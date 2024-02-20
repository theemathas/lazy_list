//! See the documentation for [`InfVec`] for usage information.

pub mod producer;
pub use producer::Producer;

use producer::{FnMutProducer, IteratorProducer};
use std::{
    array,
    cell::RefCell,
    fmt::{self, Debug},
    marker::PhantomData,
    mem::{transmute, MaybeUninit},
    ops::{Index, IndexMut},
    ptr::NonNull,
};

/// Number of elements in a chunk in the cache
const CHUNK_SIZE: usize = 64;

// TODO Cargo.toml

// TODO figure out clone (for InfVec and for producers)

// TODO more docs

// TODO map, filter, etc.

/// A lazily-populated infinite `Vec`-like data structure, the main type of
/// this crate.
///
/// An `InfVec<T, P>` contains infinitely many values of type `T`.  It can be
/// iterated over and indexed into similarly to a `Vec`. Elements are produced
/// on-demand by a [`Producer`] with type `P` which is specified upon creation
/// of the `InfVec`. The elements can be later accessed by index, and can be
/// modified.
///
/// `InfVec` is currently not thread-safe. It is also invariant, as opposed to
/// covariant.
///
/// Despite the name, `InfVec` doesn't actually store elements continuously like
/// a `Vec`. Instead, it stores elements in 64-element chunks. This is so that
/// we can hand out references to elements, and not have them be invalidated as
/// more elements are added to the cache.
///
/// If you don't want to specify a producer type as a type parameter, you can
/// use the [`InfVecBoxed`] type alias.
pub struct InfVec<T, P>(RefCell<InfVecInner<T, P>>, PhantomData<(Vec<T>, P)>);

/// A type alias for an [`InfVec`] with a boxed [`Producer`] for convenience.
///
/// In most cases, `InfVecBoxed<'static, T>` is the type you want.
pub type InfVecBoxed<'a, T> = InfVec<T, Box<dyn Producer<T> + 'a>>;

type Chunk<T> = NonNull<[MaybeUninit<T>; CHUNK_SIZE]>;

/// SAFETY invariant: Each chunk points to is a valid allocation, and the first
/// `cached_len` `T` elements of `cached_chunks` are initialized. (Exception:
/// the "must be initialized" invariant doesn't apply when `InfVec` is inside an
/// `IntoIter`.)
///
/// Non-safety invariant when panics do not occur: After the first `cached_len`
/// elements, the remaining elements are uninitialized.
struct InfVecInner<T, P> {
    /// The elements that have been produced so far and cached. The i'th cached
    /// element is at `cached_chunks[i / CHUNK_SIZE][i % CHUNK_SIZE]`.
    ///
    /// We use chunks like this instead of a simple `Vec<T>` so we can hand out
    /// references to elements, and not have them be invalidated when we cache
    /// more elements. This is because reallocating the `Vec` doesn't move the
    /// chunks.
    cached_chunks: Vec<Chunk<T>>,
    /// The number of elements that have been produced and cached so far.
    num_cached: usize,
    /// The producer of elements, which will be called to acquire elements that
    /// haven't been cached yet.
    producer: P,
}

/// Iterator over immutable references to the elements of an [`InfVec`].
///
/// This struct is created by the [`iter`](InfVec::iter) method on [`InfVec`].
#[derive(Debug)]
pub struct Iter<'a, T, P> {
    vec: &'a InfVec<T, P>,
    index: usize,
}

/// Iterator over mutable references to the elements of an [`InfVec`].
///
/// This struct is created by the [`iter_mut`](InfVec::iter_mut) method on
/// [`InfVec`].
#[derive(Debug)]
pub struct IterMut<'a, T, P> {
    vec: &'a mut InfVec<T, P>,
    index: usize,
}

// TODO Debug impl
/// Iterator that moves elements out of an [`InfVec`].
///
/// This struct is created by the `into_iter` method on [`InfVec`].
pub struct IntoIter<T, P> {
    // SAFETY invariant: Instead of `InfVec` having elements `0..num_cached`
    // initialized, this one has elements `index..num_cached` initialized.
    vec: InfVec<T, P>,
    index: usize,
}

// SAFETY: We don't rely on everything being in one thread. We don't have `Sync`
// though.
unsafe impl<T: Send, P: Send> Send for InfVec<T, P> {}

impl<T, P> Drop for InfVec<T, P> {
    fn drop(&mut self) {
        let mut inner = self.0.borrow_mut();
        // SAFETY: Only drops initialized elements as per the invariant of
        // `InfVecInner`. If any element's destructor panics, nothing else is
        // dropped.
        for i in 0..inner.num_cached {
            // Drop each element
            unsafe {
                inner.cached_chunks[i / CHUNK_SIZE].as_mut()[i % CHUNK_SIZE].assume_init_drop();
            }
        }
        for chunk in inner.cached_chunks.drain(..) {
            // Drop each chunk
            unsafe {
                drop(Box::from_raw(chunk.as_ptr()));
            }
        }
    }
}

impl<T, P> Drop for IntoIter<T, P> {
    fn drop(&mut self) {
        let mut inner = self.vec.0.borrow_mut();

        // Reset the InfVec to have no cached elements, so that its drop impl
        // doesn't drop any of the `T` elements, even if panics happen.
        let actual_num_cached = inner.num_cached;
        inner.num_cached = 0;

        // SAFETY: Only drops initialized elements as per the invariant of
        // `InfVecInner`. If any element's destructor panics, nothing else is
        // dropped.
        for i in self.index..actual_num_cached {
            // Drop each element
            unsafe {
                inner.cached_chunks[i / CHUNK_SIZE].as_mut()[i % CHUNK_SIZE].assume_init_drop();
            }
        }
        for chunk in inner.cached_chunks.drain(..) {
            // Drop each chunk
            unsafe {
                drop(Box::from_raw(chunk.as_ptr()));
            }
        }
    }
}

impl<T, P> InfVecInner<T, P> {
    fn index_ptr(&self, i: usize) -> *const MaybeUninit<T> {
        let chunk = self.cached_chunks[i / CHUNK_SIZE];
        // We do this janky way to avoid creating a reference to the chunk,
        // since someone else might be holding a reference into the chunk.
        let first_element_ptr = chunk.as_ptr() as *const MaybeUninit<T>;
        // SAFETY: `i % CHUNK_SIZE` is in `0..CHUNK_SIZE` so it's in bounds.
        unsafe { first_element_ptr.add(i % CHUNK_SIZE) }
    }

    fn index_ptr_mut(&mut self, i: usize) -> *mut MaybeUninit<T> {
        let chunk = self.cached_chunks[i / CHUNK_SIZE];
        // We do this janky way to avoid creating a reference to the chunk,
        // since someone else might be holding a reference into the chunk.
        let first_element_ptr = chunk.as_ptr().cast::<MaybeUninit<T>>();
        // SAFETY: `i % CHUNK_SIZE` is in `0..CHUNK_SIZE` so it's in bounds.
        unsafe { first_element_ptr.add(i % CHUNK_SIZE) }
    }
}

impl<T, P> InfVec<T, P> {
    /// Creates a new `InfVec` with the given producer. The resulting `InfVec`
    /// is conceptually initialized by values from successive calls to
    /// [produce](Producer::produce).
    pub const fn new(producer: P) -> InfVec<T, P> {
        InfVec(
            RefCell::new(InfVecInner {
                cached_chunks: Vec::new(),
                num_cached: 0,
                producer,
            }),
            PhantomData,
        )
    }

    /// Returns the number of elements that have been produced and cached so far.
    pub fn num_cached(&self) -> usize {
        self.0.borrow().num_cached
    }
}

impl<T, F: FnMut() -> T> InfVec<T, FnMutProducer<F>> {
    /// Creates a new `InfVec` from a closure. The resulting `InfVec` is
    /// conceptually initialized by values from successive calls to the closure.
    pub fn from_fn(f: F) -> InfVec<T, FnMutProducer<F>> {
        InfVec::new(FnMutProducer(f))
    }
}

impl<I: Iterator> InfVec<I::Item, IteratorProducer<I>> {
    /// Creates a new `InfVec` from an iterator, conceptually containing the
    /// stream of elements produced by the iterator. Operations on the resulting
    /// `InfVec` might panic if the iterator is not infinite.
    pub fn from_infinite_iter(iter: I) -> InfVec<I::Item, IteratorProducer<I>> {
        InfVec::new(IteratorProducer(iter))
    }
}

impl<'a, T> InfVecBoxed<'a, T> {
    /// Creates an [`InfVecBoxed`] from a closure. This method boxes the
    /// closure for you. The resulting `InfVecBoxed` is conceptually initialized
    /// by values from successive calls to the closure.
    pub fn boxed_from_fn(f: impl FnMut() -> T + 'a) -> InfVecBoxed<'a, T> {
        InfVecBoxed::new(Box::new(FnMutProducer(f)))
    }

    /// Creates an [`InfVecBoxed`] from an iterator. This method boxes the
    /// iterator for you. The resulting `InfVec` conceptually contains the
    /// stream of elements produced by the iterator. Operations on the resulting
    /// `InfVecBoxed` might panic if the iterator is not infinite.
    pub fn boxed_from_infinite_iter(iter: impl Iterator<Item = T> + 'a) -> InfVecBoxed<'a, T> {
        InfVecBoxed::new(Box::new(IteratorProducer(iter)))
    }
}

impl<T, P: Producer<T>> Index<usize> for InfVec<T, P> {
    type Output = T;

    fn index(&self, index: usize) -> &T {
        self.ensure_cached(index);
        let guard = self.0.borrow();
        let inner = &*guard;
        // SAFETY: `ensure_cached` already ensures that the given element is
        // initialized. Additionally, pushing a new chunk does not invalidate
        // references to existing chunks.
        unsafe { (*inner.index_ptr(index)).assume_init_ref() }
    }
}

impl<T, P: Producer<T>> IndexMut<usize> for InfVec<T, P> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        self.ensure_cached(index);
        let mut guard = self.0.borrow_mut();
        let inner = &mut *guard;
        // SAFETY: `ensure_cached` already ensures that the given element is
        // initialized. Additionally, pushing a new chunk does not invalidate
        // references to existing chunks. And we have a mutable reference to
        // self, so we can't have any other references to the same data.
        unsafe { (*inner.index_ptr_mut(index)).assume_init_mut() }
    }
}

impl<T, P: Producer<T>> InfVec<T, P> {
    /// Ensures that element `i` is cached.
    fn ensure_cached(&self, i: usize) {
        let mut guard = self.0.borrow_mut();
        let inner = &mut *guard;
        // While the element is not cached, cache one more element.
        while i >= inner.num_cached {
            // If all chunks are exactly full. Allocate a new chunk.
            if inner.num_cached % CHUNK_SIZE == 0 {
                assert!(inner.num_cached == inner.cached_chunks.len() * CHUNK_SIZE);
                inner.cached_chunks.push(
                    Chunk::new(Box::into_raw(Box::new(array::from_fn(|_| {
                        MaybeUninit::uninit()
                    }))))
                    .unwrap(),
                );
            }

            // Fill in the next element.
            let target_element_ptr: *mut MaybeUninit<T> = inner.index_ptr_mut(inner.num_cached);
            // SAFETY: the allocation is valid.
            let target_element_ref: &mut MaybeUninit<T> = unsafe { &mut *target_element_ptr };
            // A panic here won't corrupt anything since we haven't updated
            // cached_len yet.
            target_element_ref.write(inner.producer.produce());

            inner.num_cached += 1;
        }
    }

    /// Returns an infinite iterator over the elements of the `InfVec`.
    pub fn iter(&self) -> Iter<T, P> {
        Iter {
            vec: self,
            index: 0,
        }
    }

    /// Returns a mutable infinite iterator over the elements of the `InfVec`.
    pub fn iter_mut(&mut self) -> IterMut<T, P> {
        IterMut {
            vec: self,
            index: 0,
        }
    }
}

impl<T, P: Producer<T>> IntoIterator for InfVec<T, P> {
    type Item = T;
    type IntoIter = IntoIter<T, P>;

    fn into_iter(self) -> IntoIter<T, P> {
        IntoIter {
            vec: self,
            index: 0,
        }
    }
}

impl<'a, T, P: Producer<T>> IntoIterator for &'a InfVec<T, P> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T, P>;

    fn into_iter(self) -> Iter<'a, T, P> {
        self.iter()
    }
}

impl<'a, T, P: Producer<T>> IntoIterator for &'a mut InfVec<T, P> {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T, P>;

    fn into_iter(self) -> IterMut<'a, T, P> {
        self.iter_mut()
    }
}

impl<'a, T, P: Producer<T>> Iterator for Iter<'a, T, P> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        let result = &self.vec[self.index];
        self.index += 1;
        Some(result)
    }
}

impl<'a, T, P: Producer<T>> Iterator for IterMut<'a, T, P> {
    type Item = &'a mut T;

    fn next<'b>(&'b mut self) -> Option<&'a mut T> {
        let result = &mut self.vec[self.index];
        self.index += 1;
        // SAFETY: the reference to the element existing does not depend on the
        // the self reference existing.
        unsafe { Some(transmute::<&'b mut T, &'a mut T>(result)) }
    }
}

impl<T, P: Producer<T>> Iterator for IntoIter<T, P> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        let mut inner = self.vec.0.borrow_mut();
        if self.index < inner.num_cached {
            let result = unsafe {
                // SAFETY: `index` is in bounds, and elements in the range
                // `index..num_cached` are initialized.
                inner.index_ptr(self.index).read().assume_init()
            };
            self.index += 1;
            Some(result)
        } else {
            Some(inner.producer.produce())
        }
    }
}

impl<T: Debug, P> Debug for InfVec<T, P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Hack until DebugList::entries is stabilized
        struct DebugEllipsis;
        impl Debug for DebugEllipsis {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("...")
            }
        }

        let inner = self.0.borrow();
        f.debug_list()
            .entries((0..inner.num_cached).map(|i| unsafe {
                // SAFETY: Elements in the range `0..num_cached` are
                // initialized.
                (*inner.index_ptr(i)).assume_init_ref()
            }))
            .entry(&DebugEllipsis)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::atomic::AtomicUsize;

    #[test]
    fn test_indexing() {
        let mut x = 0;
        let mut vec = InfVec::from_fn(move || {
            x += 1;
            x
        });
        for i in 0..100 {
            assert_eq!(vec[i], i + 1);
        }
        for i in 0..100 {
            assert_eq!(vec[i], i + 1);
        }
        for i in 0..200 {
            vec[i] = 10 * i;
        }
        for i in 0..200 {
            assert_eq!(vec[i], 10 * i);
        }
    }

    #[test]
    fn test_boxed_from_fn() {
        let mut x = 0;
        let vec: InfVecBoxed<'_, _> = InfVecBoxed::boxed_from_fn(move || {
            x += 1;
            x
        });
        for i in 0..100 {
            assert_eq!(vec[i], i + 1);
        }
    }

    #[test]
    fn test_boxed_from_infinite_iter() {
        let vec: InfVecBoxed<'_, _> = InfVecBoxed::boxed_from_infinite_iter((0..).map(|x| x * x));
        for i in 0..100 {
            assert_eq!(vec[i], i * i);
        }
    }

    #[test]
    fn test_iter() {
        let mut x = 0;
        let vec = InfVec::from_fn(move || {
            x += 1;
            x
        });
        let mut iter = vec.iter();
        for i in 0..100 {
            assert_eq!(iter.next(), Some(&(i + 1)));
        }
        let mut iter2 = vec.iter();
        for i in 0..200 {
            assert_eq!(iter2.next(), Some(&(i + 1)));
        }
        for i in 0..200 {
            assert_eq!(iter.next(), Some(&(i + 101)));
        }
    }

    #[test]
    fn test_iter_mut() {
        let mut x = 0;
        let mut vec = InfVec::from_fn(move || {
            x += 1;
            x
        });
        let mut iter = vec.iter_mut();
        for i in 0..100 {
            let element_ref = iter.next().unwrap();
            assert_eq!(element_ref, &mut (i + 1));
            *element_ref += 1000;
        }
        let mut iter2 = vec.iter_mut();
        for i in 0..100 {
            assert_eq!(iter2.next(), Some(&mut (i + 1001)));
        }
        for i in 0..100 {
            assert_eq!(iter2.next(), Some(&mut (i + 101)));
        }
    }

    #[test]
    fn test_into_iter() {
        struct DropCounter<'a>(&'a AtomicUsize, usize);
        impl Drop for DropCounter<'_> {
            fn drop(&mut self) {
                self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }

        let drop_counter = AtomicUsize::new(0);
        let drop_counter_ref = &drop_counter;
        let mut x = 0;
        let vec = InfVec::from_fn(move || {
            x += 1;
            DropCounter(drop_counter_ref, x)
        });

        let mut iter = vec.iter();
        for _ in 0..200 {
            // Fill the cache
            iter.next();
        }
        let mut into_iter = vec.into_iter();
        for i in 0..100 {
            assert_eq!(into_iter.next().map(|x| x.1), Some(i + 1));
        }

        // Iterating through `into_iter` should consume the iterated elements.
        assert_eq!(drop_counter.load(std::sync::atomic::Ordering::Relaxed), 100);
        // Dropping the `into_iter` should deallocate the cached elements.
        drop(into_iter);
        assert_eq!(drop_counter.load(std::sync::atomic::Ordering::Relaxed), 200);
    }

    #[test]
    fn test_debug() {
        let mut x = 0;
        let vec = InfVec::from_fn(move || {
            x += 1;
            x
        });
        assert_eq!(format!("{:?}", vec), "[...]");
        vec[9];
        assert_eq!(format!("{:?}", vec), "[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ...]");
    }
}
