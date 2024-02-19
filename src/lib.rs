//! See the documentation for [`InfVec`] for usage information.

use std::{
    array,
    cell::RefCell,
    fmt::{self, Debug},
    marker::PhantomData,
    mem::MaybeUninit,
    ops::{Index, IndexMut},
    ptr::NonNull,
};

/// Number of elements in a chunk in the cache
const CHUNK_SIZE: usize = 64;

// TODO Cargo.toml

/// A trait for types that can produce an infinite stream of elements of type
/// `T`.
pub trait Producer<T> {
    fn produce(&mut self) -> T;
}

impl<T, P: Producer<T> + ?Sized> Producer<T> for &mut P {
    fn produce(&mut self) -> T {
        (**self).produce()
    }
}

impl<T, P: Producer<T> + ?Sized> Producer<T> for Box<P> {
    fn produce(&mut self) -> T {
        (**self).produce()
    }
}

/// A [`Producer`] that produces elements by repeatedly calling a closure.
pub struct FnMutProducer<F>(pub F);

impl<T, F: FnMut() -> T> Producer<T> for FnMutProducer<F> {
    fn produce(&mut self) -> T {
        self.0()
    }
}

/// A [`Producer`] that produces elements from an iterator which is assumed to
/// be infinite. Panics if the iterator runs out of elements.
pub struct IteratorProducer<I>(pub I);

impl<I: Iterator> Producer<I::Item> for IteratorProducer<I> {
    fn produce(&mut self) -> I::Item {
        self.0
            .next()
            .expect("An IteratorProducer should never run out of elements")
    }
}

// TODO more docs

// TODO iterators
// TODO map, filter, etc.

/// An infinite `Vec`-like data structure. The main type of this crate.
///
/// An `InfVec<T, P>` contains infinitely many values of type `T`.  It can be
/// iterated over and indexed into similarly to a `Vec`. Elements are produced
/// on-demand by a [`Producer`] with type `P` which is specified upon creation
/// of the `InfVec`.
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
pub type InfVecBoxed<T> = InfVec<T, Box<dyn Producer<T> + 'static>>;

type Chunk<T> = NonNull<[MaybeUninit<T>; CHUNK_SIZE]>;

/// SAFETY invariant: Each chunk points to is a valid allocation, and the first
/// `cached_len` `T` elements of `cached_chunks` are initialized.
///
/// Invariant when panics do not occur: After the first `cached_len` elements,
/// the remaining elements are uninitialized. Additionally, the last chunk in
/// `cached_chunk` has at least one initialized element.
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
        for chunk in &inner.cached_chunks {
            // Drop each chunk
            unsafe {
                drop(Box::from_raw(chunk.as_ptr()));
            }
        }
    }
}

impl<T, P> InfVecInner<T, P> {
    fn index_ptr(&self, i: usize) -> *const MaybeUninit<T> {
        let chunk = &self.cached_chunks[i / CHUNK_SIZE];
        // We do this janky way to avoid creating a reference to the chunk,
        // since someone else might be holding a reference into the chunk.
        let first_element_ptr = chunk.as_ptr() as *const MaybeUninit<T>;
        // SAFETY: `i % CHUNK_SIZE` is in `0..CHUNK_SIZE` so it's in bounds.
        unsafe { first_element_ptr.add(i % CHUNK_SIZE) }
    }

    fn index_ptr_mut(&mut self, i: usize) -> *mut MaybeUninit<T> {
        let chunk = &self.cached_chunks[i / CHUNK_SIZE];
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

impl<T> InfVecBoxed<T> {
    /// Creates an [`InfVecBoxed`] from a closure. This method boxes the
    /// closure for you. The resulting `InfVecBoxed` is conceptually initialized
    /// by values from successive calls to the closure.
    pub fn boxed_from_fn(f: impl FnMut() -> T + 'static) -> InfVecBoxed<T> {
        InfVecBoxed::new(Box::new(FnMutProducer(f)))
    }

    /// Creates an [`InfVecBoxed`] from an iterator. This method boxes the
    /// iterator for you. The resulting `InfVec` conceptually contains the
    /// stream of elements produced by the iterator. Operations on the resulting
    /// `InfVecBoxed` might panic if the iterator is not infinite.
    pub fn boxed_from_infinite_iter(iter: impl Iterator<Item = T> + 'static) -> InfVecBoxed<T> {
        InfVecBoxed::new(Box::new(IteratorProducer(iter)))
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
}

impl<T, P> Debug for InfVec<T, P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!()
    }
}

impl<T, P> Clone for InfVec<T, P> {
    fn clone(&self) -> InfVec<T, P> {
        todo!()
    }
}

impl<T, P: Producer<T>> Index<usize> for InfVec<T, P> {
    type Output = T;

    fn index(&self, index: usize) -> &T {
        self.ensure_cached(index);
        let guard = self.0.borrow();
        let inner = &*guard;
        // SAFETY: `ensure_cached` already ensures that the given element is initialized.
        unsafe { (*inner.index_ptr(index)).assume_init_ref() }
    }
}

impl<T, P: Producer<T>> IndexMut<usize> for InfVec<T, P> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        self.ensure_cached(index);
        let mut guard = self.0.borrow_mut();
        let inner = &mut *guard;
        // SAFETY: `ensure_cached` already ensures that the given element is initialized.
        unsafe { (*inner.index_ptr_mut(index)).assume_init_mut() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let vec: InfVecBoxed<_> = InfVecBoxed::boxed_from_fn(move || {
            x += 1;
            x
        });
        for i in 0..100 {
            assert_eq!(vec[i], i + 1);
        }
    }

    #[test]
    fn test_boxed_from_infinite_iter() {
        let vec: InfVecBoxed<_> = InfVecBoxed::boxed_from_infinite_iter((0..).map(|x| x * x));
        for i in 0..100 {
            assert_eq!(vec[i], i * i);
        }
    }
}
