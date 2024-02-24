//! See the documentation for [`InfVec`] for usage information.

mod chunked_vec;

use std::{
    fmt::{self, Debug},
    iter,
    mem::transmute,
    ops::{Index, IndexMut},
    sync::{Arc, Mutex},
};

use chunked_vec::ChunkedVec;

// TODO figure out clone
// TODO more docs
// TODO map, filter, etc.
// TODO Debug impl for iterator

const FINITE_ITERATOR_PANIC_MESSAGE: &str =
    "The iterator used to construct an InfVec should be infinite";

/// A lazily-populated infinite `Vec`-like data structure, the main type of this
/// crate.
///
/// An `InfVec<T, I>` contains infinitely many values of type `T`.  It can be
/// iterated over and indexed into similarly to a `Vec`. Elements are produced
/// on-demand by a iterator with type `I` which is specified upon creation of
/// the `InfVec`. The elements can be later accessed by index, and can be
/// modified.
///
/// If the iterator is not infinite, operations on the `InfVec` might panic.
///
/// `InfVec` is currently invariant, as opposed to covariant, if that matters to
/// you.
///
/// Despite the name, `InfVec` doesn't actually store elements continuously like
/// a `Vec`. Instead, it stores elements in 64-element chunks. This is so that
/// we can hand out references to elements, and not have them be invalidated as
/// more elements are added to the cache.
///
/// If you don't want to specify an iterator type as a type parameter, you can
/// use the [`InfVecBoxed`] type alias.
pub struct InfVec<T, I> {
    cached: ChunkedVec<T>,
    remaining: Mutex<I>,
}

/// Type alias for an [`InfVec`] with an unknown iterator type, which might
/// borrow data with lifetime `'a`.
///
/// In most cases, [`InfVecOwned`] is the type you want.
pub type InfVecBoxed<'a, T> = InfVec<T, Box<dyn Iterator<Item = T> + 'a>>;

/// Type alias for an [`InfVec`] with an unknown iterator type, which has no
/// borrowed data.
pub type InfVecOwned<T> = InfVecBoxed<'static, T>;

/// Iterator over immutable references to the elements of an [`InfVec`].
///
/// This struct is created by the [`iter`](InfVec::iter) method on [`InfVec`].
pub struct Iter<'a, T, I> {
    inf_vec: &'a InfVec<T, I>,
    index: usize,
}

/// Iterator over mutable references to the elements of an [`InfVec`].
///
/// This struct is created by the [`iter_mut`](InfVec::iter_mut) method on
/// [`InfVec`].
pub struct IterMut<'a, T, I> {
    inf_vec: &'a mut InfVec<T, I>,
    index: usize,
}

/// Iterator that moves elements out of an [`InfVec`].
///
/// This struct is created by the `into_iter` method on [`InfVec`].
pub struct IntoIter<T, I>(iter::Chain<chunked_vec::IntoIter<T>, I>);

impl<T, I> InfVec<T, I> {
    /// Creates an [`InfVec`] from an iterator. The resulting `InfVec`
    /// conceptually contains the stream of elements produced by the iterator.
    /// Operations on the resulting `InfVec` might panic if the iterator is not
    /// infinite.
    ///
    /// Equivalent to [`IteratorInfExt::collect_inf`].
    pub const fn new(iterator: I) -> InfVec<T, I> {
        InfVec {
            cached: ChunkedVec::new(),
            remaining: Mutex::new(iterator),
        }
    }

    /// Returns the number of elements that have been produced and cached so far.
    pub fn num_cached(&self) -> usize {
        self.cached.len()
    }
}

impl<T, F: FnMut() -> T> InfVec<T, iter::RepeatWith<F>> {
    /// Creates an [`InfVec`] that conceptually contains elements from
    /// successive calls to a given closure.
    pub fn repeat_with(f: F) -> InfVec<T, iter::RepeatWith<F>> {
        InfVec::new(iter::repeat_with(f))
    }
}

impl<'a, T: 'a> InfVecBoxed<'a, T> {
    /// Creates a recursive `InfVec`. The closure should take a reference to the
    /// `InfVec` itself and an index, then return the element at that index. The
    /// closure should only attempt to access prior elements of the `InfVec`, or
    /// a deadlock will occur.
    pub fn recursive<F: FnMut(&InfVecBoxed<'a, T>, usize) -> T + 'a>(
        mut f: F,
    ) -> Arc<InfVecBoxed<'a, T>> {
        Arc::new_cyclic(|weak| {
            let weak = weak.clone();
            InfVec::new((0..).map(move |i| f(&weak.upgrade().unwrap(), i))).boxed()
        })
    }
}

impl<'a, T, I: Iterator<Item = T> + 'a> InfVec<T, I> {
    pub fn boxed(self) -> InfVecBoxed<'a, T> {
        InfVec {
            cached: self.cached,
            remaining: Mutex::new(Box::new(self.remaining.into_inner().unwrap())),
        }
    }
}

impl<T, I: Iterator<Item = T>> Index<usize> for InfVec<T, I> {
    type Output = T;

    fn index(&self, index: usize) -> &T {
        self.ensure_cached(index);
        // SAFETY: Shared reference to self ensures that no other mutable
        // references to contents exist.
        unsafe { self.cached.index(index) }
    }
}

impl<T, I: Iterator<Item = T>> IndexMut<usize> for InfVec<T, I> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        self.ensure_cached(index);
        // SAFETY: Exclusive reference to self ensures that no other references
        // to contents exist.
        unsafe { self.cached.index_mut(index) }
    }
}

impl<T, I: Iterator<Item = T>> InfVec<T, I> {
    /// Ensures that element `index` is cached.
    fn ensure_cached(&self, index: usize) {
        // Don't lock if the element is already cached.
        if self.cached.len() > index {
            return;
        }
        let mut guard = self.remaining.lock().unwrap();
        while self.cached.len() <= index {
            let element = guard.next().expect(FINITE_ITERATOR_PANIC_MESSAGE);
            self.cached.push(element);
        }
    }

    /// Returns an infinite iterator over the elements of the `InfVec`.
    pub fn iter(&self) -> Iter<T, I> {
        Iter {
            inf_vec: self,
            index: 0,
        }
    }

    /// Returns a mutable infinite iterator over the elements of the `InfVec`.
    pub fn iter_mut(&mut self) -> IterMut<T, I> {
        IterMut {
            inf_vec: self,
            index: 0,
        }
    }
}

impl<T, I: Iterator<Item = T>> IntoIterator for InfVec<T, I> {
    type Item = T;
    type IntoIter = IntoIter<T, I>;

    fn into_iter(self) -> IntoIter<T, I> {
        IntoIter(
            self.cached
                .into_iter()
                .chain(self.remaining.into_inner().unwrap()),
        )
    }
}

impl<'a, T, I: Iterator<Item = T>> IntoIterator for &'a InfVec<T, I> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T, I>;

    fn into_iter(self) -> Iter<'a, T, I> {
        self.iter()
    }
}

impl<'a, T, I: Iterator<Item = T>> IntoIterator for &'a mut InfVec<T, I> {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T, I>;

    fn into_iter(self) -> IterMut<'a, T, I> {
        self.iter_mut()
    }
}

impl<'a, T, I: Iterator<Item = T>> Iterator for Iter<'a, T, I> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        let result = &self.inf_vec[self.index];
        self.index += 1;
        Some(result)
    }
}

impl<'a, T, I: Iterator<Item = T>> Iterator for IterMut<'a, T, I> {
    type Item = &'a mut T;

    fn next<'b>(&'b mut self) -> Option<&'a mut T> {
        let result = &mut self.inf_vec[self.index];
        self.index += 1;
        // SAFETY: the reference to the element existing does not depend on the
        // the self reference existing.
        unsafe { Some(transmute::<&'b mut T, &'a mut T>(result)) }
    }
}

impl<T, I: Iterator<Item = T>> Iterator for IntoIter<T, I> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        self.0.next()
    }
}

impl<T: Debug, I> Debug for InfVec<T, I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Hack until DebugList::entry_with is stabilized
        struct DebugEllipsis;
        impl Debug for DebugEllipsis {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("...")
            }
        }

        f.debug_list()
            .entries((0..self.cached.len()).map(|i| unsafe {
                // SAFETY: Shared reference to self ensures that no other mutable
                // references to contents exist.
                self.cached.index(i)
            }))
            .entry(&DebugEllipsis)
            .finish()
    }
}

pub trait IteratorInfExt: Iterator + Sized {
    /// Collects the elements of an iterator into an [`InfVec`]. If the iterator
    /// isn't infinite, operations on the resulting `InfVec` might panic.
    ///
    /// Equivalent to [`InfVec::new`].
    fn collect_inf<T>(self) -> InfVec<T, Self>;
}
impl<I: Iterator + Sized> IteratorInfExt for I {
    fn collect_inf<T>(self) -> InfVec<T, Self> {
        InfVec::new(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::{iter, sync::atomic::AtomicUsize};

    #[test]
    fn test_indexing() {
        let mut x = 0;
        let mut vec = InfVec::new(iter::repeat_with(move || {
            x += 1;
            x
        }));
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
    fn test_boxed_from_infinite_iter() {
        let vec: InfVecBoxed<'_, _> = InfVec::new((0..).map(|x| x * x)).boxed();
        for i in 0..100 {
            assert_eq!(vec[i], i * i);
        }
    }

    #[test]
    fn test_iter() {
        let mut x = 0;
        let vec = InfVec::new(iter::repeat_with(move || {
            x += 1;
            x
        }));
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
        let mut vec = InfVec::new(iter::repeat_with(move || {
            x += 1;
            x
        }));
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
        let vec = InfVec::new(iter::repeat_with(move || {
            x += 1;
            DropCounter(drop_counter_ref, x)
        }));

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
        println!("dropping");
        drop(into_iter);
        assert_eq!(drop_counter.load(std::sync::atomic::Ordering::Relaxed), 200);
    }

    #[test]
    fn test_debug() {
        let mut x = 0;
        let vec = InfVec::new(iter::repeat_with(move || {
            x += 1;
            x
        }));
        assert_eq!(format!("{:?}", vec), "[...]");
        vec[9];
        assert_eq!(format!("{:?}", vec), "[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ...]");
    }

    #[test]
    fn test_recursive() {
        let evens = InfVec::recursive(|evens_ref, i| if i == 0 { 0 } else { evens_ref[i - 1] + 2 });
        assert_eq!(
            evens.iter().copied().take(10).collect::<Vec<_>>(),
            [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
        );
    }
}
