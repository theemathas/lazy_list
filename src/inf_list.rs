//! This module provides [`InfList`], A lazily-populated infinite list.
//!
//! An `InfList<T, I>` contains infinitely many values of type `T`.  It can be
//! iterated over and indexed into similarly to a `Vec`. Elements are produced
//! on-demand by a iterator with type `I` which is specified upon creation of
//! the `InfList`. The elements can be later accessed by index, and can be
//! modified.
//!
//! If the iterator is not infinite, operations on the `InfList` might panic.
//!
//! `InfList` is currently invariant, as opposed to covariant, if that matters
//! to you.
//!
//! An immutable `InfList` is thread-safe, so you can put it in `static`
//! variables.
//!
//! Internally, `InfList` stores elements in 64-element chunks. This is so that
//! we can hand out references to elements, and not have them be invalidated as
//! more elements are added to the cache.
//!
//! If you don't want to specify an iterator type as a type parameter, you can
//! use the [`InfListBoxed`] or [`InfListOwned`] type aliases.
//!
//! # Examples
//!
//! Mutation of an `InfList`:
//! ```
//! use lazy_list::InfList;
//!
//! let mut list = InfList::new(0..);
//! assert_eq!(
//!     list.iter().take(10).copied().collect::<Vec<_>>(),
//!     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
//! );
//! list[3] = 100;
//! assert_eq!(
//!     list.iter().take(10).copied().collect::<Vec<_>>(),
//!     [0, 1, 2, 100, 4, 5, 6, 7, 8, 9]
//! );
//! ```
//!
//! Reusing a static `InfList`:
//! ```
//! use lazy_list::{InfList, InfListOwned, IteratorInfExt};
//! use once_cell::sync::Lazy;
//!
//! // Note that each element will only ever be produced once.
//! static EVENS: Lazy<InfListOwned<i32>> =
//!     Lazy::new(|| (0..).map(|x| x * 2).collect_inf().boxed());
//!
//! fn evens_with_property(mut predicate: impl FnMut(i32) -> bool) -> impl Iterator<Item = i32> {
//!     EVENS.iter().copied().filter(move |&x| predicate(x))
//! }
//!
//! assert_eq!(
//!     evens_with_property(|x| x % 3 == 0)
//!         .take(5)
//!         .collect::<Vec<_>>(),
//!     [0, 6, 12, 18, 24]
//! );
//! assert_eq!(
//!     evens_with_property(|x| x % 5 == 0)
//!         .take(5)
//!         .collect::<Vec<_>>(),
//!     [0, 10, 20, 30, 40]
//! );
//! ```
//!
//! Recursive `InfList`:
//! ```
//! use lazy_list::{InfList, InfListBoxed};
//! use std::sync::Arc;
//!
//! let fibonacci: Arc<InfListBoxed<i32>> = InfList::recursive(|fibonacci_ref, i| {
//!     if i < 2 {
//!         1
//!     } else {
//!         fibonacci_ref[i - 1] + fibonacci_ref[i - 2]
//!     }
//! });
//! assert_eq!(
//!     fibonacci.iter().take(10).copied().collect::<Vec<_>>(),
//!     [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
//! );
//! ```

// TODO: Debug impl for iterators

use std::{
    fmt::{self, Debug},
    iter,
    mem::transmute,
    ops::{Index, IndexMut},
    sync::{Arc, Mutex},
};

use crate::chunked_vec::{self, ChunkedVec};

const FINITE_ITERATOR_PANIC_MESSAGE: &str =
    "The iterator used to construct an InfList should be infinite";

/// A lazily-populated infinite list.
///
/// See the [module-level documentation](crate::inf_list) for more information.
pub struct InfList<T, I> {
    cached: ChunkedVec<T>,
    remaining: Mutex<I>,
}

/// Type alias for an [`InfList`] with an unknown iterator type, which might
/// borrow data with lifetime `'a`.
///
/// This type can be constructed by manually boxing the iterator, or by calling
/// [`InfList::boxed`].
///
/// In most cases, [`InfListOwned`] is the type you want.
pub type InfListBoxed<'a, T> = InfList<T, Box<dyn Iterator<Item = T> + Send + 'a>>;

/// Type alias for an [`InfList`] with an unknown iterator type, which has no
/// borrowed data.
///
/// This type can be constructed by manually boxing the iterator, or by calling
/// [`InfList::boxed`].
pub type InfListOwned<T> = InfListBoxed<'static, T>;

/// Iterator over immutable references to the elements of an [`InfList`].
///
/// This struct is created by the [`iter`](InfList::iter) method on [`InfList`].
pub struct Iter<'a, T, I> {
    inf_list: &'a InfList<T, I>,
    index: usize,
}

/// Iterator over mutable references to the elements of an [`InfList`].
///
/// This struct is created by the [`iter_mut`](InfList::iter_mut) method on
/// [`InfList`].
pub struct IterMut<'a, T, I> {
    inf_list: &'a mut InfList<T, I>,
    index: usize,
}

/// Iterator that moves elements out of an [`InfList`].
///
/// This struct is created by the `into_iter` method on [`InfList`].
pub struct IntoIter<T, I>(iter::Chain<chunked_vec::IntoIter<T>, I>);

impl<T, I> InfList<T, I> {
    /// Creates an [`InfList`] from an iterator. The resulting `InfList`
    /// conceptually contains the stream of elements produced by the iterator.
    /// Operations on the resulting `InfList` might panic if the iterator is not
    /// infinite.
    ///
    /// Equivalent to [`IteratorInfExt::collect_inf`].
    pub const fn new(iterator: I) -> InfList<T, I> {
        InfList {
            cached: ChunkedVec::new(),
            remaining: Mutex::new(iterator),
        }
    }

    /// Returns the number of elements that have been produced and cached so far.
    pub fn num_cached(&self) -> usize {
        self.cached.len()
    }
}

impl<T, F: FnMut() -> T> InfList<T, iter::RepeatWith<F>> {
    /// Creates an [`InfList`] that conceptually contains elements from
    /// successive calls to a given closure.
    pub fn repeat_with(f: F) -> InfList<T, iter::RepeatWith<F>> {
        InfList::new(iter::repeat_with(f))
    }
}

impl<'a, T: Send + Sync + 'a> InfListBoxed<'a, T> {
    /// Creates a recursive `InfList`. The closure should take a reference to the
    /// `InfList` itself and an index, then return the element at that index. The
    /// closure should only attempt to access prior elements of the `InfList`, or
    /// a deadlock will occur.
    pub fn recursive<F: FnMut(&InfListBoxed<'a, T>, usize) -> T + Send + 'a>(
        mut f: F,
    ) -> Arc<InfListBoxed<'a, T>> {
        Arc::new_cyclic(|weak| {
            let weak = weak.clone();
            InfList::new((0..).map(move |i| f(&weak.upgrade().unwrap(), i))).boxed()
        })
    }
}

impl<'a, T, I: Iterator<Item = T> + Send + 'a> InfList<T, I> {
    /// Returns a boxed version of the `InfList`. This is useful when you don't
    /// want to write out the iterator type as a type parameter.
    pub fn boxed(self) -> InfListBoxed<'a, T> {
        InfList {
            cached: self.cached,
            remaining: Mutex::new(Box::new(self.remaining.into_inner().unwrap())),
        }
    }
}

impl<T, I: Iterator<Item = T>> Index<usize> for InfList<T, I> {
    type Output = T;

    fn index(&self, index: usize) -> &T {
        self.ensure_cached(index);
        // SAFETY: Shared reference to self ensures that no other mutable
        // references to contents exist.
        unsafe { self.cached.index(index) }
    }
}

impl<T, I: Iterator<Item = T>> IndexMut<usize> for InfList<T, I> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        self.ensure_cached(index);
        // SAFETY: Exclusive reference to self ensures that no other references
        // to contents exist.
        unsafe { self.cached.index_mut(index) }
    }
}

impl<T, I: Iterator<Item = T>> InfList<T, I> {
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

    /// Returns an infinite iterator over the elements of the `InfList`.
    pub fn iter(&self) -> Iter<T, I> {
        Iter {
            inf_list: self,
            index: 0,
        }
    }

    /// Returns a mutable infinite iterator over the elements of the `InfList`.
    pub fn iter_mut(&mut self) -> IterMut<T, I> {
        IterMut {
            inf_list: self,
            index: 0,
        }
    }
}

impl<T, I: Iterator<Item = T>> IntoIterator for InfList<T, I> {
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

impl<'a, T, I: Iterator<Item = T>> IntoIterator for &'a InfList<T, I> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T, I>;

    fn into_iter(self) -> Iter<'a, T, I> {
        self.iter()
    }
}

impl<'a, T, I: Iterator<Item = T>> IntoIterator for &'a mut InfList<T, I> {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T, I>;

    fn into_iter(self) -> IterMut<'a, T, I> {
        self.iter_mut()
    }
}

impl<'a, T, I: Iterator<Item = T>> Iterator for Iter<'a, T, I> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        let result = &self.inf_list[self.index];
        self.index += 1;
        Some(result)
    }
}

impl<'a, T, I: Iterator<Item = T>> Iterator for IterMut<'a, T, I> {
    type Item = &'a mut T;

    fn next<'b>(&'b mut self) -> Option<&'a mut T> {
        let result = &mut self.inf_list[self.index];
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

impl<T: Debug, I> Debug for InfList<T, I> {
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

/// Extension trait for [`Iterator`].
///
/// This trait provides the [`collect_inf`] method, which is a method version of
/// [`InfList::new`].
pub trait IteratorInfExt: Iterator + Sized {
    /// Collects the elements of an iterator into an [`InfList`]. If the iterator
    /// isn't infinite, operations on the resulting `InfList` might panic.
    ///
    /// Equivalent to [`InfList::new`].
    fn collect_inf<T>(self) -> InfList<T, Self>;
}
impl<I: Iterator + Sized> IteratorInfExt for I {
    fn collect_inf<T>(self) -> InfList<T, Self> {
        InfList::new(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::{iter, sync::atomic::AtomicUsize};

    #[test]
    fn test_indexing() {
        let mut x = 0;
        let mut list = InfList::new(iter::repeat_with(move || {
            x += 1;
            x
        }));
        for i in 0..100 {
            assert_eq!(list[i], i + 1);
        }
        for i in 0..100 {
            assert_eq!(list[i], i + 1);
        }
        for i in 0..200 {
            list[i] = 10 * i;
        }
        for i in 0..200 {
            assert_eq!(list[i], 10 * i);
        }
    }

    #[test]
    fn test_boxed_from_infinite_iter() {
        let list: InfListBoxed<'_, _> = InfList::new((0..).map(|x| x * x)).boxed();
        for i in 0..100 {
            assert_eq!(list[i], i * i);
        }
    }

    #[test]
    fn test_iter() {
        let mut x = 0;
        let list = InfList::new(iter::repeat_with(move || {
            x += 1;
            x
        }));
        let mut iter = list.iter();
        for i in 0..100 {
            assert_eq!(iter.next(), Some(&(i + 1)));
        }
        let mut iter2 = list.iter();
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
        let mut list = InfList::new(iter::repeat_with(move || {
            x += 1;
            x
        }));
        let mut iter = list.iter_mut();
        for i in 0..100 {
            let element_ref = iter.next().unwrap();
            assert_eq!(element_ref, &mut (i + 1));
            *element_ref += 1000;
        }
        let mut iter2 = list.iter_mut();
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
        let list = InfList::new(iter::repeat_with(move || {
            x += 1;
            DropCounter(drop_counter_ref, x)
        }));

        let mut iter = list.iter();
        for _ in 0..200 {
            // Fill the cache
            iter.next();
        }
        let mut into_iter = list.into_iter();
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
        let list = InfList::new(iter::repeat_with(move || {
            x += 1;
            x
        }));
        assert_eq!(format!("{:?}", list), "[...]");
        list[9];
        assert_eq!(
            format!("{:?}", list),
            "[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ...]"
        );
    }

    #[test]
    fn test_recursive() {
        let evens =
            InfList::recursive(|evens_ref, i| if i == 0 { 0 } else { evens_ref[i - 1] + 2 });
        assert_eq!(
            evens.iter().copied().take(10).collect::<Vec<_>>(),
            [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
        );
    }
}
