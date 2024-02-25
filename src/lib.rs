#![warn(missing_docs)]
#![allow(clippy::missing_panics_doc, clippy::module_name_repetitions)]
//! This crate provides [`LazyList`], A lazily-populated potentially-infinite
//! list.
//!
//! An `LazyList<T, I>` can be indexed and have its elements modified, similarly
//! to a `Vec<T>`. However, elements are produced on-demand by a
//! potentially-infinite iterator with type `I` which is specified upon creation
//! of the `LazyList`. Once an element is produced, it is cached for later
//! access.
//!
//! If you don't want to specify an iterator type as a type parameter, you can
//! use the [`LazyListBoxed`] or [`LazyListOwned`] type aliases.
//!
//! `LazyList` is currently invariant, as opposed to covariant, if that matters
//! to you.
//!
//! An immutable `LazyList` is thread-safe.
//!
//! Internally, `LazyList` stores elements in 64-element chunks. This is so that
//! we can hand out references to elements, and not have them be invalidated as
//! more elements are added to the cache.
//!
//! # Examples
//!
//! Basic usage:
//! ```
//! use lazy_list::LazyList;
//!
//! // Finite list
//! let list: LazyList<i32, _> = LazyList::new(0..100);
//! assert_eq!(list.into_iter().sum::<i32>(), 4950);
//! // Infinite list
//! let list: LazyList<i32, _>  = LazyList::new(0..);
//! assert_eq!(list.into_iter().take(100).sum::<i32>(), 4950);
//! ```
//!
//! Mutation of an `LazyList`:
//! ```
//! use lazy_list::LazyList;
//!
//! let mut list = LazyList::new(0..);
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
//! Reusing a static `LazyList`:
//! ```
//! use lazy_list::{LazyList, LazyListOwned, IteratorLazyExt};
//! use once_cell::sync::Lazy;
//!
//! // Note that each element will only ever be produced once.
//! static EVENS: Lazy<LazyListOwned<i32>> =
//!     Lazy::new(|| (0..).map(|x| x * 2).collect_lazy().boxed());
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
//! Recursive `LazyList`:
//! ```
//! use lazy_list::{LazyList, LazyListBoxed};
//! use std::sync::Arc;
//!
//! let fibonacci: Arc<LazyListBoxed<i32>> = LazyList::recursive(|fibonacci_ref, i| {
//!     if i < 2 {
//!         Some(1)
//!     } else {
//!         Some(fibonacci_ref[i - 1] + fibonacci_ref[i - 2])
//!     }
//! });
//! assert_eq!(
//!     fibonacci.iter().take(10).copied().collect::<Vec<_>>(),
//!     [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
//! );
//! ```

mod chunked_vec;

// TODO: Debug impl for iterators

use std::{
    fmt::{self, Debug},
    iter,
    ops::{Index, IndexMut},
    option,
    sync::{Arc, Mutex},
};

use crate::chunked_vec::ChunkedVec;

/// A lazily-populated list.
///
/// See the [crate-level documentation](crate) for more information.
pub struct LazyList<T, I> {
    cached: ChunkedVec<T>,
    // A None here means the iterator is exhausted.
    remaining: Mutex<Option<I>>,
}

/// Type alias for a [`LazyList`] with an unknown iterator type, which might
/// borrow data with lifetime `'a`.
///
/// This type can be constructed via [`LazyList::boxed`], or by manually boxing
/// the iterator.
///
/// In most cases, [`LazyListOwned`] is the type you want.
pub type LazyListBoxed<'a, T> = LazyList<T, Box<dyn Iterator<Item = T> + Send + 'a>>;

/// Type alias for a [`LazyList`] with an unknown iterator type, which has no
/// borrowed data.
///
/// This type can be constructed via [`LazyList::boxed`], or by manually boxing
/// the iterator.
pub type LazyListOwned<T> = LazyListBoxed<'static, T>;

/// Iterator over immutable references to the elements of a [`LazyList`].
///
/// This struct is created by the [`iter`](LazyList::iter) method on
/// [`LazyList`].
pub struct Iter<'a, T, I> {
    list: &'a LazyList<T, I>,
    inner: chunked_vec::Iter<'a, T>,
}

/// Iterator over mutable references to the elements of a [`LazyList`].
///
/// This struct is created by the [`iter_mut`](LazyList::iter_mut) method on
/// [`LazyList`].
pub struct IterMut<'a, T, I> {
    list: &'a LazyList<T, I>,
    inner: chunked_vec::IterMut<'a, T>,
}

/// Iterator that moves elements out of an [`LazyList`].
///
/// This struct is created by the `into_iter` method on [`LazyList`].
pub struct IntoIter<T, I: Iterator>(
    iter::Chain<chunked_vec::IntoIter<T>, iter::Flatten<option::IntoIter<I>>>,
);

impl<T, I> LazyList<T, I> {
    /// Creates an [`LazyList`] from an iterator. The resulting `LazyList`
    /// conceptually contains the list of elements that the iterator would
    /// produce.
    ///
    /// Equivalent to [`crate::IteratorLazyExt::collect_lazy`].
    pub const fn new(iterator: I) -> LazyList<T, I> {
        LazyList {
            cached: ChunkedVec::new(),
            remaining: Mutex::new(Some(iterator)),
        }
    }

    /// Returns the number of elements that have been produced and cached so far.
    pub fn num_cached(&self) -> usize {
        self.cached.len()
    }
}

impl<'a, T: Send + Sync + 'a> LazyListBoxed<'a, T> {
    /// Creates a recursively-defined `LazyList`. The closure should take a
    /// reference to the `LazyList` itself and an index, then return the element
    /// at that index, or `None` if there are no more elements. The closure
    /// should only attempt to access prior elements of the `LazyList`, or a
    /// deadlock will occur.
    pub fn recursive<F: FnMut(&LazyListBoxed<'a, T>, usize) -> Option<T> + Send + 'a>(
        mut f: F,
    ) -> Arc<LazyListBoxed<'a, T>> {
        Arc::new_cyclic(|weak| {
            let weak = weak.clone();
            LazyList::new((0..).map_while(move |i| f(&weak.upgrade().unwrap(), i))).boxed()
        })
    }
}

impl<'a, T, I: Iterator<Item = T> + Send + 'a> LazyList<T, I> {
    /// Returns a boxed version of the `LazyList`. This is useful when you don't
    /// want to write out the iterator type as a type parameter.
    pub fn boxed(self) -> LazyListBoxed<'a, T> {
        LazyList {
            cached: self.cached,
            remaining: Mutex::new(
                self.remaining
                    .into_inner()
                    .unwrap()
                    .map(|iter| Box::new(iter) as Box<dyn Iterator<Item = T> + Send + 'a>),
            ),
        }
    }
}

impl<T, I: Iterator<Item = T>> Index<usize> for LazyList<T, I> {
    type Output = T;

    fn index(&self, index: usize) -> &T {
        self.get(index).expect("index out of bounds")
    }
}

impl<T, I: Iterator<Item = T>> IndexMut<usize> for LazyList<T, I> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        self.get_mut(index).expect("index out of bounds")
    }
}

impl<T, I: Iterator<Item = T>> LazyList<T, I> {
    /// Ensures that element `index`, if such an element exists, is cached.
    fn ensure_cached(&self, index: usize) {
        // Don't lock if the element is already cached.
        if self.cached.len() > index {
            return;
        }
        let mut guard = self.remaining.lock().unwrap();
        let iter_option: &mut Option<I> = &mut guard;
        while self.cached.len() <= index {
            let element_option: Option<T> = iter_option.as_mut().and_then(Iterator::next);
            if let Some(element) = element_option {
                self.cached.push(element);
            } else {
                *iter_option = None;
                break;
            }
        }
    }

    /// Returns a reference to the element at index `index`, or `None` if the
    /// index is out of bounds.
    pub fn get(&self, index: usize) -> Option<&T> {
        self.ensure_cached(index);
        if index < self.cached.len() {
            // SAFETY: Shared reference to self ensure that no other mutable references to contents exist.
            unsafe { Some(self.cached.index(index)) }
        } else {
            None
        }
    }

    /// Returns a mutable reference to the element at index `index`, or `None`
    /// if the index is out of bounds.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        self.ensure_cached(index);
        if index < self.cached.len() {
            // SAFETY: Exclusive reference to self ensure that no other
            // references to contents exist.
            unsafe { Some(self.cached.index_mut(index)) }
        } else {
            None
        }
    }

    /// Returns an iterator over the elements of the `LazyList`.
    pub fn iter(&self) -> Iter<T, I> {
        Iter {
            list: self,
            // SAFETY: Shared reference to self ensures that no other mutable
            // references to contents exist.
            inner: unsafe { self.cached.iter() },
        }
    }

    /// Returns a mutable iterator over the elements of the `LazyList`.
    pub fn iter_mut(&mut self) -> IterMut<T, I> {
        IterMut {
            list: self,
            // SAFETY: Exclusive reference to self ensure that no other
            // references to contents exist.
            inner: unsafe { self.cached.iter_mut() },
        }
    }
}

impl<T, I: Iterator<Item = T>> IntoIterator for LazyList<T, I> {
    type Item = T;
    type IntoIter = IntoIter<T, I>;

    fn into_iter(self) -> IntoIter<T, I> {
        IntoIter(
            self.cached
                .into_iter()
                .chain(self.remaining.into_inner().unwrap().into_iter().flatten()),
        )
    }
}

impl<'a, T, I: Iterator<Item = T>> IntoIterator for &'a LazyList<T, I> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T, I>;

    fn into_iter(self) -> Iter<'a, T, I> {
        self.iter()
    }
}

impl<'a, T, I: Iterator<Item = T>> IntoIterator for &'a mut LazyList<T, I> {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T, I>;

    fn into_iter(self) -> IterMut<'a, T, I> {
        self.iter_mut()
    }
}

impl<'a, T, I: Iterator<Item = T>> Iterator for Iter<'a, T, I> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        self.list.ensure_cached(self.inner.next_index());
        self.inner.next()
    }
}

impl<'a, T, I: Iterator<Item = T>> Iterator for IterMut<'a, T, I> {
    type Item = &'a mut T;

    fn next<'b>(&'b mut self) -> Option<&'a mut T> {
        self.list.ensure_cached(self.inner.next_index());
        self.inner.next()
    }
}

impl<T, I: Iterator<Item = T>> Iterator for IntoIter<T, I> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        self.0.next()
    }
}

impl<T: Debug, I> Debug for LazyList<T, I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Hack until DebugList::entry_with is stabilized
        struct DebugEllipsis;
        impl Debug for DebugEllipsis {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("...")
            }
        }

        let mut debug_list = f.debug_list();
        debug_list.entries((0..self.cached.len()).map(|i| unsafe {
            // SAFETY: Shared reference to self ensures that no other mutable
            // references to contents exist.
            self.cached.index(i)
        }));
        let has_remaining = self.remaining.lock().unwrap().is_some();
        if has_remaining {
            debug_list.entry(&DebugEllipsis);
        }
        debug_list.finish()
    }
}

/// Extension trait for [`Iterator`], providing lazy list operations.
pub trait IteratorLazyExt: Iterator + Sized {
    /// Collects the elements of an iterator into a [`LazyList`].
    ///
    /// Equivalent to [`LazyList::new`].
    fn collect_lazy<T>(self) -> LazyList<T, Self>;
}
impl<I: Iterator + Sized> IteratorLazyExt for I {
    fn collect_lazy<T>(self) -> LazyList<T, Self> {
        LazyList::new(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{iter, sync::atomic::AtomicUsize};

    #[test]
    fn test_indexing_infinite() {
        let mut list = LazyList::new(0..);
        for i in 0..100 {
            assert_eq!(list[i], i);
        }
        for i in 0..100 {
            assert_eq!(list[i], i);
        }
        for i in 0..200 {
            list[i] = 10 * i;
        }
        for i in 0..200 {
            assert_eq!(list[i], 10 * i);
        }
    }

    #[test]
    fn test_indexing_finite() {
        let mut list = LazyList::new(0..200);
        for i in 0..100 {
            assert_eq!(list[i], i);
        }
        for i in 0..100 {
            assert_eq!(list[i], i);
        }
        for i in 0..200 {
            list[i] = 10 * i;
        }
        for i in 0..200 {
            assert_eq!(list[i], 10 * i);
        }
        for i in 200..300 {
            assert_eq!(list.get(i), None);
        }
    }

    #[test]
    fn test_boxed() {
        let list: LazyListBoxed<'_, _> = LazyList::new(0..100).boxed();
        for i in 0..100 {
            assert_eq!(list[i], i);
        }
        for i in 100..200 {
            assert_eq!(list.get(i), None);
        }
    }

    #[test]
    fn test_iter_infinite() {
        let list = LazyList::new(0..);
        let mut iter = list.iter();
        for i in 0..100 {
            assert_eq!(iter.next(), Some(&(i)));
        }
        let mut iter2 = list.iter();
        for i in 0..200 {
            assert_eq!(iter2.next(), Some(&(i)));
        }
        for i in 0..200 {
            assert_eq!(iter.next(), Some(&(i + 100)));
        }
    }

    #[test]
    fn test_iter_finite() {
        let list = LazyList::new(0..300);
        let mut iter = list.iter();
        for i in 0..100 {
            assert_eq!(iter.next(), Some(&(i)));
        }
        let mut iter2 = list.iter();
        for i in 0..200 {
            assert_eq!(iter2.next(), Some(&(i)));
        }
        for i in 0..200 {
            assert_eq!(iter.next(), Some(&(i + 100)));
        }
        assert_eq!(iter.next(), None);
        // Doesn't cause UB
        for _ in 0..100 {
            let _ = iter.next();
        }
    }

    #[test]
    fn test_iter_mut() {
        let mut list = LazyList::new(0..);
        let mut iter = list.iter_mut();
        for i in 0..100 {
            let element_ref = iter.next().unwrap();
            assert_eq!(element_ref, &i);
            *element_ref += 1000;
        }
        let mut iter2 = list.iter_mut();
        for i in 0..100 {
            assert_eq!(iter2.next(), Some(&mut (i + 1000)));
        }
        for i in 0..100 {
            assert_eq!(iter2.next(), Some(&mut (i + 100)));
        }
    }

    #[test]
    fn test_into_iter_infinite() {
        struct DropCounter<'a>(&'a AtomicUsize, usize);
        impl Drop for DropCounter<'_> {
            fn drop(&mut self) {
                self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }

        let drop_counter = AtomicUsize::new(0);
        let drop_counter_ref = &drop_counter;
        let mut x = 0;
        let list = LazyList::new(iter::repeat_with(move || {
            let result = x;
            x += 1;
            DropCounter(drop_counter_ref, result)
        }));

        let mut iter = list.iter();
        for _ in 0..200 {
            // Fill the cache
            iter.next();
        }
        assert_eq!(drop_counter.load(std::sync::atomic::Ordering::Relaxed), 0);
        let mut into_iter = list.into_iter();
        for i in 0..100 {
            assert_eq!(into_iter.next().map(|x| x.1), Some(i));
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
        let list = LazyList::new(0..);
        assert_eq!(format!("{:?}", list), "[...]");
        list[9];
        assert_eq!(format!("{:?}", list), "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ...]");
        let list2 = LazyList::new(0..10);
        assert_eq!(format!("{:?}", list2), "[...]");
        list2[9];
        assert_eq!(
            format!("{:?}", list2),
            "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ...]"
        );
        list2.get(10);
        assert_eq!(format!("{:?}", list2), "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]");
    }

    #[test]
    fn test_recursive() {
        let evens =
            LazyList::recursive(|evens_ref, i| Some(if i == 0 { 0 } else { evens_ref[i - 1] + 2 }));
        assert_eq!(
            evens.iter().copied().take(10).collect::<Vec<_>>(),
            [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
        );
    }
}
