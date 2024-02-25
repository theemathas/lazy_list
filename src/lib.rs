#![warn(missing_docs)]
#![allow(clippy::missing_panics_doc, clippy::module_name_repetitions)]
//! This crate provides lazily-populated lists, finite or infinite.
//!
//! For usage examples, see the documentation for the [`lazy_list`] and
//! [`inf_list`] modules.

mod chunked_vec;
pub mod lazy_list;

pub use lazy_list::{LazyList, LazyListBoxed, LazyListOwned};

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
