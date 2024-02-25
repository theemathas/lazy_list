#![warn(missing_docs)]
#![allow(clippy::missing_panics_doc, clippy::module_name_repetitions)]
//! This crate provides lazily-populated lists, finite or infinite.
//!
//! For usage examples, see the documentation for the [`lazy_list`] and
//! [`inf_list`] modules.

mod chunked_vec;
pub mod inf_list;
pub mod lazy_list;

pub use inf_list::{InfList, InfListBoxed, InfListOwned};
pub use lazy_list::{LazyList, LazyListBoxed, LazyListOwned};

/// Extension trait for [`Iterator`], providing lazy list operations.
pub trait IteratorLazyExt: Iterator + Sized {
    /// Collects the elements of an iterator into an [`InfList`]. If the iterator
    /// isn't infinite, operations on the resulting `InfList` might panic.
    ///
    /// Equivalent to [`InfList::new`].
    fn collect_inf<T>(self) -> InfList<T, Self>;
    /// Collects the elements of an iterator into a [`LazyList`].
    ///
    /// Equivalent to [`LazyList::new`].
    fn collect_lazy<T>(self) -> LazyList<T, Self>;
}
impl<I: Iterator + Sized> IteratorLazyExt for I {
    fn collect_inf<T>(self) -> InfList<T, Self> {
        InfList::new(self)
    }
    fn collect_lazy<T>(self) -> LazyList<T, Self> {
        LazyList::new(self)
    }
}
