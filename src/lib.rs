#![warn(missing_docs)]
#![allow(clippy::missing_panics_doc, clippy::module_name_repetitions)]
//! This crate provides lazily-populated lists, infinite or infinite.
//!
//! For usage examples, see the documentation for the [`inf_list`] module.

mod chunked_vec;
pub mod inf_list;

pub use inf_list::{InfList, InfListBoxed, InfListOwned};

/// Extension trait for [`Iterator`], providing lazy list operations.
pub trait IteratorLazyExt: Iterator + Sized {
    /// Collects the elements of an iterator into an [`InfList`]. If the iterator
    /// isn't infinite, operations on the resulting `InfList` might panic.
    ///
    /// Equivalent to [`InfList::new`].
    fn collect_inf<T>(self) -> InfList<T, Self>;
}
impl<I: Iterator + Sized> IteratorLazyExt for I {
    fn collect_inf<T>(self) -> InfList<T, Self> {
        InfList::new(self)
    }
}
