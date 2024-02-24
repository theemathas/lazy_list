#![warn(missing_docs)]
#![allow(clippy::missing_panics_doc, clippy::module_name_repetitions)]
//! This crate provides lazily-populated lists, infinite or infinite.
//!
//! For usage examples, see the documentation for the [`inf_list`] module.

mod chunked_vec;
pub mod inf_list;

pub use inf_list::{InfList, InfListBoxed, InfListOwned, IteratorInfExt};
