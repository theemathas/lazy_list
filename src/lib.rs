#![warn(missing_docs)]
#![allow(clippy::missing_panics_doc)]
//! This crate provides lazily-populated lists, infinite or infinite.
//!
//! For usage examples, see the documentation for the [`inf_list`] module.

mod chunked_vec;
pub mod inf_list;

pub use inf_list::{InfVec, InfVecBoxed, InfVecOwned, IteratorInfExt};
