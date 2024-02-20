//! The [`Producer`] trait and some implementations.

#![allow(clippy::module_name_repetitions)]

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
#[derive(Debug)]
pub struct FnMutProducer<F>(pub F);

impl<T, F: FnMut() -> T> Producer<T> for FnMutProducer<F> {
    fn produce(&mut self) -> T {
        self.0()
    }
}

/// A [`Producer`] that produces elements from an iterator which is assumed to
/// be infinite. Panics if the iterator runs out of elements.
#[derive(Debug)]
pub struct IteratorProducer<I>(pub I);

impl<I: Iterator> Producer<I::Item> for IteratorProducer<I> {
    fn produce(&mut self) -> I::Item {
        self.0
            .next()
            .expect("An IteratorProducer should never run out of elements")
    }
}
