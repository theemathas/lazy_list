use std::{
    array,
    cell::UnsafeCell,
    marker::PhantomData,
    mem::{ManuallyDrop, MaybeUninit},
};

/// Number of elements in a chunk
const CHUNK_SIZE: usize = 64;

type ChunkArray<T> = [MaybeUninit<T>; CHUNK_SIZE];
type Chunk<T> = Box<ChunkArray<T>>;

/// A `Vec`-like data structure that stores elements in chunks of fixed size. It
/// supports mutation only by appending elements to the end. As a result, once
/// an element is pushed into a `ChunkedVec`, it will never be moved for as long
/// as the `ChunkedVec` exists. It has interior mutability, so its elements `T`
/// may be mutated through a `ChunkedVec`.
pub struct ChunkedVec<T> {
    /// The i'th element is at `chunks[i / CHUNK_SIZE][i % CHUNK_SIZE]`.
    ///
    /// SAFETY invariant: Each chunk points to a valid allocation, and the first
    /// `len` elements of this `ChunkedVec` are initialized.
    ///
    /// SAFETY invariant:
    /// * No references to the ChunkedVecInner exist except when a method is
    ///   running.
    /// * No references to the ChunkedVecInner or to the contents of the Vec
    ///   exist at any point.
    /// * Any number of mutable or shared references to the contents of the Vec
    ///   may exist, but they must obey the usual aliasing rules.
    inner: UnsafeCell<ChunkedVecInner<T>>,
    _marker: PhantomData<Vec<T>>,
}

struct ChunkedVecInner<T> {
    chunks: Vec<Chunk<T>>,
    len: usize,
}

pub struct IntoIter<T> {
    /// Same as `ChunkedVec`, except that in the SAFETY invariant, the
    /// initialized values are at indexes `current..len`, and there are no
    /// restrictions on when references to the `ChunkedVecInner` may exist.
    inner: ChunkedVecInner<T>,
    current: usize,
}

impl<T> ChunkedVecInner<T> {
    /// Returns a pointer to the i'th element of the `ChunkedVec`. Panics if the
    /// index is greater than the capacity.
    ///
    /// SAFETY: Always returns a valid pointer. The data there might or might
    /// not be initialized.
    fn index_ptr(&mut self, index: usize) -> *mut MaybeUninit<T> {
        unsafe {
            assert!(
                index / CHUNK_SIZE < self.chunks.len(),
                "index greater than capacity"
            );
            // SAFETY: No references to the contents of the Vec.
            let chunk: *mut ChunkArray<T> = self
                .chunks
                .as_mut_ptr()
                .add(index / CHUNK_SIZE)
                .cast::<*mut ChunkArray<T>>()
                .read();
            chunk.cast::<MaybeUninit<T>>().add(index % CHUNK_SIZE)
        }
    }
}

impl<T> ChunkedVec<T> {
    /// Creates a new, empty `ChunkedVec`.
    pub const fn new() -> Self {
        Self {
            inner: UnsafeCell::new(ChunkedVecInner {
                chunks: Vec::new(),
                len: 0,
            }),
            _marker: PhantomData,
        }
    }

    /// The number of elements in the `ChunkedVec`.
    pub fn len(&self) -> usize {
        // SAFETY: No other references to the `ChunkedVecInner` exist.
        unsafe { (*self.inner.get()).len }
    }

    /// Appends an element to the end of the `ChunkedVec`.
    pub fn push(&self, value: T) {
        unsafe {
            // SAFETY: No other references to the `ChunkedVecInner` exist.
            let inner: &mut ChunkedVecInner<T> = &mut *self.inner.get();
            // If capacity is full, add a new chunk
            if inner.len == inner.chunks.len() * CHUNK_SIZE {
                inner
                    .chunks
                    .push(Box::new(array::from_fn(|_| MaybeUninit::uninit())));
            }
            let new_element: *mut MaybeUninit<T> = inner.index_ptr(inner.len);
            new_element.write(MaybeUninit::new(value));
            inner.len += 1;
        }
    }

    /// Returns a reference to the i'th element of the `ChunkedVec`. Panics if
    /// out of bounds.
    ///
    /// SAFETY: The caller must ensure that nobody is holding a mutable
    /// reference to this exact element.
    pub unsafe fn index(&self, index: usize) -> &T {
        // SAFETY: No other references to the `ChunkedVecInner` exist.
        let inner: &mut ChunkedVecInner<T> = &mut *self.inner.get();
        assert!(index < inner.len, "index out of bounds");
        let ptr: *mut MaybeUninit<T> = inner.index_ptr(index);
        // SAFETY: This index is already initialized, and won't be moved.
        (*ptr).as_ptr().as_ref().unwrap()
    }

    /// Returns a mutable reference to the i'th element of the `ChunkedVec`.
    /// Panics if out of bounds.
    ///
    /// SAFETY: The caller must ensure that nobody is holding a reference to
    /// this exact element.
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn index_mut(&self, index: usize) -> &mut T {
        // SAFETY: No other references to the `ChunkedVecInner` exist.
        let inner: &mut ChunkedVecInner<T> = &mut *self.inner.get();
        assert!(index < inner.len, "index out of bounds");
        let ptr: *mut MaybeUninit<T> = inner.index_ptr(index);
        // SAFETY: This index is already initialized, and won't be moved.
        (*ptr).as_mut_ptr().as_mut().unwrap()
    }
}

impl<T> IntoIterator for ChunkedVec<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;
    /// Returns an iterator that consumes the `ChunkedVec`.
    fn into_iter(self) -> IntoIter<T> {
        // SAFETY: No other references to the `ChunkedVecInner` exist.
        unsafe {
            let self_manually_drop = ManuallyDrop::new(self);
            let inner_ptr: *mut ChunkedVecInner<T> = self_manually_drop.inner.get();
            IntoIter {
                inner: inner_ptr.read(),
                current: 0,
            }
        }
    }
}

impl<T> Drop for ChunkedVec<T> {
    fn drop(&mut self) {
        unsafe {
            // SAFETY: No other references to the `ChunkedVecInner` exist.
            let inner: &mut ChunkedVecInner<T> = &mut *self.inner.get();
            // SAFETY: We're dropping self already, so nobody else can have
            // references to things inside it.
            for i in 0..inner.len {
                (*inner.index_ptr(i)).assume_init_drop();
            }
        }
        // The Vec and the chunks are automatically dropped.
    }
}

impl<T> Drop for IntoIter<T> {
    fn drop(&mut self) {
        unsafe {
            // SAFETY: No other references to the `ChunkedVecInner` exist.
            // SAFETY: We're dropping self already, so nobody else can have
            // references to things inside it.
            for i in self.current..self.inner.len {
                (*self.inner.index_ptr(i)).assume_init_drop();
            }
        }
        // The Vec and the chunks are automatically dropped.
    }
}

impl<T> Iterator for IntoIter<T> {
    type Item = T;
    fn next(&mut self) -> Option<T> {
        if self.current < self.inner.len {
            let ptr = self.inner.index_ptr(self.current);
            self.current += 1;
            // SAFETY: The index is initialized, and incrementing the index
            // means it won't be consumed again.
            Some(unsafe { ptr.read().assume_init() })
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunked_vec_basic() {
        let vec = ChunkedVec::new();
        vec.push(1);
        vec.push(2);
        vec.push(3);
        assert_eq!(vec.len(), 3);
        let ref0 = unsafe { vec.index(0) };
        let ref1 = unsafe { vec.index(1) };
        let ref2 = unsafe { vec.index(2) };
        assert_eq!(*ref0, 1);
        assert_eq!(*ref1, 2);
        assert_eq!(*ref2, 3);
        assert_eq!(*ref0, 1);
        assert_eq!(*ref1, 2);
        assert_eq!(*ref2, 3);
        // This invalidates ref0
        let ref0_mut = unsafe { vec.index_mut(0) };
        *ref0_mut = 4;
        assert_eq!(*ref0_mut, 4);
        assert_eq!(*ref1, 2);
        assert_eq!(*ref2, 3);
    }

    #[test]
    fn test_chunked_vec_large() {
        let vec = ChunkedVec::new();
        for i in 0..1000 {
            vec.push(i);
        }
        assert_eq!(vec.len(), 1000);
        let mut refs_vec = Vec::new();
        for i in 0..1000 {
            let ref_i = unsafe { vec.index(i) };
            assert_eq!(*ref_i, i);
            refs_vec.push(ref_i);
        }
        for i in 1000..2000 {
            vec.push(i);
        }
        assert_eq!(vec.len(), 2000);
        for i in 0..1000 {
            assert_eq!(*refs_vec[i], i);
        }
    }

    #[test]
    fn test_into_iter() {
        let vec = ChunkedVec::new();
        for i in 0..1000 {
            vec.push(i);
        }
        let mut iter = vec.into_iter();
        for i in 0..1000 {
            assert_eq!(iter.next(), Some(i));
        }
        assert_eq!(iter.next(), None);

        let vec = ChunkedVec::new();
        for i in 0..1000 {
            vec.push(i);
        }
        let mut iter = vec.into_iter();
        for i in 0..500 {
            assert_eq!(iter.next(), Some(i));
        }
        // Drop the iterator
    }
}
