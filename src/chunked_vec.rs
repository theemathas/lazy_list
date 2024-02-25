use std::{
    array,
    marker::PhantomData,
    mem::{ManuallyDrop, MaybeUninit},
    ptr::addr_of,
    sync::{
        atomic::{AtomicUsize, Ordering::SeqCst},
        Mutex,
    },
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
    /// * No references to the ChunkedVecInner exist except as allowed by the
    ///   mutex. Mutating the len field also requires holding the mutex lock.
    /// * No references to the contents of the Vec exist at any point.
    /// * Any number of mutable or shared references to the contents of the
    ///   chunks may exist, but they must obey the usual aliasing rules.
    inner: Mutex<ChunkedVecInner<T>>,
    len: AtomicUsize,
    _marker: PhantomData<Vec<T>>,
}

struct ChunkedVecInner<T> {
    chunks: Vec<Chunk<T>>,
}

pub struct Iter<'a, T> {
    chunked_vec: &'a ChunkedVec<T>,
    next_index: usize,
    /// SAFETY: This, if Some, is a valid pointer pointing to where the next
    /// element would be if it existed.
    next_ptr: Option<*const MaybeUninit<T>>,
}

pub struct IterMut<'a, T> {
    chunked_vec: &'a ChunkedVec<T>,
    next_index: usize,
    /// SAFETY: This, if Some, is a valid pointer pointing to where the next
    /// element would be if it existed.
    next_ptr: Option<*mut MaybeUninit<T>>,
}

pub struct IntoIter<T> {
    /// Same as `ChunkedVec`, except that in the SAFETY invariant, the
    /// initialized values are at indexes `current..len`, and there are no
    /// restrictions on when references to the `ChunkedVecInner` may exist.
    inner: ChunkedVecInner<T>,
    len: AtomicUsize,
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
            inner: Mutex::new(ChunkedVecInner { chunks: Vec::new() }),
            len: AtomicUsize::new(0),
            _marker: PhantomData,
        }
    }

    /// The number of elements in the `ChunkedVec`.
    pub fn len(&self) -> usize {
        self.len.load(SeqCst)
    }

    /// Appends an element to the end of the `ChunkedVec`.
    pub fn push(&self, value: T) {
        unsafe {
            // SAFETY: After locking, we gain mutable access to the len.
            let mut inner = self.inner.lock().unwrap();
            // If capacity is full, add a new chunk
            if self.len() == inner.chunks.len() * CHUNK_SIZE {
                inner
                    .chunks
                    .push(Box::new(array::from_fn(|_| MaybeUninit::uninit())));
            }
            let new_element: *mut MaybeUninit<T> = inner.index_ptr(self.len());
            new_element.write(MaybeUninit::new(value));
            self.len.fetch_add(1, SeqCst);
        }
    }

    /// Returns a reference to the i'th element of the `ChunkedVec`. Panics if
    /// out of bounds.
    ///
    /// SAFETY: The caller must ensure that nobody is holding a mutable
    /// reference to this exact element.
    pub unsafe fn index(&self, index: usize) -> &T {
        assert!(index < self.len(), "index out of bounds");
        let mut inner = self.inner.lock().unwrap();
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
        assert!(index < self.len(), "index out of bounds");
        let mut inner = self.inner.lock().unwrap();
        let ptr: *mut MaybeUninit<T> = inner.index_ptr(index);
        // SAFETY: This index is already initialized, and won't be moved.
        (*ptr).as_mut_ptr().as_mut().unwrap()
    }

    /// Returns an iterator over the elements of the `ChunkedVec`.
    ///
    /// SAFETY: The caller must ensure that nobody is holding a mutable
    /// reference to any element of the `ChunkedVec`.
    pub unsafe fn iter(&self) -> Iter<'_, T> {
        Iter {
            chunked_vec: self,
            next_index: 0,
            next_ptr: None,
        }
    }

    /// Returns a mutable iterator over the elements of the `ChunkedVec`.
    ///
    /// SAFETY: The caller must ensure that nobody is holding a reference to
    /// any element of the `ChunkedVec`.
    pub unsafe fn iter_mut(&self) -> IterMut<'_, T> {
        IterMut {
            chunked_vec: self,
            next_index: 0,
            next_ptr: None,
        }
    }
}

impl<T> IntoIterator for ChunkedVec<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;
    /// Returns an iterator that consumes the `ChunkedVec`.
    fn into_iter(self) -> IntoIter<T> {
        // SAFETY: No other references to the `ChunkedVecInner` exist.
        let self_manually_drop = ManuallyDrop::new(self);
        let mutex = unsafe { addr_of!(self_manually_drop.inner).read() };
        let inner = mutex.into_inner().unwrap();
        unsafe {
            IntoIter {
                inner,
                len: addr_of!(self_manually_drop.len).read(),
                current: 0,
            }
        }
    }
}

impl<T> Drop for ChunkedVec<T> {
    fn drop(&mut self) {
        unsafe {
            let inner: &mut ChunkedVecInner<T> = &mut self.inner.lock().unwrap();
            // SAFETY: We're dropping self already, so nobody else can have
            // references to things inside it.
            for i in 0..self.len() {
                (*inner.index_ptr(i)).assume_init_drop();
            }
        }
        // The Vec and the chunks are automatically dropped.
    }
}

impl<T> Drop for IntoIter<T> {
    fn drop(&mut self) {
        unsafe {
            // SAFETY: We're dropping self already, so nobody else can have
            // references to things inside it.
            for i in self.current..self.len.load(SeqCst) {
                (*self.inner.index_ptr(i)).assume_init_drop();
            }
        }
        // The Vec and the chunks are automatically dropped.
    }
}

impl<'a, T> Iter<'a, T> {
    pub fn next_index(&self) -> usize {
        self.next_index
    }
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<&'a T> {
        if self.next_index < self.chunked_vec.len() {
            let current_ptr = self.next_ptr.unwrap_or_else(|| {
                self.chunked_vec
                    .inner
                    .lock()
                    .unwrap()
                    .index_ptr(self.next_index)
            });
            self.next_index += 1;
            self.next_ptr = if self.next_index % CHUNK_SIZE == 0 {
                // End of the current chunk
                None
            } else {
                // SAFETY: We're not at the end of the current chunk.
                unsafe { Some(current_ptr.add(1)) }
            };
            // SAFETY: The client must ensure that nobody is holding a mutable
            // reference to any element of the `ChunkedVec`.
            Some(unsafe { &*current_ptr.cast::<T>() })
        } else {
            None
        }
    }
}

impl<'a, T> IterMut<'a, T> {
    pub fn next_index(&self) -> usize {
        self.next_index
    }
}

impl<'a, T> Iterator for IterMut<'a, T> {
    type Item = &'a mut T;
    fn next(&mut self) -> Option<&'a mut T> {
        if self.next_index < self.chunked_vec.len() {
            let current_ptr = self.next_ptr.unwrap_or_else(|| {
                self.chunked_vec
                    .inner
                    .lock()
                    .unwrap()
                    .index_ptr(self.next_index)
            });
            self.next_index += 1;
            self.next_ptr = if self.next_index % CHUNK_SIZE == 0 {
                // End of the current chunk
                None
            } else {
                // SAFETY: We're not at the end of the current chunk.
                unsafe { Some(current_ptr.add(1)) }
            };
            // SAFETY: The client must ensure that nobody is holding a reference
            // to any element of the `ChunkedVec`. And we increment stuff so
            // this element isn't returned from this iterator again.
            Some(unsafe { &mut *current_ptr.cast::<T>() })
        } else {
            None
        }
    }
}

impl<T> Iterator for IntoIter<T> {
    type Item = T;
    fn next(&mut self) -> Option<T> {
        if self.current < self.len.load(SeqCst) {
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
    fn test_iter() {
        let vec = ChunkedVec::new();
        for i in 0..1000 {
            vec.push(i);
        }
        let mut iter = unsafe { vec.iter() };
        for i in 0..1000 {
            assert_eq!(iter.next(), Some(&i));
        }
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_iter_mut() {
        let vec = ChunkedVec::new();
        for i in 0..1000 {
            vec.push(i);
        }
        let mut iter = unsafe { vec.iter_mut() };
        for i in 0..1000 {
            assert_eq!(iter.next(), Some(&mut { i }));
        }
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_iter_mut_always_near_end() {
        let vec = ChunkedVec::new();
        let mut iter = unsafe { vec.iter_mut() };
        for i in 0..1000 {
            vec.push(i);
            assert_eq!(iter.next(), Some(&mut { i }));
        }
        assert_eq!(iter.next(), None);
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
