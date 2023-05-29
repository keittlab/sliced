#![allow(dead_code)]

use std::{
    ops::{Index, IndexMut, Range},
    ptr,
    slice::{Chunks, ChunksMut, Iter, IterMut},
};

/// A segmented vector for iterating over slices of constant length.
///
/// Storing vectors within vectors is convenient but means that each
/// stored vector will allocate on the heap and drop when removed. VecVec
/// stores constant-length segments within a single vector so that `push`
/// within the storage capacity will not allocate and `truncate` will not
/// deallocate from the heap. Benchmarks indicate that this strategy is not
/// always faster for repeated cycles of `push` and `swap_remove`. This is
/// likely because the overhead of swapping a larger number of elements. `Vec`
/// within `Vec` only has to swap the pointers of the stored `Vec` objects
/// whereas `VecVec` has to swap an entire segment of values. In a few cases,
/// `VecVec` has proven about twice as fast, but you will need to test your
/// cases. `VecVec` is nonetheless convenient for organizing segmented storage,
/// such as a collection of image rows, and so on.
///
/// # Example
///
/// ```
/// use rand::{rngs::SmallRng, Rng, SeedableRng};
/// use vecvec::VecVec;
/// let mut rng = SmallRng::from_entropy();
/// let mut x: VecVec<usize> = VecVec::with_capacity(100, 20);
/// let mut gen_vec = || std::iter::repeat_with(|| rng.gen()).take(20).collect::<Vec<usize>>();
/// for _ in 0..100 {
///     x.push(gen_vec().as_slice())
/// }
/// for _ in 0..100 {
///     for i in (0..50).step_by(2) {
///         x.swap_truncate(i);
///     }
///     for _ in 0..50 {
///         x.push(gen_vec().as_slice())
///     }
/// }
/// ```
#[derive(Debug)]
pub struct VecVec<T>
where
    T: Copy + Clone,
{
    storage: Vec<T>,
    segment_len: usize,
}

impl<T> VecVec<T>
where
    T: Copy + Clone,
{
    /// Initialize a `VecVec` and set the segment size.
    ///
    /// Panics if `segment_len` is zero.
    pub fn new(segment_len: usize) -> Self {
        assert_ne!(segment_len, 0);
        Self {
            storage: Vec::new(),
            segment_len,
        }
    }
    /// Initialize a `VecVec` and set the capacity and segment size.
    ///
    /// Panics if `segment_len` is zero.
    pub fn with_capacity(size: usize, segment_len: usize) -> Self {
        assert_ne!(segment_len, 0);
        Self {
            storage: Vec::with_capacity(size * segment_len),
            segment_len,
        }
    }
    /// Returns the number of internal segments
    pub fn len(&self) -> usize {
        self.storage.len() / self.segment_len
    }
    /// Get the capacity in number of segments
    pub fn capacity(&self) -> usize {
        self.storage_capacity() / self.segment_len
    }
    /// Returns the length of the underlying storage
    pub fn storage_len(&self) -> usize {
        self.storage.len()
    }
    /// Get the capacity of the underlying storage
    pub fn storage_capacity(&self) -> usize {
        self.storage.capacity()
    }
    /// Append the contents of another `VecVec`.
    ///
    /// Complexity is the length of `other`, plus any
    /// allocation required. `other` is drained after call.
    ///
    /// # Example
    ///
    /// ```
    /// use vecvec::{vecvec, VecVec};
    /// let mut a = vecvec![[1, 2, 3], [4, 5, 6]];
    /// let mut b = vecvec![[7, 8, 9], [3, 2, 1]];
    /// a.append(&mut b);
    /// assert_eq!(a.len(), 4);
    /// assert_eq!(b.len(), 0);
    /// ```
    ///
    ///  Panics if the segment size of `other` is different.
    pub fn append(&mut self, other: &mut Self) {
        assert_eq!(other.segment_len, self.segment_len);
        self.storage.append(&mut other.storage)
    }
    /// Insert a slice at position `index`.
    /// 
    /// Complexity is linear in `storage_len`.
    /// 
    /// Panics if `index` is out of bounds or if the
    /// length of `segment` is not the native segment
    /// size of the `VecVec`.
    pub fn insert(&mut self, index: usize, segment: &[T]) {
        assert!(index < self.len());
        assert_eq!(segment.len(), self.segment_len);
        let orig_last_index = self.last_index();
        self.storage.extend_from_within(self.storage_range_last());
        if index < orig_last_index {
            let src = self.storage_range_range(index, orig_last_index - 1);
            let dst = self.storage_begin(index + 1);
            self.storage.copy_within(src, dst);
        }
        unsafe { self.overwrite(index, segment) }
    }
    /// Add one or more segments to the end.
    /// 
    /// Complexity is amortized the segment size.
    /// 
    /// Panics if the length of the slice is not
    /// a multiple of the segment length.
    /// 
    /// # Example
    /// 
    /// ```
    /// use vecvec::*;
    /// let mut a = vecvec![[1, 2, 3]];
    /// a.push(&[4, 5, 6, 7, 8, 9]); // any multiple of segment length
    /// assert_eq!(a.len(), 3);
    /// assert_eq!(a.storage_len(), 9);
    /// ```
    /// 
    pub fn push(&mut self, segment: &[T]) {
        assert!(self.is_valid_length(segment));
        self.storage.extend_from_slice(segment)
    }
    /// Add one or more segments contained in a `Vec`.
    /// 
    /// Complexity is amortized the length of
    /// the slice.
    /// 
    /// Panics if the length of the slice is not
    /// a multiple of the segment length.
    pub fn push_vec(&mut self, segment: &Vec<T>) {
        self.push(segment.as_slice())
    }
    /// Get a reference to a segment.
    /// 
    /// Returns `None` if `index` is out of range.
    pub fn get(&self, index: usize) -> Option<&[T]> {
        self.storage.get(self.storage_range(index))
    }
    /// Get a mutable reference to a segment.
    /// 
    /// Returns `None` if `index` is out of range.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut [T]> {
        let range = self.storage_range(index);
        self.storage.get_mut(range)
    }
    /// Remove and return a segment.
    /// 
    /// Does not preserve the order of segments.
    /// Complexity is the segment length.
    /// 
    /// Panics if index is out of range.
    pub fn swap_remove(&mut self, index: usize) -> Vec<T> {
        debug_assert!(index < self.len());
        if index != self.last_index() {
            self.storage_range(index)
                .zip(self.storage_range_last())
                .for_each(|(i, j)| self.storage.swap(i, j))
        }
        self.storage
            .drain(self.storage_range_last())
            .as_slice()
            .into()
    }
    /// Swap a segment and truncate its storage.
    /// 
    /// Does not preserve the order of segments. The
    /// `VecVec` length will be reduced by one segment.
    /// Complexity is the segment length.
    /// 
    /// Panics if `index` is out of bounds.
    pub fn swap_truncate(&mut self, index: usize) {
        debug_assert!(index < self.len());
        if index != self.last_index() {
            let src = self.storage_range_last();
            let dst = self.storage_begin(index);
            self.storage.copy_within(src, dst)
        }
        self.storage.truncate(self.storage.len() - self.segment_len)
    }
    /// Non-order-preserving insert.
    /// 
    /// Appends the contents of the segment at `index`
    /// to the end of the storage and then overwrites
    /// the segment with the new values. Complexity is
    /// the twice the segment length.
    /// 
    /// Panics if `index` is out of bounds.
    pub fn swap_insert(&mut self, index: usize, segment: &[T]) {
        debug_assert!(index < self.len());
        assert_eq!(segment.len(), self.segment_len);
        self.storage.extend_from_within(self.storage_range(index));
        unsafe { self.overwrite(index, segment) }
    }
    /// Return a chunked iterator.
    /// 
    /// Allows iteration over segments as slices.
    pub fn iter(&self) -> Chunks<'_, T> {
        self.storage.chunks(self.segment_len)
    }
    /// Return a mutable chunked iterator.
    /// 
    /// Allows iteration and modification of segments.
    pub fn iter_mut(&mut self) -> ChunksMut<'_, T> {
        self.storage.chunks_mut(self.segment_len)
    }
    /// Iterate over the raw storage.
    pub fn iter_storage(&self) -> Iter<'_, T> {
        self.storage.iter()
    }
    /// Mutable iteration over the raw storage.
    pub fn iter_mut_storage(&mut self) -> IterMut<'_, T> {
        self.storage.iter_mut()
    }
    /// Clear the contents.
    pub fn clear(&mut self) {
        self.storage.clear()
    }
    /// Test if storage length is zero.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn storage_begin(&self, index: usize) -> usize {
        debug_assert!(index < self.len());
        index * self.segment_len
    }
    fn storage_end(&self, index: usize) -> usize {
        debug_assert!(index < self.len());
        self.storage_begin(index) + self.segment_len
    }
    fn storage_range(&self, index: usize) -> Range<usize> {
        debug_assert!(index < self.len());
        self.storage_begin(index)..self.storage_end(index)
    }
    fn storage_range_range(&self, begin: usize, end: usize) -> Range<usize> {
        self.storage_begin(begin)..self.storage_end(end)
    }
    fn storage_range_last(&self) -> Range<usize> {
        self.storage_range(self.last_index())
    }
    fn last_index(&self) -> usize {
        self.len() - 1
    }
    unsafe fn overwrite(&mut self, index: usize, segment: &[T]) {
        debug_assert!(index < self.len());
        debug_assert_eq!(self.segment_len, segment.len());
        ptr::copy(
            segment.as_ptr(),
            self.storage.as_mut_ptr().add(self.storage_begin(index)),
            self.segment_len,
        )
    }
    fn is_valid_length(&self, data: &[T]) -> bool {
        data.len() % self.segment_len == 0
    }
}

impl<T> Index<usize> for VecVec<T>
where
    T: Copy + Clone,
{
    type Output = [T];
    fn index(&self, index: usize) -> &Self::Output {
        &self.storage[self.storage_range(index)]
    }
}

impl<T> IndexMut<usize> for VecVec<T>
where
    T: Copy + Clone,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let range = self.storage_range(index);
        &mut self.storage[range]
    }
}

/// Contruct a `VecVec` from a list of arrays
///
/// # Example
///
/// ```
/// use vecvec::{vecvec, VecVec};
/// let x = vecvec![[1, 2, 3], [4, 5, 6]];
/// assert_eq!(x.len(), 2);
/// ```
///
/// Panics if array lengths do not match.
#[macro_export]
macro_rules! vecvec {
    ( $first:expr$(, $the_rest:expr )*$(,)? ) => {
        {
            let mut temp_vec = VecVec::new($first.len());
            temp_vec.push($first.as_slice());
            $(
                temp_vec.push($the_rest.as_slice());
            )*
            temp_vec
        }
    }
}

#[cfg(test)]
mod tests {
    use super::VecVec;

    #[test]
    fn test_vecvec() {
        let mut a = vecvec!([1, 2, 3], [4, 5, 6], [7, 8, 9]);
        assert!(a.is_valid_length(&[1, 2, 3, 4, 5, 6]));
        assert_eq!(&a[0], &[1, 2, 3]);
        assert_eq!(&a[1], &[4, 5, 6]);
        assert_eq!(&a[2], &[7, 8, 9]);
        assert_eq!(a.swap_remove(1), &[4, 5, 6]);
        assert_eq!(a.len(), 2);
        assert_eq!(&a[1], &[7, 8, 9]);
        a.append(&mut vecvec!(&[3, 6, 9]));
        assert_eq!(&a[2], &[3, 6, 9]);
        a.insert(1, &[3, 2, 1]);
        assert_eq!(&a[3], &[3, 6, 9]);
        assert_eq!(&a[1], &[3, 2, 1]);
        a.swap_insert(1, &[2, 2, 2]);
        assert_eq!(&a[4], &[3, 2, 1]);
        assert_eq!(&a[1], &[2, 2, 2]);
        let mut v: VecVec<i32> = VecVec::new(3);
        assert_eq!(v.len(), 0);
        v.push(&[1, 2, 3]);
        assert_eq!(v.len(), 1);
        assert_eq!(v.get(0), Some([1, 2, 3].as_slice()));
        v.push(&[4, 5, 6]);
        assert_eq!(v.len(), 2);
        assert_eq!(v.get(0).unwrap(), &[1, 2, 3]);
        assert_eq!(v.get(1).unwrap(), &[4, 5, 6]);
        let s: i32 = v.iter().map(|x| x.iter().sum::<i32>()).sum();
        assert_eq!(s, 21);
        let lens = v.iter().map(|x| x.len()).collect::<Vec<_>>();
        assert_eq!(lens, vec![3, 3]);
        assert_eq!(v.swap_remove(0), &[1, 2, 3]);
        assert_eq!(v.get(0).unwrap(), &[4, 5, 6]);
        v.iter_mut().for_each(|x| x.clone_from_slice(&[7, 8, 9]));
        assert_eq!(v.get(0).unwrap(), &[7, 8, 9]);
        let mut w: VecVec<i32> = VecVec::with_capacity(20, 5);
        w.push(&[1, 2, 3, 4, 5]);
        let x = w.get_mut(0).unwrap();
        assert_eq!(x, &[1, 2, 3, 4, 5]);
        x.clone_from_slice(&[5, 4, 3, 2, 1]);
        assert_eq!(x, &[5, 4, 3, 2, 1]);
        assert_eq!(&w[0], &[5, 4, 3, 2, 1]);
        assert_eq!(w[0][2], 3);
        let z = w.get_mut(0).unwrap();
        z[2] = 0;
        assert_eq!(z[2], 0);
        assert_eq!(w.get(0).unwrap()[2], 0);
        w.push(&[10, 20, 30, 40, 50]);
        w.push(&[100, 200, 300, 400, 500]);
        w.swap_truncate(0);
        assert_eq!(w.len(), 2);
        assert_eq!(&w[0], &[100, 200, 300, 400, 500]);
        assert_eq!(&w[1], &[10, 20, 30, 40, 50]);
        w.swap_truncate(1);
        assert_eq!(w.len(), 1);
        assert_eq!(&w[0], &[100, 200, 300, 400, 500]);
        w.swap_truncate(0);
        assert_eq!(w.len(), 0);
        assert!(w.is_empty());
    }
}
