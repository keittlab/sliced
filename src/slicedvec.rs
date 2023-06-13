use std::{
    ops::{Index, IndexMut, Range},
    ptr,
};

/// A segmented vector for iterating over slices of constant length.
#[derive(Debug)]
pub struct SlicedVec<T>
where
    T: Copy + Clone,
{
    pub(crate) storage: Vec<T>,
    segment_len: usize,
}

impl<T> SlicedVec<T>
where
    T: Copy + Clone,
{
    /// Initialize a `SlicedVec` and set the segment size.
    ///
    /// # Example
    /// ```
    /// use sliced::SlicedVec;
    /// let mut sv = SlicedVec::new(10);
    /// sv.push_vec((0..10).collect());
    /// assert_eq!(sv.segment_len(), 10);
    /// assert_eq!(sv.len(), 1);
    /// ```
    /// # Panics
    /// If `segment_len` is zero.
    pub fn new(segment_len: usize) -> Self {
        assert_ne!(segment_len, 0);
        Self {
            storage: Vec::new(),
            segment_len,
        }
    }
    /// Initialize a `SlicedVec` and set the capacity and segment size.
    ///
    /// # Example
    /// ```
    /// use sliced::SlicedVec;
    /// let mut sv = SlicedVec::with_capacity(10, 1000);
    /// sv.push_vec((0..10).collect());
    /// assert_eq!(sv.storage_capacity(), 10000);
    /// assert_eq!(sv.capacity(), 1000);
    /// ```
    /// # Panics
    /// If `segment_len` is zero.
    pub fn with_capacity(segment_len: usize, size: usize) -> Self {
        assert_ne!(segment_len, 0);
        Self {
            storage: Vec::with_capacity(size * segment_len),
            segment_len,
        }
    }
    /// Initialize a `SlicedVec` from a vector.
    ///
    /// # Example
    /// ```
    /// use sliced::SlicedVec;
    /// let sv = SlicedVec::from_vec(3, (1..=9).collect());
    /// assert_eq!(sv[0], [1, 2, 3]);
    /// ```
    /// # Panics
    /// If `segment_len` is zero or the length of `data`
    /// is not a multiple of `segment_len`.
    pub fn from_vec(segment_len: usize, data: Vec<T>) -> Self {
        assert_ne!(segment_len, 0);
        assert_eq!(data.len() % segment_len, 0);
        Self {
            storage: data,
            segment_len,
        }
    }
    /// Get the internal segment length.
    ///
    /// # Example
    /// ```
    /// use sliced::{slicedvec, SlicedVec};
    /// let sv = slicedvec![[1, 2], [3, 4, 5, 6]];
    /// assert_eq!(sv.segment_len(), 2);
    /// ```
    pub fn segment_len(&self) -> usize {
        self.segment_len
    }
    /// Returns the number of internal segments.
    /// 
    /// # Example
    /// ```
    /// use sliced::{slicedvec, SlicedVec};
    /// let sv = slicedvec![[1, 2], [3, 4, 5, 6]];
    /// assert_eq!(sv.len(), 3);
    /// ```
    pub fn len(&self) -> usize {
        self.storage.len() / self.segment_len
    }
    /// Get the capacity in number of segments.
    pub fn capacity(&self) -> usize {
        self.storage_capacity() / self.segment_len
    }
    /// Returns the length of the underlying storage.
    pub fn storage_len(&self) -> usize {
        self.storage.len()
    }
    /// Get the capacity of the underlying storage.
    pub fn storage_capacity(&self) -> usize {
        self.storage.capacity()
    }
    /// Call `shrink_to_fit` on the storage.
    pub fn shrink_to_fit(&mut self) {
        self.storage.shrink_to_fit()
    }
    /// Append the contents of another `SlicedVec`.
    ///
    /// Complexity is the length of `other`, plus any
    /// allocation required. `other` is drained after call.
    ///
    /// # Example
    ///
    /// ```
    /// use sliced::{slicedvec, SlicedVec};
    /// let mut a = slicedvec![[1, 2, 3], [4, 5, 6]];
    /// let mut b = slicedvec![[7, 8, 9], [3, 2, 1]];
    /// a.append(&mut b);
    /// assert_eq!(a.len(), 4);
    /// assert_eq!(b.len(), 0);
    /// ```
    ///  # Panics
    /// If the segment size of `other` is different.
    pub fn append(&mut self, other: &mut Self) {
        assert_eq!(other.segment_len, self.segment_len);
        self.storage.append(&mut other.storage)
    }
    /// Insert a slice at position `index`.
    ///
    /// Complexity is linear in `storage_len`.
    /// # Example
    /// ```
    /// use sliced::{slicedvec, SlicedVec};
    /// let mut sv = slicedvec![[1, 2],[3, 4]];
    /// sv.insert(0, &[5, 6]); // [5,6][1,2][3,4]
    /// assert_eq!(sv.len(), 3);
    /// assert_eq!(sv[0], [5, 6]);
    /// ```
    /// # Panics
    /// If `index` is out of bounds or if the
    /// length of `segment` is not the native segment
    /// size of the `SlicedVec`.
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
        // Safety: index is range-checked and segment length is correct
        unsafe { self.overwrite(index, segment) }
    }
    /// Add one or more segments to the end.
    ///
    /// Complexity is amortized the segment size.
    /// # Example
    ///
    /// ```
    /// use sliced::*;
    /// let mut a = slicedvec![[1, 2, 3]];
    /// a.push(&[4, 5, 6, 7, 8, 9]); // any multiple of segment length
    /// assert_eq!(a.len(), 3);
    /// assert_eq!(a.storage_len(), 9);
    /// ```
    /// # Panics
    /// If the length of the slice is not
    /// a multiple of the segment length.
    pub fn push(&mut self, segment: &[T]) {
        assert!(self.is_valid_length(segment));
        self.storage.extend_from_slice(segment)
    }
    /// Add one or more segments contained in a `Vec`.
    ///
    /// Complexity is amortized the length of
    /// the slice.
    /// # Panics
    /// If the length of the slice is not
    /// a multiple of the segment length.
    pub fn push_vec(&mut self, segment: Vec<T>) {
        self.push(segment.as_slice())
    }
    /// Pop and return last segment.
    ///
    /// Returns `None` if empty.
    /// 
    /// # Example
    /// ```
    /// use sliced::{slicedvec, SlicedVec};
    /// let mut sv = slicedvec![[1, 2, 3], [4, 5, 6, 7, 8, 9]];
    /// let last = sv.pop();
    /// assert_eq!(last, Some(vec![7, 8, 9]));
    /// assert_eq!(sv.len(), 2);
    /// ```
    pub fn pop(&mut self) -> Option<Vec<T>> {
        if self.is_empty() {
            None
        } else {
            Some(
                self.storage
                    .drain(self.storage_range_last())
                    .as_slice()
                    .into(),
            )
        }
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
    /// Get a reference to the first segment.
    ///
    /// Returns `None` if `index` is out of range.
    pub fn first(&self) -> Option<&[T]> {
        self.get(0)
    }
    /// Get a mutable reference to the first segment.
    ///
    /// Returns `None` if `index` is out of range.
    pub fn first_mut(&mut self) -> Option<&mut [T]> {
        self.get_mut(0)
    }
    /// Get a reference to the last segment.
    ///
    /// Returns `None` if `index` is out of range.
    pub fn last(&self) -> Option<&[T]> {
        self.get(self.last_index())
    }
    /// Get a mutable reference to the last segment.
    ///
    /// Returns `None` if `index` is out of range.
    pub fn last_mut(&mut self) -> Option<&mut [T]> {
        self.get_mut(self.last_index())
    }
    /// Remove and return a segment.
    ///
    /// Does not preserve the order of segments.
    /// Complexity is the segment length.
    /// # Example
    /// ```
    /// use sliced::{slicedvec, SlicedVec};
    /// let mut sv = slicedvec![[1, 2, 3], [4, 5, 6, 7, 8, 9]];
    /// let first = sv.swap_remove(0);
    /// assert_eq!(first, vec![1, 2, 3]);
    /// assert_eq!(sv[0], [7, 8, 9]);
    /// ```
    ///  # Panics
    /// If index is out of range.
    pub fn swap_remove(&mut self, index: usize) -> Vec<T> {
        assert!(index < self.len());
        if index != self.last_index() {
            self.swap(index, self.last_index());
        }
        self.storage
            .drain(self.storage_range_last())
            .as_slice()
            .into()
    }
    /// Swap the contents of two segments.
    ///
    /// # Example
    /// ```
    /// use sliced::{slicedvec, SlicedVec};
    /// let mut sv = slicedvec![[1, 2, 3], [4, 5, 6, 7, 8, 9]];
    /// sv.swap(0, 2);
    /// assert_eq!(sv[0], [7, 8, 9]);
    /// assert_eq!(sv[1], [4, 5, 6]);
    /// assert_eq!(sv[2], [1, 2, 3]);
    /// ```
    pub fn swap(&mut self, i: usize, j: usize) {
        self.storage_range(i)
            .zip(self.storage_range(j))
            .for_each(|(a, b)| self.storage.swap(a, b))
    }
    /// Overwrite a segment from last and then truncate.
    ///
    /// Does not preserve the order of segments. The
    /// `SlicedVec` length will be reduced by one segment.
    /// Complexity is the segment length.
    /// # Example
    /// ```
    /// use sliced::{slicedvec, SlicedVec};
    /// let mut sv = slicedvec![[1, 2, 3], [4, 5, 6, 7, 8, 9]];
    /// sv.overwrite_remove(1);
    /// assert_eq!(sv[1], [7, 8, 9]);
    /// assert_eq!(sv.len(), 2);
    /// ```
    /// # Panics
    /// If `index` is out of bounds.
    pub fn overwrite_remove(&mut self, index: usize) {
        assert!(index < self.len());
        if index != self.last_index() {
            let src = self.storage_range_last();
            let dst = self.storage_begin(index);
            self.storage.copy_within(src, dst)
        }
        self.truncate(self.last_index());
    }
    /// Truncate the storage to `len` segments.
    ///
    /// If `len` is greater than the number of
    /// segments, nothing happens.
    /// # Example
    /// ```
    /// use sliced::SlicedVec;
    /// let mut sv = SlicedVec::<i32>::new(3);
    /// sv.truncate(1);
    /// assert_eq!(sv.len(), 0);
    /// sv.push_vec((1..=9).collect());
    /// sv.truncate(2);
    /// assert_eq!(sv.last(), Some([4, 5, 6].as_slice()));
    /// assert_eq!(sv.len(), 2);
    /// assert_eq!(sv.storage_len(), 6);
    /// ```
    pub fn truncate(&mut self, len: usize) {
        self.storage.truncate(len * self.segment_len);
    }
    /// Non-order-preserving, constant-time insert.
    ///
    /// Appends the contents of the segment at `index`
    /// to the end of the storage and then overwrites
    /// the segment with the new values.
    /// # Example
    /// ```
    /// use sliced::SlicedVec;
    /// let mut sv = SlicedVec::from_vec(3, (1..=9).collect());
    /// sv.relocate_insert(0, &[1, 2, 3]);
    /// assert_eq!(sv.first(), sv.last());
    /// ```
    /// # Panics
    /// If `index` is out of range.
    pub fn relocate_insert(&mut self, index: usize, segment: &[T]) {
        assert!(index < self.len());
        assert_eq!(segment.len(), self.segment_len);
        self.storage.extend_from_within(self.storage_range(index));
        // Safety: index range-checked and segment length matches
        unsafe { self.overwrite(index, segment) }
    }
    /// Return a chunked iterator.
    ///
    /// Allows iteration over segments as slices.
    /// # Example
    /// ```
    /// use sliced::{slicedvec, SlicedVec};
    /// let sv = slicedvec![[1, 2, 3], [4, 5, 6, 7, 8, 9]];
    /// for slice in sv.iter() {
    ///     assert_eq!(slice.len(), 3);
    /// }
    /// ```
    pub fn iter(&self) -> impl Iterator<Item = &[T]> {
        self.storage.chunks(self.segment_len)
    }
    /// Return a mutable chunked iterator.
    ///
    /// Allows iteration and modification of segments.
    /// # Example
    /// ```
    /// use sliced::{slicedvec, SlicedVec};
    /// let mut sv = slicedvec![[1, 2, 3], [4, 5, 6, 7, 8, 9]];
    /// sv.iter_mut().for_each(|slice| slice.swap(0, 2));
    /// assert_eq!(sv[0], [3, 2, 1]);
    /// ```
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut [T]> {
        self.storage.chunks_mut(self.segment_len)
    }
    /// Return a chunked iterator.
    ///
    /// Allows iteration over segments as slices.
    pub fn enumerate(&self) -> impl Iterator<Item = (usize, &[T])> {
        self.storage.chunks(self.segment_len).enumerate()
    }
    /// Iterate over the raw storage.
    pub fn iter_storage(&self) -> impl Iterator<Item = &T> {
        self.storage.iter()
    }
    /// Mutable iteration over the raw storage.
    pub fn iter_mut_storage(&mut self) -> impl Iterator<Item = &mut T> {
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
    pub(crate) fn storage_begin(&self, index: usize) -> usize {
        index * self.segment_len
    }
    pub(crate) fn storage_end(&self, index: usize) -> usize {
        self.storage_begin(index) + self.segment_len
    }
    pub(crate) fn storage_range(&self, index: usize) -> Range<usize> {
        self.storage_begin(index)..self.storage_end(index)
    }
    pub(crate) fn storage_range_range(&self, begin: usize, end: usize) -> Range<usize> {
        self.storage_begin(begin)..self.storage_end(end)
    }
    pub(crate) fn storage_range_last(&self) -> Range<usize> {
        self.storage_range(self.last_index())
    }
    // Caller is responsible for ensuring length is sufficient
    pub(crate) fn last_index(&self) -> usize {
        debug_assert!(!self.is_empty());
        self.len() - 1
    }
    // Caller is responsible for ensuring bounds are safe
    pub(crate) unsafe fn overwrite(&mut self, index: usize, segment: &[T]) {
        debug_assert!(index < self.len());
        debug_assert_eq!(self.segment_len, segment.len());
        ptr::copy(
            segment.as_ptr(),
            self.storage.as_mut_ptr().add(self.storage_begin(index)),
            self.segment_len,
        )
    }
    pub(crate) fn is_valid_length(&self, data: &[T]) -> bool {
        data.len() % self.segment_len == 0 && !data.is_empty()
    }
}

impl<T> Index<usize> for SlicedVec<T>
where
    T: Copy + Clone,
{
    type Output = [T];
    fn index(&self, index: usize) -> &Self::Output {
        &self.storage[self.storage_range(index)]
    }
}

impl<T> IndexMut<usize> for SlicedVec<T>
where
    T: Copy + Clone,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let range = self.storage_range(index);
        &mut self.storage[range]
    }
}

#[allow(clippy::from_over_into)]
impl<T> Into<Vec<T>> for SlicedVec<T>
where
    T: Copy + Clone,
{
    fn into(self) -> Vec<T> {
        self.storage
    }
}

/// Construct a `SlicedVec` from a list of arrays
///
/// # Example
/// ```
/// use sliced::{slicedvec, SlicedVec};
/// let x = slicedvec![[1, 2, 3], [4, 5, 6]];
/// assert_eq!(x.get(0), Some([1, 2, 3].as_slice()));
/// assert_eq!(x.get(2), None);
/// assert_eq!(x[1], [4, 5, 6]);
/// assert_eq!(x.len(), 2);
/// ```
/// # Panics
/// If array lengths do not match.
#[macro_export]
macro_rules! slicedvec {
    ( $first:expr$(, $the_rest:expr )*$(,)? ) => {
        {
            let mut temp_vec = SlicedVec::new($first.len());
            temp_vec.push($first.as_slice());
            $(
                temp_vec.push($the_rest.as_slice());
            )*
            temp_vec
        }
    }
}
