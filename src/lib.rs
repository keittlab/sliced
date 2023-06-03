#![allow(dead_code)]
//! Two structs are provided: `SlicedVec` and `SlicedSlab`. `SlicedVec` stores a
//! collection of uniformly sized slices in a single vector. The segment length is determined at run-time
//! during initialization. Methods are available for constant-time, non-order-preserving insertion and deletion.
//! The erase-remove idiom is supported for for segments containing multiple values.
//!
//! `SlicedSlab` is built on `SlicedVec` and returns stable keys to allocated sequences of values. Methods are
//! provided for re-keying and compacting the slab if it becomes too sparse. Open slots are stored in a `BTreeSet`
//! so that new insert occur as close to the beginning of the storage as possible thereby reducing fragmentation.
//!
//! # Example
//!
//! ```
//! use rand::{rngs::SmallRng, Rng, SeedableRng, seq::SliceRandom};
//! use rand_distr::StandardNormal;
//! use sliced::{SlicedVec, SlicedSlab};
//! let mut rng = SmallRng::from_entropy();
//! let vals = (&mut rng).sample_iter(StandardNormal).take(1600).collect::<Vec<f32>>();
//! let mut svec = SlicedVec::from_vec(16, vals);
//! for _ in 0..100 {
//!     let i = (&mut rng).gen_range(0..svec.len());
//!     svec.overwrite_remove(i);
//!     svec.push_vec((&mut rng).sample_iter(StandardNormal).take(16).collect::<Vec<f32>>());
//! }
//! let mut slab = SlicedSlab::new(16);
//! let mut keys = Vec::new();
//! svec.iter().for_each(|segment| keys.push(slab.insert(segment)));
//! for _ in 0..50 {
//!     let i = keys.swap_remove((&mut rng).gen_range(0..keys.len()));
//!     slab.release(i);
//! }
//! for _ in 0..50 {
//!     let i = (&mut rng).gen_range(0..svec.len());
//!     keys.push(slab.insert(&svec[i]))
//! }
//! let rows = 100;
//! let cols = 100;
//! let data = (&mut rng).sample_iter(StandardNormal).take(rows * cols).collect::<Vec<f32>>();
//! let mut rast = SlicedVec::from_vec(cols, data);
//! for row in 1..(rows - 1) {
//!     for col in 1..(cols - 1) {
//!         rast[row][col] = rast[row][col - 1] + rast[row][col + 1] + 
//!                          rast[row - 1][col] + rast[row + 1][col]
//!     }
//! }
//! ```

use std::{
    collections::BTreeSet,
    ops::{Index, IndexMut, Range},
    ptr,
};

/// A segmented vector for iterating over slices of constant length.
#[derive(Debug)]
pub struct SlicedVec<T>
where
    T: Copy + Clone,
{
    storage: Vec<T>,
    segment_len: usize,
}

impl<T> SlicedVec<T>
where
    T: Copy + Clone,
{
    /// Initialize a `SlicedVec` and set the segment size.
    ///
    /// Panics if `segment_len` is zero.
    /// 
    /// # Example
    /// ```
    /// use sliced::SlicedVec;
    /// let mut sv = SlicedVec::new(10);
    /// sv.push_vec((0..10).collect());
    /// assert_eq!(sv.segment_len(), 10);
    /// assert_eq!(sv.len(), 1);
    /// ```
    pub fn new(segment_len: usize) -> Self {
        assert_ne!(segment_len, 0);
        Self {
            storage: Vec::new(),
            segment_len,
        }
    }
    /// Initialize a `SlicedVec` and set the capacity and segment size.
    ///
    /// Panics if `segment_len` is zero.
    /// 
    /// # Example
    /// ```
    /// use sliced::SlicedVec;
    /// let mut sv = SlicedVec::with_capacity(1000, 10);
    /// sv.push_vec((0..10).collect());
    /// assert_eq!(sv.storage_capacity(), 10000);
    /// assert_eq!(sv.capacity(), 1000);
    /// ```
    pub fn with_capacity(size: usize, segment_len: usize) -> Self {
        assert_ne!(segment_len, 0);
        Self {
            storage: Vec::with_capacity(size * segment_len),
            segment_len,
        }
    }
    /// Initialize a `SlicedVec` from a vector.
    ///
    /// Panics if `segment_len` is zero or the length of `data`
    /// is not a multiple of `segment_len`.
    ///
    /// # Example
    /// ```
    /// use sliced::SlicedVec;
    /// let sv = SlicedVec::from_vec(3, (1..=9).collect());
    /// assert_eq!(sv[0], [1, 2, 3]);
    /// ```
    pub fn from_vec(segment_len: usize, data: Vec<T>) -> Self {
        assert_ne!(segment_len, 0);
        assert_eq!(data.len() % segment_len, 0);
        Self {
            storage: data,
            segment_len,
        }
    }
    /// Get the internal segment length
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
    /// Returns the number of internal segments
    /// # Example
    /// ```
    /// use sliced::{slicedvec, SlicedVec};
    /// let sv = slicedvec![[1, 2], [3, 4, 5, 6]];
    /// assert_eq!(sv.len(), 3);
    /// ```
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
    /// size of the `SlicedVec`.
    ///
    /// # Example
    /// ```
    /// use sliced::{slicedvec, SlicedVec};
    /// let mut sv = slicedvec![[1, 2],[3, 4]];
    /// // [1,2][3,4]
    /// sv.insert(0, &[5, 6]);
    /// // [5,6][1,2][3,4]
    /// assert_eq!(sv.len(), 3);
    /// assert_eq!(sv[0], [5, 6]);
    /// ```
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
        // Requires index in bounds
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
    /// use sliced::*;
    /// let mut a = slicedvec![[1, 2, 3]];
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
    pub fn push_vec(&mut self, segment: Vec<T>) {
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
    ///
    /// Panics if index is out of range.
    ///
    /// # Example
    /// ```
    /// use sliced::{slicedvec, SlicedVec};
    /// let mut sv = slicedvec![[1, 2, 3], [4, 5, 6, 7, 8, 9]];
    /// let first = sv.swap_remove(0);
    /// assert_eq!(first, vec![1, 2, 3]);
    /// assert_eq!(sv[0], [7, 8, 9]);
    /// ```
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
    ///
    /// Panics if `index` is out of bounds.
    ///
    /// # Example
    /// ```
    /// use sliced::{slicedvec, SlicedVec};
    /// let mut sv = slicedvec![[1, 2, 3], [4, 5, 6, 7, 8, 9]];
    /// sv.overwrite_remove(1);
    /// assert_eq!(sv[1], [7, 8, 9]);
    /// assert_eq!(sv.len(), 2);
    /// ```
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
    ///
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
    /// the segment with the new values. If `index` is
    /// greater than or equal to `self.len()`, then the
    /// segments is repeatedly pushed until it fills the
    /// location given by `index`.
    ///
    /// Panics if `index` is out of range.
    ///
    /// # Example
    /// ```
    /// use sliced::SlicedVec;
    /// let mut sv = SlicedVec::from_vec(3, (1..=9).collect());
    /// sv.relocate_insert(0, &[1, 2, 3]);
    /// assert_eq!(sv.first(), sv.last());
    /// ```
    pub fn relocate_insert(&mut self, index: usize, segment: &[T]) {
        assert!(index < self.len());
        assert_eq!(segment.len(), self.segment_len);
        self.storage.extend_from_within(self.storage_range(index));
        // Requires index in bounds
        unsafe { self.overwrite(index, segment) }
    }
    /// Return a chunked iterator.
    ///
    /// Allows iteration over segments as slices.
    ///
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
    ///
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
    fn storage_begin(&self, index: usize) -> usize {
        index * self.segment_len
    }
    fn storage_end(&self, index: usize) -> usize {
        self.storage_begin(index) + self.segment_len
    }
    fn storage_range(&self, index: usize) -> Range<usize> {
        self.storage_begin(index)..self.storage_end(index)
    }
    fn storage_range_range(&self, begin: usize, end: usize) -> Range<usize> {
        self.storage_begin(begin)..self.storage_end(end)
    }
    fn storage_range_last(&self) -> Range<usize> {
        self.storage_range(self.last_index())
    }
    // Caller is responsible for ensuring length is sufficient
    fn last_index(&self) -> usize {
        debug_assert!(!self.is_empty());
        self.len() - 1
    }
    // Caller is responsible for ensuring bounds are safe
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
///
/// ```
/// use sliced::{slicedvec, SlicedVec};
/// let x = slicedvec![[1, 2, 3], [4, 5, 6]];
/// assert_eq!(x.get(0), Some([1, 2, 3].as_slice()));
/// assert_eq!(x.get(2), None);
/// assert_eq!(x[1], [4, 5, 6]);
/// assert_eq!(x.len(), 2);
/// ```
///
/// Panics if array lengths do not match.
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

/// A segmented slab with stable keys.
///
/// Maintains a `SlicedVec` and a `BTreeSet` of
/// available slots. Given sufficient capacity, no
/// allocation will occur on insert or removal. Look
/// up of available slots is logarithmic in the number
/// of open slots.
#[derive(Debug)]
pub struct SlicedSlab<T>
where
    T: Copy + Clone,
{
    slots: SlicedVec<T>,
    open_slots: BTreeSet<usize>,
}

impl<T> SlicedSlab<T>
where
    T: Copy + Clone,
{
    /// Construct a new `SlicedSlab`.
    ///
    /// Panics if `segment_len` is zero.
    pub fn new(segment_len: usize) -> Self {
        assert_ne!(segment_len, 0);
        Self {
            slots: SlicedVec::new(segment_len),
            open_slots: BTreeSet::new(),
        }
    }
    /// Initialize a `SlicedSlab` and set the capacity and segment size.
    ///
    /// Panics if `segment_len` is zero.
    pub fn with_capacity(size: usize, segment_len: usize) -> Self {
        assert_ne!(segment_len, 0);
        Self {
            slots: SlicedVec::with_capacity(size, segment_len),
            open_slots: BTreeSet::new(),
        }
    }
    /// Initialize a `SlicedSlab` and set the capacity and segment size.
    ///
    /// Panics if `segment_len` is zero.
    ///
    /// # Example
    /// ```
    /// use sliced::SlicedSlab;
    /// let mut ss = SlicedSlab::from_vec(3, (1..=9).collect());
    /// ss.release(1);
    /// assert_eq!(ss.get_keys(), vec![0, 2]);
    /// ```
    pub fn from_vec(segment_len: usize, data: Vec<T>) -> Self {
        assert_ne!(segment_len, 0);
        Self {
            slots: SlicedVec::from_vec(segment_len, data),
            open_slots: BTreeSet::new(),
        }
    }
    /// Iterate over active keys.
    ///
    /// This will be slow if the slab is sparse.
    ///
    /// # Example
    /// ```
    /// use sliced::{SlicedVec, SlicedSlab};
    /// let mut sv = SlicedVec::new(3);
    /// let mut ss = SlicedSlab::from_vec(3, (1..=9).collect());
    /// ss.release(1);
    /// ss.iter_keys().for_each(|key| sv.push(&ss[key]));
    /// assert_eq!(sv[1], ss[2]);
    /// ```
    pub fn iter_keys(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.slots.len()).filter(|key| !self.open_slots.contains(key))
    }
    /// Get active keys.
    ///
    /// This will be slow if the slab is sparse.
    pub fn get_keys(&self) -> Vec<usize> {
        self.iter_keys().collect()
    }
    /// Insert a segment into the slab.
    ///
    /// The first available slot is overwritten
    /// with the contents of the slice. Otherwise,
    /// the slice is appended to the storage. Returns
    /// a key for later retrieval.
    ///
    /// Panics if the length of the slice does
    /// not match the segments size of the slab.
    pub fn insert(&mut self, segment: &[T]) -> usize {
        assert_eq!(segment.len(), self.slots.segment_len());
        match self.open_slots.pop_first() {
            Some(key) => {
                debug_assert!(key < self.slots.len());
                unsafe {
                    // Requires key is in bounds
                    self.slots.overwrite(key, segment);
                }
                key
            }
            None => {
                let key = self.slots.len();
                self.slots.push(segment);
                key
            }
        }
    }
    /// Insert a vector into the slab.
    ///
    /// # Example
    /// ```
    /// use sliced::SlicedSlab;
    /// let mut ss = SlicedSlab::new(3);
    /// assert_eq!(ss.insert_vec((1..=3).collect()), 0);
    /// ```
    pub fn insert_vec(&mut self, data: Vec<T>) -> usize {
        self.insert(data.as_slice())
    }
    /// Copy a segment and return a new key.
    ///
    /// If there exists an open slot closer to the
    /// start of the slab, then the data pointed
    /// to by `oldkey` will be moved there and
    /// a new key will be returned. Otherwise, no
    /// action is taken and `oldkey` is returned
    /// unchanged.
    ///
    /// Panics if the old key is unoccupied.
    ///
    /// # Example
    /// ```
    /// use sliced::SlicedSlab;
    /// let mut ss = SlicedSlab::new(3);
    /// assert_eq!(ss.insert(&[1, 2, 3]), 0);
    /// assert_eq!(ss.insert(&[4, 5, 6]), 1);
    /// // [occ][occ]
    /// ss.release(0);
    /// // [vac][occ]
    /// assert_eq!(ss.rekey(1), 0);
    /// // [occ][vac]
    /// assert_eq!(ss[0], [4, 5, 6]);
    /// ```
    pub fn rekey(&mut self, oldkey: usize) -> usize {
        debug_assert!(oldkey < self.slots.len());
        if self.open_slots.first() < Some(&oldkey) {
            match self.acquire() {
                Some(newkey) => {
                    self.release(oldkey);
                    debug_assert!(newkey < self.slots.len());
                    let src = self.slots.storage_range(oldkey);
                    let dst = self.slots.storage_begin(newkey);
                    self.slots.storage.copy_within(src, dst);
                    newkey
                }
                None => oldkey,
            }
        } else {
            oldkey
        }
    }
    /// Removes open slots at the end of the slab.
    ///
    /// If all key-holders have called rekey, this
    /// function will remove all open slots, thus
    /// fully compacting the slab. The storage capacity
    /// is not affected. This will greatly increase the
    /// speed of key look-ups as there will be no open
    /// slots to search. Subsequent insertions will all
    /// be pushed to the end of the storage. Or if all
    /// slots are open, the slab will be empty after
    /// this call.
    ///
    /// # Example
    /// ```
    /// use sliced::SlicedSlab;
    /// let mut ss = SlicedSlab::new(3);
    /// assert_eq!(ss.insert(&[1, 2, 3]), 0);
    /// assert_eq!(ss.insert(&[4, 5, 6]), 1);
    /// assert_eq!(ss.insert(&[7, 8, 9]), 2);
    /// // [occ][occ][occ]
    /// ss.release(1);
    /// // [occ][vac][occ]
    /// assert_eq!(ss.sparsity(), 1./3.);
    /// ss.compact();
    /// // [occ][vac][occ]
    /// assert_eq!(ss.sparsity(), 1./3.);
    /// assert_eq!(ss.get(1), None);
    /// assert_eq!(ss.rekey(0), 0);
    /// // [occ][vac][occ]
    /// assert_eq!(ss.get(2), Some([7, 8, 9].as_slice()));
    /// assert_eq!(ss.rekey(2), 1);
    /// // [occ][occ][vac]
    /// assert_eq!(ss.get(1), Some([7, 8, 9].as_slice()));
    /// assert_eq!(ss.get(2), None);
    /// ss.compact();
    /// // [occ][occ]
    /// assert_eq!(ss.sparsity(), 0.0);
    /// ss.release(1);
    /// ss.compact();
    /// assert_eq!(ss.iter_keys().max().unwrap(), 0);
    /// ss.release(0);
    /// ss.compact();
    /// assert!(ss.get_keys().is_empty());
    /// ```
    pub fn compact(&mut self) {
        if self.open_slots.len() == self.slots.len() {
            // Covers empty case
            self.open_slots.clear();
            self.slots.clear()
        } else {
            debug_assert!(!self.slots.is_empty());
            debug_assert!(self.open_slots.len() < self.slots.len());
            let mut len = self.slots.len();
            while self.open_slots.last() == Some(&(len - 1)) {
                self.open_slots.pop_last();
                debug_assert!(len > 0);
                len -= 1;
            }
            self.slots.storage.truncate(len * self.slots.segment_len);
            debug_assert!(self.open_slots.len() <= self.slots.len());
            debug_assert!(self.open_slots.last() < Some(&(self.slots.len())));
        }
    }
    /// Compute the proportion of open slots.
    ///
    /// A sparsity of 0.0 indicates no open slots and
    /// insertions will be pushed at the end of the storage.
    /// A sparsity of 1.0 indicates only open slots and compaction
    /// will lead to an empty slab.
    pub fn sparsity(&self) -> f32 {
        self.open_slots.len() as f32 / self.slots.len() as f32
    }
    /// Mark the slot as open for future overwrite.
    ///
    /// Keys are not globally unique. They will be reused.
    /// Marking the slot unoccupied is logarithmic in the
    /// number of open slots.
    ///
    /// Panics of the slot is already marked as open.
    pub fn release(&mut self, key: usize) {
        assert!(key < self.slots.len());
        // This is the only site where keys are added
        // The assertion ensures that no key is out of bounds
        assert!(self.open_slots.insert(key));
        debug_assert!(self.open_slots.len() <= self.slots.len());
    }
    /// Acquire a previously released slot.
    ///
    /// This allows one to directly update
    /// the internal storage. Returns `None`
    /// if there are no open slots.
    ///
    /// # Example
    /// ```
    /// use sliced::SlicedSlab;
    /// let mut ss = SlicedSlab::from_vec(2, (1..=8).collect());
    /// assert_eq!(ss.acquire(), None);
    /// ss.release(2);
    /// let key = ss.acquire().expect("No empty slots!");
    /// assert_eq!(key, 2);
    /// ss[key].iter_mut().for_each(|value| *value = 0);
    /// assert_eq!(ss[key], [0, 0]);
    /// ```
    pub fn acquire(&mut self) -> Option<usize> {
        self.open_slots.pop_first()
    }
    /// Get a reference to a segment.
    ///
    /// Returns `None` if `key` is out of range
    /// or the slot is marked as unoccupied. Key
    /// look up is logarithmic in the number of
    /// open slots.
    pub fn get(&self, key: usize) -> Option<&[T]> {
        if self.open_slots.contains(&key) {
            return None;
        }
        self.slots.get(key)
    }
    /// Get a mutable reference to a segment.
    ///
    /// Returns `None` if `key` is out of range
    /// or the slot is marked as unoccupied. Key
    /// look up is logarithmic in the number of
    /// open slots.
    pub fn get_mut(&mut self, key: usize) -> Option<&mut [T]> {
        if self.open_slots.contains(&key) {
            return None;
        }
        self.slots.get_mut(key)
    }
    /// Iterate over key, slice pairs.
    ///
    /// This will be slow if the slab is very sparse.
    ///
    /// # Example
    /// ```
    /// use sliced::SlicedSlab;
    /// let mut ss = SlicedSlab::from_vec(3, (1..=9).collect());
    /// ss.release(1);
    /// let s: usize = ss.enumerate()
    ///     .map(|(key, slice)| key * slice.len())
    ///     .sum();
    /// assert_eq!(s, 6);
    /// ```
    pub fn enumerate(&self) -> impl Iterator<Item = (usize, &[T])> {
        self.slots
            .enumerate()
            .filter(|(key, _)| !self.open_slots.contains(key))
    }
}

/// Get segment from slab.
///
/// This will return whatever it finds at index
/// regardless of whether it is occupied
/// or released.
///
/// Panics if `index` is out of range.
///
/// # Example
/// ```
/// use sliced::SlicedSlab;
/// let mut ss = SlicedSlab::from_vec(3, (1..=9).collect());
/// ss.release(1);
/// assert_eq!(ss[1], [4, 5, 6]);
/// assert_eq!(ss.insert(&[3, 2, 1]), 1);
/// assert_eq!(ss[1], [3, 2, 1]);
/// ```
impl<T> Index<usize> for SlicedSlab<T>
where
    T: Copy + Clone,
{
    type Output = [T];
    fn index(&self, index: usize) -> &Self::Output {
        &self.slots[index]
    }
}

/// Get segment from slab.
///
/// This will return whatever it finds at index
/// regardless of whether it is occupied
/// or released.
///
/// Panics if `index` is out of range.
/// # Example
/// ```
/// use sliced::SlicedSlab;
/// let mut ss = SlicedSlab::from_vec(3, (1..=9).collect());
/// ss.release(1);
/// ss[1][1] = 0;
/// assert_eq!(ss[1], [4, 0, 6]);
/// ```
impl<T> IndexMut<usize> for SlicedSlab<T>
where
    T: Copy + Clone,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.slots[index]
    }
}

#[cfg(test)]
mod tests {
    use super::SlicedVec;

    #[test]
    fn test_slicedvec() {
        let mut a = slicedvec!([1, 2, 3], [4, 5, 6], [7, 8, 9]);
        assert!(a.is_valid_length(&[1, 2, 3, 4, 5, 6]));
        assert_eq!(a.segment_len(), 3);
        assert_eq!(&a[0], &[1, 2, 3]);
        assert_eq!(&a[1], &[4, 5, 6]);
        assert_eq!(&a[2], &[7, 8, 9]);
        assert_eq!(a.swap_remove(1), &[4, 5, 6]);
        assert_eq!(a.len(), 2);
        assert_eq!(&a[1], &[7, 8, 9]);
        a.append(&mut slicedvec!(&[3, 6, 9]));
        assert_eq!(&a[2], &[3, 6, 9]);
        a.insert(1, &[3, 2, 1]);
        assert_eq!(&a[3], &[3, 6, 9]);
        assert_eq!(&a[1], &[3, 2, 1]);
        a.relocate_insert(1, &[2, 2, 2]);
        assert_eq!(&a[4], &[3, 2, 1]);
        assert_eq!(&a[1], &[2, 2, 2]);
        let mut v: SlicedVec<i32> = SlicedVec::new(3);
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
        let mut w: SlicedVec<i32> = SlicedVec::with_capacity(20, 5);
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
        w.overwrite_remove(0);
        assert_eq!(w.len(), 2);
        assert_eq!(&w[0], &[100, 200, 300, 400, 500]);
        assert_eq!(&w[1], &[10, 20, 30, 40, 50]);
        w.overwrite_remove(1);
        assert_eq!(w.len(), 1);
        assert_eq!(&w[0], &[100, 200, 300, 400, 500]);
        w.overwrite_remove(0);
        assert_eq!(w.len(), 0);
        assert!(w.is_empty());
        let a = slicedvec![[1, 2, 3], [4, 5, 6]];
        let aa: Vec<_> = a.into();
        assert_eq!(aa.len(), 6);
    }
}
