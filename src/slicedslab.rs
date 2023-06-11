use std::{collections::BTreeSet, ops::{IndexMut, Index}};
use crate::slicedvec::*;

/// A segmented slab with stable keys.
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
    /// # Panics
    /// If `segment_len` is zero.
    pub fn new(segment_len: usize) -> Self {
        assert_ne!(segment_len, 0);
        Self {
            slots: SlicedVec::new(segment_len),
            open_slots: BTreeSet::new(),
        }
    }
    /// Initialize a `SlicedSlab` and set the capacity and segment size.
    /// # Panics
    /// If `segment_len` is zero.
    pub fn with_capacity(segment_len: usize, size: usize) -> Self {
        assert_ne!(segment_len, 0);
        Self {
            slots: SlicedVec::with_capacity(segment_len, size),
            open_slots: BTreeSet::new(),
        }
    }
    /// Initialize a `SlicedSlab` from a vector.
    /// # Example
    /// ```
    /// use sliced::SlicedSlab;
    /// let mut ss = SlicedSlab::from_vec(3, (1..=9).collect());
    /// ```
    /// # Panics
    /// If `segment_len` is zero.
    pub fn from_vec(segment_len: usize, data: Vec<T>) -> Self {
        assert_ne!(segment_len, 0);
        Self {
            slots: SlicedVec::from_vec(segment_len, data),
            open_slots: BTreeSet::new(),
        }
    }
    /// Iterate over active keys.
    /// # Example
    /// ```
    /// use sliced::*;
    /// let mut ss = SlicedSlab::from_vec(3, (1..=9).collect());
    /// ss.release(1);
    /// let mut sv = SlicedVec::new(3);
    /// ss.iter_keys().for_each(|key| sv.push(&ss[key]));
    /// assert_eq!(sv[1], ss[2]);
    /// ```
    pub fn iter_keys(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.slots.len()).filter(|key| !self.open_slots.contains(key))
    }
    /// Get active keys.
    /// 
    /// # Example
    /// ```
    /// use sliced::*;
    /// let mut ss = SlicedSlab::from_vec(2, (0..10).collect());
    /// ss.release(1);
    /// ss.release(3);
    /// assert_eq!(ss.get_keys(), vec![0, 2, 4]);
    /// ```
    pub fn get_keys(&self) -> Vec<usize> {
        self.iter_keys().collect()
    }
    /// Insert a segment into the slab.
    /// 
    /// The first available slot is overwritten
    /// with the contents of the slice. Otherwise,
    /// the slice is appended to the storage. Returns
    /// a key for later retrieval.
    /// # Example
    /// ```
    /// use sliced::*;
    /// let mut ss = SlicedSlab::new(2);
    /// let key = ss.insert(&[1, 2]);
    /// assert_eq!(ss[key], [1, 2]);
    /// ```
    /// # Panics
    /// If the length of the slice does
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
    /// # Panics
    /// If the old key is released.
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
    /// ss.release(1);
    /// assert_eq!(ss.sparsity(), 1./3.);
    /// ss.release(2);
    /// assert_eq!(ss.sparsity(), 2./3.);
    /// ss.compact();
    /// assert_eq!(ss.sparsity(), 0.0);
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
            self.slots.truncate(len);
            debug_assert!(self.open_slots.len() <= self.slots.len());
            debug_assert!(self.open_slots.last() < Some(&self.slots.len()));
        }
    }
    /// Call `shrink_to_fit` on the storage.
    pub fn shrink_to_fit(&mut self) {
        self.slots.shrink_to_fit()
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
