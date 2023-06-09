use std::ops::{Index, IndexMut, Range};

/// A segmented vector with variable length segments.
#[derive(Debug)]
pub struct VarSlicedVec<T>
where
    T: Copy + Clone,
{
    storage: Vec<T>,
    extents: Vec<usize>,
}

impl<T> VarSlicedVec<T>
where
    T: Copy + Clone,
{
    /// Initialize a `VarSlicedVec`.
    ///
    /// # Example
    /// ```
    /// use sliced::*;
    /// let mut vv = VarSlicedVec::new();
    /// vv.push_vec((0..10).collect());
    /// assert_eq!(vv.segment_len(0), 10);
    /// assert_eq!(vv.len(), 1);
    /// ```
    pub fn new() -> Self {
        Self {
            storage: Vec::new(),
            extents: vec![0],
        }
    }
    /// Initialize a `VarSlicedVec` and set the capacity.
    ///
    /// # Example
    /// ```
    /// use sliced::VarSlicedVec;
    /// let mut vv = VarSlicedVec::<u8>::with_capacity(1000);
    /// assert_eq!(vv.storage_capacity(), 1000);
    /// ```
    pub fn with_capacity(size: usize) -> Self {
        Self {
            storage: Vec::with_capacity(size),
            extents: vec![0],
        }
    }
    /// Append the contents of another `VarSlicedVec`.
    ///
    /// `other` is drained after call.
    ///
    /// # Example
    ///
    /// ```
    /// use sliced::*;
    /// let mut a = varslicedvec![[1, 2], [3], [4, 5, 6]];
    /// let mut b = varslicedvec![[7], [8, 9], [3, 2, 1]];
    /// a.append(&mut b);
    /// assert_eq!(a.len(), 6);
    /// assert_eq!(b.len(), 0);
    /// assert_eq!(a.lengths(), vec![2, 1, 3, 1, 2, 3]);
    /// let mut c = VarSlicedVec::new();
    /// a.append(&mut c);
    /// assert_eq!(a.len(), 6);
    /// assert_eq!(a.lengths(), vec![2, 1, 3, 1, 2, 3]);
    /// ```
    pub fn append(&mut self, other: &mut Self) {
        other
            .lengths()
            .iter()
            .for_each(|length| self.extents.push(self.last_extent() + length));
        other.extents.truncate(1);
        self.storage.append(&mut other.storage);
        debug_assert!(self.check_invariants());
        debug_assert!(other.check_invariants());
    }
    /// Add a segments to the end.
    ///
    /// Complexity is amortized the segment size.
    ///
    /// # Example
    ///
    /// ```
    /// use sliced::*;
    /// let mut vv = VarSlicedVec::new();
    /// vv.push(&[1, 2, 3]);
    /// vv.push(&[4, 5]);
    /// assert_eq!(vv[0], [1, 2, 3]);
    /// assert_eq!(vv[1], [4, 5]);
    /// vv.push(&[]);
    /// assert_eq!(vv.len(), 3);
    /// assert_eq!(vv[2], []);
    /// vv.push(&[1]);
    /// assert_eq!(vv.len(), 4);
    /// assert_eq!(vv[2], []);
    /// assert_eq!(vv[3], [1]);
    /// assert_eq!(vv.pop().unwrap(), [1]);
    /// assert_eq!(vv.pop().unwrap(), [])
    /// ```
    ///
    pub fn push(&mut self, segment: &[T]) {
        self.extents.push(self.last_extent() + segment.len());
        self.storage.extend_from_slice(segment);
        debug_assert!(self.check_invariants());
    }
    /// Add one or more segments contained in a `Vec`.
    ///
    /// Complexity is amortized the length of
    /// the slice.
    ///
    /// Panics if the length of the slice is not
    /// a multiple of the segment length.
    /// # Example
    ///
    /// ```
    /// use sliced::*;
    /// let mut vv = VarSlicedVec::with_capacity(1024);
    /// vv.push_vec((1..=512).collect());
    /// vv.push_vec(vec![0]);
    /// assert_eq!(vv[0].len(), 512);
    /// assert_eq!(vv[1], [0]);
    /// ```
    pub fn push_vec(&mut self, segment: Vec<T>) {
        self.push(segment.as_slice())
    }
    /// Pop and return last segment.
    ///
    /// Returns `None` if empty.
    ///
    /// # Example
    /// ```
    /// use sliced::*;
    /// let mut vv = varslicedvec![[1, 2, 3], [4, 5, 6, 7, 8, 9]];
    /// assert_eq!(vv.pop(), Some(vec![4, 5, 6, 7, 8, 9]));
    /// assert_eq!(vv.len(), 1);
    /// assert_eq!(vv.pop(), Some(vec![1, 2, 3]));
    /// assert_eq!(vv.pop(), None);
    /// ```
    pub fn pop(&mut self) -> Option<Vec<T>> {
        if self.is_empty() {
            None
        } else {
            let range = self.storage_range(self.len() - 1);
            self.extents.pop();
            Some(self.storage.drain(range).as_slice().into())
        }
    }
    /// Split container into twp parts.
    ///
    /// # Example
    /// ```
    /// use sliced::*;
    /// let mut vv1 = varslicedvec![[1], [2, 3], [4, 5, 6]];
    /// let vv2 = vv1.split_off(1);
    /// assert_eq!(vv1[0], [1]);
    /// assert_eq!(vv1.lengths(), vec![1]);
    /// assert_eq!(vv2[0], [2, 3]);
    /// assert_eq!(vv2.lengths(), vec![2, 3]);
    /// ```
    pub fn split_off(&mut self, at: usize) -> Self {
        debug_assert!(self.check_invariants());
        Self {
            storage: self.storage.split_off(self.storage_begin(at)),
            extents: [0]
                .into_iter()
                .chain(
                    self.extents
                        .split_off(at + 1)
                        .into_iter()
                        .map(|extent| extent - self.storage_begin(at)),
                )
                .collect::<Vec<usize>>(),
        }
    }
    /// Insert a segment into the container.
    ///
    /// # Example
    /// ```
    /// use sliced::*;
    /// let mut vv = varslicedvec![[1], [2, 3]];
    /// vv.insert(0, &[0]);
    /// assert_eq!(vv[0], [0]);
    /// vv.insert(2, &[4]);
    /// assert_eq!(vv[2], [4]);
    /// assert_eq!(vv[3], [2, 3]);
    /// ```
    pub fn insert(&mut self, at: usize, segment: &[T]) {
        let mut back = self.split_off(at);
        self.push(segment);
        self.append(&mut back);
        debug_assert!(self.check_invariants());
    }
    /// Remove and return a segment
    /// 
    /// # Example
    /// ```
    /// use sliced::*;
    /// let mut vv = varslicedvec![[1], [2, 3], [4, 5, 6]];
    /// assert_eq!(vv.remove(1), [2, 3]);
    /// assert_eq!(vv[1], [4, 5, 6]);
    /// ```
    pub fn remove(&mut self, index: usize) -> Vec<T> {
        assert!(index < self.len());
        if index == self.len() - 1 {
            self.pop().unwrap()
        } else {
            let mut back = self.split_off(index + 1);
            let segment = self.pop();
            self.append(&mut back);
            segment.unwrap()
        }
    }
    /// Get a reference to a segment.
    ///
    /// Returns `None` if `index` is out of range.
    ///
    /// # Example
    /// ```
    /// use sliced::*;
    /// let vv = varslicedvec![[1, 2, 3], [4, 5], [6]];
    /// assert_eq!(vv.get(0), Some([1, 2, 3].as_slice()));
    /// assert_eq!(vv.get(1), Some([4, 5].as_slice()));
    /// assert_eq!(vv.get(2), Some([6].as_slice()));
    /// assert_eq!(vv.get(3), None);
    /// ```
    pub fn get(&self, index: usize) -> Option<&[T]> {
        debug_assert!(self.check_invariants());
        if index < self.len() {
            // Safety: index range is checked
            unsafe {
                let range = self.storage_range_unchecked(index);
                Some(self.storage.get_unchecked(range))
            }
        } else {
            None
        }
    }
    /// Get a mutable reference to a segment.
    ///
    /// Returns `None` if `index` is out of range.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut [T]> {
        debug_assert!(self.check_invariants());
        if index < self.len() {
            // Safety: index range is checked
            unsafe {
                let range = self.storage_range_unchecked(index);
                Some(self.storage.get_unchecked_mut(range))
            }
        } else {
            None
        }
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
        self.get(self.len() - 1)
    }
    /// Get a mutable reference to the last segment.
    ///
    /// Returns `None` if `index` is out of range.
    pub fn last_mut(&mut self) -> Option<&mut [T]> {
        self.get_mut(self.len() - 1)
    }
    /// Get the segment length at `index`.
    ///
    /// # Example
    /// ```
    /// use sliced::*;
    /// let vv = varslicedvec![[1, 2], [3, 4, 5, 6]];
    /// assert_eq!(vv.segment_len(0), 2);
    /// assert_eq!(vv.segment_len(1), 4);
    /// assert_eq!(vv.segment_len(2), 0);
    /// ```
    pub fn segment_len(&self, index: usize) -> usize {
        if index < self.len() {
            self.extents[index + 1] - self.extents[index]
        } else {
            0
        }
    }
    /// Return a vector of segment lengths.
    ///
    /// # Example
    /// ```
    /// use sliced::*;
    /// let vv = varslicedvec![[1, 2], [3, 4, 5, 6]];
    /// assert_eq!(vv.lengths(), vec![2, 4]);
    /// ```
    pub fn lengths(&self) -> Vec<usize> {
        self.extents.windows(2).map(|x| x[1] - x[0]).collect()
    }
    /// Returns the number of internal segments.
    ///
    /// # Example
    /// ```
    /// use sliced::VarSlicedVec;
    /// let mut vv = VarSlicedVec::new();
    /// vv.push(&[1]);
    /// vv.push(&[2, 3]);
    /// vv.push(&[4, 5, 6]);
    /// assert_eq!(vv.len(), 3);
    /// ```
    pub fn len(&self) -> usize {
        self.extents.len() - 1
    }
    /// Clear the contents
    pub fn clear(&mut self) {
        self.storage.clear();
        self.extents.truncate(1);
        debug_assert!(self.check_invariants());
    }
    /// Test if length is zero.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Get the capacity of the underlying storage.
    pub fn storage_capacity(&self) -> usize {
        self.storage.capacity()
    }
    /// Get storage range of index
    fn storage_range(&self, index: usize) -> Range<usize> {
        self.storage_begin(index)..self.storage_end(index)
    }
    /// Get start of segment storage.
    fn storage_begin(&self, index: usize) -> usize {
        self.extents[index]
    }
    /// Get end of segment storage.
    fn storage_end(&self, index: usize) -> usize {
        self.extents[index + 1]
    }
    /// Get storage range of index.
    unsafe fn storage_range_unchecked(&self, index: usize) -> Range<usize> {
        debug_assert!(self.check_invariants());
        self.storage_begin_unchecked(index)..self.storage_end_unchecked(index)
    }
    /// Get start of segment storage.
    unsafe fn storage_begin_unchecked(&self, index: usize) -> usize {
        debug_assert!(self.check_invariants());
        *self.extents.get_unchecked(index)
    }
    /// Get end of segment storage.
    unsafe fn storage_end_unchecked(&self, index: usize) -> usize {
        debug_assert!(self.check_invariants());
        *self.extents.get_unchecked(index + 1)
    }
    /// Get last extent
    fn last_extent(&self) -> usize {
        debug_assert!(!self.extents.is_empty());
        let i = self.extents.len() - 1;
        // Safety: extents is never empty
        unsafe { *self.extents.get_unchecked(i) }
    }
    /// Debugging sanity check
    fn check_invariants(&self) -> bool {
        (!self.extents.is_empty())
            && self.extents[0] == 0
            && self.extents.last().unwrap() == &self.storage.len()
            && self.extents_are_monotonic()
    }
    /// Extents must not decrease
    fn extents_are_monotonic(&self) -> bool {
        if self.extents.len() > 1 {
            self
            .extents
            .windows(2)
            .map(|x| x[1] >= x[0])
            .fold(true, |aggr, cond| aggr & cond)
        } else {
            true
        } 
    }
    /// Return iterator over slices
    ///
    /// # Example
    /// ```
    /// use sliced::*;
    /// let vv = varslicedvec![[1], [2, 3], [4, 5, 6], [7, 8], [9]];
    /// vv.iter().for_each(|slice| assert!(slice[0] < 10));
    /// let third = vv.iter().take(3).last();
    /// assert_eq!(third, Some([4, 5, 6].as_slice()));
    /// let lens = vv.iter().map(|slice| slice.len()).collect::<Vec<usize>>();
    /// assert_eq!(lens, vec![1, 2, 3, 2, 1]);
    /// ```
    pub fn iter(&self) -> VarSlicedVecIter<T> {
        VarSlicedVecIter { data: self, i: 0 }
    }
}

impl<T> Index<usize> for VarSlicedVec<T>
where
    T: Copy + Clone,
{
    type Output = [T];
    fn index(&self, index: usize) -> &Self::Output {
        let range = self.storage_range(index);
        // Safety: above will panic if out of range
        unsafe { self.storage.get_unchecked(range) }
    }
}

impl<T> IndexMut<usize> for VarSlicedVec<T>
where
    T: Copy + Clone,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let range = self.storage_range(index);
        // Safety: above will panic if out of range
        unsafe { self.storage.get_unchecked_mut(range) }
    }
}

impl<T> Default for VarSlicedVec<T>
where
    T: Copy + Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Iterator over slices
pub struct VarSlicedVecIter<'a, T>
where
    T: Copy + Clone,
{
    data: &'a VarSlicedVec<T>,
    i: usize,
}

impl<'a, T> Iterator for VarSlicedVecIter<'a, T>
where
    T: Copy + Clone,
{
    type Item = &'a [T];
    fn next(&mut self) -> Option<Self::Item> {
        debug_assert!(self.data.check_invariants());
        if self.i < self.data.len() {
            let range = self.data.storage_range(self.i);
            self.i += 1;
            // Safety: i cannot be out of range
            unsafe { Some(self.data.storage.get_unchecked(range)) }
        } else {
            None
        }
    }
}

/*
/// Iterator over slices
pub struct VarSlicedVecIterMut<'a, T>
where T: Copy + Clone
{
    data: &'a mut VarSlicedVec<T>,
    i: usize,
}

impl<'a, T> Iterator for VarSlicedVecIterMut<'a, T>
where T: Copy + Clone
{
    type Item = &'a mut[T];
    fn next(&mut self) -> Option<Self::Item> {
        if self.i < self.data.len() {
            let range = self.data.storage_range(self.i);
            self.i += 1;
            unsafe { let ret = self.data.storage.get_unchecked_mut(range);
            Some(&'a mut *ret) }
        } else {
            None
        }
    }
}
*/

/// Construct a `VarSlicedVec` from a list of arrays
///
/// # Example
///
/// ```
/// use sliced::*;
/// let x = varslicedvec![[1, 2], [3, 4, 5, 6], [7, 8, 9]];
/// assert_eq!(x.get(0), Some([1, 2].as_slice()));
/// assert_eq!(x.get(3), None);
/// assert_eq!(x[1], [3, 4, 5, 6]);
/// assert_eq!(x.len(), 3);
/// ```
///
/// Panics if array lengths do not match.
#[macro_export]
macro_rules! varslicedvec {
    ( $first:expr$(, $the_rest:expr )*$(,)? ) => {
        {
            let mut temp_vec = VarSlicedVec::new();
            temp_vec.push($first.as_slice());
            $(
                temp_vec.push($the_rest.as_slice());
            )*
            temp_vec
        }
    }
}
