#![allow(dead_code)]

use std::{
    ops::{Index, IndexMut, Range},
    slice::{Chunks, ChunksMut, Iter, IterMut},
};

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
    pub fn new(segment_len: usize) -> Self {
        assert_ne!(segment_len, 0);
        Self {
            storage: Vec::new(),
            segment_len,
        }
    }
    pub fn with_capacity(size: usize, segment_len: usize) -> Self {
        assert_ne!(segment_len, 0);
        Self {
            storage: Vec::with_capacity(size * segment_len),
            segment_len,
        }
    }
    pub fn len(&self) -> usize {
        self.storage.len() / self.segment_len
    }
    pub fn append(&mut self, other: &mut Self) {
        assert_eq!(other.segment_len, self.segment_len);
        self.storage.append(&mut other.storage)
    }
    pub fn push(&mut self, segment: &[T]) {
        assert_eq!(segment.len(), self.segment_len);
        self.storage.extend_from_slice(segment)
    }
    pub fn get(&self, index: usize) -> Option<&[T]> {
        self.storage.get(self.storage_range(index))
    }
    pub fn get_mut(&mut self, index: usize) -> Option<&mut [T]> {
        let range = self.storage_range(index);
        self.storage.get_mut(range)
    }
    pub fn swap_remove(&mut self, index: usize) -> Vec<T> {
        assert!(index < self.len());
        let last_index = self.len() - 1;
        if index != last_index {
            self.storage_range(index)
                .zip(self.storage_range(last_index))
                .for_each(|(i, j)| self.storage.swap(i, j))
        }
        self.storage
            .drain(self.storage_range(last_index))
            .as_slice()
            .into()
    }
    pub fn overwrite_remove(&mut self, index: usize) {
        assert!(index < self.len());
        let last_index = self.len() - 1;
        if index != last_index {
            let source_range = self.storage_range(last_index);
            self.storage
                .copy_within(source_range, index * self.segment_len)
        }
        self.storage.truncate(self.storage.len() - self.segment_len)
    }
    pub fn swap_insert(&mut self, index: usize, segment: &[T]) {
        assert!(index < self.len());
        assert_eq!(segment.len(), self.segment_len);
        self.storage.extend_from_within(self.storage_range(index));
        // finish
    }
    pub fn iter(&self) -> Chunks<'_, T> {
        self.storage.chunks(self.segment_len)
    }
    pub fn iter_mut(&mut self) -> ChunksMut<'_, T> {
        self.storage.chunks_mut(self.segment_len)
    }
    pub fn iter_storage(&self) -> Iter<'_, T> {
        self.storage.iter()
    }
    pub fn iter_mut_storage(&mut self) -> IterMut<'_, T> {
        self.storage.iter_mut()
    }
    pub fn clear(&mut self) {
        self.storage.clear()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn storage_range(&self, index: usize) -> Range<usize> {
        assert!(index < self.len());
        let start = index * self.segment_len;
        let end = start + self.segment_len;
        start..end
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

#[macro_export]
macro_rules! vecvec {
    ( $first:expr$(, $the_rest:expr )*$(,)? ) => {
        {
            let mut temp_vec = VecVec::new($first.len());
            temp_vec.push($first);
            $(
                temp_vec.push($the_rest);
            )*
            temp_vec
        }
    };
}

#[cfg(test)]
mod tests {
    use super::VecVec;

    #[test]
    fn test_vecvec() {
        let mut a = vecvec!(&[1, 2, 3], &[4, 5, 6], &[7, 8, 9]);
        assert_eq!(&a[0], &[1, 2, 3]);
        assert_eq!(&a[1], &[4, 5, 6]);
        assert_eq!(&a[2], &[7, 8, 9]);
        assert_eq!(a.swap_remove(1), &[4, 5, 6]);
        assert_eq!(a.len(), 2);
        assert_eq!(&a[1], &[7, 8, 9]);
        a.append(&mut vecvec!(&[3, 6, 9]));
        assert_eq!(&a[2], &[3, 6, 9]);
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
    }
}
