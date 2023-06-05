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
//! let genseq = |n: usize, rng: &mut SmallRng|
//!     rng.sample_iter(StandardNormal)
//!     .take(n).collect::<Vec<f32>>();
//! let sample_range = |upper: usize, rng: &mut SmallRng|
//!     rng.gen_range(0..upper);
//! // Constant time insertion and deletion in contigous memory
//! let vals = genseq(1600, &mut rng);
//! let mut svec = SlicedVec::from_vec(16, vals);
//! for _ in 0..100 {
//!     let i = sample_range(svec.len(), &mut rng);
//!     svec.overwrite_remove(i);
//!     svec.push_vec(genseq(16, &mut rng))
//! }
//! // Key-based access in pre-allocated memory
//! let mut slab = SlicedSlab::with_capacity(16, 100);
//! let mut keys = Vec::new();
//! svec.iter().for_each(|segment| keys.push(slab.insert(segment)));
//! for _ in 0..50 {
//!     let i = keys.swap_remove(sample_range(keys.len(), &mut rng));
//!     slab.release(i)
//! }
//! keys.iter_mut().for_each(|key| *key = slab.rekey(*key));
//! slab.compact();
//! for _ in 0..50 {
//!     let i = sample_range(svec.len(), &mut rng);
//!     keys.push(slab.insert(&svec.swap_remove(i)))
//! }
//! let sum = keys.iter().map(|&key| slab[key].iter().sum::<f32>()).sum::<f32>();
//! // 4-point Laplace operator on grid
//! let rows = 256;
//! let cols = 128;
//! let mut rast = SlicedVec::from_vec(cols, genseq(rows * cols, &mut rng));
//! for row in 1..(rows - 1) {
//!     for col in 1..(cols - 1) {
//!         rast[row][col] = rast[row][col - 1] + rast[row][col + 1] + 
//!                          rast[row - 1][col] + rast[row + 1][col] - 4. * rast[row][col]
//!     }
//! }
//! ```

mod slicedvec;
pub use slicedvec::*;

mod slicedslab;
pub use slicedslab::*;

mod varslicedvec;
pub use varslicedvec::*;

#[cfg(test)]
mod tests {
    use super::*;

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
        let mut w: SlicedVec<i32> = SlicedVec::with_capacity(5, 20);
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
