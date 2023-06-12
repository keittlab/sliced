#![allow(dead_code)]
//! The `sliced` crate is a thin wrapper around `Vec` that returns slices over internal storage rather
//! than individual elements. It is useful in cases where you need to store and repeatedly manipulate a
//! large collection of relatively short runs of numbers with the run lengths determined at run-time rather
//! than during compilation. Using `Vec<Vec<T>>` means that each insert and remove will allocate and deallocate heap
//! storage for the inner `Vec`, whereas sliced storage will use a single growable buffer.
//! 
//! For variable length slices, `VarSlicedVec` stores the sequences in a single `Vec` along with their extents using
//! a compressed sparse layout.
//! ```
//! use sliced::*;
//! let mut vv = VarSlicedVec::new();
//! vv.push(&[1, 2, 3]);
//! vv.push(&[4, 5]);
//! vv.push(&[6]);
//! assert_eq!(vv.remove(1), [4, 5]);
//! assert_eq!(vv.pop(), Some(vec![6]));
//! assert_eq!(vv[0], [1, 2, 3]);
//! ```
//! 
//! For strings of equal length set at run-time, `SlicedVec` allows for constant-time insertion and
//! removal without extra allocation if there is sufficient spare storage capacity.
//! ```
//! use sliced::*;
//! let mut sv = SlicedVec::new(3);
//! sv.push(&[1, 2, 3]);
//! sv.push(&[4, 5, 6]);
//! sv.push(&[7, 8, 9]);
//! assert_eq!(sv.swap_remove(1), [4, 5, 6]);
//! assert_eq!(sv.pop(), Some(vec![7, 8, 9]));
//! assert_eq!(sv[0], [1, 2, 3]);
//! ```
//! 
//! `SlicedSlab` is also provided for accessing segments using a key.
//! ```
//! use sliced::*;
//! let mut ss = SlicedSlab::from_vec(3, (1..=9).collect());
//! assert_eq!(ss.get_keys(), vec![0, 1, 2]);
//! assert_eq!(ss[1], [4, 5, 6]);
//! ss.release(1);
//! assert_eq!(ss.insert(&[6, 5, 4]), 1);
//! assert_eq!(ss[1], [6, 5, 4]);
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
