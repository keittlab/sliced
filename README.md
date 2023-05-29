# slicedvec

This is [a rust crate](https://docs.rs/slicedvec) that provides a type `SlicedVec<T>` that emulates aspects of `Vec<Vec<T>>` using a single `Vec<T>` for storage. The main purpose is to support the swap-remove idiom. When repeatedly creating and dropping many objects, the swap-remove idiom can be used to eliminate most allocations. This does not work however if the stored objects themselves allocate, as happens with `Vec<Vec<T>>`.
  
