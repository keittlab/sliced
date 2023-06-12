# sliced

The `sliced` crate is a thin wrapper around `Vec` that returns slices over internal storage rather
than individual elements. It is useful in cases where you need to store and repeatedly manipulate a
large collection of relatively short runs of numbers with the run lengths determined at run-time rather
than during compilation. Using `Vec<Vec<T>>` means that each insert and remove will allocate and deallocate heap
storage for the inner `Vec`, whereas sliced storage will use a single growable buffer.

For variable length slices, `VarSlicedVec` stores the sequences in a single `Vec` along with their extents using
a compressed sparse layout.
```rust
use sliced::*;
let mut vv = VarSlicedVec::new();
vv.push(&[1, 2, 3]);
vv.push(&[4, 5]);
vv.push(&[6]);
assert_eq!(vv.remove(1), [4, 5]);
assert_eq!(vv.pop(), Some(vec![6]));
assert_eq!(vv[0], [1, 2, 3]);
```

For strings of equal length set at run-time, `SlicedVec` allows for constant-time insertion and
removal without extra allocation if there is sufficient spare storage capacity.
```rust
use sliced::*;
let mut sv = SlicedVec::new(3);
sv.push(&[1, 2, 3]);
sv.push(&[4, 5, 6]);
sv.push(&[7, 8, 9]);
assert_eq!(sv.swap_remove(1), [4, 5, 6]);
assert_eq!(sv.pop(), Some(vec![7, 8, 9]));
assert_eq!(sv[0], [1, 2, 3]);
```

`SlicedSlab` is also provided for accessing segments using a key.
```rust
use sliced::*;
let mut ss = SlicedSlab::from_vec(3, (1..=9).collect());
assert_eq!(ss.get_keys(), vec![0, 1, 2]);
assert_eq!(ss[1], [4, 5, 6]);
ss.release(1);
assert_eq!(ss.insert(&[6, 5, 4]), 1);
assert_eq!(ss[1], [6, 5, 4]);
```

License: MIT
