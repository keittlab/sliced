# [sliced](https://docs.rs/sliced)

Two structs are provided: `SlicedVec` and `SlicedSlab`. `SlicedVec` stores a
collection of uniformly sized slices in a single vector. The segment length is determined at run-time
during initialization. Methods are available for constant-time, non-order-preserving insertion and deletion.
The erase-remove idiom is supported for for segments containing multiple values.

`SlicedSlab` is built on `SlicedVec` and returns stable keys to allocated sequences of values. Methods are
provided for re-keying and compacting the slab if it becomes too sparse. Open slots are stored in a `BTreeSet`
so that new insert occur as close to the beginning of the storage as possible thereby reducing fragmentation.

# Example
```rust
use rand::{rngs::SmallRng, Rng, SeedableRng, seq::SliceRandom};
use rand_distr::StandardNormal;
use sliced::{SlicedVec, SlicedSlab};
let mut rng = SmallRng::from_entropy();
let vals = (&mut rng).sample_iter(StandardNormal).take(1600).collect::<Vec<f32>>();
let mut svec = SlicedVec::from_vec(16, vals);
for _ in 0..100 {
    let i = (&mut rng).gen_range(0..svec.len());
    svec.overwrite_remove(i);
    svec.push_vec((&mut rng).sample_iter(StandardNormal).take(16).collect::<Vec<f32>>());
}
let mut slab = SlicedSlab::new(16);
let mut keys = Vec::new();
svec.iter().for_each(|segment| keys.push(slab.insert(segment)));
for _ in 0..50 {
    let i = keys.swap_remove((&mut rng).gen_range(0..keys.len()));
    slab.release(i);
}
for _ in 0..50 {
    let i = (&mut rng).gen_range(0..svec.len());
    keys.push(slab.insert(&svec[i]))
}
let rows = 100;
let cols = 100;
let data = (&mut rng).sample_iter(StandardNormal).take(rows * cols).collect::<Vec<f32>>();
let mut rast = SlicedVec::from_vec(cols, data);
for row in 1..(rows - 1) {
    for col in 1..(cols - 1) {
        rast[row][col] = rast[row][col - 1] + rast[row][col + 1] + 
                         rast[row - 1][col] + rast[row + 1][col] - 4. * rast[row][col]
    }
}
```


  
