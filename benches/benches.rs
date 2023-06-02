use criterion::{criterion_group, criterion_main, Criterion};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use sliced::*;

fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = SmallRng::from_entropy();
    let mut x1 = SlicedVec::with_capacity(1000, 20);
    x1.push_vec(
        std::iter::repeat_with(|| rng.gen())
            .take(20 * 1000)
            .collect::<Vec<_>>(),
    );
    let mut x2: Vec<Vec<usize>> =
        std::iter::repeat_with(|| std::iter::repeat_with(|| rng.gen()).take(20).collect())
            .take(1000)
            .collect();
    let x1_insert: Vec<Vec<usize>> =
        std::iter::repeat_with(|| std::iter::repeat_with(|| rng.gen()).take(20).collect())
            .take(500)
            .collect();
    let x2_insert: Vec<Vec<usize>> =
        std::iter::repeat_with(|| std::iter::repeat_with(|| rng.gen()).take(20).collect())
            .take(500)
            .collect();
    c.bench_function("slicedvec", |b| {
        b.iter(|| {
            for i in 0..500 {
                x1.overwrite_remove(i);
            }
            x1_insert.iter().for_each(|segment| x1.push(segment));
        })
    });
    c.bench_function("vec_of_vec", |b| {
        b.iter(|| {
            for i in 0..500 {
                x2.swap_remove(i);
            }
            x2_insert
                .iter()
                .for_each(|segment| x2.push(segment.clone()));
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
