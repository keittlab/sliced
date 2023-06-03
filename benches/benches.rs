use criterion::{criterion_group, criterion_main, Criterion};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use rand_distr::StandardNormal;
use sliced::*;

fn criterion_benchmark(c: &mut Criterion) {
    let sample_range = |upper: usize, rng: &mut SmallRng| rng.gen_range(0..upper);
    let mut rng = SmallRng::from_entropy();
    c.bench_function("slicedvec", |b| {
        let mut x1 = SlicedVec::from_vec(
            20,
            std::iter::repeat_with(|| rng.sample(StandardNormal))
                .take(20 * 1000)
                .collect::<Vec<f32>>(),
        );
        let x1_insert = SlicedVec::from_vec(
            20,
            std::iter::repeat_with(|| rng.sample(StandardNormal))
                .take(20 * 1000)
                .collect::<Vec<f32>>(),
        );
        assert_eq!(x1.len(), 1000);
        assert_eq!(x1_insert.len(), 1000);
        b.iter(|| {
            for _ in 0..500 {
                let i = sample_range(x1.len(), &mut rng);
                x1.overwrite_remove(i);
            }
            for _ in 0..500 {
                let i = sample_range(1000, &mut rng);
                x1.push(&x1_insert[i]);
            }
        })
    });
    c.bench_function("vec_of_vec", |b| {
        let mut x2 = std::iter::repeat_with(|| {
            std::iter::repeat_with(|| rng.sample(StandardNormal))
                .take(20)
                .collect::<Vec<f32>>()
        })
        .take(1000)
        .collect::<Vec<Vec<f32>>>();
        let x2_insert = std::iter::repeat_with(|| {
            std::iter::repeat_with(|| rng.sample(StandardNormal))
                .take(20)
                .collect::<Vec<f32>>()
        })
        .take(1000)
        .collect::<Vec<Vec<f32>>>();
        assert_eq!(x2.len(), 1000);
        assert_eq!(x2_insert.len(), 1000);
        b.iter(|| {
            for _ in 0..500 {
                let i = sample_range(x2.len(), &mut rng);
                x2.swap_remove(i);
            }
            for _ in 0..500 {
                let i = sample_range(1000, &mut rng);
                x2.push(x2_insert[i].clone());
            }
        })
    });
    c.bench_function("slicedslab", |b| {
        let mut x3 = SlicedSlab::from_vec(
            20,
            std::iter::repeat_with(|| rng.sample(StandardNormal))
                .take(20 * 1000)
                .collect::<Vec<f32>>(),
        );
        let x3_insert = SlicedVec::from_vec(
            20,
            std::iter::repeat_with(|| rng.sample(StandardNormal))
                .take(20 * 1000)
                .collect::<Vec<f32>>(),
        );
        assert_eq!(x3.get_keys().len(), 1000);
        assert_eq!(x3_insert.len(), 1000);    
        b.iter(|| {
            let mut x3_keys: Vec<usize> = (0..1000).collect();
            for _ in 0..500 {
                let i = sample_range(x3_keys.len(), &mut rng);
                let key = x3_keys.swap_remove(i);
                x3.release(key);
            }
            for _ in 0..500 {
                let i = sample_range(1000, &mut rng);
                x3_keys.push(x3.insert(&x3_insert[i]));
            }
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
