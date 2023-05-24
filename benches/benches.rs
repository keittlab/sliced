use criterion::{criterion_group, criterion_main, Criterion};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use vecvec::VecVec;

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("vecvec", |b| {
        b.iter(|| {
            let mut rng = SmallRng::from_entropy();
            let mut x: VecVec<usize> = VecVec::with_capacity(1024, 20);
            for _ in 0..1024 {
                x.push(
                    std::iter::repeat_with(|| rng.gen())
                        .take(20)
                        .collect::<Vec<usize>>()
                        .as_slice(),
                )
            }
            for _ in 0..1024 {
                for i in (0..512).step_by(2) {
                    x.swap_truncate(i);
                }
                for _ in 0..512 {
                    x.push(
                        std::iter::repeat_with(|| rng.gen())
                            .take(20)
                            .collect::<Vec<usize>>()
                            .as_slice(),
                    )
                }
            }
        })
    });
    c.bench_function("vec", |b| {
        b.iter(|| {
            let mut rng = SmallRng::from_entropy();
            let mut x: Vec<Vec<usize>> = Vec::with_capacity(1024); // allocate 8MB
            for _ in 0..1024 {
                x.push(
                    std::iter::repeat_with(|| rng.gen())
                        .take(20)
                        .collect::<Vec<usize>>()
                )
            }
            for _ in 0..1024 {
                for i in (0..512).step_by(2) {
                    x.swap_remove(i);
                }
                for _ in 0..512 {
                    x.push(
                        std::iter::repeat_with(|| rng.gen())
                            .take(20)
                            .collect::<Vec<usize>>()
                    )
                }
            }
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
