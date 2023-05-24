use criterion::{criterion_group, criterion_main, Criterion};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use vecvec::VecVec;

fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = SmallRng::from_entropy();
    let mut gen_vec = || {
        std::iter::repeat_with(|| rng.gen())
            .take(20)
            .collect::<Vec<usize>>()
    };
    let mut x1: VecVec<usize> = VecVec::with_capacity(1024, 20);
    for _ in 0..1024 {
        x1.push(gen_vec().as_slice())
    }
    let mut x2: Vec<Vec<usize>> = Vec::with_capacity(1024);
    for _ in 0..1024 {
        x2.push(gen_vec())
    }
    c.bench_function("vecvec", |b| {
        b.iter(|| {
            for _ in 0..1024 {
                for i in (0..512).step_by(2) {
                    x1.swap_truncate(i);
                }
                for _ in 0..512 {
                    x1.push(gen_vec().as_slice())
                }
            }
        })
    });
    c.bench_function("vec", |b| {
        b.iter(|| {
            for _ in 0..1024 {
                for i in (0..512).step_by(2) {
                    x2.swap_remove(i);
                }
                for _ in 0..512 {
                    x2.push(gen_vec())
                }
            }
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
