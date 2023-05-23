use criterion::{criterion_group, criterion_main, Criterion};
use vecvec::VecVec;

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("vecvec", |b| {
        b.iter(|| {
            let mut x: VecVec<usize> = VecVec::with_capacity(1024, 1024); // allocate 8MB
            let segment = (0..1024).collect::<Vec<_>>();
            for _ in 0..1024 {
                x.push(segment.as_slice())
            } // no allocation
            for _ in 0..1024 {
                for i in (0..512).step_by(2) {
                    x.swap_truncate(i); // capacity unchanged
                }
                for _ in 0..512 {
                    x.push(segment.as_slice()); // capacity unchanged
                }
            }
        })
    });
    c.bench_function("vec", |b| {
        b.iter(|| {
            let mut x: Vec<Vec<usize>> = Vec::with_capacity(1024); // allocate 8MB
            let segment = (0..1024).collect::<Vec<_>>();
            for _ in 0..1024 {
                x.push(segment.clone())
            } // no allocation
            for _ in 0..1024 {
                for i in (0..512).step_by(2) {
                    x.swap_remove(i); // capacity unchanged
                }
                for _ in 0..512 {
                    x.push(segment.clone()); // capacity unchanged
                }
            }
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
