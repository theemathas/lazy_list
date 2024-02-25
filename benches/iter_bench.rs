use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lazy_list::LazyList;

pub fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("Iterate 10000", |b| {
        b.iter(|| {
            for x in &LazyList::new(0..10000) {
                black_box(x);
            }
        });
    });
    c.bench_function("Iterate 1000, repeat 100", |b| {
        b.iter(|| {
            let list = LazyList::new(0..1000);
            for _ in 0..100 {
                for x in &list {
                    black_box(x);
                }
            }
        });
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
