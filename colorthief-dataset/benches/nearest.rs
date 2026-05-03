//! Head-to-head benchmark: scalar vs every available SIMD backend on
//! the same deterministic 4096-point LAB query grid.
//!
//! Run with: `cargo bench -p colorthief-dataset --bench nearest`.
//!
//! Per-target backend coverage:
//! - aarch64: scalar + NEON
//! - x86_64: scalar + (SSE4.1 / AVX2 — only the variants the host
//!   actually supports, gated via `is_x86_feature_detected!`)
//! - wasm32 + simd128: scalar + WASM SIMD128
//! - other: scalar only
//!
//! `Color::nearest_to` (the public dispatched API) is benched too as
//! a sanity check that the dispatcher itself doesn't add measurable
//! overhead vs calling the chosen backend directly.

use std::hint::black_box;

use colorthief_dataset::{__bench, Color};
use criterion::{Criterion, criterion_group, criterion_main};

/// Build a deterministic, varied set of LAB queries from a 16-step RGB
/// grid (16³ = 4096 points). Pre-computing keeps the benchmark loop
/// focused on the NN scan rather than RGB→LAB conversion.
fn lab_query_grid() -> Vec<[f32; 3]> {
  let mut out = Vec::with_capacity(4096);
  for r in (0..256).step_by(16) {
    for g in (0..256).step_by(16) {
      for b in (0..256).step_by(16) {
        out.push(__bench::rgb_to_lab([r as u8, g as u8, b as u8]));
      }
    }
  }
  out
}

fn bench_nearest_idx(c: &mut Criterion) {
  let queries = lab_query_grid();
  let mut group = c.benchmark_group("nearest_idx");

  // Set throughput so criterion reports time-per-element.
  group.throughput(criterion::Throughput::Elements(1));

  // Delta E 76 scalar baseline — always present.
  group.bench_function("scalar", |b| {
    let mut iter = queries.iter().cycle();
    b.iter(|| {
      let q = *iter.next().unwrap();
      black_box(__bench::scalar_nearest_idx(black_box(q)))
    })
  });

  // CIEDE2000 scalar — the perceptual gold-standard distance metric.
  // Scalar-only public path (atan2 / sin / cos / exp don't vectorise
  // cleanly). Bench expectation: ~50–100× slower than the Delta E 76
  // scalar baseline per the formula's transcendental count.
  group.bench_function("ciede2000_scalar", |b| {
    let mut iter = queries.iter().cycle();
    b.iter(|| {
      let q = *iter.next().unwrap();
      black_box(__bench::ciede2000_nearest_idx(black_box(q)))
    })
  });

  // CIEDE2000 with Delta E 76 prefilter. Hierarchical: scan all 949
  // entries with the cheap squared-Euclidean LAB metric, keep the
  // top-K (K=96, validated for zero divergences on the 17³ grid),
  // re-rank with full CIEDE2000. Empirically equivalent to the
  // full-scan reference on every grid query.
  group.bench_function("ciede2000_prefiltered", |b| {
    let mut iter = queries.iter().cycle();
    b.iter(|| {
      let q = *iter.next().unwrap();
      black_box(__bench::ciede2000_prefiltered_nearest_idx(black_box(q)))
    })
  });

  // CIE94 scalar reference.
  group.bench_function("cie94_scalar", |b| {
    let mut iter = queries.iter().cycle();
    b.iter(|| {
      let q = *iter.next().unwrap();
      black_box(__bench::cie94_nearest_idx(black_box(q)))
    })
  });

  // CIE94 NEON.
  #[cfg(target_arch = "aarch64")]
  group.bench_function("cie94_aarch64_neon", |b| {
    let mut iter = queries.iter().cycle();
    b.iter(|| {
      let q = *iter.next().unwrap();
      black_box(__bench::cie94_aarch64_neon_nearest_idx(black_box(q)))
    })
  });

  // CIE94 x86 SIMD — only bench what the host actually supports.
  #[cfg(target_arch = "x86_64")]
  {
    if std::is_x86_feature_detected!("sse4.1") {
      group.bench_function("cie94_x86_sse41", |b| {
        let mut iter = queries.iter().cycle();
        b.iter(|| {
          let q = *iter.next().unwrap();
          // SAFETY: feature verified.
          black_box(unsafe { __bench::cie94_x86_sse41_nearest_idx(black_box(q)) })
        })
      });
    }
    if std::is_x86_feature_detected!("avx2") {
      group.bench_function("cie94_x86_avx2", |b| {
        let mut iter = queries.iter().cycle();
        b.iter(|| {
          let q = *iter.next().unwrap();
          // SAFETY: feature verified.
          black_box(unsafe { __bench::cie94_x86_avx2_nearest_idx(black_box(q)) })
        })
      });
    }
  }

  // CIE94 WASM SIMD128.
  #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
  group.bench_function("cie94_wasm_simd128", |b| {
    let mut iter = queries.iter().cycle();
    b.iter(|| {
      let q = *iter.next().unwrap();
      black_box(__bench::cie94_wasm_simd128_nearest_idx(black_box(q)))
    })
  });

  // No SIMD CIEDE2000 bench arm. A NEON attempt benchmarked at
  // 115.9 µs/query against the scalar baseline's 85.9 µs/query on
  // 2026-05-03 — the transcendentals (atan2, sin, cos, exp ×
  // ~10K/query) dominate everything and SIMD only adds load/store
  // overhead. Discarded per the protocol: try, bench, keep only if
  // it wins.

  // aarch64 NEON.
  #[cfg(target_arch = "aarch64")]
  group.bench_function("aarch64_neon", |b| {
    let mut iter = queries.iter().cycle();
    b.iter(|| {
      let q = *iter.next().unwrap();
      black_box(__bench::aarch64_neon_nearest_idx(black_box(q)))
    })
  });

  // x86 backends — only bench the ones the host CPU actually supports.
  #[cfg(target_arch = "x86_64")]
  {
    if std::is_x86_feature_detected!("sse4.1") {
      group.bench_function("x86_sse41", |b| {
        let mut iter = queries.iter().cycle();
        b.iter(|| {
          let q = *iter.next().unwrap();
          // SAFETY: feature just verified; the function is
          // `#[target_feature(enable = "sse4.1")]`.
          black_box(unsafe { __bench::x86_sse41_nearest_idx(black_box(q)) })
        })
      });
    }
    if std::is_x86_feature_detected!("avx2") {
      group.bench_function("x86_avx2", |b| {
        let mut iter = queries.iter().cycle();
        b.iter(|| {
          let q = *iter.next().unwrap();
          // SAFETY: feature just verified.
          black_box(unsafe { __bench::x86_avx2_nearest_idx(black_box(q)) })
        })
      });
    }
  }

  // WASM SIMD128.
  #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
  group.bench_function("wasm_simd128", |b| {
    let mut iter = queries.iter().cycle();
    b.iter(|| {
      let q = *iter.next().unwrap();
      black_box(__bench::wasm_simd128_nearest_idx(black_box(q)))
    })
  });

  group.finish();
}

/// Benchmark the public `Color::nearest_to(rgb)` — includes the RGB→LAB
/// conversion plus dispatch overhead. This is what production callers
/// actually pay per call.
fn bench_color_nearest_to(c: &mut Criterion) {
  let mut rgb_queries: Vec<[u8; 3]> = Vec::with_capacity(4096);
  for r in (0..256).step_by(16) {
    for g in (0..256).step_by(16) {
      for b in (0..256).step_by(16) {
        rgb_queries.push([r as u8, g as u8, b as u8]);
      }
    }
  }

  let mut group = c.benchmark_group("Color::nearest_to");
  group.throughput(criterion::Throughput::Elements(1));
  group.bench_function("dispatched", |b| {
    let mut iter = rgb_queries.iter().cycle();
    b.iter(|| {
      let rgb = *iter.next().unwrap();
      black_box(Color::nearest_to(black_box(rgb)))
    })
  });
  group.finish();
}

criterion_group!(benches, bench_nearest_idx, bench_color_nearest_to);
criterion_main!(benches);
