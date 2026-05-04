//! `colorthief::extract` benchmark — establishes the MMCQ scaling
//! profile against scalar before any SIMD work on the histogram /
//! vbox sums lands.
//!
//! Run with: `cargo bench -p colorthief --bench extract`.
//!
//! Three frame sizes:
//! - 64×64 (4096 px, 12 KB) — small enough that fixed MMCQ overhead
//!   (priority-queue churn, vbox traversal) dominates.
//! - 256×256 (65k px, 192 KB) — typical thumbnail.
//! - 1024×1024 (1M px, 3 MB) — proxy for 1080p (~2M px, 6 MB)
//!   without the bigger allocation; same order of magnitude for
//!   per-pixel costs.
//!
//! Reading the table: if extract time scales ~linearly with pixel
//! count, the histogram build (`build_histogram`) is the dominant
//! cost and SIMD-ing its inner loop is high-ROI. If most of the time
//! is fixed (small change between sizes), the priority-queue + vbox
//! math is the bottleneck and SIMD on `VBox::count` / `median_cut`
//! partial sums matters more.

use std::hint::black_box;

use colorthief::{RgbFrame, extract};
use criterion::{Criterion, criterion_group, criterion_main};

/// Generate a tile of 64 distinct colors arranged in a deterministic
/// 8×8 pattern, then repeat it to fill `width × height`. This gives
/// MMCQ a non-trivial palette to quantize at every size — a single-
/// colour frame would short-circuit through the degenerate path and
/// bench nothing useful.
fn synthetic_frame(width: u32, height: u32) -> Vec<u8> {
  let mut buf = Vec::with_capacity((width * height) as usize * 3);
  for y in 0..height {
    for x in 0..width {
      // 64-color tile: 4 bits per channel, derived from the lower 6
      // bits of the (y, x) coords.
      let r = (((x ^ y) & 0x3F) as u8).wrapping_mul(4);
      let g = (((x.wrapping_add(y)) & 0x3F) as u8).wrapping_mul(4);
      let b = ((x.wrapping_mul(y) & 0x3F) as u8).wrapping_mul(4);
      buf.extend_from_slice(&[r, g, b]);
    }
  }
  buf
}

fn bench_extract(c: &mut Criterion) {
  // Stride = 3 * width (no row padding) for every fixture.
  let frames: Vec<(&str, u32, u32, Vec<u8>)> = [
    ("64x64", 64u32, 64u32),
    ("256x256", 256, 256),
    ("1024x1024", 1024, 1024),
  ]
  .into_iter()
  .map(|(label, w, h)| (label, w, h, synthetic_frame(w, h)))
  .collect();

  let mut group = c.benchmark_group("extract");
  // 5 dominants per call — typical search-vocabulary size.
  let count: u8 = 5;
  for (label, w, h, buf) in &frames {
    group.throughput(criterion::Throughput::Elements((*w * *h) as u64));
    group.bench_function(*label, |b| {
      b.iter(|| {
        let frame = RgbFrame::try_new(buf, *w, *h, *w * 3).expect("frame");
        black_box(extract(black_box(frame), count))
      })
    });
  }
  group.finish();
}

criterion_group!(benches, bench_extract);
criterion_main!(benches);
