//! Exhaustive bit-parity sweeps across the full 256³ RGB cube.
//!
//! Each test in this file iterates all 16,777,216 distinct u8 RGB
//! inputs and asserts that every reachable SIMD backend (or the
//! CIEDE2000 prefilter) returns the bit-identical `usize` index as
//! the scalar reference. This is the airtight version of the 17³ =
//! 4913-point grid tests already living inline in
//! `src/nearest/mod.rs::tests`; those run on every `cargo test` for
//! quick feedback, while the tests here are
//!
//! - marked `#[ignore]` (run via `cargo test --ignored`),
//! - and meant to be invoked with `cargo test --release` — debug-mode
//!   runtime is impractical (~hours for the prefilter test).
//!
//! Approximate runtime on Apple Silicon, release mode:
//!
//! | Test                                              | Duration |
//! |---------------------------------------------------|----------|
//! | `parity_de76_*_vs_scalar_256_grid`                | ~30 sec  |
//! | `parity_cie94_*_vs_scalar_256_grid`               | ~45 sec  |
//! | `parity_ciede2000_prefilter_vs_exact_256_grid`    | ~25 min  |

#![allow(unsafe_code)]

use colorthief_dataset::__bench::*;

const TOTAL: u64 = 256 * 256 * 256;

fn rgb_iter() -> impl Iterator<Item = [u8; 3]> {
  (0..256u32).flat_map(|r| {
    (0..256u32).flat_map(move |g| (0..256u32).map(move |b| [r as u8, g as u8, b as u8]))
  })
}

// =====================================================================
// aarch64 NEON
// =====================================================================

/// aarch64 NEON Delta E 76 ↔ scalar across all 16,777,216 u8 RGB
/// inputs. Strict bit-parity: a deviation here would imply the
/// underlying f32 `(dl² + da²) + db²` computations diverged for at
/// least one query.
#[test]
#[ignore = "256³ = 16.8M queries; run with `cargo test --release --ignored`"]
#[cfg(target_arch = "aarch64")]
fn parity_de76_neon_vs_scalar_256_grid() {
  let mut count = 0u32;
  for rgb in rgb_iter() {
    let q = rgb_to_lab(rgb);
    let s = scalar_nearest_idx(q);
    let n = aarch64_neon_nearest_idx(q);
    if s != n {
      count += 1;
      if count <= 5 {
        eprintln!("rgb={rgb:?} scalar={s} neon={n}");
      }
    }
  }
  assert_eq!(count, 0, "{count} divergences across {TOTAL} queries");
}

/// aarch64 NEON CIE94 ↔ scalar across all 16,777,216 u8 RGB inputs.
#[test]
#[ignore = "256³ = 16.8M queries; run with `cargo test --release --ignored`"]
#[cfg(target_arch = "aarch64")]
fn parity_cie94_neon_vs_scalar_256_grid() {
  let mut count = 0u32;
  for rgb in rgb_iter() {
    let q = rgb_to_lab(rgb);
    let s = cie94_nearest_idx(q);
    let n = cie94_aarch64_neon_nearest_idx(q);
    if s != n {
      count += 1;
      if count <= 5 {
        eprintln!("rgb={rgb:?} scalar={s} neon={n}");
      }
    }
  }
  assert_eq!(count, 0, "{count} divergences across {TOTAL} queries");
}

// =====================================================================
// x86_64 — runtime feature-detected
// =====================================================================

/// x86 AVX-512F Delta E 76 ↔ scalar. Skipped if AVX-512F is not
/// detected on the host running the test binary.
#[test]
#[ignore = "256³ = 16.8M queries; run with `cargo test --release --ignored`"]
#[cfg(target_arch = "x86_64")]
fn parity_de76_avx512_vs_scalar_256_grid() {
  if !std::is_x86_feature_detected!("avx512f") {
    eprintln!("skipping: AVX-512F not detected on this host");
    return;
  }
  let mut count = 0u32;
  for rgb in rgb_iter() {
    let q = rgb_to_lab(rgb);
    let s = scalar_nearest_idx(q);
    // SAFETY: feature just verified.
    let v = unsafe { x86_avx512_nearest_idx(q) };
    if s != v {
      count += 1;
      if count <= 5 {
        eprintln!("rgb={rgb:?} scalar={s} avx512={v}");
      }
    }
  }
  assert_eq!(count, 0, "{count} divergences across {TOTAL} queries");
}

/// x86 AVX2 Delta E 76 ↔ scalar.
#[test]
#[ignore = "256³ = 16.8M queries; run with `cargo test --release --ignored`"]
#[cfg(target_arch = "x86_64")]
fn parity_de76_avx2_vs_scalar_256_grid() {
  if !std::is_x86_feature_detected!("avx2") {
    eprintln!("skipping: AVX2 not detected on this host");
    return;
  }
  let mut count = 0u32;
  for rgb in rgb_iter() {
    let q = rgb_to_lab(rgb);
    let s = scalar_nearest_idx(q);
    // SAFETY: feature just verified.
    let v = unsafe { x86_avx2_nearest_idx(q) };
    if s != v {
      count += 1;
      if count <= 5 {
        eprintln!("rgb={rgb:?} scalar={s} avx2={v}");
      }
    }
  }
  assert_eq!(count, 0, "{count} divergences across {TOTAL} queries");
}

/// x86 SSE4.1 Delta E 76 ↔ scalar.
#[test]
#[ignore = "256³ = 16.8M queries; run with `cargo test --release --ignored`"]
#[cfg(target_arch = "x86_64")]
fn parity_de76_sse41_vs_scalar_256_grid() {
  if !std::is_x86_feature_detected!("sse4.1") {
    eprintln!("skipping: SSE4.1 not detected on this host");
    return;
  }
  let mut count = 0u32;
  for rgb in rgb_iter() {
    let q = rgb_to_lab(rgb);
    let s = scalar_nearest_idx(q);
    // SAFETY: feature just verified.
    let v = unsafe { x86_sse41_nearest_idx(q) };
    if s != v {
      count += 1;
      if count <= 5 {
        eprintln!("rgb={rgb:?} scalar={s} sse41={v}");
      }
    }
  }
  assert_eq!(count, 0, "{count} divergences across {TOTAL} queries");
}

/// x86 AVX-512F CIE94 ↔ scalar.
#[test]
#[ignore = "256³ = 16.8M queries; run with `cargo test --release --ignored`"]
#[cfg(target_arch = "x86_64")]
fn parity_cie94_avx512_vs_scalar_256_grid() {
  if !std::is_x86_feature_detected!("avx512f") {
    eprintln!("skipping: AVX-512F not detected on this host");
    return;
  }
  let mut count = 0u32;
  for rgb in rgb_iter() {
    let q = rgb_to_lab(rgb);
    let s = cie94_nearest_idx(q);
    // SAFETY: feature just verified.
    let v = unsafe { cie94_x86_avx512_nearest_idx(q) };
    if s != v {
      count += 1;
      if count <= 5 {
        eprintln!("rgb={rgb:?} scalar={s} avx512={v}");
      }
    }
  }
  assert_eq!(count, 0, "{count} divergences across {TOTAL} queries");
}

/// x86 AVX2 CIE94 ↔ scalar.
#[test]
#[ignore = "256³ = 16.8M queries; run with `cargo test --release --ignored`"]
#[cfg(target_arch = "x86_64")]
fn parity_cie94_avx2_vs_scalar_256_grid() {
  if !std::is_x86_feature_detected!("avx2") {
    eprintln!("skipping: AVX2 not detected on this host");
    return;
  }
  let mut count = 0u32;
  for rgb in rgb_iter() {
    let q = rgb_to_lab(rgb);
    let s = cie94_nearest_idx(q);
    // SAFETY: feature just verified.
    let v = unsafe { cie94_x86_avx2_nearest_idx(q) };
    if s != v {
      count += 1;
      if count <= 5 {
        eprintln!("rgb={rgb:?} scalar={s} avx2={v}");
      }
    }
  }
  assert_eq!(count, 0, "{count} divergences across {TOTAL} queries");
}

/// x86 SSE4.1 CIE94 ↔ scalar.
#[test]
#[ignore = "256³ = 16.8M queries; run with `cargo test --release --ignored`"]
#[cfg(target_arch = "x86_64")]
fn parity_cie94_sse41_vs_scalar_256_grid() {
  if !std::is_x86_feature_detected!("sse4.1") {
    eprintln!("skipping: SSE4.1 not detected on this host");
    return;
  }
  let mut count = 0u32;
  for rgb in rgb_iter() {
    let q = rgb_to_lab(rgb);
    let s = cie94_nearest_idx(q);
    // SAFETY: feature just verified.
    let v = unsafe { cie94_x86_sse41_nearest_idx(q) };
    if s != v {
      count += 1;
      if count <= 5 {
        eprintln!("rgb={rgb:?} scalar={s} sse41={v}");
      }
    }
  }
  assert_eq!(count, 0, "{count} divergences across {TOTAL} queries");
}

// =====================================================================
// WASM SIMD128
// =====================================================================

/// WASM SIMD128 Delta E 76 ↔ scalar.
#[test]
#[ignore = "256³ = 16.8M queries; run with `cargo test --release --ignored`"]
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn parity_de76_wasm_simd128_vs_scalar_256_grid() {
  let mut count = 0u32;
  for rgb in rgb_iter() {
    let q = rgb_to_lab(rgb);
    let s = scalar_nearest_idx(q);
    let v = wasm_simd128_nearest_idx(q);
    if s != v {
      count += 1;
      if count <= 5 {
        eprintln!("rgb={rgb:?} scalar={s} simd128={v}");
      }
    }
  }
  assert_eq!(count, 0, "{count} divergences across {TOTAL} queries");
}

/// WASM SIMD128 CIE94 ↔ scalar.
#[test]
#[ignore = "256³ = 16.8M queries; run with `cargo test --release --ignored`"]
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn parity_cie94_wasm_simd128_vs_scalar_256_grid() {
  let mut count = 0u32;
  for rgb in rgb_iter() {
    let q = rgb_to_lab(rgb);
    let s = cie94_nearest_idx(q);
    let v = cie94_wasm_simd128_nearest_idx(q);
    if s != v {
      count += 1;
      if count <= 5 {
        eprintln!("rgb={rgb:?} scalar={s} simd128={v}");
      }
    }
  }
  assert_eq!(count, 0, "{count} divergences across {TOTAL} queries");
}

// =====================================================================
// CIEDE2000 prefilter ↔ exact full-scan
// =====================================================================

/// CIEDE2000 prefilter (K=96) ↔ exact full-scan across all 16,777,216
/// u8 RGB inputs. The 17³ inline test only checks 4913 queries; this
/// is the airtight version that proves K=96 is bit-equivalent across
/// every reachable u8 RGB palette query.
///
/// Slow: ~25 min on Apple Silicon in release mode (each query runs
/// both metrics, ≈ 92 µs combined). Run nightly in CI rather than on
/// every PR.
#[test]
#[ignore = "256³ × ~92µs/query ≈ 25 min release; run nightly"]
fn parity_ciede2000_prefilter_vs_exact_256_grid() {
  let mut count = 0u32;
  for rgb in rgb_iter() {
    let q = rgb_to_lab(rgb);
    let exact = ciede2000_nearest_idx(q);
    let prefilter = ciede2000_prefiltered_nearest_idx(q);
    if exact != prefilter {
      count += 1;
      if count <= 5 {
        eprintln!("rgb={rgb:?} exact={exact} prefilter={prefilter}");
      }
    }
  }
  assert_eq!(count, 0, "{count} divergences across {TOTAL} queries");
}
