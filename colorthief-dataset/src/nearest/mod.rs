//! Nearest-neighbor lookup against the xkcd LAB palette.
//!
//! Public entry point is [`crate::Color::nearest_to`]; this module
//! owns the actual scan over the 949-entry palette plus per-arch SIMD
//! specialisation, hidden behind a single internal [`nearest_idx`]
//! dispatcher.
//!
//! # Backends
//!
//! - [`scalar`] — always compiled, the reference implementation.
//! - [`aarch64_neon`] — `cfg(target_arch = "aarch64")`, 4 entries/iter
//!   via 128-bit NEON. Compile-time gated; NEON is mandatory in
//!   Armv8-A.
//! - [`x86_sse41`] — `cfg(target_arch = "x86_64")`, 4 entries/iter via
//!   128-bit SSE4.1. Runtime feature-detected (`std`-only).
//! - [`x86_avx2`] — `cfg(target_arch = "x86_64")`, 8 entries/iter via
//!   256-bit AVX2. Runtime feature-detected (`std`-only).
//! - [`x86_avx512`] — `cfg(target_arch = "x86_64")`, 16 entries/iter
//!   via 512-bit AVX-512F. Runtime feature-detected (`std`-only).
//!   Requires Rust 1.89+ for stable `_mm512_*` intrinsics; the
//!   workspace MSRV is 1.95.
//! - [`wasm_simd128`] — `cfg(all(target_arch = "wasm32",
//!   target_feature = "simd128"))`, 4 entries/iter via WASM SIMD128.
//!   Compile-time gated.
//!
//! # Dispatch
//!
//! - On aarch64 → `aarch64_neon` (compile-time).
//! - On x86_64 with `feature = "std"` → runtime detection picks the
//!   highest-tier available (`avx2` > `sse4.1` > `scalar`). On
//!   `no_std` x86 we fall through to scalar — runtime detection
//!   needs `std`.
//! - On wasm32 with `target_feature = "simd128"` → `wasm_simd128`
//!   (compile-time).
//! - Else → scalar.
//!
//! Pattern mirrors the colconv project's `src/row/arch/` layout.
//!
//! # Bit-parity contract
//!
//! Every backend evaluates the squared distance with the same
//! associativity (`(dl² + da²) + db²`) and uses plain mul/add (no
//! FMA), so they produce bit-identical `f32` results on the same
//! inputs. The grid-parity tests in this module enforce this against
//! a representative RGB grid for every backend reachable on the
//! current target.

use crate::{
  Color,
  generated::{COLORS, LABS_A, LABS_B, LABS_C, LABS_L},
};

pub(crate) mod scalar;

/// CIEDE2000 — scalar-only on every target. See [`ciede2000`] for why
/// SIMD isn't worth pursuing here. A NEON attempt was benchmarked
/// against the scalar baseline on 2026-05-03 and regressed by ~35%
/// (115.9 µs vs 85.9 µs / query) — the transcendental-heavy formula
/// can't usefully parallelise, so we keep the scalar path.
pub(crate) mod ciede2000;

/// CIEDE2000 candidate-set LUT — gated on `feature = "lut"` (default
/// on). The LUT is pre-computed at xtask codegen time: each of the
/// 32³ cells stores the small set of palette indices that are the
/// CIEDE2000-nearest at *some* RGB inside the cell's 8×8×8 box. At
/// runtime, the cell lookup is O(1) and the candidate scan is bounded
/// by the per-cell max (10 in the current palette), which collapses
/// the per-query CIEDE2000 cost from ~71 µs (full scan over 949) to a
/// few hundred ns. Provably exact at u8 RGB resolution.
#[cfg(feature = "lut")]
pub(crate) mod ciede2000_lut;

/// CIE94 (Delta E 94) — scalar reference. The SIMD-friendly formula
/// (no `atan2` / `sin` / `cos` / `exp`; only `sqrt` + arithmetic) has
/// per-arch backends below mirroring Delta E 76.
pub(crate) mod cie94;

// `target_feature = "neon"` (not just `target_arch = "aarch64"`):
// `aarch64-unknown-none-softfloat` is a Tier-2 target with
// `target_arch = "aarch64"` but no `target_feature = "neon"`, and
// calling a `#[target_feature(enable = "neon")]` fn there is UB per
// the Rust reference. Other aarch64 targets (-linux-gnu, -apple-darwin,
// -unknown-none, etc.) all have NEON in the default feature set, so
// this gate excludes only the softfloat variant.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub(crate) mod cie94_aarch64_neon;

#[cfg(target_arch = "x86_64")]
pub(crate) mod cie94_x86_sse41;

#[cfg(target_arch = "x86_64")]
pub(crate) mod cie94_x86_avx2;

#[cfg(target_arch = "x86_64")]
pub(crate) mod cie94_x86_avx512;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub(crate) mod cie94_wasm_simd128;

// See the comment on `cie94_aarch64_neon` above for why we gate on
// `target_feature = "neon"` rather than just `target_arch = "aarch64"`.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub(crate) mod aarch64_neon;

#[cfg(target_arch = "x86_64")]
pub(crate) mod x86_sse41;

#[cfg(target_arch = "x86_64")]
pub(crate) mod x86_avx2;

#[cfg(target_arch = "x86_64")]
pub(crate) mod x86_avx512;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub(crate) mod wasm_simd128;

/// Internal dispatcher: returns the index into [`COLORS`] of the entry
/// whose pre-computed LAB is closest to `query` by Delta E 76 (squared
/// Euclidean — `sqrt` is monotonic, no need to take it).
///
/// # Tier-forcing cfg flags
///
/// Mirrors colconv's coverage strategy. Each flag short-circuits the
/// dispatcher to a lower tier so coverage runs can exercise every
/// branch even on a host whose CPU only naturally hits the top tier:
///
/// - `--cfg colorthief_force_scalar` — bypass every SIMD backend and
///   call the scalar reference unconditionally.
/// - `--cfg colorthief_disable_avx512` — on x86_64, skip the AVX-512F
///   tier so the dispatcher falls through to AVX2 (or lower).
/// - `--cfg colorthief_disable_avx2` — on x86_64, skip the AVX2 tier
///   so the dispatcher falls through to SSE4.1 (or scalar if SSE4.1
///   is also unavailable at runtime). Stacks with
///   `colorthief_disable_avx512` to force the SSE4.1 path.
///
/// These flags are declared in the workspace's
/// `[workspace.lints.rust] unexpected_cfgs.check-cfg` so passing them
/// via `RUSTFLAGS` doesn't trip the unexpected-cfgs lint.
///
/// `#[allow(unsafe_code)]` is scoped here because the x86 backends are
/// `unsafe fn` (the `#[target_feature]` attribute requires it) and we
/// call them inside `is_x86_feature_detected!` guards. The aarch64
/// and WASM backends expose safe wrappers so they don't need the
/// allow.
///
/// `#[allow(unreachable_code)]` because each per-arch cfg branch
/// `return`s and on a target that hits Tier 1 the trailing scalar
/// fallback is unreachable. The trailing call exists for x86_64 (when
/// no SIMD feature detects), no_std x86_64, every other arch, and the
/// `colorthief_force_scalar` coverage runs.
#[allow(unsafe_code)]
#[allow(unreachable_code)]
#[inline]
pub(crate) fn nearest_idx(query: [f32; 3]) -> usize {
  // Tier 1: aarch64 NEON. NEON is part of the default feature set on
  // every aarch64 target Rust supports *except*
  // `aarch64-unknown-none-softfloat` (Tier 2, soft-float embedded). We
  // gate on `target_feature = "neon"` rather than just `target_arch`
  // so the softfloat target falls through to scalar — calling a
  // `#[target_feature(enable = "neon")]` fn without the feature in
  // scope is UB per the Rust reference.
  #[cfg(all(
    target_arch = "aarch64",
    target_feature = "neon",
    not(colorthief_force_scalar)
  ))]
  {
    return aarch64_neon::nearest_idx(query);
  }

  // Tier 1: WASM SIMD128. Compile-time gated; the module is only
  // declared when `target_feature = "simd128"`.
  #[cfg(all(
    target_arch = "wasm32",
    target_feature = "simd128",
    not(colorthief_force_scalar)
  ))]
  {
    return wasm_simd128::nearest_idx(query);
  }

  // Tier 1-3: x86_64 std runtime feature detection. AVX-512F →
  // AVX2 → SSE4.1 cascade; the `colorthief_disable_avx512` and
  // `colorthief_disable_avx2` flags force coverage runs through the
  // lower tiers even on machines that natively support the higher
  // ones. The `is_x86_feature_detected!` macro caches the lookup in
  // an atomic so per-call overhead is a single relaxed load.
  #[cfg(all(target_arch = "x86_64", feature = "std", not(colorthief_force_scalar)))]
  {
    if !cfg!(colorthief_disable_avx512) && std::is_x86_feature_detected!("avx512f") {
      // SAFETY: feature just verified; `x86_avx512::nearest_idx`
      // carries `#[target_feature(enable = "avx512f")]`.
      return unsafe { x86_avx512::nearest_idx(query) };
    }
    if !cfg!(colorthief_disable_avx2) && std::is_x86_feature_detected!("avx2") {
      // SAFETY: feature just verified.
      return unsafe { x86_avx2::nearest_idx(query) };
    }
    if std::is_x86_feature_detected!("sse4.1") {
      // SAFETY: feature just verified.
      return unsafe { x86_sse41::nearest_idx(query) };
    }
  }

  // Fallback: scalar.
  scalar::nearest_idx(query)
}

/// Convenience wrapper used by [`crate::Color::nearest_to`].
#[inline]
pub(crate) fn nearest(query: [f32; 3]) -> &'static Color {
  COLORS[nearest_idx(query)]
}

/// CIEDE2000 nearest-neighbor — the dispatcher behind both
/// [`crate::Color::nearest_to_ciede2000`] and
/// [`crate::Color::nearest_to_ciede2000_exact`].
///
/// When `feature = "lut"` is enabled (the default), routes through
/// the candidate-set LUT in [`ciede2000_lut`] — provably exact at u8
/// RGB resolution, ~few-hundred-ns/query. When the feature is
/// disabled, falls through to the full-scan reference
/// [`ciede2000::nearest_idx`] (~71 µs/query, also provably exact).
///
/// The Delta E 76 prefilter at `K = 96` is **not** used as a
/// production path: a 256³ exhaustive sweep
/// (`tests/parity_exhaustive.rs::parity_ciede2000_prefilter_vs_exact_256_grid`)
/// showed 2283 divergences vs. full-scan, so the prefilter can't
/// claim strict exactness. It's retained as a benchmark baseline only.
#[cfg(feature = "lut")]
#[inline]
pub(crate) fn nearest_ciede2000(rgb: [u8; 3]) -> &'static Color {
  let query = crate::rgb_to_lab(rgb);
  COLORS[ciede2000_lut::nearest_idx(rgb, query)]
}

#[cfg(not(feature = "lut"))]
#[inline]
pub(crate) fn nearest_ciede2000(rgb: [u8; 3]) -> &'static Color {
  let query = crate::rgb_to_lab(rgb);
  COLORS[ciede2000::nearest_idx(query)]
}

/// CIE94 (Delta E 94) nearest-neighbor with the same SIMD dispatch
/// cascade as [`nearest_idx`] for Delta E 76. Honours the same
/// coverage cfg flags (`colorthief_force_scalar`,
/// `colorthief_disable_avx2`).
#[allow(unsafe_code)]
#[allow(unreachable_code)]
#[inline]
pub(crate) fn nearest_cie94(query: [f32; 3]) -> &'static Color {
  // Tier 1: aarch64 NEON. See `nearest_idx` above for why we gate on
  // `target_feature = "neon"` rather than just `target_arch`.
  #[cfg(all(
    target_arch = "aarch64",
    target_feature = "neon",
    not(colorthief_force_scalar)
  ))]
  {
    return COLORS[cie94_aarch64_neon::nearest_idx(query)];
  }

  // Tier 1: WASM SIMD128 (compile-time gated).
  #[cfg(all(
    target_arch = "wasm32",
    target_feature = "simd128",
    not(colorthief_force_scalar)
  ))]
  {
    return COLORS[cie94_wasm_simd128::nearest_idx(query)];
  }

  // Tier 1-3: x86_64 std runtime feature detection. AVX-512F → AVX2
  // → SSE4.1, same cascade as Delta E 76's `nearest_idx`. Gated on
  // `feature = "std"` because `is_x86_feature_detected!` requires
  // `std`; on `no_std` x86_64 we fall through to scalar (matches the
  // Delta E 76 cascade above at line 170).
  #[cfg(all(target_arch = "x86_64", feature = "std", not(colorthief_force_scalar)))]
  {
    if !cfg!(colorthief_disable_avx512) && std::is_x86_feature_detected!("avx512f") {
      // SAFETY: feature just verified.
      return COLORS[unsafe { cie94_x86_avx512::nearest_idx(query) }];
    }
    if !cfg!(colorthief_disable_avx2) && std::is_x86_feature_detected!("avx2") {
      // SAFETY: feature just verified.
      return COLORS[unsafe { cie94_x86_avx2::nearest_idx(query) }];
    }
    if std::is_x86_feature_detected!("sse4.1") {
      // SAFETY: feature just verified.
      return COLORS[unsafe { cie94_x86_sse41::nearest_idx(query) }];
    }
  }

  COLORS[cie94::nearest_idx(query)]
}

#[cfg(test)]
#[allow(unsafe_code)]
mod tests {
  use super::*;

  /// Iterate the standard parity grid (17³ = 4913 RGB points evenly
  /// spaced 16 apart). Reused across every backend's parity test.
  ///
  /// Gated on `feature = "std"` to match the parity tests below — they
  /// all need `Vec` to collect mismatches, which requires `alloc` (and
  /// the test harness itself needs std). On `cargo hack test
  /// --no-default-features --features alloc` no SIMD-arch test is
  /// reachable on Linux/Windows runners (target_arch = x86_64 with
  /// `feature = "std"` filter excludes them), so this helper would
  /// otherwise become dead code under `-Dwarnings`.
  #[cfg(feature = "std")]
  fn parity_grid() -> impl Iterator<Item = [u8; 3]> {
    (0..256u32).step_by(16).flat_map(move |r| {
      (0..256u32).step_by(16).flat_map(move |g| {
        (0..256u32)
          .step_by(16)
          .map(move |b| [r as u8, g as u8, b as u8])
      })
    })
  }

  /// SoA arrays must align with the AoS [`COLORS`] indexing: every
  /// `LABS_*[i]` matches `COLORS[i].lab()`. Pins the xtask invariant
  /// that the SoA write order matches the const emission order.
  #[test]
  fn soa_lab_arrays_align_with_aos_colors() {
    assert_eq!(LABS_L.len(), COLORS.len());
    assert_eq!(LABS_A.len(), COLORS.len());
    assert_eq!(LABS_B.len(), COLORS.len());
    for (i, c) in COLORS.iter().enumerate() {
      let lab = c.lab();
      assert_eq!(LABS_L[i], lab[0], "L mismatch at index {i}");
      assert_eq!(LABS_A[i], lab[1], "a mismatch at index {i}");
      assert_eq!(LABS_B[i], lab[2], "b mismatch at index {i}");
    }
  }

  /// aarch64 NEON ↔ scalar. Needs `feature = "std"` for `Vec` and
  /// the test harness; under `--no-default-features --features alloc`
  /// the test is skipped (the standard test runner requires std).
  #[test]
  #[cfg(all(target_arch = "aarch64", target_feature = "neon", feature = "std"))]
  fn neon_and_scalar_agree_across_grid() {
    let mut mismatches = Vec::new();
    for rgb in parity_grid() {
      let query = crate::rgb_to_lab(rgb);
      let s = scalar::nearest_idx(query);
      let n = aarch64_neon::nearest_idx(query);
      if s != n {
        mismatches.push((rgb, s, n));
      }
    }
    assert!(
      mismatches.is_empty(),
      "{} scalar/NEON mismatches across the 17³ grid; first few: {:?}",
      mismatches.len(),
      &mismatches[..mismatches.len().min(5)]
    );
  }

  /// x86 SSE4.1 ↔ scalar (runs only when SSE4.1 is detected on the
  /// host running the test binary).
  #[test]
  #[cfg(all(target_arch = "x86_64", feature = "std"))]
  fn sse41_and_scalar_agree_across_grid() {
    if !std::is_x86_feature_detected!("sse4.1") {
      eprintln!("skipping: SSE4.1 not detected on this host");
      return;
    }
    let mut mismatches = Vec::new();
    for rgb in parity_grid() {
      let query = crate::rgb_to_lab(rgb);
      let s = scalar::nearest_idx(query);
      // SAFETY: feature just verified.
      let v = unsafe { x86_sse41::nearest_idx(query) };
      if s != v {
        mismatches.push((rgb, s, v));
      }
    }
    assert!(
      mismatches.is_empty(),
      "{} scalar/SSE4.1 mismatches; first few: {:?}",
      mismatches.len(),
      &mismatches[..mismatches.len().min(5)]
    );
  }

  /// x86 AVX-512F ↔ scalar (runs only when AVX-512F is detected on
  /// the host).
  #[test]
  #[cfg(all(target_arch = "x86_64", feature = "std"))]
  fn avx512_and_scalar_agree_across_grid() {
    if !std::is_x86_feature_detected!("avx512f") {
      eprintln!("skipping: AVX-512F not detected on this host");
      return;
    }
    let mut mismatches = Vec::new();
    for rgb in parity_grid() {
      let query = crate::rgb_to_lab(rgb);
      let s = scalar::nearest_idx(query);
      // SAFETY: feature just verified.
      let v = unsafe { x86_avx512::nearest_idx(query) };
      if s != v {
        mismatches.push((rgb, s, v));
      }
    }
    assert!(
      mismatches.is_empty(),
      "{} scalar/AVX-512F mismatches; first few: {:?}",
      mismatches.len(),
      &mismatches[..mismatches.len().min(5)]
    );
  }

  /// CIE94 x86 AVX-512F ↔ scalar (runs only when AVX-512F is detected
  /// on the host).
  #[test]
  #[cfg(all(target_arch = "x86_64", feature = "std"))]
  fn cie94_avx512_and_scalar_agree_across_grid() {
    if !std::is_x86_feature_detected!("avx512f") {
      eprintln!("skipping: AVX-512F not detected on this host");
      return;
    }
    let mut mismatches = Vec::new();
    for rgb in parity_grid() {
      let query = crate::rgb_to_lab(rgb);
      let s = cie94::nearest_idx(query);
      // SAFETY: feature verified.
      let v = unsafe { cie94_x86_avx512::nearest_idx(query) };
      if s != v {
        mismatches.push((rgb, s, v));
      }
    }
    assert!(
      mismatches.is_empty(),
      "{} CIE94 scalar/AVX-512F mismatches; first few: {:?}",
      mismatches.len(),
      &mismatches[..mismatches.len().min(5)]
    );
  }

  /// x86 AVX2 ↔ scalar (runs only when AVX2 is detected on the host).
  #[test]
  #[cfg(all(target_arch = "x86_64", feature = "std"))]
  fn avx2_and_scalar_agree_across_grid() {
    if !std::is_x86_feature_detected!("avx2") {
      eprintln!("skipping: AVX2 not detected on this host");
      return;
    }
    let mut mismatches = Vec::new();
    for rgb in parity_grid() {
      let query = crate::rgb_to_lab(rgb);
      let s = scalar::nearest_idx(query);
      // SAFETY: feature just verified.
      let v = unsafe { x86_avx2::nearest_idx(query) };
      if s != v {
        mismatches.push((rgb, s, v));
      }
    }
    assert!(
      mismatches.is_empty(),
      "{} scalar/AVX2 mismatches; first few: {:?}",
      mismatches.len(),
      &mismatches[..mismatches.len().min(5)]
    );
  }

  /// CIE94 aarch64 NEON ↔ scalar.
  #[test]
  #[cfg(all(target_arch = "aarch64", target_feature = "neon", feature = "std"))]
  fn cie94_neon_and_scalar_agree_across_grid() {
    let mut mismatches = Vec::new();
    for rgb in parity_grid() {
      let query = crate::rgb_to_lab(rgb);
      let s = cie94::nearest_idx(query);
      let n = cie94_aarch64_neon::nearest_idx(query);
      if s != n {
        mismatches.push((rgb, s, n));
      }
    }
    assert!(
      mismatches.is_empty(),
      "{} CIE94 scalar/NEON mismatches; first few: {:?}",
      mismatches.len(),
      &mismatches[..mismatches.len().min(5)]
    );
  }

  /// CIE94 x86 SSE4.1 ↔ scalar.
  #[test]
  #[cfg(all(target_arch = "x86_64", feature = "std"))]
  fn cie94_sse41_and_scalar_agree_across_grid() {
    if !std::is_x86_feature_detected!("sse4.1") {
      eprintln!("skipping: SSE4.1 not detected");
      return;
    }
    let mut mismatches = Vec::new();
    for rgb in parity_grid() {
      let query = crate::rgb_to_lab(rgb);
      let s = cie94::nearest_idx(query);
      // SAFETY: feature verified.
      let v = unsafe { cie94_x86_sse41::nearest_idx(query) };
      if s != v {
        mismatches.push((rgb, s, v));
      }
    }
    assert!(
      mismatches.is_empty(),
      "{} CIE94 scalar/SSE4.1 mismatches; first few: {:?}",
      mismatches.len(),
      &mismatches[..mismatches.len().min(5)]
    );
  }

  /// CIE94 x86 AVX2 ↔ scalar.
  #[test]
  #[cfg(all(target_arch = "x86_64", feature = "std"))]
  fn cie94_avx2_and_scalar_agree_across_grid() {
    if !std::is_x86_feature_detected!("avx2") {
      eprintln!("skipping: AVX2 not detected");
      return;
    }
    let mut mismatches = Vec::new();
    for rgb in parity_grid() {
      let query = crate::rgb_to_lab(rgb);
      let s = cie94::nearest_idx(query);
      // SAFETY: feature verified.
      let v = unsafe { cie94_x86_avx2::nearest_idx(query) };
      if s != v {
        mismatches.push((rgb, s, v));
      }
    }
    assert!(
      mismatches.is_empty(),
      "{} CIE94 scalar/AVX2 mismatches; first few: {:?}",
      mismatches.len(),
      &mismatches[..mismatches.len().min(5)]
    );
  }

  /// CIE94 WASM SIMD128 ↔ scalar.
  #[test]
  #[cfg(all(target_arch = "wasm32", target_feature = "simd128", feature = "std"))]
  fn cie94_wasm_simd128_and_scalar_agree_across_grid() {
    let mut mismatches = Vec::new();
    for rgb in parity_grid() {
      let query = crate::rgb_to_lab(rgb);
      let s = cie94::nearest_idx(query);
      let v = cie94_wasm_simd128::nearest_idx(query);
      if s != v {
        mismatches.push((rgb, s, v));
      }
    }
    assert!(
      mismatches.is_empty(),
      "{} CIE94 scalar/WASM SIMD128 mismatches; first few: {:?}",
      mismatches.len(),
      &mismatches[..mismatches.len().min(5)]
    );
  }

  /// WASM SIMD128 ↔ scalar.
  #[test]
  #[cfg(all(target_arch = "wasm32", target_feature = "simd128", feature = "std"))]
  fn wasm_simd128_and_scalar_agree_across_grid() {
    let mut mismatches = Vec::new();
    for rgb in parity_grid() {
      let query = crate::rgb_to_lab(rgb);
      let s = scalar::nearest_idx(query);
      let v = wasm_simd128::nearest_idx(query);
      if s != v {
        mismatches.push((rgb, s, v));
      }
    }
    assert!(
      mismatches.is_empty(),
      "{} scalar/WASM SIMD128 mismatches; first few: {:?}",
      mismatches.len(),
      &mismatches[..mismatches.len().min(5)]
    );
  }
}
