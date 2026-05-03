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
//! - [`wasm_simd128`] — `cfg(all(target_arch = "wasm32",
//!   target_feature = "simd128"))`, 4 entries/iter via WASM SIMD128.
//!   Compile-time gated.
//!
//! AVX-512 is intentionally absent: its `_mm512_*` intrinsics
//! stabilised in Rust 1.89 and the workspace MSRV is 1.85; revisit
//! after the next MSRV bump.
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
  generated::{COLORS, LABS_A, LABS_B, LABS_L},
};

pub(crate) mod scalar;

/// CIEDE2000 — scalar-only on every target. See [`ciede2000`] for why
/// SIMD isn't worth pursuing here. A NEON attempt was benchmarked
/// against the scalar baseline on 2026-05-03 and regressed by ~35%
/// (115.9 µs vs 85.9 µs / query) — the transcendental-heavy formula
/// can't usefully parallelise, so we keep the scalar path.
pub(crate) mod ciede2000;

/// CIE94 (Delta E 94) — scalar implementation. SIMD-friendly (no
/// `atan2`, `sin`, `cos`, `exp`); SIMD backends to be added in
/// follow-up work.
pub(crate) mod cie94;

#[cfg(target_arch = "aarch64")]
pub(crate) mod aarch64_neon;

#[cfg(target_arch = "x86_64")]
pub(crate) mod x86_sse41;

#[cfg(target_arch = "x86_64")]
pub(crate) mod x86_avx2;

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
/// - `--cfg colorthief_disable_avx2` — on x86_64, skip the AVX2 tier
///   so the dispatcher falls through to SSE4.1 (or scalar if SSE4.1
///   is also unavailable at runtime).
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
  // Tier 1: aarch64 NEON. NEON is mandatory in Armv8-A so no runtime
  // feature detection is needed; compile-time cfg is sufficient.
  #[cfg(all(target_arch = "aarch64", not(colorthief_force_scalar)))]
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

  // Tier 1-2: x86_64 std runtime feature detection. AVX2 first, then
  // SSE4.1; `colorthief_disable_avx2` forces a drop to SSE4.1 for
  // coverage. The `is_x86_feature_detected!` macro caches the lookup
  // in an atomic so per-call overhead is a single relaxed load.
  #[cfg(all(target_arch = "x86_64", feature = "std", not(colorthief_force_scalar)))]
  {
    if !cfg!(colorthief_disable_avx2) && std::is_x86_feature_detected!("avx2") {
      // SAFETY: feature just verified; `x86_avx2::nearest_idx`
      // carries `#[target_feature(enable = "avx2")]`.
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

/// CIEDE2000 nearest-neighbor convenience wrapper used by
/// [`crate::Color::nearest_to_ciede2000_exact`]. Full-scan reference
/// implementation — always scalar; CIEDE2000's `atan2` / `sin` /
/// `exp` / branchy hue wraparound don't vectorise usefully on any of
/// our SIMD backends.
#[inline]
pub(crate) fn nearest_ciede2000(query: [f32; 3]) -> &'static Color {
  COLORS[ciede2000::nearest_idx(query)]
}

/// CIEDE2000 nearest-neighbor with a Delta E 76 prefilter — the
/// default behind [`crate::Color::nearest_to_ciede2000`]. Stage 1
/// scans every entry under the cheap Delta E 76 metric, keeps the
/// top-K (K = [`ciede2000::PREFILTER_K`]) candidates; stage 2
/// re-ranks those K with the full CIEDE2000 formula.
///
/// ~5× faster than [`nearest_ciede2000`] on Apple Silicon and
/// validated to agree with it on the 17³ RGB grid at K = 96 (zero
/// divergences). For inputs outside the validation grid where strict
/// full-scan semantics are required, callers reach for
/// [`crate::Color::nearest_to_ciede2000_exact`].
#[inline]
pub(crate) fn nearest_ciede2000_prefiltered(query: [f32; 3]) -> &'static Color {
  COLORS[ciede2000::nearest_idx_prefiltered(query)]
}

/// CIE94 (Delta E 94) nearest-neighbor convenience wrapper used by
/// [`crate::Color::nearest_to_cie94`]. Currently scalar-only; SIMD
/// backends would mirror the Delta E 76 module structure since the
/// formula has no transcendentals beyond `sqrt`.
#[inline]
pub(crate) fn nearest_cie94(query: [f32; 3]) -> &'static Color {
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
  #[cfg(all(target_arch = "aarch64", feature = "std"))]
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
