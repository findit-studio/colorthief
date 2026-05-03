//! Property-based tests for the nearest-neighbor pipeline.
//!
//! Complements the deterministic 17³ grid tests in
//! `src/nearest/mod.rs::tests` and the airtight 256³ sweeps in
//! `tests/parity_exhaustive.rs`. proptest contributes random-input
//! coverage that doesn't fall on grid boundaries — useful for catching
//! off-by-one issues in tail loops, cancellation near LAB equality,
//! and shrinking-driven minimal failure cases.

use colorthief_dataset::{__bench::*, COLORS};
use proptest::prelude::*;

// LAB ranges:
//   L: [0, 100] in normal sRGB; allow [-10, 110] for out-of-gamut headroom.
//   a, b: roughly [-128, 127] in sRGB; allow [-200, 200] for safety.

proptest! {
  /// Self-distance is zero: `delta_e_76_sq(lab, lab) = 0` for any LAB
  /// triple (subtraction of equal f32 values is exact).
  #[test]
  fn prop_de76_self_distance_zero(
    l in -10.0f32..110.0,
    a in -200.0f32..200.0,
    b in -200.0f32..200.0,
  ) {
    let lab = [l, a, b];
    prop_assert_eq!(delta_e_76_sq(lab, lab), 0.0);
  }

  /// Symmetric: `delta_e_76_sq(a, b) == delta_e_76_sq(b, a)`. Squared
  /// Euclidean is symmetric by construction; pin the property so any
  /// future formula refactor that breaks symmetry trips here.
  #[test]
  fn prop_de76_symmetric(
    l1 in -10.0f32..110.0, a1 in -200.0f32..200.0, b1 in -200.0f32..200.0,
    l2 in -10.0f32..110.0, a2 in -200.0f32..200.0, b2 in -200.0f32..200.0,
  ) {
    let lab1 = [l1, a1, b1];
    let lab2 = [l2, a2, b2];
    prop_assert_eq!(delta_e_76_sq(lab1, lab2), delta_e_76_sq(lab2, lab1));
  }

  /// Non-negative: distance is the sum of squares; always ≥ 0.
  #[test]
  fn prop_de76_nonneg(
    l1 in -10.0f32..110.0, a1 in -200.0f32..200.0, b1 in -200.0f32..200.0,
    l2 in -10.0f32..110.0, a2 in -200.0f32..200.0, b2 in -200.0f32..200.0,
  ) {
    let lab1 = [l1, a1, b1];
    let lab2 = [l2, a2, b2];
    prop_assert!(delta_e_76_sq(lab1, lab2) >= 0.0);
  }

  /// CIE94 self-distance is zero (every Δ is zero, ΔH² clamp doesn't
  /// fire).
  #[test]
  fn prop_cie94_self_distance_zero(
    l in -10.0f32..110.0,
    a in -200.0f32..200.0,
    b in -200.0f32..200.0,
  ) {
    let lab = [l, a, b];
    prop_assert_eq!(delta_e_94_sq(lab, lab), 0.0);
  }

  /// CIE94 is non-negative.
  #[test]
  fn prop_cie94_nonneg(
    l1 in -10.0f32..110.0, a1 in -200.0f32..200.0, b1 in -200.0f32..200.0,
    l2 in -10.0f32..110.0, a2 in -200.0f32..200.0, b2 in -200.0f32..200.0,
  ) {
    let lab1 = [l1, a1, b1];
    let lab2 = [l2, a2, b2];
    prop_assert!(delta_e_94_sq(lab1, lab2) >= 0.0);
  }

  /// CIEDE2000 self-distance is zero.
  #[test]
  fn prop_ciede2000_self_distance_zero(
    l in -10.0f32..110.0,
    a in -200.0f32..200.0,
    b in -200.0f32..200.0,
  ) {
    let lab = [l, a, b];
    prop_assert_eq!(delta_e_2000_sq(lab, lab), 0.0);
  }

  /// `nearest_idx` always returns a valid index into `COLORS`.
  #[test]
  fn prop_de76_nearest_returns_valid_idx(rgb: [u8; 3]) {
    let q = rgb_to_lab(rgb);
    prop_assert!(scalar_nearest_idx(q) < COLORS.len());
  }

  /// CIE94 lookup returns a valid index.
  #[test]
  fn prop_cie94_nearest_returns_valid_idx(rgb: [u8; 3]) {
    let q = rgb_to_lab(rgb);
    prop_assert!(cie94_nearest_idx(q) < COLORS.len());
  }

  /// CIEDE2000 lookup returns a valid index.
  #[test]
  fn prop_ciede2000_nearest_returns_valid_idx(rgb: [u8; 3]) {
    let q = rgb_to_lab(rgb);
    prop_assert!(ciede2000_nearest_idx(q) < COLORS.len());
    prop_assert!(ciede2000_prefiltered_nearest_idx(q) < COLORS.len());
  }

  /// Querying with a palette entry's own LAB returns either that
  /// entry or another with bit-equal LAB. Implicit verification that
  /// self-distance is the global minimum: if it weren't, a different
  /// entry could win and the LABs would differ.
  #[test]
  fn prop_de76_idempotent_for_palette_lab(i in 0usize..COLORS.len()) {
    let entry_lab = COLORS[i].lab();
    let j = scalar_nearest_idx(entry_lab);
    let returned_lab = COLORS[j].lab();
    prop_assert_eq!(returned_lab, entry_lab);
  }
}

// SIMD ↔ scalar parity on random RGB inputs. Quick continuous check
// that complements the exhaustive 256³ sweep in
// `tests/parity_exhaustive.rs` (which is `#[ignore]`-gated). Runs on
// every `cargo test`.

#[cfg(target_arch = "aarch64")]
proptest! {
  #[test]
  fn prop_de76_neon_matches_scalar_random_rgb(rgb: [u8; 3]) {
    let q = rgb_to_lab(rgb);
    prop_assert_eq!(scalar_nearest_idx(q), aarch64_neon_nearest_idx(q));
  }

  #[test]
  fn prop_cie94_neon_matches_scalar_random_rgb(rgb: [u8; 3]) {
    let q = rgb_to_lab(rgb);
    prop_assert_eq!(cie94_nearest_idx(q), cie94_aarch64_neon_nearest_idx(q));
  }
}
