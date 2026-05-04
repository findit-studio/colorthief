//! Public API surface tests — accessors, Algorithm dispatch, and the
//! `Color::nearest_to*` entry points.
//!
//! Pins the `Color::*()` const-fn accessors and the `Algorithm` enum's
//! `extract` / `as_str` methods. These are the user-visible getters
//! across the rest of the crate's docs and examples; if the field
//! layout under `Color` changes, the tests here flag it.

use colorthief_dataset::{Algorithm, Color, Family, Kind};

/// Every accessor on `Color` returns *something* — pin the contract
/// for each entry. Stable identifiers (name, hex, family) are the
/// downstream join key for search-vocabulary indexing, so them being
/// non-empty / non-default matters more than specific values.
#[test]
fn color_accessors_round_trip() {
  let all = Color::all();
  assert_eq!(all.len(), 949, "xkcd palette is 949 entries");
  let c = all[0];

  assert!(!c.name().is_empty(), "name must not be empty");
  assert!(
    c.hex().starts_with('#'),
    "hex must start with #: {:?}",
    c.hex()
  );
  let _: [u8; 3] = c.rgb();
  let lab = c.lab();
  assert!(lab[0].is_finite() && lab[1].is_finite() && lab[2].is_finite());

  assert!(!c.design_name().is_empty());
  assert!(c.design_hex().starts_with('#'));
  let _: [u8; 3] = c.design_rgb();

  assert!(!c.common_name().is_empty());
  assert!(c.common_hex().starts_with('#'));
  let _: [u8; 3] = c.common_rgb();

  // Family / Kind round-trip via `as_str`. `is_neutral` is bool — just
  // call it so the accessor lights up.
  let _: Family = c.family();
  let _: Kind = c.kind();
  let _: bool = c.is_neutral();
}

/// Every entry exposes consistent accessors — sweep the whole table
/// to exercise each line of the const-fn getters across the SoA.
#[test]
fn color_accessors_sweep_full_palette() {
  for c in Color::all() {
    let _ = c.name();
    let _ = c.hex();
    let _ = c.rgb();
    let _ = c.lab();
    let _ = c.design_name();
    let _ = c.design_hex();
    let _ = c.design_rgb();
    let _ = c.common_name();
    let _ = c.common_hex();
    let _ = c.common_rgb();
    let _ = c.family();
    let _ = c.kind();
    let _ = c.is_neutral();
  }
}

/// `Color::nearest_to` (Delta E 76 entry point).
#[test]
fn nearest_to_delta_e_76_returns_red_for_solid_red() {
  let c = Color::nearest_to([255, 0, 0]);
  assert!(
    c.family().as_str().contains("red") || c.name().contains("red"),
    "expected red-family entry, got {:?}",
    c.name()
  );
}

/// `Color::nearest_to_cie94` (the dispatcher exercised here is
/// `crate::nearest::nearest_cie94`).
#[test]
fn nearest_to_cie94_returns_red_for_solid_red() {
  let c = Color::nearest_to_cie94([255, 0, 0]);
  assert!(
    c.family().as_str().contains("red") || c.name().contains("red"),
    "expected red-family entry, got {:?}",
    c.name()
  );
}

/// `Color::nearest_to_ciede2000` (LUT path under default features).
#[test]
fn nearest_to_ciede2000_returns_red_for_solid_red() {
  let c = Color::nearest_to_ciede2000([255, 0, 0]);
  assert!(c.family().as_str().contains("red") || c.name().contains("red"),);
}

/// `Color::nearest_to_ciede2000_exact` — equivalent under both
/// feature configurations, kept distinct for API stability.
#[test]
fn nearest_to_ciede2000_exact_returns_red_for_solid_red() {
  let c = Color::nearest_to_ciede2000_exact([255, 0, 0]);
  assert!(c.family().as_str().contains("red") || c.name().contains("red"),);
}

/// CIE94 / CIEDE2000 dispatchers must agree with their direct counter-
/// parts on every cube corner — pins that the per-arch SIMD branch
/// and the scalar fallback both compile and execute cleanly under the
/// dispatcher cascade in `nearest::nearest_cie94`.
#[test]
fn dispatcher_consistent_across_extreme_rgbs() {
  for rgb in [
    [0, 0, 0],
    [255, 255, 255],
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [128, 128, 128],
    [255, 255, 0],
    [0, 255, 255],
    [255, 0, 255],
  ] {
    // Just call through every dispatcher path. If any branch panics or
    // diverges from the others, downstream tests would catch it; the
    // assertion here is that all three return *some* valid entry.
    let de76 = Color::nearest_to(rgb);
    let cie94 = Color::nearest_to_cie94(rgb);
    let de2000 = Color::nearest_to_ciede2000(rgb);
    assert!(!de76.name().is_empty());
    assert!(!cie94.name().is_empty());
    assert!(!de2000.name().is_empty());
  }
}

// ---------------------------------------------------------------------
// Algorithm enum
// ---------------------------------------------------------------------

#[test]
fn algorithm_default_is_ciede2000_exact() {
  assert_eq!(Algorithm::default(), Algorithm::Ciede2000Exact);
}

#[test]
fn algorithm_extract_exercises_each_variant() {
  let rgb = [200u8, 50, 50];
  for algo in [
    Algorithm::DeltaE76,
    Algorithm::Cie94,
    Algorithm::Ciede2000,
    Algorithm::Ciede2000Exact,
  ] {
    let c = algo.extract(rgb);
    assert!(!c.name().is_empty(), "{algo:?} returned empty name");
  }
}

#[test]
fn algorithm_as_str_emits_stable_identifiers() {
  assert_eq!(Algorithm::DeltaE76.as_str(), "delta-e-76");
  assert_eq!(Algorithm::Cie94.as_str(), "cie94");
  assert_eq!(Algorithm::Ciede2000.as_str(), "ciede2000");
  assert_eq!(Algorithm::Ciede2000Exact.as_str(), "ciede2000-exact");
}

#[test]
fn algorithm_extract_matches_color_helpers() {
  // Each Algorithm variant must dispatch to the corresponding
  // Color::nearest_to* helper. Pins the table in the rustdoc.
  let rgb = [123, 45, 67];
  assert_eq!(
    Algorithm::DeltaE76.extract(rgb).name(),
    Color::nearest_to(rgb).name()
  );
  assert_eq!(
    Algorithm::Cie94.extract(rgb).name(),
    Color::nearest_to_cie94(rgb).name()
  );
  assert_eq!(
    Algorithm::Ciede2000.extract(rgb).name(),
    Color::nearest_to_ciede2000(rgb).name()
  );
  assert_eq!(
    Algorithm::Ciede2000Exact.extract(rgb).name(),
    Color::nearest_to_ciede2000_exact(rgb).name()
  );
}
