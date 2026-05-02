//! Static xkcd color-hierarchy table for human-vocabulary color naming.
//!
//! Each entry exposes five hierarchical names (xkcd → design → common →
//! family → kind), plus the entry's RGB and a pre-computed CIE LAB triple
//! used by [`Color::nearest_to`] for nearest-neighbor lookup.
//!
//! # Why LAB is pre-computed at codegen time
//!
//! The sRGB → LAB pipeline involves a power function (gamma decode) and a
//! cube root (LAB f-function), both transcendental. Computing those for
//! all 950 entries on every query would dwarf the actual nearest-neighbor
//! scan. Pre-computing at `cargo xtask codegen` time pushes that cost out
//! of the hot path; the runtime cost of `nearest_to` is one sRGB→LAB on
//! the query plus 950 squared-distance comparisons (no `sqrt` — squared
//! distance preserves ordering).
//!
//! # Distance metric
//!
//! [`Color::nearest_to`] uses **Delta E 76** (Euclidean distance in LAB).
//! It's adequate for naming against a well-separated 950-color palette;
//! the (~50-line) CIEDE2000 upgrade would only change borderline cases
//! near the gray and yellow regions and is left as a follow-up if real
//! usage shows naming drift.
//!
//! # Attribution
//!
//! The color hierarchy is sourced from Stitch Fix's `colornamer` (Apache
//! 2.0); see `THIRD_PARTY_NOTICES.md` for the full upstream attribution.

#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(docsrs, allow(unused_attributes))]
#![deny(missing_docs)]
// `unsafe_code` is `deny`-not-`forbid` because the per-arch NEON kernel in
// `nearest::aarch64_neon` needs `unsafe` to call `core::arch::aarch64`
// intrinsics (the NEON entry function is `#[target_feature(enable = "neon")]`
// and therefore `unsafe fn`). That module carries a local
// `#[allow(unsafe_code)]` and is the ONLY place unsafe code is allowed.
#![deny(unsafe_code)]

mod generated;
mod nearest;

pub use generated::{COLORS, Family, Kind};

/// **Not a stable API.**
///
/// Hidden helpers used by `colorthief-dataset/benches/nearest.rs` to
/// compare backends head-to-head. Calling these directly bypasses the
/// dispatcher in [`Color::nearest_to`] — production code should use
/// the public method.
#[doc(hidden)]
pub mod __bench {
  pub use crate::nearest::{
    ciede2000::nearest_idx as ciede2000_nearest_idx, scalar::nearest_idx as scalar_nearest_idx,
  };

  #[cfg(target_arch = "aarch64")]
  pub use crate::nearest::aarch64_neon::nearest_idx as aarch64_neon_nearest_idx;

  // x86 backends are `unsafe fn` (the `#[target_feature]` attribute
  // enforces the safety boundary). Re-export them as-is — the bench
  // wraps the `unsafe { ... }` after a runtime feature check.
  #[cfg(target_arch = "x86_64")]
  pub use crate::nearest::x86_avx2::nearest_idx as x86_avx2_nearest_idx;
  #[cfg(target_arch = "x86_64")]
  pub use crate::nearest::x86_sse41::nearest_idx as x86_sse41_nearest_idx;

  #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
  pub use crate::nearest::wasm_simd128::nearest_idx as wasm_simd128_nearest_idx;

  /// Public re-export of the crate-private `rgb_to_lab` so benches can
  /// pre-convert RGB queries without duplicating the math.
  pub fn rgb_to_lab(rgb: [u8; 3]) -> [f32; 3] {
    super::rgb_to_lab(rgb)
  }
}

/// One named entry in the xkcd color hierarchy.
///
/// Carries every column from the upstream `color_hierarchy.csv`:
/// xkcd / design / common name, hex, and RGB triples for each level,
/// plus the family / kind / neutrality classification. The xkcd LAB
/// triple is pre-computed at codegen time for nearest-neighbor lookup
/// in [`Self::nearest_to`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Color {
  pub(crate) name: &'static str,
  pub(crate) hex: &'static str,
  pub(crate) rgb: [u8; 3],
  pub(crate) lab: [f32; 3],
  pub(crate) design_name: &'static str,
  pub(crate) design_hex: &'static str,
  pub(crate) design_rgb: [u8; 3],
  pub(crate) common_name: &'static str,
  pub(crate) common_hex: &'static str,
  pub(crate) common_rgb: [u8; 3],
  pub(crate) family: Family,
  pub(crate) kind: Kind,
  pub(crate) is_neutral: bool,
}

impl Color {
  /// xkcd-survey name (~950 unique values, e.g. `"burnt orange"`,
  /// `"vermilion"`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn name(&self) -> &'static str {
    self.name
  }

  /// xkcd hex string, e.g. `"#bd6c48"`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn hex(&self) -> &'static str {
    self.hex
  }

  /// xkcd RGB triple, e.g. `[189, 108, 72]`. The exact 8-bit value the
  /// xkcd survey reports for this name.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn rgb(&self) -> [u8; 3] {
    self.rgb
  }

  /// Pre-computed CIE LAB (D65 illuminant, 2° observer) for [`Self::rgb`].
  /// Used internally by [`Self::nearest_to`]; exposed publicly so callers
  /// can implement their own distance metric (e.g. CIEDE2000) on top of
  /// the same cached values.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn lab(&self) -> [f32; 3] {
    self.lab
  }

  /// Coarser design-palette name (~250 unique, e.g. `"russet brown"`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn design_name(&self) -> &'static str {
    self.design_name
  }

  /// Hex string for the design-palette anchor color.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn design_hex(&self) -> &'static str {
    self.design_hex
  }

  /// RGB triple for the design-palette anchor color (the canonical
  /// 8-bit representation of [`Self::design_name`]). Differs from
  /// [`Self::rgb`] when the xkcd entry sits at the edge of its
  /// design-palette bucket.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn design_rgb(&self) -> [u8; 3] {
    self.design_rgb
  }

  /// Coarser still common name (~120 unique, e.g. `"sienna"`). The
  /// search-friendly default for indexing pipelines.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn common_name(&self) -> &'static str {
    self.common_name
  }

  /// Hex string for the common-name anchor color.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn common_hex(&self) -> &'static str {
    self.common_hex
  }

  /// RGB triple for the common-name anchor color.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn common_rgb(&self) -> [u8; 3] {
    self.common_rgb
  }

  /// Color family classification (26 values, e.g. [`Family::Yellow`],
  /// [`Family::BlueGreen`], [`Family::Neutral`]). Call
  /// [`Family::as_str`] for the original CSV string.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn family(&self) -> Family {
    self.family
  }

  /// Color kind / texture classification (11 values, e.g.
  /// [`Kind::NeonColor`], [`Kind::PainterlyNeutral`]). Call
  /// [`Kind::as_str`] for the original CSV string.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn kind(&self) -> Kind {
    self.kind
  }

  /// `true` if the entry is classified as a neutral (vs a chromatic
  /// color). Drives the `color_or_neutral` axis in the original Stitch
  /// Fix taxonomy.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn is_neutral(&self) -> bool {
    self.is_neutral
  }

  /// Every entry in the dataset, in CSV (alphabetical-by-`name`) order.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn all() -> &'static [&'static Color] {
    COLORS
  }

  /// Find the entry whose pre-computed LAB is closest to the given query
  /// RGB by **Delta E 76** (squared Euclidean LAB).
  ///
  /// Always returns an entry — `COLORS` is non-empty and verified at
  /// codegen time. The scan dispatches to a per-arch SIMD backend
  /// (NEON / AVX2 / SSE4.1 / WASM SIMD128) on every target that has
  /// one; other targets fall through to the scalar reference. Every
  /// backend is bit-identical — see [`crate::nearest`] for the
  /// dispatch contract and parity tests.
  ///
  /// # When to use this vs. [`Self::nearest_to_ciede2000`]
  ///
  /// Delta E 76 is the **default** for a reason: against this
  /// crate's well-clustered 949-entry xkcd palette it picks the same
  /// named entry as CIEDE2000 in the overwhelming majority of cases,
  /// runs ~2× over scalar on aarch64 NEON, and has bit-identical
  /// SIMD parity. Reach for [`Self::nearest_to_ciede2000`] only when
  /// you've measured Delta E 76 mis-naming colours on real footage,
  /// typically near the gray / yellow boundary.
  pub fn nearest_to(rgb: [u8; 3]) -> &'static Color {
    crate::nearest::nearest(rgb_to_lab(rgb))
  }

  /// Find the entry whose pre-computed LAB is closest to the given
  /// query RGB by **CIEDE2000** — the modern perceptual gold-standard
  /// colour-difference formula.
  ///
  /// CIEDE2000 corrects Delta E 76's known biases (over-weighting
  /// yellows, under-weighting blues, hue-rotation in the saturated
  /// blue region). The cost: it pulls in `atan2` / `sin` / `cos` /
  /// `exp` / branchy hue-wraparound logic, which combine to be
  /// ~5–10× slower than [`Self::nearest_to`] per query.
  ///
  /// **Scalar-only** by design: see [`crate::nearest::ciede2000`] for
  /// the rationale (transcendentals don't vectorise cleanly on any
  /// 128/256-bit ISA we target). Reach for this when you've measured
  /// drift on real footage and need the accuracy more than the
  /// throughput; otherwise prefer [`Self::nearest_to`].
  pub fn nearest_to_ciede2000(rgb: [u8; 3]) -> &'static Color {
    crate::nearest::nearest_ciede2000(rgb_to_lab(rgb))
  }
}

/// Convert sRGB byte triple → CIE LAB (D65 illuminant, 2° observer).
///
/// Pipeline: byte → normalized [0, 1] → linearized (sRGB EOTF) → XYZ
/// (sRGB→XYZ matrix, D65) → LAB (CIE 1976 transfer).
pub(crate) fn rgb_to_lab(rgb: [u8; 3]) -> [f32; 3] {
  let r = srgb_to_linear(rgb[0] as f32 / 255.0);
  let g = srgb_to_linear(rgb[1] as f32 / 255.0);
  let b = srgb_to_linear(rgb[2] as f32 / 255.0);

  // sRGB → XYZ (D65, 2°). Coefficients from IEC 61966-2-1, rounded to
  // f32 precision (~7 decimal digits — trailing zeros that clippy flagged
  // as excessive precision are dropped here).
  let x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375;
  let y = r * 0.2126729 + g * 0.7151522 + b * 0.072175;
  let z = r * 0.0193339 + g * 0.119192 + b * 0.9503041;

  // D65 reference white (CIE 1931, 2°).
  const XN: f32 = 0.95047;
  const YN: f32 = 1.00000;
  const ZN: f32 = 1.08883;

  let fx = lab_f(x / XN);
  let fy = lab_f(y / YN);
  let fz = lab_f(z / ZN);

  let l = 116.0 * fy - 16.0;
  let a = 500.0 * (fx - fy);
  let b_lab = 200.0 * (fy - fz);
  [l, a, b_lab]
}

/// sRGB electro-optical transfer function (gamma decode).
fn srgb_to_linear(c: f32) -> f32 {
  if c <= 0.04045 {
    c / 12.92
  } else {
    libm::powf((c + 0.055) / 1.055, 2.4)
  }
}

/// CIE LAB transfer function (`f` in the standard).
fn lab_f(t: f32) -> f32 {
  // delta = 6/29; threshold where the linear/cube-root segments meet.
  const DELTA_CUBED: f32 = 216.0 / 24389.0; // (6/29)^3
  const KAPPA_OVER_3: f32 = 841.0 / 108.0; // 1 / (3 * (6/29)^2)
  const OFFSET: f32 = 4.0 / 29.0;
  if t > DELTA_CUBED {
    libm::cbrtf(t)
  } else {
    KAPPA_OVER_3 * t + OFFSET
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  /// The dataset must always be non-empty (xtask validates at codegen
  /// time, but a runtime smoke test catches a regression where someone
  /// edits `generated.rs` by hand).
  #[test]
  fn dataset_is_non_empty() {
    assert!(!COLORS.is_empty());
  }

  /// Snapshot the entry count. Acts as a canary if upstream colornamer
  /// updates the CSV: a count change is a deliberate regen, not a silent
  /// drift. Updating this number is the right action when intentional.
  #[test]
  fn dataset_entry_count_matches_csv() {
    assert_eq!(
      COLORS.len(),
      949,
      "regenerate via `cargo xtask codegen` if the upstream CSV changed",
    );
  }

  /// Pure sRGB red must map to an entry that's at least red-flavored.
  /// Strict equality on the xkcd label is fragile (the closest match
  /// depends on the palette and could be `"red"`, `"bright red"`,
  /// `"true red"`, etc.); the family is a stable invariant.
  #[test]
  fn nearest_to_pure_red_is_in_red_family() {
    let c = Color::nearest_to([255, 0, 0]);
    assert!(
      c.family().as_str().contains("red") || c.name().contains("red"),
      "nearest to pure red was name={:?} family={:?}",
      c.name(),
      c.family().as_str(),
    );
  }

  /// Pure sRGB blue must map to a blue-family entry.
  #[test]
  fn nearest_to_pure_blue_is_in_blue_family() {
    let c = Color::nearest_to([0, 0, 255]);
    assert!(
      c.family().as_str().contains("blue") || c.name().contains("blue"),
      "nearest to pure blue was name={:?} family={:?}",
      c.name(),
      c.family().as_str(),
    );
  }

  /// Mid-gray must map to a neutral. Tests that the `is_neutral` axis
  /// reaches readers correctly via the lookup path.
  #[test]
  fn nearest_to_mid_gray_is_neutral() {
    let c = Color::nearest_to([128, 128, 128]);
    assert!(
      c.is_neutral(),
      "nearest to (128,128,128) was name={:?} is_neutral={}",
      c.name(),
      c.is_neutral(),
    );
  }

  /// sRGB → LAB on the D65 reference white must produce L=100, a=0, b=0
  /// to within float tolerance.
  #[test]
  fn rgb_to_lab_d65_white() {
    let lab = rgb_to_lab([255, 255, 255]);
    assert!((lab[0] - 100.0).abs() < 0.01, "L was {}", lab[0]);
    assert!(lab[1].abs() < 0.01, "a was {}", lab[1]);
    assert!(lab[2].abs() < 0.01, "b was {}", lab[2]);
  }

  /// sRGB → LAB on absolute black must produce L=0, a=0, b=0 exactly
  /// (the linear segment of the sRGB EOTF carries 0 through unchanged).
  #[test]
  fn rgb_to_lab_black() {
    let lab = rgb_to_lab([0, 0, 0]);
    assert!(lab[0].abs() < 1e-6, "L was {}", lab[0]);
    assert!(lab[1].abs() < 1e-6, "a was {}", lab[1]);
    assert!(lab[2].abs() < 1e-6, "b was {}", lab[2]);
  }
}
