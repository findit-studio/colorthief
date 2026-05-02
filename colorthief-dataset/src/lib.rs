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
#![forbid(unsafe_code)]

mod generated;

pub use generated::COLORS;

/// One named entry in the xkcd color hierarchy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Color {
  pub(crate) name: &'static str,
  pub(crate) rgb: [u8; 3],
  pub(crate) lab: [f32; 3],
  pub(crate) design_name: &'static str,
  pub(crate) common_name: &'static str,
  pub(crate) family: &'static str,
  pub(crate) kind: &'static str,
  pub(crate) is_neutral: bool,
}

impl Color {
  /// xkcd-survey name (~950 unique values, e.g. `"burnt orange"`,
  /// `"vermilion"`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn name(&self) -> &'static str {
    self.name
  }

  /// xkcd RGB triple, e.g. `[189, 108, 72]`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn rgb(&self) -> [u8; 3] {
    self.rgb
  }

  /// Pre-computed CIE LAB (D65 illuminant, 2° observer) for the entry's
  /// RGB. Used internally by [`Self::nearest_to`]; exposed publicly so
  /// callers can implement their own distance metric (e.g. CIEDE2000) on
  /// top of the same cached values.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn lab(&self) -> [f32; 3] {
    self.lab
  }

  /// Coarser design-palette name (~250 unique, e.g. `"russet brown"`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn design_name(&self) -> &'static str {
    self.design_name
  }

  /// Coarser still common name (~120 unique, e.g. `"sienna"`). The
  /// search-friendly default for indexing pipelines.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn common_name(&self) -> &'static str {
    self.common_name
  }

  /// Color family (26 values, e.g. `"blue green"`, `"neutral"`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn family(&self) -> &'static str {
    self.family
  }

  /// Color type (11 values, e.g. `"neon color"`, `"painterly neutral"`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn kind(&self) -> &'static str {
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
  /// RGB (Delta E 76).
  ///
  /// Always returns an entry — `COLORS` is non-empty and verified at
  /// codegen time.
  pub fn nearest_to(rgb: [u8; 3]) -> &'static Color {
    let query = rgb_to_lab(rgb);
    let (mut best, rest) = COLORS
      .split_first()
      .expect("colorthief-dataset must have at least one entry");
    let mut best_d2 = lab_distance_sq(query, best.lab);
    for entry in rest {
      let d2 = lab_distance_sq(query, entry.lab);
      if d2 < best_d2 {
        best = entry;
        best_d2 = d2;
      }
    }
    best
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

  // sRGB → XYZ (D65, 2°). Coefficients from IEC 61966-2-1.
  let x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375;
  let y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750;
  let z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041;

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

/// Squared Euclidean distance in LAB. Squared, not square-rooted —
/// preserves ordering and saves a `sqrt` per query.
fn lab_distance_sq(a: [f32; 3], b: [f32; 3]) -> f32 {
  let dl = a[0] - b[0];
  let da = a[1] - b[1];
  let db = a[2] - b[2];
  dl * dl + da * da + db * db
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
