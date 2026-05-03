//! CIEDE2000 candidate-set LUT lookup.
//!
//! When the `lut` feature is enabled, [`crate::Color::nearest_to_ciede2000`]
//! and [`crate::Color::nearest_to_ciede2000_exact`] dispatch through this
//! module instead of the full-scan reference in [`super::ciede2000`].
//!
//! # Algorithm
//!
//! 1. Quantize the query RGB to a 32³ cell index by dropping the bottom
//!    3 bits per channel: `cell = ((r >> 3) << 10) | ((g >> 3) << 5) |
//!    (b >> 3)`. Cell range: `[0, 32768)`.
//! 2. Look up the cell's candidate slice in the CSR-encoded LUT
//!    (`LUT_CELL_OFFSETS[cell..=cell + 1]` ranges into
//!    `LUT_CELL_INDICES`).
//! 3. Scan the candidates with the exact CIEDE2000 metric, return the
//!    minimiser.
//!
//! # Provable exactness at u8 RGB resolution
//!
//! At codegen time (`cargo xtask codegen`), the LUT is built by
//! sampling **every** RGB inside each cell's `8×8×8` box and unioning
//! the CIEDE2000 full-scan winners. Since u8 RGB inputs are exactly
//! the codegen-time samples (every `[u8; 3]` falls into exactly one
//! cell, and every value in the cell was visited), the cell's
//! candidate set is **guaranteed** to contain the true CIEDE2000
//! nearest for any u8 RGB query. The runtime small-set scan then
//! picks the right one.
//!
//! Empirical stats from the current palette (949 entries, 32,768
//! cells): 81,875 total candidates, avg 2.50 per cell, max 10 per
//! cell. Per query the candidate scan does at most 10 CIEDE2000
//! evaluations vs. 949 for the full-scan baseline — ~95× fewer
//! transcendental-heavy calls.

use super::{LABS_A, LABS_B, LABS_L, ciede2000::delta_e_2000_sq};
use crate::generated::{LUT_CELL_INDICES, LUT_CELL_OFFSETS};

/// LUT-routed CIEDE2000 nearest-neighbor.
///
/// `rgb` drives the cell lookup; `query` is its pre-converted LAB
/// (the caller computes it once via `crate::rgb_to_lab` before
/// dispatch — passing both avoids re-converting).
pub fn nearest_idx(rgb: [u8; 3], query: [f32; 3]) -> usize {
  let cell =
    (((rgb[0] >> 3) as usize) << 10) | (((rgb[1] >> 3) as usize) << 5) | ((rgb[2] >> 3) as usize);

  let start = LUT_CELL_OFFSETS[cell] as usize;
  let end = LUT_CELL_OFFSETS[cell + 1] as usize;

  // The LUT is built so every cell has at least one candidate. The
  // first candidate seeds the running minimum; the rest of the slice
  // is scanned linearly.
  let first = LUT_CELL_INDICES[start] as usize;
  let mut best_idx = first;
  let mut best_d2 = delta_e_2000_sq([LABS_L[first], LABS_A[first], LABS_B[first]], query);

  for &idx in &LUT_CELL_INDICES[start + 1..end] {
    let i = idx as usize;
    let d2 = delta_e_2000_sq([LABS_L[i], LABS_A[i], LABS_B[i]], query);
    if d2 < best_d2 {
      best_d2 = d2;
      best_idx = i;
    }
  }
  best_idx
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::generated::COLORS;

  /// Smoke test: returns a finite valid index for a small RGB grid.
  #[test]
  fn lookup_returns_valid_index_across_grid() {
    for r in (0..=255u32).step_by(64) {
      for g in (0..=255u32).step_by(64) {
        for b in (0..=255u32).step_by(64) {
          let rgb = [r as u8, g as u8, b as u8];
          let q = crate::rgb_to_lab(rgb);
          let idx = nearest_idx(rgb, q);
          assert!(
            idx < COLORS.len(),
            "rgb={rgb:?} idx={idx} >= {}",
            COLORS.len()
          );
        }
      }
    }
  }

  /// Querying with a palette entry's own RGB returns that entry (or
  /// one with bit-equal LAB if the palette has duplicates). Catches
  /// regressions where the cell containing an entry's RGB wouldn't
  /// list that entry as a candidate.
  #[test]
  fn idempotent_for_palette_rgb() {
    for entry in COLORS.iter() {
      let rgb = entry.rgb();
      let q = crate::rgb_to_lab(rgb);
      let idx = nearest_idx(rgb, q);
      let returned = COLORS[idx];
      assert_eq!(
        returned.lab(),
        entry.lab(),
        "entry {:?} (rgb={:?}) returned {:?} via LUT",
        entry.name(),
        rgb,
        returned.name(),
      );
    }
  }
}
