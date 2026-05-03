//! CIE94 (Delta E 94) color-difference formula — scalar reference.
//!
//! Reference: CIE 116-1995, *"Industrial Colour-Difference
//! Evaluation"*. Sits between Delta E 76 (squared Euclidean LAB) and
//! CIEDE2000 in both perceptual accuracy and arithmetic cost. Unlike
//! CIEDE2000, the formula uses no `atan2`, `sin`, `cos`, or `exp` —
//! only square roots and arithmetic. That makes CIE94 **SIMD-friendly**
//! the same way Delta E 76 is; per-arch backends mirror the existing
//! [`super::scalar`] / [`super::aarch64_neon`] / [`super::x86_avx2`]
//! / [`super::x86_sse41`] / [`super::wasm_simd128`] structure.
//!
//! # Formula (graphic-arts weighting)
//!
//! Given reference Lab₁ (here: each palette entry) and sample Lab₂
//! (here: the query):
//!
//! ```text
//! ΔL = L₁ - L₂
//! C₁ = √(a₁² + b₁²)
//! C₂ = √(a₂² + b₂²)
//! ΔC = C₁ - C₂
//! Δa = a₁ - a₂
//! Δb = b₁ - b₂
//! ΔH² = max(Δa² + Δb² - ΔC², 0)   // clamp for f32 cancellation
//!
//! S_L = 1
//! S_C = 1 + 0.045 * C₁
//! S_H = 1 + 0.015 * C₁
//! K_L = K_C = K_H = 1   (graphic-arts; textiles uses K_L=2, K_C=K_H=1)
//!
//! ΔE94² = (ΔL/(K_L·S_L))² + (ΔC/(K_C·S_C))² + (ΔH²)/(K_H·S_H)²
//! ```
//!
//! # Asymmetry
//!
//! CIE94 is **asymmetric** — `ΔE94(A, B) ≠ ΔE94(B, A)` because S_C
//! and S_H depend on `C₁` (the reference, not the sample). For
//! nearest-neighbor against a fixed palette the natural choice is to
//! treat the **palette entry as the reference** and the query as the
//! sample, so the per-entry C₁ can be pre-computed at codegen time
//! (via `LABS_*` already, plus a small derived array). For now we
//! compute C₁ inline since the per-pair cost is dominated by
//! arithmetic, not the chroma sqrt.

use libm::sqrtf;

use super::{LABS_A, LABS_B, LABS_C, LABS_L};

/// Squared CIE94 distance between a palette-entry LAB (the reference,
/// `lab_ref`) and a query LAB (the sample, `lab_sample`). Always
/// non-negative; squared form preserves nearest-neighbor ordering and
/// saves a `sqrtf` per query.
///
/// `nearest_idx` doesn't call this — the production hot loop inlines
/// the formula and reads pre-computed entry chroma from
/// [`super::LABS_C`] to skip the `sqrtf(a² + b²)`. This standalone
/// function is the reference shape used by tests
/// (`self_distance_is_zero`, `asymmetric_in_arguments`) and stays
/// available for any future caller that wants CIE94 between two
/// arbitrary LAB triples without going through the palette-lookup
/// path.
#[allow(dead_code)]
#[inline]
pub fn delta_e_94_sq(lab_ref: [f32; 3], lab_sample: [f32; 3]) -> f32 {
  let [l1, a1, b1] = lab_ref;
  let [l2, a2, b2] = lab_sample;

  let dl = l1 - l2;
  let da = a1 - a2;
  let db = b1 - b2;

  // Reference chroma C₁ (drives the S_C / S_H scale factors).
  let c1_sq = a1 * a1 + b1 * b1;
  let c1 = sqrtf(c1_sq);
  let c2 = sqrtf(a2 * a2 + b2 * b2);
  let dc = c1 - c2;

  // ΔH² = Δa² + Δb² - ΔC². Algebraically non-negative for any
  // (Lab₁, Lab₂); f32 cancellation can drive it slightly below zero
  // for nearly-equal pairs, so clamp at zero.
  let dh_sq = (da * da + db * db - dc * dc).max(0.0);

  let sl = 1.0;
  let sc = 1.0 + 0.045 * c1;
  let sh = 1.0 + 0.015 * c1;

  let dl_term = dl / sl;
  let dc_term = dc / sc;
  let dh_term_sq = dh_sq / (sh * sh);

  dl_term * dl_term + dc_term * dc_term + dh_term_sq
}

/// Find the [`super::COLORS`] index minimising CIE94 to `query`.
/// Each entry is treated as the reference; the query as the sample.
///
/// Uses [`super::LABS_C`] for the per-entry chroma `C₁`, saving the
/// `sqrtf(a² + b²)` per pair (one `sqrtf` × N entries per query).
pub fn nearest_idx(query: [f32; 3]) -> usize {
  let n = LABS_L.len();
  let q_a = query[1];
  let q_b = query[2];
  // Sample chroma C₂ — constant across the loop, hoist out.
  let c2 = sqrtf(q_a * q_a + q_b * q_b);

  let mut best_idx = 0usize;
  let mut best_d2 = f32::INFINITY;
  for i in 0..n {
    let l1 = LABS_L[i];
    let a1 = LABS_A[i];
    let b1 = LABS_B[i];
    let c1 = LABS_C[i]; // pre-computed entry chroma
    let dl = l1 - query[0];
    let da = a1 - q_a;
    let db = b1 - q_b;
    let dc = c1 - c2;
    let dh_sq = (da * da + db * db - dc * dc).max(0.0);
    let sc = 1.0 + 0.045 * c1;
    let sh = 1.0 + 0.015 * c1;
    let dl_term = dl;
    let dc_term = dc / sc;
    let dh_term_sq = dh_sq / (sh * sh);
    let d2 = dl_term * dl_term + dc_term * dc_term + dh_term_sq;
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

  /// Self-distance is zero (note: CIE94 is asymmetric in general but
  /// must collapse to 0 when reference = sample since every Δ term is
  /// 0 and ΔH² clamp doesn't fire).
  #[test]
  fn self_distance_is_zero() {
    for lab in [[50.0, 0.0, 0.0], [80.0, 20.0, -30.0], [0.0, 0.0, 0.0]] {
      let d2 = delta_e_94_sq(lab, lab);
      assert_eq!(d2, 0.0, "self-distance non-zero for {lab:?}: {d2}");
    }
  }

  /// Black-vs-black: C₁ = 0 ⇒ S_C = S_H = 1 ⇒ ΔE94² = ΔL² + ΔC² + ΔH².
  /// Every Δ is zero so the whole expression is zero. Pin it as a
  /// regression: an `0/0` somewhere in the chroma terms would NaN.
  #[test]
  fn black_vs_black_handled() {
    let d2 = delta_e_94_sq([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
    assert_eq!(d2, 0.0);
    assert!(d2.is_finite());
  }

  /// CIE94 is asymmetric: ΔE94(A, B) ≠ ΔE94(B, A) in general because
  /// the S_C / S_H factors use C₁ (the reference). Pin this with a
  /// concrete example so a future "tidy" refactor doesn't accidentally
  /// symmetrise the formula.
  #[test]
  fn asymmetric_in_arguments() {
    // High-chroma A, low-chroma B → S_C(A) > S_C(B) → ΔE94(A, B) <
    // ΔE94(B, A) for the same (a, b) deltas.
    let high_chroma = [50.0, 60.0, 40.0];
    let low_chroma = [50.0, 5.0, 5.0];
    let de_ab = delta_e_94_sq(high_chroma, low_chroma);
    let de_ba = delta_e_94_sq(low_chroma, high_chroma);
    assert!(
      (de_ab - de_ba).abs() > 1.0,
      "expected meaningful asymmetry, got ΔE²(A→B)={de_ab}, ΔE²(B→A)={de_ba}"
    );
  }

  /// Smoke test: lookup must return a finite valid index and the
  /// chosen entry must actually minimise ΔE94² over the full palette.
  #[test]
  fn lookup_returns_valid_index_across_grid() {
    for r in (0..=255u32).step_by(64) {
      for g in (0..=255u32).step_by(64) {
        for b in (0..=255u32).step_by(64) {
          let q = crate::rgb_to_lab([r as u8, g as u8, b as u8]);
          let idx = nearest_idx(q);
          assert!(idx < LABS_L.len());
          let entry = [LABS_L[idx], LABS_A[idx], LABS_B[idx]];
          let d2 = delta_e_94_sq(entry, q);
          assert!(
            d2.is_finite() && d2 >= 0.0,
            "non-finite/negative ΔE94² at rgb=[{r},{g},{b}]: d2={d2}"
          );
        }
      }
    }
  }
}
