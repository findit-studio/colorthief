//! aarch64 NEON CIE94 backend — 4 entries/iter via 128-bit
//! `vld1q_f32` loads against the SoA `LABS_*` arrays.
//!
//! CIE94 has no transcendentals beyond `sqrt`, so unlike CIEDE2000 the
//! formula vectorises cleanly. Per-chunk work is identical in shape
//! to the Delta E 76 NEON backend plus a chroma `vsqrtq_f32` per
//! entry and the chroma-scaled denominator divisions.

#![allow(unsafe_code, dead_code)]

use core::arch::aarch64::*;

use libm::sqrtf;

use super::{LABS_A, LABS_B, LABS_C, LABS_L};

/// CIE94 nearest-neighbor scan.
///
/// Treats every palette entry as the reference (so its C₁ drives the
/// S_C / S_H factors) and the query as the sample — same convention
/// as the scalar [`super::cie94::nearest_idx`].
pub fn nearest_idx(query: [f32; 3]) -> usize {
  // SAFETY: this module is only compiled under
  // `cfg(target_feature = "neon")` (see `super` mod decl), so calling
  // a `#[target_feature(enable = "neon")]` fn is sound.
  // `aarch64-unknown-none-softfloat` falls through to scalar instead
  // of compiling this path.
  unsafe { nearest_idx_neon(query) }
}

#[target_feature(enable = "neon")]
unsafe fn nearest_idx_neon(query: [f32; 3]) -> usize {
  // Broadcast query (the "sample" L, a, b) and pre-compute the query
  // chroma C₂ = sqrt(a₂² + b₂²) once outside the loop. Note that
  // unlike the entry C₁, C₂ doesn't drive any scale factor —
  // S_C / S_H use C₁ — so it only matters for the ΔC term.
  //
  // Plain mul + add (not FMA) to keep the result bit-identical to the
  // scalar `a*a + b*b` — FMA produces a single-rounded result that
  // diverges from the scalar reference at ~5 RGB inputs across the
  // 256³ cube (caught by `tests/parity_exhaustive.rs`).
  let l2 = vdupq_n_f32(query[0]);
  let a2 = vdupq_n_f32(query[1]);
  let b2 = vdupq_n_f32(query[2]);
  let c2_sq_v = vaddq_f32(vmulq_f32(a2, a2), vmulq_f32(b2, b2));
  let c2_v = vsqrtq_f32(c2_sq_v);

  let n = LABS_L.len();
  let chunks = n / 4;

  let mut best_d2 = f32::INFINITY;
  let mut best_idx: usize = 0;

  let l_ptr = LABS_L.as_ptr();
  let a_ptr = LABS_A.as_ptr();
  let b_ptr = LABS_B.as_ptr();
  // Pre-computed reference chroma C₁ — one `vld1q_f32` per chunk
  // replaces `vsqrtq_f32(a₁² + b₁²)`.
  let c_ptr = LABS_C.as_ptr();

  for chunk in 0..chunks {
    let i = chunk * 4;

    // SAFETY: chunks = n / 4 floor; last load is at offset
    // (chunks - 1) * 4, at most n - 4.
    let l1 = unsafe { vld1q_f32(l_ptr.add(i)) };
    let a1 = unsafe { vld1q_f32(a_ptr.add(i)) };
    let b1 = unsafe { vld1q_f32(b_ptr.add(i)) };
    let c1 = unsafe { vld1q_f32(c_ptr.add(i)) };

    // ΔL, Δa, Δb (reference - sample, matching scalar's `lab_ref - lab_sample`).
    let dl = vsubq_f32(l1, l2);
    let da = vsubq_f32(a1, a2);
    let db = vsubq_f32(b1, b2);

    // ΔC = C₁ - C₂. ΔH² = max(Δa² + Δb² - ΔC², 0).
    // Plain mul + add (no FMA) — see the c2_sq comment above.
    let dc = vsubq_f32(c1, c2_v);
    let dab_sq = vaddq_f32(vmulq_f32(da, da), vmulq_f32(db, db));
    let dc_sq = vmulq_f32(dc, dc);
    let dh_sq_raw = vsubq_f32(dab_sq, dc_sq);
    // Clamp ΔH² ≥ 0 to absorb f32 cancellation when (Δa² + Δb²) ≈ ΔC².
    let dh_sq = vmaxq_f32(dh_sq_raw, vdupq_n_f32(0.0));

    // S_C = 1 + 0.045·C₁; S_H = 1 + 0.015·C₁; S_L = 1.
    // Plain mul + add (no FMA) for the same parity reason.
    let one = vdupq_n_f32(1.0);
    let sc = vaddq_f32(one, vmulq_f32(vdupq_n_f32(0.045), c1));
    let sh = vaddq_f32(one, vmulq_f32(vdupq_n_f32(0.015), c1));

    // Terms: dl_term² (sl=1 so just dl²), (dc/sc)², dh_sq/(sh²).
    let dl_sq = vmulq_f32(dl, dl);
    let dc_term = vdivq_f32(dc, sc);
    let dc_term_sq = vmulq_f32(dc_term, dc_term);
    let sh_sq = vmulq_f32(sh, sh);
    let dh_term_sq = vdivq_f32(dh_sq, sh_sq);

    let d2 = vaddq_f32(vaddq_f32(dl_sq, dc_term_sq), dh_term_sq);

    // Min-reduce: spill to a 4-element buffer, scan for the lane.
    let mut buf = [0f32; 4];
    // SAFETY: 16-byte aligned write into a 16-byte buffer.
    unsafe { vst1q_f32(buf.as_mut_ptr(), d2) };
    for (lane, d) in buf.iter().enumerate() {
      if *d < best_d2 {
        best_d2 = *d;
        best_idx = i + lane;
      }
    }
  }

  // Tail: remaining entries past the last full 4-chunk. Use the
  // pre-computed `LABS_C[i]` to keep parity with the SIMD path.
  let c2_scalar = sqrtf(query[1] * query[1] + query[2] * query[2]);
  for i in (chunks * 4)..n {
    let l1 = LABS_L[i];
    let a1 = LABS_A[i];
    let b1 = LABS_B[i];
    let c1 = LABS_C[i];
    let dl = l1 - query[0];
    let da = a1 - query[1];
    let db = b1 - query[2];
    let dc = c1 - c2_scalar;
    let dh_sq = (da * da + db * db - dc * dc).max(0.0);
    let sc = 1.0 + 0.045 * c1;
    let sh = 1.0 + 0.015 * c1;
    let d2 = dl * dl + (dc / sc) * (dc / sc) + dh_sq / (sh * sh);
    if d2 < best_d2 {
      best_d2 = d2;
      best_idx = i;
    }
  }

  best_idx
}
