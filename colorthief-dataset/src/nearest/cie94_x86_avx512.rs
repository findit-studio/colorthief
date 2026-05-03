//! x86 AVX-512F CIE94 backend — 16 entries/iter via 512-bit
//! `_mm512_loadu_ps` against the SoA `LABS_*` arrays.
//!
//! Requires Rust 1.89+ for stable `_mm512_*` intrinsics.

#![allow(unsafe_code, dead_code)]

use core::arch::x86_64::*;

use libm::sqrtf;

use super::{LABS_A, LABS_B, LABS_C, LABS_L};

/// CIE94 nearest-neighbor scan (AVX-512F).
///
/// # Safety
///
/// Caller must guarantee AVX-512F is available at runtime.
#[target_feature(enable = "avx512f")]
pub unsafe fn nearest_idx(query: [f32; 3]) -> usize {
  let l2 = _mm512_set1_ps(query[0]);
  let a2 = _mm512_set1_ps(query[1]);
  let b2 = _mm512_set1_ps(query[2]);
  let c2_sq = _mm512_add_ps(_mm512_mul_ps(a2, a2), _mm512_mul_ps(b2, b2));
  let c2_v = _mm512_sqrt_ps(c2_sq);

  let n = LABS_L.len();
  let chunks = n / 16;

  let mut best_d2 = f32::INFINITY;
  let mut best_idx: usize = 0;

  let l_ptr = LABS_L.as_ptr();
  let a_ptr = LABS_A.as_ptr();
  let b_ptr = LABS_B.as_ptr();
  let c_ptr = LABS_C.as_ptr();

  let zero = _mm512_setzero_ps();
  let one = _mm512_set1_ps(1.0);
  let kc = _mm512_set1_ps(0.045);
  let kh = _mm512_set1_ps(0.015);

  for chunk in 0..chunks {
    let i = chunk * 16;

    // SAFETY: chunks = n / 16 floor.
    let l1 = unsafe { _mm512_loadu_ps(l_ptr.add(i)) };
    let a1 = unsafe { _mm512_loadu_ps(a_ptr.add(i)) };
    let b1 = unsafe { _mm512_loadu_ps(b_ptr.add(i)) };
    let c1 = unsafe { _mm512_loadu_ps(c_ptr.add(i)) };

    let dl = _mm512_sub_ps(l1, l2);
    let da = _mm512_sub_ps(a1, a2);
    let db = _mm512_sub_ps(b1, b2);

    let dc = _mm512_sub_ps(c1, c2_v);
    let dab_sq = _mm512_add_ps(_mm512_mul_ps(da, da), _mm512_mul_ps(db, db));
    let dc_sq = _mm512_mul_ps(dc, dc);
    let dh_sq = _mm512_max_ps(_mm512_sub_ps(dab_sq, dc_sq), zero);

    let sc = _mm512_add_ps(one, _mm512_mul_ps(kc, c1));
    let sh = _mm512_add_ps(one, _mm512_mul_ps(kh, c1));

    let dl_sq = _mm512_mul_ps(dl, dl);
    let dc_term = _mm512_div_ps(dc, sc);
    let dc_term_sq = _mm512_mul_ps(dc_term, dc_term);
    let sh_sq = _mm512_mul_ps(sh, sh);
    let dh_term_sq = _mm512_div_ps(dh_sq, sh_sq);

    let d2 = _mm512_add_ps(_mm512_add_ps(dl_sq, dc_term_sq), dh_term_sq);

    let mut buf = [0f32; 16];
    // SAFETY: 64-byte write into a 64-byte buffer.
    unsafe { _mm512_storeu_ps(buf.as_mut_ptr(), d2) };
    for (lane, d) in buf.iter().enumerate() {
      if *d < best_d2 {
        best_d2 = *d;
        best_idx = i + lane;
      }
    }
  }

  // Tail.
  let c2_scalar = sqrtf(query[1] * query[1] + query[2] * query[2]);
  for i in (chunks * 16)..n {
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
