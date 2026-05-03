//! x86 AVX2 CIE94 backend — 8 entries/iter via 256-bit
//! `_mm256_loadu_ps` against the SoA `LABS_*` arrays. Halves the
//! iteration count vs. the SSE4.1 version.

#![allow(unsafe_code, dead_code)]

use core::arch::x86_64::*;

use libm::sqrtf;

use super::{LABS_A, LABS_B, LABS_L};

/// CIE94 nearest-neighbor scan (AVX2).
///
/// # Safety
///
/// Caller must guarantee AVX2 is available at runtime.
#[target_feature(enable = "avx2")]
pub unsafe fn nearest_idx(query: [f32; 3]) -> usize {
  let l2 = _mm256_set1_ps(query[0]);
  let a2 = _mm256_set1_ps(query[1]);
  let b2 = _mm256_set1_ps(query[2]);
  let c2_sq = _mm256_add_ps(_mm256_mul_ps(a2, a2), _mm256_mul_ps(b2, b2));
  let c2_v = _mm256_sqrt_ps(c2_sq);

  let n = LABS_L.len();
  let chunks = n / 8;

  let mut best_d2 = f32::INFINITY;
  let mut best_idx: usize = 0;

  let l_ptr = LABS_L.as_ptr();
  let a_ptr = LABS_A.as_ptr();
  let b_ptr = LABS_B.as_ptr();

  let zero = _mm256_setzero_ps();
  let one = _mm256_set1_ps(1.0);
  let kc = _mm256_set1_ps(0.045);
  let kh = _mm256_set1_ps(0.015);

  for chunk in 0..chunks {
    let i = chunk * 8;

    // SAFETY: chunks = n / 8 floor.
    let l1 = unsafe { _mm256_loadu_ps(l_ptr.add(i)) };
    let a1 = unsafe { _mm256_loadu_ps(a_ptr.add(i)) };
    let b1 = unsafe { _mm256_loadu_ps(b_ptr.add(i)) };

    let dl = _mm256_sub_ps(l1, l2);
    let da = _mm256_sub_ps(a1, a2);
    let db = _mm256_sub_ps(b1, b2);

    let c1_sq = _mm256_add_ps(_mm256_mul_ps(a1, a1), _mm256_mul_ps(b1, b1));
    let c1 = _mm256_sqrt_ps(c1_sq);

    let dc = _mm256_sub_ps(c1, c2_v);
    let dab_sq = _mm256_add_ps(_mm256_mul_ps(da, da), _mm256_mul_ps(db, db));
    let dc_sq = _mm256_mul_ps(dc, dc);
    let dh_sq = _mm256_max_ps(_mm256_sub_ps(dab_sq, dc_sq), zero);

    let sc = _mm256_add_ps(one, _mm256_mul_ps(kc, c1));
    let sh = _mm256_add_ps(one, _mm256_mul_ps(kh, c1));

    let dl_sq = _mm256_mul_ps(dl, dl);
    let dc_term = _mm256_div_ps(dc, sc);
    let dc_term_sq = _mm256_mul_ps(dc_term, dc_term);
    let sh_sq = _mm256_mul_ps(sh, sh);
    let dh_term_sq = _mm256_div_ps(dh_sq, sh_sq);

    let d2 = _mm256_add_ps(_mm256_add_ps(dl_sq, dc_term_sq), dh_term_sq);

    let mut buf = [0f32; 8];
    // SAFETY: 32-byte write into a 32-byte buffer.
    unsafe { _mm256_storeu_ps(buf.as_mut_ptr(), d2) };
    for (lane, d) in buf.iter().enumerate() {
      if *d < best_d2 {
        best_d2 = *d;
        best_idx = i + lane;
      }
    }
  }

  // Tail (n % 8 — currently 5 entries for the 949-entry palette).
  for i in (chunks * 8)..n {
    let l1 = LABS_L[i];
    let a1 = LABS_A[i];
    let b1 = LABS_B[i];
    let dl = l1 - query[0];
    let da = a1 - query[1];
    let db = b1 - query[2];
    let c1 = sqrtf(a1 * a1 + b1 * b1);
    let c2 = sqrtf(query[1] * query[1] + query[2] * query[2]);
    let dc = c1 - c2;
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
