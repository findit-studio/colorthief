//! WASM SIMD128 CIE94 backend — 4 entries/iter via 128-bit
//! `v128_load` against the SoA `LABS_*` arrays.

#![allow(unsafe_code, dead_code)]

use core::arch::wasm32::*;

use libm::sqrtf;

use super::{LABS_A, LABS_B, LABS_L};

/// CIE94 nearest-neighbor scan (WASM SIMD128).
pub fn nearest_idx(query: [f32; 3]) -> usize {
  // SAFETY: SIMD128 statically guaranteed by the cfg gate on the
  // module's declaration in `super`.
  unsafe { nearest_idx_simd128(query) }
}

#[target_feature(enable = "simd128")]
unsafe fn nearest_idx_simd128(query: [f32; 3]) -> usize {
  let l2 = f32x4_splat(query[0]);
  let a2 = f32x4_splat(query[1]);
  let b2 = f32x4_splat(query[2]);
  let c2_sq = f32x4_add(f32x4_mul(a2, a2), f32x4_mul(b2, b2));
  let c2_v = f32x4_sqrt(c2_sq);

  let n = LABS_L.len();
  let chunks = n / 4;

  let mut best_d2 = f32::INFINITY;
  let mut best_idx: usize = 0;

  let l_ptr = LABS_L.as_ptr();
  let a_ptr = LABS_A.as_ptr();
  let b_ptr = LABS_B.as_ptr();

  let zero = f32x4_splat(0.0);
  let one = f32x4_splat(1.0);
  let kc = f32x4_splat(0.045);
  let kh = f32x4_splat(0.015);

  for chunk in 0..chunks {
    let i = chunk * 4;

    // SAFETY: chunks = n / 4 floor.
    let l1 = unsafe { v128_load(l_ptr.add(i) as *const v128) };
    let a1 = unsafe { v128_load(a_ptr.add(i) as *const v128) };
    let b1 = unsafe { v128_load(b_ptr.add(i) as *const v128) };

    let dl = f32x4_sub(l1, l2);
    let da = f32x4_sub(a1, a2);
    let db = f32x4_sub(b1, b2);

    let c1_sq = f32x4_add(f32x4_mul(a1, a1), f32x4_mul(b1, b1));
    let c1 = f32x4_sqrt(c1_sq);

    let dc = f32x4_sub(c1, c2_v);
    let dab_sq = f32x4_add(f32x4_mul(da, da), f32x4_mul(db, db));
    let dc_sq = f32x4_mul(dc, dc);
    let dh_sq = f32x4_max(f32x4_sub(dab_sq, dc_sq), zero);

    let sc = f32x4_add(one, f32x4_mul(kc, c1));
    let sh = f32x4_add(one, f32x4_mul(kh, c1));

    let dl_sq = f32x4_mul(dl, dl);
    let dc_term = f32x4_div(dc, sc);
    let dc_term_sq = f32x4_mul(dc_term, dc_term);
    let sh_sq = f32x4_mul(sh, sh);
    let dh_term_sq = f32x4_div(dh_sq, sh_sq);

    let d2 = f32x4_add(f32x4_add(dl_sq, dc_term_sq), dh_term_sq);

    let mut buf = [0f32; 4];
    // SAFETY: 16-byte write into a 16-byte buffer.
    unsafe { v128_store(buf.as_mut_ptr() as *mut v128, d2) };
    for (lane, d) in buf.iter().enumerate() {
      if *d < best_d2 {
        best_d2 = *d;
        best_idx = i + lane;
      }
    }
  }

  // Tail.
  for i in (chunks * 4)..n {
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
