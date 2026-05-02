//! x86 AVX2 backend — 8 entries/iter via 256-bit `_mm256_loadu_ps`
//! loads against the SoA `LABS_*` arrays.
//!
//! Compile-time gated to `target_arch = "x86_64"`; runtime-gated by
//! the dispatcher via `is_x86_feature_detected!("avx2")`. 8-lane width
//! halves the iteration count vs SSE4.1; the same plain-mul-and-add
//! discipline keeps the result bit-identical to scalar.

#![allow(unsafe_code)]

use core::arch::x86_64::*;

use super::{LABS_A, LABS_B, LABS_L};

/// AVX2 nearest-neighbor scan.
///
/// # Safety
///
/// Caller must guarantee that the AVX2 instruction set is available
/// at runtime; the dispatcher in [`super::nearest_idx`] verifies this
/// via [`std::is_x86_feature_detected!`].
#[target_feature(enable = "avx2")]
pub unsafe fn nearest_idx(query: [f32; 3]) -> usize {
  let ql = _mm256_set1_ps(query[0]);
  let qa = _mm256_set1_ps(query[1]);
  let qb = _mm256_set1_ps(query[2]);

  let n = LABS_L.len();
  let chunks = n / 8;

  let mut best_d2 = f32::INFINITY;
  let mut best_idx: usize = 0;

  let l_ptr = LABS_L.as_ptr();
  let a_ptr = LABS_A.as_ptr();
  let b_ptr = LABS_B.as_ptr();

  for chunk in 0..chunks {
    let i = chunk * 8;

    // SAFETY: `i + 8 <= n`; chunks = n / 8 floor.
    let l = unsafe { _mm256_loadu_ps(l_ptr.add(i)) };
    let a = unsafe { _mm256_loadu_ps(a_ptr.add(i)) };
    let b = unsafe { _mm256_loadu_ps(b_ptr.add(i)) };

    let dl = _mm256_sub_ps(ql, l);
    let da = _mm256_sub_ps(qa, a);
    let db = _mm256_sub_ps(qb, b);

    // Plain mul + add — no FMA. AVX2 has FMA via the FMA3 ISA, but
    // mixing it in would diverge from the scalar baseline by 1 ulp.
    let dl_sq = _mm256_mul_ps(dl, dl);
    let da_sq = _mm256_mul_ps(da, da);
    let db_sq = _mm256_mul_ps(db, db);
    let partial = _mm256_add_ps(dl_sq, da_sq);
    let d2 = _mm256_add_ps(partial, db_sq);

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

  // Tail: any leftover entries (n % 8 — currently 5 entries).
  for i in (chunks * 8)..n {
    let dl = query[0] - LABS_L[i];
    let da = query[1] - LABS_A[i];
    let db = query[2] - LABS_B[i];
    let d2 = (dl * dl + da * da) + db * db;
    if d2 < best_d2 {
      best_d2 = d2;
      best_idx = i;
    }
  }

  best_idx
}
