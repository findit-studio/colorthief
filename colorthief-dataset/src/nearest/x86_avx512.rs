//! x86 AVX-512F Delta E 76 backend — 16 entries/iter via 512-bit
//! `_mm512_loadu_ps` against the SoA `LABS_*` arrays. Halves the
//! iteration count vs. the AVX2 version (8 → 16 lanes).
//!
//! Requires Rust 1.89+ for stable `_mm512_*` intrinsics; the
//! workspace MSRV is 1.95, so this is always available.

#![allow(unsafe_code, dead_code)]

use core::arch::x86_64::*;

use super::{LABS_A, LABS_B, LABS_L};

/// Delta E 76 nearest-neighbor scan (AVX-512F).
///
/// # Safety
///
/// Caller must guarantee AVX-512F is available at runtime.
#[target_feature(enable = "avx512f")]
pub unsafe fn nearest_idx(query: [f32; 3]) -> usize {
  let ql = _mm512_set1_ps(query[0]);
  let qa = _mm512_set1_ps(query[1]);
  let qb = _mm512_set1_ps(query[2]);

  let n = LABS_L.len();
  let chunks = n / 16;

  let mut best_d2 = f32::INFINITY;
  let mut best_idx: usize = 0;

  let l_ptr = LABS_L.as_ptr();
  let a_ptr = LABS_A.as_ptr();
  let b_ptr = LABS_B.as_ptr();

  for chunk in 0..chunks {
    let i = chunk * 16;

    // SAFETY: chunks = n / 16 floor.
    let l = unsafe { _mm512_loadu_ps(l_ptr.add(i)) };
    let a = unsafe { _mm512_loadu_ps(a_ptr.add(i)) };
    let b = unsafe { _mm512_loadu_ps(b_ptr.add(i)) };

    let dl = _mm512_sub_ps(ql, l);
    let da = _mm512_sub_ps(qa, a);
    let db = _mm512_sub_ps(qb, b);

    // Plain mul + add — no FMA — to match the scalar baseline.
    let dl_sq = _mm512_mul_ps(dl, dl);
    let da_sq = _mm512_mul_ps(da, da);
    let db_sq = _mm512_mul_ps(db, db);
    let partial = _mm512_add_ps(dl_sq, da_sq);
    let d2 = _mm512_add_ps(partial, db_sq);

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

  // Tail: any leftover entries (n % 16 — currently 5 for the
  // 949-entry palette: 949 = 59*16 + 5).
  for i in (chunks * 16)..n {
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
