//! x86 SSE4.1 backend — 4 entries/iter via 128-bit `_mm_loadu_ps`
//! loads against the SoA `LABS_*` arrays.
//!
//! Compile-time gated to `target_arch = "x86_64"`; runtime-gated by
//! the dispatcher via `is_x86_feature_detected!("sse4.1")`. Mirrors
//! the aarch64 NEON shape exactly (4-lane width, plain mul+add, scalar
//! reduction over a 4-element spill buffer) so the bit-parity test
//! applies the same expectation.

#![allow(unsafe_code)]

use core::arch::x86_64::*;

use super::{LABS_A, LABS_B, LABS_L};

/// SSE4.1 nearest-neighbor scan.
///
/// # Safety
///
/// Caller must guarantee that the SSE4.1 instruction set is available
/// at runtime; the dispatcher in [`super::nearest_idx`] verifies this
/// via [`std::is_x86_feature_detected!`].
#[target_feature(enable = "sse4.1")]
pub unsafe fn nearest_idx(query: [f32; 3]) -> usize {
  let ql = _mm_set1_ps(query[0]);
  let qa = _mm_set1_ps(query[1]);
  let qb = _mm_set1_ps(query[2]);

  let n = LABS_L.len();
  let chunks = n / 4;

  let mut best_d2 = f32::INFINITY;
  let mut best_idx: usize = 0;

  let l_ptr = LABS_L.as_ptr();
  let a_ptr = LABS_A.as_ptr();
  let b_ptr = LABS_B.as_ptr();

  for chunk in 0..chunks {
    let i = chunk * 4;

    // SAFETY: `i + 4 <= n`; chunks = n / 4 floor. Unaligned loads —
    // SoA arrays are `f32`-aligned, not `[f32; 4]`-aligned, so use
    // `loadu` rather than `load`. Cost is identical on Sandy Bridge+.
    let l = unsafe { _mm_loadu_ps(l_ptr.add(i)) };
    let a = unsafe { _mm_loadu_ps(a_ptr.add(i)) };
    let b = unsafe { _mm_loadu_ps(b_ptr.add(i)) };

    let dl = _mm_sub_ps(ql, l);
    let da = _mm_sub_ps(qa, a);
    let db = _mm_sub_ps(qb, b);

    // Plain mul + add — no FMA — to match the scalar baseline.
    let dl_sq = _mm_mul_ps(dl, dl);
    let da_sq = _mm_mul_ps(da, da);
    let db_sq = _mm_mul_ps(db, db);
    let partial = _mm_add_ps(dl_sq, da_sq);
    let d2 = _mm_add_ps(partial, db_sq);

    let mut buf = [0f32; 4];
    // SAFETY: 16-byte write into a 16-byte buffer (loadu equivalent
    // for store; alignment-tolerant).
    unsafe { _mm_storeu_ps(buf.as_mut_ptr(), d2) };
    for (lane, d) in buf.iter().enumerate() {
      if *d < best_d2 {
        best_d2 = *d;
        best_idx = i + lane;
      }
    }
  }

  // Tail: any leftover entries (n % 4 — currently 1 entry).
  for i in (chunks * 4)..n {
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
