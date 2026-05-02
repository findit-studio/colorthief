//! Aarch64 NEON backend — 4 entries/iter via 128-bit `vld1q_f32`
//! loads against the SoA `LABS_*` arrays.
//!
//! Compile-time gated to `target_arch = "aarch64"`. NEON is mandatory
//! in Armv8-A so no runtime feature detection is needed.

#![allow(unsafe_code)]

use core::arch::aarch64::*;

use super::{LABS_A, LABS_B, LABS_L};

/// NEON entry. Static target gating means NEON is always available
/// here. The `unsafe` block delegates to the
/// `#[target_feature(enable = "neon")]` worker; the safety
/// requirement (NEON available at runtime) is satisfied by the
/// architectural mandate.
pub fn nearest_idx(query: [f32; 3]) -> usize {
  // SAFETY: `target_arch = "aarch64"` implies Armv8-A, where NEON is
  // mandatory; there is no aarch64 build target without NEON.
  unsafe { nearest_idx_neon(query) }
}

#[target_feature(enable = "neon")]
unsafe fn nearest_idx_neon(query: [f32; 3]) -> usize {
  // Broadcast each query channel across all 4 lanes.
  let ql = vdupq_n_f32(query[0]);
  let qa = vdupq_n_f32(query[1]);
  let qb = vdupq_n_f32(query[2]);

  let n = LABS_L.len();
  let chunks = n / 4;

  let mut best_d2 = f32::INFINITY;
  let mut best_idx: usize = 0;

  let l_ptr = LABS_L.as_ptr();
  let a_ptr = LABS_A.as_ptr();
  let b_ptr = LABS_B.as_ptr();

  for chunk in 0..chunks {
    let i = chunk * 4;

    // SAFETY: `i + 4 <= n`; chunks = n / 4 floor, so the last load is
    // at offset `(chunks - 1) * 4`, at most `n - 4`.
    let l = unsafe { vld1q_f32(l_ptr.add(i)) };
    let a = unsafe { vld1q_f32(a_ptr.add(i)) };
    let b = unsafe { vld1q_f32(b_ptr.add(i)) };

    let dl = vsubq_f32(ql, l);
    let da = vsubq_f32(qa, a);
    let db = vsubq_f32(qb, b);

    // Plain mul + add — no FMA — to match the scalar backend's
    // round-by-round result. `vfmaq_f32` would fuse and produce
    // 1-ulp-different sums.
    let dl_sq = vmulq_f32(dl, dl);
    let da_sq = vmulq_f32(da, da);
    let db_sq = vmulq_f32(db, db);
    let partial = vaddq_f32(dl_sq, da_sq);
    let d2 = vaddq_f32(partial, db_sq);

    // 4 squared distances out as scalars to find the (per-iteration)
    // min and its lane.
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
