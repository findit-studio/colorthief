//! Always-compiled scalar reference implementation.
//!
//! Iterates the SoA arrays in lockstep, computing
//! `(query - entry).abs²` summed across the three channels and
//! tracking the running min. Used by the dispatcher when no SIMD
//! backend is reachable on the current target, and by every parity
//! test as the bit-identical baseline.

use super::{LABS_A, LABS_B, LABS_L};

/// Reference implementation. On targets where the dispatcher selects
/// a SIMD backend in production builds (e.g. aarch64 NEON), this
/// function is only called from parity tests and the bench harness;
/// the `#[allow(dead_code)]` keeps clippy quiet about that — keeping
/// the scalar reference alive is the whole point.
#[allow(dead_code)]
pub fn nearest_idx(query: [f32; 3]) -> usize {
  let [ql, qa, qb] = query;
  let mut best_idx: usize = 0;
  let mut best_d2 = f32::INFINITY;
  for i in 0..LABS_L.len() {
    let dl = ql - LABS_L[i];
    let da = qa - LABS_A[i];
    let db = qb - LABS_B[i];
    // Order of operations matches every SIMD backend:
    //   ((dl*dl) + (da*da)) + (db*db)
    // Bit-identical when the SIMD path also avoids FMA.
    let d2 = (dl * dl + da * da) + db * db;
    if d2 < best_d2 {
      best_d2 = d2;
      best_idx = i;
    }
  }
  best_idx
}
