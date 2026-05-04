//! WASM SIMD128 backend — 4 entries/iter via 128-bit `v128_load`
//! loads against the SoA `LABS_*` arrays.
//!
//! Compile-time gated to `cfg(all(target_arch = "wasm32",
//! target_feature = "simd128"))`. Mirrors the aarch64 NEON shape
//! exactly (4-lane width, plain mul+add, scalar reduction over a
//! 4-element spill buffer).

#![allow(unsafe_code)]

use core::arch::wasm32::*;

use super::{LABS_A, LABS_B, LABS_L};

/// SIMD128 nearest-neighbor scan.
///
/// Static `cfg(target_feature = "simd128")` guarantees the
/// instructions are available; the safe wrapper just delegates to the
/// `#[target_feature(enable = "simd128")]` worker.
pub fn nearest_idx(query: [f32; 3]) -> usize {
  // SAFETY: SIMD128 is statically guaranteed by the cfg gate on this
  // module's declaration in `mod.rs`.
  unsafe { nearest_idx_simd128(query) }
}

#[target_feature(enable = "simd128")]
unsafe fn nearest_idx_simd128(query: [f32; 3]) -> usize {
  let ql = f32x4_splat(query[0]);
  let qa = f32x4_splat(query[1]);
  let qb = f32x4_splat(query[2]);

  let n = LABS_L.len();
  let chunks = n / 4;

  let mut best_d2 = f32::INFINITY;
  let mut best_idx: usize = 0;

  let l_ptr = LABS_L.as_ptr();
  let a_ptr = LABS_A.as_ptr();
  let b_ptr = LABS_B.as_ptr();

  for chunk in 0..chunks {
    let i = chunk * 4;

    // SAFETY: `i + 4 <= n`; chunks = n / 4 floor.
    let l = unsafe { v128_load(l_ptr.add(i) as *const v128) };
    let a = unsafe { v128_load(a_ptr.add(i) as *const v128) };
    let b = unsafe { v128_load(b_ptr.add(i) as *const v128) };

    let dl = f32x4_sub(ql, l);
    let da = f32x4_sub(qa, a);
    let db = f32x4_sub(qb, b);

    // Plain mul + add — no FMA — to match the scalar baseline.
    let dl_sq = f32x4_mul(dl, dl);
    let da_sq = f32x4_mul(da, da);
    let db_sq = f32x4_mul(db, db);
    let partial = f32x4_add(dl_sq, da_sq);
    let d2 = f32x4_add(partial, db_sq);

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
