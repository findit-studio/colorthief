//! SIMD helpers used by the hot paths in MMCQ (`VBox::count`,
//! `VBox::avg`, and `median_cut` partial sums).
//!
//! The MMCQ benchmark in `benches/extract.rs` showed that for a
//! 64×64 frame `extract` takes ~458 µs and for a 1024×1024 frame
//! ~965 µs — a 256× pixel-count increase costs only ~2.1× wall time.
//! That scaling profile pins the bottleneck on the **fixed-cost**
//! MMCQ work (priority queue, vbox traversal) rather than the
//! per-pixel histogram build. The SIMD here targets the inner loop
//! that traverses the 5-bit b-axis of a vbox: `b1..=b2` is contiguous
//! u32 memory in the `histo` array (because `histo_index` puts `b`
//! in the low bits), so a `vld1q_u32`-style 4-lane parallel add
//! against a 4-wide accumulator is an exact fit.
//!
//! # Backends
//!
//! - [`scalar::sum_u32_slice`] — always compiled, the reference.
//! - [`aarch64_neon::sum_u32_slice`] — NEON, 4 u32 lanes, accumulates
//!   into u64 to dodge intermediate-overflow surprises (the reduced
//!   sum is then saturating-clamped back to u32 to match the scalar
//!   `saturating_add` semantics).
//! - x86 SSE4.1 / AVX2 and WASM SIMD128 follow the same shape.
//!
//! # Bit-parity contract
//!
//! Each backend computes the true (saturating-clamped) sum of the
//! input slice; for any histogram our MMCQ pipeline produces (per-bin
//! counts ≤ pixel count ≤ ~32M for 4K frames) saturation never fires,
//! so all backends produce bit-identical `u32` results. The parity
//! tests in this module enforce that.

/// Dispatcher: returns `slice.iter().fold(0u32, |a, b| a.saturating_add(*b))`
/// for any backend that's reachable on the current target. Compile-time
/// gated only — same `--cfg colorthief_force_scalar` flag the
/// `colorthief-dataset/src/nearest` dispatcher honours short-circuits
/// every SIMD backend here too.
#[allow(unsafe_code)]
#[inline]
pub(crate) fn sum_u32_slice(slice: &[u32]) -> u32 {
  #[cfg(all(target_arch = "aarch64", not(colorthief_force_scalar)))]
  {
    return aarch64_neon::sum_u32_slice(slice);
  }

  #[cfg(all(
    target_arch = "wasm32",
    target_feature = "simd128",
    not(colorthief_force_scalar)
  ))]
  {
    return wasm_simd128::sum_u32_slice(slice);
  }

  #[cfg(all(target_arch = "x86_64", not(colorthief_force_scalar)))]
  {
    if !cfg!(colorthief_disable_avx2) && std::is_x86_feature_detected!("avx2") {
      // SAFETY: feature just verified.
      return unsafe { x86_avx2::sum_u32_slice(slice) };
    }
    if std::is_x86_feature_detected!("sse4.1") {
      // SAFETY: feature just verified.
      return unsafe { x86_sse41::sum_u32_slice(slice) };
    }
  }

  #[allow(unreachable_code)]
  scalar::sum_u32_slice(slice)
}

pub(crate) mod scalar {
  /// Saturating sum of every `u32` in `slice`. Reference impl.
  #[allow(dead_code)]
  pub fn sum_u32_slice(slice: &[u32]) -> u32 {
    let mut sum: u32 = 0;
    for &x in slice {
      sum = sum.saturating_add(x);
    }
    sum
  }
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
pub(crate) mod aarch64_neon {
  // The dispatcher in `super::sum_u32_slice` only routes here when
  // `not(colorthief_force_scalar)`. Under `--cfg colorthief_force_scalar`
  // (the coverage-side flag in `coverage.yml` and the
  // `test-force-scalar` job in `simd.yml`) the dispatcher short-circuits
  // straight to the scalar path, so this module's functions become
  // dead — but on natural-build aarch64 every CI runner exercises them
  // via the standard test job. Same allow-pattern that
  // `colorthief-dataset/src/nearest/scalar.rs` uses for the inverse
  // case.
  #![allow(dead_code)]

  use core::arch::aarch64::*;

  /// NEON saturating sum. Accumulates u32×4 lanes pairwise into a
  /// u64×2 accumulator — no overflow possible until the running sum
  /// exceeds 2⁶⁴ (impossible for any plausible histogram). The final
  /// reduce clamps back to `u32::MAX`, matching scalar's
  /// `saturating_add`.
  pub fn sum_u32_slice(slice: &[u32]) -> u32 {
    // SAFETY: NEON is mandatory on aarch64.
    unsafe { sum_u32_slice_neon(slice) }
  }

  #[target_feature(enable = "neon")]
  unsafe fn sum_u32_slice_neon(slice: &[u32]) -> u32 {
    let mut acc = vdupq_n_u64(0);
    let chunks = slice.chunks_exact(4);
    let remainder = chunks.remainder();
    for chunk in chunks {
      // SAFETY: chunks_exact guarantees 4 u32 = 16 bytes here.
      let v32 = unsafe { vld1q_u32(chunk.as_ptr()) };
      // Pairwise add u32×4 → u64×2: (lane0+lane1, lane2+lane3).
      let widened = vpaddlq_u32(v32);
      acc = vaddq_u64(acc, widened);
    }
    // Horizontal reduce u64×2 → u64.
    let total64: u64 = vaddvq_u64(acc);
    // Tail (≤3 elements).
    let mut total64 = total64;
    for &x in remainder {
      total64 = total64.saturating_add(x as u64);
    }
    total64.min(u32::MAX as u64) as u32
  }
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
pub(crate) mod x86_sse41 {
  #![allow(dead_code)]

  use core::arch::x86_64::*;

  /// SSE4.1 saturating sum. 4 u32 lanes per iteration, widened to
  /// u64 via `_mm_unpacklo_epi32` / `_mm_unpackhi_epi32` against zero
  /// (zero-extension), accumulated into a u64×2 register.
  ///
  /// # Safety
  ///
  /// Caller must guarantee SSE4.1 is available at runtime; the
  /// dispatcher in [`super::sum_u32_slice`] verifies via
  /// [`std::is_x86_feature_detected!`].
  #[target_feature(enable = "sse4.1")]
  pub unsafe fn sum_u32_slice(slice: &[u32]) -> u32 {
    let mut acc = _mm_setzero_si128();
    let zero = _mm_setzero_si128();
    let chunks = slice.chunks_exact(4);
    let remainder = chunks.remainder();
    for chunk in chunks {
      // SAFETY: chunks_exact guarantees 4 u32 = 16 bytes here.
      let v32 = unsafe { _mm_loadu_si128(chunk.as_ptr() as *const __m128i) };
      // Zero-extend u32×4 → u64×2 + u64×2 (low pair, high pair).
      let lo64 = _mm_unpacklo_epi32(v32, zero);
      let hi64 = _mm_unpackhi_epi32(v32, zero);
      acc = _mm_add_epi64(acc, lo64);
      acc = _mm_add_epi64(acc, hi64);
    }
    // Horizontal reduce u64×2 → u64.
    let mut buf = [0u64; 2];
    // SAFETY: 16-byte write into a 16-byte buffer.
    unsafe { _mm_storeu_si128(buf.as_mut_ptr() as *mut __m128i, acc) };
    let mut total64 = buf[0].saturating_add(buf[1]);
    for &x in remainder {
      total64 = total64.saturating_add(x as u64);
    }
    total64.min(u32::MAX as u64) as u32
  }
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
pub(crate) mod x86_avx2 {
  #![allow(dead_code)]

  use core::arch::x86_64::*;

  /// AVX2 saturating sum. 8 u32 lanes per iteration; each iteration
  /// produces a u64×4 partial sum via `_mm256_unpacklo_epi32` /
  /// `_mm256_unpackhi_epi32` against zero. Halves the iteration
  /// count vs SSE4.1.
  ///
  /// # Safety
  ///
  /// Caller must guarantee AVX2 is available at runtime.
  #[target_feature(enable = "avx2")]
  pub unsafe fn sum_u32_slice(slice: &[u32]) -> u32 {
    let mut acc = _mm256_setzero_si256();
    let zero = _mm256_setzero_si256();
    let chunks = slice.chunks_exact(8);
    let remainder = chunks.remainder();
    for chunk in chunks {
      // SAFETY: chunks_exact guarantees 8 u32 = 32 bytes here.
      let v32 = unsafe { _mm256_loadu_si256(chunk.as_ptr() as *const __m256i) };
      let lo64 = _mm256_unpacklo_epi32(v32, zero);
      let hi64 = _mm256_unpackhi_epi32(v32, zero);
      acc = _mm256_add_epi64(acc, lo64);
      acc = _mm256_add_epi64(acc, hi64);
    }
    let mut buf = [0u64; 4];
    // SAFETY: 32-byte write into a 32-byte buffer.
    unsafe { _mm256_storeu_si256(buf.as_mut_ptr() as *mut __m256i, acc) };
    let mut total64 = buf[0]
      .saturating_add(buf[1])
      .saturating_add(buf[2])
      .saturating_add(buf[3]);
    for &x in remainder {
      total64 = total64.saturating_add(x as u64);
    }
    total64.min(u32::MAX as u64) as u32
  }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[allow(unsafe_code)]
pub(crate) mod wasm_simd128 {
  #![allow(dead_code)]

  use core::arch::wasm32::*;

  /// WASM SIMD128 saturating sum. Same shape as SSE4.1 — 4 u32 lanes
  /// per iter, widened to u64×2 via the `u64x2_extend_low_u32x4` /
  /// `u64x2_extend_high_u32x4` ops, accumulated.
  pub fn sum_u32_slice(slice: &[u32]) -> u32 {
    // SAFETY: SIMD128 statically guaranteed by the cfg gate on the
    // module's declaration in `super::sum_u32_slice`.
    unsafe { sum_u32_slice_simd128(slice) }
  }

  #[target_feature(enable = "simd128")]
  unsafe fn sum_u32_slice_simd128(slice: &[u32]) -> u32 {
    let mut acc = u64x2_splat(0);
    let chunks = slice.chunks_exact(4);
    let remainder = chunks.remainder();
    for chunk in chunks {
      // SAFETY: chunks_exact guarantees 4 u32 = 16 bytes here.
      let v32 = unsafe { v128_load(chunk.as_ptr() as *const v128) };
      let lo64 = u64x2_extend_low_u32x4(v32);
      let hi64 = u64x2_extend_high_u32x4(v32);
      acc = u64x2_add(acc, lo64);
      acc = u64x2_add(acc, hi64);
    }
    let lane0 = u64x2_extract_lane::<0>(acc);
    let lane1 = u64x2_extract_lane::<1>(acc);
    let mut total64 = lane0.saturating_add(lane1);
    for &x in remainder {
      total64 = total64.saturating_add(x as u64);
    }
    total64.min(u32::MAX as u64) as u32
  }
}

#[cfg(test)]
#[allow(unsafe_code)]
mod tests {
  use super::*;

  /// Three sample sizes — empty, partial-chunk-only, and chunk +
  /// non-empty tail — exercise both the vectorised core and the
  /// scalar tail of every backend.
  fn parity_inputs() -> Vec<Vec<u32>> {
    vec![
      vec![],
      vec![1, 2, 3],
      vec![1, 2, 3, 4],
      vec![1, 2, 3, 4, 5],
      vec![10, 20, 30, 40, 50, 60, 70, 80, 90],
      // Histogram-shaped: many zeros + a few populated bins.
      {
        let mut v = vec![0u32; 31];
        v[0] = 100;
        v[5] = 200;
        v[20] = 50;
        v[30] = 1;
        v
      },
      // Saturation-corner: one bin near u32::MAX. Scalar saturates
      // immediately; SIMD widens to u64 and clamps. Both must agree.
      {
        let mut v = vec![1u32; 5];
        v[0] = u32::MAX - 2;
        v
      },
    ]
  }

  #[test]
  fn scalar_matches_naive_fold() {
    for input in parity_inputs() {
      let naive: u32 = input.iter().fold(0u32, |a, b| a.saturating_add(*b));
      assert_eq!(scalar::sum_u32_slice(&input), naive);
    }
  }

  #[test]
  #[cfg(target_arch = "aarch64")]
  fn neon_matches_scalar() {
    for input in parity_inputs() {
      let s = scalar::sum_u32_slice(&input);
      let n = aarch64_neon::sum_u32_slice(&input);
      assert_eq!(n, s, "NEON divergence on input of len {}", input.len());
    }
  }

  #[test]
  #[cfg(target_arch = "x86_64")]
  fn sse41_matches_scalar() {
    if !std::is_x86_feature_detected!("sse4.1") {
      eprintln!("skipping: SSE4.1 not detected");
      return;
    }
    for input in parity_inputs() {
      let s = scalar::sum_u32_slice(&input);
      // SAFETY: feature verified above.
      let v = unsafe { x86_sse41::sum_u32_slice(&input) };
      assert_eq!(v, s, "SSE4.1 divergence on input of len {}", input.len());
    }
  }

  #[test]
  #[cfg(target_arch = "x86_64")]
  fn avx2_matches_scalar() {
    if !std::is_x86_feature_detected!("avx2") {
      eprintln!("skipping: AVX2 not detected");
      return;
    }
    for input in parity_inputs() {
      let s = scalar::sum_u32_slice(&input);
      // SAFETY: feature verified above.
      let v = unsafe { x86_avx2::sum_u32_slice(&input) };
      assert_eq!(v, s, "AVX2 divergence on input of len {}", input.len());
    }
  }

  #[test]
  #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
  fn wasm_simd128_matches_scalar() {
    for input in parity_inputs() {
      let s = scalar::sum_u32_slice(&input);
      let v = wasm_simd128::sum_u32_slice(&input);
      assert_eq!(v, s);
    }
  }
}
