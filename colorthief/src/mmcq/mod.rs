//! MMCQ (Modified Median Cut Quantization).
//!
//! Faithful Rust port of color-thief's TypeScript implementation
//! (`color-thief/src/quantizers/mmcq.ts`), itself derived from
//! Lokesh Dhakar's `quantize.js` and ultimately Leptonica's MMCQ.
//!
//! # Algorithm sketch
//!
//! 1. Quantize each pixel to 5-bit channels and bin into a 32K-entry
//!    histogram.
//! 2. Initial bounding "vbox" covers all populated bins.
//! 3. Two-phase iterative splitting on a priority queue:
//!    - Phase 1 (until `0.75 * target` boxes): split the box with the
//!      largest pixel count along its longest axis at the
//!      population median.
//!    - Phase 2 (until `target` boxes): re-sort by `population *
//!      volume` and continue splitting.
//! 4. Each surviving box's pixel-count-weighted average is one
//!    dominant.

mod simd;

use core::mem::MaybeUninit;

use crate::Buffer;

const SIGBITS: u32 = 5;
const RSHIFT: u32 = 8 - SIGBITS;
const HISTO_LEVELS: usize = 1 << SIGBITS; // 32
const HISTO_SIZE: usize = 1 << (3 * SIGBITS); // 32768
const MAX_ITERATIONS: usize = 1000;
/// MMCQ never produces more than `target` boxes, and `target` is
/// clamped to 256. Plus a small safety margin for the in-loop
/// "exhausted" pile that's drained back at the end. 256 is enough.
const MAX_BOXES: usize = 256;

// =====================================================================
// BoxArena — fixed-capacity, push/pop/sort-by-key inline storage for
// VBox. Replaces the `Vec<VBox>` that MMCQ used pre-no_alloc-refactor.
// =====================================================================

/// Inline-array `Vec<VBox>` replacement. ~6 KB inline, no heap.
///
/// Used in two places: as the splittable queue inside [`Mmcq::boxes`]
/// and as the side "exhausted" pile inside `iterate_split`. Every
/// public method matches the `Vec<VBox>` API piece used by the
/// pre-refactor MMCQ (push/pop/len/sort/extend) so the algorithm
/// translation is one-to-one.
pub(crate) struct BoxArena {
  data: [MaybeUninit<VBox>; MAX_BOXES],
  len: usize,
}

#[allow(unsafe_code)]
impl BoxArena {
  /// Empty arena. `const fn` so [`Mmcq::new`] is `const`.
  pub const fn new() -> Self {
    Self {
      data: [const { MaybeUninit::uninit() }; MAX_BOXES],
      len: 0,
    }
  }

  pub fn len(&self) -> usize {
    self.len
  }

  /// Push `v`. Returns `false` if at capacity. MMCQ's internals never
  /// produce more than `MAX_BOXES` boxes given the `target` clamp; the
  /// `false` return is a safety net rather than an expected path.
  pub fn push(&mut self, v: VBox) -> bool {
    if self.len >= MAX_BOXES {
      return false;
    }
    self.data[self.len].write(v);
    self.len += 1;
    true
  }

  pub fn pop(&mut self) -> Option<VBox> {
    if self.len == 0 {
      return None;
    }
    self.len -= 1;
    // SAFETY: `self.data[..self.len + 1]` were initialized via `push`;
    // we just decremented `len` so `self.data[self.len]` is still
    // initialized and we read it out, leaving the slot uninitialized
    // (which is fine — `len` reflects the new boundary).
    Some(unsafe { self.data[self.len].assume_init_read() })
  }

  pub fn as_mut_slice(&mut self) -> &mut [VBox] {
    // SAFETY: `self.data[..self.len]` are initialized.
    unsafe { core::slice::from_raw_parts_mut(self.data.as_mut_ptr() as *mut VBox, self.len) }
  }

  /// Drop all initialized elements; reset `len` to 0.
  pub fn clear(&mut self) {
    while self.pop().is_some() {}
  }

  /// Move every element from `self` into `dst`. After return, `self`
  /// is empty. Used at the end of `iterate_split` to merge the
  /// "exhausted" pile back into the splittable queue.
  pub fn drain_into(&mut self, dst: &mut Self) {
    while let Some(v) = self.pop() {
      // MMCQ invariant: total boxes never exceeds MAX_BOXES, so
      // `dst.push` should never fail in practice.
      let _ = dst.push(v);
    }
  }
}

impl Drop for BoxArena {
  fn drop(&mut self) {
    self.clear();
  }
}

/// Encode a 5-bit (R, G, B) coord into a flat histogram index.
#[inline]
fn histo_index(r: u32, g: u32, b: u32) -> usize {
  ((r << (2 * SIGBITS)) + (g << SIGBITS) + b) as usize
}

/// A 3-D bounding box in 5-bit RGB space. Bounds are inclusive
/// (`r1..=r2`). Pixel count + average color are cached on first access.
#[derive(Clone)]
pub(crate) struct VBox {
  r1: u32,
  r2: u32,
  g1: u32,
  g2: u32,
  b1: u32,
  b2: u32,
  count_cache: Option<u32>,
  avg_cache: Option<[u8; 3]>,
}

impl VBox {
  fn volume(&self) -> u32 {
    (self.r2 - self.r1 + 1) * (self.g2 - self.g1 + 1) * (self.b2 - self.b1 + 1)
  }

  fn count(&mut self, histo: &[u32]) -> u32 {
    if let Some(c) = self.count_cache {
      return c;
    }
    // Inner b-axis is contiguous in `histo` (`histo_index` puts `b`
    // in the low bits). Hand each (r, g) row's bin slice to
    // [`simd::sum_u32_slice`] which reduces the per-row sum on
    // whichever SIMD backend the dispatcher selects (NEON / SSE4.1 /
    // AVX2 / WASM SIMD128 / scalar). This is the bench-identified
    // hottest path — see `benches/extract.rs` for the scaling profile
    // that motivated targeting `count` first.
    let mut npix: u32 = 0;
    for r in self.r1..=self.r2 {
      for g in self.g1..=self.g2 {
        let lo = histo_index(r, g, self.b1);
        let hi = histo_index(r, g, self.b2);
        let row_sum = simd::sum_u32_slice(&histo[lo..=hi]);
        npix = npix.saturating_add(row_sum);
      }
    }
    self.count_cache = Some(npix);
    npix
  }

  /// Pixel-count-weighted average of the box's bins, mapped back to
  /// 8-bit RGB. Each populated bin contributes
  /// `pop * (idx + 0.5) * 2^RSHIFT` (the `+ 0.5` puts the contribution
  /// at the bin center, which spans `2^RSHIFT` 8-bit values), then the
  /// per-channel mean is truncated to `u8`.
  ///
  /// The TS reference (and earlier rounds of this Rust port) carried
  /// a single-bin special case that returned `idx << RSHIFT` (the
  /// bin's lower edge) instead of the centered formula. That biased
  /// every solid-color or low-variance frame down by `2^(RSHIFT - 1)`
  /// units — pure white mapped to `[248, 248, 248]` instead of
  /// `[252, 252, 252]`.
  fn avg(&mut self, histo: &[u32]) -> [u8; 3] {
    if let Some(a) = self.avg_cache {
      return a;
    }
    let mult = 1u32 << RSHIFT;

    let mut ntot: u64 = 0;
    let mut rsum: f64 = 0.0;
    let mut gsum: f64 = 0.0;
    let mut bsum: f64 = 0.0;
    for r in self.r1..=self.r2 {
      for g in self.g1..=self.g2 {
        for b in self.b1..=self.b2 {
          let pop = histo[histo_index(r, g, b)] as u64;
          if pop == 0 {
            continue;
          }
          ntot += pop;
          let popf = pop as f64;
          rsum += popf * (r as f64 + 0.5) * mult as f64;
          gsum += popf * (g as f64 + 0.5) * mult as f64;
          bsum += popf * (b as f64 + 0.5) * mult as f64;
        }
      }
    }

    let out = if ntot > 0 {
      let n = ntot as f64;
      [(rsum / n) as u8, (gsum / n) as u8, (bsum / n) as u8]
    } else {
      // Empty box: 8-bit RGB at the box's geometric center.
      let center = |a: u32, b: u32| -> u8 { ((mult * (a + b + 1)) / 2).min(255) as u8 };
      [
        center(self.r1, self.r2),
        center(self.g1, self.g2),
        center(self.b1, self.b2),
      ]
    };
    self.avg_cache = Some(out);
    out
  }
}

/// Build the 32K-entry histogram from an iterator of u8 RGB pixels
/// into a caller-provided slice.
///
/// `histo` must be `HISTO_SIZE` entries; passed by mutable reference
/// so the caller decides where the 128 KB lives (typically inside
/// [`Mmcq`], on the heap or in `static mut`). Reset to zero before
/// counting.
///
/// Generic over the pixel source so callers can feed in
/// [`crate::RgbFrame::pixels`] (8-bit packed) or
/// [`crate::Rgb48Frame::pixels`] (16-bit packed, downscaled to u8 in
/// the iterator) without duplicating MMCQ.
fn build_histogram<I: Iterator<Item = [u8; 3]>>(pixels: I, histo: &mut [u32; HISTO_SIZE]) {
  histo.fill(0);
  for [r, g, b] in pixels {
    let rv = (r as u32) >> RSHIFT;
    let gv = (g as u32) >> RSHIFT;
    let bv = (b as u32) >> RSHIFT;
    histo[histo_index(rv, gv, bv)] = histo[histo_index(rv, gv, bv)].saturating_add(1);
  }
}

/// Initial bounding box covering all populated histogram bins. Returns
/// `None` on an empty histogram (frame had zero pixels — shouldn't
/// happen, [`RgbFrame::try_new`] rejects zero-dimension frames).
fn initial_vbox(histo: &[u32]) -> Option<VBox> {
  let mut rmin = u32::MAX;
  let mut rmax = 0;
  let mut gmin = u32::MAX;
  let mut gmax = 0;
  let mut bmin = u32::MAX;
  let mut bmax = 0;
  let mut any = false;
  for r in 0..HISTO_LEVELS as u32 {
    for g in 0..HISTO_LEVELS as u32 {
      for b in 0..HISTO_LEVELS as u32 {
        if histo[histo_index(r, g, b)] > 0 {
          any = true;
          if r < rmin {
            rmin = r;
          }
          if r > rmax {
            rmax = r;
          }
          if g < gmin {
            gmin = g;
          }
          if g > gmax {
            gmax = g;
          }
          if b < bmin {
            bmin = b;
          }
          if b > bmax {
            bmax = b;
          }
        }
      }
    }
  }
  if !any {
    return None;
  }
  Some(VBox {
    r1: rmin,
    r2: rmax,
    g1: gmin,
    g2: gmax,
    b1: bmin,
    b2: bmax,
    count_cache: None,
    avg_cache: None,
  })
}

#[derive(Clone, Copy)]
enum Axis {
  R,
  G,
  B,
}

/// Median-cut split. Returns `Some((left, Some(right)))` on a successful
/// split, `Some((self, None))` for boxes that can't be split further
/// (single bin or single pixel), or `None` if the box is empty.
fn median_cut(vbox: &VBox, histo: &[u32]) -> Option<(VBox, Option<VBox>)> {
  let mut probe = vbox.clone();
  let count = probe.count(histo);
  if count == 0 {
    return None;
  }
  if count == 1 {
    return Some((vbox.clone(), None));
  }

  let rw = vbox.r2 - vbox.r1 + 1;
  let gw = vbox.g2 - vbox.g1 + 1;
  let bw = vbox.b2 - vbox.b1 + 1;

  // Single-bin defensive guard (rw == gw == bw == 1): not splittable.
  // The TS reference doesn't check this explicitly and ends up
  // producing a degenerate (empty, full) split that the priority
  // queue iterates over until MAX_ITERATIONS. We short-circuit.
  if rw == 1 && gw == 1 && bw == 1 {
    return Some((vbox.clone(), None));
  }

  let maxw = rw.max(gw).max(bw);
  let axis = if maxw == rw {
    Axis::R
  } else if maxw == gw {
    Axis::G
  } else {
    Axis::B
  };

  // Build cumulative population along the chosen axis. `partialsum[i]`
  // is the cumulative sum of pixels in slices `[lo..=i]` along the cut
  // axis (`i` is the absolute 5-bit coordinate, NOT an offset), so we
  // size the array to `HISTO_LEVELS` for direct indexing.
  let (lo, hi) = match axis {
    Axis::R => (vbox.r1, vbox.r2),
    Axis::G => (vbox.g1, vbox.g2),
    Axis::B => (vbox.b1, vbox.b2),
  };

  let mut partialsum = [0u32; HISTO_LEVELS];
  let mut total: u32 = 0;
  for i in lo..=hi {
    let mut sum: u32 = 0;
    match axis {
      Axis::R => {
        for g in vbox.g1..=vbox.g2 {
          for b in vbox.b1..=vbox.b2 {
            sum = sum.saturating_add(histo[histo_index(i, g, b)]);
          }
        }
      }
      Axis::G => {
        for r in vbox.r1..=vbox.r2 {
          for b in vbox.b1..=vbox.b2 {
            sum = sum.saturating_add(histo[histo_index(r, i, b)]);
          }
        }
      }
      Axis::B => {
        for r in vbox.r1..=vbox.r2 {
          for g in vbox.g1..=vbox.g2 {
            sum = sum.saturating_add(histo[histo_index(r, g, i)]);
          }
        }
      }
    }
    total = total.saturating_add(sum);
    partialsum[i as usize] = total;
  }

  let lookaheadsum: [u32; HISTO_LEVELS] =
    core::array::from_fn(|i| total.saturating_sub(partialsum[i]));

  // Find the first slice where the cumulative population crosses the
  // halfway mark, then nudge the cut so neither side is empty.
  for i in lo..=hi {
    if partialsum[i as usize] <= total / 2 {
      continue;
    }
    let left = i - lo;
    let right = hi - i;

    // Center the cut between i and the box edge that's farther away.
    // `as i64` math avoids u32 underflow for the `i - 1 - left/2` branch.
    let d2_initial: i64 = if left <= right {
      let candidate = i as i64 + (right / 2) as i64;
      candidate.min(hi as i64 - 1)
    } else {
      let candidate = i as i64 - 1 - (left / 2) as i64;
      candidate.max(lo as i64)
    };
    let mut d2 = d2_initial.clamp(lo as i64, hi as i64) as u32;

    // Walk forward to a populated slice if d2 landed on a hole.
    while d2 < hi && partialsum[d2 as usize] == 0 {
      d2 += 1;
    }
    // If the right half is empty, walk d2 backward through populated
    // slices until it isn't. Stop at lo; never go below.
    while d2 > lo && lookaheadsum[d2 as usize] == 0 && partialsum[(d2 - 1) as usize] != 0 {
      d2 -= 1;
    }

    // d2 must stay strictly inside [lo, hi-1] so the right half is
    // non-degenerate — otherwise the right vbox starts at hi+1.
    let d2 = d2.min(hi.saturating_sub(1)).max(lo);

    let mut left_box = vbox.clone();
    let mut right_box = vbox.clone();
    match axis {
      Axis::R => {
        left_box.r2 = d2;
        right_box.r1 = d2 + 1;
      }
      Axis::G => {
        left_box.g2 = d2;
        right_box.g1 = d2 + 1;
      }
      Axis::B => {
        left_box.b2 = d2;
        right_box.b1 = d2 + 1;
      }
    }
    left_box.count_cache = None;
    left_box.avg_cache = None;
    right_box.count_cache = None;
    right_box.avg_cache = None;
    return Some((left_box, Some(right_box)));
  }

  // Population never crossed total/2 — degenerate box, can't split.
  Some((vbox.clone(), None))
}

/// Iterative splitting against a `score(box) -> u64` ordering. Each
/// iteration sorts ascending and splits the highest-score box.
///
/// Empty children are dropped at the source rather than counted toward
/// `target`. Without this, `median_cut` can produce a `(populated,
/// empty)` pair for a sparse parent box (one populated bin inside a
/// wide range on one axis), the loop would terminate at `boxes.len()
/// == target` with the empty half consuming a slot, and lower-scored
/// splittable boxes would never get popped — the caller's
/// `quantize` filter then drops the empty box and `extract`
/// underfills relative to `count`.
///
/// Boxes that can't be split further (single-bin, or median-cut
/// returns `Some((_, None))`) are moved to a side `exhausted` pile so
/// they don't get popped again, but the loop keeps trying lower-
/// scored boxes — they may still be productively splittable.
fn iterate_split<F>(boxes: &mut BoxArena, target: usize, histo: &[u32], score: F)
where
  F: Fn(&mut VBox, &[u32]) -> u64,
{
  let mut iters = 0;
  let mut exhausted = BoxArena::new();

  while boxes.len() + exhausted.len() < target && iters < MAX_ITERATIONS {
    iters += 1;
    // `sort_unstable_by_key` (not `sort_by_key`) — the unstable in-
    // place sort is `core::slice` and works in `no_alloc`. Tie-break
    // ordering doesn't affect MMCQ correctness; the score is the
    // only thing that matters for splittable selection.
    boxes.as_mut_slice().sort_unstable_by_key(|b| {
      // sort_by_key needs a fresh `b` each call; clone is cheap (no
      // alloc — VBox is `Copy`-like aside from the `Option` caches).
      let mut probe = b.clone();
      score(&mut probe, histo)
    });
    let mut top = match boxes.pop() {
      Some(b) => b,
      None => break,
    };
    if top.count(histo) == 0 {
      // No populated boxes left in the splittable pile.
      break;
    }
    match median_cut(&top, histo) {
      Some((mut left, Some(mut right))) => {
        let l_pop = left.count(histo);
        let r_pop = right.count(histo);
        match (l_pop, r_pop) {
          // `top` was non-empty above so this should be unreachable;
          // defensive: keep `top` as exhausted rather than fabricating
          // empty children.
          (0, 0) => {
            let _ = exhausted.push(top);
          }
          // `(populated, empty)` or `(empty, populated)` — push only
          // the populated child. Net: the queue shrinks by one
          // splittable candidate but `boxes.len() + exhausted.len()`
          // doesn't grow toward `target`, so the loop continues with
          // the next-highest box.
          (0, _) => {
            let _ = boxes.push(right);
          }
          (_, 0) => {
            let _ = boxes.push(left);
          }
          (_, _) => {
            let _ = boxes.push(left);
            let _ = boxes.push(right);
          }
        }
      }
      Some((only, None)) => {
        // Unsplittable (single-bin or median-cut found no productive
        // split). Set aside so we don't pop it again, but don't break
        // — lower-scored boxes might still split.
        let _ = exhausted.push(only);
      }
      None => {
        // `top.count() > 0` was just verified, so median_cut returning
        // `None` is unreachable in practice. Defensive: mark exhausted.
        let _ = exhausted.push(top);
      }
    }
  }

  // Re-merge so callers see the full set (splittable + exhausted).
  exhausted.drain_into(boxes);
}

// =====================================================================
// Mmcq — MMCQ workspace + extract method.
// =====================================================================

/// MMCQ workspace. Holds the 32K-entry histogram and the box queue
/// inline as fixed-size arrays — no heap allocations for either.
///
/// Total in-place footprint: ~134 KB (`[u32; 32_768]` = 128 KB +
/// `BoxArena` ~6 KB + `usize`).
///
/// # Construction
///
/// Use [`Mmcq::new_boxed`] (when `alloc` is enabled) for runtime
/// construction — it places the workspace on the heap via
/// `Box::new_zeroed` and avoids a 134 KB stack frame. Use
/// [`Mmcq::new`] (a `const fn`) for `static` placement on `no_alloc`
/// targets:
///
/// ```ignore
/// // alloc available:
/// let mut mmcq = colorthief::Mmcq::new_boxed();
///
/// // bare metal / wasm / other no_alloc:
/// static mut MMCQ: colorthief::Mmcq = colorthief::Mmcq::new();
/// ```
///
/// **Do not** call `Mmcq::new()` in a non-`static` non-`const`
/// context — it will create a 134 KB stack frame and almost certainly
/// blow embedded stacks (and is risky even on desktop in deep call
/// chains).
pub struct Mmcq {
  histogram: [u32; HISTO_SIZE],
  boxes: BoxArena,
}

impl Mmcq {
  /// Zero-initialized workspace. **Only call this in a `static` /
  /// `const` context** (or any context where you've confirmed your
  /// stack budget can absorb 134 KB).
  ///
  /// ```ignore
  /// static mut MMCQ: colorthief::Mmcq = colorthief::Mmcq::new();
  /// ```
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {
      histogram: [0u32; HISTO_SIZE],
      boxes: BoxArena::new(),
    }
  }

  /// Heap-allocated workspace constructor — avoids the 134 KB stack
  /// frame `Mmcq::new()` would produce in the caller.
  ///
  /// Uses `Box::new_zeroed` (stable since Rust 1.82) which allocates
  /// and zero-fills directly on the heap; `assume_init` is a bit-level
  /// cast with no copy.
  #[cfg(any(feature = "alloc", feature = "std"))]
  #[cfg_attr(docsrs, doc(cfg(any(feature = "alloc", feature = "std"))))]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn new_boxed() -> std::boxed::Box<Self> {
    // SAFETY: every field of `Mmcq` has a valid all-zero bit-pattern:
    //
    //   - histogram: `[u32; HISTO_SIZE]`, zero is valid u32
    //   - boxes: `BoxArena { data: [MaybeUninit<VBox>; N], len: 0 }`
    //     `MaybeUninit` accepts any pattern, `len = 0` is valid.
    #[allow(unsafe_code)]
    unsafe {
      std::boxed::Box::<Self>::new_zeroed().assume_init()
    }
  }

  /// Run MMCQ on a pixel iterator, naming each dominant via `algo`,
  /// pushing the named [`crate::Dominant`]s into `out` (sorted
  /// descending by population). Stops early on the first `out`
  /// rejection, leaving the buffer in whatever state it was in.
  ///
  /// **Allocates nothing** — all working storage lives in `&mut self`.
  /// Resets the histogram and box queue at entry, so a single `Mmcq`
  /// can be reused across many calls (e.g. via `thread_local!`).
  pub fn extract<I, B>(&mut self, pixels: I, count: u8, algo: crate::Algorithm, out: &mut B)
  where
    I: Iterator<Item = [u8; 3]>,
    B: Buffer<crate::Dominant>,
  {
    if count == 0 {
      return;
    }

    // Reset state from previous calls.
    self.boxes.clear();

    // Build histogram.
    build_histogram(pixels, &mut self.histogram);

    // Total pixel count = sum of all histogram bins. Used to derive
    // each dominant's `percentage` of the source frame. One pass
    // over the histogram (32 768 u32 adds ≈ a few µs) is independent
    // of frame size, unlike incrementing a counter per pixel.
    // Saturating-add for the unlikely > 4 G pixel case.
    let total_pixels: u32 = self
      .histogram
      .iter()
      .fold(0u32, |a, b| a.saturating_add(*b));
    let inv_total_pct: f32 = if total_pixels == 0 {
      0.0
    } else {
      100.0 / total_pixels as f32
    };

    // Initial bounding box. Empty histogram → no work.
    let initial = match initial_vbox(&self.histogram) {
      Some(b) => b,
      None => return,
    };
    let _ = self.boxes.push(initial);

    // MMCQ is undefined outside [2, 256]. Saturate to that range.
    let target = (count as usize).clamp(2, 256);

    // Phase 1: split by raw population. Phase target uses the
    // `ceil(0.75 * target)` shape the TS reference relies on:
    // `(3 * target).div_ceil(4)` matches `(0.75 * target).ceil() as
    // usize` exactly for every `target ∈ [2, 256]` (verified across
    // the range) and stays in `core` (`f64::ceil` is std-only).
    let phase1_target = (3 * target).div_ceil(4);
    iterate_split(&mut self.boxes, phase1_target, &self.histogram, |b, h| {
      b.count(h) as u64
    });

    // Phase 2: split by population * volume.
    iterate_split(&mut self.boxes, target, &self.histogram, |b, h| {
      (b.count(h) as u64) * (b.volume() as u64)
    });

    // Sort the final palette descending by population so the caller
    // gets the most-dominant color first.
    let histo: &[u32] = &self.histogram;
    self.boxes.as_mut_slice().sort_unstable_by_key(|b| {
      let mut probe = b.clone();
      core::cmp::Reverse(probe.count(histo))
    });

    // Push dominants to `out`, capped at `count`, skipping zero-
    // population boxes (sparse parent split into populated+empty).
    let mut written: usize = 0;
    let max_out = count as usize;
    for vbox in self.boxes.as_mut_slice().iter_mut() {
      if written >= max_out {
        break;
      }
      let pop = vbox.count(histo);
      if pop == 0 {
        continue;
      }
      let rgb = vbox.avg(histo);
      let dominant = crate::Dominant {
        rgb,
        color: algo.extract(rgb),
        population: pop,
        percentage: pop as f32 * inv_total_pct,
      };
      if out.try_push(dominant).is_some() {
        // Buffer full; stop.
        break;
      }
      written += 1;
    }
  }
}

// =====================================================================
// alloc-tier convenience wrapper (used by lib.rs's `extract*`).
//
// Two impls of `quantize` are defined below, mutually-exclusive on
// the `std` feature so callers always see one signature:
//
//   1. `feature = "std"`: thread_local!-cached `Box<Mmcq>` — one
//      ~134 KB allocation per OS thread for the program's lifetime;
//      zero per-call. Per-thread isolation makes this sound under
//      any threading model.
//   2. `feature = "alloc"` (no `std`): per-call `Mmcq::new_boxed()`.
//      ~134 KB heap alloc per call. Stateless, sound under any
//      threading model.
//
// A previous design exposed a `single-threaded` feature that wrapped
// a `OnceCell<RefCell<Box<Mmcq>>>` in an `unsafe impl Sync` shim, on
// the assumption that flipping the feature flag was the user's
// promise of single-threaded access. Codex adversarial review caught
// this as a soundness hole: Cargo features are global build-time
// configuration, not per-call invariants, so a multi-threaded RTOS
// build with `single-threaded` enabled (intentionally or
// transitively) hit data-race UB through fully-safe `extract` calls.
// The feature was removed; users in genuinely-single-threaded
// `no_std + alloc` environments who want a cached workspace should
// place an `Mmcq` in their own `static mut` and call `Mmcq::extract`
// directly — the `unsafe` then sits at their call site, not silently
// inside this crate.
// =====================================================================

// --- Tier 1: std (thread_local-cached) ------------------------------
#[cfg(feature = "std")]
std::thread_local! {
  static MMCQ_TLS: core::cell::RefCell<std::boxed::Box<Mmcq>> =
    core::cell::RefCell::new(Mmcq::new_boxed());
}

/// Run MMCQ on a pixel iterator and return up to `count` named
/// dominants in a `Vec`. See module-level comment above for the
/// per-tier caching strategy.
#[cfg(feature = "std")]
pub(crate) fn quantize<I: Iterator<Item = [u8; 3]>>(
  pixels: I,
  count: u8,
  algo: crate::Algorithm,
) -> std::vec::Vec<crate::Dominant> {
  MMCQ_TLS.with(|cell| {
    let mut mmcq = cell.borrow_mut();
    let mut out: std::vec::Vec<crate::Dominant> = std::vec::Vec::new();
    mmcq.extract(pixels, count, algo, &mut out);
    out
  })
}

// --- Tier 2: alloc-only (per-call) ----------------------------------
#[cfg(all(feature = "alloc", not(feature = "std")))]
pub(crate) fn quantize<I: Iterator<Item = [u8; 3]>>(
  pixels: I,
  count: u8,
  algo: crate::Algorithm,
) -> std::vec::Vec<crate::Dominant> {
  let mut mmcq = Mmcq::new_boxed();
  let mut out: std::vec::Vec<crate::Dominant> = std::vec::Vec::new();
  mmcq.extract(pixels, count, algo, &mut out);
  out
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::RgbFrame;

  /// Build a synthetic RgbFrame from a Vec of [R,G,B] triples so tests
  /// don't have to open image files.
  fn make_frame(width: u32, height: u32, pixels: &[[u8; 3]]) -> Vec<u8> {
    assert_eq!(pixels.len() as u32, width * height);
    let mut buf = Vec::with_capacity(pixels.len() * 3);
    for p in pixels {
      buf.extend_from_slice(p);
    }
    buf
  }

  #[test]
  fn solid_red_frame_yields_a_red_dominant() {
    // 4x4 solid red, no padding.
    let pixels = vec![[255, 0, 0]; 16];
    let buf = make_frame(4, 4, &pixels);
    let frame = RgbFrame::try_new(&buf, 4, 4, 12).expect("frame");
    let dominants = quantize(frame.pixels(), 5, crate::Algorithm::default());
    assert!(!dominants.is_empty(), "MMCQ produced zero dominants");
    let top = &dominants[0];
    // 5-bit quantization shifts pure red (255,0,0) → bin (31,0,0).
    // avg() returns the bin center: ((31 + 0.5) * 8, 4, 4) = (252, 4, 4).
    assert!(top.rgb[0] > 200, "expected R>200, got {:?}", top.rgb);
    assert!(top.rgb[1] < 30, "expected G<30, got {:?}", top.rgb);
    assert!(top.rgb[2] < 30, "expected B<30, got {:?}", top.rgb);
  }

  /// pre-fix, the single-bin
  /// shortcut in `avg()` returned `idx << RSHIFT` (the bin's lower
  /// edge), which biased every solid-color frame down by half a bin
  /// width. Pure white was reported as `[248, 248, 248]` instead of
  /// the bin center `[252, 252, 252]`. After deleting the shortcut,
  /// the general weighted-average path produces the correct centered
  /// result for single-bin boxes.
  #[test]
  fn solid_white_recovered_at_bin_center() {
    let pixels = vec![[255u8, 255, 255]; 16];
    let buf = make_frame(4, 4, &pixels);
    let frame = RgbFrame::try_new(&buf, 4, 4, 12).expect("frame");
    let dominants = quantize(frame.pixels(), 5, crate::Algorithm::default());
    let top = &dominants[0];
    // bin (31, 31, 31); center = (31.5 * 8) = 252.
    assert_eq!(top.rgb, [252, 252, 252], "expected bin-center white");
  }

  /// Pure black: bin (0, 0, 0); center = (0.5 * 8) = 4.
  /// Pre-fix this was `[0, 0, 0]` (lower-edge bias).
  #[test]
  fn solid_black_recovered_at_bin_center() {
    let pixels = vec![[0u8, 0, 0]; 16];
    let buf = make_frame(4, 4, &pixels);
    let frame = RgbFrame::try_new(&buf, 4, 4, 12).expect("frame");
    let dominants = quantize(frame.pixels(), 5, crate::Algorithm::default());
    let top = &dominants[0];
    assert_eq!(top.rgb, [4, 4, 4], "expected bin-center black");
  }

  /// Two distinct 8-bit values that fall into the same 5-bit bin
  /// must report the same dominant — the bin-center, not either of
  /// the source values. Bin 1 covers 8-bit [8, 15] and centers on
  /// `(1.5 * 8) = 12`. Pre-fix, both `[8,8,8]` and `[15,15,15]`
  /// reported `[8, 8, 8]` (lower-edge bias).
  #[test]
  fn bin_edge_inputs_collapse_to_bin_center() {
    for value in [8u8, 15u8] {
      let pixels = vec![[value; 3]; 16];
      let buf = make_frame(4, 4, &pixels);
      let frame = RgbFrame::try_new(&buf, 4, 4, 12).expect("frame");
      let dominants = quantize(frame.pixels(), 5, crate::Algorithm::default());
      let top = &dominants[0];
      assert_eq!(
        top.rgb,
        [12, 12, 12],
        "input [{value};3] should collapse to bin-center [12;3], got {:?}",
        top.rgb
      );
    }
  }

  #[test]
  fn checkerboard_red_blue_yields_two_dominants() {
    // 4x4 alternating red/blue. Should produce >=2 dominants.
    let red = [255, 0, 0];
    let blue = [0, 0, 255];
    let mut pixels = Vec::with_capacity(16);
    for i in 0..16 {
      pixels.push(if i % 2 == 0 { red } else { blue });
    }
    let buf = make_frame(4, 4, &pixels);
    let frame = RgbFrame::try_new(&buf, 4, 4, 12).expect("frame");
    let dominants = quantize(frame.pixels(), 5, crate::Algorithm::default());
    assert!(
      dominants.len() >= 2,
      "expected at least 2 dominants, got {}",
      dominants.len()
    );
    // Verify the dominant set covers both red and blue regions.
    let has_red = dominants.iter().any(|d| d.rgb[0] > 200 && d.rgb[2] < 50);
    let has_blue = dominants.iter().any(|d| d.rgb[2] > 200 && d.rgb[0] < 50);
    assert!(
      has_red && has_blue,
      "expected red AND blue dominants, got {:?}",
      dominants.iter().map(|d| d.rgb).collect::<Vec<_>>()
    );
  }

  #[test]
  fn padded_stride_is_respected() {
    // 2x2 frame with 8-byte stride (vs minimum 6 bytes per row).
    // Padding bytes should be ignored.
    let mut buf = Vec::new();
    // Row 0: red, red, then 2 bytes of garbage.
    buf.extend_from_slice(&[255, 0, 0, 255, 0, 0, 0xFF, 0xFF]);
    // Row 1: red, red, then 2 bytes of garbage.
    buf.extend_from_slice(&[255, 0, 0, 255, 0, 0, 0xFF, 0xFF]);
    let frame = RgbFrame::try_new(&buf, 2, 2, 8).expect("frame with padding");
    let dominants = quantize(frame.pixels(), 5, crate::Algorithm::default());
    let top = &dominants[0];
    // If padding leaked in, we'd see white-ish (255,255,255) dominate.
    // The stride-respecting path keeps it red.
    assert!(
      top.rgb[0] > 200 && top.rgb[1] < 30 && top.rgb[2] < 30,
      "padding leaked into the histogram; top dominant was {:?}",
      top.rgb
    );
  }

  /// Empty pixel iterator → empty histogram → `initial_vbox` returns
  /// `None` → `extract` returns immediately. Pins the early-return
  /// path that protects against zero-pixel frames slipping through.
  #[test]
  fn extract_with_empty_pixel_iterator_yields_nothing() {
    let mut mmcq = Mmcq::new_boxed();
    let mut out: Vec<crate::Dominant> = Vec::new();
    mmcq.extract(
      core::iter::empty::<[u8; 3]>(),
      5,
      crate::Algorithm::default(),
      &mut out,
    );
    assert!(out.is_empty(), "empty input must produce no dominants");
  }

  /// `count == 0` short-circuits before any work runs. Tests the
  /// direct `Mmcq::extract` path (the alloc-tier `extract` wrapper has
  /// its own count-zero test in `tests/extract.rs`).
  #[test]
  fn mmcq_extract_count_zero_does_no_work() {
    let mut mmcq = Mmcq::new_boxed();
    let buf = vec![255u8, 0, 0, 0, 255, 0, 0, 0, 255]; // 1x1, 3 bytes/pixel padded weird? actually 3x1
    let frame = RgbFrame::try_new(&buf, 3, 1, 9).expect("frame");
    let mut out: Vec<crate::Dominant> = Vec::new();
    mmcq.extract(frame.pixels(), 0, crate::Algorithm::default(), &mut out);
    assert!(out.is_empty(), "count=0 must produce no dominants");
  }

  /// Output buffer fills before MMCQ produces every requested dominant
  /// → the inner loop's `try_push` returns `Some(_)` and `extract`
  /// breaks, leaving the buffer at its capacity. Pins the back-pressure
  /// contract for the `Buffer` trait under fixed-size arrays.
  #[test]
  fn mmcq_extract_stops_when_output_buffer_full() {
    // 4x4 frame split half red / half blue → MMCQ produces ≥2 dominants.
    let pixels: Vec<[u8; 3]> = (0..16)
      .map(|i| if i < 8 { [255, 0, 0] } else { [0, 0, 255] })
      .collect();
    let buf = make_frame(4, 4, &pixels);
    let frame = RgbFrame::try_new(&buf, 4, 4, 12).expect("frame");

    // Buffer with capacity 1 — second push from MMCQ will be rejected.
    let mut out: [Option<crate::Dominant>; 1] = [const { None }; 1];
    let mut mmcq = Mmcq::new_boxed();
    mmcq.extract(frame.pixels(), 5, crate::Algorithm::default(), &mut out);

    assert!(out[0].is_some(), "first slot must be filled");
  }

  /// `Mmcq::new()` is the `const fn` placement constructor. Calling it
  /// from a runtime context (vs `static MMCQ: Mmcq = Mmcq::new()` which
  /// evaluates at const time and is invisible to LLVM coverage
  /// instrumentation) ensures the constructor body is measured. The
  /// 134 KB stack frame is the documented footgun this constructor
  /// has — `new_boxed` is the right call for non-static placement —
  /// but the test runner has plenty of stack budget.
  #[test]
  fn mmcq_new_const_is_callable() {
    static MMCQ: Mmcq = Mmcq::new();
    assert_eq!(MMCQ.histogram[0], 0);
    assert_eq!(MMCQ.boxes.len(), 0);

    // Force a runtime evaluation of `Mmcq::new()` so its body lights
    // up in coverage instrumentation. Box-and-drop keeps the 134 KB
    // alive only for the duration of this expression.
    let runtime = std::boxed::Box::new(Mmcq::new());
    assert_eq!(runtime.histogram[0], 0);
  }

  /// Reusing an `Mmcq` across two extract calls exercises the
  /// histogram reset and `boxes.clear()` paths at the entry of
  /// `extract`. The integration test `mmcq_reuse_resets_state_between_calls`
  /// covers the same property end-to-end through the alloc API; this
  /// inline version pins the direct `Mmcq::extract` path so the
  /// internal reset logic stays measured even if the integration test
  /// is removed.
  #[test]
  fn mmcq_reuse_via_direct_extract_resets_state() {
    let mut mmcq = Mmcq::new_boxed();
    let red_pixels = vec![[200u8, 30, 30]; 16];
    let blue_pixels = vec![[30u8, 30, 200]; 16];

    let red_buf = make_frame(4, 4, &red_pixels);
    let blue_buf = make_frame(4, 4, &blue_pixels);
    let red_frame = RgbFrame::try_new(&red_buf, 4, 4, 12).expect("frame");
    let blue_frame = RgbFrame::try_new(&blue_buf, 4, 4, 12).expect("frame");

    let mut red_out: Vec<crate::Dominant> = Vec::new();
    mmcq.extract(
      red_frame.pixels(),
      3,
      crate::Algorithm::default(),
      &mut red_out,
    );
    let mut blue_out: Vec<crate::Dominant> = Vec::new();
    mmcq.extract(
      blue_frame.pixels(),
      3,
      crate::Algorithm::default(),
      &mut blue_out,
    );

    let red_top = red_out[0].rgb;
    let blue_top = blue_out[0].rgb;
    assert!(red_top[0] > 100 && red_top[2] < 100);
    assert!(blue_top[2] > 100 && blue_top[0] < 100);
  }

  /// `BoxArena::push` returns `false` once the inline capacity is
  /// reached. The MMCQ pipeline never produces more than `MAX_BOXES`
  /// in practice (the iterate_split loop is bounded by `target ≤ 256`),
  /// so this is a defensive guard. Test it directly so the overflow
  /// branch is exercised.
  #[test]
  fn box_arena_push_rejects_when_full() {
    let mut arena = BoxArena::new();
    let template = VBox {
      r1: 0,
      r2: 0,
      g1: 0,
      g2: 0,
      b1: 0,
      b2: 0,
      count_cache: Some(1),
      avg_cache: Some([0, 0, 0]),
    };
    for _ in 0..MAX_BOXES {
      assert!(arena.push(template.clone()));
    }
    assert!(!arena.push(template), "push must reject when at capacity");
  }

  /// `VBox::avg` caches its result in `avg_cache` after the first
  /// call. Re-invoking returns the cached value without re-walking the
  /// histogram. Exercises the cache-hit early return.
  #[test]
  fn vbox_avg_cache_hit_returns_same_value() {
    let pixels = vec![[120u8, 80, 200]; 16];
    let buf = make_frame(4, 4, &pixels);
    let frame = RgbFrame::try_new(&buf, 4, 4, 12).expect("frame");
    let mut histo = [0u32; HISTO_SIZE];
    build_histogram(frame.pixels(), &mut histo);
    let mut vbox = initial_vbox(&histo).expect("non-empty histogram");
    let first = vbox.avg(&histo);
    let second = vbox.avg(&histo);
    assert_eq!(first, second, "avg must be stable across calls");
  }

  /// `VBox::avg` on an empty box returns the geometric center as the
  /// "no pixels" fallback. The MMCQ pipeline filters empty boxes out
  /// before they reach `avg`, but the fallback is reachable directly
  /// for callers that construct a `VBox` against a histogram that
  /// doesn't populate the box's range.
  #[test]
  fn vbox_avg_on_empty_box_returns_geometric_center() {
    let histo = [0u32; HISTO_SIZE]; // all-zero — every bin empty.
    let mut empty = VBox {
      r1: 0,
      r2: 7,
      g1: 0,
      g2: 7,
      b1: 0,
      b2: 7,
      count_cache: None,
      avg_cache: None,
    };
    let avg = empty.avg(&histo);
    // Center across (0..=7) on each axis: ((1<<RSHIFT) * (0+7+1)) / 2
    // = 8 * 8 / 2 = 32. Just pin "non-zero, deterministic, identical
    // on each channel" — the formula is internal and may be tuned.
    assert_eq!(avg[0], avg[1]);
    assert_eq!(avg[1], avg[2]);
    assert!(avg[0] > 0);
  }
}
