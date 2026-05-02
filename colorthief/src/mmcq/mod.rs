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

use crate::RgbFrame;

mod simd;

const SIGBITS: u32 = 5;
const RSHIFT: u32 = 8 - SIGBITS;
const HISTO_LEVELS: usize = 1 << SIGBITS; // 32
const HISTO_SIZE: usize = 1 << (3 * SIGBITS); // 32768
const MAX_ITERATIONS: usize = 1000;
const FRACT_BY_POPULATIONS: f64 = 0.75;

/// One dominant returned by [`quantize`]. `population` is the number
/// of source pixels assigned to this box; the public `extract` wraps
/// this into [`crate::Dominant`] alongside the named [`Color`] match.
pub(crate) struct Dominant {
  pub rgb: [u8; 3],
  pub population: u32,
}

/// Encode a 5-bit (R, G, B) coord into a flat histogram index.
#[inline]
fn histo_index(r: u32, g: u32, b: u32) -> usize {
  ((r << (2 * SIGBITS)) + (g << SIGBITS) + b) as usize
}

/// A 3-D bounding box in 5-bit RGB space. Bounds are inclusive
/// (`r1..=r2`). Pixel count + average color are cached on first access.
#[derive(Clone)]
struct VBox {
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
  /// 8-bit RGB. Mirrors color-thief's TS `avg()` byte-for-byte: each
  /// bin contributes `pop * (idx + 0.5) * 2^RSHIFT`, then we round
  /// the per-channel mean down (`as u8` truncates positive floats).
  fn avg(&mut self, histo: &[u32]) -> [u8; 3] {
    if let Some(a) = self.avg_cache {
      return a;
    }
    let mult = 1u32 << RSHIFT;

    if self.r1 == self.r2 && self.g1 == self.g2 && self.b1 == self.b2 {
      let out = [
        (self.r1 << RSHIFT) as u8,
        (self.g1 << RSHIFT) as u8,
        (self.b1 << RSHIFT) as u8,
      ];
      self.avg_cache = Some(out);
      return out;
    }

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

/// Build the 32K-entry histogram from a frame's pixels.
fn build_histogram(frame: RgbFrame<'_>) -> Vec<u32> {
  let mut histo = vec![0u32; HISTO_SIZE];
  for [r, g, b] in frame.pixels() {
    let rv = (r as u32) >> RSHIFT;
    let gv = (g as u32) >> RSHIFT;
    let bv = (b as u32) >> RSHIFT;
    histo[histo_index(rv, gv, bv)] = histo[histo_index(rv, gv, bv)].saturating_add(1);
  }
  histo
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
fn iterate_split<F>(boxes: &mut Vec<VBox>, target: usize, histo: &[u32], score: F)
where
  F: Fn(&mut VBox, &[u32]) -> u64,
{
  let mut iters = 0;
  while boxes.len() < target && iters < MAX_ITERATIONS {
    iters += 1;
    boxes.sort_by_key(|b| {
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
      // No populated boxes left.
      boxes.push(top);
      break;
    }
    match median_cut(&top, histo) {
      Some((left, Some(right))) => {
        boxes.push(left);
        boxes.push(right);
      }
      Some((only, None)) => {
        // Box can't split further; keep it and stop iterating
        // (no other box is bigger by the current ordering).
        boxes.push(only);
        break;
      }
      None => break,
    }
  }
}

/// Run MMCQ on `frame` and return up to `max_colors` dominants.
pub(crate) fn quantize(frame: RgbFrame<'_>, max_colors: u8) -> Vec<Dominant> {
  // MMCQ is undefined outside [2, 256]. Saturate to that range.
  let target = (max_colors as usize).clamp(2, 256);

  let histo = build_histogram(frame);
  let initial = match initial_vbox(&histo) {
    Some(b) => b,
    None => return Vec::new(),
  };

  let mut boxes = vec![initial];

  // Phase 1: split by raw population.
  let phase1_target = ((FRACT_BY_POPULATIONS * target as f64) as usize).max(1);
  iterate_split(&mut boxes, phase1_target, &histo, |b, h| b.count(h) as u64);

  // Phase 2: split by population * volume, finishing the split tree.
  iterate_split(&mut boxes, target, &histo, |b, h| {
    (b.count(h) as u64) * (b.volume() as u64)
  });

  // Sort the final palette descending by population so the caller gets
  // the most-dominant color first.
  boxes.sort_by_key(|b| {
    let mut probe = b.clone();
    core::cmp::Reverse(probe.count(&histo))
  });

  // Filter zero-population boxes. `median_cut` can split a sparsely-
  // populated parent into a (populated, empty) pair when the cut
  // axis only has one occupied slice, and `iterate_split` accepts
  // both halves. Without this filter the empty box's `avg()` would
  // fall through to the geometric-center fallback, fabricating a
  // dominant color that never appeared in the frame. (Codex
  // adversarial review, 2026-05-02.)
  boxes
    .into_iter()
    .filter_map(|mut b| {
      let pop = b.count(&histo);
      if pop == 0 {
        None
      } else {
        Some(Dominant {
          rgb: b.avg(&histo),
          population: pop,
        })
      }
    })
    .collect()
}

#[cfg(test)]
mod tests {
  use super::*;

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
    let dominants = quantize(frame, 5);
    assert!(!dominants.is_empty(), "MMCQ produced zero dominants");
    let top = &dominants[0];
    // The 5-bit quantization shifts pure red (255,0,0) → (31,0,0), and
    // avg() maps it back via `bin << 3 = 248`. So the recovered red is
    // 248-ish, not 255 exactly. Still squarely in the red region.
    assert!(top.rgb[0] > 200, "expected R>200, got {:?}", top.rgb);
    assert!(top.rgb[1] < 30, "expected G<30, got {:?}", top.rgb);
    assert!(top.rgb[2] < 30, "expected B<30, got {:?}", top.rgb);
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
    let dominants = quantize(frame, 5);
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
    let dominants = quantize(frame, 5);
    let top = &dominants[0];
    // If padding leaked in, we'd see white-ish (255,255,255) dominate.
    // The stride-respecting path keeps it red.
    assert!(
      top.rgb[0] > 200 && top.rgb[1] < 30 && top.rgb[2] < 30,
      "padding leaked into the histogram; top dominant was {:?}",
      top.rgb
    );
  }
}
