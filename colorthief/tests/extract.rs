//! End-to-end integration tests for [`colorthief::extract`].
//!
//! Synthetic-frame coverage that crosses both pipeline stages:
//!   pixels → MMCQ dominants → CIEDE76 nearest xkcd entry.
//! The `mmcq::tests` unit tests cover MMCQ in isolation; the
//! `colorthief_dataset::tests` cover NN lookup. These tests pin the
//! composition.
//!
//! Asserting on family / hue rather than exact xkcd names keeps the
//! tests robust across xkcd-dataset regenerations.

use colorthief::{
  Algorithm, Dominant, Mmcq, Rgb48Frame, RgbFrame, RgbFrameError, extract, extract_rgb48,
};

fn solid_color_frame(rgb: [u8; 3], width: u32, height: u32) -> Vec<u8> {
  let mut buf = Vec::with_capacity((width * height) as usize * 3);
  for _ in 0..width * height {
    buf.extend_from_slice(&rgb);
  }
  buf
}

#[test]
fn extract_on_solid_red_returns_a_red_named_color() {
  let buf = solid_color_frame([255, 0, 0], 8, 8);
  let frame = RgbFrame::try_new(&buf, 8, 8, 24).expect("frame");
  let dominants = extract(frame, 5);
  assert!(!dominants.is_empty(), "expected at least one dominant");
  let top = dominants[0];
  assert!(
    top.color.family().as_str().contains("red") || top.color.name().contains("red"),
    "top dominant on solid red was rgb={:?} name={:?} family={:?}",
    top.rgb,
    top.color.name(),
    top.color.family().as_str(),
  );
  assert!(top.population > 0, "population must be non-zero");
}

#[test]
fn extract_on_solid_blue_returns_a_blue_named_color() {
  let buf = solid_color_frame([0, 0, 255], 8, 8);
  let frame = RgbFrame::try_new(&buf, 8, 8, 24).expect("frame");
  let dominants = extract(frame, 5);
  assert!(!dominants.is_empty());
  let top = dominants[0];
  assert!(
    top.color.family().as_str().contains("blue") || top.color.name().contains("blue"),
    "top dominant on solid blue was rgb={:?} name={:?} family={:?}",
    top.rgb,
    top.color.name(),
    top.color.family().as_str(),
  );
}

#[test]
fn extract_count_zero_returns_empty() {
  let buf = solid_color_frame([128, 128, 128], 4, 4);
  let frame = RgbFrame::try_new(&buf, 4, 4, 12).expect("frame");
  let dominants = extract(frame, 0);
  assert!(dominants.is_empty());
}

#[test]
fn extract_on_red_blue_split_recovers_both_hues() {
  // 8x8 frame split half red / half blue (rows 0-3 red, rows 4-7 blue).
  // The dominant set should cover both regions.
  let mut buf = Vec::with_capacity(64 * 3);
  for row in 0..8 {
    for _ in 0..8 {
      let rgb = if row < 4 { [255, 0, 0] } else { [0, 0, 255] };
      buf.extend_from_slice(&rgb);
    }
  }
  let frame = RgbFrame::try_new(&buf, 8, 8, 24).expect("frame");
  let dominants = extract(frame, 5);
  assert!(dominants.len() >= 2);
  let has_red = dominants
    .iter()
    .any(|d| d.color.family().as_str().contains("red") || d.color.name().contains("red"));
  let has_blue = dominants
    .iter()
    .any(|d| d.color.family().as_str().contains("blue") || d.color.name().contains("blue"));
  assert!(
    has_red && has_blue,
    "expected red and blue named entries, got: {:?}",
    dominants
      .iter()
      .map(|d| (d.color.name(), d.rgb, d.population))
      .collect::<Vec<_>>()
  );
}

/// Pins the population-descending sort on the public API.
#[test]
fn extract_dominants_sorted_by_population_descending() {
  // 8x8 frame: 75% red, 25% blue. Red must come first.
  let mut buf = Vec::with_capacity(64 * 3);
  for row in 0..8 {
    for _ in 0..8 {
      let rgb = if row < 6 { [255, 0, 0] } else { [0, 0, 255] };
      buf.extend_from_slice(&rgb);
    }
  }
  let frame = RgbFrame::try_new(&buf, 8, 8, 24).expect("frame");
  let dominants = extract(frame, 5);
  assert!(dominants.len() >= 2);
  for window in dominants.windows(2) {
    assert!(
      window[0].population >= window[1].population,
      "dominants must be sorted by descending population: {:?}",
      dominants
        .iter()
        .map(|d| (d.color.name(), d.population))
        .collect::<Vec<_>>()
    );
  }
  let top = dominants[0];
  assert!(
    top.color.family().as_str().contains("red") || top.color.name().contains("red"),
    "75%-red frame: top dominant should be red, got {:?}",
    top.color.name()
  );
}

/// `extract(frame, 1)` was returning two entries because MMCQ's
/// internal `target` clamps to ≥2. The public `count` must be a
/// hard upper bound.
#[test]
fn extract_count_one_returns_at_most_one() {
  // 4x4 half-red / half-blue — at least 2 distinct populated bins,
  // so prior to the fix MMCQ would emit 2 dominants.
  let mut buf = Vec::with_capacity(16 * 3);
  for i in 0..16 {
    let rgb = if i < 8 { [255, 0, 0] } else { [0, 0, 255] };
    buf.extend_from_slice(&rgb);
  }
  let frame = RgbFrame::try_new(&buf, 4, 4, 12).expect("frame");
  let dominants = extract(frame, 1);
  assert!(
    dominants.len() <= 1,
    "extract(_, 1) must return at most 1 entry, got {}: {:?}",
    dominants.len(),
    dominants.iter().map(|d| d.rgb).collect::<Vec<_>>(),
  );
}

/// `phase1_target = (0.75 * count) as usize` truncated toward zero,
/// so for `count` values with a fractional `0.75 * count` (3, 5, 6,
/// 7, 9, …) phase 1 ended one box early vs. the TS reference's
/// effective ceil semantics. That ceded a split to phase 2's
/// `population * volume` scoring, which can pick a different box
/// than phase 1's pure-population scoring. Without an external
/// reference oracle to assert against, the regression here pins the
/// count-exactness invariant at both fractional boundaries.
///
/// Helper: build an `n × n` frame whose `count` palette colors are
/// each used in equal-area stripes.
fn striped_palette_frame(palette: &[[u8; 3]], width: u32, height: u32) -> Vec<u8> {
  let total = (width * height) as usize;
  let mut buf = Vec::with_capacity(total * 3);
  for i in 0..total {
    buf.extend_from_slice(&palette[i % palette.len()]);
  }
  buf
}

#[test]
fn extract_count_3_returns_full_count() {
  // 3-color palette, count=3 → phase1 target = ceil(2.25) = 3.
  // All splits decided by population scoring; phase 2 is a no-op.
  let palette = [[200u8, 30, 30], [30, 200, 30], [30, 30, 200]];
  let buf = striped_palette_frame(&palette, 8, 8);
  let frame = RgbFrame::try_new(&buf, 8, 8, 24).expect("frame");
  let dominants = extract(frame, 3);
  assert_eq!(dominants.len(), 3);
  for d in &dominants {
    assert!(d.population > 0);
  }
}

#[test]
fn extract_count_7_returns_full_count() {
  // 7-color palette, count=7 → phase1 target = ceil(5.25) = 6.
  // Phase 1 produces 6 boxes via population scoring; phase 2 adds
  // the 7th via population*volume.
  let palette = [
    [200u8, 30, 30],
    [30, 200, 30],
    [30, 30, 200],
    [200, 200, 30],
    [30, 200, 200],
    [200, 30, 200],
    [128, 128, 128],
  ];
  let buf = striped_palette_frame(&palette, 8, 8);
  let frame = RgbFrame::try_new(&buf, 8, 8, 24).expect("frame");
  let dominants = extract(frame, 7);
  assert_eq!(dominants.len(), 7);
  for d in &dominants {
    assert!(d.population > 0);
  }
}

/// `iterate_split` was counting empty children of a sparse box's
/// median-cut split toward `target`, so when the frame had `count`
/// distinct colors but one of them sat in a wide-empty-range parent,
/// the loop would terminate with an empty half consuming a target
/// slot. The post-quantize zero-population filter then dropped that
/// empty box, and `extract` underfilled (returned `< count` even
/// though `count` real distinct colors were available).
///
/// Construct a frame with 5 distinct quantized colors arranged so
/// one of them is widely separated on the R axis from the others —
/// MMCQ's first split on R will produce a populated/empty pair for
/// the high-R color's parent.
#[test]
fn extract_with_count_distinct_colors_returns_full_count() {
  // Five distinct colors; the last one sits at R=255 while the rest
  // cluster near R=0..32, so the first R-axis split has a wide
  // empty range on the right of the populated low-R cluster.
  let palette: [[u8; 3]; 5] = [
    [10, 200, 10],  // green-ish, low R
    [10, 10, 200],  // blue-ish, low R
    [200, 10, 200], // magenta-ish, mid R
    [10, 200, 200], // cyan-ish, low R
    [255, 10, 10],  // red, high R — the sparse one
  ];
  // 8x8 frame, equal-area distribution of the 5 colors (with 4
  // remainder pixels reused on the first color so totals are
  // balanced enough that all five survive Phase 1's count-only
  // priority).
  let mut buf = Vec::with_capacity(64 * 3);
  for i in 0..64 {
    let rgb = palette[i % 5];
    buf.extend_from_slice(&rgb);
  }
  let frame = RgbFrame::try_new(&buf, 8, 8, 24).expect("frame");
  let dominants = extract(frame, 5);
  assert_eq!(
    dominants.len(),
    5,
    "5 distinct colors but extract returned {} dominants: {:?}",
    dominants.len(),
    dominants
      .iter()
      .map(|d| (d.rgb, d.population))
      .collect::<Vec<_>>(),
  );
  for d in &dominants {
    assert!(
      d.population > 0,
      "zero-population dominant: rgb={:?}",
      d.rgb
    );
  }
}

/// `median_cut` could produce a (populated, empty) split that
/// `iterate_split` accepted, and `quantize` then mapped the empty
/// box's geometric-center fallback to a fabricated `Dominant`. With
/// fewer distinct colors than `count`, the result must contain only
/// real (population > 0) entries.
#[test]
fn extract_no_zero_population_dominants_below_distinct_color_floor() {
  // 8x8 frame with only 2 populated bins (pure red, pure blue).
  // Request 5 dominants — MMCQ will fail to split productively
  // beyond 2 boxes; pre-fix, the surplus 3 boxes were zero-population
  // entries with fabricated geometric-center colors.
  let mut buf = Vec::with_capacity(64 * 3);
  for i in 0..64 {
    let rgb = if i < 32 { [255, 0, 0] } else { [0, 0, 255] };
    buf.extend_from_slice(&rgb);
  }
  let frame = RgbFrame::try_new(&buf, 8, 8, 24).expect("frame");
  let dominants = extract(frame, 5);
  assert!(
    dominants.len() <= 2,
    "frame has 2 distinct colors but extract returned {} dominants: {:?}",
    dominants.len(),
    dominants
      .iter()
      .map(|d| (d.rgb, d.population))
      .collect::<Vec<_>>(),
  );
  for d in &dominants {
    assert!(
      d.population > 0,
      "zero-population dominant in result: rgb={:?} name={:?}",
      d.rgb,
      d.color.name(),
    );
  }
}

// ---------------------------------------------------------------------
// Rgb48Frame (16-bit-per-channel) integration tests
// ---------------------------------------------------------------------

/// Build a u16 RGB plane filled with one repeated pixel. Stride
/// equals `3 * width` u16 elements (no padding).
fn solid_color_frame_u16(rgb: [u16; 3], width: u32, height: u32) -> Vec<u16> {
  let mut buf = Vec::with_capacity((width * height) as usize * 3);
  for _ in 0..width * height {
    buf.extend_from_slice(&rgb);
  }
  buf
}

/// Smoke test: pure red u16 input must reach a red-family dominant
/// after the `>> 8` downscale + MMCQ + naming pipeline.
#[test]
fn extract_rgb48_on_solid_red_returns_a_red_named_color() {
  let buf = solid_color_frame_u16([0xFF00, 0x0000, 0x0000], 8, 8);
  let frame = Rgb48Frame::try_new(&buf, 8, 8, 24).expect("frame");
  let dominants = extract_rgb48(frame, 5);
  assert!(!dominants.is_empty(), "expected at least one dominant");
  let top = dominants[0];
  assert!(
    top.color.family().as_str().contains("red") || top.color.name().contains("red"),
    "top dominant on solid red u16 was rgb={:?} name={:?} family={:?}",
    top.rgb,
    top.color.name(),
    top.color.family().as_str(),
  );
}

/// Equivalence: `extract_rgb48(u16-widened-from-u8)` must produce
/// exactly the same dominants as `extract(u8)`. The downscale
/// `(u8 << 8) >> 8 == u8` is bit-exact, so MMCQ + naming sees
/// identical data on both paths.
#[test]
fn extract_rgb48_widened_matches_extract_u8() {
  // Mixed-color frame to exercise more than one MMCQ box.
  let mut buf_u8 = Vec::with_capacity(64 * 3);
  for i in 0..64 {
    let rgb = match i % 4 {
      0 => [200, 30, 30],
      1 => [30, 30, 200],
      2 => [30, 200, 30],
      _ => [180, 180, 180],
    };
    buf_u8.extend_from_slice(&rgb);
  }
  let buf_u16: Vec<u16> = buf_u8.iter().map(|&b| (b as u16) << 8).collect();

  let frame_u8 = RgbFrame::try_new(&buf_u8, 8, 8, 24).expect("u8 frame");
  let frame_u16 = Rgb48Frame::try_new(&buf_u16, 8, 8, 24).expect("u16 frame");

  let d_u8 = extract(frame_u8, 5);
  let d_u16 = extract_rgb48(frame_u16, 5);

  assert_eq!(d_u8.len(), d_u16.len(), "dominant counts must match");
  for (a, b) in d_u8.iter().zip(d_u16.iter()) {
    assert_eq!(
      a.rgb, b.rgb,
      "u8 and u16-widened paths produced different RGBs"
    );
    assert_eq!(a.population, b.population, "populations diverged");
    assert_eq!(
      a.color.name(),
      b.color.name(),
      "named colors diverged: u8={} u16={}",
      a.color.name(),
      b.color.name(),
    );
  }
}

/// `try_new` rejects zero-dimension frames.
#[test]
fn rgb48_try_new_rejects_zero_dimension() {
  let buf = vec![0u16; 12];
  let err = Rgb48Frame::try_new(&buf, 0, 4, 12).unwrap_err();
  assert!(matches!(err, RgbFrameError::ZeroDimension { .. }));
}

/// `try_new` rejects stride < 3 * width (in u16 elements).
#[test]
fn rgb48_try_new_rejects_stride_too_small() {
  let buf = vec![0u16; 16];
  // width = 4 → min_stride = 12 u16 elements; supply 8.
  let err = Rgb48Frame::try_new(&buf, 4, 2, 8).unwrap_err();
  assert!(
    matches!(
      err,
      RgbFrameError::StrideTooSmall {
        min_stride: 12,
        stride: 8
      }
    ),
    "expected StrideTooSmall, got {err:?}",
  );
}

/// `try_new` rejects buffers shorter than `stride * height` u16
/// elements.
#[test]
fn rgb48_try_new_rejects_plane_too_short() {
  // width = 4, height = 4, stride = 12 → need 48 u16 elements; supply 30.
  let buf = vec![0u16; 30];
  let err = Rgb48Frame::try_new(&buf, 4, 4, 12).unwrap_err();
  assert!(
    matches!(
      err,
      RgbFrameError::PlaneTooShort {
        expected: 48,
        actual: 30
      }
    ),
    "expected PlaneTooShort, got {err:?}",
  );
}

/// `extract_rgb48(_, 0)` returns an empty Vec — same contract as
/// [`extract`].
#[test]
fn extract_rgb48_count_zero_returns_empty() {
  let buf = solid_color_frame_u16([0x8000, 0x4000, 0x2000], 4, 4);
  let frame = Rgb48Frame::try_new(&buf, 4, 4, 12).expect("frame");
  let dominants = extract_rgb48(frame, 0);
  assert!(dominants.is_empty());
}

// ---------------------------------------------------------------------
// no_alloc-path API: Mmcq::extract with a Buffer (fixed-size array).
// ---------------------------------------------------------------------

/// `Mmcq::extract` writing into a `[Option<Dominant>; N]` buffer —
/// the canonical no_alloc usage. Exercises the same pipeline as
/// `extract` but without the convenience `Vec` wrapper.
#[test]
fn mmcq_extract_into_array_buffer_recovers_red() {
  let buf = solid_color_frame([200, 50, 50], 8, 8);
  let frame = RgbFrame::try_new(&buf, 8, 8, 24).expect("frame");

  let mut mmcq = Mmcq::new_boxed();
  let mut out: [Option<Dominant>; 5] = [const { None }; 5];
  mmcq.extract(frame.pixels(), 5, Algorithm::default(), &mut out);

  let first = out
    .iter()
    .find_map(|o| o.as_ref())
    .expect("expected at least one dominant");
  assert!(
    first.color.family().as_str().contains("red") || first.color.name().contains("red"),
    "expected red-family dominant, got {:?}",
    first.color.name(),
  );
  assert!(first.population > 0);
}

/// `Mmcq` reuse across calls — verify the workspace can be reused
/// without leaking state from previous calls. Critical for the
/// thread_local-cache pattern.
#[test]
fn mmcq_reuse_resets_state_between_calls() {
  let mut mmcq = Mmcq::new_boxed();

  // First call: red frame.
  let red_buf = solid_color_frame([220, 20, 20], 8, 8);
  let red_frame = RgbFrame::try_new(&red_buf, 8, 8, 24).expect("frame");
  let mut red_out: [Option<Dominant>; 3] = [const { None }; 3];
  mmcq.extract(red_frame.pixels(), 3, Algorithm::default(), &mut red_out);
  let red_first = red_out
    .iter()
    .find_map(|o| o.as_ref())
    .expect("red dominant");
  assert!(
    red_first.color.family().as_str().contains("red") || red_first.color.name().contains("red")
  );

  // Second call: blue frame. Same Mmcq, must NOT remember red state.
  let blue_buf = solid_color_frame([20, 20, 220], 8, 8);
  let blue_frame = RgbFrame::try_new(&blue_buf, 8, 8, 24).expect("frame");
  let mut blue_out: [Option<Dominant>; 3] = [const { None }; 3];
  mmcq.extract(blue_frame.pixels(), 3, Algorithm::default(), &mut blue_out);
  let blue_first = blue_out
    .iter()
    .find_map(|o| o.as_ref())
    .expect("blue dominant");
  assert!(
    blue_first.color.family().as_str().contains("blue") || blue_first.color.name().contains("blue"),
    "second-call dominant should be blue, got {:?}",
    blue_first.color.name(),
  );
}
