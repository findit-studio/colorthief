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

use colorthief::{RgbFrame, extract};

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

/// Regression for Codex adversarial review finding [medium]:
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

/// Regression for Codex adversarial-review round 5 finding [medium]:
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

/// Regression for Codex adversarial-review round 2 finding [high]:
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

/// Regression for Codex adversarial review finding [high] (round 1):
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
