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
  let colors = extract(&frame, 5);
  assert!(!colors.is_empty(), "expected at least one named color");
  let top = colors[0];
  assert!(
    top.family().contains("red") || top.name().contains("red"),
    "top dominant on solid red was name={:?} family={:?}",
    top.name(),
    top.family(),
  );
}

#[test]
fn extract_on_solid_blue_returns_a_blue_named_color() {
  let buf = solid_color_frame([0, 0, 255], 8, 8);
  let frame = RgbFrame::try_new(&buf, 8, 8, 24).expect("frame");
  let colors = extract(&frame, 5);
  assert!(!colors.is_empty());
  let top = colors[0];
  assert!(
    top.family().contains("blue") || top.name().contains("blue"),
    "top dominant on solid blue was name={:?} family={:?}",
    top.name(),
    top.family(),
  );
}

#[test]
fn extract_count_zero_returns_empty() {
  let buf = solid_color_frame([128, 128, 128], 4, 4);
  let frame = RgbFrame::try_new(&buf, 4, 4, 12).expect("frame");
  let colors = extract(&frame, 0);
  assert!(colors.is_empty());
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
  let colors = extract(&frame, 5);
  assert!(colors.len() >= 2);
  let has_red = colors
    .iter()
    .any(|c| c.family().contains("red") || c.name().contains("red"));
  let has_blue = colors
    .iter()
    .any(|c| c.family().contains("blue") || c.name().contains("blue"));
  assert!(
    has_red && has_blue,
    "expected red and blue named entries, got: {:?}",
    colors.iter().map(|c| c.name()).collect::<Vec<_>>()
  );
}
