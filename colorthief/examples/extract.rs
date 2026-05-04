//! Extract dominant colors from a synthetic frame and print each
//! with its human-vocabulary name.
//!
//! Run with: `cargo run --release --example extract -p colorthief`
//!
//! Demonstrates both the default extraction (CIEDE2000Exact via the
//! 32³ LUT) and explicit-algorithm selection via `extract_with`.

use colorthief::{Algorithm, RgbFrame, extract, extract_with};

fn main() {
  // Synthetic 32×32 frame: 75% red, 25% blue. Stride is `3 * width`
  // (no row padding).
  let mut buf = Vec::with_capacity(32 * 32 * 3);
  for row in 0..32 {
    for _ in 0..32 {
      let rgb = if row < 24 {
        [220, 30, 30]
      } else {
        [30, 30, 220]
      };
      buf.extend_from_slice(&rgb);
    }
  }
  let frame = RgbFrame::try_new(&buf, 32, 32, 96).expect("frame");

  println!("== default (CIEDE2000Exact) ==");
  for d in extract(frame, 5) {
    println!(
      "  rgb={:?}  name={:?}  family={:?}  pop={}",
      d.rgb(),
      d.color().name(),
      d.color().family().as_str(),
      d.population(),
    );
  }

  println!("\n== Algorithm::DeltaE76 (fastest) ==");
  for d in extract_with(frame, 5, Algorithm::DeltaE76) {
    println!(
      "  rgb={:?}  name={:?}  pop={}",
      d.rgb(),
      d.color().name(),
      d.population(),
    );
  }
}
