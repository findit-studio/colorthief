//! Extract dominant colors from a synthetic 16-bit-per-channel HDR
//! frame.
//!
//! Run with: `cargo run --release --example extract_rgb48 -p colorthief`
//!
//! `Rgb48Frame` mirrors the buffer layout colconv's
//! `MixedSinker::with_rgb_u16` writes to: `&[u16]` of length
//! `3 * width * height`, R/G/B interleaved per pixel, stride in u16
//! elements. Each channel is downscaled to u8 via `>> 8` at pixel
//! iteration before MMCQ.

use colorthief::{Rgb48Frame, extract_rgb48};

fn main() {
  // Synthetic 32×32 HDR frame: 50% green / 50% magenta in u16 sRGB
  // (top 8 bits carry the equivalent u8 value; bottom 8 bits zero).
  let mut buf = Vec::with_capacity(32 * 32 * 3);
  for row in 0..32 {
    for _ in 0..32 {
      let rgb = if row < 16 {
        [0x2000_u16, 0xC000, 0x2000] // muted green
      } else {
        [0xC000, 0x2000, 0xC000] // muted magenta
      };
      buf.extend_from_slice(&rgb);
    }
  }
  // Stride is in u16 elements (>= 3 * width), matching colconv's API.
  let frame = Rgb48Frame::try_new(&buf, 32, 32, 96).expect("frame");

  for d in extract_rgb48(frame, 5) {
    println!(
      "rgb={:?}  name={:?}  family={:?}  pop={}",
      d.rgb(),
      d.color().name(),
      d.color().family().as_str(),
      d.population(),
    );
  }
}
