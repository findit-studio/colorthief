//! The `no_std + no_alloc` extraction pattern: caller-managed `Mmcq`
//! workspace in `static mut` + fixed-size `[Option<Dominant>; N]`
//! output buffer (no `Vec`, no heap).
//!
//! Run with: `cargo run --release --example extract_no_alloc -p colorthief`
//!
//! Demonstrated here under `std` for runnability — the same pattern
//! works under `--no-default-features` (no `std`, no `alloc`).
//! Memory layout:
//! - `MMCQ` lives in `.bss` (static): ~134 KB program-image cost,
//!   zero runtime allocation.
//! - `out` lives on the stack: `5 * sizeof(Option<Dominant>)` ≈ 80 B.
//! - The frame buffer here is also a stack array; on real targets the
//!   pixel buffer typically comes from a sink/decoder and is borrowed.

use colorthief::{Algorithm, Dominant, Mmcq, RgbFrame};

// Workspace placement: `static mut` (no_alloc) or `Mmcq::new_boxed()`
// (alloc — see `examples/extract.rs`).
static mut MMCQ: Mmcq = Mmcq::new();

const W: u32 = 16;
const H: u32 = 16;
const STRIDE: u32 = W * 3;

fn main() {
  // Synthetic 16×16 frame on the stack — 75% red, 25% blue.
  let mut buf = [0u8; (STRIDE * H) as usize];
  for row in 0..H as usize {
    let rgb = if row < 12 {
      [220, 30, 30]
    } else {
      [30, 30, 220]
    };
    for col in 0..W as usize {
      let off = row * STRIDE as usize + col * 3;
      buf[off..off + 3].copy_from_slice(&rgb);
    }
  }
  let frame = RgbFrame::try_new(&buf, W, H, STRIDE).expect("frame");

  // Output buffer: fixed-capacity, every slot starts as `None` and
  // `Mmcq::extract` fills the first N positions via the
  // `impl Buffer<Dominant> for [Option<Dominant>; N]` blanket.
  let mut out: [Option<Dominant>; 5] = [const { None }; 5];

  // SAFETY: this is the only access to MMCQ in this single-threaded
  // example. Real `no_std + alloc` consumers should keep the same
  // single-threaded discipline (typical wasm32-unknown-unknown,
  // interrupt-free bare metal); multi-threaded environments should
  // either provide their own synchronization or use the per-call
  // `Mmcq::new_boxed()` path instead.
  #[allow(static_mut_refs)]
  unsafe {
    (*core::ptr::addr_of_mut!(MMCQ)).extract(frame.pixels(), 5, Algorithm::default(), &mut out);
  }

  for slot in out.iter().flatten() {
    println!(
      "rgb={:?}  name={:?}  family={:?}  pop={}  ({:.1}%)",
      slot.rgb(),
      slot.color().name(),
      slot.color().family().as_str(),
      slot.population(),
      slot.percentage(),
    );
  }
}
