# colorthief

Dominant-color extraction (MMCQ) and human-vocabulary naming for packed-RGB
video keyframes — CIEDE2000 (default), CIE94, or Delta E 76 nearest-neighbor
against the [xkcd color hierarchy].

[xkcd color hierarchy]: https://blog.xkcd.com/2010/05/03/color-survey-results/

## Pipeline

1. [`extract`] runs MMCQ (Modified Median Cut Quantization) over the pixels of
   an [`RgbFrame`] (8-bit) or [`Rgb48Frame`] (16-bit per channel HDR), producing
   up to `count` dominant RGB values plus the pixel population behind each.
2. Each dominant is mapped to its nearest entry in the
   [`colorthief-dataset`](https://crates.io/crates/colorthief-dataset)
   xkcd-hierarchy table via the algorithm chosen by [`Algorithm`] (default
   [`Algorithm::Ciede2000Exact`], the modern perceptual gold-standard).

## Feature flags

| Feature | Description |
|---|---|
| `std` (default) | `thread_local!`-cached MMCQ workspace — zero-alloc-per-call after the first call per thread. Implies `alloc`. |
| `alloc` | Heap allocator available; enables `Vec<Dominant>`-returning convenience APIs and `Mmcq::new_boxed()`. |
| `single-threaded` | `OnceCell + AssumeSync`-cached static workspace for `no_std + alloc` consumers who can guarantee single-threaded access (typical wasm32-unknown-unknown / interrupt-free bare metal). Opt-in only. |
| `lut` (default) | 32³ candidate-set LUT for CIEDE2000 — ~256 KB binary cost for ~300× speedup vs full scan. |

## No-std support

`colorthief` works under `no_std + no_alloc` with caller-supplied workspace and
output buffer:

```rust,ignore
use colorthief::{Algorithm, Buffer, Dominant, Mmcq, RgbFrame};

// Workspace placement: `static` (no_alloc) or `Box::new_boxed()` (alloc).
static mut MMCQ: Mmcq = Mmcq::new();

let frame = RgbFrame::try_new(rgb_bytes, width, height, stride)?;
let mut out: [Option<Dominant>; 5] = [const { None }; 5];

// SAFETY: caller guarantees single-threaded access to `MMCQ`.
unsafe {
    (*core::ptr::addr_of_mut!(MMCQ)).extract(
        frame.pixels(), 5, Algorithm::default(), &mut out,
    );
}
```

For consumers with an allocator but `no_std`, `Mmcq::new_boxed()` avoids the
134 KB stack frame `Mmcq::new()` would otherwise produce.

## License

Dual-licensed under MIT or Apache-2.0.
