# 0.1.0 (2026-05-04)

Initial release. Two crates: `colorthief` (dominant-color extraction
with human-vocabulary naming) and `colorthief-dataset` (the static
xkcd palette + nearest-neighbor lookup), plus a build-time `xtask`
codegen tool.

## `colorthief-dataset`

Static xkcd color-hierarchy table (949 named colors, sourced from
Stitch Fix's `colornamer`, Apache-2.0). Pre-computed CIE LAB at
codegen time; runtime is `no_std + no_alloc`, every entry
`&'static`.

- Three color-difference algorithms behind a `#[non_exhaustive] #[repr(u8)]`
  `Algorithm` enum:
  - `DeltaE76` — squared Euclidean LAB
  - `Cie94` — graphic-arts weighting (asymmetric)
  - `Ciede2000` / `Ciede2000Exact` — modern perceptual gold standard
    (default)
- SIMD backends for `DeltaE76` and `Cie94`, bit-identical to the
  scalar reference and verified across 17³ inline + 256³ exhaustive
  parity sweeps:
  - `aarch64`: NEON (gated on `target_feature = "neon"` so
    `aarch64-unknown-none-softfloat` falls through to scalar)
  - `x86_64`: SSE4.1 / AVX2 / AVX-512F runtime feature-detected
  - `wasm32`: SIMD128 (gated on `target_feature = "simd128"`)
- **CIEDE2000 candidate-set LUT** (`feature = "lut"`, default on).
  32³ cells × ~2.5-avg / 10-max candidates per cell, pre-computed at
  xtask time by sweeping every u8 RGB through the full-scan
  reference. ~230 ns/query vs 71.5 µs full scan (~310× speedup),
  provably exact at u8 RGB resolution.
- Pre-computed entry chroma `LABS_C` saves one `sqrtf` per pair in
  the CIE94 and CIEDE2000 inner loops.
- Coverage-friendly tier-forcing cfgs:
  `colorthief_force_scalar`, `colorthief_disable_avx512`,
  `colorthief_disable_avx2`.
- Property-based tests (`proptest`) for metric invariants
  (self-distance = 0, non-negative, symmetric, idempotent under
  palette LAB).

## `colorthief`

- `RgbFrame<'a>` — packed 8-bit sRGB, mirrors `colconv::Rgb24Frame`
  field-for-field (no `colconv` dep — license firewall keeps the
  workspace MIT/Apache).
- `Rgb48Frame<'a>` — packed 16-bit-per-channel sRGB for HDR sources;
  mirrors `colconv::sinker::MixedSinker::with_rgb_u16`. Per-channel
  `>> 8` downscale at pixel iteration preserves the LUT correctness
  guarantee.
- `extract` / `extract_with` and `extract_rgb48` /
  `extract_rgb48_with` (alloc-gated) — return `Vec<Dominant>` with
  per-tier workspace caching.
- `Mmcq` — caller-owned MMCQ workspace, 128 KB inline histogram +
  256-VBox arena, no `Vec` internally. `Mmcq::new()` is `const fn`
  for `static mut` placement; `Mmcq::new_boxed()` (alloc) avoids
  the 134 KB stack frame.
- `Buffer<T>` trait — `try_push(&mut self, val: T) -> Option<T>`
  (returns `Some(val)` on overflow). Default impls for `Vec<T>`
  (alloc), `[Option<T>; N]`, `&mut [Option<T>]`; consumers can plug
  in `arrayvec::ArrayVec` / `heapless::Vec` / custom types with a
  one-line `impl`.
- Three-tier feature design:
  - `std` (default): `thread_local!`-cached `Mmcq` — zero-alloc-per-
    call after first call per thread. Sound under any threading
    model (per-thread isolation).
  - `alloc`: `Mmcq::new_boxed()` per call (~134 KB heap each call).
    Stateless, sound under any threading model.
  - no features (`no_std + no_alloc`): caller-managed workspace via
    `Mmcq::extract` directly. The `unsafe` for global state lives
    at the user's call site, not silently inside this crate.
- MMCQ port of color-thief's TS reference, producing up to `count`
  dominants sorted descending by population. SIMD-accelerated
  histogram-row sums on aarch64 NEON / x86 SSE4.1+AVX2 / WASM
  SIMD128.

## `xtask`

- `cargo run --release -p xtask -- codegen` — reads the upstream
  CSV, computes LAB + chroma + the 32³ CIEDE2000 LUT (rayon-parallel,
  ~3 min on Apple Silicon), and emits a `rustfmt`-formatted
  `generated.rs`.

## Compatibility

- MSRV: **Rust 1.95** (required for stable AVX-512F intrinsics and
  `core::error::Error` in `no_std` builds via `thiserror` 2 without
  its `std` feature).
- Licenses: MIT OR Apache-2.0 (workspace crates). Upstream
  `colornamer` data redistributed under Apache-2.0; see
  `THIRD_PARTY_NOTICES.md`.
