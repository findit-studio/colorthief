<div align="center">
<h1>Color Thief</h1>
</div>
<div align="center">

Dominant colors with human-vocabulary names for video keyframes — MMCQ extraction + nearest-neighbor lookup against the xkcd color survey.

[<img alt="github" src="https://img.shields.io/badge/github-findit--ai/colorthief-8da0cb?style=for-the-badge&logo=Github" height="22">][Github-url]
<img alt="LoC" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fal8n%2F327b2a8aef9003246e45c6e47fe63937%2Fraw%2Fcolorthief" height="22">
[<img alt="Build" src="https://img.shields.io/github/actions/workflow/status/findit-ai/colorthief/ci.yml?logo=Github-Actions&style=for-the-badge" height="22">][CI-url]
[<img alt="codecov" src="https://img.shields.io/codecov/c/gh/findit-ai/colorthief?style=for-the-badge&token=6R3QFWRWHL&logo=codecov" height="22">][codecov-url]

[<img alt="docs.rs" src="https://img.shields.io/badge/docs.rs-colorthief-66c2a5?style=for-the-badge&labelColor=555555&logo=data:image/svg+xml;base64,PHN2ZyByb2xlPSJpbWciIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgdmlld0JveD0iMCAwIDUxMiA1MTIiPjxwYXRoIGZpbGw9IiNmNWY1ZjUiIGQ9Ik00ODguNiAyNTAuMkwzOTIgMjE0VjEwNS41YzAtMTUtOS4zLTI4LjQtMjMuNC0zMy43bC0xMDAtMzcuNWMtOC4xLTMuMS0xNy4xLTMuMS0yNS4zIDBsLTEwMCAzNy41Yy0xNC4xIDUuMy0yMy40IDE4LjctMjMuNCAzMy43VjIxNGwtOTYuNiAzNi4yQzkuMyAyNTUuNSAwIDI2OC45IDAgMjgzLjlWMzk0YzAgMTMuNiA3LjcgMjYuMSAxOS45IDMyLjJsMTAwIDUwYzEwLjEgNS4xIDIyLjEgNS4xIDMyLjIgMGwxMDMuOS01MiAxMDMuOSA1MmMxMC4xIDUuMSAyMi4xIDUuMSAzMi4yIDBsMTAwLTUwYzEyLjItNi4xIDE5LjktMTguNiAxOS45LTMyLjJWMjgzLjljMC0xNS05LjMtMjguNC0yMy40LTMzLjd6TTM1OCAyMTQuOGwtODUgMzEuOXYtNjguMmw4NS0zN3Y3My4zek0xNTQgMTA0LjFsMTAyLTM4LjIgMTAyIDM4LjJ2LjZsLTEwMiA0MS40LTEwMi00MS40di0uNnptODQgMjkxLjFsLTg1IDQyLjV2LTc5LjFsODUtMzguOHY3NS40em0wLTExMmwtMTAyIDQxLjQtMTAyLTQxLjR2LS42bDEwMi0zOC4yIDEwMiAzOC4ydi42em0yNDAgMTEybC04NSA0Mi41di03OS4xbDg1LTM4Ljh2NzUuNHptMC0xMTJsLTEwMiA0MS40LTEwMi00MS40di0uNmwxMDItMzguMiAxMDIgMzguMnYuNnoiPjwvcGF0aD48L3N2Zz4K" height="20">][doc-url]
[<img alt="crates.io" src="https://img.shields.io/crates/v/colorthief?style=for-the-badge&logo=data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iaXNvLTg4NTktMSI/Pg0KPCEtLSBHZW5lcmF0b3I6IEFkb2JlIElsbHVzdHJhdG9yIDE5LjAuMCwgU1ZHIEV4cG9ydCBQbHVnLUluIC4gU1ZHIFZlcnNpb246IDYuMDAgQnVpbGQgMCkgIC0tPg0KPHN2ZyB2ZXJzaW9uPSIxLjEiIGlkPSJMYXllcl8xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHhtbG5zOnhsaW5rPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hsaW5rIiB4PSIwcHgiIHk9IjBweCINCgkgdmlld0JveD0iMCAwIDUxMiA1MTIiIHhtbDpzcGFjZT0icHJlc2VydmUiPg0KPGc+DQoJPGc+DQoJCTxwYXRoIGQ9Ik0yNTYsMEwzMS41MjgsMTEyLjIzNnYyODcuNTI4TDI1Niw1MTJsMjI0LjQ3Mi0xMTIuMjM2VjExMi4yMzZMMjU2LDB6IE0yMzQuMjc3LDQ1Mi41NjRMNzQuOTc0LDM3Mi45MTNWMTYwLjgxDQoJCQlsMTU5LjMwMyw3OS42NTFWNDUyLjU2NHogTTEwMS44MjYsMTI1LjY2MkwyNTYsNDguNTc2bDE1NC4xNzQsNzcuMDg3TDI1NiwyMDIuNzQ5TDEwMS44MjYsMTI1LjY2MnogTTQzNy4wMjYsMzcyLjkxMw0KCQkJbC0xNTkuMzAzLDc5LjY1MVYyNDAuNDYxbDE1OS4zMDMtNzkuNjUxVjM3Mi45MTN6IiBmaWxsPSIjRkZGIi8+DQoJPC9nPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPC9zdmc+DQo=" height="22">][crates-url]
[<img alt="crates.io" src="https://img.shields.io/crates/d/colorthief?color=critical&logo=data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBzdGFuZGFsb25lPSJubyI/PjwhRE9DVFlQRSBzdmcgUFVCTElDICItLy9XM0MvL0RURCBTVkcgMS4xLy9FTiIgImh0dHA6Ly93d3cudzMub3JnL0dyYXBoaWNzL1NWRy8xLjEvRFREL3N2ZzExLmR0ZCI+PHN2ZyB0PSIxNjQ1MTE3MzMyOTU5IiBjbGFzcz0iaWNvbiIgdmlld0JveD0iMCAwIDEwMjQgMTAyNCIgdmVyc2lvbj0iMS4xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHAtaWQ9IjM0MjEiIGRhdGEtc3BtLWFuY2hvci1pZD0iYTMxM3guNzc4MTA2OS4wLmkzIiB3aWR0aD0iNDgiIGhlaWdodD0iNDgiIHhtbG5zOnhsaW5rPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hsaW5rIj48ZGVmcz48c3R5bGUgdHlwZT0idGV4dC9jc3MiPjwvc3R5bGU+PC9kZWZzPjxwYXRoIGQ9Ik00NjkuMzEyIDU3MC4yNHYtMjU2aDg1LjM3NnYyNTZoMTI4TDUxMiA3NTYuMjg4IDM0MS4zMTIgNTcwLjI0aDEyOHpNMTAyNCA2NDAuMTI4QzEwMjQgNzgyLjkxMiA5MTkuODcyIDg5NiA3ODcuNjQ4IDg5NmgtNTEyQzEyMy45MDQgODk2IDAgNzYxLjYgMCA1OTcuNTA0IDAgNDUxLjk2OCA5NC42NTYgMzMxLjUyIDIyNi40MzIgMzAyLjk3NiAyODQuMTYgMTk1LjQ1NiAzOTEuODA4IDEyOCA1MTIgMTI4YzE1Mi4zMiAwIDI4Mi4xMTIgMTA4LjQxNiAzMjMuMzkyIDI2MS4xMkM5NDEuODg4IDQxMy40NCAxMDI0IDUxOS4wNCAxMDI0IDY0MC4xOTJ6IG0tMjU5LjItMjA1LjMxMmMtMjQuNDQ4LTEyOS4wMjQtMTI4Ljg5Ni0yMjIuNzItMjUyLjgtMjIyLjcyLTk3LjI4IDAtMTgzLjA0IDU3LjM0NC0yMjQuNjQgMTQ3LjQ1NmwtOS4yOCAyMC4yMjQtMjAuOTI4IDIuOTQ0Yy0xMDMuMzYgMTQuNC0xNzguMzY4IDEwNC4zMi0xNzguMzY4IDIxNC43MiAwIDExNy45NTIgODguODMyIDIxNC40IDE5Ni45MjggMjE0LjRoNTEyYzg4LjMyIDAgMTU3LjUwNC03NS4xMzYgMTU3LjUwNC0xNzEuNzEyIDAtODguMDY0LTY1LjkyLTE2NC45MjgtMTQ0Ljk2LTE3MS43NzZsLTI5LjUwNC0yLjU2LTUuODg4LTMwLjk3NnoiIGZpbGw9IiNmZmZmZmYiIHAtaWQ9IjM0MjIiIGRhdGEtc3BtLWFuY2hvci1pZD0iYTMxM3guNzc4MTA2OS4wLmkwIiBjbGFzcz0iIj48L3BhdGg+PC9zdmc+&style=for-the-badge" height="22">][crates-url]
<img alt="license" src="https://img.shields.io/badge/License-Apache%202.0/MIT-blue.svg?style=for-the-badge&fontColor=white&logoColor=f5c076&logo=data:image/svg+xml;base64,PCFET0NUWVBFIHN2ZyBQVUJMSUMgIi0vL1czQy8vRFREIFNWRyAxLjEvL0VOIiAiaHR0cDovL3d3dy53My5vcmcvR3JhcGhpY3MvU1ZHLzEuMS9EVEQvc3ZnMTEuZHRkIj4KDTwhLS0gVXBsb2FkZWQgdG86IFNWRyBSZXBvLCB3d3cuc3ZncmVwby5jb20sIFRyYW5zZm9ybWVkIGJ5OiBTVkcgUmVwbyBNaXhlciBUb29scyAtLT4KPHN2ZyBmaWxsPSIjZmZmZmZmIiBoZWlnaHQ9IjgwMHB4IiB3aWR0aD0iODAwcHgiIHZlcnNpb249IjEuMSIgaWQ9IkNhcGFfMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgdmlld0JveD0iMCAwIDI3Ni43MTUgMjc2LjcxNSIgeG1sOnNwYWNlPSJwcmVzZXJ2ZSIgc3Ryb2tlPSIjZmZmZmZmIj4KDTxnIGlkPSJTVkdSZXBvX2JnQ2FycmllciIgc3Ryb2tlLXdpZHRoPSIwIi8+Cg08ZyBpZD0iU1ZHUmVwb190cmFjZXJDYXJyaWVyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiLz4KDTxnIGlkPSJTVkdSZXBvX2ljb25DYXJyaWVyIj4gPGc+IDxwYXRoIGQ9Ik0xMzguMzU3LDBDNjIuMDY2LDAsMCw2Mi4wNjYsMCwxMzguMzU3czYyLjA2NiwxMzguMzU3LDEzOC4zNTcsMTM4LjM1N3MxMzguMzU3LTYyLjA2NiwxMzguMzU3LTEzOC4zNTcgUzIxNC42NDgsMCwxMzguMzU3LDB6IE0xMzguMzU3LDI1OC43MTVDNzEuOTkyLDI1OC43MTUsMTgsMjA0LjcyMywxOCwxMzguMzU3UzcxLjk5MiwxOCwxMzguMzU3LDE4IHMxMjAuMzU3LDUzLjk5MiwxMjAuMzU3LDEyMC4zNTdTMjA0LjcyMywyNTguNzE1LDEzOC4zNTcsMjU4LjcxNXoiLz4gPHBhdGggZD0iTTE5NC43OTgsMTYwLjkwM2MtNC4xODgtMi42NzctOS43NTMtMS40NTQtMTIuNDMyLDIuNzMyYy04LjY5NCwxMy41OTMtMjMuNTAzLDIxLjcwOC0zOS42MTQsMjEuNzA4IGMtMjUuOTA4LDAtNDYuOTg1LTIxLjA3OC00Ni45ODUtNDYuOTg2czIxLjA3Ny00Ni45ODYsNDYuOTg1LTQ2Ljk4NmMxNS42MzMsMCwzMC4yLDcuNzQ3LDM4Ljk2OCwyMC43MjMgYzIuNzgyLDQuMTE3LDguMzc1LDUuMjAxLDEyLjQ5NiwyLjQxOGM0LjExOC0yLjc4Miw1LjIwMS04LjM3NywyLjQxOC0xMi40OTZjLTEyLjExOC0xNy45MzctMzIuMjYyLTI4LjY0NS01My44ODItMjguNjQ1IGMtMzUuODMzLDAtNjQuOTg1LDI5LjE1Mi02NC45ODUsNjQuOTg2czI5LjE1Miw2NC45ODYsNjQuOTg1LDY0Ljk4NmMyMi4yODEsMCw0Mi43NTktMTEuMjE4LDU0Ljc3OC0zMC4wMDkgQzIwMC4yMDgsMTY5LjE0NywxOTguOTg1LDE2My41ODIsMTk0Ljc5OCwxNjAuOTAzeiIvPiA8L2c+IDwvZz4KDTwvc3ZnPg==" height="22">

</div>

## Overview

`colorthief` extracts dominant colors from packed-RGB video keyframes
and maps each to its closest entry in a 949-color human-vocabulary
table sourced from the [xkcd color survey][xkcd]. Built for video
indexing and search-vocabulary pipelines: every output dominant
carries both the actual MMCQ-extracted RGB (for swatch rendering)
and the named `Color` (for search-index vocabulary), sorted
descending by population.

## Crates in this workspace

| Crate | Purpose |
|---|---|
| [`colorthief`](./colorthief) | Dominant-color extraction (MMCQ) + naming pipeline. `RgbFrame<'a>` (8-bit) / `Rgb48Frame<'a>` (16-bit HDR) input. |
| [`colorthief-dataset`](./colorthief-dataset) | Static xkcd palette + nearest-neighbor lookup with three color-difference metrics (CIEDE2000, CIE94, Delta E 76). `no_std + no_alloc`. |
| `xtask` | Build-time codegen — re-runs offline to regenerate the static dataset and CIEDE2000 LUT from the upstream CSV. Not published. |

## Installation

```toml
[dependencies]
colorthief = "0.1"

# Or, if you only need the static palette + nearest-neighbor lookup
# (no MMCQ; works in no_std + no_alloc):
colorthief-dataset = "0.1"
```

Minimum supported Rust version: **1.95** (required for stable AVX-512F
intrinsics and `core::error::Error` in `no_std` builds via
`thiserror` 2 without its `std` feature).

## Examples

| Example | Crate | Run |
|---|---|---|
| [`extract`](./colorthief/examples/extract.rs) | `colorthief` | `cargo run --release --example extract -p colorthief` |
| [`extract_rgb48`](./colorthief/examples/extract_rgb48.rs) (HDR / 16-bit) | `colorthief` | `cargo run --release --example extract_rgb48 -p colorthief` |
| [`extract_no_alloc`](./colorthief/examples/extract_no_alloc.rs) (`static mut Mmcq` + fixed buffer) | `colorthief` | `cargo run --release --example extract_no_alloc -p colorthief` |
| [`lookup`](./colorthief-dataset/examples/lookup.rs) (name-only, no MMCQ) | `colorthief-dataset` | `cargo run --release --example lookup -p colorthief-dataset` |

See more details in [examples](./colorthief/examples) and
[examples](./colorthief-dataset/examples).

## Algorithms

Three nearest-neighbor metrics, behind a `#[non_exhaustive] #[repr(u8)]` enum:

| `Algorithm` | Speed (NEON) | Notes |
|---|---:|---|
| `Ciede2000Exact` (default) | ~230 ns/query (LUT) or 71.5 µs (full scan) | Modern perceptual gold-standard. Provably exact at u8 RGB resolution when `lut` feature is on. |
| `Cie94` | ~510 ns/query | Asymmetric (palette = reference). Mid-accuracy. |
| `DeltaE76` | ~470 ns/query | Squared Euclidean LAB. Fastest, but well-known biases in the saturated blue / yellow regions. |

The default `Ciede2000Exact` is ~310× faster than naive full-scan
thanks to a pre-computed 32³ candidate-set LUT (see Architecture
below).

## Feature flags

`colorthief`:

| Feature | Default | Effect |
|---|:---:|---|
| `std` | ✓ | `thread_local!`-cached MMCQ workspace; zero-alloc-per-call after first call per thread. Implies `alloc`. |
| `alloc` | | Heap allocator available; enables `Vec<Dominant>`-returning APIs and `Mmcq::new_boxed()`. |
| `lut` | ✓ | 32³ candidate-set LUT for CIEDE2000 — ~256 KB binary cost, ~310× CIEDE2000 speedup. |

`colorthief-dataset`:

| Feature | Default | Effect |
|---|:---:|---|
| `std` | ✓ | Enables x86_64 runtime CPU-feature detection. |
| `alloc` | | Forward-compat hook (current API is `no_alloc`). |
| `lut` | ✓ | The 32³ CIEDE2000 LUT — propagated from `colorthief/lut`. |

## No-std + no-alloc support

Both crates are usable in `no_std + no_alloc` environments. Caller
manages the MMCQ workspace (a `static mut Mmcq` placed in `.bss`) and
the output buffer (a fixed-size `[Option<Dominant>; N]`). See the
[`extract_no_alloc`](./colorthief/examples/extract_no_alloc.rs)
example for the full pattern.

The `Buffer<T>` trait abstracts the output: `Vec<T>` (alloc-gated),
`[Option<T>; N]`, `&mut [Option<T>]` ship by default; consumers can
plug in `arrayvec::ArrayVec` / `heapless::Vec` / custom types with
a one-line `impl Buffer<T>`.

For zero-alloc-per-call in single-threaded `no_std + alloc`
environments (typical wasm32-unknown-unknown / interrupt-free bare
metal), place an `Mmcq` in `static mut` yourself — the `unsafe`
then sits at your call site, not silently inside this crate.

## SIMD backends

`Color::nearest_to` (Delta E 76) and `Color::nearest_to_cie94`
dispatch to per-arch SIMD backends:

| Backend | ISA | Lanes | Detection |
|---|---|---:|---|
| `aarch64_neon` | NEON | 4 (128-bit) | compile-time (`target_feature = "neon"`) |
| `x86_avx512` | AVX-512F | 16 (512-bit) | runtime (`is_x86_feature_detected!`) |
| `x86_avx2` | AVX2 | 8 (256-bit) | runtime |
| `x86_sse41` | SSE4.1 | 4 (128-bit) | runtime |
| `wasm_simd128` | SIMD128 | 4 (128-bit) | compile-time (`target_feature = "simd128"`) |
| `scalar` | — | 1 | always available |

Every backend is **bit-identical** to the scalar reference — plain
`mul + add` (no FMA) — and verified against a 17³ = 4913-point
inline parity grid plus an exhaustive 256³ = 16,777,216-point sweep
(`#[ignore]`-gated; run via `cargo test --release --ignored`).

CIEDE2000 is scalar-only by design — its `atan2` / `sin` / `cos` /
`exp` and branchy hue-wraparound logic don't vectorize cleanly; an
attempt regressed by ~35% vs the scalar baseline.

## Codegen pipeline

`colorthief-dataset/src/generated.rs` is produced offline by
`cargo run --release -p xtask -- codegen`. The xtask:

1. Parses `colorthief-dataset/assets/color_hierarchy.csv` (sourced
   from Stitch Fix's [`colornamer`][colornamer], Apache-2.0).
2. Computes CIE LAB (D65, 2°) per entry.
3. Computes the 32³ CIEDE2000 candidate-set LUT (rayon-parallel,
   ~3 min on Apple Silicon — every u8 RGB swept through the
   full-scan reference).
4. Emits two `#[non_exhaustive] #[repr(u8)]` enums (`Family`, `Kind`)
   covering every distinct value in the CSV.
5. Pretty-prints + `rustfmt`s the result so it passes `cargo fmt
   --check`.

CI's `codegen-up-to-date` job re-runs the xtask and fails if
`generated.rs` would change — guarantees no drift between `assets/`
and the committed source.

## Coverage-side cfgs

For coverage runs that need to exercise lower-tier SIMD branches on
hardware that natively supports a higher tier:

- `--cfg colorthief_force_scalar` — bypass every SIMD backend.
- `--cfg colorthief_disable_avx512` — drop x86_64 from AVX-512F to AVX2.
- `--cfg colorthief_disable_avx2` — drop x86_64 to SSE4.1.

These flags are also exercised by the `simd.yml` CI workflow.

## License

`colorthief` is dual-licensed under MIT or Apache-2.0 at your
option.

See [LICENSE-APACHE](LICENSE-APACHE), [LICENSE-MIT](LICENSE-MIT) for
details.

The upstream xkcd color-survey data is public domain (Randall
Munroe); Stitch Fix's hierarchical name layers are Apache-2.0
(attribution in [`THIRD_PARTY_NOTICES.md`](./THIRD_PARTY_NOTICES.md)).

Copyright (c) 2026 FinDIT Studio authors.

[Github-url]: https://github.com/findit-ai/colorthief/
[CI-url]: https://github.com/findit-ai/colorthief/actions/workflows/ci.yml
[doc-url]: https://docs.rs/colorthief
[crates-url]: https://crates.io/crates/colorthief
[codecov-url]: https://app.codecov.io/gh/findit-ai/colorthief/
[zh-cn-url]: https://github.com/findit-ai/colorthief/tree/main/README-zh_CN.md
[xkcd]: https://blog.xkcd.com/2010/05/03/color-survey-results/
[colornamer]: https://github.com/stitchfix/colornamer
