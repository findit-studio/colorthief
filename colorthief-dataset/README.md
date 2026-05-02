# colorthief-dataset

Static [xkcd color survey](https://blog.xkcd.com/2010/05/03/color-survey-results/)
hierarchy with pre-computed CIE LAB, used by
[`colorthief`](https://github.com/findit-ai/colorthief) for human-vocabulary
color naming. **`no_std` + no-allocation** — every entry is `&'static`, every
lookup is a stack-only scan with optional SIMD acceleration.

## Installation

```toml
[dependencies]
colorthief-dataset = "0.0"
```

## What it ships

The dataset is sourced from Stitch Fix's
[`colornamer`](https://github.com/stitchfix/colornamer) (Apache 2.0;
attribution in [`THIRD_PARTY_NOTICES.md`](./THIRD_PARTY_NOTICES.md)),
extending the original 949-entry xkcd survey with three nested
human-readable name layers. Each [`Color`] entry exposes:

| Layer | Cardinality | Example |
|-------|------------:|---------|
| `name` (xkcd) | 949 | `"burnt orange"` |
| `design_name` | ~250 | `"russet brown"` |
| `common_name` | ~120 | `"sienna"` |
| `family` ([`Family`] enum) | 26 | [`Family::OrangeRed`] |
| `kind` ([`Kind`] enum) | 11 | [`Kind::PainterlyNeutral`] |
| `is_neutral` | bool | `false` |

Plus per-layer `hex` / `rgb` representations, and a pre-computed `lab`
triple (D65, 2°) for nearest-neighbor lookup.

## Usage

### Find the named entry closest to an arbitrary RGB

```rust
use colorthief_dataset::Color;

let c = Color::nearest_to([189, 108, 72]);
assert_eq!(c.name(), "adobe");
assert_eq!(c.common_name(), "sienna");
assert_eq!(c.family().as_str(), "sienna");
assert!(c.is_neutral());
```

### Iterate the full table

```rust
use colorthief_dataset::{Color, Family};

let blues = Color::all().iter().filter(|c| c.family() == Family::Blue).count();
println!("{blues} entries in the Blue family");
```

## Codegen pipeline

`generated.rs` is produced offline by `cargo xtask codegen` against
`assets/color_hierarchy.csv`. The xtask:

1. Parses every CSV row into a `Color` literal.
2. Computes CIE LAB (D65, 2°, sRGB → linear → XYZ → LAB) per entry —
   pre-computing means runtime nearest-neighbor pays only one
   transcendental conversion per query, not 950.
3. Emits two `#[non_exhaustive] #[repr(u8)]` enums (`Family`, `Kind`)
   covering every distinct value in the CSV.
4. Pretty-prints + `rustfmt`s the result so it passes
   `cargo fmt --check`.

The CI's `codegen-up-to-date` job re-runs the xtask and fails if
`generated.rs` would change — guarantees no drift between `assets/` and
the committed source.

## SIMD nearest-neighbor

[`Color::nearest_to`] dispatches to a per-arch backend at runtime
(x86_64) or compile-time (aarch64, wasm32):

| Backend | ISA | Lanes |
|---------|-----|------:|
| `aarch64_neon` | `target_arch = "aarch64"` | 4 (128-bit) |
| `x86_avx2` | `is_x86_feature_detected!("avx2")` | 8 (256-bit) |
| `x86_sse41` | `is_x86_feature_detected!("sse4.1")` | 4 (128-bit) |
| `wasm_simd128` | `target_feature = "simd128"` | 4 (128-bit) |
| `scalar` | always-compiled fallback | 1 |

Every backend is bit-identical to scalar (plain `mul`+`add`, no FMA);
the parity tests in `nearest::tests` enforce this against a
17³ = 4913-point RGB grid. On Apple Silicon, NEON gives ~2.0× over
scalar for the inner scan.

## Features

| Feature | Default | Effect |
|---------|:-------:|--------|
| `std` | ✓ | Enables runtime CPU feature detection on x86_64; safe to disable for `no_std` deployments. |
| `alloc` | | Forward-compat hook; current API is `no_alloc`. |

## Coverage-side cfgs

For coverage runs that need to exercise lower-tier SIMD branches on
AVX2-capable hardware:

- `--cfg colorthief_force_scalar` — bypass every SIMD backend.
- `--cfg colorthief_disable_avx2` — drop x86_64 from AVX2 to SSE4.1.

These mirror the same flags consumed by `colorthief`'s MMCQ SIMD
helper module.

## License

MIT OR Apache-2.0. Upstream xkcd survey is public domain (Munroe);
Stitch Fix's hierarchical roll-ups are Apache-2.0 (see
[`THIRD_PARTY_NOTICES.md`](./THIRD_PARTY_NOTICES.md)).
