//! Codegen for `colorthief-dataset/src/generated.rs`.
//!
//! Reads `colorthief-dataset/assets/color_hierarchy.csv` (sourced from
//! Stitch Fix's `colornamer`), pre-computes CIE LAB for each entry's
//! xkcd RGB, and emits a Rust source file containing one `const` per
//! entry plus a `pub static COLORS: &[&Color]` slice.
//!
//! Run with: `cargo xtask codegen`.

use std::{
  collections::{BTreeMap, HashSet},
  path::PathBuf,
};

use heck::{ToShoutySnakeCase, ToUpperCamelCase};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use serde::Deserialize;

/// One row of `color_hierarchy.csv`. Every column from the upstream
/// dataset is captured and emitted into `generated.rs` — the runtime
/// `Color` exposes all 18 of them so consumers can pick the level of
/// granularity (xkcd / design / common name + hex + rgb) that fits
/// their use case.
#[derive(Debug, Deserialize)]
struct CsvRow {
  xkcd_color: String,
  xkcd_color_hex: String,
  xkcd_r: u8,
  xkcd_g: u8,
  xkcd_b: u8,
  design_color: String,
  design_color_hex: String,
  design_r: u8,
  design_g: u8,
  design_b: u8,
  common_color: String,
  common_color_hex: String,
  common_r: u8,
  common_g: u8,
  common_b: u8,
  color_family: String,
  color_type: String,
  color_or_neutral: String,
}

fn main() {
  let mut args = std::env::args().skip(1);
  match args.next().as_deref() {
    Some("codegen") | None => codegen(),
    Some(other) => {
      eprintln!("unknown xtask command: {other}");
      eprintln!("usage: cargo xtask [codegen]");
      std::process::exit(1);
    }
  }
}

fn workspace_root() -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    .parent()
    .expect("xtask crate must live one level under the workspace root")
    .to_path_buf()
}

fn codegen() {
  let root = workspace_root();
  let csv_path = root.join("colorthief-dataset/assets/color_hierarchy.csv");
  let out_path = root.join("colorthief-dataset/src/generated.rs");

  // 1. Parse CSV.
  let mut rdr = csv::ReaderBuilder::new()
    .has_headers(true)
    .from_path(&csv_path)
    .unwrap_or_else(|e| panic!("open {}: {e}", csv_path.display()));
  let rows: Vec<CsvRow> = rdr
    .deserialize::<CsvRow>()
    .map(|r| r.expect("parse csv row"))
    .collect();
  assert!(!rows.is_empty(), "CSV must have at least one entry");

  // 2a. Collect every distinct `color_family` and `color_type` value
  // across the CSV — these become enum variants in `generated.rs`. Use
  // `BTreeMap` for alphabetical, deterministic variant ordering.
  let family_variants = collect_enum_variants(&rows, |r| &r.color_family, "color_family");
  let kind_variants = collect_enum_variants(&rows, |r| &r.color_type, "color_type");

  // 2b. Build per-entry tokens; track ident uniqueness as we go.
  // SoA accumulators feed `LABS_L`/`LABS_A`/`LABS_B` — the NEON
  // `nearest_to` backend reads these as dense `vld1q_f32` chunks rather
  // than gathering through `&[&Color]` indirection.
  let mut seen_idents = HashSet::<String>::new();
  let mut consts: Vec<TokenStream> = Vec::with_capacity(rows.len());
  let mut idents: Vec<syn::Ident> = Vec::with_capacity(rows.len());
  let mut labs_l: Vec<TokenStream> = Vec::with_capacity(rows.len());
  let mut labs_a: Vec<TokenStream> = Vec::with_capacity(rows.len());
  let mut labs_b: Vec<TokenStream> = Vec::with_capacity(rows.len());
  // Pre-computed CIE94 reference chroma C₁ = sqrt(a² + b²) per
  // entry. Saves the chroma `sqrtf` inside CIE94's per-pair inner
  // loop (one `sqrtf` × 949 entries per query → ~30% scalar
  // speedup, ~5% NEON speedup).
  let mut labs_c: Vec<TokenStream> = Vec::with_capacity(rows.len());
  // Parallel f32 storage feeds the LUT computation below — the
  // `labs_*: Vec<TokenStream>` above are quote-tokens for source
  // emission, not callable values.
  let mut palette_labs: Vec<[f32; 3]> = Vec::with_capacity(rows.len());
  for row in &rows {
    let ident = name_to_const_ident(&row.xkcd_color);
    if !seen_idents.insert(ident.to_string()) {
      panic!(
        "duplicate const ident `{ident}` (xkcd name: {:?}); two CSV rows \
         collide after SHOUTY_SNAKE_CASE conversion. Disambiguate the \
         names upstream, or add a deterministic suffix in `name_to_const_ident`.",
        row.xkcd_color,
      );
    }

    let [r, g, b] = [row.xkcd_r, row.xkcd_g, row.xkcd_b];
    let [dr, dg, db] = [row.design_r, row.design_g, row.design_b];
    let [cr, cg, cb] = [row.common_r, row.common_g, row.common_b];
    let lab = rgb_to_lab([r, g, b]);
    palette_labs.push(lab);
    let l_lit = float_lit(lab[0]);
    let a_lit = float_lit(lab[1]);
    let b_lit = float_lit(lab[2]);
    labs_l.push(l_lit.clone());
    labs_a.push(a_lit.clone());
    labs_b.push(b_lit.clone());

    // Reference chroma C₁ = sqrt(a² + b²). Pre-compute via libm so
    // the rounding matches the runtime `sqrtf` exactly (consumers
    // using the SoA `LABS_C` array are bit-equivalent to a runtime
    // `sqrtf(a*a + b*b)`).
    let c = libm::sqrtf(lab[1] * lab[1] + lab[2] * lab[2]);
    labs_c.push(float_lit(c));

    let xkcd_name = &row.xkcd_color;
    let xkcd_hex = &row.xkcd_color_hex;
    let design_name = &row.design_color;
    let design_hex = &row.design_color_hex;
    let common_name = &row.common_color;
    let common_hex = &row.common_color_hex;
    let family_variant = family_variants
      .get(&row.color_family)
      .expect("collected from this row");
    let kind_variant = kind_variants
      .get(&row.color_type)
      .expect("collected from this row");
    let is_neutral = parse_color_or_neutral(&row.color_or_neutral, xkcd_name);

    consts.push(quote! {
      const #ident: &Color = &Color {
        name: #xkcd_name,
        hex: #xkcd_hex,
        rgb: [#r, #g, #b],
        lab: [#l_lit, #a_lit, #b_lit],
        design_name: #design_name,
        design_hex: #design_hex,
        design_rgb: [#dr, #dg, #db],
        common_name: #common_name,
        common_hex: #common_hex,
        common_rgb: [#cr, #cg, #cb],
        family: Family::#family_variant,
        kind: Kind::#kind_variant,
        is_neutral: #is_neutral,
      };
    });
    idents.push(ident);
  }

  // 2c. Compute CIEDE2000 candidate-set LUT (parallelized via rayon).
  // Cells are independent — each runs 512 full-scans over the 949-entry
  // palette and unions the winners.
  let (lut_offsets, lut_indices) = compute_lut(&palette_labs);

  // 3. Assemble the file body and pretty-print.
  let count = idents.len();
  let count_doc = format!(" All {count} entries in the dataset, in CSV order.");
  let family_enum = build_enum_tokens(
    "Family",
    &family_variants,
    "Color family classification",
    "color_family",
  );
  let kind_enum = build_enum_tokens(
    "Kind",
    &kind_variants,
    "Color kind / texture classification",
    "color_type",
  );
  let labs_doc = " Structure-of-arrays projection of every entry's pre-computed LAB \
                   channel, in the same index order as [`COLORS`]. Used by the \
                   `nearest` module's NEON backend so per-component loads are dense \
                   `vld1q_f32`s rather than gathers through `&[&Color]`.";
  let body = quote! {
    use super::Color;

    #family_enum

    #kind_enum

    #(#consts)*

    #[doc = #count_doc]
    pub static COLORS: &[&Color] = &[
      #(#idents),*
    ];

    #[doc = #labs_doc]
    pub(crate) static LABS_L: &[f32] = &[#(#labs_l),*];
    #[doc = #labs_doc]
    pub(crate) static LABS_A: &[f32] = &[#(#labs_a),*];
    #[doc = #labs_doc]
    pub(crate) static LABS_B: &[f32] = &[#(#labs_b),*];

    /// Pre-computed CIE94 reference chroma `C₁ = sqrt(a² + b²)` per
    /// entry, in the same index order as [`COLORS`]. Saves the
    /// chroma `sqrtf` from CIE94's per-pair inner loop.
    pub(crate) static LABS_C: &[f32] = &[#(#labs_c),*];

    /// CIEDE2000 candidate-set LUT — CSR offsets array. Length
    /// `32³ + 1 = 32,769`. For cell `c` (where `c = ((r >> 3) << 10) |
    /// ((g >> 3) << 5) | (b >> 3)`), the candidate indices are
    /// `LUT_CELL_INDICES[LUT_CELL_OFFSETS[c]..LUT_CELL_OFFSETS[c + 1]]`.
    /// Generated by `cargo xtask codegen`; do not edit by hand.
    #[cfg(feature = "lut")]
    pub(crate) static LUT_CELL_OFFSETS: &[u32] = &[#(#lut_offsets),*];

    /// CIEDE2000 candidate-set LUT — flat candidate indices indexed
    /// via [`LUT_CELL_OFFSETS`]. Each value is an index into
    /// [`COLORS`]. Provably exact at u8 RGB resolution: every
    /// reachable u8 RGB was sampled at codegen time, so the runtime
    /// LUT path's small candidate scan is guaranteed to contain the
    /// true CIEDE2000 nearest for any u8 RGB query.
    #[cfg(feature = "lut")]
    pub(crate) static LUT_CELL_INDICES: &[u16] = &[#(#lut_indices),*];
  };

  let pretty = prettyplease::unparse(
    &syn::parse2::<syn::File>(body).expect("generated tokens must parse as a Rust file"),
  );
  let header = "// This file is generated by `cargo xtask codegen`, do not edit it manually.\n\n";
  let output = format!("{header}{pretty}");

  std::fs::write(&out_path, output).unwrap_or_else(|e| panic!("write {}: {e}", out_path.display()));

  // prettyplease emits 4-space indent unconditionally; the workspace
  // rustfmt.toml uses `tab_spaces = 2`. Shell out to rustfmt so the
  // generated file passes `cargo fmt --check` like the hand-written ones.
  let status = std::process::Command::new("rustfmt")
    .arg("--edition=2024")
    .arg(&out_path)
    .status()
    .expect("rustfmt is required on PATH for `cargo xtask codegen`");
  assert!(
    status.success(),
    "rustfmt {out} failed with status {status}",
    out = out_path.display(),
  );
  println!("wrote {} ({count} entries)", out_path.display());
}

fn name_to_const_ident(name: &str) -> syn::Ident {
  // Pre-substitute `/` with a sentinel word so it survives heck's
  // word-boundary handling as a distinct token. Without this, `blue/green`
  // and `blue green` both shouty-snake-case to `BLUE_GREEN` (11 such
  // pairs in the xkcd dataset).
  let preprocessed = name.replace('/', " slash ");
  let raw = preprocessed.to_shouty_snake_case();
  // Replace any leftover non-ident chars defensively. heck handles spaces
  // and hyphens but xkcd names occasionally contain apostrophes
  // ("robin's egg") that would slip through.
  let cleaned: String = raw
    .chars()
    .map(|c| {
      if c.is_ascii_alphanumeric() || c == '_' {
        c
      } else {
        '_'
      }
    })
    .collect();
  // Identifiers can't start with a digit; prefix on the rare chance an xkcd
  // name happens to start with one.
  let cleaned = if cleaned
    .chars()
    .next()
    .map(|c| c.is_ascii_digit())
    .unwrap_or(false)
  {
    format!("C_{cleaned}")
  } else {
    cleaned
  };
  format_ident!("{}", cleaned)
}

/// Walk every CSV row, extract the chosen string column, and build a
/// `value -> variant_ident` map. Panics on duplicate variant idents
/// from distinct CSV values (which would mean the upstream dataset
/// added a value that collides with an existing one after
/// `to_upper_camel_case` — the right fix is to disambiguate the new
/// value upstream or extend `enum_variant_ident`'s sanitisation).
fn collect_enum_variants(
  rows: &[CsvRow],
  get: impl Fn(&CsvRow) -> &str,
  column_label: &str,
) -> BTreeMap<String, syn::Ident> {
  let mut by_value: BTreeMap<String, syn::Ident> = BTreeMap::new();
  for row in rows {
    let value = get(row).trim().to_string();
    if value.is_empty() {
      panic!(
        "row {:?}: column `{column_label}` is empty; every entry must \
         have a non-empty {column_label} so the enum is exhaustive",
        row.xkcd_color,
      );
    }
    let ident = enum_variant_ident(&value);
    if let Some(existing) = by_value.get(&value) {
      // Same value, same ident — already recorded, skip.
      debug_assert_eq!(existing, &ident);
      continue;
    }
    if let Some((other_value, _)) = by_value.iter().find(|(_, v)| **v == ident) {
      panic!(
        "{column_label}: variant `{ident}` collision between \
         {other_value:?} and {value:?}; adjust enum_variant_ident to \
         disambiguate (the two CSV values would generate the same Rust \
         identifier).",
      );
    }
    by_value.insert(value, ident);
  }
  by_value
}

/// Convert a CSV value (lowercase, possibly multi-word) into a Rust
/// `UpperCamelCase` enum variant ident.
fn enum_variant_ident(value: &str) -> syn::Ident {
  let camel = value.to_upper_camel_case();
  // Defensive sanitisation — heck handles spaces and hyphens, but slip
  // past anything else so we get a panic from `format_ident!` rather
  // than a confusing emit-time error.
  let cleaned: String = camel
    .chars()
    .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
    .collect();
  let cleaned = if cleaned
    .chars()
    .next()
    .map(|c| c.is_ascii_digit())
    .unwrap_or(false)
  {
    format!("V_{cleaned}")
  } else {
    cleaned
  };
  format_ident!("{}", cleaned)
}

/// Build a single `pub enum NAME { ... }` + its `as_str()` impl.
fn build_enum_tokens(
  type_name: &str,
  values: &BTreeMap<String, syn::Ident>,
  doc_summary: &str,
  csv_column: &str,
) -> TokenStream {
  let type_ident = format_ident!("{}", type_name);
  let type_doc = format!(
    " {doc_summary} sourced from the upstream `color_hierarchy.csv` \n\
     `{csv_column}` column. Marked `#[non_exhaustive]` so adding a \n\
     new upstream value is a non-breaking change for downstream \n\
     consumers; call [`{type_name}::as_str`] to get the original \n\
     string back when you need to feed it into a search index."
  );
  let as_str_doc = format!(
    " The original `{csv_column}` string for this variant — exactly \
     what appears in `color_hierarchy.csv`."
  );

  let variants = values.iter().map(|(value, ident)| {
    let variant_doc = format!(" Variant for `{value}`.");
    quote! {
      #[doc = #variant_doc]
      #ident,
    }
  });
  let arms = values.iter().map(|(value, ident)| {
    quote! {
      Self::#ident => #value,
    }
  });

  // Variant counts in this dataset are 26 (Family) and 11 (Kind) — both
  // well under 256, so `#[repr(u8)]` is safely future-proof and gives
  // every enum value a predictable 1-byte layout. If upstream ever
  // grows past 256 variants the codegen will emit invalid tokens and
  // the next `cargo build` will fail loudly, which is what we want.
  let variant_count = values.len();
  assert!(
    variant_count <= 256,
    "{type_name}: {variant_count} unique values exceeds u8 repr capacity; \
     widen the repr or split into multiple enums",
  );

  quote! {
    #[doc = #type_doc]
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
    #[non_exhaustive]
    #[repr(u8)]
    pub enum #type_ident {
      #(#variants)*
    }

    impl #type_ident {
      #[doc = #as_str_doc]
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub const fn as_str(&self) -> &'static str {
        match self {
          #(#arms)*
        }
      }
    }
  }
}

fn parse_color_or_neutral(value: &str, name: &str) -> bool {
  match value.trim() {
    "color" => false,
    "neutral" => true,
    other => {
      panic!("row {name:?}: `color_or_neutral` must be \"color\" or \"neutral\", got {other:?}",)
    }
  }
}

/// Format an `f32` so the generated source round-trips back to the same
/// bit pattern. `{value:?}` uses Rust's shortest-round-trip representation
/// (the `Debug` impl for floats), which is what we want here.
fn float_lit(value: f32) -> TokenStream {
  let s = format!("{value:?}");
  // Ensure the literal carries an `f32` suffix so it parses unambiguously.
  let suffixed = if s.ends_with("f32") {
    s
  } else {
    format!("{s}_f32")
  };
  syn::parse_str::<TokenStream>(&suffixed)
    .unwrap_or_else(|e| panic!("invalid f32 literal {suffixed:?}: {e}"))
}

// ---------------------------------------------------------------------------
// LAB math — must stay byte-for-byte identical to
// `colorthief-dataset/src/lib.rs`. Both paths use libm so pre-computed and
// runtime-computed LAB values agree exactly on the same input.
// ---------------------------------------------------------------------------

fn rgb_to_lab(rgb: [u8; 3]) -> [f32; 3] {
  let r = srgb_to_linear(rgb[0] as f32 / 255.0);
  let g = srgb_to_linear(rgb[1] as f32 / 255.0);
  let b = srgb_to_linear(rgb[2] as f32 / 255.0);

  // Coefficients must stay byte-for-byte identical to
  // `colorthief-dataset/src/lib.rs::rgb_to_lab`. Trailing zeros that
  // exceed f32's ~7-digit precision are dropped to satisfy clippy.
  let x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375;
  let y = r * 0.2126729 + g * 0.7151522 + b * 0.072175;
  let z = r * 0.0193339 + g * 0.119192 + b * 0.9503041;

  const XN: f32 = 0.95047;
  const YN: f32 = 1.00000;
  const ZN: f32 = 1.08883;

  let fx = lab_f(x / XN);
  let fy = lab_f(y / YN);
  let fz = lab_f(z / ZN);

  [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
}

fn srgb_to_linear(c: f32) -> f32 {
  if c <= 0.04045 {
    c / 12.92
  } else {
    libm::powf((c + 0.055) / 1.055, 2.4)
  }
}

fn lab_f(t: f32) -> f32 {
  const DELTA_CUBED: f32 = 216.0 / 24389.0;
  const KAPPA_OVER_3: f32 = 841.0 / 108.0;
  const OFFSET: f32 = 4.0 / 29.0;
  if t > DELTA_CUBED {
    libm::cbrtf(t)
  } else {
    KAPPA_OVER_3 * t + OFFSET
  }
}

// ---------------------------------------------------------------------------
// CIEDE2000 metric — duplicated from
// `colorthief-dataset/src/nearest/ciede2000.rs` so xtask stays standalone.
// Both implementations use libm transcendentals so codegen-time and
// runtime distance values agree bit-for-bit on the same inputs.
// ---------------------------------------------------------------------------

const TWENTY_FIVE_POW_7: f32 = 25.0_f32 * 25.0 * 25.0 * 25.0 * 25.0 * 25.0 * 25.0;
const RAD_TO_DEG: f32 = 180.0 / std::f32::consts::PI;
const DEG_TO_RAD: f32 = std::f32::consts::PI / 180.0;

fn pow7(x: f32) -> f32 {
  let x2 = x * x;
  let x4 = x2 * x2;
  x4 * x2 * x
}

fn hue_atan2_deg(y: f32, x: f32) -> f32 {
  if y == 0.0 && x == 0.0 {
    return 0.0;
  }
  let h = libm::atan2f(y, x) * RAD_TO_DEG;
  if h < 0.0 { h + 360.0 } else { h }
}

fn delta_e_2000_sq(lab1: [f32; 3], lab2: [f32; 3]) -> f32 {
  let [l1, a1, b1] = lab1;
  let [l2, a2, b2] = lab2;
  let c1 = libm::sqrtf(a1 * a1 + b1 * b1);
  let c2 = libm::sqrtf(a2 * a2 + b2 * b2);

  let cbar = 0.5 * (c1 + c2);
  let cbar7 = pow7(cbar);
  let g = 0.5 * (1.0 - libm::sqrtf(cbar7 / (cbar7 + TWENTY_FIVE_POW_7)));

  let one_plus_g = 1.0 + g;
  let a1p = a1 * one_plus_g;
  let a2p = a2 * one_plus_g;
  let c1p = libm::sqrtf(a1p * a1p + b1 * b1);
  let c2p = libm::sqrtf(a2p * a2p + b2 * b2);

  let h1p = hue_atan2_deg(b1, a1p);
  let h2p = hue_atan2_deg(b2, a2p);

  let dlp = l2 - l1;
  let dcp = c2p - c1p;

  let dhp = if c1p * c2p == 0.0 {
    0.0
  } else {
    let diff = h2p - h1p;
    if diff > 180.0 {
      diff - 360.0
    } else if diff < -180.0 {
      diff + 360.0
    } else {
      diff
    }
  };

  let dh_cap = 2.0 * libm::sqrtf(c1p * c2p) * libm::sinf(0.5 * dhp * DEG_TO_RAD);

  let lp_bar = 0.5 * (l1 + l2);
  let cp_bar = 0.5 * (c1p + c2p);

  let hp_bar = if c1p * c2p == 0.0 {
    h1p + h2p
  } else {
    let abs_diff = (h1p - h2p).abs();
    let sum = h1p + h2p;
    if abs_diff <= 180.0 {
      0.5 * sum
    } else if sum < 360.0 {
      0.5 * (sum + 360.0)
    } else {
      0.5 * (sum - 360.0)
    }
  };

  let hp_rad = hp_bar * DEG_TO_RAD;
  let t = 1.0 - 0.17 * libm::cosf(hp_rad - 30.0 * DEG_TO_RAD)
    + 0.24 * libm::cosf(2.0 * hp_rad)
    + 0.32 * libm::cosf(3.0 * hp_rad + 6.0 * DEG_TO_RAD)
    - 0.20 * libm::cosf(4.0 * hp_rad - 63.0 * DEG_TO_RAD);

  let d_theta_arg = (hp_bar - 275.0) / 25.0;
  let d_theta = 30.0 * libm::expf(-(d_theta_arg * d_theta_arg));

  let cp_bar7 = pow7(cp_bar);
  let rc = 2.0 * libm::sqrtf(cp_bar7 / (cp_bar7 + TWENTY_FIVE_POW_7));

  let lp_minus_50 = lp_bar - 50.0;
  let sl = 1.0 + 0.015 * lp_minus_50 * lp_minus_50 / libm::sqrtf(20.0 + lp_minus_50 * lp_minus_50);
  let sc = 1.0 + 0.045 * cp_bar;
  let sh = 1.0 + 0.015 * cp_bar * t;

  let rt = -libm::sinf(2.0 * d_theta * DEG_TO_RAD) * rc;

  let dl_term = dlp / sl;
  let dc_term = dcp / sc;
  let dh_term = dh_cap / sh;
  dl_term * dl_term + dc_term * dc_term + dh_term * dh_term + rt * dc_term * dh_term
}

fn nearest_palette_idx(query: [f32; 3], palette_labs: &[[f32; 3]]) -> u16 {
  let mut best_idx = 0usize;
  let mut best_d2 = f32::INFINITY;
  for (i, &entry) in palette_labs.iter().enumerate() {
    let d2 = delta_e_2000_sq(entry, query);
    if d2 < best_d2 {
      best_d2 = d2;
      best_idx = i;
    }
  }
  // Palette has 949 entries — fits in u16 with a 65× headroom.
  u16::try_from(best_idx).expect("palette index must fit in u16")
}

// ---------------------------------------------------------------------------
// LUT computation: 32³ candidate-set grid for CIEDE2000.
// ---------------------------------------------------------------------------

/// 32 cells per RGB axis × 3 = 32,768 cells. Each cell covers an
/// 8×8×8 RGB box and stores the indices of every palette entry that
/// is the CIEDE2000-nearest at *some* RGB inside the box.
const LUT_AXIS: usize = 32;
const N_CELLS: usize = LUT_AXIS * LUT_AXIS * LUT_AXIS;

/// Compute the CIEDE2000 candidate-set LUT.
///
/// For each of `32³` cells, sample all `8³ = 512` RGB inputs the cell
/// covers, compute the true CIEDE2000-nearest palette index for each,
/// and union them into a per-cell candidate set. Returns CSR-style:
/// `(offsets, indices)` where `offsets[cell..=cell+1]` gives the
/// `indices` slice belonging to that cell.
///
/// Provably exact at u8 RGB resolution: every reachable u8 RGB is
/// sampled at codegen, so the runtime LUT path's small candidate scan
/// is guaranteed to contain the true nearest for any u8 RGB query.
///
/// Parallelizes across cells via rayon — each cell's work is
/// independent (512 full-scans of the 949-entry palette). On Apple
/// Silicon with 8 P-cores, total runtime is ~2.5 min in release mode.
fn compute_lut(palette_labs: &[[f32; 3]]) -> (Vec<u32>, Vec<u16>) {
  use std::collections::BTreeSet;

  use rayon::prelude::*;

  eprintln!(
    "  computing CIEDE2000 LUT: {N_CELLS} cells × 512 RGB inputs × {} palette entries...",
    palette_labs.len()
  );
  let start = std::time::Instant::now();

  let cells: Vec<Vec<u16>> = (0..N_CELLS)
    .into_par_iter()
    .map(|cell| {
      // Cell ID layout: top 5 bits of R | top 5 of G | top 5 of B.
      let cr = (cell >> 10) & 0x1F;
      let cg = (cell >> 5) & 0x1F;
      let cb = cell & 0x1F;

      let mut candidates = BTreeSet::<u16>::new();
      for dr in 0..8u32 {
        for dg in 0..8u32 {
          for db in 0..8u32 {
            let r = ((cr as u32) << 3 | dr) as u8;
            let g = ((cg as u32) << 3 | dg) as u8;
            let b = ((cb as u32) << 3 | db) as u8;
            let lab = rgb_to_lab([r, g, b]);
            let idx = nearest_palette_idx(lab, palette_labs);
            candidates.insert(idx);
          }
        }
      }
      candidates.into_iter().collect()
    })
    .collect();

  let elapsed = start.elapsed();
  let total: usize = cells.iter().map(Vec::len).sum();
  let max_cell = cells.iter().map(Vec::len).max().unwrap_or(0);
  let avg = total as f64 / N_CELLS as f64;
  eprintln!(
    "  LUT done in {:.2}s: {total} candidates total, avg {avg:.2}/cell, max {max_cell}/cell",
    elapsed.as_secs_f64()
  );

  // Build CSR from the per-cell vecs.
  let mut offsets = Vec::with_capacity(N_CELLS + 1);
  let mut indices = Vec::with_capacity(total);
  offsets.push(0u32);
  for cell_candidates in &cells {
    indices.extend_from_slice(cell_candidates);
    offsets.push(u32::try_from(indices.len()).expect("LUT index count must fit in u32"));
  }
  (offsets, indices)
}
