//! Static xkcd color-hierarchy table for human-vocabulary color naming.
//!
//! Each entry exposes five hierarchical names (xkcd → design → common →
//! family → kind), plus the entry's RGB and a pre-computed CIE LAB triple
//! used by [`Color::nearest_to`] for nearest-neighbor lookup.
//!
//! # Why LAB is pre-computed at codegen time
//!
//! The sRGB → LAB pipeline involves a power function (gamma decode) and a
//! cube root (LAB f-function), both transcendental. Computing those for
//! all 950 entries on every query would dwarf the actual nearest-neighbor
//! scan. Pre-computing at `cargo xtask codegen` time pushes that cost out
//! of the hot path; the runtime cost of `nearest_to` is one sRGB→LAB on
//! the query plus 950 squared-distance comparisons (no `sqrt` — squared
//! distance preserves ordering).
//!
//! # Permanent ids
//!
//! Every entry carries a [`ColorId`] — a permanent, stable handle you can
//! store and resolve back to the entry with [`Color::from_id`]. The pair
//! ([`Color::id`], [`Color::from_id`]) is a bijection onto the assigned
//! ids, which is what lets a database, an index, or a wire format keep a
//! two-byte reference to a named color instead of minting an identifier
//! of its own.
//!
//! The ids obey one discipline, and it is the reason they are worth
//! storing:
//!
//! - An id is assigned once and **never changes**. Correcting an entry's
//!   name, design/common columns, RGB, hex, family or kind keeps its id.
//! - A **deleted entry's id is never reused**. It stays retired;
//!   [`Color::from_id`] returns `None` for it forever after.
//! - A **new entry mints a fresh id**, above every id ever assigned.
//!
//! Ids start at 1, so `ColorId::new(0)` is never a valid entry and a
//! zeroed column is always detectably wrong. The assignment lives in
//! `assets/color_ids.csv` and is pinned by the crate's tests: a
//! regeneration that renumbers anything fails them loudly.
//!
//! An id is **not** a position in [`COLORS`]. Retirements make the two
//! diverge, and only the id is stable across dataset revisions — index
//! into [`COLORS`] for iteration, store a [`ColorId`] for reference.
//!
//! # Distance metric
//!
//! Choose via [`Algorithm`]; the default ([`Algorithm::Ciede2000Exact`])
//! is the modern perceptual gold-standard. Faster alternatives are
//! available via [`Color::nearest_to`] (Delta E 76, ~470 ns NEON) and
//! [`Color::nearest_to_cie94`] (CIE94, ~620 ns NEON) for throughput-
//! sensitive callers willing to trade borderline accuracy for speed.
//!
//! # Attribution
//!
//! The color hierarchy is sourced from Stitch Fix's `colornamer` (Apache
//! 2.0); see `THIRD_PARTY_NOTICES.md` for the full upstream attribution.

#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(docsrs, allow(unused_attributes))]
#![deny(missing_docs)]
// `unsafe_code` is `deny`-not-`forbid` because the per-arch NEON kernel in
// `nearest::aarch64_neon` needs `unsafe` to call `core::arch::aarch64`
// intrinsics (the NEON entry function is `#[target_feature(enable = "neon")]`
// and therefore `unsafe fn`). That module carries a local
// `#[allow(unsafe_code)]` and is the ONLY place unsafe code is allowed.
#![deny(unsafe_code)]

mod generated;
mod nearest;

use generated::BY_ID;
pub use generated::{COLORS, Family, Kind};
// `Algorithm` and `ColorId` are defined below alongside `Color`;
// re-exported to crate root for ergonomics —
// `colorthief_dataset::Algorithm` is where users expect to find it.

/// **Not a stable API.**
///
/// Hidden helpers used by `colorthief-dataset/benches/nearest.rs` to
/// compare backends head-to-head. Calling these directly bypasses the
/// dispatcher in [`Color::nearest_to`] — production code should use
/// the public method.
#[doc(hidden)]
pub mod __bench {
  pub use crate::nearest::{
    cie94::{delta_e_94_sq, nearest_idx as cie94_nearest_idx},
    ciede2000::{
      delta_e_2000_sq, nearest_idx as ciede2000_nearest_idx,
      nearest_idx_prefiltered as ciede2000_prefiltered_nearest_idx,
    },
    scalar::{delta_e_76_sq, nearest_idx as scalar_nearest_idx},
  };

  #[cfg(feature = "lut")]
  pub use crate::nearest::ciede2000_lut::nearest_idx as ciede2000_lut_nearest_idx;

  // `target_feature = "neon"` (not just `target_arch = "aarch64"`):
  // see `nearest::aarch64_neon` mod decl for the
  // `aarch64-unknown-none-softfloat` rationale.
  #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
  pub use crate::nearest::cie94_aarch64_neon::nearest_idx as cie94_aarch64_neon_nearest_idx;
  #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
  pub use crate::nearest::cie94_wasm_simd128::nearest_idx as cie94_wasm_simd128_nearest_idx;
  #[cfg(target_arch = "x86_64")]
  pub use crate::nearest::cie94_x86_avx2::nearest_idx as cie94_x86_avx2_nearest_idx;
  #[cfg(target_arch = "x86_64")]
  pub use crate::nearest::cie94_x86_avx512::nearest_idx as cie94_x86_avx512_nearest_idx;
  #[cfg(target_arch = "x86_64")]
  pub use crate::nearest::cie94_x86_sse41::nearest_idx as cie94_x86_sse41_nearest_idx;

  #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
  pub use crate::nearest::aarch64_neon::nearest_idx as aarch64_neon_nearest_idx;

  // x86 backends are `unsafe fn` (the `#[target_feature]` attribute
  // enforces the safety boundary). Re-export them as-is — the bench
  // wraps the `unsafe { ... }` after a runtime feature check.
  #[cfg(target_arch = "x86_64")]
  pub use crate::nearest::x86_avx2::nearest_idx as x86_avx2_nearest_idx;
  #[cfg(target_arch = "x86_64")]
  pub use crate::nearest::x86_avx512::nearest_idx as x86_avx512_nearest_idx;
  #[cfg(target_arch = "x86_64")]
  pub use crate::nearest::x86_sse41::nearest_idx as x86_sse41_nearest_idx;

  #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
  pub use crate::nearest::wasm_simd128::nearest_idx as wasm_simd128_nearest_idx;

  /// Public re-export of the crate-private `rgb_to_lab` so benches can
  /// pre-convert RGB queries without duplicating the math.
  pub fn rgb_to_lab(rgb: [u8; 3]) -> [f32; 3] {
    super::rgb_to_lab(rgb)
  }
}

/// A permanent, stable handle on one [`Color`] in the dataset.
///
/// Store this — not a name, not a hex string, not a [`COLORS`] index —
/// when something outside the crate needs to refer to a palette entry
/// across time: a database column, a search index, a wire message.
/// [`Color::from_id`] resolves it back.
///
/// # The guarantee
///
/// An id is assigned once and never changes; a corrected entry keeps
/// its id, a retired entry's id is never handed out again, and a new
/// entry mints a fresh one. See the [crate docs](crate#permanent-ids)
/// for the full discipline.
///
/// Ids start at 1. `ColorId::new(0)` is well-formed but is never an
/// assigned id, so a zeroed column always fails to resolve rather than
/// silently naming the first entry.
///
/// # Not a validated type
///
/// Constructing a `ColorId` asserts nothing — any `u16` makes one, and
/// unassigned and retired ids are both representable. [`Color::from_id`]
/// is the total lookup that decides: it answers `None` for every id the
/// dataset does not carry.
///
/// ```
/// use colorthief_dataset::{Color, ColorId};
///
/// let entry = Color::all()[0];
/// let stored: u16 = entry.id().get();
///
/// // ... a round trip through a database column, a wire format, a file ...
/// let recovered = Color::from_id(ColorId::new(stored)).expect("assigned id");
/// assert_eq!(recovered.name(), entry.name());
///
/// // 0 is never assigned.
/// assert!(Color::from_id(ColorId::new(0)).is_none());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct ColorId(u16);

impl ColorId {
  /// Wrap a raw id — typically one read back out of storage.
  ///
  /// Performs no validation; see the [type docs](Self) for why, and
  /// [`Color::from_id`] for the lookup that does decide.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(id: u16) -> Self {
    Self(id)
  }

  /// The raw id, for handing to storage.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn get(&self) -> u16 {
    self.0
  }
}

impl core::fmt::Display for ColorId {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    core::fmt::Display::fmt(&self.0, f)
  }
}

/// One named entry in the xkcd color hierarchy.
///
/// Carries every column from the upstream `color_hierarchy.csv`:
/// xkcd / design / common name, hex, and RGB triples for each level,
/// plus the family / kind / neutrality classification. The xkcd LAB
/// triple is pre-computed at codegen time for nearest-neighbor lookup
/// in [`Self::nearest_to`], and a permanent [`ColorId`] for storing a
/// reference to the entry.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Color {
  pub(crate) id: ColorId,
  pub(crate) name: &'static str,
  pub(crate) hex: &'static str,
  pub(crate) rgb: [u8; 3],
  pub(crate) lab: [f32; 3],
  pub(crate) design_name: &'static str,
  pub(crate) design_hex: &'static str,
  pub(crate) design_rgb: [u8; 3],
  pub(crate) common_name: &'static str,
  pub(crate) common_hex: &'static str,
  pub(crate) common_rgb: [u8; 3],
  pub(crate) family: Family,
  pub(crate) kind: Kind,
  pub(crate) is_neutral: bool,
}

impl Color {
  /// This entry's permanent id — the handle to store when something
  /// outside the crate needs to name it later.
  ///
  /// Round-trips through [`Self::from_id`] for every entry in the
  /// dataset. See the [crate docs](crate#permanent-ids) for what the
  /// permanence guarantee does and does not cover.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn id(&self) -> ColorId {
    self.id
  }

  /// The entry carrying `id`, or `None` if the dataset has no such id.
  ///
  /// Total over every `u16`: unassigned ids, ids retired by a past
  /// dataset revision, and the reserved 0 all answer `None` rather than
  /// resolving to some neighbouring color.
  ///
  /// This is the reverse of [`Self::id`], and the pair is a bijection
  /// onto the assigned ids — `Color::from_id(c.id())` is `c` for every
  /// `c` in [`Color::all`], and an id that resolves always resolves to
  /// the entry that claims it.
  ///
  /// O(1): a bounds check and one load through a dense id → entry table
  /// generated alongside [`COLORS`], so resolving an id per row of a
  /// query result is free next to the nearest-neighbor scan that
  /// produced it.
  ///
  /// ```
  /// use colorthief_dataset::{Color, ColorId};
  ///
  /// let c = Color::nearest_to([189, 108, 72]);
  /// assert_eq!(Color::from_id(c.id()).map(Color::name), Some(c.name()));
  /// assert!(Color::from_id(ColorId::new(u16::MAX)).is_none());
  /// ```
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn from_id(id: ColorId) -> Option<&'static Color> {
    let slot = id.get() as usize;
    if slot < BY_ID.len() {
      BY_ID[slot]
    } else {
      None
    }
  }

  /// xkcd-survey name (~950 unique values, e.g. `"burnt orange"`,
  /// `"vermilion"`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn name(&self) -> &'static str {
    self.name
  }

  /// xkcd hex string, e.g. `"#bd6c48"`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn hex(&self) -> &'static str {
    self.hex
  }

  /// xkcd RGB triple, e.g. `[189, 108, 72]`. The exact 8-bit value the
  /// xkcd survey reports for this name.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn rgb(&self) -> [u8; 3] {
    self.rgb
  }

  /// Pre-computed CIE LAB (D65 illuminant, 2° observer) for [`Self::rgb`].
  /// Used internally by [`Self::nearest_to`]; exposed publicly so callers
  /// can implement their own distance metric (e.g. CIEDE2000) on top of
  /// the same cached values.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn lab(&self) -> [f32; 3] {
    self.lab
  }

  /// Coarser design-palette name (~250 unique, e.g. `"russet brown"`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn design_name(&self) -> &'static str {
    self.design_name
  }

  /// Hex string for the design-palette anchor color.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn design_hex(&self) -> &'static str {
    self.design_hex
  }

  /// RGB triple for the design-palette anchor color (the canonical
  /// 8-bit representation of [`Self::design_name`]). Differs from
  /// [`Self::rgb`] when the xkcd entry sits at the edge of its
  /// design-palette bucket.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn design_rgb(&self) -> [u8; 3] {
    self.design_rgb
  }

  /// Coarser still common name (~120 unique, e.g. `"sienna"`). The
  /// search-friendly default for indexing pipelines.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn common_name(&self) -> &'static str {
    self.common_name
  }

  /// Hex string for the common-name anchor color.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn common_hex(&self) -> &'static str {
    self.common_hex
  }

  /// RGB triple for the common-name anchor color.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn common_rgb(&self) -> [u8; 3] {
    self.common_rgb
  }

  /// Color family classification (26 values, e.g. [`Family::Yellow`],
  /// [`Family::BlueGreen`], [`Family::Neutral`]). Call
  /// [`Family::as_str`] for the original CSV string.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn family(&self) -> Family {
    self.family
  }

  /// Color kind / texture classification (11 values, e.g.
  /// [`Kind::NeonColor`], [`Kind::PainterlyNeutral`]). Call
  /// [`Kind::as_str`] for the original CSV string.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn kind(&self) -> Kind {
    self.kind
  }

  /// `true` if the entry is classified as a neutral (vs a chromatic
  /// color). Drives the `color_or_neutral` axis in the original Stitch
  /// Fix taxonomy.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn is_neutral(&self) -> bool {
    self.is_neutral
  }

  /// Every entry in the dataset, in CSV (alphabetical-by-`name`) order.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn all() -> &'static [&'static Color] {
    COLORS
  }

  /// Find the entry whose pre-computed LAB is closest to the given query
  /// RGB by **Delta E 76** (squared Euclidean LAB).
  ///
  /// Always returns an entry — `COLORS` is non-empty and verified at
  /// codegen time. The scan dispatches to a per-arch SIMD backend
  /// (NEON / AVX2 / SSE4.1 / WASM SIMD128) on every target that has
  /// one; other targets fall through to the scalar reference. Every
  /// backend is bit-identical — see [`crate::nearest`] for the
  /// dispatch contract and parity tests.
  ///
  /// # When to use this method
  ///
  /// Delta E 76 is the *fast* metric: against this crate's well-
  /// clustered 949-entry xkcd palette it picks the same named entry
  /// as CIEDE2000 in the overwhelming majority of cases, at ~150×
  /// the throughput of [`Algorithm::Ciede2000Exact`] (the default
  /// returned by [`Algorithm::default`]). Reach for it when you've
  /// measured the slower default bottlenecking real workloads and
  /// can tolerate borderline misnamings near the gray / yellow
  /// boundary.
  pub fn nearest_to(rgb: [u8; 3]) -> &'static Color {
    crate::nearest::nearest(rgb_to_lab(rgb))
  }

  /// Find the entry whose pre-computed LAB is closest to the given
  /// query RGB by **CIEDE2000** — the modern perceptual gold-standard
  /// colour-difference formula.
  ///
  /// CIEDE2000 corrects Delta E 76's known biases (over-weighting
  /// yellows, under-weighting blues, hue-rotation in the saturated
  /// blue region) at the cost of `atan2` / `sin` / `cos` / `exp` per
  /// pair plus branchy hue-wraparound logic.
  ///
  /// # Implementation
  ///
  /// - With `feature = "lut"` (the default): O(1) cell lookup → small
  ///   candidate scan via the pre-computed candidate-set LUT
  ///   ([`crate::nearest::ciede2000_lut`]). Provably exact at u8 RGB
  ///   resolution; ~few-hundred-ns/query.
  /// - Without `feature = "lut"`: full-scan reference over all 949
  ///   palette entries (~71 µs/query). Same correctness guarantee,
  ///   slower.
  ///
  /// Both modes are scalar — CIEDE2000's transcendentals don't
  /// vectorise usefully (see [`crate::nearest::ciede2000`]).
  pub fn nearest_to_ciede2000(rgb: [u8; 3]) -> &'static Color {
    crate::nearest::nearest_ciede2000(rgb)
  }

  /// Strict CIEDE2000 nearest-neighbor. Behaviorally equivalent to
  /// [`Self::nearest_to_ciede2000`] — both are exact under both
  /// feature configurations (the LUT path is provably exact, the
  /// no-LUT path is full-scan). Retained as a distinct entry point
  /// for API stability; consumers picking between the two by name
  /// should prefer [`Self::nearest_to_ciede2000`].
  pub fn nearest_to_ciede2000_exact(rgb: [u8; 3]) -> &'static Color {
    crate::nearest::nearest_ciede2000(rgb)
  }

  /// Find the entry whose pre-computed LAB is closest to the given
  /// query RGB by **CIE94** (Delta E 94, graphic-arts weighting).
  ///
  /// CIE94 sits between Delta E 76 and CIEDE2000 in both perceptual
  /// accuracy and arithmetic cost. It uses no transcendentals beyond
  /// `sqrt`, so unlike CIEDE2000 the formula vectorises cleanly —
  /// SIMD backends are a planned follow-up that will mirror the
  /// Delta E 76 module structure.
  ///
  /// CIE94 is **asymmetric** (the S_C / S_H scale factors depend on
  /// the reference's chroma C₁); this implementation treats the
  /// palette entry as the reference and the query as the sample.
  pub fn nearest_to_cie94(rgb: [u8; 3]) -> &'static Color {
    crate::nearest::nearest_cie94(rgb_to_lab(rgb))
  }
}

/// Color-difference algorithm used to map an arbitrary RGB query to
/// its nearest [`Color`] in the xkcd palette.
///
/// Each variant corresponds to one of the per-metric
/// `Color::nearest_to_*` methods. The enum exists so callers can
/// store the choice as a value and pass it to higher-level APIs like
/// `colorthief::extract_with` without committing to a specific
/// `Color::nearest_to_*` callsite.
///
/// Marked `#[non_exhaustive]` so adding a future variant (e.g. CMC,
/// CIEDE2010) is a non-breaking change for downstream consumers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
#[non_exhaustive]
#[repr(u8)]
pub enum Algorithm {
  /// **Delta E 76** — squared Euclidean LAB distance. Fastest by a
  /// wide margin (~470 ns/query on Apple Silicon NEON, ~940 ns
  /// scalar). SIMD-dispatched on every supported arch with bit-
  /// identical parity. Recommended for search-vocabulary indexing
  /// where throughput matters more than borderline accuracy.
  DeltaE76,

  /// **CIE94 (Delta E 94, graphic-arts weighting)** — middle ground
  /// between Delta E 76's speed and CIEDE2000's accuracy. Uses only
  /// `sqrt` for transcendentals, so it vectorises cleanly. ~900 ns
  /// NEON / ~4.4 µs scalar. Asymmetric: the palette entry is the
  /// reference, the query is the sample.
  Cie94,

  /// **CIEDE2000** — the modern perceptual gold-standard formula.
  /// With `feature = "lut"` (the default) routes through the
  /// candidate-set LUT (~230 ns/query, provably exact at u8 RGB);
  /// without the feature falls back to the full-scan reference
  /// (~71 µs/query, also provably exact). Behaviorally equivalent
  /// to [`Self::Ciede2000Exact`] under both feature configurations.
  Ciede2000,

  /// **CIEDE2000**, retained as a distinct variant for API
  /// stability. Behaviorally equivalent to [`Self::Ciede2000`] —
  /// both go through the LUT when `feature = "lut"` is enabled and
  /// the full-scan reference otherwise. **Default.** Returned by
  /// [`Algorithm::default`] so consumers of [`Algorithm::extract`]
  /// and crate-level entry points like `colorthief::extract` get
  /// the perceptual gold-standard out of the box.
  #[default]
  Ciede2000Exact,
}

impl Algorithm {
  /// Find the [`Color`] whose pre-computed LAB is closest to the
  /// given RGB under this algorithm's distance metric.
  ///
  /// Equivalent to dispatching to the corresponding
  /// `Color::nearest_to*` method by hand:
  ///
  /// | Variant                    | Equivalent call                                |
  /// |----------------------------|------------------------------------------------|
  /// | [`Self::DeltaE76`]         | [`Color::nearest_to`]                          |
  /// | [`Self::Cie94`]            | [`Color::nearest_to_cie94`]                    |
  /// | [`Self::Ciede2000`]        | [`Color::nearest_to_ciede2000`]                |
  /// | [`Self::Ciede2000Exact`]   | [`Color::nearest_to_ciede2000_exact`]          |
  #[inline]
  pub fn extract(&self, rgb: [u8; 3]) -> &'static Color {
    match self {
      Self::DeltaE76 => Color::nearest_to(rgb),
      Self::Cie94 => Color::nearest_to_cie94(rgb),
      Self::Ciede2000 => Color::nearest_to_ciede2000(rgb),
      Self::Ciede2000Exact => Color::nearest_to_ciede2000_exact(rgb),
    }
  }
}

impl core::fmt::Display for Algorithm {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.write_str(self.as_str())
  }
}

// R1 Codex finding (medium): `Serialize` delegated to the exhaustive
// `as_str` match, but `FromStr` (an `if`/`else` chain) and `ROSTER` (a
// plain array) were two MORE hand-written copies of the same variant
// list. `as_str`'s match is compiler-checked exhaustive over `Algorithm`
// (matches within the defining crate are exhaustive regardless of
// `#[non_exhaustive]`), so a new variant forces it to be updated — but
// nothing forced `FromStr` or `ROSTER` to follow. A new variant could
// compile, serialize its word via `Serialize` (which calls `as_str`),
// and then fail to parse that same word back via `FromStr`/`Deserialize`
// — a one-way trip through JSON/YAML/TOML.
//
// Fix: one authoritative variant-to-word list, generating all three
// faces so they cannot disagree. `Family`/`Kind::as_str` (this crate's
// existing precedent) get their single-source guarantee from the xtask
// codegen tool reading `assets/color_hierarchy.csv` once and emitting
// every derived face into `generated.rs`; `Algorithm` is a small,
// hand-authored, non-generated enum, so the equivalent here is a local
// declarative macro rather than a build-time tool — same principle
// (generate every face from one list), sized to a four-entry enum
// instead of a 949-row CSV.
// R2 Codex finding (medium): the R1 fix forces every VARIANT to have a
// word (an omitted one fails to compile), but nothing forced the WORDS
// to be pairwise distinct. `NewVariant => "CIE94"` beside the existing
// `Cie94 => "cie94"` would compile: `Serialize` emits `"CIE94"` for
// `NewVariant`, but `FromStr`'s `if`/`else` chain checks `Cie94` first
// (`"CIE94".eq_ignore_ascii_case("cie94")` is `true`) and silently
// returns `Cie94` instead — a variant substitution, not a parse error,
// on the JSON/YAML/TOML round trip.
//
// Fix: two compile-time assertions make `algorithm_words!`'s mapping an
// actual bijection instead of merely a total function. Both run via
// `const _: () = assert!(...)`, a `no_std`-safe pattern — a
// `const`-context panic is a compiler diagnostic, never a runtime one,
// so it needs no panic handler.
const fn str_eq_ignore_ascii_case(a: &str, b: &str) -> bool {
  // `str::eq_ignore_ascii_case` is not a `const fn` (as of this crate's
  // MSRV), so the same byte-for-byte comparison is spelled out by hand
  // here for use inside a `const` context — but each byte still goes
  // through `u8::eq_ignore_ascii_case` (also `const fn`) rather than a
  // manual `to_ascii_lowercase` compare, per `str::as_bytes`/`[u8]`
  // indexing, which are `const fn` too.
  let a = a.as_bytes();
  let b = b.as_bytes();
  if a.len() != b.len() {
    return false;
  }
  let mut i = 0;
  while i < a.len() {
    if !a[i].eq_ignore_ascii_case(&b[i]) {
      return false;
    }
    i += 1;
  }
  true
}

/// `true` if no two `words` are equal under ASCII case-insensitive
/// comparison — the same equality [`core::str::FromStr`] uses to accept
/// a word. Backs `algorithm_words!`'s compile-time check that two
/// variants can never share a word: if they did, `FromStr` would
/// silently return whichever variant's `if` branch runs first, no
/// matter which variant `Serialize` actually emitted the word for.
const fn pairwise_unique_ignore_ascii_case(words: &[&str]) -> bool {
  let mut i = 0;
  while i < words.len() {
    let mut j = i + 1;
    while j < words.len() {
      if str_eq_ignore_ascii_case(words[i], words[j]) {
        return false;
      }
      j += 1;
    }
    i += 1;
  }
  true
}

/// `true` if no two `u8` values repeat. Backs `algorithm_words!`'s
/// compile-time check that no `Algorithm` variant identifier is listed
/// twice: since `Algorithm` is `#[repr(u8)]`, two entries naming the
/// same variant produce the same discriminant here, independently of
/// whether the resulting duplicate `as_str` match arm also happens to
/// be caught by the `unreachable_patterns` lint (which this workspace
/// only denies under `-D warnings`, not by default).
const fn pairwise_unique_u8(values: &[u8]) -> bool {
  let mut i = 0;
  while i < values.len() {
    let mut j = i + 1;
    while j < values.len() {
      if values[i] == values[j] {
        return false;
      }
      j += 1;
    }
    i += 1;
  }
  true
}

/// Declares every `Algorithm` variant and its [`Algorithm::as_str`] word
/// exactly once, and generates the three faces that must agree with each
/// other from that single list: `as_str` (still an ordinary `match`, so
/// this crate fails to compile the moment a variant is added here
/// without a word), [`core::str::FromStr`], and [`ROSTER`].
///
/// Two compile-time assertions make the variant/word mapping an actual
/// bijection: every word belongs to exactly one variant (enforced by
/// [`pairwise_unique_ignore_ascii_case`] over the same case-insensitive
/// equality `FromStr` parses with), and no variant is listed twice
/// (enforced by [`pairwise_unique_u8`] over the `#[repr(u8)]`
/// discriminants, independent of `unreachable_patterns` lint level).
macro_rules! algorithm_words {
  ($( $variant:ident => $word:literal ),+ $(,)?) => {
    impl Algorithm {
      /// Stable string identifier for this algorithm — useful for log
      /// lines, telemetry, and search-index metadata. Mirrors the
      /// [`Family::as_str`] / [`Kind::as_str`] convention.
      #[inline]
      pub const fn as_str(&self) -> &'static str {
        match self {
          $( Self::$variant => $word, )+
        }
      }
    }

    impl core::str::FromStr for Algorithm {
      type Err = ParseAlgorithmError;

      fn from_str(s: &str) -> Result<Self, Self::Err> {
        $(
          if s.eq_ignore_ascii_case($word) {
            return Ok(Self::$variant);
          }
        )+
        Err(ParseAlgorithmError(()))
      }
    }

    /// Every accepted [`Algorithm::as_str`] word, in declaration order —
    /// generated by `algorithm_words!` from the same list as `as_str`
    /// and [`core::str::FromStr`]. Backs [`ParseAlgorithmError`]'s
    /// message and, behind `feature = "serde"`, serde's own
    /// `unknown_variant` refusal.
    const ROSTER: &[&str] = &[ $( $word ),+ ];

    const _: () = assert!(
      pairwise_unique_ignore_ascii_case(ROSTER),
      "algorithm_words!: words must be unique ignoring ASCII case (FromStr could not tell them apart)",
    );

    const _: () = assert!(
      pairwise_unique_u8(&[ $( Algorithm::$variant as u8 ),+ ]),
      "algorithm_words!: the same Algorithm variant was listed more than once",
    );
  };
}

algorithm_words! {
  DeltaE76 => "delta-e-76",
  Cie94 => "cie94",
  Ciede2000 => "ciede2000",
  Ciede2000Exact => "ciede2000-exact",
}

/// [`Algorithm::from_str`] rejects anything but the four
/// [`Algorithm::as_str`] words (ASCII case-insensitive).
///
/// Deliberately opaque: naming the caller's rejected word back would
/// need an owned copy — the input `&str`'s lifetime cannot flow through
/// [`core::str::FromStr::Err`], so holding it at all means allocating
/// one, which this type (`Copy`, feature-free, `no_std`) does not do.
/// [`Display`](core::fmt::Display) names the accepted roster instead —
/// `'static`, so no allocation is needed to hold or hand it out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParseAlgorithmError(());

impl core::fmt::Display for ParseAlgorithmError {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.write_str("unknown algorithm; expected one of ")?;
    for (i, word) in ROSTER.iter().enumerate() {
      if i > 0 {
        f.write_str(", ")?;
      }
      write!(f, "{word:?}")?;
    }
    Ok(())
  }
}

impl core::error::Error for ParseAlgorithmError {}

#[cfg(feature = "serde")]
#[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
impl serde::Serialize for Algorithm {
  fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
  where
    S: serde::Serializer,
  {
    serializer.serialize_str(self.as_str())
  }
}

#[cfg(feature = "serde")]
#[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
impl<'de> serde::Deserialize<'de> for Algorithm {
  fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
  where
    D: serde::Deserializer<'de>,
  {
    struct AlgorithmVisitor;

    impl serde::de::Visitor<'_> for AlgorithmVisitor {
      type Value = Algorithm;

      fn expecting(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("an algorithm name")
      }

      fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
      where
        E: serde::de::Error,
      {
        v.parse().map_err(|_| E::unknown_variant(v, ROSTER))
      }
    }

    deserializer.deserialize_str(AlgorithmVisitor)
  }
}

/// Convert sRGB byte triple → CIE LAB (D65 illuminant, 2° observer).
///
/// Pipeline: byte → normalized [0, 1] → linearized (sRGB EOTF) → XYZ
/// (sRGB→XYZ matrix, D65) → LAB (CIE 1976 transfer).
pub(crate) fn rgb_to_lab(rgb: [u8; 3]) -> [f32; 3] {
  let r = srgb_to_linear(rgb[0] as f32 / 255.0);
  let g = srgb_to_linear(rgb[1] as f32 / 255.0);
  let b = srgb_to_linear(rgb[2] as f32 / 255.0);

  // sRGB → XYZ (D65, 2°). Coefficients from IEC 61966-2-1, rounded to
  // f32 precision (~7 decimal digits — trailing zeros that clippy flagged
  // as excessive precision are dropped here).
  let x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375;
  let y = r * 0.2126729 + g * 0.7151522 + b * 0.072175;
  let z = r * 0.0193339 + g * 0.119192 + b * 0.9503041;

  // D65 reference white (CIE 1931, 2°).
  const XN: f32 = 0.95047;
  const YN: f32 = 1.00000;
  const ZN: f32 = 1.08883;

  let fx = lab_f(x / XN);
  let fy = lab_f(y / YN);
  let fz = lab_f(z / ZN);

  let l = 116.0 * fy - 16.0;
  let a = 500.0 * (fx - fy);
  let b_lab = 200.0 * (fy - fz);
  [l, a, b_lab]
}

/// sRGB electro-optical transfer function (gamma decode).
fn srgb_to_linear(c: f32) -> f32 {
  if c <= 0.04045 {
    c / 12.92
  } else {
    libm::powf((c + 0.055) / 1.055, 2.4)
  }
}

/// CIE LAB transfer function (`f` in the standard).
fn lab_f(t: f32) -> f32 {
  // delta = 6/29; threshold where the linear/cube-root segments meet.
  const DELTA_CUBED: f32 = 216.0 / 24389.0; // (6/29)^3
  const KAPPA_OVER_3: f32 = 841.0 / 108.0; // 1 / (3 * (6/29)^2)
  const OFFSET: f32 = 4.0 / 29.0;
  if t > DELTA_CUBED {
    libm::cbrtf(t)
  } else {
    KAPPA_OVER_3 * t + OFFSET
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  /// The dataset must always be non-empty (xtask validates at codegen
  /// time, but a runtime smoke test catches a regression where someone
  /// edits `generated.rs` by hand).
  #[test]
  fn dataset_is_non_empty() {
    assert!(!COLORS.is_empty());
  }

  /// Snapshot the entry count. Acts as a canary if upstream colornamer
  /// updates the CSV: a count change is a deliberate regen, not a silent
  /// drift. Updating this number is the right action when intentional.
  ///
  /// A count change means entries arrived or left, so `tests/ids.rs`
  /// will need updating too — check there that the ids of everything
  /// that stayed did not move.
  #[test]
  fn dataset_entry_count_matches_csv() {
    assert_eq!(
      COLORS.len(),
      949,
      "regenerate via `cargo xtask codegen` if the upstream CSV changed",
    );
  }

  /// Pure sRGB red must map to an entry that's at least red-flavored.
  /// Strict equality on the xkcd label is fragile (the closest match
  /// depends on the palette and could be `"red"`, `"bright red"`,
  /// `"true red"`, etc.); the family is a stable invariant.
  #[test]
  fn nearest_to_pure_red_is_in_red_family() {
    let c = Color::nearest_to([255, 0, 0]);
    assert!(
      c.family().as_str().contains("red") || c.name().contains("red"),
      "nearest to pure red was name={:?} family={:?}",
      c.name(),
      c.family().as_str(),
    );
  }

  /// Pure sRGB blue must map to a blue-family entry.
  #[test]
  fn nearest_to_pure_blue_is_in_blue_family() {
    let c = Color::nearest_to([0, 0, 255]);
    assert!(
      c.family().as_str().contains("blue") || c.name().contains("blue"),
      "nearest to pure blue was name={:?} family={:?}",
      c.name(),
      c.family().as_str(),
    );
  }

  /// Mid-gray must map to a neutral. Tests that the `is_neutral` axis
  /// reaches readers correctly via the lookup path.
  #[test]
  fn nearest_to_mid_gray_is_neutral() {
    let c = Color::nearest_to([128, 128, 128]);
    assert!(
      c.is_neutral(),
      "nearest to (128,128,128) was name={:?} is_neutral={}",
      c.name(),
      c.is_neutral(),
    );
  }

  /// sRGB → LAB on the D65 reference white must produce L=100, a=0, b=0
  /// to within float tolerance.
  #[test]
  fn rgb_to_lab_d65_white() {
    let lab = rgb_to_lab([255, 255, 255]);
    assert!((lab[0] - 100.0).abs() < 0.01, "L was {}", lab[0]);
    assert!(lab[1].abs() < 0.01, "a was {}", lab[1]);
    assert!(lab[2].abs() < 0.01, "b was {}", lab[2]);
  }

  /// sRGB → LAB on absolute black must produce L=0, a=0, b=0 exactly
  /// (the linear segment of the sRGB EOTF carries 0 through unchanged).
  #[test]
  fn rgb_to_lab_black() {
    let lab = rgb_to_lab([0, 0, 0]);
    assert!(lab[0].abs() < 1e-6, "L was {}", lab[0]);
    assert!(lab[1].abs() < 1e-6, "a was {}", lab[1]);
    assert!(lab[2].abs() < 1e-6, "b was {}", lab[2]);
  }

  const ALL_ALGORITHMS: [Algorithm; 4] = [
    Algorithm::DeltaE76,
    Algorithm::Cie94,
    Algorithm::Ciede2000,
    Algorithm::Ciede2000Exact,
  ];

  /// Feature-free: `Display` is exactly `as_str`, for every variant.
  #[test]
  fn algorithm_display_matches_as_str() {
    for algo in ALL_ALGORITHMS {
      assert_eq!(algo.to_string(), algo.as_str());
    }
  }

  /// Feature-free: every `as_str` word parses back to its variant.
  #[test]
  fn algorithm_fromstr_roundtrips_all_words() {
    for algo in ALL_ALGORITHMS {
      assert_eq!(algo.as_str().parse::<Algorithm>().unwrap(), algo);
    }
  }

  /// Feature-free: parsing ignores ASCII case.
  #[test]
  fn algorithm_fromstr_is_case_insensitive() {
    assert_eq!(
      "DELTA-E-76".parse::<Algorithm>().unwrap(),
      Algorithm::DeltaE76
    );
    assert_eq!("Cie94".parse::<Algorithm>().unwrap(), Algorithm::Cie94);
    assert_eq!(
      "CIEDE2000".parse::<Algorithm>().unwrap(),
      Algorithm::Ciede2000
    );
    assert_eq!(
      "Ciede2000-Exact".parse::<Algorithm>().unwrap(),
      Algorithm::Ciede2000Exact,
    );
  }

  /// Feature-free: a near-miss is rejected, and the refusal names every
  /// word in the roster rather than echoing the rejected input back
  /// (see [`ParseAlgorithmError`] for why).
  #[test]
  fn algorithm_fromstr_rejects_unknown_names_roster() {
    let message = "ciede-2000".parse::<Algorithm>().unwrap_err().to_string();
    for &word in ROSTER {
      assert!(message.contains(word), "{message:?} does not name {word:?}");
    }
  }

  /// Guards `algorithm_words!`'s completeness independently of the
  /// macro-generated `as_str` match: the inline match below fails to
  /// compile the moment `Algorithm` gains a variant that
  /// `algorithm_words!` (and therefore `ROSTER`) does not — the same
  /// class of gap R1's Codex review found (`as_str` alone updated,
  /// `FromStr`/`ROSTER` silently left behind).
  #[test]
  fn algorithm_roster_covers_every_variant() {
    for algo in ALL_ALGORITHMS {
      match algo {
        Algorithm::DeltaE76
        | Algorithm::Cie94
        | Algorithm::Ciede2000
        | Algorithm::Ciede2000Exact => {}
      }
    }
    assert_eq!(ROSTER.len(), ALL_ALGORITHMS.len());
  }

  /// The predicate behind `algorithm_words!`'s compile-time
  /// word-uniqueness check (R2 Codex finding), exercised directly since
  /// the crate has no `trybuild`-style compile-fail test infrastructure
  /// to exercise the `const` assertion itself.
  #[test]
  fn algorithm_word_uniqueness_helper_is_correct() {
    assert!(!pairwise_unique_ignore_ascii_case(&["cie94", "CIE94"]));
    assert!(pairwise_unique_ignore_ascii_case(ROSTER));
  }

  /// The predicate behind `algorithm_words!`'s compile-time
  /// variant-uniqueness check, exercised directly for the same reason.
  #[test]
  fn algorithm_discriminant_uniqueness_helper_is_correct() {
    assert!(!pairwise_unique_u8(&[1, 2, 2, 3]));
    assert!(pairwise_unique_u8(&[0, 1, 2, 3]));
  }

  /// `feature = "serde"`: every word round-trips through JSON as the
  /// plain `as_str` string.
  #[test]
  #[cfg(feature = "serde")]
  fn algorithm_serde_json_roundtrips_all_words() {
    for algo in ALL_ALGORITHMS {
      let json = serde_json::to_string(&algo).unwrap();
      assert_eq!(json, format!("\"{}\"", algo.as_str()));
      assert_eq!(serde_json::from_str::<Algorithm>(&json).unwrap(), algo);
    }
  }

  /// `feature = "serde"`: an unrecognized JSON string is refused via
  /// serde's own `unknown_variant`, which also names the roster.
  #[test]
  #[cfg(feature = "serde")]
  fn algorithm_serde_json_rejects_unknown_names_roster() {
    let message = serde_json::from_str::<Algorithm>("\"ciede-2000\"")
      .unwrap_err()
      .to_string();
    for &word in ROSTER {
      assert!(message.contains(word), "{message:?} does not name {word:?}");
    }
  }
}
