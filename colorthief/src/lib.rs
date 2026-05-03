//! Dominant-color extraction (MMCQ) + human-vocabulary naming for packed
//! sRGB video keyframes.
//!
//! # Pipeline
//!
//! 1. [`extract`] runs MMCQ (Modified Median Cut Quantization) over the
//!    pixels of an [`RgbFrame`], producing up to `count` dominant RGB
//!    values plus the pixel population behind each.
//! 2. Each dominant is mapped to its nearest entry in
//!    [`colorthief_dataset`]'s xkcd-hierarchy table via Delta E 76 (LAB
//!    Euclidean) distance.
//!
//! The result is a `Vec<`[`Dominant`]`>` carrying both the actual
//! extracted RGB (for swatch rendering) and the named [`Color`] (for
//! search-vocabulary), sorted descending by `population`. The caller
//! picks which name level (`name()`, `common_name()`, `family()`, …)
//! is right for their indexing needs.
//!
//! # Frame input
//!
//! [`RgbFrame`] is a borrowed-byte-slice newtype shaped like
//! `colconv::Rgb24Frame`: packed 8-bit RGB, one plane, `R, G, B` byte
//! order, byte stride ≥ `3 * width`. **Not** a `colconv` re-export —
//! that crate is GPL-3.0-or-later and we keep this workspace under
//! MIT/Apache. Bridge from a real `colconv::Rgb24Frame` at the call
//! site:
//!
//! ```ignore
//! let frame = colorthief::RgbFrame::try_new(
//!     rgb24.rgb(), rgb24.width(), rgb24.height(), rgb24.stride()
//! )?;
//! let colors = colorthief::extract(&frame, 5);
//! ```
//!
//! [`colorthief_dataset`]: ../colorthief_dataset/index.html

#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(docsrs, allow(unused_attributes))]
#![deny(missing_docs)]
// `unsafe_code` is `deny`-not-`forbid` because `mmcq::simd` needs
// per-arch `#[target_feature(enable = ...)]` SIMD intrinsics, which
// require `unsafe`. Each arch submodule in `mmcq::simd` carries a
// local `#[allow(unsafe_code)]`; that's the only place unsafe is
// permitted.
#![deny(unsafe_code)]

pub use colorthief_dataset::{Algorithm, Color, Family, Kind};

mod mmcq;

use thiserror::Error;

/// One entry in an [`extract`] result: the actual MMCQ-extracted RGB,
/// the closest xkcd-hierarchy [`Color`] for naming, and the pixel-count
/// weight behind that color.
///
/// `rgb` is what MMCQ produced from the source frame (post-5-bit
/// quantization average mapped back to 8-bit, so it round-trips to the
/// nearest bin-center step). `color` is the closest xkcd entry to that
/// RGB — its `rgb()` will differ slightly because it's snapped to the
/// named-color palette.
///
/// Use `rgb` for rendering swatches that match the source frame; use
/// `color.name()` / `.common_name()` / `.family()` etc. for search-
/// index vocabulary. `population` is the relative weight, useful for
/// ranking, merging across frames, or thresholding visual significance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dominant {
  /// MMCQ-extracted dominant RGB.
  pub rgb: [u8; 3],
  /// Closest entry in the xkcd hierarchy to `rgb`, by Delta E 76 in
  /// LAB space.
  pub color: &'static Color,
  /// Number of source-frame pixels assigned to this dominant's box.
  pub population: u32,
}

/// Errors returned by [`RgbFrame::try_new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Error)]
#[non_exhaustive]
pub enum RgbFrameError {
  /// `width` or `height` was zero.
  #[error("width ({width}) or height ({height}) is zero")]
  ZeroDimension {
    /// The supplied width.
    width: u32,
    /// The supplied height.
    height: u32,
  },
  /// `stride < 3 * width`.
  #[error("stride ({stride}) is smaller than 3 * width ({min_stride})")]
  StrideTooSmall {
    /// Required minimum stride (`3 * width`).
    min_stride: u32,
    /// The supplied stride.
    stride: u32,
  },
  /// Plane is shorter than `stride * height` bytes.
  #[error("RGB plane has {actual} bytes but at least {expected} are required")]
  PlaneTooShort {
    /// Minimum bytes required.
    expected: usize,
    /// Actual bytes supplied.
    actual: usize,
  },
  /// `stride * height` overflows `usize`.
  #[error("declared geometry overflows usize: stride={stride} * rows={rows}")]
  GeometryOverflow {
    /// Stride that overflowed.
    stride: u32,
    /// Row count that overflowed against the stride.
    rows: u32,
  },
  /// `3 * width` overflows `u32`.
  #[error("3 * width overflows u32 ({width} too large)")]
  WidthOverflow {
    /// The supplied width.
    width: u32,
  },
}

/// A validated borrow over a packed sRGB 8-bit frame.
///
/// One plane, 3 bytes per pixel, byte order `R, G, B`. Byte stride
/// (`>= 3 * width`) lets the caller pass FFmpeg-style padded frames
/// without copying. Shape mirrors `colconv::Rgb24Frame` field-for-field.
#[derive(Debug, Clone, Copy)]
pub struct RgbFrame<'a> {
  rgb: &'a [u8],
  width: u32,
  height: u32,
  stride: u32,
}

impl<'a> RgbFrame<'a> {
  /// Construct a new [`RgbFrame`], validating dimensions and plane
  /// length.
  pub const fn try_new(
    rgb: &'a [u8],
    width: u32,
    height: u32,
    stride: u32,
  ) -> Result<Self, RgbFrameError> {
    if width == 0 || height == 0 {
      return Err(RgbFrameError::ZeroDimension { width, height });
    }
    let min_stride = match width.checked_mul(3) {
      Some(v) => v,
      None => return Err(RgbFrameError::WidthOverflow { width }),
    };
    if stride < min_stride {
      return Err(RgbFrameError::StrideTooSmall { min_stride, stride });
    }
    let plane_min = match (stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(RgbFrameError::GeometryOverflow {
          stride,
          rows: height,
        });
      }
    };
    if rgb.len() < plane_min {
      return Err(RgbFrameError::PlaneTooShort {
        expected: plane_min,
        actual: rgb.len(),
      });
    }
    Ok(Self {
      rgb,
      width,
      height,
      stride,
    })
  }

  /// Packed RGB plane bytes (`R, G, B, R, G, B, …` per row).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn rgb(&self) -> &'a [u8] {
    self.rgb
  }

  /// Frame width in pixels.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn width(&self) -> u32 {
    self.width
  }

  /// Frame height in pixels.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn height(&self) -> u32 {
    self.height
  }

  /// Byte stride (`>= 3 * width`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn stride(&self) -> u32 {
    self.stride
  }

  /// Iterate the frame's pixels as `[R, G, B]` triples in raster order.
  /// Skips the row-padding bytes between `3 * width` and `stride`.
  pub(crate) fn pixels(&self) -> impl Iterator<Item = [u8; 3]> + '_ {
    let row_bytes = self.width as usize * 3;
    let stride = self.stride as usize;
    (0..self.height as usize).flat_map(move |row| {
      let start = row * stride;
      self.rgb[start..start + row_bytes]
        .chunks_exact(3)
        .map(|c| [c[0], c[1], c[2]])
    })
  }
}

/// Extract up to `count` dominant colors from `frame`, each mapped to
/// its nearest entry in the xkcd color hierarchy and weighted by the
/// number of source pixels behind it.
///
/// Returns fewer than `count` entries if the frame has fewer distinct
/// colors than requested. Returns an empty `Vec` when `count == 0`.
/// The returned `Vec` is sorted descending by `population` so
/// `extract(...)[0]` is always the most-dominant color.
///
/// MMCQ's internal `target` is clamped to `[2, 256]` (the algorithm
/// is undefined outside that range), but the public `count` is
/// honoured as a strict upper bound — the result is truncated to
/// `count` so `extract(frame, 1)` returns at most one entry.
///
/// Naming uses [`Algorithm::default`] (Delta E 76, SIMD-dispatched).
/// To pick a different metric explicitly use [`extract_with`].
#[inline]
pub fn extract(frame: RgbFrame<'_>, count: u8) -> Vec<Dominant> {
  extract_with(frame, count, Algorithm::default())
}

/// Same as [`extract`] but the per-dominant naming step uses the
/// algorithm specified by `algo`. See [`Algorithm`] for the variants
/// and their speed/accuracy trade-offs.
///
/// The MMCQ extraction stage is identical regardless of `algo`; only
/// the RGB → named-`Color` lookup differs.
pub fn extract_with(frame: RgbFrame<'_>, count: u8, algo: Algorithm) -> Vec<Dominant> {
  if count == 0 {
    return Vec::new();
  }
  let mut dominants: Vec<Dominant> = mmcq::quantize(frame, count)
    .into_iter()
    .map(|d| Dominant {
      rgb: d.rgb,
      color: algo.extract(d.rgb),
      population: d.population,
    })
    .collect();
  // MMCQ runs with an internal target of `max(count, 2)` — clamping
  // up keeps the algorithm well-defined. Truncate the (already
  // population-descending) output back to the public `count` cap so
  // callers using `count` as a hard top-N bound never see overrun.
  // (Codex adversarial review, 2026-05-02.)
  dominants.truncate(count as usize);
  dominants
}
