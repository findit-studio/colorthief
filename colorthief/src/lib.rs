#![doc = include_str!("../README.md")]
#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(docsrs, allow(unused_attributes))]
#![allow(clippy::new_without_default)]
#![deny(missing_docs)]
// `unsafe_code` is `deny`-not-`forbid` because `mmcq::simd` needs
// per-arch `#[target_feature(enable = ...)]` SIMD intrinsics, which
// require `unsafe`. Each arch submodule in `mmcq::simd` carries a
// local `#[allow(unsafe_code)]`; that's the only place unsafe is
// permitted. The `Mmcq::new_boxed` constructor also opts in via a
// scoped `#[allow(unsafe_code)]` for `Box::new_zeroed().assume_init()`.
#![deny(unsafe_code)]

#[cfg(all(feature = "alloc", not(feature = "std")))]
extern crate alloc as std;

#[cfg(feature = "std")]
extern crate std;

#[cfg(any(feature = "alloc", feature = "std"))]
use std::vec::Vec;

pub use colorthief_dataset::{Algorithm, Color, ColorId, Family, Kind};

mod buffer;
pub use buffer::Buffer;

mod mmcq;
pub use mmcq::Mmcq;

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
  pub(crate) rgb: [u8; 3],
  pub(crate) color: &'static Color,
  pub(crate) population: u32,
  pub(crate) percentage: f32,
}

impl Dominant {
  /// MMCQ-extracted dominant RGB (post-5-bit-bin centered).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn rgb(&self) -> [u8; 3] {
    self.rgb
  }

  /// Closest entry in the xkcd hierarchy to [`Self::rgb`], under
  /// the algorithm passed to [`extract_with`] (or
  /// [`Algorithm::default`] when [`extract`] is used).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn color(&self) -> &'static Color {
    self.color
  }

  /// Number of source-frame pixels assigned to this dominant's box.
  /// For an absolute "how many of the source pixels does this color
  /// account for" answer; see [`Self::percentage`] for the relative
  /// area.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn population(&self) -> u32 {
    self.population
  }

  /// Share of the source frame this dominant's box covers, as a
  /// percentage in `[0.0, 100.0]`. Computed at extraction time as
  /// `(population / total_frame_pixels) * 100`.
  ///
  /// Sums of percentages across one [`extract`] call are at most
  /// `100.0` and may be less when `count` truncates the dominant
  /// list before every box is emitted (the dropped boxes' pixels
  /// are simply unaccounted for in the returned set).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn percentage(&self) -> f32 {
    self.percentage
  }
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
  #[cfg_attr(not(tarpaulin), inline(always))]
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
  ///
  /// Public because `no_alloc` callers feed it into [`Mmcq::extract`]
  /// directly: `mmcq.extract(frame.pixels(), count, algo, &mut out)`.
  /// Alloc-tier callers use [`extract`] / [`extract_with`] which wrap
  /// this internally.
  pub fn pixels(&self) -> impl Iterator<Item = [u8; 3]> + '_ {
    let row_bytes = self.width as usize * 3;
    let stride = self.stride as usize;
    (0..self.height as usize).flat_map(move |row| {
      let start = row * stride;
      self.rgb[start..start + row_bytes]
        .as_chunks::<3>()
        .0
        .iter()
        .map(|c| [c[0], c[1], c[2]])
    })
  }
}

/// A validated borrow over a packed sRGB 16-bit-per-channel frame.
///
/// One plane, 3 u16 elements per pixel, channel order `R, G, B`.
/// Stride is in **u16 elements** (`>= 3 * width`) — matches the
/// buffer layout `colconv::sinker::MixedSinker::with_rgb_u16`
/// writes to, so HDR-source pipelines that produce u16 RGB output
/// can hand the buffer straight to colorthief.
///
/// # Downscale to u8
///
/// At pixel iteration time each channel is downscaled to u8 via the
/// natural `>> 8` truncation. For sRGB-encoded u16 (which is what
/// `with_rgb_u16` emits from sRGB-encoded sources), this is exactly
/// the inverse of `u8 << 8` and recovers the equivalent u8 sRGB
/// pixel — the dropped bottom 8 bits are sub-`ΔE_1` in LAB and
/// well below the perceptual threshold for color naming, plus
/// MMCQ's 5-bit binning further dwarfs the precision difference.
///
/// MMCQ and the CIEDE2000 LUT downstream operate on u8, so the
/// LUT's "provably exact at u8 RGB resolution" guarantee carries
/// through.
///
/// # Color space caveat
///
/// The downscale assumes sRGB-encoded u16 input. For BT.2020 PQ /
/// HLG sources, the upstream colconv sink must be configured to
/// produce sRGB-anchored RGB before reaching colorthief — the
/// `>> 8` truncation does **not** colour-space-convert.
#[derive(Debug, Clone, Copy)]
pub struct Rgb48Frame<'a> {
  rgb16: &'a [u16],
  width: u32,
  height: u32,
  stride: u32,
}

impl<'a> Rgb48Frame<'a> {
  /// Construct a new [`Rgb48Frame`], validating dimensions and plane
  /// length. The plane length and stride are measured in **u16
  /// elements**, not bytes.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn try_new(
    rgb16: &'a [u16],
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
    if rgb16.len() < plane_min {
      return Err(RgbFrameError::PlaneTooShort {
        expected: plane_min,
        actual: rgb16.len(),
      });
    }
    Ok(Self {
      rgb16,
      width,
      height,
      stride,
    })
  }

  /// 16-bit-per-channel R, G, B plane elements (interleaved per
  /// pixel).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn rgb16(&self) -> &'a [u16] {
    self.rgb16
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

  /// Stride in **u16 elements** (`>= 3 * width`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn stride(&self) -> u32 {
    self.stride
  }

  /// Iterate the frame's pixels as `[R, G, B]` u8 triples in raster
  /// order, downscaled from u16 via `>> 8`. Skips the row-padding
  /// u16 elements between `3 * width` and `stride`.
  ///
  /// Public for the same reason as [`RgbFrame::pixels`] — `no_alloc`
  /// callers feed it into [`Mmcq::extract`] directly.
  pub fn pixels(&self) -> impl Iterator<Item = [u8; 3]> + '_ {
    let row_u16 = self.width as usize * 3;
    let stride_u16 = self.stride as usize;
    (0..self.height as usize).flat_map(move |row| {
      let start = row * stride_u16;
      self.rgb16[start..start + row_u16]
        .as_chunks::<3>()
        .0
        .iter()
        .map(|p| [(p[0] >> 8) as u8, (p[1] >> 8) as u8, (p[2] >> 8) as u8])
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
/// Naming uses [`Algorithm::default`] — currently
/// [`Algorithm::Ciede2000Exact`], the modern perceptual gold-standard
/// (~71 µs/lookup scalar). Throughput-sensitive consumers should pick
/// a faster metric via [`extract_with`].
///
/// **Requires the `alloc` feature** — disabled in `no_std + no_alloc`
/// builds. For those, use `Mmcq::extract` with a caller-supplied
/// [`Buffer`] (once that lands in task #19).
#[cfg(any(feature = "alloc", feature = "std"))]
#[cfg_attr(docsrs, doc(cfg(any(feature = "alloc", feature = "std"))))]
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
#[cfg(any(feature = "alloc", feature = "std"))]
#[cfg_attr(docsrs, doc(cfg(any(feature = "alloc", feature = "std"))))]
pub fn extract_with(frame: RgbFrame<'_>, count: u8, algo: Algorithm) -> Vec<Dominant> {
  if count == 0 {
    return Vec::new();
  }
  extract_dominants_from_pixels(frame.pixels(), count, algo)
}

/// 16-bit-per-channel variant of [`extract`] for HDR sources. Mirrors
/// the colconv `MixedSinker::with_rgb_u16` output buffer. Each u16
/// channel is downscaled to u8 via `>> 8` at pixel iteration before
/// MMCQ — see [`Rgb48Frame`] for the rationale and color-space
/// caveat.
#[cfg(any(feature = "alloc", feature = "std"))]
#[cfg_attr(docsrs, doc(cfg(any(feature = "alloc", feature = "std"))))]
#[inline]
pub fn extract_rgb48(frame: Rgb48Frame<'_>, count: u8) -> Vec<Dominant> {
  extract_rgb48_with(frame, count, Algorithm::default())
}

/// 16-bit variant of [`extract_with`].
#[cfg(any(feature = "alloc", feature = "std"))]
#[cfg_attr(docsrs, doc(cfg(any(feature = "alloc", feature = "std"))))]
pub fn extract_rgb48_with(frame: Rgb48Frame<'_>, count: u8, algo: Algorithm) -> Vec<Dominant> {
  if count == 0 {
    return Vec::new();
  }
  extract_dominants_from_pixels(frame.pixels(), count, algo)
}

/// Common tail shared by both [`extract_with`] and
/// [`extract_rgb48_with`]: run MMCQ on the supplied pixel iterator
/// with the chosen algorithm and return the named-dominant `Vec`.
///
/// MMCQ's internal `target` is clamped to `[2, 256]` (the algorithm
/// is undefined outside that range), but the public `count` is the
/// hard upper bound `Mmcq::extract` honours via the `Buffer::try_push`
/// caller-side cap — `count = 1` returns at most 1 entry.
#[cfg(any(feature = "alloc", feature = "std"))]
#[cfg_attr(docsrs, doc(cfg(any(feature = "alloc", feature = "std"))))]
fn extract_dominants_from_pixels<I: Iterator<Item = [u8; 3]>>(
  pixels: I,
  count: u8,
  algo: Algorithm,
) -> Vec<Dominant> {
  mmcq::quantize(pixels, count, algo)
}
