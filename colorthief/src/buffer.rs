//! Output buffer abstraction for `Mmcq::extract` and the alloc-tier
//! `extract_into*` family.
//!
//! [`Buffer<T>`] lets callers plug any container with push semantics
//! into colorthief's extract pipeline:
//!
//! - `Vec<T>` — unbounded, requires `feature = "alloc"`.
//! - `[Option<T>; N]` — fixed-capacity, no allocator. Linear-scan
//!   push is O(N) — suitable for the typical `count ≤ 32` regime;
//!   for larger N use a length-tracking wrapper.
//! - `&mut [Option<T>]` — same shape, borrowed.
//! - Any other type the user wants to plug in (e.g.
//!   `heapless::Vec<T, N>`, `arrayvec::ArrayVec<T, N>`) via a
//!   one-line `impl Buffer<T>`.

/// Push-shaped output buffer.
///
/// On success [`Buffer::try_push`] returns `None`; on overflow it
/// returns `Some(val)` so the caller can recover the value (typical
/// pattern: stop the producer loop on first `Some`). The method is
/// named `try_push` rather than `push` because `Vec` already has an
/// inherent `push(&mut self, T) -> ()` — Rust's method resolution
/// would otherwise pick the inherent `()` return over the trait's
/// `Option<T>`.
pub trait Buffer<T> {
  /// Compile-time capacity hint. `None` for unbounded buffers
  /// (e.g. `Vec`); `Some(n)` for fixed-capacity buffers
  /// (e.g. `[Option<T>; N]`).
  ///
  /// Internal callers can use this to pre-cap target counts and avoid
  /// generating dominants that would just be rejected. Slice-shaped
  /// buffers whose capacity is only known at runtime report `None`
  /// even though they are bounded — `try_push` is still the source of
  /// truth.
  const MAX_CAP: Option<usize>;

  /// Push `val` to the buffer.
  ///
  /// Returns `None` on success. Returns `Some(val)` if the buffer is
  /// full, handing the value back to the caller. The buffer is
  /// otherwise unchanged.
  fn try_push(&mut self, val: T) -> Option<T>;
}

#[cfg(any(feature = "alloc", feature = "std"))]
#[cfg_attr(docsrs, doc(cfg(any(feature = "alloc", feature = "std"))))]
impl<T> Buffer<T> for std::vec::Vec<T> {
  const MAX_CAP: Option<usize> = None;

  #[inline]
  fn try_push(&mut self, val: T) -> Option<T> {
    std::vec::Vec::push(self, val);
    None
  }
}

/// Fixed-size buffer using `Option<T>` slots. `push` linear-scans for
/// the first `None` and inserts there — O(N) per push. Good ergonomics
/// for small `N` (the typical extract `count ≤ 32`); for larger `N`
/// or O(1) push, use a length-tracking wrapper (`arrayvec::ArrayVec`,
/// `heapless::Vec`, …).
impl<T, const N: usize> Buffer<T> for [Option<T>; N] {
  const MAX_CAP: Option<usize> = Some(N);

  fn try_push(&mut self, val: T) -> Option<T> {
    for slot in self.iter_mut() {
      if slot.is_none() {
        *slot = Some(val);
        return None;
      }
    }
    Some(val)
  }
}

/// Same linear-scan push as the array impl, for borrowed slices.
/// `MAX_CAP` is `None` because slice length is a runtime value, not a
/// compile-time constant.
impl<T> Buffer<T> for &mut [Option<T>] {
  const MAX_CAP: Option<usize> = None;

  fn try_push(&mut self, val: T) -> Option<T> {
    for slot in self.iter_mut() {
      if slot.is_none() {
        *slot = Some(val);
        return None;
      }
    }
    Some(val)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn array_buffer_push_until_full() {
    let mut buf: [Option<u32>; 3] = [const { None }; 3];
    assert_eq!(buf.try_push(1), None);
    assert_eq!(buf.try_push(2), None);
    assert_eq!(buf.try_push(3), None);
    // Buffer is full; subsequent pushes return the value back.
    assert_eq!(buf.try_push(4), Some(4));
    assert_eq!(buf.try_push(5), Some(5));
    // Confirm contents are 1, 2, 3 in order.
    assert_eq!(buf, [Some(1), Some(2), Some(3)]);
  }

  #[test]
  fn slice_buffer_push_until_full() {
    let mut underlying: [Option<&'static str>; 2] = [const { None }; 2];
    let mut buf: &mut [Option<&str>] = &mut underlying;
    assert_eq!(buf.try_push("a"), None);
    assert_eq!(buf.try_push("b"), None);
    assert_eq!(buf.try_push("c"), Some("c"));
    assert_eq!(underlying, [Some("a"), Some("b")]);
  }

  #[cfg(any(feature = "alloc", feature = "std"))]
  #[test]
  fn vec_buffer_is_unbounded() {
    use std::vec::Vec;

    let mut buf: Vec<u32> = Vec::new();
    for i in 0..1000 {
      assert_eq!(buf.try_push(i), None);
    }
    assert_eq!(buf.len(), 1000);
    assert_eq!(<Vec<u32> as Buffer<u32>>::MAX_CAP, None);
  }

  #[test]
  fn max_cap_constants_are_correct() {
    assert_eq!(<[Option<u32>; 8] as Buffer<u32>>::MAX_CAP, Some(8));
    assert_eq!(<[Option<u32>; 0] as Buffer<u32>>::MAX_CAP, Some(0));
    assert_eq!(<&mut [Option<u32>] as Buffer<u32>>::MAX_CAP, None);
  }
}
