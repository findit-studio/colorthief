//! Codegen for `colorthief-dataset/src/generated.rs`.
//!
//! Reads `colorthief-dataset/assets/color_hierarchy.csv` (sourced from
//! Stitch Fix's `colornamer`), pre-computes CIE LAB for each entry's
//! xkcd RGB, and emits a Rust source file containing one `const` per
//! entry plus a `pub static COLORS: &[&Color]` slice.
//!
//! Entry ids come from `colorthief-dataset/assets/color_ids.csv`, the
//! permanent-id ledger — see [`Ledger`] for the discipline it enforces.
//! The ledger is read *and rewritten* by this tool; both it and
//! `generated.rs` are committed, and CI fails on drift in either.
//!
//! Run with: `cargo xtask codegen`.

use std::{
  collections::{BTreeMap, BTreeSet, HashSet},
  path::{Path, PathBuf},
};

use heck::{ToShoutySnakeCase, ToUpperCamelCase};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use serde::{Deserialize, Serialize};

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

/// One row of `assets/color_ids.csv`, the permanent-id ledger.
///
/// `retired` is the row's liveness *as of the last committed run* — the
/// one fact the generator cannot recompute from the upstream CSV, and
/// the one every gate below needs. Without it the generator knows which
/// names are live now but not which were live before, so it can tell
/// neither "this entry retired in this run" from "this entry retired
/// three revisions ago", nor spot a surviving name landing on a
/// tombstone and walking off with its id.
#[derive(Debug, Deserialize, Serialize)]
struct LedgerRow {
  id: u16,
  xkcd_color: String,
  retired: bool,
}

/// Prologue rewritten verbatim on top of `assets/color_ids.csv` every
/// codegen run. The ledger carries its own law so a curator editing it
/// by hand reads the discipline before touching a number.
const LEDGER_HEADER: &str = "\
# colorthief-dataset — the permanent color-id ledger.
#
# THIS FILE IS THE AUTHORITY FOR `ColorId`. `cargo run --release -p xtask
# -- codegen` reads it, mints ids for any new `color_hierarchy.csv` row,
# rewrites it, and emits the ids into `src/generated.rs`, where they are
# part of the crate's PUBLIC API. Downstream stores the id and looks the
# row back up; a renumbering silently repoints stored data at the wrong
# color.
#
# The id discipline — these ids are PERMANENT:
#
#   * An id is assigned once and NEVER changes. Correcting an entry's
#     name, design/common columns, rgb, hex, family or kind KEEPS its id.
#   * A deleted entry's id is NEVER reused. Its row stays here, retired,
#     so the mint can never hand the number out again. If that same
#     color later returns, recovering its own id is correct — but
#     codegen cannot tell it from a DIFFERENT color arriving under a
#     name some earlier entry used to hold, so it refuses either way
#     until you pass --allow-revival to say which it is.
#   * A new entry mints a fresh id: the high-water mark plus one. Ids
#     start at 1 — 0 is never assigned, so a zeroed id is always
#     detectably invalid.
#
# EVERY ROW HERE IS LOAD-BEARING, retired ones included. A retired row
# is the only record that its number was ever handed out; delete it and
# the next mint can hand the same number to a different color. Rows are
# only ever added or edited in place — never removed, never renumbered,
# and the file is never regenerated from scratch. `tests/ids.rs` pins
# every row, retired ones included, and CI re-runs codegen and fails on
# any drift in this file.
#
# The `retired` column is the row's liveness as of THIS commit, and it
# is load-bearing too: it is the only record of which names were live
# last time. The generator reads it to tell a retirement that happened
# in the run it is executing from one that happened revisions ago, and
# to notice a surviving name landing on a tombstone. Flipping it by
# hand re-arms or disarms those gates — leave it to the generator.
#
# `xkcd_color` is the join key the generator matches CSV rows on, so
# renaming an entry upstream is a CORRECTION, not a delete-and-insert:
# edit the name in place HERE, keeping its id, in the same commit that
# changes `color_hierarchy.csv`. Dropping the row instead would give the
# color an id it did not hold before — a freshly minted one, or a
# retired one if the new name is already here as a tombstone — and break
# every id already stored downstream. Codegen cannot tell a rename from
# a delete-and-insert, so it refuses any run that retires a name live
# last run while some other name picks up an id it did not hold, until
# you either fix the name in place or pass --allow-retire-and-mint to
# confirm the events are unrelated.
";

/// The permanent-id ledger: the id half of the `ColorId` bijection,
/// kept outside `color_hierarchy.csv` so the upstream file stays
/// verbatim-replaceable on a `colornamer` refresh.
///
/// The ledger holds *every* name ever seen, live and retired. Retirement
/// is what burns an id: a row with no matching CSV row is never matched
/// and still counts toward [`Ledger::high_water`], so its number can
/// never be minted a second time.
struct Ledger {
  /// Name → id for every row ever minted, live and retired.
  by_name: BTreeMap<String, u16>,
  /// What the file held on entry. Re-checked before writing so a future
  /// edit to this type cannot silently move an already-assigned id.
  loaded: BTreeMap<String, u16>,
  /// Which of [`Self::loaded`] were live as of the last committed run,
  /// straight off the ledger's `retired` column.
  ///
  /// The generator recomputes liveness for *this* run from the upstream
  /// CSV, so without the previous run's answer it can only ask "is this
  /// name live now". Every interesting question is a difference between
  /// the two: retired in this run, or come back from the dead in it.
  loaded_live: BTreeSet<String>,
  /// Highest id ever minted, retired rows included.
  high_water: u16,
  /// Names matched against a CSV row this run; the complement is retired.
  live: BTreeSet<String>,
  /// Ids minted this run, for the codegen log.
  minted: Vec<(u16, String)>,
  /// Where these rows came from, for error messages.
  source: String,
  /// The exact bytes [`Self::load`] read, or `None` when the file did
  /// not exist. [`Self::write`] refuses to commit unless the ledger on
  /// disk is still byte-for-byte this.
  ///
  /// The gap between load and commit spans the whole LUT computation —
  /// minutes — so "the file I am about to replace is the file I derived
  /// these ids from" is not something this tool may assume.
  loaded_source: Option<String>,
}

impl Ledger {
  /// Validate a set of ledger rows and take ownership of them.
  ///
  /// Panics on a malformed ledger — a duplicate id, a duplicate name, or
  /// the reserved id 0 — rather than generating a table whose ids are
  /// not a bijection. Split from [`Self::load`] so the discipline can be
  /// exercised in unit tests without touching the filesystem.
  fn from_rows(rows: Vec<LedgerRow>, source: &str) -> Self {
    let mut by_name = BTreeMap::<String, u16>::new();
    let mut by_id = BTreeMap::<u16, String>::new();
    let mut loaded_live = BTreeSet::<String>::new();

    for LedgerRow {
      id,
      xkcd_color,
      retired,
    } in rows
    {
      assert!(
        id != 0,
        "{source}: id 0 is reserved as \"never assigned\" (row {xkcd_color:?})",
      );
      if !retired {
        loaded_live.insert(xkcd_color.clone());
      }
      if let Some(other) = by_id.insert(id, xkcd_color.clone()) {
        panic!(
          "{source}: id {id} is assigned twice, to {other:?} and {xkcd_color:?}; \
           ids are permanent and unique — resolve by hand, never by renumbering",
        );
      }
      if let Some(other) = by_name.insert(xkcd_color.clone(), id) {
        panic!(
          "{source}: {xkcd_color:?} appears twice, as id {other} and id {id}; \
           a name is the ledger's join key and must be unique",
        );
      }
    }

    let high_water = by_id.keys().next_back().copied().unwrap_or(0);
    Self {
      loaded: by_name.clone(),
      by_name,
      loaded_live,
      high_water,
      live: BTreeSet::new(),
      minted: Vec::new(),
      source: source.to_string(),
      loaded_source: None,
    }
  }

  /// Read the committed ledger.
  ///
  /// A missing file is a hard error unless `bootstrap` is set. That
  /// asymmetry is deliberate: the ledger is the only record that a
  /// retired id was ever handed out, so silently regenerating it from
  /// the upstream CSV would renumber the dataset and remint every
  /// retired number. Creating one is an explicit, once-ever act
  /// (`--bootstrap-ledger`), not a fallback.
  fn load(path: &Path, bootstrap: bool) -> Self {
    let source = path.display().to_string();

    if !path.exists() {
      assert!(
        bootstrap,
        "{source}: the permanent-id ledger is missing. It is the only \
         record of which ids have been handed out, including retired \
         ones — regenerating it from the upstream CSV would renumber the \
         dataset and remint retired ids. Restore the file from version \
         control. Pass --bootstrap-ledger only to create one for the \
         first time, on a dataset that has never shipped an id.",
      );
      return Self::from_rows(Vec::new(), &source);
    }

    let raw = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {source}: {e}"));

    // A ledger cut short still parses. Rows go out in id order, so any
    // prefix is a well-formed, non-empty, ascending ledger that merely
    // reports a lower high-water mark — and the next run hands the ids
    // it lost to whichever colors come first. `write` commits through a
    // rename so this generator can never produce one, but a bad merge,
    // a full disk or a half-finished editor save still can, and the
    // trailing newline is the cheapest evidence that the last row
    // arrived whole.
    assert!(
      raw.is_empty() || raw.ends_with('\n'),
      "{source}: the permanent-id ledger does not end in a newline, so \
       its last row was cut short. A truncated ledger still parses and \
       silently lowers the high-water mark, freeing burned ids to be \
       reminted for different colors. Restore the file from version \
       control rather than re-running codegen over it.",
    );

    let mut rdr = csv::ReaderBuilder::new()
      .has_headers(true)
      .comment(Some(b'#'))
      .from_reader(raw.as_bytes());
    let rows: Vec<LedgerRow> = rdr
      .deserialize::<LedgerRow>()
      .map(|row| row.unwrap_or_else(|e| panic!("parse {source}: {e}")))
      .collect();

    // An emptied ledger is a lost ledger wearing a header. It would
    // reset the high-water mark to 0 and remint the whole dataset from
    // 1, and — unlike a rename — it retires nothing, so the
    // retire-and-mint gate would not fire either. Treat it exactly like
    // a missing file.
    assert!(
      bootstrap || !rows.is_empty(),
      "{source}: the permanent-id ledger has no rows. An empty ledger \
       remints every id from 1 and loses every retired one. Restore the \
       file from version control. Pass --bootstrap-ledger only to create \
       one for the first time, on a dataset that has never shipped an id.",
    );

    let mut ledger = Self::from_rows(rows, &source);
    ledger.loaded_source = Some(raw);
    ledger
  }

  /// The id for `name`: its existing one, or a freshly minted one above
  /// the high-water mark. Marks the name live either way.
  fn id_for(&mut self, name: &str) -> u16 {
    self.live.insert(name.to_string());
    if let Some(&id) = self.by_name.get(name) {
      return id;
    }
    let id = self
      .high_water
      .checked_add(1)
      .expect("permanent color ids exhausted u16; widen ColorId's repr");
    self.high_water = id;
    self.by_name.insert(name.to_string(), id);
    self.minted.push((id, name.to_string()));
    id
  }

  /// Every ledger row with no matching CSV row this run — the complete
  /// tombstone set, this run's retirements and every earlier one alike.
  ///
  /// This is the *reporting* view: their ids stay burned, and the list
  /// is printed so a retirement is never silent. It is deliberately not
  /// what the gates key on — see [`Self::newly_retired`].
  fn tombstones(&self) -> Vec<(u16, &str)> {
    let mut out: Vec<(u16, &str)> = self
      .by_name
      .iter()
      .filter(|(name, _)| !self.live.contains(*name))
      .map(|(name, &id)| (id, name.as_str()))
      .collect();
    out.sort_unstable();
    out
  }

  /// Names that were live in the last committed run and are not live
  /// now: the retirements *this* run performs.
  ///
  /// [`Self::tombstones`] cannot serve here. It reports every name not
  /// live now, so it stays non-empty forever once anything has ever
  /// retired — a gate keyed on it would fire on every later run that
  /// mints anything, and the operator would learn to wave the override
  /// through as a matter of routine, disarming the one check that
  /// catches a real rename.
  fn newly_retired(&self) -> Vec<(u16, &str)> {
    let mut out: Vec<(u16, &str)> = self
      .loaded_live
      .iter()
      .filter(|name| !self.live.contains(*name))
      .map(|name| (self.by_name[name], name.as_str()))
      .collect();
    out.sort_unstable();
    out
  }

  /// Tombstones that are live again this run: a name the last run had
  /// retired now matches a CSV row, so it recovers its original id
  /// instead of minting.
  ///
  /// Legitimate on its own — a color that left and came back is the
  /// same color, and getting its id back is the point. But it is also
  /// what a rename ONTO a dead name looks like, and that hands a
  /// retired id to a different color while minting nothing, so the gate
  /// has to see it. Invisible to [`Self::minted`] by construction.
  fn reactivated(&self) -> Vec<(u16, &str)> {
    let mut out: Vec<(u16, &str)> = self
      .live
      .iter()
      .filter(|name| self.loaded.contains_key(*name) && !self.loaded_live.contains(*name))
      .map(|name| (self.by_name[name], name.as_str()))
      .collect();
    out.sort_unstable();
    out
  }

  /// Refuse a run that both retires a name and mints a new one, unless
  /// the operator has said the two are unrelated.
  ///
  /// The generator matches CSV rows on a *mutable* column, so an
  /// upstream rename is indistinguishable from a deletion plus an
  /// insertion: it would retire the old name, mint a fresh id for the
  /// new one, and silently break every id already stored for that
  /// color — precisely the correction the "an id survives a rename"
  /// clause exists to protect. Only the operator knows which happened,
  /// so codegen stops and asks rather than guessing.
  ///
  /// A run that only mints (new entries) or only retires (deletions) is
  /// unambiguous and passes untouched, and so is one whose only
  /// retirements happened in some earlier revision.
  ///
  /// This is the *minting* half of a rename: a name that vanished this
  /// run, paired with a name arriving on a freshly minted id. The other
  /// half — a name arriving on a **retired** id — mints nothing, so it
  /// is invisible here and has its own gate,
  /// [`Self::assert_revival_approved`].
  fn assert_rename_resolved(&self, allow_retire_and_mint: bool) {
    let retired = self.newly_retired();
    if allow_retire_and_mint || retired.is_empty() || self.minted.is_empty() {
      return;
    }

    let minted = self
      .minted
      .iter()
      .map(|(id, name)| format!("\n    would mint id {id} for {name:?}"))
      .collect::<String>();
    let retired = retired
      .iter()
      .map(|(id, name)| format!("\n    would retire id {id}, held by {name:?}"))
      .collect::<String>();
    panic!(
      "{}: this run retires a name that was live last run, and in the \
       same breath mints a fresh id for another — which is what an \
       upstream RENAME looks like from here, and would break every id \
       already stored for the renamed color.\
       \n{retired}{minted}\
       \n\n  If a retirement above is really the same color under a new \
       name, that is a CORRECTION: edit the name in place in the ledger, \
       keeping its id, and re-run. The pairing disappears and codegen \
       proceeds.\
       \n  If they are genuinely unrelated — a color left and a different \
       one arrived — re-run with --allow-retire-and-mint to say so.",
      self.source,
    );
  }

  /// Refuse to hand a retired id back out without the operator saying so.
  ///
  /// Fails closed on **every** revival, not only on one paired with a
  /// retirement in the same run. Pairing was the wrong test: an operator
  /// can retire "gray" in one commit and let the old tombstone "grey"
  /// come back as its replacement in the next, and then neither run
  /// looks suspicious on its own — the first only retires, the second
  /// only revives — while across the two a burned id has moved to a
  /// different color. Nothing in the ledger records which logical color
  /// a name once meant, so no amount of state can tell that apart from
  /// the legitimate case.
  ///
  /// So the legitimate case pays the same toll. A color that genuinely
  /// left and came back does recover its original id — that is the
  /// point of keeping the row — but the operator has to assert it with
  /// `--allow-revival`, because from here it is indistinguishable from
  /// a different color moving in on a dead name, and getting that wrong
  /// silently repoints every id already stored for the old one.
  ///
  /// This is the same posture the rest of the ledger takes: a missing
  /// ledger, a retire-and-mint pairing and now a revival all stop and
  /// ask rather than guess. Minting a fresh id above the high-water mark
  /// stays the only way a name can acquire an id unattended.
  fn assert_revival_approved(&self, allow_revival: bool) {
    let reactivated = self.reactivated();
    if allow_revival || reactivated.is_empty() {
      return;
    }

    let revived = reactivated
      .iter()
      .map(|(id, name)| format!("\n    would hand retired id {id} back to {name:?}"))
      .collect::<String>();
    panic!(
      "{}: this run gives a name an id that was RETIRED — a number the \
       ledger has already spent and promised never to hand out again.\
       \n{revived}\
       \n\n  If that really is the same color returning after an \
       absence, recovering its own id is correct: re-run with \
       --allow-revival to say so.\
       \n  If it is a different color that has arrived under a name some \
       earlier entry used to hold, do NOT revive it. Give the ledger row \
       a name that is free — or leave the newcomer to mint a fresh id — \
       so the retired number stays retired.",
      self.source,
    );
  }

  /// Stage the ledger beside its destination, ready to be committed.
  ///
  /// Staging only writes a private temporary; nothing the crate reads
  /// moves until [`Staged::commit`]. That split is what lets the ledger
  /// and `generated.rs` land together instead of one at a time, so a
  /// refused run leaves BOTH untouched rather than a rewritten table
  /// paired with the ledger it was supposed to have come from.
  ///
  /// The seal is checked here: every id present on entry still maps to
  /// the same name, so no in-memory edit can have moved one.
  fn stage(&self, path: &Path) -> Staged {
    for (name, &id) in &self.loaded {
      let now = self.by_name.get(name).copied();
      assert_eq!(
        now,
        Some(id),
        "{}: {name:?} held id {id} on entry and {now:?} on exit; color ids \
         are PERMANENT — a correction keeps its id and a retired id is \
         never reused",
        path.display(),
      );
    }

    Staged::write(path, self.to_csv().as_bytes())
  }

  /// Take the commit lock, an OS advisory lock on a stable file beside
  /// the ledger.
  ///
  /// Held only across the re-read and the two renames — microseconds —
  /// not across the minutes of LUT computation, so it costs a concurrent
  /// run nothing but the commit itself. The lock lives on its own file
  /// rather than on the ledger, because committing *renames over* the
  /// ledger: a lock held on the old inode would be invisible to whoever
  /// opened the new one.
  ///
  /// This is an OS lock, not a lock-by-file-existence, and the
  /// difference is the whole reason it is safe to use here. The kernel
  /// drops it when the process exits — including when a developer
  /// interrupts a run that looks like it has hung — so it cannot go
  /// stale and leave every later run refusing until somebody deletes a
  /// file by hand.
  fn lock_for_commit(path: &Path) -> std::fs::File {
    let lock_path = path.with_extension("csv.lock");
    let file = std::fs::OpenOptions::new()
      .create(true)
      .truncate(false)
      .write(true)
      .open(&lock_path)
      .unwrap_or_else(|e| panic!("open commit lock {}: {e}", lock_path.display()));
    file
      .lock()
      .unwrap_or_else(|e| panic!("acquire commit lock {}: {e}", lock_path.display()));
    file
  }

  /// Confirm the ledger on disk is still byte-for-byte what
  /// [`Self::load`] read. Call under [`Self::lock_for_commit`].
  ///
  /// This covers the ledger only. The upstream `color_hierarchy.csv` is
  /// checked separately by [`assert_input_unchanged`], because a change
  /// there can move every LAB value and the whole LUT while leaving the
  /// ledger byte-identical.
  ///
  /// The gap between load and commit spans the whole LUT computation, so
  /// a run that derived its ids from a file that has since changed would
  /// commit an assignment built from a ledger nobody has any more,
  /// silently dropping whatever rows or tombstones the newer one gained.
  /// A stale run stops having written nothing, and re-running from the
  /// current ledger is a clean retry.
  fn assert_unchanged_on_disk(&self, path: &Path) {
    let on_disk = std::fs::read_to_string(path).ok();
    assert!(
      on_disk == self.loaded_source,
      "{}: the permanent-id ledger changed on disk while this codegen \
       was running. The ids just built came from the file as it was \
       when the run started, so committing them now would overwrite \
       whatever rows or tombstones it has gained since — and an id that \
       vanishes is an id the next run can hand to a different color. \
       Nothing has been written. Re-run codegen against the current \
       ledger.",
      path.display(),
    );
  }

  /// Serialize the whole ledger — header prologue, then every row, live
  /// and retired, sorted by id.
  ///
  /// Deterministic down to the byte: the row order is the id order and
  /// the terminator is always LF, so CI's byte comparison of a
  /// regenerated ledger holds across platforms.
  fn to_csv(&self) -> String {
    let mut rows: Vec<LedgerRow> = self
      .by_name
      .iter()
      .map(|(xkcd_color, &id)| LedgerRow {
        id,
        retired: !self.live.contains(xkcd_color),
        xkcd_color: xkcd_color.clone(),
      })
      .collect();
    rows.sort_unstable_by_key(|r| r.id);

    // LF explicitly: CI's `codegen-up-to-date` job compares bytes, so a
    // regeneration on Windows must produce the same file as one on Linux.
    let mut wtr = csv::WriterBuilder::new()
      .terminator(csv::Terminator::Any(b'\n'))
      .from_writer(Vec::<u8>::new());
    for row in &rows {
      wtr.serialize(row).expect("serialize ledger row");
    }
    let body = wtr.into_inner().expect("flush ledger writer");
    let body = String::from_utf8(body).expect("ledger rows are utf-8");

    format!("{LEDGER_HEADER}{body}")
  }
}

/// A generated artifact written beside its destination and not yet in
/// place.
///
/// Codegen produces two files that have to agree — the ledger and
/// `generated.rs` — and writing either one directly means a run that
/// fails partway leaves the pair disagreeing: a table whose ids the
/// ledger does not record, or a ledger describing a table that was
/// never written. Both are staged first and renamed at the end, so a
/// refused or failed run leaves both exactly as they were.
///
/// The temporary is named uniquely per process and per call and created
/// with `create_new`, so it can never be another run's file. A fixed
/// name would be worse than no temporary at all: a second process
/// truncating and writing the *same* temporary while this one is
/// mid-write leaves both streams interleaved in one file, and once
/// either renames it, that interleaving IS the committed artifact.
///
/// Dropping without committing removes the temporary, so an error path
/// or a panic between staging and commit leaves nothing behind.
struct Staged {
  tmp: PathBuf,
  dst: PathBuf,
  committed: bool,
}

impl Staged {
  /// Write `contents` to a private temporary beside `dst`.
  ///
  /// `suffix` carries the destination's own extension so a staged Rust
  /// file is still named `*.rs` — rustfmt is run over the staged file
  /// before it is committed, and it needs to recognise it as Rust.
  fn write(dst: &Path, contents: &[u8]) -> Self {
    let extension = dst
      .extension()
      .and_then(|e| e.to_str())
      .unwrap_or("tmp")
      .to_string();
    let stamp = std::time::SystemTime::now()
      .duration_since(std::time::UNIX_EPOCH)
      .map(|d| d.as_nanos())
      .unwrap_or(0);
    // Same directory, so the rename stays within one filesystem and
    // cannot degrade into a copy.
    let tmp = dst.with_extension(format!("staged-{}-{stamp}.{extension}", std::process::id()));

    let mut file =
      std::fs::File::create_new(&tmp).unwrap_or_else(|e| panic!("create {}: {e}", tmp.display()));
    // `sync_all` before the rename: the rename is atomic with respect to
    // readers, but only durable if the bytes reached the disk first.
    let written = std::io::Write::write_all(&mut file, contents).and_then(|()| file.sync_all());
    drop(file);
    if let Err(e) = written {
      std::fs::remove_file(&tmp).ok();
      panic!("write {}: {e}", tmp.display());
    }

    Self {
      tmp,
      dst: dst.to_path_buf(),
      committed: false,
    }
  }

  /// The staged file's path, for a step that has to run over it before
  /// it is committed.
  fn path(&self) -> &Path {
    &self.tmp
  }

  /// Flush the staged file again, after something outside this type
  /// rewrote it.
  ///
  /// [`Self::write`] syncs what it wrote, but rustfmt then replaces the
  /// staged source with reformatted bytes that nothing has flushed. The
  /// rename would otherwise publish a directory entry for contents that
  /// may not have reached the disk, which after a crash is a corrupt
  /// `generated.rs` beside a ledger that has already advanced.
  /// Opened for WRITING, and without truncating. `sync_all` is
  /// `FlushFileBuffers` on Windows, which requires write access to the
  /// handle, so a read-only reopen would fail there — the durability
  /// this exists for would be unavailable on exactly one platform, and
  /// codegen would stop rather than quietly skip it.
  fn resync(&self) {
    let file = std::fs::OpenOptions::new()
      .write(true)
      .truncate(false)
      .open(&self.tmp)
      .unwrap_or_else(|e| panic!("reopen {} for sync: {e}", self.tmp.display()));
    file
      .sync_all()
      .unwrap_or_else(|e| panic!("sync {}: {e}", self.tmp.display()));
  }

  /// Rename the staged file into place.
  ///
  /// A rename is atomic on POSIX and on Windows, so a reader sees the
  /// old artifact or the new one and never a prefix of either. A
  /// truncating write would instead leave a *valid shorter ledger*: rows
  /// go out in id order, so any prefix parses, passes the non-empty
  /// check, and simply reports a lower high-water mark — and the next
  /// run hands the ids it lost to whichever colors come first.
  fn commit(self) {
    let dst = self.rename_into_place();
    sync_parent_dir(&dst);
  }

  /// Rename, and nothing else. Both commit paths go through here so the
  /// directory sync is chosen once, by the caller that knows whether
  /// anything is ordered against it.
  fn rename_into_place(mut self) -> PathBuf {
    std::fs::rename(&self.tmp, &self.dst).unwrap_or_else(|e| {
      panic!(
        "commit {} over {}: {e}",
        self.tmp.display(),
        self.dst.display()
      )
    });
    self.committed = true;
    self.dst.clone()
  }

  /// Commit, and confirm the directory entry reached the disk.
  ///
  /// The ledger goes first and through this door: the source rename that
  /// follows must not proceed on an unconfirmed ledger rename, or a
  /// crash could leave a table shipping ids the authority does not
  /// record — and an unrecorded id is one a later run can hand to a
  /// different color. A swallowed error here would be that ordering
  /// silently not holding.
  fn commit_durably(self) {
    let dst = self.rename_into_place();
    sync_dir_or_panic(&dst);
  }
}

impl Drop for Staged {
  fn drop(&mut self) {
    if !self.committed {
      std::fs::remove_file(&self.tmp).ok();
    }
  }
}

/// Flush the directory entry a rename just created.
///
/// `sync_all` on the temporary makes its *contents* durable, but the
/// rename is a change to the parent directory, and until that is flushed
/// a power loss can roll it back — leaving the old ledger beside the new
/// generated source, which is precisely the disagreement the staged
/// commit exists to prevent.
///
/// POSIX only. A directory cannot be opened as a file on Windows, and
/// `std::fs::rename` there does not expose a write-through option, so
/// this reports `Unsupported` rather than pretending: the guarantee is
/// genuinely not available from the standard library on that platform.
/// What is left there is the ordinary failure rather than a silent one —
/// both artifacts are committed to the repository and CI re-runs codegen
/// and fails on any drift, so an interrupted commit surfaces as a red
/// build rather than as an id quietly reused.
#[cfg(unix)]
fn sync_dir(path: &Path) -> std::io::Result<()> {
  let dir = path
    .parent()
    .filter(|p| !p.as_os_str().is_empty())
    .unwrap_or_else(|| Path::new("."));
  std::fs::File::open(dir)?.sync_all()
}

#[cfg(not(unix))]
fn sync_dir(_path: &Path) -> std::io::Result<()> {
  Err(std::io::Error::new(
    std::io::ErrorKind::Unsupported,
    "directory sync is not available through the standard library here",
  ))
}

/// Confirm a generated artifact's input is still what this run read.
///
/// Call under the commit lock. The ledger's own snapshot check cannot
/// cover this: an upstream edit to an rgb, a hex, a family, a type or
/// the row order rewrites `generated.rs` and its LUT while leaving the
/// ledger byte-identical, so a run started before that edit would pass
/// the ledger check and commit its stale table over a newer one, while
/// reporting success.
fn assert_input_unchanged(path: &Path, snapshot: &str) {
  let on_disk = std::fs::read_to_string(path).ok();
  assert!(
    on_disk.as_deref() == Some(snapshot),
    "{}: the upstream color CSV changed on disk while this codegen was \
     running. Everything just built — the LAB values, the enum variants \
     and the whole candidate-set LUT — came from the file as it was when \
     the run started, so committing now would publish a table that does \
     not match its own input, and would overwrite a newer run that does. \
     Nothing has been written. Re-run codegen.",
    path.display(),
  );
}

/// Best-effort flush, for a rename nothing else is ordered against.
fn sync_parent_dir(path: &Path) {
  sync_dir(path).ok();
}

/// Flush and insist, for a rename that a later one depends on.
///
/// On a platform that cannot offer the guarantee the error is
/// `Unsupported`; that is not a failure of this run and must not abort
/// it. Anything else is a real I/O error, and the run stops rather than
/// committing the next artifact against an unconfirmed one.
fn sync_dir_or_panic(path: &Path) {
  match sync_dir(path) {
    Ok(()) => {}
    Err(e) if e.kind() == std::io::ErrorKind::Unsupported => {
      // Expected off unix, where the standard library offers no
      // directory sync at all. On unix it means this filesystem does not
      // support it, which is worth saying out loud: the commit went
      // through, but its durability is not confirmed.
      if cfg!(unix) {
        eprintln!(
          "  warning: the directory holding {} does not support syncing, so \
           the commit is not confirmed durable",
          path.display(),
        );
      }
    }
    Err(e) => panic!(
      "the directory holding {} was not synced after the commit, so the \
       rename is not known to have reached the disk and the artifacts \
       that follow must not be committed against it: {e}",
      path.display(),
    ),
  }
}

/// The escape hatches on the ledger's discipline. All default to off,
/// and each exists for the same reason: the generator sees names, not
/// colors, so it cannot tell a rename from a delete-plus-insert, a
/// returning color from a newcomer wearing a dead name, nor a
/// never-created ledger from a lost one. Where it cannot tell, it stops
/// and asks rather than guessing at an id that downstream has stored.
#[derive(Debug, Default, Clone, Copy)]
struct CodegenOptions {
  /// Create the ledger when it does not exist, instead of refusing.
  bootstrap_ledger: bool,
  /// Accept a run that both retires and mints as genuinely unrelated
  /// events rather than an unrecorded rename.
  allow_retire_and_mint: bool,
  /// Accept that a name matching a retired row really is that same
  /// color returning, so recovering its original id is correct.
  allow_revival: bool,
}

const USAGE: &str = "usage: cargo xtask [codegen] [--bootstrap-ledger] \
                     [--allow-retire-and-mint] [--allow-revival]";

fn main() {
  let mut options = CodegenOptions::default();
  for arg in std::env::args().skip(1) {
    match arg.as_str() {
      "codegen" => {}
      "--bootstrap-ledger" => options.bootstrap_ledger = true,
      "--allow-retire-and-mint" => options.allow_retire_and_mint = true,
      "--allow-revival" => options.allow_revival = true,
      other => {
        eprintln!("unknown xtask argument: {other}");
        eprintln!("{USAGE}");
        std::process::exit(1);
      }
    }
  }
  codegen(options);
}

fn workspace_root() -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    .parent()
    .expect("xtask crate must live one level under the workspace root")
    .to_path_buf()
}

fn codegen(options: CodegenOptions) {
  let root = workspace_root();
  let csv_path = root.join("colorthief-dataset/assets/color_hierarchy.csv");
  let ledger_path = root.join("colorthief-dataset/assets/color_ids.csv");
  let out_path = root.join("colorthief-dataset/src/generated.rs");

  // 1. Parse the CSV, from a retained snapshot of its bytes.
  //
  // Everything below is derived from these bytes: the ids, the LAB
  // values, the enum variants, the LUT. The snapshot is re-checked under
  // the commit lock, because the ledger cannot stand in for it — an
  // upstream edit to an rgb, a hex, a family or the row order changes
  // `generated.rs` and leaves the ledger byte-identical, so a slower run
  // started before that edit would pass the ledger check and commit its
  // stale table over a newer one.
  let upstream = std::fs::read_to_string(&csv_path)
    .unwrap_or_else(|e| panic!("read {}: {e}", csv_path.display()));
  let mut rdr = csv::ReaderBuilder::new()
    .has_headers(true)
    .from_reader(upstream.as_bytes());
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

  // 2a-bis. Load the permanent-id ledger. Every CSV row resolves to its
  // existing id or mints a fresh one; nothing already assigned moves.
  let mut ledger = Ledger::load(&ledger_path, options.bootstrap_ledger);

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
  // Permanent id per entry, positionally aligned with `idents`/`COLORS`.
  let mut ids: Vec<u16> = Vec::with_capacity(rows.len());
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
    let id = ledger.id_for(xkcd_name);
    ids.push(id);

    consts.push(quote! {
      const #ident: &Color = &Color {
        id: ColorId::new(#id),
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

  // 2c. Clear the ledger's gates and build the id → entry reverse
  // table. The table is dense over `1..=max live id`, so
  // `Color::from_id` is a bounds check plus one load; index 0 and any
  // retired id hold `None`.
  //
  // The gates run here, before anything is written, so a refused run
  // leaves both artifacts untouched. Between them they cover every way
  // a name can end up on an id it did not hold in the previous
  // revision: onto a freshly minted one, or onto a retired one.
  ledger.assert_rename_resolved(options.allow_retire_and_mint);
  ledger.assert_revival_approved(options.allow_revival);
  for (id, name) in &ledger.minted {
    println!("  minted permanent id {id} for {name:?}");
  }
  for (id, name) in ledger.tombstones() {
    println!("  id {id} stays retired (no CSV row for {name:?}); never reused");
  }
  let max_id = ids.iter().copied().max().expect("CSV is non-empty");
  let mut by_id: Vec<TokenStream> = vec![quote! { None }; usize::from(max_id) + 1];
  for (&id, ident) in ids.iter().zip(&idents) {
    by_id[usize::from(id)] = quote! { Some(#ident) };
  }

  // 2d. Compute CIEDE2000 candidate-set LUT (parallelized via rayon).
  // Cells are independent — each runs 512 full-scans over the 949-entry
  // palette and unions the winners.
  let (lut_offsets, lut_indices) = compute_lut(&palette_labs);

  // 3. Assemble the file body and pretty-print.
  let count = idents.len();
  let count_doc = format!(" All {count} entries in the dataset, in CSV order.");
  let by_id_doc = format!(
    " Permanent id → entry, indexed by [`ColorId::get`]. Dense over \
      `0..={max_id}`: slot 0 is `None` (0 is never assigned) and so is \
      any id whose entry has been retired. Backs [`Color::from_id`], the \
      reverse half of the id bijection. Generated by `cargo xtask \
      codegen` from `assets/color_ids.csv`; do not edit by hand."
  );
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
    use super::{Color, ColorId};

    #family_enum

    #kind_enum

    #(#consts)*

    #[doc = #count_doc]
    pub static COLORS: &[&Color] = &[
      #(#idents),*
    ];

    #[doc = #by_id_doc]
    pub(crate) static BY_ID: &[Option<&Color>] = &[
      #(#by_id),*
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

  // 4. Stage both artifacts. Nothing the crate reads has moved yet.
  let staged_source = Staged::write(&out_path, output.as_bytes());

  // prettyplease emits 4-space indent unconditionally; the workspace
  // rustfmt.toml uses `tab_spaces = 2`. Shell out to rustfmt so the
  // generated file passes `cargo fmt --check` like the hand-written
  // ones. Run over the STAGED file, so a rustfmt that is missing or
  // fails leaves the committed source untouched.
  let status = std::process::Command::new("rustfmt")
    .arg("--edition=2024")
    .arg(staged_source.path())
    .status()
    .expect("rustfmt is required on PATH for `cargo xtask codegen`");
  assert!(
    status.success(),
    "rustfmt {out} failed with status {status}",
    out = staged_source.path().display(),
  );
  // rustfmt replaced the bytes `Staged::write` flushed, so flush again.
  // Without this the rename publishes a directory entry for contents
  // that may not have reached the disk.
  staged_source.resync();

  let staged_ledger = ledger.stage(&ledger_path);

  // 5. Commit both, together and last.
  //
  // The ledger and `generated.rs` have to agree: the table ships the
  // ids, the ledger is the record that they were spent. Moving either
  // one alone leaves the pair disagreeing — a table whose ids the ledger
  // does not record, or a ledger describing a table nobody wrote — and
  // the id authority must never advance past a run that produced
  // nothing. So every fallible step above happens first, and the two
  // renames happen here, under the commit lock and after one last check
  // that the ledger this run derived its ids from is still the ledger on
  // disk.
  //
  // Re-running after a failure is then a clean retry: the ids are still
  // unminted, and the CSV row order that decides them has not moved.
  let lock = Ledger::lock_for_commit(&ledger_path);
  assert_input_unchanged(&csv_path, &upstream);
  ledger.assert_unchanged_on_disk(&ledger_path);
  // The ledger goes first, and durably: it is the authority, and a
  // ledger that is ahead is harmless — the ids are recorded but not yet
  // shipped, and a re-run reproduces exactly the same table. A source
  // that is ahead is not: it would ship ids the authority does not
  // record, and an unrecorded id is one a later run can hand to a
  // different color.
  staged_ledger.commit_durably();
  staged_source.commit();
  drop(lock);

  println!(
    "wrote {} ({count} entries, ids 1..={max_id}) and {}",
    out_path.display(),
    ledger_path.display(),
  );
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

#[cfg(test)]
mod tests {
  use super::*;

  /// Build a ledger of all-live rows from `(id, name)` pairs, as if read
  /// from a file whose every entry matched a CSV row last run.
  fn ledger(rows: &[(u16, &str)]) -> Ledger {
    let rows: Vec<(u16, &str, bool)> = rows.iter().map(|&(id, name)| (id, name, false)).collect();
    ledger_with_liveness(&rows)
  }

  /// Build a ledger from `(id, name, retired)` triples — the general
  /// form, for the cases where a tombstone from an earlier revision is
  /// the whole point.
  fn ledger_with_liveness(rows: &[(u16, &str, bool)]) -> Ledger {
    Ledger::from_rows(
      rows
        .iter()
        .map(|&(id, xkcd_color, retired)| LedgerRow {
          id,
          xkcd_color: xkcd_color.to_string(),
          retired,
        })
        .collect(),
      "test ledger",
    )
  }

  /// Resolve a run's worth of CSV names against a ledger, in order.
  fn resolve(ledger: &mut Ledger, names: &[&str]) -> Vec<u16> {
    names.iter().map(|name| ledger.id_for(name)).collect()
  }

  #[test]
  fn existing_names_keep_their_ids() {
    let mut l = ledger(&[(1, "red"), (2, "green"), (3, "blue")]);
    assert_eq!(resolve(&mut l, &["red", "green", "blue"]), [1, 2, 3]);
    assert!(l.minted.is_empty(), "nothing new should have been minted");
    assert!(l.tombstones().is_empty());
  }

  /// CSV row order must not touch the assignment — the ledger is keyed
  /// by name, not by position.
  #[test]
  fn reordered_input_does_not_renumber() {
    let mut l = ledger(&[(1, "red"), (2, "green"), (3, "blue")]);
    assert_eq!(resolve(&mut l, &["blue", "red", "green"]), [3, 1, 2]);
    assert!(l.minted.is_empty());
  }

  #[test]
  fn a_new_name_mints_above_the_high_water_mark() {
    let mut l = ledger(&[(1, "red"), (7, "green")]);
    assert_eq!(resolve(&mut l, &["red", "green", "violet"]), [1, 7, 8]);
    assert_eq!(l.minted, vec![(8, "violet".to_string())]);
  }

  /// The core promise. An entry retires while holding the highest id;
  /// its row stays; the next newcomer must NOT receive that number.
  #[test]
  fn a_retired_high_water_id_is_never_reminted() {
    let mut l = ledger(&[(1, "red"), (2, "green"), (3, "blue")]);
    // "blue" (id 3, the high-water mark) is gone from the CSV.
    assert_eq!(resolve(&mut l, &["red", "green", "violet"]), [1, 2, 4]);
    assert_eq!(l.tombstones(), vec![(3, "blue")]);
    assert_eq!(l.newly_retired(), vec![(3, "blue")]);
    assert_eq!(l.minted, vec![(4, "violet".to_string())]);
    assert!(
      l.to_csv().contains("3,blue"),
      "the retired row must survive the rewrite; it is the only record \
       that 3 was ever handed out",
    );
  }

  /// Falsification of the clause above: this is what goes wrong when a
  /// retired row is deleted from the ledger by hand. Nothing in the
  /// generator can detect it — the tombstone IS the memory — which is
  /// why `tests/ids.rs` pins every ledger row, retired ones included.
  #[test]
  fn deleting_a_tombstone_lets_its_id_be_reminted() {
    let mut without_tombstone = ledger(&[(1, "red"), (2, "green")]);
    assert_eq!(
      resolve(&mut without_tombstone, &["red", "green", "violet"]),
      [1, 2, 3],
      "with blue's row deleted, 3 is handed to a different color",
    );

    let mut with_tombstone = ledger(&[(1, "red"), (2, "green"), (3, "blue")]);
    assert_eq!(
      resolve(&mut with_tombstone, &["red", "green", "violet"]),
      [1, 2, 4],
      "with the row kept, the same input mints 4 instead",
    );
  }

  /// An entry that leaves and later comes back is the same color, so it
  /// gets its original id back rather than a fresh one.
  #[test]
  fn a_returning_name_recovers_its_original_id() {
    let mut gone = ledger(&[(1, "red"), (2, "green")]);
    assert_eq!(resolve(&mut gone, &["red"]), [1]);
    assert_eq!(gone.tombstones(), vec![(2, "green")]);

    // The ledger as the run above would have rewritten it: green is
    // still there, now marked retired.
    let mut back = ledger_with_liveness(&[(1, "red", false), (2, "green", true)]);
    assert_eq!(resolve(&mut back, &["red", "green"]), [1, 2]);
    assert!(back.minted.is_empty(), "green must not be re-minted");
    assert_eq!(back.reactivated(), vec![(2, "green")]);
    assert!(
      back.newly_retired().is_empty(),
      "nothing left in this run, so the revival stands alone",
    );
    // Recovering the id is the right answer, but the generator cannot
    // know this is the same green rather than a newcomer wearing a dead
    // name, so the operator has to say so.
    back.assert_rename_resolved(false);
    back.assert_revival_approved(true);
  }

  /// The same revival without the operator asserting it: refused.
  #[test]
  #[should_panic(expected = "an id that was RETIRED")]
  fn an_unapproved_revival_is_refused() {
    let mut back = ledger_with_liveness(&[(1, "red", false), (2, "green", true)]);
    resolve(&mut back, &["red", "green"]);
    back.assert_revival_approved(false);
  }

  /// The rename the pairing test could not see, because its two halves
  /// land in DIFFERENT runs. "gray" retires in run N; in run N+1 the
  /// older tombstone "grey" comes back as its replacement. Neither run
  /// looks suspicious alone — the first only retires, the second only
  /// revives — yet across the two, id 2 has moved to a color it was
  /// never assigned to. Only a gate that fails closed on every revival
  /// catches it.
  #[test]
  #[should_panic(expected = "would hand retired id 2 back to")]
  fn a_rename_split_across_two_runs_is_refused() {
    // Run N: "gray" (id 3) retires. "grey" (id 2) has been a tombstone
    // since some earlier revision.
    let mut run_n =
      ledger_with_liveness(&[(1, "red", false), (2, "grey", true), (3, "gray", false)]);
    resolve(&mut run_n, &["red"]);
    assert_eq!(run_n.newly_retired(), vec![(3, "gray")]);
    assert!(run_n.reactivated().is_empty());
    // Nothing revives, so run N itself is clean.
    run_n.assert_rename_resolved(false);
    run_n.assert_revival_approved(false);

    // Run N+1 reads the ledger run N would have written: both grey and
    // gray retired. Upstream now carries "grey".
    let mut run_n1 =
      ledger_with_liveness(&[(1, "red", false), (2, "grey", true), (3, "gray", true)]);
    resolve(&mut run_n1, &["red", "grey"]);
    assert!(
      run_n1.newly_retired().is_empty(),
      "nothing retires in run N+1 — which is exactly why pairing missed this",
    );
    assert!(run_n1.minted.is_empty(), "and nothing mints either");
    assert_eq!(run_n1.reactivated(), vec![(2, "grey")]);
    run_n1.assert_rename_resolved(false);
    run_n1.assert_revival_approved(false);
  }

  /// A rename reaches the generator as a retirement plus a mint, and
  /// minting would break every stored id for that color. Codegen must
  /// refuse rather than guess.
  #[test]
  #[should_panic(expected = "mints a fresh id for another")]
  fn a_rename_is_refused_until_the_operator_resolves_it() {
    let mut l = ledger(&[(1, "red"), (2, "grey")]);
    resolve(&mut l, &["red", "gray"]);
    l.assert_rename_resolved(false);
  }

  /// The rename the old gate could not see. "grey" is already here as a
  /// tombstone, so renaming the live "gray" onto it MINTS NOTHING — it
  /// simply walks off with a retired id. A gate watching only the mint
  /// list waves this through, and id 2 then names a color it was never
  /// assigned to.
  #[test]
  #[should_panic(expected = "would hand retired id 2 back to")]
  fn a_rename_onto_a_tombstone_is_refused() {
    let mut l = ledger_with_liveness(&[(1, "red", false), (2, "grey", true), (3, "gray", false)]);
    resolve(&mut l, &["red", "grey"]);
    assert!(
      l.minted.is_empty(),
      "the whole point: this rename mints nothing",
    );
    assert_eq!(l.newly_retired(), vec![(3, "gray")]);
    assert_eq!(l.reactivated(), vec![(2, "grey")]);
    // The minting gate cannot see it — nothing was minted.
    l.assert_rename_resolved(false);
    l.assert_revival_approved(false);
  }

  /// A tombstone from an EARLIER revision must not arm the gate. Once
  /// anything has ever retired, a gate keyed on the whole tombstone set
  /// fires on every later run that mints, and the operator learns to
  /// pass the override as a matter of routine — which disarms the check
  /// for the real rename it exists to catch.
  #[test]
  fn a_mint_after_an_older_retirement_needs_no_override() {
    let mut l = ledger_with_liveness(&[(1, "red", false), (2, "green", true)]);
    assert_eq!(resolve(&mut l, &["red", "violet"]), [1, 3]);
    assert_eq!(l.minted, vec![(3, "violet".to_string())]);
    assert_eq!(
      l.tombstones(),
      vec![(2, "green")],
      "the old tombstone is still reported, and its id still burned",
    );
    assert!(
      l.newly_retired().is_empty(),
      "green retired in an earlier run, not in this one",
    );
    l.assert_rename_resolved(false);
  }

  /// The same shape one revision further on: the override must not
  /// become permanently necessary as tombstones accumulate.
  #[test]
  fn mints_stay_unblocked_however_many_tombstones_have_accumulated() {
    let mut l = ledger_with_liveness(&[
      (1, "red", false),
      (2, "green", true),
      (3, "blue", true),
      (4, "grey", true),
    ]);
    assert_eq!(resolve(&mut l, &["red", "violet"]), [1, 5]);
    assert_eq!(l.tombstones().len(), 3);
    assert!(l.newly_retired().is_empty());
    l.assert_rename_resolved(false);
  }

  /// Liveness survives the write/read round trip — it is the only fact
  /// in the file the generator cannot recompute, so losing it in
  /// serialization would silently restore both bugs above.
  #[test]
  fn the_retired_column_round_trips() {
    let mut l = ledger(&[(1, "red"), (2, "green"), (3, "blue")]);
    resolve(&mut l, &["red", "blue"]);

    let serialized = l.to_csv();
    assert!(serialized.contains("1,red,false"));
    assert!(
      serialized.contains("2,green,true"),
      "green retired this run"
    );
    assert!(serialized.contains("3,blue,false"));

    let mut rdr = csv::ReaderBuilder::new()
      .has_headers(true)
      .comment(Some(b'#'))
      .from_reader(serialized.as_bytes());
    let rows: Vec<LedgerRow> = rdr
      .deserialize::<LedgerRow>()
      .map(|r| r.expect("re-read own output"))
      .collect();
    let reread = Ledger::from_rows(rows, "round trip");
    assert_eq!(
      reread
        .loaded_live
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>(),
      ["blue", "red"],
      "green must come back as a tombstone, not as a live row",
    );
    assert_eq!(reread.high_water, 3, "the tombstone still burns its id");
  }

  /// Resolving the rename the documented way — editing the name in
  /// place, keeping the id — makes the pairing disappear.
  #[test]
  fn editing_the_name_in_place_preserves_the_id_and_clears_the_gate() {
    let mut l = ledger(&[(1, "red"), (2, "gray")]);
    assert_eq!(resolve(&mut l, &["red", "gray"]), [1, 2]);
    assert!(l.minted.is_empty());
    assert!(l.tombstones().is_empty());
    l.assert_rename_resolved(false);
  }

  /// A genuinely unrelated delete and insert is allowed, but only with
  /// the operator saying so.
  #[test]
  fn unrelated_retire_and_mint_passes_with_the_flag() {
    let mut l = ledger(&[(1, "red"), (2, "grey")]);
    resolve(&mut l, &["red", "violet"]);
    l.assert_rename_resolved(true);
  }

  /// Retiring alone is unambiguous — no gate.
  #[test]
  fn retiring_without_minting_is_unambiguous() {
    let mut l = ledger(&[(1, "red"), (2, "grey")]);
    resolve(&mut l, &["red"]);
    l.assert_rename_resolved(false);
  }

  /// Minting alone is unambiguous — no gate.
  #[test]
  fn minting_without_retiring_is_unambiguous() {
    let mut l = ledger(&[(1, "red")]);
    resolve(&mut l, &["red", "violet"]);
    l.assert_rename_resolved(false);
  }

  #[test]
  #[should_panic(expected = "assigned twice")]
  fn a_duplicate_id_is_rejected() {
    ledger(&[(1, "red"), (1, "green")]);
  }

  #[test]
  #[should_panic(expected = "appears twice")]
  fn a_duplicate_name_is_rejected() {
    ledger(&[(1, "red"), (2, "red")]);
  }

  #[test]
  #[should_panic(expected = "id 0 is reserved")]
  fn the_reserved_id_zero_is_rejected() {
    ledger(&[(0, "red")]);
  }

  #[test]
  #[should_panic(expected = "exhausted u16")]
  fn exhausting_the_id_space_is_a_hard_error() {
    let mut l = ledger(&[(u16::MAX, "the last color")]);
    resolve(&mut l, &["the last color", "one too many"]);
  }

  /// A name needing CSV quoting must survive a write/read round trip —
  /// the ledger's own parser and its writer have to agree over the
  /// whole domain of upstream names.
  #[test]
  fn a_name_needing_quoting_round_trips() {
    let awkward = "red, orange \"vivid\"";
    let mut l = ledger(&[(1, "red")]);
    assert_eq!(resolve(&mut l, &["red", awkward]), [1, 2]);

    let serialized = l.to_csv();
    let mut rdr = csv::ReaderBuilder::new()
      .has_headers(true)
      .comment(Some(b'#'))
      .from_reader(serialized.as_bytes());
    let rows: Vec<LedgerRow> = rdr
      .deserialize::<LedgerRow>()
      .map(|r| r.expect("re-read own output"))
      .collect();
    let reread = Ledger::from_rows(rows, "round trip");
    assert_eq!(reread.by_name.get(awkward).copied(), Some(2));
    assert_eq!(reread.high_water, 2);
  }

  /// The serialized form is byte-stable: same ledger, same bytes, with
  /// rows in id order regardless of insertion order.
  #[test]
  fn serialization_is_deterministic_and_id_ordered() {
    let mut a = ledger(&[(2, "green"), (1, "red")]);
    let mut b = ledger(&[(1, "red"), (2, "green")]);
    resolve(&mut a, &["red", "green"]);
    resolve(&mut b, &["green", "red"]);
    assert_eq!(a.to_csv(), b.to_csv());

    let body = a.to_csv();
    let red = body.find("1,red").expect("red row");
    let green = body.find("2,green").expect("green row");
    assert!(red < green, "rows must be sorted by id");
    assert!(body.ends_with('\n'));
    assert!(!body.contains('\r'), "the ledger is always LF");
  }

  /// A missing ledger is a hard error: it is the only record of retired
  /// ids, so regenerating it from the CSV would remint them.
  #[test]
  #[should_panic(expected = "permanent-id ledger is missing")]
  fn a_missing_ledger_is_refused_without_the_bootstrap_flag() {
    Ledger::load(Path::new("/nonexistent/color_ids.csv"), false);
  }

  /// An emptied ledger — header present, no rows — resets the
  /// high-water mark and remints everything from 1, and because it
  /// retires nothing the rename gate stays silent. It has to be refused
  /// like a missing file.
  #[test]
  #[should_panic(expected = "has no rows")]
  fn an_empty_ledger_is_refused_without_the_bootstrap_flag() {
    let dir = std::env::temp_dir().join(format!("dsid-empty-ledger-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let path = dir.join("color_ids.csv");
    std::fs::write(&path, "# prologue\nid,xkcd_color\n").expect("write empty ledger");
    let result = std::panic::catch_unwind(|| Ledger::load(&path, false));
    std::fs::remove_dir_all(&dir).ok();
    match result {
      Ok(_) => panic!("an empty ledger was accepted"),
      Err(payload) => std::panic::resume_unwind(payload),
    }
  }

  #[test]
  fn bootstrap_creates_an_empty_ledger() {
    let l = Ledger::load(Path::new("/nonexistent/color_ids.csv"), true);
    assert_eq!(l.high_water, 0);
    assert!(l.by_name.is_empty());
  }

  /// Bootstrapping mints from 1, never 0.
  #[test]
  fn bootstrap_mints_from_one() {
    let mut l = Ledger::load(Path::new("/nonexistent/color_ids.csv"), true);
    assert_eq!(resolve(&mut l, &["red", "green"]), [1, 2]);
  }

  /// The commit sequence `codegen` performs for the ledger, as one call:
  /// stage, take the lock, re-check the snapshot, rename into place.
  fn commit_ledger(ledger: &Ledger, path: &Path) {
    let staged = ledger.stage(path);
    let lock = Ledger::lock_for_commit(path);
    ledger.assert_unchanged_on_disk(path);
    staged.commit();
    drop(lock);
  }

  /// Every file the commit sequence leaves in `dir` other than the
  /// ledger itself and its lock. Must always be empty: a staged
  /// temporary that outlived its commit is a leak.
  fn strays(dir: &Path, ledger: &Path) -> Vec<PathBuf> {
    std::fs::read_dir(dir)
      .expect("scratch dir")
      .map(|e| e.expect("dir entry").path())
      .filter(|p| p != ledger && p.extension().and_then(|e| e.to_str()) != Some("lock"))
      .collect()
  }

  /// A private scratch directory, removed however the test ends.
  struct Scratch(PathBuf);

  impl Scratch {
    fn new(tag: &str) -> Self {
      // Thread id as well as process id: `cargo test` runs these
      // concurrently in one process.
      let dir = std::env::temp_dir().join(format!(
        "dsid-{tag}-{}-{:?}",
        std::process::id(),
        std::thread::current().id(),
      ));
      std::fs::create_dir_all(&dir).expect("scratch dir");
      Self(dir)
    }

    fn path(&self, name: &str) -> PathBuf {
      self.0.join(name)
    }
  }

  impl Drop for Scratch {
    fn drop(&mut self) {
      std::fs::remove_dir_all(&self.0).ok();
    }
  }

  /// A ledger cut short still parses — rows go out in id order, so any
  /// prefix is a well-formed ascending ledger that merely reports a
  /// lower high-water mark, and the next run hands the ids it lost to
  /// different colors. The trailing newline is the evidence the last row
  /// arrived whole.
  #[test]
  #[should_panic(expected = "does not end in a newline")]
  fn a_truncated_ledger_is_refused() {
    let scratch = Scratch::new("truncated");
    let path = scratch.path("color_ids.csv");
    std::fs::write(
      &path,
      "# prologue\nid,xkcd_color,retired\n1,red,false\n2,gre",
    )
    .expect("write truncated ledger");
    let result = std::panic::catch_unwind(|| Ledger::load(&path, false));
    drop(scratch);
    match result {
      Ok(_) => panic!("a truncated ledger was accepted"),
      Err(payload) => std::panic::resume_unwind(payload),
    }
  }

  /// Falsification of the clause above: without the guard, the prefix
  /// above loads clean and the lost row's id is free to be reminted.
  #[test]
  fn a_truncated_ledger_would_otherwise_remint_the_lost_id() {
    // The prefix a half-finished write leaves behind, parsed directly.
    let mut prefix = ledger(&[(1, "red")]);
    assert_eq!(
      resolve(&mut prefix, &["red", "violet"]),
      [1, 2],
      "with the second row lost, id 2 goes to a different color",
    );

    let mut whole = ledger(&[(1, "red"), (2, "green")]);
    assert_eq!(
      resolve(&mut whole, &["red", "violet"]),
      [1, 3],
      "with the row intact, the same input mints 3",
    );
  }

  /// A run whose ledger changed underneath it must NOT commit. Its ids
  /// were derived from the file as it was minutes earlier; writing them
  /// now would erase whatever rows or tombstones the file has gained,
  /// and a vanished row is an id the next run can hand to another color.
  #[test]
  #[should_panic(expected = "changed on disk while this codegen was running")]
  fn a_stale_run_refuses_to_commit_over_a_changed_ledger() {
    let scratch = Scratch::new("stale");
    let path = scratch.path("color_ids.csv");
    std::fs::write(&path, "id,xkcd_color,retired\n1,red,false\n").expect("seed ledger");

    let mut stale = Ledger::load(&path, false);
    resolve(&mut stale, &["red"]);

    // Another run commits first, adding a row this one never saw.
    std::fs::write(&path, "id,xkcd_color,retired\n1,red,false\n2,green,false\n")
      .expect("competing commit");

    let result = std::panic::catch_unwind(|| commit_ledger(&stale, &path));
    let after = std::fs::read_to_string(&path).expect("ledger still readable");
    drop(scratch);
    assert!(
      after.contains("2,green,false"),
      "the winner's row must survive the refused commit",
    );
    match result {
      Ok(()) => panic!("a stale run committed over a changed ledger"),
      Err(payload) => std::panic::resume_unwind(payload),
    }
  }

  /// The ledger check cannot stand in for the input check. An upstream
  /// edit to an rgb leaves the ledger byte-identical — no name changed,
  /// so nothing mints or retires — while rewriting every LAB value and
  /// the whole LUT. A run started before that edit must not commit its
  /// stale table over a newer one.
  #[test]
  #[should_panic(expected = "upstream color CSV changed on disk")]
  fn a_stale_run_refuses_when_only_the_upstream_csv_moved() {
    let scratch = Scratch::new("stale-input");
    let csv = scratch.path("color_hierarchy.csv");
    std::fs::write(&csv, "xkcd_color,xkcd_r\nred,255\n").expect("seed csv");
    let snapshot = std::fs::read_to_string(&csv).expect("snapshot");

    // Another run edits an rgb and commits. No name moved, so a ledger
    // check would see nothing at all.
    std::fs::write(&csv, "xkcd_color,xkcd_r\nred,254\n").expect("competing edit");

    assert_input_unchanged(&csv, &snapshot);
  }

  /// The control for the check above: an untouched input passes.
  #[test]
  fn an_untouched_upstream_csv_passes_the_input_check() {
    let scratch = Scratch::new("fresh-input");
    let csv = scratch.path("color_hierarchy.csv");
    std::fs::write(&csv, "xkcd_color,xkcd_r\nred,255\n").expect("seed csv");
    let snapshot = std::fs::read_to_string(&csv).expect("snapshot");
    assert_input_unchanged(&csv, &snapshot);
  }

  /// A staged file rewritten in place — as rustfmt rewrites the staged
  /// source — must be re-flushed and must commit the REWRITTEN bytes,
  /// not the ones staging happened to sync.
  #[test]
  fn a_rewritten_staged_file_is_resynced_and_commits_its_new_bytes() {
    let scratch = Scratch::new("resync");
    let dst = scratch.path("generated.rs");

    let staged = Staged::write(&dst, b"// staged\n");
    assert!(
      staged.path().extension().and_then(|e| e.to_str()) == Some("rs"),
      "a staged Rust file must still look like Rust to rustfmt: {:?}",
      staged.path(),
    );
    // Stand in for rustfmt: replace the staged bytes in place.
    std::fs::write(staged.path(), b"// reformatted\n").expect("rewrite staged");
    staged.resync();
    staged.commit();

    assert_eq!(
      std::fs::read_to_string(&dst).expect("committed"),
      "// reformatted\n",
    );
    assert!(strays(&scratch.0, &dst).is_empty());
  }

  /// Dropping a staged file without committing removes it, so a panic or
  /// an early return between staging and commit leaves nothing behind
  /// and, crucially, leaves the destination untouched.
  #[test]
  fn an_uncommitted_staged_file_is_cleaned_up_and_leaves_the_destination_alone() {
    let scratch = Scratch::new("staged-drop");
    let dst = scratch.path("generated.rs");
    std::fs::write(&dst, b"// original\n").expect("seed destination");

    let tmp = {
      let staged = Staged::write(&dst, b"// never committed\n");
      staged.path().to_path_buf()
    };

    assert!(
      !tmp.exists(),
      "the staged temporary must be removed on drop"
    );
    assert_eq!(
      std::fs::read_to_string(&dst).expect("destination"),
      "// original\n",
      "an abandoned staging must not touch the destination",
    );
    assert!(strays(&scratch.0, &dst).is_empty());
  }

  /// The unchanged case is the control: a ledger nobody touched commits
  /// normally, so the snapshot check is not simply refusing everything.
  #[test]
  fn an_untouched_ledger_commits_normally() {
    let scratch = Scratch::new("untouched");
    let path = scratch.path("color_ids.csv");
    std::fs::write(&path, "id,xkcd_color,retired\n1,red,false\n").expect("seed ledger");

    let mut l = Ledger::load(&path, false);
    resolve(&mut l, &["red", "violet"]);
    commit_ledger(&l, &path);

    let after = std::fs::read_to_string(&path).expect("committed");
    assert!(
      after.contains("2,violet,false"),
      "the mint reached the file"
    );
  }

  /// Two overlapping runs must never share a temporary. A fixed name is
  /// worse than none: the second process truncates and writes the same
  /// file while the first is mid-write, and whichever renames it commits
  /// the interleaving as the ledger.
  #[test]
  fn each_run_writes_a_private_temporary() {
    let scratch = Scratch::new("private-tmp");
    let path = scratch.path("color_ids.csv");
    std::fs::write(&path, "id,xkcd_color,retired\n1,red,false\n").expect("seed ledger");

    let mut first = Ledger::load(&path, false);
    resolve(&mut first, &["red"]);
    commit_ledger(&first, &path);

    // A second commit in the same process and directory must not be
    // blocked or corrupted by anything the first left behind.
    let mut second = Ledger::load(&path, false);
    resolve(&mut second, &["red", "violet"]);
    commit_ledger(&second, &path);

    let after = std::fs::read_to_string(&path).expect("committed");
    assert!(after.contains("2,violet,false"));

    let leftovers = strays(&scratch.0, &path);
    assert!(
      leftovers.is_empty(),
      "no temporary may outlive its commit: {leftovers:?}",
    );
  }

  /// `write` commits through a rename, so the ledger is never observable
  /// in a half-written state and no temporary file survives the run.
  #[test]
  fn write_commits_atomically_and_leaves_no_temporary() {
    let scratch = Scratch::new("atomic");
    let path = scratch.path("color_ids.csv");

    let mut l = ledger(&[(1, "red"), (2, "green")]);
    resolve(&mut l, &["red"]);
    commit_ledger(&l, &path);

    let written = std::fs::read_to_string(&path).expect("ledger was committed");
    assert_eq!(written, l.to_csv(), "committed bytes are the whole ledger");
    assert!(written.ends_with('\n'));

    let leftovers = strays(&scratch.0, &path);
    assert!(
      leftovers.is_empty(),
      "the temporary must be renamed away, not left behind: {leftovers:?}",
    );

    // And the committed file is a ledger the loader accepts, with the
    // retirement recorded rather than dropped.
    let reloaded = Ledger::load(&path, false);
    assert_eq!(reloaded.high_water, 2);
    assert_eq!(
      reloaded
        .loaded_live
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>(),
      ["red"],
    );
  }
}
