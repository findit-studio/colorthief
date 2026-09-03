//! Permanent color ids: the [`ColorId`] bijection and the canary that
//! pins the whole assignment.
//!
//! Downstream stores a `ColorId` and resolves it back later, so these
//! ids are load-bearing in a way the rest of the table is not: a
//! renumbering does not fail a build or a lookup, it silently repoints
//! every stored id at a different color. Four things guard that here.
//!
//! 1. **The bijection** — `from_id(c.id())` is `c` for every entry, ids
//!    are unique, and `from_id` is total over `u16`: an id the dataset
//!    does not carry answers `None` rather than a neighbour.
//! 2. **The table pin** — [`ID_ASSIGNMENT_FINGERPRINT`] hashes the
//!    complete `(id, name)` assignment the crate ships. Any renumbering
//!    changes it, and the cross-check proves the table still matches
//!    the committed `assets/color_ids.csv`.
//! 3. **The ledger pin** — [`LEDGER_FINGERPRINT`] hashes every ledger
//!    row, *retired ones included*. Those are invisible to the table
//!    pin, and losing one un-burns its id: the generator's high-water
//!    mark drops and the next new color is minted that number. Nothing
//!    else notices, because the table, the generated file and the
//!    codegen diff all agree with the shortened ledger.
//! 4. **The probes** — [`renumber_probe_trips_the_pin`] and
//!    [`tombstone_loss_trips_the_ledger_pin`] break the assignment on
//!    purpose and assert the pins move, so neither can pass by being
//!    insensitive to what it claims to catch.
//! 5. **The liveness column** —
//!    [`ledger_liveness_matches_the_shipped_table`] checks the ledger's
//!    `retired` flag against what the crate actually ships. No runtime
//!    code reads that column, so nothing else would notice it drifting;
//!    the generator reads it to date a retirement and to catch a live
//!    name landing on a tombstone, and a wrong flag disarms both.
//!
//! These tests touch no SIMD backend, so unlike `tests/api.rs` and
//! `tests/properties.rs` they are left to run under Miri too.

use std::collections::{BTreeMap, BTreeSet};

use colorthief_dataset::{Color, ColorId};

/// The committed permanent-id ledger, the authority `cargo xtask
/// codegen` assigns from.
const LEDGER: &str = include_str!("../assets/color_ids.csv");

/// FNV-1a 64 over the complete `(id, name)` assignment in id order —
/// see [`fingerprint`].
///
/// **This number changing means the dataset's identity changed.** Ids
/// are permanent: the only legitimate reasons to update it are a new
/// entry minting a fresh id, an entry retiring, or an upstream
/// correction to an entry's *name*. Before touching it, confirm from
/// the `assets/color_ids.csv` diff that no existing id moved to a
/// different color — that would break every id already stored
/// downstream, and no rebuild would say so.
const ID_ASSIGNMENT_FINGERPRINT: u64 = 0x2025_1b10_41d8_7686;

/// FNV-1a 64 over **every row of the ledger**, live and retired.
///
/// [`ID_ASSIGNMENT_FINGERPRINT`] cannot see a retired row: it hashes
/// the shipped table, and a retired entry is by definition not in it.
/// But a retired row is the only record that its number was ever handed
/// out — delete it and the generator's high-water mark drops, so the
/// next new entry is minted that number and every id stored for the old
/// color silently resolves to the new one. Nothing else catches that:
/// not the table, not the generated file, not the codegen diff, because
/// all three agree with the shortened ledger.
///
/// So this pins the tombstones. Update it for a mint, a retirement, or
/// a name correction — never for a row that *disappeared*.
///
/// It equals [`ID_ASSIGNMENT_FINGERPRINT`] today, and that is not a
/// copy-paste: nothing has retired yet, so the ledger's rows are
/// exactly the shipped table's. The first retirement separates them
/// forever. [`tombstone_loss_trips_the_ledger_pin`] asserts that
/// relationship rather than leaving it to coincidence.
const LEDGER_FINGERPRINT: u64 = 0x2025_1b10_41d8_7686;

/// Rows in the ledger — every id ever assigned, live or retired. Only
/// ever grows.
const EXPECTED_LEDGER_ROWS: usize = 949;

/// Entries in the dataset today. Pinned alongside the fingerprint so a
/// failure reads as a summary before the reader reaches the hash.
const EXPECTED_ENTRIES: usize = 949;

/// Lowest and highest assigned id today. Ids start at 1 and are dense
/// so far — nothing has retired yet — but neither property is promised
/// by the format, only observed by this pin.
const EXPECTED_ID_RANGE: (u16, u16) = (1, 949);

/// FNV-1a 64 over `(id, name)` pairs sorted by id.
///
/// Binds each name to its number, so the hash moves for a renumbering
/// even when the same set of ids and the same set of names survive it —
/// two entries trading ids is the failure mode that matters most and
/// the one an id-set or name-set checksum would miss.
///
/// Sorting first makes the hash independent of `COLORS` order: this
/// pins the *assignment*, not the table's iteration order, which
/// `dataset_entry_count_matches_csv` and the CSV itself already cover.
fn fingerprint<S: AsRef<str>>(assignment: &[(u16, S)]) -> u64 {
  const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
  const PRIME: u64 = 0x0000_0100_0000_01b3;

  let mut sorted: Vec<(u16, &str)> = assignment
    .iter()
    .map(|(id, name)| (*id, name.as_ref()))
    .collect();
  sorted.sort_unstable();

  let mut hash = OFFSET;
  let mut eat = |byte: u8| {
    hash ^= u64::from(byte);
    hash = hash.wrapping_mul(PRIME);
  };
  for (id, name) in &sorted {
    for byte in id.to_le_bytes() {
      eat(byte);
    }
    for &byte in name.as_bytes() {
      eat(byte);
    }
    // Record separator, so `(1, "ab")` and `(1, "a") (_, "b")` cannot
    // collide by concatenation.
    eat(0xff);
  }
  hash
}

/// The `(id, name)` assignment as the shipped table carries it.
fn table_assignment() -> Vec<(u16, &'static str)> {
  Color::all()
    .iter()
    .map(|c| (c.id().get(), c.name()))
    .collect()
}

/// Parse a ledger: `#` comment prologue, one header row, then
/// `id,xkcd_color,retired`.
///
/// Read through a conforming CSV reader in the same dialect the
/// generator writes, so every name the writer can emit is accepted —
/// including one that needs quoting, like `red, orange`. A hand-rolled
/// splitter here would reject valid generator output, turning a
/// legitimate upstream name into a test failure.
fn parse_ledger(source: &str) -> Vec<(u16, String, bool)> {
  let mut reader = csv::ReaderBuilder::new()
    .has_headers(true)
    .comment(Some(b'#'))
    .from_reader(source.as_bytes());

  let headers: Vec<String> = reader
    .headers()
    .expect("ledger has a header row")
    .iter()
    .map(str::to_string)
    .collect();
  assert_eq!(
    headers,
    ["id", "xkcd_color", "retired"],
    "unexpected ledger columns; this test reads them positionally",
  );

  reader
    .records()
    .map(|record| {
      let record = record.expect("ledger row parses as CSV");
      let raw = record.get(0).expect("id column");
      let id: u16 = raw
        .parse()
        .unwrap_or_else(|e| panic!("ledger id {raw:?} is not a u16: {e}"));
      let raw_retired = record.get(2).expect("retired column");
      let retired: bool = raw_retired
        .parse()
        .unwrap_or_else(|e| panic!("ledger retired {raw_retired:?} is not a bool: {e}"));
      (id, record.get(1).expect("name column").to_string(), retired)
    })
    .collect()
}

/// Every row of the committed ledger with its liveness, live and
/// retired alike.
fn ledger_rows_with_liveness() -> Vec<(u16, String, bool)> {
  parse_ledger(LEDGER)
}

/// Every row of the committed ledger as an `(id, name)` assignment —
/// the shape both fingerprints hash. Liveness is pinned separately by
/// [`ledger_liveness_matches_the_shipped_table`]; folding it into the
/// hash would make an ordinary retirement look like the renumbering
/// these constants exist to catch.
fn ledger_rows() -> Vec<(u16, String)> {
  parse_ledger(LEDGER)
    .into_iter()
    .map(|(id, name, _)| (id, name))
    .collect()
}

// ---------------------------------------------------------------------
// The bijection
// ---------------------------------------------------------------------

/// Every entry resolves back to *itself* through its id. Identity is
/// checked by pointer: `Color` derives `PartialEq`, so a value
/// comparison would also accept a different entry that happened to be
/// field-identical.
#[test]
fn from_id_round_trips_every_entry() {
  for entry in Color::all() {
    let back = Color::from_id(entry.id()).unwrap_or_else(|| {
      panic!(
        "{:?} carries id {} but from_id returned None",
        entry.name(),
        entry.id(),
      )
    });
    assert!(
      std::ptr::eq(back, *entry),
      "id {} resolved to {:?}, not to {:?}",
      entry.id(),
      back.name(),
      entry.name(),
    );
  }
}

/// Ids are injective — no two entries share one. Without this,
/// `from_id` could round-trip every entry and still lose a color.
#[test]
fn ids_are_unique_across_the_table() {
  let mut by_id = BTreeMap::<u16, &str>::new();
  for entry in Color::all() {
    if let Some(other) = by_id.insert(entry.id().get(), entry.name()) {
      panic!(
        "id {} is carried by both {:?} and {:?}",
        entry.id(),
        other,
        entry.name(),
      );
    }
  }
  assert_eq!(by_id.len(), Color::all().len());
}

/// `from_id` is total over the whole `u16` space: every id either
/// resolves to the entry that claims it, or resolves to nothing. The
/// failure this rules out is an unassigned id quietly landing on a
/// neighbouring color.
#[test]
fn from_id_is_total_over_u16() {
  let assigned: BTreeMap<u16, &str> = table_assignment().into_iter().collect();

  for raw in 0..=u16::MAX {
    match Color::from_id(ColorId::new(raw)) {
      Some(entry) => {
        assert_eq!(
          entry.id().get(),
          raw,
          "from_id({raw}) returned {:?}, which claims id {}",
          entry.name(),
          entry.id(),
        );
        assert_eq!(
          assigned.get(&raw),
          Some(&entry.name()),
          "from_id({raw}) resolved to an entry the table does not list under that id",
        );
      }
      None => assert!(
        !assigned.contains_key(&raw),
        "id {raw} is carried by {:?} but from_id returned None",
        assigned[&raw],
      ),
    }
  }
}

/// 0 is reserved as "never assigned", so a zeroed storage column fails
/// to resolve instead of naming whichever entry sits first.
#[test]
fn id_zero_is_never_assigned() {
  assert!(Color::from_id(ColorId::new(0)).is_none());
  for entry in Color::all() {
    assert_ne!(entry.id().get(), 0, "{:?} carries id 0", entry.name());
  }
}

// ---------------------------------------------------------------------
// The pin
// ---------------------------------------------------------------------

/// Pin the complete assignment. A regeneration that renumbers anything
/// fails here.
#[test]
fn id_assignment_is_pinned() {
  let assignment = table_assignment();

  assert_eq!(
    assignment.len(),
    EXPECTED_ENTRIES,
    "entry count changed; see `dataset_entry_count_matches_csv` first",
  );

  let ids: BTreeSet<u16> = assignment.iter().map(|&(id, _)| id).collect();
  let range = (
    *ids.iter().next().expect("non-empty"),
    *ids.iter().next_back().expect("non-empty"),
  );
  assert_eq!(
    range, EXPECTED_ID_RANGE,
    "the live id range moved. A mint legitimately raises the top and a \
     retirement at either end legitimately moves that end — but an \
     existing entry's id changing does not. Check the ledger diff before \
     updating this.",
  );

  assert_eq!(
    fingerprint(&assignment),
    ID_ASSIGNMENT_FINGERPRINT,
    "the permanent-id assignment changed. Read the \
     `assets/color_ids.csv` diff: if an existing id now names a \
     different color, that is the defect — ids are PERMANENT, and every \
     id already stored downstream now points at the wrong entry. Only \
     update this constant once the diff shows nothing but fresh mints, \
     retirements, or upstream name corrections that kept their ids.",
  );
}

/// The shipped table and the committed ledger agree, in both
/// directions. `generated.rs` is machine-written and unreviewable at
/// 949 entries; the ledger is the reviewable form of the same
/// assignment, and this is what keeps the two from drifting apart.
#[test]
fn table_and_ledger_agree() {
  let rows = ledger_rows();
  let mut ledger = BTreeMap::<u16, String>::new();
  let mut by_name = BTreeMap::<String, u16>::new();
  let mut previous = 0u16;
  for (id, name) in rows {
    assert_ne!(id, 0, "ledger assigns the reserved id 0 to {name:?}");
    assert!(
      id > previous,
      "ledger is not sorted by ascending id: {id} follows {previous}",
    );
    previous = id;
    if let Some(other) = ledger.insert(id, name.clone()) {
      panic!("ledger assigns id {id} to both {other:?} and {name:?}");
    }
    if let Some(other) = by_name.insert(name.clone(), id) {
      panic!("ledger lists {name:?} twice, as id {other} and id {id}");
    }
  }

  // Every shipped entry is in the ledger under the same id.
  for entry in Color::all() {
    let id = entry.id().get();
    assert_eq!(
      ledger.get(&id).map(String::as_str),
      Some(entry.name()),
      "{:?} ships with id {id}, which the ledger does not assign to it",
      entry.name(),
    );
  }

  // And every *live* ledger row is in the table. Rows the table does
  // not carry are retirements: legitimate, and their ids must stay
  // unresolvable rather than being handed to someone else.
  let shipped: BTreeMap<u16, &str> = table_assignment().into_iter().collect();
  for (&id, name) in &ledger {
    match shipped.get(&id) {
      Some(&live) => assert_eq!(
        live, name,
        "ledger id {id} names {name:?} but the table ships {live:?} under it",
      ),
      None => assert!(
        Color::from_id(ColorId::new(id)).is_none(),
        "id {id} ({name:?}) is retired from the table but still resolves",
      ),
    }
  }
}

/// The ledger's `retired` column agrees with what the crate ships.
///
/// That column is the generator's memory of which names were live last
/// run, and it is the only input that tells a retirement made by the
/// current run from one made three revisions ago, or spots a surviving
/// name landing on a tombstone. Nothing in the crate reads it at
/// runtime, so nothing else would notice it drifting — and a drifted
/// column silently re-arms exactly the two ways an id can move to a
/// different color.
#[test]
fn ledger_liveness_matches_the_shipped_table() {
  let shipped: BTreeMap<u16, &str> = table_assignment().into_iter().collect();

  let mut live = 0usize;
  for (id, name, retired) in ledger_rows_with_liveness() {
    match shipped.get(&id) {
      Some(&table_name) => {
        assert!(
          !retired,
          "ledger marks id {id} ({name:?}) retired, but the crate ships            {table_name:?} under it — the generator would read this as a            color coming back from the dead",
        );
        live += 1;
      }
      None => assert!(
        retired,
        "ledger marks id {id} ({name:?}) live, but the crate ships no          entry under it — the generator would read this as a retirement          happening in whatever run comes next",
      ),
    }
  }

  assert_eq!(
    live, EXPECTED_ENTRIES,
    "every shipped entry must have exactly one live ledger row",
  );
}

/// Pin every ledger row, tombstones included.
///
/// This is the guard the shipped-table pin cannot be: a retired row is
/// invisible to [`ID_ASSIGNMENT_FINGERPRINT`], and losing one lets the
/// generator remint its id for a different color with every other check
/// still green. See [`LEDGER_FINGERPRINT`].
#[test]
fn ledger_including_retired_rows_is_pinned() {
  let rows = ledger_rows();

  assert_eq!(
    rows.len(),
    EXPECTED_LEDGER_ROWS,
    "the ledger row count changed. It may only ever GROW: a retirement \
     keeps its row so the id stays burned. If this went down, a row was \
     deleted and its id can now be reminted for a different color.",
  );

  assert_eq!(
    fingerprint(&rows),
    LEDGER_FINGERPRINT,
    "the ledger changed. Read the assets/color_ids.csv diff: a row that \
     DISAPPEARED is the defect — its id is no longer burned and the next \
     mint can hand it to a different color, breaking every id already \
     stored for the old one. Update this constant only for added rows or \
     for a name corrected in place with its id kept.",
  );
}

/// Falsification: [`ID_ASSIGNMENT_FINGERPRINT`] must actually be
/// sensitive to renumbering. Each case below is a way the assignment
/// could break; all three must move the fingerprint.
#[test]
fn renumber_probe_trips_the_pin() {
  let real = table_assignment();
  assert_eq!(
    fingerprint(&real),
    ID_ASSIGNMENT_FINGERPRINT,
    "control: the untouched assignment must match the pin",
  );

  // (a) Two entries trade ids. Same id set, same name set, same entry
  //     count — nothing but the binding moved, and every id stored for
  //     either color now names the other.
  let mut traded = real.clone();
  let (first, second) = (traded[0].0, traded[1].0);
  traded[0].0 = second;
  traded[1].0 = first;
  assert_ne!(
    fingerprint(&traded),
    ID_ASSIGNMENT_FINGERPRINT,
    "two entries traded ids and the pin did not notice",
  );

  // (b) The whole table renumbered — e.g. a generator regressing to
  //     0-based positional ids.
  let shifted: Vec<(u16, &str)> = real.iter().map(|&(id, name)| (id - 1, name)).collect();
  assert_ne!(
    fingerprint(&shifted),
    ID_ASSIGNMENT_FINGERPRINT,
    "the whole assignment shifted and the pin did not notice",
  );

  // (c) An entry retires and its id is handed to a newcomer — exactly
  //     what the "never reused" clause forbids.
  let mut reused = real.clone();
  let (retired_id, _) = reused.remove(0);
  reused.push((retired_id, "a color that did not exist before"));
  assert_ne!(
    fingerprint(&reused),
    ID_ASSIGNMENT_FINGERPRINT,
    "a retired id was reused and the pin did not notice",
  );

  // Negative control: reordering the pairs is not a renumbering, and
  // must not trip the pin — otherwise a `COLORS` reordering would be
  // reported as an identity change.
  let mut reordered = real.clone();
  reordered.reverse();
  assert_eq!(
    fingerprint(&reordered),
    ID_ASSIGNMENT_FINGERPRINT,
    "the pin is sensitive to table order; it must pin the assignment only",
  );
}

/// Falsification for the ledger pin, covering the case the table pin
/// structurally cannot: a tombstone going missing.
#[test]
fn tombstone_loss_trips_the_ledger_pin() {
  let real = ledger_rows();
  assert_eq!(
    fingerprint(&real),
    LEDGER_FINGERPRINT,
    "control: the untouched ledger must match the pin",
  );

  // Drop the highest-id row, as if an editor had tidied away a retired
  // entry. The generator's high-water mark would fall with it and the
  // next new color would be minted that number.
  let mut without_highest = real.clone();
  let highest = without_highest
    .iter()
    .enumerate()
    .max_by_key(|(_, (id, _))| *id)
    .map(|(index, _)| index)
    .expect("ledger is non-empty");
  without_highest.remove(highest);
  assert_ne!(
    fingerprint(&without_highest),
    LEDGER_FINGERPRINT,
    "a ledger row disappeared and the pin did not notice",
  );

  // The same loss is invisible to the table pin whenever the lost row
  // is a tombstone — which is exactly why both pins exist. Assert that
  // asymmetry rather than assuming it.
  let live: Vec<(u16, &str)> = real
    .iter()
    .filter(|(_, name)| Color::all().iter().any(|c| c.name() == name))
    .map(|(id, name)| (*id, name.as_str()))
    .collect();
  assert_eq!(
    fingerprint(&live),
    ID_ASSIGNMENT_FINGERPRINT,
    "the live subset of the ledger must reproduce the table pin",
  );
}

/// The ledger parser must accept everything the generator's writer can
/// emit. A name containing a comma or a quote is legitimate upstream
/// and gets CSV-quoted on the way out; reading it back must recover the
/// original string rather than panicking or splitting it.
#[test]
fn parser_accepts_names_that_need_quoting() {
  let source = concat!(
    "# a comment prologue, skipped\n",
    "id,xkcd_color,retired\n",
    "1,plain name,false\n",
    "2,\"red, orange\",false\n",
    "3,\"vivid \"\"blue\"\"\",true\n",
    // A newline inside a quoted field, and a line that then *starts*
    // with the comment marker — the field must survive whole rather
    // than half of it being eaten as a comment.
    "4,\"a name\n# not a comment\",false\n",
  );

  let rows = parse_ledger(source);
  assert_eq!(
    rows,
    vec![
      (1, "plain name".to_string(), false),
      (2, "red, orange".to_string(), false),
      (3, "vivid \"blue\"".to_string(), true),
      (4, "a name\n# not a comment".to_string(), false),
    ],
  );
}

// ---------------------------------------------------------------------
// ColorId itself
// ---------------------------------------------------------------------

/// `new` / `get` are inverse, over the whole `u16` space including the
/// ids the dataset does not carry.
#[test]
fn color_id_wraps_any_u16() {
  for raw in [0, 1, 42, 949, 950, u16::MAX] {
    assert_eq!(ColorId::new(raw).get(), raw);
  }
}

/// `Display` is the storage-facing face of the id — a bare number, the
/// same text a database column or a log line would carry.
#[test]
fn color_id_displays_as_its_number() {
  assert_eq!(ColorId::new(0).to_string(), "0");
  assert_eq!(ColorId::new(949).to_string(), "949");
  assert_eq!(ColorId::new(u16::MAX).to_string(), "65535");
}

/// Ordering follows the number, so ids sort and range-scan the way a
/// storage layer expects.
#[test]
fn color_id_orders_by_number() {
  assert!(ColorId::new(1) < ColorId::new(2));
  assert_eq!(ColorId::new(7), ColorId::new(7));
  let mut ids = [ColorId::new(9), ColorId::new(1), ColorId::new(5)];
  ids.sort_unstable();
  assert_eq!(ids, [ColorId::new(1), ColorId::new(5), ColorId::new(9)]);
}
