//! Look up the named xkcd palette entry closest to an arbitrary RGB
//! (no MMCQ — direct nearest-neighbor lookup).
//!
//! Run with: `cargo run --release --example lookup -p colorthief-dataset`

use colorthief_dataset::{Color, ColorId, Family};

fn main() {
  // Find the palette entry closest to a specific RGB.
  let c = Color::nearest_to([189, 108, 72]);
  println!("RGB [189, 108, 72]:");
  println!("  id          = {}", c.id());
  println!("  name        = {:?}", c.name());
  println!("  common_name = {:?}", c.common_name());
  println!("  design_name = {:?}", c.design_name());
  println!("  family      = {:?}", c.family().as_str());
  println!("  kind        = {:?}", c.kind().as_str());
  println!("  hex         = {:?}", c.hex());
  println!("  is_neutral  = {}", c.is_neutral());

  // The id is what you put in a database column or a wire message —
  // it is permanent, so it still names this entry after the dataset is
  // corrected or regenerated. Round-trip it back.
  let stored: u16 = c.id().get();
  let recovered = Color::from_id(ColorId::new(stored)).expect("an assigned id always resolves");
  println!(
    "\nstored id {stored} resolves back to {:?}",
    recovered.name()
  );

  // Ids the dataset does not carry resolve to nothing — 0 is reserved,
  // so a zeroed column fails loudly instead of naming the first entry.
  println!("id 0 resolves to {:?}", Color::from_id(ColorId::new(0)));

  // Iterate the full table — count entries by family.
  let blues = Color::all()
    .iter()
    .filter(|c| c.family() == Family::Blue)
    .count();
  println!("\n{blues} entries in the Blue family");
}
