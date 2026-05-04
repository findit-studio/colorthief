//! Look up the named xkcd palette entry closest to an arbitrary RGB
//! (no MMCQ — direct nearest-neighbor lookup).
//!
//! Run with: `cargo run --release --example lookup -p colorthief-dataset`

use colorthief_dataset::{Color, Family};

fn main() {
  // Find the palette entry closest to a specific RGB.
  let c = Color::nearest_to([189, 108, 72]);
  println!("RGB [189, 108, 72]:");
  println!("  name        = {:?}", c.name());
  println!("  common_name = {:?}", c.common_name());
  println!("  design_name = {:?}", c.design_name());
  println!("  family      = {:?}", c.family().as_str());
  println!("  kind        = {:?}", c.kind().as_str());
  println!("  hex         = {:?}", c.hex());
  println!("  is_neutral  = {}", c.is_neutral());

  // Iterate the full table — count entries by family.
  let blues = Color::all()
    .iter()
    .filter(|c| c.family() == Family::Blue)
    .count();
  println!("\n{blues} entries in the Blue family");
}
