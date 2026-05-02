# Third-Party Notices for `colorthief-dataset`

This crate redistributes color-hierarchy metadata from Stitch Fix's
`colornamer` and generates static Rust lookup tables from it.

## xkcd color survey + Stitch Fix hierarchy

The file `assets/color_hierarchy.csv` is sourced from Stitch Fix's
[`colornamer`](https://github.com/stitchfix/colornamer) and is used to
generate `src/generated.rs`.

- Upstream repository: <https://github.com/stitchfix/colornamer>
- License: Apache License 2.0
- The dataset combines:
  - The [xkcd color survey](https://blog.xkcd.com/2010/05/03/color-survey-results/)
    (~950 named RGB colors), placed in the public domain by Randall Munroe.
  - Hierarchical roll-ups (design / common / family / kind) added by Stitch
    Fix, distributed under Apache 2.0.

Reusing the CSV verbatim and emitting derived Rust source files is permitted
under Apache 2.0; this file satisfies the attribution clause.
