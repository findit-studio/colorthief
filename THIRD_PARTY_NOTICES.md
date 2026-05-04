# Third-Party Notices

`colorthief` and `colorthief-dataset` redistribute and build on the
following third-party works. Each section reproduces the upstream
attribution so downstream consumers can satisfy their own license
obligations.

---

## 1. xkcd color survey results

- **Source:** Randall Munroe, *Color Survey Results* (2010).
  https://blog.xkcd.com/2010/05/03/color-survey-results/
- **Files used:** the 949 `(name, hex, R, G, B)` rows that anchor every
  `Color::name()` / `Color::hex()` / `Color::rgb()` accessor in
  `colorthief-dataset`. Embedded in the generated table at
  `colorthief-dataset/src/generated.rs` (codegen input lives at
  `colorthief-dataset/assets/color_hierarchy.csv`).
- **License:** Public domain. Per the survey announcement, Randall
  Munroe placed the raw color-survey results in the public domain (CC0
  / public-domain dedication). No further attribution is legally
  required, but we credit the source out of courtesy.

## 2. Stitch Fix `colornamer`

- **Source:** Stitch Fix Inc., *colornamer*.
  https://github.com/stitchfix/colornamer
- **Files used:** `static/color_hierarchy.csv` from the upstream Python
  package — specifically the `design_*`, `common_*`, `color_family`,
  `color_type`, and `color_or_neutral` columns layered on top of the
  xkcd anchors. Vendored verbatim at
  `colorthief-dataset/assets/color_hierarchy.csv` and consumed by
  `xtask` to emit `colorthief-dataset/src/generated.rs`.
- **License:** Apache License, Version 2.0. The full text is
  reproduced in `LICENSE-APACHE`. Per Section 4 of the Apache 2.0
  license the upstream copyright notice is preserved here:

  ```
  Copyright 2020 Stitch Fix, Inc.

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
  implied. See the License for the specific language governing
  permissions and limitations under the License.
  ```

  No upstream `colornamer` source code is redistributed — only the
  static color-hierarchy CSV.

## 3. Sharma et al. CIEDE2000 reference values

- **Source:** Gaurav Sharma, Wencheng Wu, Edul N. Dalal,
  *The CIEDE2000 Color-Difference Formula: Implementation Notes,
  Supplementary Test Data, and Mathematical Observations*,
  Color Research & Application, 30(1):21–30, 2005.
- **Files used:** Test vectors from "Table 1" reproduced inline in the
  unit tests at
  `colorthief-dataset/src/nearest/ciede2000.rs::tests` (e.g.
  `sharma_table_1_row_1`). Used to pin our scalar CIEDE2000
  implementation against the published reference values.
- **License:** The paper itself is copyrighted by Wiley; the test
  vectors are factual reference data, included here under fair use
  for verification purposes only. No paper text is redistributed.

---

## Crate runtime dependencies

The runtime workspace crates (`colorthief`, `colorthief-dataset`)
depend on:

| Crate | License | Purpose |
|---|---|---|
| `thiserror` | MIT OR Apache-2.0 | `#[derive(Error)]` for `RgbFrameError`. |

All runtime dependencies are MIT- or Apache-2.0-licensed and
license-compatible with the dual MIT/Apache-2.0 licensing of this
project.

Build-only dependencies (`xtask`, dev-dependencies, benchmark
harnesses) are not redistributed at runtime and are listed in the
workspace `Cargo.lock` for full reproducibility.
