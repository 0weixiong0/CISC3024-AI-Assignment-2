# Stage 04 — Frontend implementation and feasibility report

Project: CISC3024 AI Assignment #2 (Pattern Recognition, University of Macau, Fall 2026)
Milestone: M5 interface (assignment component (6), "a webpage link of your source codes")
Date: 2026-10-11
Language of this record: English, per the collaboration contract. Chat with the supervisor is Chinese;
every artefact that a marker reads is English.

---

## 1. What now exists

| Absolute path | Content | Size |
| --- | --- | --- |
| `D:\claude\workspace\CISC3024 AI Assignment 2\webapp\index.html` | the register: masthead + ten entries, static markup so structure survives a script failure | 8 kB |
| `D:\claude\workspace\CISC3024 AI Assignment 2\webapp\styles.css` | the whole visual system, hand-written, no framework | 19 kB |
| `D:\claude\workspace\CISC3024 AI Assignment 2\webapp\app.js` | hand-authored SVG plates, tables, the interactive evidence gallery | 64 kB |
| `D:\claude\workspace\CISC3024 AI Assignment 2\webapp\data\dataset.js` | the generated data bundle the page reads | 226 kB |
| `D:\claude\workspace\CISC3024 AI Assignment 2\webapp\data\manifest.json` | machine-readable reduction (all 36 quantities × 5 configurations × 3 seeds, curves, cost, variants) | 80 kB |
| `D:\claude\workspace\CISC3024 AI Assignment 2\webapp\data\samples.json` | per-image posteriors for 48 test images × 5 configurations + the Dirichlet internals of A1 | 162 kB |
| `D:\claude\workspace\CISC3024 AI Assignment 2\webapp\data\samples\*.png` | the 48 actual test images | 48 files |
| `D:\claude\workspace\CISC3024 AI Assignment 2\webapp\fonts\` | Archivo + JetBrains Mono, variable, subset to woff2, with both OFL licences | 146 kB |
| `D:\claude\workspace\CISC3024 AI Assignment 2\DESIGN.md` | the durable visual system, recorded after the finish review | — |
| `D:\claude\workspace\CISC3024 AI Assignment 2\.impeccable\surfaces\webapp-index-html.md` | the direction contract this build was judged against | — |

Total shipped folder: 937 kB, no build step, no dependency, no network call.

## 2. What the page contains

Ten register entries, navigated by an off-centre rail of ticks:

| Entry | Content | Data source |
| --- | --- | --- |
| I | five-configuration reduction, headline table + all 36 recorded quantities | `manifest.models` (3 seeds) |
| II | reliability, one plate per configuration | `manifest.curves` (seed 20261006) |
| III | risk against coverage, five curves + selective-prediction table | `manifest.curves` (seed 20261006) |
| IV | out-of-distribution controls, near (SVHN) and far (Tiny-ImageNet) | `manifest.models` (3 seeds) |
| V | epistemic / aleatoric decomposition | `manifest.models` (3 seeds) |
| VI | evidence anatomy: 48 selectable test images, per-class evidence and posterior | `samples.json` (seed 20261006) |
| VII | seed spread, plotted per run with a quantity picker | `manifest.per_seed` (3 seeds) |
| VIII | ablations: curvature identity/kron/none and spectral normalisation off | `manifest.variants` (4 runs) |
| IX | post-hoc instrument readings: lambda, evidence, temperature, Dirichlet scale | `manifest.per_seed` + `manifest.cost` |
| X | measured cost, provenance, limits, honest absences | `manifest.cost` + `manifest.notes` |

**No number is computed in the browser.** Every figure is read from the bundle, which
`src/export_webapp_data.py` generates from the same `matrix_results*.json` files the report tables use.

## 3. The contract commitment that drove the design

The direction contract fixes one rule above the others: **state is carried by line form, never by
hue.** Solid rule = measured; dashed = ablation; half-height = partial or single-seed; struck = the
quantity does not exist for that configuration.

That rule is load-bearing on this dataset, not decorative:

* B1, B2 and A2 have no epistemic term, and their error-detection AUROC is the degenerate `0.5000` a
  ranking formula returns when its score is constant. All of those are struck — 34 struck cells in
  entry I, 19 in IV, 10 in V, 7 in VIII. Nothing on the page plots them as 0.5 or as 0.0.
* The `none` curvature ablation reproduces the main run bit-for-bit because `identity` and `none` take
  the same branch in `src/laplace_icla.py:107`. The variant control is labelled "identical to main" and
  the panel states that the ablation tests nothing, matching Stage 03b §7c caveat 2.
* Entries II, III and VI come from one seed's checkpoints, so their plates are drawn with the
  half-height rule weight and their entry marks say "seed 20261006 only".

Colour is rationed to seven hairlines named after calibrated wavelengths (405/436/486/546/589/615/656
nm); five carry the series, 405 carries reference marks, 589 carries the reader's cursor. There are no
fills, no gradients and no accent surfaces anywhere, including on the miss marker in the gallery, which
is a struck rule across the thumbnail rather than a coloured badge.

## 4. Verification actually performed

| Check | Method | Result |
| --- | --- | --- |
| Console errors | live browser, both `http://` and `file://` | none |
| Contrast | script walking every text-bearing element with every entry open, computing WCAG ratio against the resolved background | 0 failures at 1430 px |
| Page overflow | `documentElement.scrollWidth − clientWidth` | 0 at 1430 px and at 380 px |
| Table overflow | per `.table-wrap` client vs scroll width | 344 px viewport holds a 5917 px table by internal scrolling |
| Fonts | `document.fonts` status + computed `font-stretch`/`font-weight` | both loaded; width axis 64–88 % and weight 400–700 measured live on the rendered elements |
| `file://` fallback | opened the page directly from disk | data, 48 images and both fonts load; all ten entries populate |
| Interactions | scripted clicks on the rail, the gallery, and both picker groups | variant/metric switches repaint, gallery repaints both plates, captions and marks update |
| Keyboard | tab order enumeration | 32 visible tab stops; ticks are real anchors, so deep links work without script |
| Focus | computed `:focus-visible` style | amber 1 px outline + offset ring |
| Reduced motion | `@media (prefers-reduced-motion)` | the ink wipe is suppressed; scroll behaviour switches to `auto` |
| Mechanical lint | `impeccable detect --json` | 16 wide-tracking warnings (adjudicated false positives, §6), 3 advisories (adjudicated, §6) |
| Struck geometry | bounding-box comparison of every struck mark and label against its plate | none outside its plate |
| Smallest chart text | `getScreenCTM()` scale on every SVG label | 8.5 px at desktop, 9 px on a phone after the fold-out fix |

## 5. Finish review and disposition of every finding

An independent reviewer was briefed with the contract and the data files and returned **FIX-FIRST** with
13 defects. Disposition:

| # | Finding | Action |
| --- | --- | --- |
| 1 | Entry VIII claimed "once without spectral normalisation" and "Four completed run variants" while only three existed | Fixed. The count is now written from the data, and the fourth variant landed 2026-10-11 00:05, so the claim is now true and self-updating |
| 2 | Entry IX plate domain started at 0 while its ticks used the padded domain, pinning all three lambda marks to the right edge | Fixed; domain is now the padded range and the plate has a numeric axis |
| 3 | Entry VII struck marks began 42 px outside the plate, over the row labels | Fixed to the plate's left inset |
| 4 | Entries II, III and VI silently used seed 20261006 while entry VI claimed to be "exactly what the report tables counted" | Fixed by disclosure, not deletion: the three entries are labelled single-seed, their plates take the half-height partial weight, and the false identity claim is gone. Exporting all three seeds' curves is listed as remaining work |
| 5 | "All five curves share the same accuracy at full coverage" is false (terminal risk .0882–.0928) | Fixed; the copy now states the real range |
| 6 | Half-height was promised four times and applied to nothing | Fixed; it now marks exactly the three single-seed plates |
| 7 | At 390 px the rail was a 380×560 sticky wall because the flex rule sat on the wrong element | Fixed; the tick container is now the flex row and the strip measures 380×84 |
| 8 | At 390 px every in-chart label rendered at 5–6.7 px | Fixed; plates now carry an intrinsic size and fold out horizontally at full scale, measured 9 px |
| 9 | Activating a rail tick did not move the reader | Fixed; ticks scroll to their entry, with the scroll gated by reduced motion |
| 10 | 56 of 59 reliability tooltips were attached to the 1 px connector instead of the rule | Fixed |
| 11 | A1's epistemic rule was drawn seven times its measured length so it would be visible | Fixed; it is now drawn at its true length and the number carries the reading |
| 12 | `svg text { fill: … }` silently overrode every inline series colour | Fixed with `:not([fill])`; the series-coloured values now render and still pass 4.5:1 |
| 13 | Minor set: "four limits" vs five items, a parameter count typed into the markup instead of the bundle, backslashes in provenance paths, plate A header overlapping the tallest bar, `background-attachment: fixed` leaving the rules behind the sheet, and a filled grayscale miss badge | All fixed. The parameter count is now exported into the bundle; the miss marker is a struck rule |

## 6. Findings deliberately not acted on

* **16 `wide-tracking` warnings.** Body copy is at `0.005em`; every rule at or above `0.05em` is a
  0.7 rem uppercase stamp (STATION, INSTRUMENT, ENTRIES, QUANTITY), which is what the register's own
  lettering requires. The reported snippets (`0.05em`, `0.06em`) do not exist in the stylesheet and are
  attributed to `index.html:0`, so the tool is bucketing rather than measuring. Accepted as false
  positives.
* **The ruled-paper `repeating-linear-gradient` advisory.** The committed world names a ruled paper
  ground; it is one hairline per 28 px at 7 % ink. Accepted, and recorded in `DESIGN.md`.
* **`#fff`/`#000` in the print stylesheet.** A deliberate addition, documented in `DESIGN.md` §Colors:
  a register is meant to be printed on office paper.

## 7. Known limits of this milestone

1. **Entries II, III and VI are single-seed.** The underlying per-seed predictive means for all three
   seeds are on disk in `predictives.npz`, so this is an export limitation, not a data limitation. It is
   disclosed on the page in three places.
2. **No client-side inference.** The brief's stage-4 wording mentions drawing on the dataset; a static
   page cannot classify a hand-drawn image, and the approved product record rules out a backend. Entry X
   states this plainly instead of faking it. If the supervisor wants a drawing affordance, the honest
   options are a downloadable notebook cell or a recorded "this is what the model saw" gallery, both of
   which need his decision.
3. **Figure 11 (interface screenshot) is not yet captured.** The in-app browser surface went hidden
   mid-session; the capture is a two-minute job once it is visible again.
4. **Nothing is pushed.** The public repository exists but is empty, and GitHub Pages is not enabled;
   both need the supervisor's go-ahead.

## 8. Cost

Frontend work occupied 22:30–00:20 on 2026-10-10/11 (about 1 h 50 m of agent time), of which roughly
40 minutes was the finish review and its 13 corrections. The data bundle regenerates in about 4 minutes
on CPU (`--derived-only`) or about 12 minutes with inference.

## 9. Recommendation

Gate 4 is complete enough to look at. Two things need the supervisor:

1. Open `D:\claude\workspace\CISC3024 AI Assignment 2\webapp\index.html` and read it as the marker will;
   tell me if any entry is missing something you want to see, because the data for most additions
   already exists.
2. Decide the drawing question in §7.2 before the report fixes component (4)'s wording.

Then Gate 5: report structure approval, and the English Word report against `Report Template.docx`.

---

# Part 2 — restructuring around the classifier experience (2026-10-11)

## 10. Why the first version was wrong about its own audience

The supervisor's verdict on Part 1: it read like the report. Ten entries of tables and curves are what
a *marker* needs and what a *visitor* cannot parse in a minute. His instruction was to make the page let
someone experience the classifier, Assignment 1 style, and to move the measurement material out of the
foreground.

Assignment 1's own frontend settled the feasibility question: `webapp_standalone/` ships
`model/model.onnx` plus `ort/ort.all.min.js` and runs real inference in the tab, so the pattern was
proven on this machine, and `onnx` 1.22.0 / `onnxruntime` 1.26.0 are already installed here. No new
dependency, no approval gate.

## 11. What the page is now

| File | Role |
| --- | --- |
| `D:\claude\workspace\CISC3024 AI Assignment 2\webapp\index.html` | **the bench**: draw a 32×32 specimen, upload a photo, or pick a reference image, and watch two instruments answer the same picture |
| `D:\claude\workspace\CISC3024 AI Assignment 2\webapp\register.html` | the ten-entry measurement register from Part 1, demoted to a secondary page and linked from the masthead |
| `D:\claude\workspace\CISC3024 AI Assignment 2\webapp\bench.js` | ORT loading, the five reductions recomputed in JavaScript, the drawing surface |
| `D:\claude\workspace\CISC3024 AI Assignment 2\webapp\model\baseline.onnx` | the cross-entropy network, 58.97 MB, outputs logits and the 512-d features |
| `D:\claude\workspace\CISC3024 AI Assignment 2\webapp\model\evidential.onnx` | the evidential network, 60.56 MB, outputs alpha, p and tau |
| `D:\claude\workspace\CISC3024 AI Assignment 2\webapp\model\verification.json` | the ONNX-versus-PyTorch parity record written by the exporter |
| `D:\claude\workspace\CISC3024 AI Assignment 2\webapp\data\instruments.json` | head weights, temperature, Dirichlet vector, Laplace MAP + posterior standard deviations, preprocessing constants, decline thresholds |
| `D:\claude\workspace\CISC3024 AI Assignment 2\src\export_onnx.py` | the exporter that writes all of the above and refuses to ship if parity fails |

Two ONNX graphs serve all five configurations: B2, A2 and A3 are post-hoc layers, so the browser derives
them from the same features and the exported parameters. The visitor sees the baseline and the evidential
network side by side, plus a five-row table of every configuration's answer, uncertainty and
whether it declines.

The decline line is not invented for the demo: it is the total uncertainty at which the test
risk-coverage curve passes 5 % risk, computed in `abstain_thresholds()` and checked against the reported
`cov_at_risk5` (0.3752 for A1 implies coverage 0.9105, reported 0.9024 across three seeds).

## 12. What the export found on the way

1. **Spectral normalisation tripled the file.** With the parametrisation traced into the graph, every
   layer exported `weight_orig` *and* `weight_v`: 118 MB per model. Folding them into plain weights
   halved it to 59 MB. The first fold attempt silently folded **zero** layers, because this torch build
   attaches spectral norm as a forward pre-hook rather than a parametrization; the guard that caught it
   is now in the script, so the same mistake cannot ship.
2. **Parity is measured against the unfolded model**, computed before folding, on real test images and
   on a blank plate, a scribble and uniform noise -- the inputs a visitor will produce and no training
   test covers. Worst relative difference across all five outputs: **3.19e-04**. The exporter returns
   non-zero if that exceeds 1e-3.
3. **The first abstain thresholds were computed backwards** (`argsort(-scores)` retained the *least*
   confident images first), which produced a coverage of 0.0001 instead of 0.91. The comparison against
   the reported `cov_at_risk5` is what exposed it; that check is now part of the export.

## 13. Verification of the browser implementation

| Check | Result |
| --- | --- |
| Model load | both graphs instantiate; WebGPU provider selected, WASM fallback wired |
| Parity, `test_00284` | browser confidence 0.96273, total evidence 3185.8, tau 5.922; **PyTorch on the identical 8-bit input gives 0.96273 / 3185.8 / 5.922** |
| Parity, `test_00284` baseline | browser 0.99967 vs report 0.99966 |
| Difference against `samples.json` | confidence 0.95933 vs 0.96273, evidence 3051.6 vs 3185.8. Cause identified and reproduced: the register's numbers come from float tensors, the bench sees the 8-bit PNG export. Disclosed on the page rather than hidden |
| Blank plate | all five configurations decline; A1 total uncertainty 0.5314 against a line at 0.3752 |
| Confidently wrong specimen (`test_02405`, true cat, predicted frog) | every configuration is confidently wrong, A1 with 12 583 units of evidence. Stated on the page as the finding it is |
| Contrast, all visible text | 0 failures at 530 px |
| Page overflow | 0; bar labels brought inside their viewBox |
| Overlapping runs | fixed by chaining inference, which also removed a WebGPU "Session already started" fault |

## 14. What the browser cannot check for me

- **Visual review at 1440 px and at 390 px.** The in-app browser surface stayed hidden all session, so
  every measurement above is computed rather than looked at. The layout numbers are recorded
  (canvas 260 px at 530, two-column grid, no overflow) but nobody has *seen* this page yet.
- **Speed on a machine without WebGPU**, and behaviour on a phone.
- **The 118 MB payload on a real connection.** It is served from the same folder, so GitHub Pages
  bandwidth is the only cost, but a marker on hotel wifi will wait.

## 15. Recommendation

Open it and tell me what it feels like: `python -m http.server 8801 --directory webapp` from the project
root, then `http://127.0.0.1:8801/`. The register still opens from disk with no server. If the bench is
the right shape, Part 1's register becomes the report's figures and appendix, and the page keeps the
experience as its front door.

---

# Part 3 — the whole station (2026-10-11, later)

## 16. A correction I owe the record

Asked whether the dataset could go to GitHub "like Assignment 1", I first checked only A1's local
working tree and its `main` branch, found no images, and reported that A1 had not shipped them. That
was wrong. A1 publishes from a **`gh-pages` branch**, which contains **27,000 EuroSAT images across
133.7 MB** (`git ls-tree -r origin/gh-pages`), and the deployed site serves them
(`https://0weixiong0.github.io/CISC3024-AI-Assignment-1/data/images/AnnualCrop/AnnualCrop_1.jpg`
returns HTTP 200). A1's own README line 104 -- "the standalone demo does not include the 27,000
dataset images" -- is true of `main` and false of what is actually published.

The precedent is therefore: **source branch clean, Pages branch carries the data.** The licence
position still differs -- EuroSAT is CC BY 4.0, CIFAR-10's official page states no licence -- but the
course precedent for publishing thumbnails of the evaluation set on the demo branch is explicit.

## 17. What was added

| File | Content |
| --- | --- |
| `D:\claude\workspace\CISC3024 AI Assignment 2\src\export_test_wall.py` | writes the wall: one PNG per test image plus the per-configuration posterior index |
| `D:\claude\workspace\CISC3024 AI Assignment 2\webapp\data\test\00000.png … 09999.png` | all 10,000 test images at the 32×32 the model saw, 22.7 MB |
| `D:\claude\workspace\CISC3024 AI Assignment 2\webapp\data\test-index.json` | 2.66 MB: label, class, and per-configuration predicted class / confidence / epistemic / aleatoric / total, for seed 20261006 |
| `webapp/index.html`, `webapp/bench.js`, `webapp/styles.css` | "The whole station": 10,000 cells, class filter, outcome filter, sort by any uncertainty quantity of any configuration |

PNG rather than JPEG because the wall's thumbnails are also the input the browser feeds the networks;
a lossy recompression would change the posteriors the page then displays. Verified: `test_00284.png`
in the wall and in the curated gallery are byte-identical.

Cells render in chunks of 240 behind an `IntersectionObserver`, images `loading="lazy"`, so the page
does not build 10,000 nodes at once. Each cell carries a hairline whose length is the sort value --
line form for magnitude, hue for the instrument, no fill and no gradient.

## 18. What the wall proved on its own

* **The decline line is self-consistent with the register.** Filtering to "declined by A1" returns
  **896 of 10,000** images, an abstention rate of 8.96 %, against the 8.95 % implied by the reported
  `cov_at_risk5 = 0.9024`. Two independent paths -- the risk-coverage curve in Python and a browser
  threshold from `instruments.json` -- land on the same number.
* **The most epistemic image in the whole test set is `#9370`**: true ship, B1 answers dog at 0.7928,
  A1 answers ship correctly at 0.4588 and then **declines**, total uncertainty 0.7165 against the line
  at 0.3752. Live browser inference on that cell reproduces the index to 0.00541 vs 0.005379 epistemic
  and 0.7165 vs 0.71653 total -- the same 8-bit quantisation gap already documented, in the direction
  and size expected.
* That single cell is the demo the page has been looking for: **the baseline is confidently wrong, the
  evidential model is right and refuses to say so.** It is now the first thing the wall shows when
  sorted by epistemic uncertainty.

## 19. Open before publication

1. **Where the 22.7 MB of images and 119 MB of models live.** Following A1, they belong on the
   `gh-pages` branch and not on `main`. That needs a packager (A1 had `src/package_standalone.py`) and
   a `.gitignore` rule keeping `webapp/data/test/` out of `main`. Not yet written.
2. **Visual review still outstanding.** Every number in Part 3 was measured through the DOM; nobody
   has looked at the wall yet.
3. `webapp/` is now 192 MB in total. GitHub Pages handles it; a marker on a slow connection does not.
   The plate and the register both work without the wall, so a failed image fetch degrades to numbers
   rather than to a broken page.

## 20. The wall moved next to the plate (after the supervisor looked at it)

He reported that picking a specimen at the bottom of the page was awkward, and asked for the
reference-thumbnail strip to become an Assignment 1 style class picker, collapsed by default. Done:

* The fixed 48-thumbnail strip beside the plate is gone. In its place, `<details class="picker">` --
  **"Choose a specimen — all 10 000 test images"** -- a 484×340 scroll box with the class chips,
  outcome chips, sort-quantity chips and sort-model chips above it. It was built collapsed; the
  supervisor looked at it and asked for it open, so it now carries `open` and reveals itself without
  a click. The disclosure stays a `<details>` so the structure is still one keystroke from folding.
* The bottom "whole station" section was deleted rather than duplicated: one picker, next to the
  instrument it feeds. Its uncertainty hairline and the struck diagonal for baseline misses moved with
  it.
* Chunk size dropped from 240 to 60 cells, and the `IntersectionObserver` root is now the scroll box,
  so scrolling inside the picker extends the selection.
* Dead work removed: the bench no longer fetches `samples.json` (162 kB) or runs `specimenStrip()` /
  `preferred()`, which the picker replaced. `samples.json` is still used by the register.
* Verified live with the picker open: class "ship" + sort "A1 epistemic" puts `#9370` first, clicking
  it loads the plate and reproduces the index numbers (epistemic 0.00541 live vs 0.005379 recorded),
  the verdict line reads "Declining to answer", and the picker stays open for the next pick. No page
  overflow, no contrast failure at 530 px.
