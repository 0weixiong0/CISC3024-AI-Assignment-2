---
version: 1
slug: "webapp-index-html"
primary_target: "webapp/register.html"
related_targets: ["webapp/styles.css","webapp/register.js"]
---

# Surface brief: webapp/register.html

> **Repointed 2026-10-11.** This brief was written for `webapp/index.html`. After the supervisor's
> review the register moved to `webapp/register.html` and `index.html` became the live classifier
> bench, which is a different surface with a different job and will need its own brief if it is
> revised. Every decision below still governs the register, and the OWN-WORLD and FORM grammar is
> shared by the bench.

Mode: read
Target: webapp/register.html
Related: webapp/styles.css, webapp/register.js, webapp/data/
Seed key: 135333b1 (scope direction, mode read)
Build path: code-led (recorded in .impeccable/config.json)

## Scope

A single static page visualising the AI Assignment #2 experiment: five configurations, three
seeds, CIFAR-10 plus two out-of-distribution controls, 48 gallery images. Ten panels. No backend,
no build step, no third-party chart library, no live inference.

## Direction contract

THESIS: the page is an observation register, not a dashboard. Every result is one ruled entry with a
station number, a measured value, and its uncertainty written the way an observatory writes a
transit determination - value, then spread, then the conditions under which it was taken. It refuses
the category-default arrangement of equal cards on a neutral grid, and refuses the hero-metric
template.

OWN-WORLD: cool paper ground with a faint ruled texture; ink is near-black blue, never pure black.
Colour exists only as hairlines at seven calibrated wavelengths (405/436/486/546/589/615/656 nm),
one per data series, donated from the spectrographic challenger - no fills, no gradients, no accent
surfaces. One grotesque at essentially one size, ranked by tracking and ink weight; monospace
reserved strictly for measured values, which is data, not costume. A single off-centre rail carries
navigation as ticks; the open tick blooms into its entry, the rest stay hairline stubs.

STORY: the marker arrives and within seconds knows what was measured, on what data, under which
frozen split, and can move from the headline table to a single image's Dirichlet evidence without
leaving the page's grammar. They should believe the numbers because the page keeps showing their
provenance, not because they look polished.

FIRST VIEWPORT: the register masthead across the top - station, instrument, epoch, observer, split
seed - set as a ruled header band with hairline rules. Below it, the rail sits at about 18% from the
left edge; the first entry, the five-configuration comparison table, occupies the remaining width at
full scale, its numeric columns set in tabular monospace with the standard deviation as a trailing
hairline whisker rather than a second number. The primary action is a keyboard-reachable list of
rail ticks; no hero image, no CTA pill.

FORM: the assigned direction from seed 135333b1, index 3 of the grounded list (observatory transit
register), raised by one donated discipline from the competitive challenger
`operate-c-emission-line-rail`: state is carried by line form, never hue - solid rule for a live
result, dashed for an ablation, half-height for a partial or single-seed result, and a struck rule
for a quantity that does not exist for a configuration (a deterministic model has no epistemic term,
which is shown as struck, not as 0.5).

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the
verdict, DESIGN.md, and every shipping raster carrying its provenance.

## Unresolved

- ~~Fonts~~ resolved at build time: Archivo and JetBrains Mono, both variable, subset to woff2 and
  self-hosted under `webapp/fonts/` with their OFL licences. Archivo's subset lacks α, λ, τ, Σ and ‖, so
  Greek appears only in monospace (plates and numeric cells) and every grotesque label spells the name.
- Entries II, III and VI read from seed 20261006's exported curves and checkpoints. Disclosed on the
  page in three places and drawn with the partial-run rule weight; per-seed curves exist in
  `predictives.npz` if a later pass wants to plot all three.
