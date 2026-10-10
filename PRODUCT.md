<!-- impeccable:product-schema 1 -->

# Product

## Platform

web

## Stack

Plain static HTML and CSS with hand-authored SVG charts written by hand in JavaScript. No build
step, no package manager, no third-party charting library. Chosen by the supervisor on 2026-10-10 for
a GitHub Pages deployment with no backend, so the repository stays small and the page is readable
without a bundler.

Live browser inference is explicitly out of scope for this version: the page renders a precomputed
gallery rather than running the model in WebAssembly.

## Users

Primary: a university pattern-recognition course marker reading this page as part of a graded
assignment submission. They are checking, within minutes, that a real experiment was run, that the
reported numbers are the numbers the model produced, and that the uncertainty quantities shown mean
what the report says they mean.

Secondary: the supervisor of the project, who commissioned the AI-led research workflow and needs to
inspect intermediate quantities (per-class evidence, posterior spread, prior precision) rather than
only headline accuracy.

## Product Purpose

A single-page visualiser for the experimental results of CISC3024 AI Assignment #2: a recent Bayesian
classification algorithm (F-EDL, flexible-Dirichlet evidential deep learning, NeurIPS 2025) with a
post-hoc Bayesian comparator (identity-curvature last-layer Laplace, after WACV 2025) and deterministic
baselines, evaluated on CIFAR-10 with two out-of-distribution control sets.

Success means the marker can see, without reading the PDF, what the Bayesian quantities are, how the
five configurations compare, and that the visualisation is generated from the same data as the tables.

## Positioning

Every number on the page is read from artefacts produced by the same code path that produced the
reported results — `webapp/data/manifest.json` and `samples.json`, written by
`src/export_webapp_data.py`, which rebuilds the models from the stored checkpoints and refits the
post-hoc layers. The page therefore cannot display a figure that disagrees with its own table, which
is the property a hand-built demo dashboard would not have.

## Operating Context

- Served as a static GitHub Pages site from this repository; also opened directly from disk during
  local testing.
- Companion to a Word/PDF report submitted through UMMoodle and screened by Turnitin.
- The assignment requires a webpage link to the source code, so this page and the repository are the
  same deliverable.
- Offline-capable: no CDN, no network calls, no analytics.

## Capabilities and Constraints

- Reads two local JSON files (`data/manifest.json`, `data/samples.json`) and PNG images from
  `data/samples/`.
- Ten required panels: configuration comparison with cross-seed error bars; reliability diagrams;
  risk-coverage curves; OOD detection bars; epistemic/aleatoric decomposition; per-image inspector
  with Dirichlet evidence and posterior spread; Laplace evidence-versus-precision curve; confusion
  matrix; training curves; reproducibility and provenance.
- Hard constraint: no build step, no external dependencies, works from `file://` as well as over
  HTTP. Fetching local JSON under `file://` is blocked by browsers, so the page must degrade to an
  explanatory state rather than fail silently, and the supported way to view it is a local static
  server or the deployed site.
- Data on hand is limited to 3 seeds, one dataset (CIFAR-10), 48 gallery images, and two OOD
  controls. Nothing may be shown that implies more.
- Live inference and image upload are out of scope by the supervisor's decision.

## Brand Commitments

None. This is course work, not a product with an identity.

## Evidence on Hand

- `webapp/data/manifest.json` — aggregate metrics with mean, standard deviation and seed count for
  five configurations; per-seed detail; fitted Laplace prior precision and calibrator parameters;
  dataset and experiment provenance.
- `webapp/data/samples.json` — per-image predictive mean, variance, epistemic and aleatoric scores,
  plus the raw Dirichlet quantities (`alpha`, `p`, `tau`, `alpha0`, evidence) for the evidential
  model, across all five configurations.
- `webapp/data/samples/*.png` — 48 real CIFAR-10 test images, stratified by uncertainty behaviour.
- `analysis/test_predictions.npz` — full per-image arrays for every configuration, split and seed
  (source of the curve data; a Python-side artefact, not shipped to the browser).
- `report_assets/*.png` — 18 static figures used by the PDF report.
- `runs/full_1791561662/matrix_results*.json` — raw experiment output including the curvature
  ablation.
- Absences that must not be fabricated: no user testimonials, no external validation, no comparison
  against other published Bayesian methods beyond the cited papers' own reported numbers, no
  cross-dataset results, no spectral-normalisation ablation with three seeds (single seed only, and
  it is still running).

## Product Principles

1. The page is evidence, not a demo: every visual states its configuration, its seed set, and whether
   it comes from the validation or the test split.
2. Honest absence beats decorative fill: where a quantity does not exist for a configuration, the
   page says so rather than plotting zero.
3. The Bayesian quantities are the subject. Uncertainty is shown as a distribution or a spread, not
   as a single scalar badge.
4. Legibility for print and for projection, since the marker may read it either way.
5. Reproducibility is a first-class panel, not a footer.

## Accessibility & Inclusion

No formal standard was required by the assignment. Practical requirements taken from the use scene:
text contrast adequate for projection in a lit room, all charts readable without colour alone
(patterns and direct labels in addition to hue), keyboard-operable controls, and a reduced-motion
path.
