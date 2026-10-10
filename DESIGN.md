---
name: "CISC3024 Assignment 2 — observation register"
description: "An observatory transit register: cool paper, blue-black ink, colour reserved for seven calibrated hairlines."
colors:
  paper: "#e6e9e6"
  paper-deep: "#dce0dd"
  ink: "#111a24"
  ink-mid: "#33414f"
  ink-weak: "#4a5765"
  rule: "rgba(17, 26, 36, 0.42)"
  rule-mid: "rgba(17, 26, 36, 0.24)"
  rule-weak: "rgba(17, 26, 36, 0.10)"
  wl-405: "#4a3598"
  wl-436: "#1c4fbe"
  wl-486: "#0d6ba6"
  wl-546: "#1b7a4b"
  wl-589: "#7c600c"
  wl-615: "#a8551a"
  wl-656: "#ab2a1e"
typography:
  masthead:
    fontFamily: "Archivo, 'Helvetica Neue', Arial, sans-serif"
    fontSize: "clamp(1.6rem, 4.2vw, 3.1rem)"
    fontWeight: 700
    lineHeight: 1.08
    letterSpacing: "-0.012em"
  entry-title:
    fontFamily: "Archivo, 'Helvetica Neue', Arial, sans-serif"
    fontSize: "clamp(1.06rem, 1.9vw, 1.42rem)"
    fontWeight: 640
    lineHeight: 1.16
    letterSpacing: "-0.008em"
  body:
    fontFamily: "Archivo, 'Helvetica Neue', Arial, sans-serif"
    fontSize: "16px"
    fontWeight: 400
    lineHeight: 1.75
    letterSpacing: "0.005em"
  stamp:
    fontFamily: "Archivo, 'Helvetica Neue', Arial, sans-serif"
    fontSize: "11.2px"
    fontWeight: 600
    lineHeight: 1.75
    letterSpacing: "0.16em"
  measured:
    fontFamily: "'JetBrains Mono', 'SFMono-Regular', Consolas, monospace"
    fontSize: "13.76px"
    fontWeight: 400
    lineHeight: 1.6
    letterSpacing: "-0.01em"
rounded:
  none: "0px"
spacing:
  hairline: "1px"
  double-rule: "3px"
  pad: "clamp(18px, 3.2vw, 44px)"
  line: "28px"
  tight: "8px"
  gap: "22px"
components:
  tick-open:
    textColor: "{colors.ink}"
    rounded: "{rounded.none}"
    padding: "6px 10px"
  tick-idle:
    textColor: "{colors.ink-weak}"
    rounded: "{rounded.none}"
    padding: "6px 10px"
  pick-pressed:
    textColor: "{colors.ink}"
    rounded: "{rounded.none}"
    padding: "2px 0 1px"
  entry-head-hover:
    backgroundColor: "rgba(17, 26, 36, 0.032)"
    textColor: "{colors.ink}"
    rounded: "{rounded.none}"
    padding: "16px 44px 15px"
  focus-ring:
    textColor: "{colors.wl-589}"
    rounded: "{rounded.none}"
---

# DESIGN.md — the observation register

Recorded after the Gate 4 build of `webapp/index.html`, whose direction contract lives at
`.impeccable/surfaces/webapp-index-html.md`. This file owns the durable visual decisions; the surface
brief owns the strategy for that one page.

## Overview

The page is a register of observations, not a dashboard. Every result is one ruled entry carrying a
station number, a measured value, its spread, and the conditions under which it was taken — the way an
observatory writes up a transit determination. Nothing is arranged as equal cards on a neutral grid,
and there is no hero metric.

The reader is a course marker and a supervisor who must trust numbers they did not produce. Trust comes
from provenance that keeps repeating on screen (the split seed, the file every value came from, the
limits held next to the finding), not from polish.

## Colors

Paper is `#e6e9e6` with a faint ruled texture; ink is near-black blue `#111a24`, never pure black.
Three ink steps carry hierarchy: `ink`, `ink-mid`, `ink-weak` — all clear 4.5:1 on the paper, and
`ink-weak` is the only one used for small copy.

Colour is rationed to seven hairlines named after calibrated spectral wavelengths
(405 / 436 / 486 / 546 / 589 / 615 / 656 nm). Five belong to data series; 405 is reserved for reference
marks (the ideal diagonal, chance, a closed-form value) and 589 for whatever the reader has selected.
Hue never encodes state, severity, or quality, and there are no fills, gradients, or accent surfaces
anywhere in the system.

The one sanctioned exception is the print stylesheet, which sets `#fff` paper and `#000` ink: a register
is meant to be printed on office paper, and tinting that sheet with the screen's ground would waste
ink without adding information.

## Typography

One grotesque — variable Archivo, self-hosted as a subset woff2 under SIL OFL — at essentially one size,
ranked by ink weight and tracking rather than by scale: `wdth` 64–88 with `wght` 400–700 does the
structural work. Monospace (variable JetBrains Mono, also subset and self-hosted) is reserved strictly
for measured values, which is data, not costume: tabular figures, `-0.01em`, and no running prose in it.

Archivo's subset lacks α, λ, τ, Σ, ‖ and ₖ, so Greek letters appear only inside SVG plates and numeric
cells (JetBrains Mono carries them); every label set in the grotesque spells the name out — "alpha
minus one", "prior precision lambda". Body measure is capped near 74ch; display tops out at 3.1rem.

## Layout

A full-bleed ruled masthead band states station, instrument, epoch, split seed, reduction, and observer,
closed by a double rule. Below it a two-column shell: an 18% rail of ticks on the left, the entries on
the right at full scale. The rail is sticky; each tick is a hairline stub with a Roman numeral, and the
open tick blooms to its full width and reveals its entry name. Ticks are real anchors, so deep links and
keyboard focus work without script.

Entries are one-per-view: opening an entry collapses the others, and the entry number hangs in the
margin beside its title. Spacing follows a 28px line rhythm, with more space above a heading than below
it (34px / 12px), and every table sits in its own horizontal scroller so a wide reduction never widens
the page.

## Elevation & Depth

There are no shadows. Depth is carried entirely by rule weight — `rule` (0.42) for structural edges,
`rule-mid` for groupings, `rule-weak` for the paper texture and grid lines — plus the double rule that
opens and closes the document. A zero-offset coloured halo would be decoration, and decoration of that
kind is not part of this world.

## Shapes

No radii, no borders thicker than 1px, no geometric masks. The only repeated shapes are hairline rules,
whisker end-caps, and the struck mark (a rule crossed by a slash). Thumbnails are square 32px sources
enlarged with `image-rendering: pixelated`, because the specimen is the data.

## Components

`tick` (rail navigation), `entry` (head + body, one open), `pick` (a control that names its quantity and
marks itself with an underline plus a longer stub when pressed), `plate` (a labelled SVG drawing with its
own caption carrying the measured summary), `register table` (row header = series swatch + name, numeric
columns in tabular mono, one standard deviation drawn as a trailing whisker scaled per column, a prime on
the best defined value), `readout` (a ruled micro-grid of label/value pairs), and `gallery` (48 specimens,
misses greyed and flagged).

Motion is a single authored moment: when an entry opens, its plates ink themselves left to right
(`clip-path` wipe, 0.7s, `cubic-bezier(0.16, 1, 0.3, 1)`, staggered 90ms for the second and third
figure), and `prefers-reduced-motion` suppresses it. Hover, focus, disabled, loading, error and empty
states are all styled; selection, caret, scrollbars and focus rings are themed from the palette.
Printing opens every entry, because a register is meant to be printed.

## Do's and Don'ts

- **Do carry state in line form**: solid = measured, dashed = ablation, half-height = partial or
  single-seed, struck = the quantity does not exist for that configuration.
- **Don't plot an absent quantity as a number.** A deterministic model has no epistemic term; it is
  drawn struck. A constant score gives a degenerate AUROC of 0.50; it is drawn struck, with the reason
  in the cell's accessible text.
- **Don't use hue to mean good, bad, warning, or state.** Hue identifies exactly one data series.
- **Do show provenance with the number** — the split seed, the source file, the seed count, and the
  limits that travel with a finding.
- **Don't smooth over a vacuous result.** The `none` curvature ablation reproduces the main run because
  both take the same code branch; the page says so and labels the control "identical to main".
- **Do keep the copy in the register's voice**: controls name their quantity, notes name the mechanism,
  and nothing on the page is a number that is not also in the exported files.
