# Stage 03b — Full-Scale Results, Hypothesis Verdicts and Figure Set (Gate 3b)

Project: CISC3024 AI Assignment #2 · Date: 2026-10-10 · Language: English
Milestones covered: M3 (full runs) and part of M4 (evidence/λ analysis). **M4 is not complete** — see
Section 8 for the ablations still outstanding, which is the main thing the supervisor needs to
approve continuing.

Supervisor decision this document supports: whether the backend is good enough to move to the HTML
visualisation (Gate 4) and the report (Gate 5).

---

## 1. What actually ran

Three independent seeds, each retraining both backbones and refitting every post-hoc layer on those
weights. The frozen split was **not** redrawn between seeds (seed `20261006` owns the split); what
varies between seeds is initialisation and shuffling order only. The test split was opened once per
(configuration, seed) pair.

| | seed 20261006 | seed 20261007 | seed 20261008 |
| --- | --- | --- | --- |
| B1 epochs actually run (early stop) | 36 | 36 | 40 |
| A1 epochs actually run (early stop) | 47 | 54 | 38 |
| B1 best validation NLL | 0.2817 | 0.2756 | 0.2743 |
| A1 best validation NLL | **0.2730** | 0.2855 | 0.2959 |
| ICLA prior precision λ (fitted) | 535.5 | 538.2 | 519.2 |
| λ from the closed form `P/‖W_MAP‖²` | 535.5 | 538.2 | 519.2 |
| Temperature T | 1.452 | 1.452 | 1.576 |
| Peak VRAM (B1 / A1) | 424 / 544 MB | 655 / 601 MB | 655 / 664 MB |
| Seed wall-clock | 3367 s | 3457 s | 2771 s |

Total experiment wall-clock **9595 s ≈ 2.7 hours** on the RTX 4060 Laptop, GPU utilisation 25–38 %
(the run is CPU-data-loading bound because DataLoader workers are disabled to avoid the Windows
spawn deadlock recorded in Stage 03a).

## 2. Measured cost versus the plan

| Plan figure | Was `[estimate]` | Now measured |
| --- | --- | --- |
| Full run, one backbone, 100 epochs | ≈ 84 min | 47–54 min actual (early stopping at 36–54 epochs) |
| M3 both backbones, three seeds | 2.8–4.5 h `[projection]` | **2.7 h measured** |
| Backend total including ablations | 6–9 h `[estimate]` | 2.7 h for M3; ablations still to run |
| Peak VRAM | 430 MB (pilot) | 424–664 MB |

The schedule assumption holds with room to spare.

## 3. Full results — in-distribution quality (test split, mean ± std over 3 seeds)

| Configuration | Top-1 | Macro-F1 | NLL ↓ | Brier ↓ | ECE ↓ | Adaptive ECE ↓ | Classwise ECE ↓ | AURC ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B1 cross-entropy | .9119 ± .0004 | .9117 | .2951 ± .0094 | .1337 | .0386 | .0386 | .0085 | .0121 |
| B2 CE + temperature | .9119 ± .0004 | .9117 | **.2611 ± .0033** | **.1276** | **.0080** | **.0063** | **.0041** | .0120 |
| A1 F-EDL | .9077 ± .0012 | .9075 | .3054 ± .0012 | .1401 | .0334 | .0334 | .0074 | .0138 |
| A2 F-EDL + Dirichlet scaling | .9072 ± .0013 | .9070 | .2958 ± .0076 | .1369 | .0145 | .0172 | .0050 | .0140 |
| A3 CE + ICLA (identity curvature) | .9118 ± .0005 | .9116 | .2805 ± .0045 | .1312 | .0305 | .0303 | .0072 | .0123 |

Seed-to-seed spread is tiny (top-1 std ≤ 0.13 points), so the ordering below is not noise.

## 4. Full results — uncertainty quality

Out-of-distribution detection AUROC (higher is better; a larger uncertainty score should mark the
out-of-distribution sample):

| Configuration | near-OOD SVHN | far-OOD Tiny-ImageNet |
| --- | --- | --- |
| B1 cross-entropy | .9119 ± .0244 | .8463 ± .0028 |
| B2 CE + temperature | .9177 ± .0247 | **.8568 ± .0029** |
| A1 F-EDL (epistemic) | **.9271 ± .0201** | **.8544 ± .0051** |
| A1 F-EDL (total) | .9258 ± .0167 | .8517 ± .0029 |
| A2 F-EDL + Dirichlet (total) | .9277 ± .0139 | .8547 ± .0030 |
| A3 CE + ICLA (epistemic) | .8827 ± .0194 | .8085 ± .0111 |
| A3 CE + ICLA (total) | .9096 ± .0248 | .8429 ± .0030 |

Error-detection AUROC — does the uncertainty score know when the model itself will be wrong:

| Configuration | epistemic score | aleatoric score |
| --- | --- | --- |
| B1 / B2 | .5000 (no epistemic term exists) | .9185 / .9180 |
| A1 F-EDL | **.9076 ± .0029** | .9129 |
| A2 F-EDL + Dirichlet | .5000 (calibration wrapper does not carry variance) | .9128 |
| A3 CE + ICLA | .9057 ± .0089 | .9174 |

## 5. Hypothesis verdicts

**H1 — F-EDL reaches within ±1.5 points of its published 91.19 % top-1. CONFIRMED.**
Measured 90.77 ± 0.12 %, i.e. 0.42 points below the published figure, well inside the band declared
before the run. The comparison is fair in the sense that the same backbone family, dataset and
optimiser budget are used, and honest in noting that our held-out validation split is 2,500 images the
paper may not have removed.

**H1 has an uncomfortable corollary that must be reported with it:** our plain cross-entropy baseline
on the same backbone reaches **91.19 %**, numerically identical to the published F-EDL accuracy. So
"the recent Bayesian algorithm reproduces its paper's accuracy" and "a default baseline matches that
accuracy" are both true of this experiment. Neither statement is a mistake, and the report has to carry
both.

**H2′ — F-EDL's calibration, measured rather than inherited. PARTLY CONFIRMED.**
Uncalibrated, the evidential model is better calibrated than the uncalibrated baseline
(ECE .0334 vs .0386, classwise .0074 vs .0085). But a single scalar temperature on the deterministic
baseline beats the whole evidential mechanism on every proper score: ECE .0080 vs .0145, NLL .2611 vs
.2958, Brier .1276 vs .1369. The honest reading is that F-EDL's prior helps a little and post-hoc
calibration helps more.

**H3 — a calibrated CE baseline closes the Bayesian advantage to within 2 AUROC points. CONFIRMED,
with the direction flipping between controls.**
Near-OOD the Bayesian model is ahead by 0.8–1.0 points (.9271 vs .9177); far-OOD it is behind by
0.5 points (.8544 vs .8568). Both gaps are inside the 2-point band predicted before the run, and the
sign is not stable across controls. A single-sentence summary — "Bayesian models detect out-of-
distribution inputs better" — is not supported by this data; "the difference is small and depends on
how far out the distribution is" is.

**Withdrawn statement.** On 2026-10-10 I reported that "A1's validation NLL now beats the baseline, so
H3's direction is reversed". That was read from seed 20261006 alone. Across three seeds the baseline is
better on average (.2772 vs .2848) and seed 20261006 was the only exception. The claim is withdrawn and
does not appear in the report.

## 6. What the Bayesian machinery demonstrably adds

The one axis where the deterministic models cannot compete at all is error self-prediction: A1 scores
.9076 and A3 .9057 on epistemic error-detection AUROC, while B1 and B2 are pinned at .5000 — not
because they are worse, but because they have no epistemic quantity to submit. Any comparison on that
axis is therefore not "Bayesian versus deterministic", it is "having a posterior versus not having
one". This is the cleanest defensible claim the experiment supports, and it should be the spine of the
report's discussion.

## 7. Figures

18 print-ready PNGs at 300 dpi in
`D:\claude\workspace\CISC3024 AI Assignment 2\report_assets\`:

| Figure | Content | Status |
| --- | --- | --- |
| 02a | CIFAR-10 test images, 10 classes × 3 | verified by inspection |
| 02b | The two OOD control sets (SVHN, Tiny-ImageNet) | verified by inspection |
| 03 | Training loss and validation NLL for both backbones, all three seeds, early-stop star | verified |
| 04 ×3 | Reliability diagrams, five configurations side by side, one file per seed | verified |
| 05 ×3 | Risk-coverage curves with AURC, one file per seed | generated |
| 06 | OOD AUROC grouped bars with seed error bars, near and far | generated |
| 07 ×3 | Epistemic/aleatoric distributions for correct vs wrong and ID vs OOD | generated |
| 08 ×3 | Confusion matrix for F-EDL, one file per seed | generated |
| 09 | Anatomy of contrasting test predictions with real images, posterior spread and the uncertainty split | **verified by inspection after a fix** |
| 10 | Laplace log-evidence versus prior precision with the fitted optimum, all three seeds | generated |

Figure 1 (architecture and inference flow) and figure 11 (web interface) are deliberately not here yet:
figure 11 needs the frontend, and figure 1 should be drawn after the frontend so both use the same
diagram of the pipeline.

Two figure defects were found and fixed rather than shipped:

1. **Per-seed figures were overwriting each other.** Reliability, risk-coverage, uncertainty and
   confusion were written to a single filename in a three-iteration loop, so only the last seed
   survived. Filenames now carry the seed, and the seed is printed in each title so a single-seed
   panel can never be paired with the three-seed table by mistake.
2. **Figure 9 was unusable.** It selected test images by argmax over all 10,000 predictions, but only
   the exported gallery exists on disk, so every panel fell back to placeholder text. The gallery
   selection is now stratified — confidently correct, hesitantly correct, confidently wrong, wrong with
   high epistemic, maximum epistemic, maximum aleatoric, near-tie — and the figure draws from those.
   A side effect is informative: the six criteria collapse to four distinct images, because "maximum
   epistemic" and "hesitantly correct" land on the same sample.

Figure 9 also surfaced the sharpest single example in the whole experiment: a misclassified automobile
predicted as a truck with **confidence 1.000, epistemic 0.0000, total 0.0002**. The model is not merely
wrong there; it is certain. That is the over-confidence failure mode the calibration literature names,
captured on a real test image rather than asserted.

## 7b. A real bug found through the ablation, and the test blind spot behind it

The first curvature ablation produced an absurd row: with `hessian_structure="kron"` the model reported
NLL 1.5232, ECE 0.6488, mean epistemic 0.7877 and an error-detection AUROC of **0.224, i.e. below
chance**. Those numbers were rejected on sight rather than reported, and tracing them found a genuine
implementation defect.

**The bug.** The Laplace evidence combines two quantities that must be on the same scale: the
log-likelihood at the MAP point, which is a **sum** over training samples, and the curvature, which I
had built as a **mean** (`phi.T @ phi / n`). The posterior precision was therefore understated by a
factor of n = 47,500, the posterior was far too wide, and the predictive distribution smeared towards
uniform -- mean confidence fell to about 0.26 while accuracy stayed near 0.91, which is exactly what an
ECE of 0.65 means. The fitted prior precision confirmed the diagnosis: λ = 1.4 instead of the closed-form
535.

**Why the test suite missed it.** The library cross-check built *our* side with
`curvature="identity"`, which drops the curvature term entirely. It therefore compared our
curvature-free model against the library's Kronecker model and asserted only loose agreement. The code
path that was broken was never exercised against anything independent. The test has been rewritten to
use `"kron"` on both sides, and now validates the fixed implementation:

> Same synthetic problem, same prior precision, same features: our Kronecker last-layer Laplace and
> `laplace-torch` agree to a **maximum absolute difference of 0.031 in predictive probability**
> (argmax agreement 0.930, the residual being near-tie samples that legitimately flip).

A targeted regression test was added for the mechanism itself rather than for a symptom: the curvature
trace must grow roughly linearly with the sample count (measured 965 → 17,380 for n = 1,000 → 16,000, a
ratio of 18.0 against an expected 16), which is precisely what a mean-instead-of-sum would break. An
absolute threshold on the epistemic value was tried first and deliberately dropped -- on the synthetic
construction the value plateaus near 0.54 for reasons unrelated to the bug, and asserting a threshold
there would have failed for the wrong reason.

**One more thing that is not a finding.** The `identity` and `none` arms produced bit-identical numbers.
That is not an empirical result: in this implementation both set the curvature to nothing, so they are
the same code path. They are reported as one arm. There is a mathematical reason the two would coincide
even if written differently -- an isotropic identity curvature contributes `c·I` to the precision, which
is absorbed by the fitted scalar λ -- so a genuine three-way curvature ablation is `identity(=prior
only)` versus `Kronecker`, and the `none` row should be removed from the table rather than presented as
evidence of anything.

Test suite after the fix: **19 passed**.

## 7c. Curvature ablation — the ICLA design choice, tested

Same backbones, same weights, same features; only the curvature treatment in the last-layer Laplace
layer differs. Three seeds, mean ± standard deviation.

| Metric | `identity` (curvature dropped) | `kron` (Kronecker GGN kept) | B1 (no Laplace at all) |
| --- | --- | --- | --- |
| NLL ↓ | **0.2805 ± .0045** | 0.4245 ± .0066 | 0.2951 ± .0094 |
| Brier ↓ | **0.1312 ± .0002** | 0.1742 ± .0049 | 0.1337 ± .0010 |
| ECE ↓ | **0.0305 ± .0062** | 0.1757 ± .0034 | 0.0386 ± .0057 |
| AURC ↓ | **0.0123 ± .0004** | 0.0161 ± .0008 | 0.0121 ± .0003 |
| mean epistemic | 0.0088 ± .0003 | 0.2571 ± .0065 | — |
| near-OOD AUROC (epistemic) ↑ | **0.8827 ± .0194** | 0.7060 ± .0401 | — |
| far-OOD AUROC (epistemic) ↑ | **0.8085 ± .0111** | 0.6337 ± .0417 | — |
| error-detection AUROC ↑ | **0.9057 ± .0089** | 0.7492 ± .0364 | .5000 (no such quantity) |
| fitted prior precision λ | 535.5 / 538.2 / 519.2 | 27.3 / 27.6 / 25.1 | — |
| curvature trace | 0 (by construction) | 5.4e5 / 5.6e5 / 4.2e5 | — |

**Finding.** All eight metrics favour dropping the curvature, by margins far larger than the seed-to-seed
spread. This is an independent reproduction of the result that motivated the comparator: fitting the
curvature is not merely unnecessary, it is actively harmful. The fitted precisions explain the
mechanism — with the Kronecker curvature present, the trace of the precision is dominated by the data
term (~5e5) and the evidence pushes λ down to ~27, whereas dropping the curvature leaves a single
isotropic scale near 535. The wide, anisotropic posterior under `kron` inflates the predictive spread
(mean epistemic 0.257 versus 0.0088), which is what degrades every proper score.

**Two caveats that must travel with this table.**

1. **Both Laplace arms lose to the plain deterministic baseline on out-of-distribution detection.**
   B1 reaches .9119 near and .8463 far using a max-softmax score, against .8827 and .8085 for the best
   epistemic score. The defensible statement is therefore "within the Laplace family, dropping the
   curvature is better", **not** "Laplace improves out-of-distribution detection". Where the Laplace
   layer does beat the baseline is on the probabilistic scores themselves — NLL .2805 versus .2951 and
   ECE .0305 versus .0386 — plus the error-detection axis, on which the baseline has no quantity to
   submit at all.
2. **The `none` arm was removed rather than reported.** It produced bit-identical numbers to
   `identity` because both clear the curvature term, so the agreement is a property of the
   implementation, not an experiment. A genuine three-way curvature comparison is not available inside
   this code structure.

## 7d. Spectral-normalisation ablation — added 2026-10-11, after the frontend

The last outstanding M4 ablation finished while the web app was being built: both backbones retrained
with `spectral_norm=False` on every convolutional and linear layer, same data, same split, same
schedules, same three seeds, and the post-hoc layers refitted exactly as in the main run. Mean over
three seeds, cell format **SN on / SN off**.

| Metric | B1 | B2 | A1 | A2 | A3 |
| --- | --- | --- | --- | --- | --- |
| Top-1 ↑ | .9119 / **.9130** | .9119 / **.9130** | .9077 / **.9083** | .9072 / **.9076** | .9118 / **.9134** |
| NLL ↓ | .2951 / **.2856** | .2611 / **.2582** | .3054 / **.3015** | .2958 / **.2943** | .2805 / **.2719** |
| Brier ↓ | .1337 / **.1315** | .1276 / **.1264** | .1401 / **.1395** | .1369 / .1371 | .1312 / **.1290** |
| ECE ↓ | .0386 / **.0361** | .0080 / **.0050** | .0334 / **.0324** | .0145 / **.0136** | .0305 / **.0271** |
| AURC ↓ | .0121 / **.0120** | .0120 / **.0119** | .0138 / **.0134** | .0140 / **.0139** | .0123 / **.0121** |
| Error-detection AUROC ↑ | — / — | — / — | .9076 / .9034 | — / — | .9057 / .9029 |
| Near-OOD (SVHN) AUROC ↑ | .9119 / .9222 | .9177 / .9350 | .9258 / .9102 | .9277 / .9177 | .9096 / .9198 |
| Far-OOD (Tiny-ImageNet) AUROC ↑ | .8463 / .8472 | .8568 / .8579 | .8517 / .8498 | .8547 / .8507 | .8429 / .8441 |
| Mean epistemic | — | — | .00012 / .00012 | — | .00877 / .00899 |

**Finding.** Removing the per-layer Lipschitz budget improves every in-distribution score for every
configuration, by more than the seed-to-seed spread on NLL and ECE, and costs the two models that
actually use an epistemic term a little of it: error-detection AUROC falls by .004 (A1) and .003 (A3),
and A1's near-OOD ranking falls by .016. The baseline and the temperature-scaled baseline gain on
near-OOD instead, so the direction is not uniform across the OOD axis.

**What this does and does not change.** The headline comparison is unaffected: with the budget off, B1
still beats A1 on accuracy (.9130 against .9083) exactly as it does with it on (.9119 against .9077),
so H1's verdict is robust to the architecture-level regulariser. What the ablation does establish is
that on this dataset and at this scale, spectral normalisation is not doing the work it is sometimes
assumed to do — it is a mild tax on fit that buys a small amount of epistemic quality. The main table
keeps the budget **on** for all five configurations, which is what makes the comparison fair; the
"91.19 coincidence" noted in Stage 03a is now recorded as an artefact of that shared setting rather
than a property of the evidential head.

**Caveat.** This is one setting changed at a time on a single station. It does not license a claim
about spectral normalisation in general, only about this backbone, this dataset, and this schedule.

## 8. What is still missing before the backend can be called done

Stated plainly, because Gate 3b exists to catch problems early:

1. **M4 ablations partially done.** Completed: the curvature comparison, reported in Section 7c, which
   was the most important one because it is the comparator's own claim; and the spectral-normalisation
   comparison, added as Section 7d on 2026-10-11 after the frontend was built. Still outstanding, none
   of it blocking the frontend:
   * the predictive-sample-count sweep `S ∈ {1, 10, 100}` and its effect on runtime and AUROC —
     post-hoc only, roughly 15 minutes;
   * the three calibration families plotted on one axis — the data already exists, this is a figure
     rather than an experiment.
2. **M5 is partly done.** The frozen test evaluation happened as part of M3 (one pass per configuration
   per seed), so the headline table is final. The figures are drawn from it. What remains is the
   ablation table, not the main result.
3. **No cross-dataset check yet** (plan milestone M6, EuroSAT from the local Assignment 1 copy). It was
   explicitly optional and skippable; with 2 days left I recommend skipping it and spending the time on
   the frontend and the report.
4. **The `A2` epistemic column is structurally zero** because the calibration wrapper rescales
   probabilities and does not carry the Dirichlet variance through. Its .5000 error-detection figure is
   an artefact of our implementation choice, not a result, and is labelled as such everywhere.

## 9. Recommendation to the supervisor

Approve moving to Gate 4 (frontend) now, and run the M4 ablations in the background alongside it. The
main table is stable across three seeds with tight spread, the hypotheses have verdicts that survive
the error bars, and the interesting findings (the confident automobile, the near/far direction flip, the
error-detection asymmetry) are already captured in figures.

## 10. Artifact inventory for Stage 03b

| Absolute path | Content |
| --- | --- |
| `D:\claude\workspace\CISC3024 AI Assignment 2\runs\full_1791561662\matrix_results.json` | All per-seed and aggregate metrics, fitted λ, T, Dirichlet scaling parameters, timings, VRAM |
| `D:\claude\workspace\CISC3024 AI Assignment 2\runs\full_1791561662\predictives.npz` | Per-configuration test predictive means |
| `D:\claude\workspace\CISC3024 AI Assignment 2\runs\full_1791561662\*\best.pt` | Twelve checkpoints (2 models × 3 seeds, best and last) |
| `D:\claude\workspace\CISC3024 AI Assignment 2\analysis\test_predictions.npz` | Per-image mean/variance/epistemic/aleatoric/labels for every configuration, split and seed (28 MB) |
| `D:\claude\workspace\CISC3024 AI Assignment 2\analysis\summary.json` | Figure inventory plus the aggregate table |
| `D:\claude\workspace\CISC3024 AI Assignment 2\report_assets\` | 18 figures listed in Section 7 |
| `D:\claude\workspace\CISC3024 AI Assignment 2\webapp\data\` | `manifest.json`, `samples.json` and 48 gallery PNGs (~1 MB total) |
| `D:\claude\workspace\CISC3024 AI Assignment 2\src\make_figures.py` | Collection plus all figure code, with a `--reuse` cache path |
| `D:\claude\workspace\CISC3024 AI Assignment 2\src\export_webapp_data.py` | Frontend data bridge with the stratified gallery selection |
| `D:\claude\workspace\CISC3024 AI Assignment 2\runs\FULL_RUN.log`, `runs\FIGURES.log`, `runs\EXPORT_FIG.log` | Raw run logs |
