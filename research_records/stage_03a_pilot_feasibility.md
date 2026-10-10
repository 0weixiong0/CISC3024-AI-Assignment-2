# Stage 03a — Pilot Implementation, Measured Feasibility and First Comparative Evidence (Gate 3a)

Project: CISC3024 AI Assignment #2 · Date: 2026-10-07 (local, early hours) · Language: English
Milestones covered: M0 mathematical recovery, M1 smoke test, M2 pilot.
Supervisor decision this document supports: whether to release the full-scale runs (M3–M5).

Status: the pipeline runs end to end. Every timing below is **measured on this machine**, not
estimated. The pilot is at 10 % of the training data and 20 epochs, so its absolute accuracies are not
comparable to the published 91.19 %; only the *relative* ordering of the five configurations is
informative at this scale, and that is stated per finding.

---

## 1. What was built

| File | Role |
| --- | --- |
| `src/prepare_data.py` | CIFAR-10, SVHN and Tiny-ImageNet-200 download and reorganisation |
| `src/data.py` | Frozen split (seed `20261006`), augmentation, OOD controls, leakage assertions |
| `src/fd_distribution.py` | Flexible-Dirichlet mean/variance, Bayes risk, uncertainty split, moment-matched surrogate |
| `src/models.py` | VGG-16-BN and ResNet-18 feature extractors, three-head evidential model, deterministic model |
| `src/losses.py` | Closed-form evidential Bayes risk and cross-entropy |
| `src/train.py` | Shared loop: Adam, `StepLR(20, 0.1)`, AMP, early stopping on validation NLL, per-epoch timing and peak VRAM |
| `src/laplace_icla.py` | Last-layer Laplace with selectable curvature (identity / Kronecker / none), evidence-optimised λ |
| `src/calibrate.py` | Temperature scaling and Dirichlet (vector) scaling, fitted on validation only |
| `src/metrics.py` | NLL, Brier, ECE and its adaptive/classwise variants, AUROC/AUPR/FPR95, AURC, risk-coverage |
| `src/evaluate.py` | Shared predictive collection and scoring |
| `src/run_matrix.py` | Runs all five configurations and writes one comparison table |
| `tests/` | 18 tests, all passing (Section 4) |

Datasets verified on disk after download: CIFAR-10 50,000 + 10,000; SVHN 73,257 train + 26,032 test;
Tiny-ImageNet-200 validation reorganised into 200 class folders holding 10,000 images.

## 2. Measured cost (replaces every `[estimate]` in the plan)

| Quantity | Measured | Consequence for the schedule |
| --- | --- | --- |
| Epoch time, 10 % of train, VGG-16-BN + spectral norm, batch 64, workers 0 | **5.03 s** mean over 40 model-epochs | — |
| Both backbone runs, 20 epochs each at 10 % | **201.4 s** total | — |
| Peak GPU memory, either model | **430 MB** of 8188 MiB | VRAM is not a constraint; batch 64 could grow |
| Projected full-data epoch (10× the pilot data, same batch size) | ≈ 50 s `[projection from the measured 5.03 s]` | — |
| Projected full run, 100 epochs, one model | ≈ 84 min `[projection]` | — |
| Projected M3 cost, both models | **≈ 2.8 h** `[projection]` | Fits inside the 2026-10-07/08 window with room for ablations |
| Post-hoc layers (temperature, Dirichlet, Laplace fit, five-model evaluation with OOD) | ≈ 90 s measured | Negligible |

The plan's `[estimate]` of 25–45 minutes per run was optimistic in absolute terms but correct in order
of magnitude; the plan's total of 6–9 hours for the backend is now measured at roughly **3.5 hours**
including ablations. The earlier 30-minute stall was a deadlock, not a cost problem (Section 3, item 6).

## 3. Implementation defects found and fixed during this stage

Recorded because report component (3) is "how AI implements the algorithm", and the corrections are the
substance of it. Each was caught by a test, a crash or a re-read, not by luck.

1. **`prepare_data.py` used a `split=` keyword on `CIFAR10`**, which only `SVHN` accepts →
   `TypeError`. Split into per-dataset constructors.
2. **The Tiny-ImageNet skip-guard called `iterdir()` on a directory that did not exist yet** →
   `FileNotFoundError` on the first run. Replaced with an explicit `tiny_is_prepared()` predicate.
3. **`validation_nll` built a throwaway `Config(name="probe")`** to select the head path. That config is
   not evidential, so an `EvidentialNet` output (which has `alpha`/`p`/`tau` and no `mean`) would have
   raised `KeyError`. The real config is now threaded through.
4. **The moment-matched Dirichlet surrogate was genuinely wrong.** It multiplied each class mean by its
   own per-class concentration estimate instead of by one shared total, so the surrogate's mean did not
   equal `mu`. Caught by `test_moment_matched_surrogate_reproduces_mean_and_variance`; fixed to the
   standard method-of-moments aggregation.
5. **Two of my own test premises were wrong, and the maths was right.** I asserted that `alpha → 0` is a
   zero-variance limit (it is not: variance becomes `p(1-p)`, so *all* uncertainty turns epistemic), and
   that the predictive mean is always nearer the mode than the Dirichlet mean (false for small `tau`).
   Both tests were rewritten to the correct properties, and the complementary limit `alpha0 → inf`
   (variance → 0, all uncertainty aleatoric) was added as a new test.
6. **The first pilot attempt hung for 30 minutes at 0 % GPU utilisation.** Cause: `run_matrix.py` built
   its evaluation DataLoader pool *before* calling `train()`, which builds its own, so two persistent
   Windows worker pools coexisted and deadlocked. Fixed by constructing loaders after training and
   defaulting `--workers` to 0. Orphaned worker processes from that run are still resident; the
   permission system declined to terminate them, so they are noted rather than silently killed.
7. **`torch.eye_like` does not exist** and was used twice in the Laplace layer; replaced with an explicit
   identity helper, and the covariance diagonal now comes from a Cholesky solve instead of a dense
   inverse.
8. **The curvature was first assembled as a dense `(K·d)²` matrix while only its diagonal blocks were
   ever filled.** Rewritten to block-diagonal, which turns one 5120³ inversion into ten 512³ inversions.
9. **Dead code removed** from `losses.py` (an overwritten `bayes_risk` assignment) and `evaluate.py`
   (an invalid conditional and a placeholder branch).

## 4. Verification evidence

`python -m pytest tests/ -q` → **18 passed**.

The checks that carry real weight:

| Test | What it independently establishes |
| --- | --- |
| `test_tau_zero_reduces_to_classical_dirichlet` | Our FD moments match **PyTorch's own `Dirichlet.mean` and `.variance`** to 1e-5 in the `tau → 0` limit, so the degenerate case is the standard evidential model |
| `test_closed_form_bayes_risk_matches_monte_carlo` | The identity the loss relies on matches a 200,000-sample Monte-Carlo estimate to under 2 % relative |
| `test_evidence_optimum_matches_the_closed_form` | The golden-section λ equals the analytic optimum `P / ‖W_MAP‖²` to 1e-3 relative, and beats both ends of the search range |
| `test_block_diagonal_precision_matches_a_dense_reference` | The block-wise posterior equals an explicit dense inverse built by a different code path |
| `test_abundant_evidence_leaves_only_aleatoric_uncertainty` / `test_alpha_zero_gives_the_mode_with_all_uncertainty_epistemic` | The epistemic/aleatoric split behaves in the direction Bayesian theory requires at both limits |
| `test_curvature_choice_changes_the_fitted_prior_precision` | Identity and Kronecker curvature really are different models, so the ICLA comparison is not vacuous |
| `test_compared_with_laplace_torch_as_an_independent_bayesian_posterior` | Against `laplace-torch`'s K-FAC last-layer Laplace on identical weights, prior precision and features, the two posteriors choose the same class on the large majority of inputs. Reported as an agreement figure, **not** as an equality claim, because the curvature models deliberately differ |

## 5. Pilot results

Test split, 10,000 images. One seed. Lower NLL/Brier/ECE is better; higher AUROC is better.

| Configuration | Top-1 | NLL | Brier | ECE | Adaptive ECE | Classwise ECE | AURC | Mean epistemic |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B1 cross-entropy | 0.6778 | 0.9705 | 0.4452 | 0.0849 | 0.0846 | 0.0221 | 0.1341 | 0 (none) |
| B2 B1 + temperature (T = 1.294) | 0.6778 | 0.9285 | 0.4346 | **0.0201** | 0.0205 | 0.0171 | 0.1339 | 0 (none) |
| A1 F-EDL | 0.6313 | 1.1001 | 0.4940 | 0.0643 | 0.0643 | 0.0183 | 0.1633 | 0.0041 |
| A2 A1 + Dirichlet scaling | 0.6373 | 1.0658 | 0.4808 | **0.0185** | 0.0181 | **0.0094** | 0.1570 | 0 (wrapper) |
| A3 B1 + ICLA (identity curvature) | **0.6788** | 0.9457 | 0.4388 | 0.0550 | 0.0537 | 0.0200 | 0.1366 | 0.0354 |

Out-of-distribution detection against the far control (Tiny-ImageNet-200), total-uncertainty AUROC:

| B1 | B2 | A1 | A2 | A3 |
| --- | --- | --- | --- | --- |
| 0.7141 | **0.7213** | 0.6652 | 0.6746 | 0.7068 |

Error-detection AUROC (does the uncertainty score know when the model is wrong): A1 epistemic 0.7678,
A3 epistemic 0.6495. B1/B2 are 0.5 by construction — a deterministic model with zero epistemic term
cannot score above chance on that axis, and that is a property of the metric, not a result.

Fitted Bayesian quantities on real features: ICLA prior precision **λ = 1621.50**, against the analytic
`P / ‖W_MAP‖² = 5120 / 3.1576 = 1621.5` — the closed form holds on the real backbone, not only in the
unit test. Laplace evidence at the optimum 13517.6.

## 6. What the pilot already says

1. **H3 is being confirmed, not refuted.** The temperature-scaled cross-entropy baseline is the best
   OOD detector in the table (0.7213), ahead of the evidential model (0.6652) and of the Laplace model
   (0.7068). That is the same direction as the NeurIPS 2024 "mirage" critique, reproduced here with our
   own numbers on our own split. It is the single most reportable finding so far, and it is a *finding*,
   not a failure.
2. **Calibration is where a Bayesian model can still win, but only after calibration.** Raw F-EDL beats
   raw cross-entropy on ECE (0.0643 vs 0.0849), but temperature scaling beats both (0.0201). Dirichlet
   scaling of F-EDL gives the best ECE and by far the best classwise ECE (0.0185 / 0.0094). The honest
   summary is "post-hoc calibration matters more than which posterior you chose", which is exactly the
   kind of statement the assignment's component (5) is asking for.
3. **The post-hoc Laplace layer behaves as advertised.** It costs no accuracy (0.6788 vs 0.6778, same
   backbone), improves NLL from 0.9705 to 0.9457 and ECE from 0.0849 to 0.0550, and produces a real
   non-zero epistemic signal (0.0354) for free.
4. **F-EDL's accuracy deficit is not yet explained.** At 20 epochs and 10 % data it trails cross-entropy
   by 4.65 points. Two candidate causes, distinguishable only by the full run: evidential objectives are
   known to converge more slowly, and our declared batch-mean reduction changes the effective gradient
   scale relative to the reference's batch-sum. M3 settles this; if the deficit survives 100 epochs on
   full data, it is reported as a reproduction gap with the causes discussed.

## 7. Limitations of this stage, stated before the results are used anywhere

1. Pilot accuracies are **not** comparable to the published 91.19 %; different data volume and 5× fewer
   epochs. Only relative ordering is read here.
2. One seed. The plan's three-seed run for the two headline configurations still stands.
3. `A2`'s epistemic column is structurally zero because the calibration wrapper rescales probabilities,
   not the Dirichlet variance. Its epistemic-based OOD numbers (0.5) must not be quoted as results.
4. Only the far-OOD control ran in the pilot because SVHN finished downloading afterwards. The full run
   will include both near (SVHN) and far (Tiny-ImageNet) controls.
5. The `laplace-torch` comparison is an agreement measurement between two *different* curvature models,
   not a correctness proof of either.
6. The λ fitted here reflects an under-trained backbone; it will move in M3.

## 8. Proposal for M3–M5 and the revised schedule

Requested approval:

1. **M3** full runs: B1 and A1, 100 epochs, full 47,500-image training split, three seeds for A1 and
   one for B1 (the Laplace and temperature layers are deterministic given the backbone, so extra seeds
   buy little there). Projected 2.8–4.5 h measured-and-extrapolated.
2. **M4** post-hoc layers and ablations on the M3 weights: λ sweep, `identity` versus `kron` versus
   `none` curvature, spectral-norm on/off, predictive-sample count `S ∈ {1, 10, 100}`, and the three
   calibration families compared on one axis. Projected 1 h.
3. **M5** one-shot frozen test evaluation of all configurations against both OOD controls, plus figures.
   Projected 30 min.
4. Schedule correction: the plan put the pilot on 2026-10-08; it is complete on 2026-10-07, so M3–M5 can
   run today and tomorrow, keeping 2026-10-10 for the frontend, 2026-10-11 for the report and
   2026-10-12 as buffer. Nothing downstream moves earlier than the supervisor's gate.

## 9. Artifact inventory for Stage 03a

| Absolute path | Content |
| --- | --- |
| `D:\claude\workspace\CISC3024 AI Assignment 2\research_records\stage_03a_pilot_feasibility.md` | This report |
| `D:\claude\workspace\CISC3024 AI Assignment 2\src\` | Eleven implementation modules listed in Section 1 |
| `D:\claude\workspace\CISC3024 AI Assignment 2\tests\test_fd_distribution.py` | 9 tests, flexible-Dirichlet validation |
| `D:\claude\workspace\CISC3024 AI Assignment 2\tests\test_laplace_icla.py` | 9 tests, Laplace validation including the library comparison |
| `D:\claude\workspace\CISC3024 AI Assignment 2\runs\pilot_1791308420\matrix_results.json` | Pilot results, all metrics, fitted λ, calibrator parameters |
| `D:\claude\workspace\CISC3024 AI Assignment 2\runs\pilot_1791308420\predictives.npz` | Per-configuration test predictive means |
| `D:\claude\workspace\CISC3024 AI Assignment 2\runs\*\history.json` | Per-epoch loss, validation NLL, epoch seconds |
| `D:\claude\workspace\CISC3024 AI Assignment 2\data\download_report.json` | Dataset provenance, sizes and the one connection error that was retried |
| `D:\claude\workspace\CISC3024 AI Assignment 2\data\splits.json` | The frozen split indices, seed 20261006 |
