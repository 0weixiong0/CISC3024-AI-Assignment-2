# Stage 02 — Algorithm Description, Experimental Design and Implementation Plan (Gate 2)

Project: CISC3024 AI Assignment #2 · Date: 2026-10-06 · Language: English (working-stage report)
Gate 1 outcome: the supervisor approved **Option A** — F-EDL as the algorithm under study, ICLA as the
in-experiment Bayesian comparator, cross-entropy (with and without temperature scaling) as the
baseline. Approval recorded in `conversation_log.md` as U003.

Status: this document is the Gate 2 artifact. It authorizes nothing by itself. Approving it authorises
exactly the actions listed in Section 11 and nothing beyond them.

> **SUPERSEDED IN PART — see `stage_02b_math_recovery.md` (M0, 2026-10-07).** Milestone M0 recovered the
> mathematics and hyperparameters that this document had marked "not yet recovered", and in doing so
> falsified four statements below. Where this document and the M0 record disagree, **M0 wins**. The
> superseded items are: (1) §1 hypothesis **H2 is withdrawn** — the published 99.10 figure is an
> error-detection AUROC, not a mean confidence, so the predicted over-confidence gap was based on a
> misread column; it is replaced by H2′ (measure ECE, which the paper does not report). (2) §2.1 the
> objective is **not** a sampled expectation — it is the closed-form Bayes risk
> `‖y − mu‖² + Σ var` with analytic moments, so no Monte-Carlo estimator exists. (3) §2.3 the CIFAR-10
> backbone in the reference is **VGG-16-BN with spectral normalisation**, not ResNet-18; ResNet-18 is
> their CIFAR-100 backbone and is retained here only as a declared fallback. (4) §5 the validation split
> becomes **2,500 images (0.05 of train)** to match the reference default, with the test split unchanged.
> §4, §6, §7, §9 and §10 are unaffected in scope.

---

## 1. Research question and hypotheses

**Question.** For image classification, does a recent Bayesian classification head deliver
uncertainty estimates that are measurably better than those obtained from an ordinary
cross-entropy network plus post-hoc calibration, and does a post-hoc Bayesian treatment of the
last layer deliver better uncertainty than a trained-in Bayesian head at lower total cost?

Three falsifiable hypotheses, stated before any experiment is run:

- **H1.** F-EDL reaches CIFAR-10 top-1 accuracy within ±1.5 points of its published 91.19 % (verified
  from its Table 1) when we reimplement it from the paper on our own frozen split.
- **H2.** Uncalibrated, F-EDL shows a confidence–accuracy gap larger than the cross-entropy baseline
  (the published table pairs 91.19 % accuracy with 99.10 % confidence, both verified). If H2 fails, the
  "mirage" critique loses one of its concrete targets in our setting, which is equally reportable.
- **H3.** After post-hoc calibration — temperature scaling for the deterministic baseline, Dirichlet
  scaling for the evidential model, and evidence-optimised prior precision for ICLA — the gap in
  OOD AUROC and in negative log-likelihood between the Bayesian models and the calibrated
  cross-entropy baseline narrows to within 2 AUROC points. H3 reproducing would confirm the field's
  critique with our own measurements rather than by citation.

## 2. Algorithm 1 — F-EDL (the recent algorithm under study)

Paper: *Uncertainty Estimation by Flexible Evidential Deep Learning*, NeurIPS 2025, arXiv:2510.18322
(venue verified via the arXiv Comments field, the repository description and the README).

### 2.1 Bayesian structure

| Element | Specification | Evidence status |
| --- | --- | --- |
| Random variable | The class-probability vector `π(x)` itself, not the weights | Verified |
| Prior family | Flexible Dirichlet `FD`, replacing the standard Dirichlet that classical EDL ties to evidence by `α = e + 1` | Verified (the classical `α = e + 1` form is the reference point) |
| Parameterisation | Three independent heads on the backbone feature `z`: `α_k = exp(g_φ1(z))`, `p = softmax(g_φ2(z))`, `τ = softplus(g_φ3(z))` | **Verified verbatim** from the HTML method section |
| Likelihood | Categorical over classes given a sampled `π` | Implied by the objective; exact density form **not yet recovered** |
| Objective | `L = E_{π~FD} ‖y − π‖² + ‖y − p‖²` — expected squared error under sampled probabilities, plus a mode-pinning term | **Verified as extracted**; the exact target encoding, any loss weight, and the Monte-Carlo estimator for the expectation are **not recovered** |
| Evidence / marginal likelihood | Not used — hyperparameters are selected on the validation split | Our design decision, declared |
| Inference cost | One forward pass for the mode `p`; `S` extra passes only when predictive samples are needed | Design decision |

### 2.2 What must be recovered before implementation (milestone M0)

The paper's Appendix E (epochs, optimiser, learning rate, and the parts of the `FD` definition we could
not retrieve) was unreachable through every route tried in Stage 01. M0 exists to close this. Its
success criteria and its fallback are both fixed in advance:

1. Re-attempt the arXiv HTML for a v2 and the NeurIPS proceedings page for the appendix.
2. If still unavailable, **we define it ourselves and declare it**: a flexible Dirichlet whose density
   is `p(π) ∝ Π_k π_k^{α_k−1}` modulated by the precision `τ` around the mode `p`, sampled by the
   standard Gamma-ratio construction on `α·τ`. The declared adaptation would be recorded in the report
   with the exact reason (source unavailable), and the degenerate case `τ → const`, `p` tied to `α` must
   reproduce classical EDL numerically. That consistency test is the acceptance criterion, not a
   narrative claim.
3. The third option — reading the unlicensed official repository to extract the missing mathematical
   definition — is **not** taken without the supervisor's explicit decision D-1 in Section 12.

### 2.3 Backbone (our choice, declared)

ResNet-18 in the CIFAR variant: 3×3 stride-1 stem, no max-pooling, global average pooling to a 512-d
feature `z`, then the three linear heads. The published headline setting is ResNet-18 on CIFAR-10
(verified), so the comparison stays fair; the exact stem construction is ours because the paper does not
state it in the text we could read.

## 3. Algorithm 2 — ICLA (the post-hoc Bayesian comparator)

Paper: *Identity Curvature Laplace Approximation for Improved Out-of-Distribution Detection*, WACV 2025,
arXiv:2312.10464v2 (venue verified in the CVF proceedings listing).

| Element | Specification | Evidence status |
| --- | --- | --- |
| Random variable | Only the last-layer weight matrix `W ∈ R^{512×10}`; all earlier layers stay point estimates | Verified |
| Prior | Gaussian `N(W; W_MAP, λ^{-1} I)` | Verified (identity prior precision, `λ` optimised) |
| Likelihood | Softmax categorical over the last-layer logits of frozen features `φ(x)` | Standard construction |
| Curvature | **Identity matrix replaces the Hessian/GGN entirely** — this is the paper's whole method change | Verified verbatim: "using identity curvature and optimizing prior precision" |
| Posterior | Gaussian centred at the MAP weights; with identity curvature the precision reduces to a scalar multiple of the identity | Derived from the verified method statement; the exact placement of `λ` relative to the data term is an implementation detail to pin against the paper's equations |
| Evidence | Marginal likelihood, maximised over `λ` — the only hyperparameter | Verified |
| Predictive | Draw `S` weight samples, average the softmax outputs; spread across samples = epistemic uncertainty | Verified as the standard Laplace predictive construction |
| Published cost | ResNet-18 + SGD, 100 epochs on CIFAR-10/100, 90 on ImageNet-200, OpenOOD v1.5 protocol | **Verified verbatim** |
| Published result | CIFAR-10 far-OOD AUROC 92.50 ± 0.38 (ICLA) vs 91.54 ± 0.36 (GGN curvature); calibration "similar across methods" | **Verified** |

Crucially, ICLA attaches to **the already-trained cross-entropy backbone from Section 4**, so it costs
no additional backbone training. Its point predictions are the baseline's; only its uncertainty differs.

## 4. Baselines (the models the Bayesian claims must be measured against)

| Config | Backbone training | Head | Purpose |
| --- | --- | --- | --- |
| B1 CE | 100 epochs, SGD + momentum, cosine schedule | Plain linear, cross-entropy | Reference point; also the host for ICLA |
| B2 CE + temperature | reuse B1 weights | one scalar `T` fitted on validation | The cheap deterministic calibration the "mirage" critique points to |
| A1 F-EDL | 100 epochs, same optimiser family | three-head flexible Dirichlet | The recent algorithm (H1, H2) |
| A2 F-EDL + Dirichlet scaling | reuse A1 weights | vector/matrix scaling on validation | Fair calibration treatment for the Bayesian head (H3) |
| A3 B1 + ICLA | reuse B1 weights | last-layer Gaussian posterior, `λ` by evidence | The post-hoc Bayesian comparator (H3) |

Five configurations, but only **two** backbone training runs (B1 and A1). Everything else is post-hoc
fitting on stored weights and cached features.

## 5. Data

| Item | Decision | Notes |
| --- | --- | --- |
| Primary dataset | CIFAR-10 | Verified headline setting of both candidate papers, so our numbers are directly comparable to published ones |
| Size / resolution / classes | 60,000 images (50,000 train + 10,000 test), 32×32 RGB, 10 classes | Official page |
| Download size | 163 MB (python archive) | Measured by a Stage 01 subagent from the official page |
| License | **Not verified** — the official page does not state a licence | Recorded honestly; research use only, no redistribution of images in the repository |
| Frozen split | 45,000 train / 5,000 validation / 10,000 test, seed `20261006`, indices written once to `data/splits.json` and never regenerated | Validation is carved from the official train set only; the test set is never touched before milestone M5 |
| Augmentation (train) | `RandomCrop(32, padding=4)` + `RandomHorizontalFlip`, then `ToTensor` + channel normalisation with CIFAR-10 mean/std | Standard; the validation/test pipeline has no augmentation |
| Near-OOD control | SVHN test split | Present in the official F-EDL code as an OOD set (verified) |
| Far-OOD control | Tiny-ImageNet-200 validation split | Present in the official F-EDL code as an OOD set (verified); 248.1 MB zip size measured by a subagent |
| Cross-dataset check (optional, M6) | The locally cached EuroSAT copy from Assignment 1 (234 MB, already on disk, zero download) | Only if the schedule allows; it is a different domain and has no published EDL numbers to compare against, so it is a robustness check, not a headline result |

Total new download if the plan is approved: **about 411 MB** (CIFAR-10 163 MB + SVHN ~182 MB +
Tiny-ImageNet-200 248 MB — the last is the dominant term, so if the supervisor wants a smaller
footprint, Tiny-ImageNet can be replaced by notMNIST or by dropping the far-OOD control; free space on
D: is ~53 GiB, so disk is not the constraint, download time is).

## 6. Metrics and evaluation protocol

Every configuration is scored with the same harness. The metric battery is the standard one in this
literature and is already implemented in `torch-uncertainty` (verified list): accuracy and macro-F1;
categorical negative log-likelihood; Brier score; ECE plus its Smooth, Adaptive and Classwise
variants; OOD AUROC, AUPR and FPR95; and selective-prediction AURC, AUGRC, coverage at 5 % risk and
risk at 80 % coverage.

Protocol rules fixed now, so they cannot be bent later:

1. Model selection uses validation NLL, not test accuracy.
2. Calibration parameters (`T`, Dirichlet scaling, `λ`) are fitted on the 5,000-image validation split
   only.
3. The test split is evaluated exactly once per configuration, at milestone M5, after all choices are
   frozen.
4. Every reported number carries its seed. Three seeds are run for the two headline configurations
   (A1, A3) and one seed for the rest; the pilot at M2 decides whether three seeds fit the budget.
5. Figures produced: reliability diagram per configuration, risk-coverage curve, OOD AUROC bar chart,
   training curves (loss and validation NLL), and a confusion matrix for the best model.

## 7. Overfitting control and regularisation plan

The outline asked specifically for techniques that prevent overfitting. For a Bayesian classifier the
honest list is different from the deep-network list, and it splits by mechanism:

| Mechanism | Where it applies | What we do |
| --- | --- | --- |
| Data augmentation | All configurations | Crop + flip (Section 5) |
| Weight decay | B1, A1 | `5e-4`, cosine schedule |
| Early stopping | A1 (and B1 for fairness) | Stop on validation NLL with patience 15 |
| Prior as regulariser | A3 / ICLA | `λ` selected by marginal likelihood — this is the textbook Bayesian anti-overfitting device, and it costs one 1-D search |
| Loss-embedded regulariser | A1 / F-EDL | The mode-pinning term `‖y − p‖²` is the paper's own regulariser. F-EDL's stated weakness is reliance on external regularisation (spectral norm in its README command list, verified), so we run one ablation with and without it |
| Label smoothing | Tested and reported either way | Applied to B1/B2 only; deliberately **not** applied to A1, because smoothing changes the target encoding the evidential loss is written against |
| Shrinkage / low-rank covariance | Not used | Would only apply if the closed-form family (C4) had been chosen |

Ablations at M4: `λ` sweep for ICLA; with/without spectral-norm regularisation for F-EDL;
number of predictive samples `S ∈ {1, 10, 100}` and its effect on both runtime and AUROC; temperature
versus Dirichlet versus Laplace calibration compared on the same axis.

## 8. Implementation architecture

All code is AI-generated; the supervisor writes none. English identifiers and comments throughout.

```
src/
  data.py            split freezing, augmentation, OOD loaders, licence notes
  models.py          CIFAR ResNet-18 backbone + head modules (linear, FD three-head)
  fd_distribution.py flexible-Dirichlet parameterisation, sampling, degenerate-case test
  losses.py          F-EDL objective, CE objective
  train.py           shared loop, AMP, checkpointing, per-config CLI
  laplace_icla.py    our own last-layer identity-curvature Laplace + evidence search for lambda
  calibrate.py       temperature and Dirichlet scaling
  evaluate.py        metric battery, reliability/risk-coverage figures
  export_artifacts.py  JSON results + figures for the report and the webapp
tests/
  test_fd_distribution.py   degenerate case must match classical Dirichlet/EDL numerically
  test_laplace_icla.py      our implementation must match laplace-torch on the same weights
  test_splits.py            frozen split is reproducible and test indices never leak into train
```

Two independent cross-checks are built in, and both are reported numerically rather than asserted:
our `fd_distribution` against `torch-uncertainty`'s classical EDL in the degenerate limit, and our own
`laplace_icla` against `laplace-torch` on identical weights and features. This mirrors the
analytic-Grad-CAM-versus-library check that worked in Assignment 1, and it is what makes component (3)
of the report auditable.

## 9. Milestones, time estimates and stop-loss criteria

`[estimate]` marks a number that has not been measured on this machine; milestone M2 replaces all of
them with measurements.

| # | Milestone | Deliverable | `[estimate]` | Stop-loss rule |
| --- | --- | --- | --- | --- |
| M0 | Recover the missing `FD` definition and Appendix E | `stage_02b_math_recovery.md` | 1–2 h | If unrecovered, adopt the declared adaptation in §2.2(2) — do not stall |
| M1 | Smoke test: 1 epoch on 1 % subset, all five configs | Console log + shape/gradient assertions | 20 min | Any config that cannot run 1 epoch stops the line |
| M2 | **MVP**: 10 % subset, 20 epochs, full metric harness | **Gate 3a feasibility report** with measured wall-clock, peak VRAM, sanity metrics | 1–1.5 h | If measured full-run projection exceeds 6 h, cut to one seed and drop the far-OOD control |
| M3 | Full runs B1 and A1 (100 epochs each) | Checkpoints + training curves | 1.5–2.5 h | If A1 cannot beat 85 % validation accuracy, the head or the objective is wrong — debug before continuing |
| M4 | Post-hoc layer: B2, A2, A3 + ablations | Comparison tables | 1 h | If our Laplace disagrees with `laplace-torch`, trust the discrepancy and investigate — it is report content either way |
| M5 | Frozen test evaluation, one shot per config | Final results table + figures | 30 min | No re-running after the test split is opened, except for a documented bug |
| M6 | Optional cross-dataset check on the cached EuroSAT copy | Robustness paragraph | 1–2 h | Skippable without touching the deadline |

Backend total: **6–9 hours of compute `[estimate]`**, scheduled across 2026-10-08 (M0–M2, Gate 3a) and
2026-10-09 (M3–M5, Gate 3b), leaving 2026-10-10 for the frontend and 2026-10-11 for the report.

## 10. Expected results (predictions written before the experiment)

| Configuration | Top-1 acc. | Expected ECE | Expected far-OOD AUROC | Basis |
| --- | --- | --- | --- | --- |
| B1 CE | 94–95 % | 0.04–0.10 | 85–92 % | `[estimate]` — ordinary ResNet-18/CIFAR-10 behaviour; not a paper number |
| B2 CE + temperature | same as B1 | 0.01–0.03 | 86–93 % | `[estimate]` — temperature scaling does not change accuracy |
| A1 F-EDL | 90–92 % | 0.05–0.20 | 88–93 % | Accuracy band anchored on the **verified** 91.19 ± 0.2; ECE band is our H2 prediction |
| A2 A1 + Dirichlet scaling | same as A1 | 0.02–0.05 | 89–94 % | `[estimate]` |
| A3 B1 + ICLA | same as B1 | 0.02–0.06 | 90–93 % | Anchored on the **verified** 92.50 ± 0.38 far-OOD AUROC, though their protocol differs from ours |

These are predictions, not targets. If F-EDL underperforms the band, or if the calibrated CE baseline
matches the Bayesian models, that is a reported finding under H3 and not a failure of the assignment.

## 11. Exactly what approval of this plan authorises

1. `pip install torch-uncertainty==0.13.0` (Apache-2.0) and `pip install laplace-torch==0.2.3` (MIT).
   Both confirmed available on PyPI by a metadata-only `pip index versions` probe on 2026-10-06; neither
   is installed yet. Python requirement `>=3.10,<3.14` matches the local 3.12.10.
2. Downloading CIFAR-10 (163 MB), SVHN (~182 MB) and Tiny-ImageNet-200 (248 MB) into
   `D:\claude\workspace\CISC3024 AI Assignment 2\data\`.
3. Writing the code skeleton in Section 8 and running milestones M0–M2, ending in the Gate 3a
   feasibility report.
4. Nothing else. Not the full runs (M3–M5), not the frontend, not creating the public repository, not
   publishing anything.

Already present locally and therefore not requested: numpy 2.4.6, scipy 1.17.1, scikit-learn 1.9.0,
matplotlib 3.11.0, pandas 3.0.3, torchvision 0.22.1+cu128, Pillow, tqdm, onnx 1.22.0 and
onnxruntime 1.26.0 (the last two are for the Gate 4 browser deployment).

## 12. Decision D-1 — the one integrity question that needs the supervisor

The official F-EDL repository declares **no license** (verified through the GitHub API). GitHub's terms
mean that no reuse permission is granted, and this report goes through Turnitin.

Two safe readings exist and I need the supervisor to pick one:

- **D-1(a) — paper-only.** I never open the repository source. If the appendix cannot be recovered, I
  adopt my own declared definition from §2.2(2) and state in the report that the implementation follows
  the paper text plus a declared adaptation.
- **D-1(b) — read for mathematics, rewrite independently.** I read the reference code only to establish
  the missing mathematical definition (equations and hyperparameter values are facts, not expression),
  record exactly which file answered which question, write our implementation from scratch, and
  disclose the consultation in the report's Acknowledgements.

My recommendation is **D-1(b) with the narrow scope written down**, because the alternative risks
silently mis-implementing the paper's own method — which is worse for the report than a disclosed
consultation. But this is the supervisor's call, not mine.

## 13. Risks and mitigations

| Risk | Likelihood | Mitigation |
| --- | --- | --- |
| The `FD` definition cannot be recovered, so A1 is our method rather than the paper's | Medium | §2.2 fixes the fallback and the disclosure wording in advance; H1 is then restated honestly as a variant |
| Measured runtime is far above the `[estimate]` | Medium | M2 exists precisely to measure; stop-loss in §9 caps the full run at 6 h or cuts seeds |
| F-EDL loses to calibrated CE on every uncertainty metric | Real, and the critique paper says it will happen | This is H3's predicted outcome and the strongest possible component (5) material; it is not a failure state |
| `torch-uncertainty` has pre-1.0 API churn (its own README warns "expect changes") | Low–Medium | Pin 0.13.0; if an API breaks, our own metric implementations stay authoritative and the library is only a cross-check |
| Dataset licence ambiguity for CIFAR-10 | Low | Images are not redistributed in the public repository; only split indices and code are committed |
| Scope creep from the optional EuroSAT cross-check | Medium | M6 is explicitly skippable and is not on the critical path |
