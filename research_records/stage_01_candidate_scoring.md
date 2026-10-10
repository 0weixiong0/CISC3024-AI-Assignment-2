# Stage 01 — Candidate Scoring and Gate 1 Recommendation

Project: CISC3024 AI Assignment #2 · Date: 2026-10-06 · Language: English (working-stage report)
Purpose: supervisor decision artifact. Approving one row below authorizes the design stage only; it
does not authorize downloads, training runs, or publication.

---

## 1. Scoring rule

Five dimensions, each 1–5, where **5 is best for this assignment**. Every score is followed by the
evidence line that produced it, and every number that was not read from a source is marked
`[estimate]`. Hardware scores assume the machine measured on 2026-10-05 (RTX 4060 Laptop 8188 MiB,
31.6 GiB RAM, torch 2.7.1+cu128). No candidate has been benchmarked locally yet, so hardware scores
are literature-plus-judgement, and the pilot at Gate 3a exists to falsify them.

| Dimension | What a 5 means | What a 1 means |
| --- | --- | --- |
| A. Algorithm complexity | Stated in a few equations a reader can verify on one page | Requires auxiliary machinery (hypernetworks, subset enumeration, curvature factorizations) to state at all |
| B. Local hardware requirement | Fits 8 GB VRAM with a wide margin, or needs no GPU | Marginal at 8 GB, or needs a larger model/multi-GPU |
| C. Implementation difficulty | All equations recoverable from public text; few interacting parts; an independent library exists to cross-check | Equations not retrievable, or many coupled components with no reference to check against |
| D. Novelty | 2025–2026, peer-reviewed at a strong venue, with a real mechanism change | Older than the window, or venue-weak, or a re-parameterisation of a known method |
| E. Ease of scoring on the brief | Generates rich process evidence, several graded metrics, and a genuinely visualizable Bayesian quantity | Thin implementation narrative, one metric, nothing to show in a webpage |

## 2. The four scored candidates

### C1 — F-EDL: Flexible Evidential Deep Learning (primary recommendation)

Paper: *Uncertainty Estimation by Flexible Evidential Deep Learning*, NeurIPS 2025 (verified via the
arXiv Comments field, the repository description, and the README acceptance statement),
arXiv:2510.18322. Code: `github.com/TaeseongYoon/F-EDL`, **no license declared**.

Probability core (VERIFIED from the HTML method section):
`α_k = exp(g_φ1(z))`, `p = softmax(g_φ2(z))`, `τ = softplus(g_φ3(z))` — three separate heads emit a
concentration α, a modal probability vector p and a learned precision τ, so the class-probability
posterior is `π | x ~ FD(α, p, τ)` instead of the standard Dirichlet that classical EDL ties to a
single non-negative evidence head via `α = e + 1`. Objective (VERIFIED as extracted):
`L = E_{π~FD} ‖y − π‖² + ‖y − p‖²`.

| Dimension | Score | Evidence |
| --- | --- | --- |
| A | 4 | Three extra heads plus one Dirichlet-family sample per loss term; no curvature, no ensemble, no subset lattice. The unverified part is the exact target encoding and any loss weighting |
| B | 5 | VERIFIED headline setting is ResNet-18 on CIFAR-10 at 91.19 ± 0.2 top-1 (Table 1). A ResNet-18 at 32×32 is far inside 8 GB; the head adds negligible compute |
| C | 4 | Equations are recoverable from public text, and `torch-uncertainty` (Apache-2.0, v0.13.0, Python ≥3.10 <3.14) ships classical EDL plus the whole metric battery as an independent cross-check. Down from 5 because the official repo is unlicensed and we will reimplement rather than reuse, and Appendix E hyperparameters were never returned |
| D | 5 | NeurIPS 2025, and the change is structural (decoupling α, p, τ), not cosmetic |
| E | 5 | Yields accuracy, NLL, Brier, ECE, OOD AUROC/AUPR and risk-coverage from one run, and the per-class evidence and epistemic/aleatoric split are directly plottable — strong material for report components (3), (4) and the HTML visualization |

Estimated cost on our machine `[estimate]`: one ResNet-18/CIFAR-10 run ≈ 25–45 minutes; four
comparator runs plus a small ablation ≈ 3–4 hours wall-clock total. Not measured.

Known risk, stated plainly: the strongest paper attacking this family is NeurIPS 2024's
*"Are Uncertainty Quantification Capabilities of Evidential Deep Learning a Mirage?"* (arXiv:2402.06160,
venue verified via the proceedings hash). It argues EDL works despite poor uncertainty
quantification and behaves as an energy-based OOD detector, and that the non-vanishing-uncertainty
problem originates in the KL-regulariser target (the result itself belongs to the Bengs et al. line,
which F-EDL's squared-error objective does not use — that is a point in F-EDL's favour). Our
experiment must therefore include a plain cross-entropy + temperature-scaling baseline, and if EDL
loses on some metric, that becomes a reported finding rather than a failure.

### C2 — ICLA: Identity Curvature Laplace Approximation (recommended as the in-experiment Bayesian comparator)

Paper: *Identity Curvature Laplace Approximation for Improved Out-of-Distribution Detection*, WACV
2025 (verified in the CVF open-access proceedings listing), arXiv:2312.10464v2. Code:
`github.com/maxnygma/icla` (identified, not audited).

Probability core (VERIFIED): a Gaussian posterior over **last-layer weights only**, with the curvature
term replaced by the identity matrix so no Hessian is computed, and the prior precision λ chosen by
maximising the marginal likelihood. This is the most textbook-readable Bayes chain available to us:
prior → likelihood → posterior → evidence, all on one page.

| Dimension | Score | Evidence |
| --- | --- | --- |
| A | 5 | One Gaussian posterior, one scalar hyperparameter; the whole method is "throw away the Hessian, fit λ" |
| B | 4 | VERIFIED schedule: ResNet-18 + SGD, 100 epochs on CIFAR-10/100, 90 on ImageNet-200, then one post-hoc pass. The post-hoc pass is cheap; the backbone training is the same cost as C1 |
| C | 5 | `laplace-torch` is MIT, v0.2.3 uploaded 2026-10-03, and its Calibration tutorial is exactly last-layer Laplace on WideResNet-16-4/CIFAR-10 with `hessian_structure="kron"`. We implement the math ourselves and validate against the library — the same "analytic vs library, correlation 1.0" pattern that worked in Assignment 1 |
| D | 4 | WACV 2025 is a real CV venue, but the lineage is 2021 last-layer Laplace, so novelty is a refinement rather than a new family |
| E | 4 | Predictive mean and variance, per-class uncertainty, risk-coverage curves. Slightly weaker than C1 on the "recognition accuracy" story because its published headline metric is OOD AUROC, not top-1 accuracy |

### C3 — HENN: Hyper Evidential Deep Learning (reserve)

Paper: *Hyper Evidential Deep Learning to Quantify Composite Classification Uncertainty*, ICLR 2024
(venue conflict resolved in favour of ICLR 2024 by the citation check), arXiv:2404.10980. Code:
`github.com/Hugo101/HyperEvidentialNN`, no license.

| Dimension | Score | Evidence |
| --- | --- | --- |
| A | 2 | Hypernetwork producing evidence over the subset lattice; `α = e + 1`, composite parameter `c = e`, epistemic term as Jøsang vagueness `vag(ω) = Σ b_S`, loss `UPCE + λ·KL[GDD ‖ GDD_uniform]` — all verified, but heavy to state and to get right |
| B | 3 | VERIFIED backbones are EfficientNet-b3, ResNet-50 and VGG16 on Tiny-ImageNet, Living17, Nonliving26, CIFAR-100 — the heaviest profile in this shortlist for 8 GB |
| C | 2 | Subset-lattice bookkeeping with no verified accuracy, ECE or epoch figures surfaced; no maintained library implements it |
| D | 4 | ICLR 2024, genuinely different uncertainty decomposition |
| E | 4 | The most interpretable uncertainty story of all four if it works |

Verdict: not recommended inside a 6-day window. Keep as the "we considered and rejected, here is why"
entry, which is itself useful report content.

### C4 — ProbaCLIP: training-free KDE Bayes classifier on frozen CLIP embeddings (fastest, weakest novelty)

Paper: *Training-Free Few-Shot Image Classification via Kernel Density Estimation with CLIP
Embeddings*, Mathematics 13(22):3615, 2025 (venue verified through the DOAJ record; the MDPI HTML
itself returns 403). Code: `github.com/icai-uma/ProbaCLIP`, no license, last push 2026-04-13.

| Dimension | Score | Evidence |
| --- | --- | --- |
| A | 5 | Class-conditional Gaussian-kernel KDE over frozen CLIP features after PCA to 99 % variance, priors from class frequencies, argmax posterior. Nothing else |
| B | 5 | No training at all; feature extraction plus density evaluation. VERIFIED variants: ViT-B/32 (512-d) and ViT-L/14@336px (768-d) |
| C | 5 | Reproducible in hours; scikit-learn `KernelDensity` covers the mechanism |
| D | 2 | 2025 but MDPI-tier, and our systematic title sweep of CVPR 2025 (2,871 titles), ICCV 2025, WACV 2025 (929 titles) and BMVC 2025 found **no** in-window top-venue paper using a closed-form class-conditional density plus Bayes rule as the primary classifier — such densities appear only as components inside trained systems. So "recent" is satisfied, "state of the art" is not |
| E | 2 | The report's component (3) "how AI implements the algorithm" and component (4) "experiment settings and results" both become thin: no training loop, no overfitting story, no regularization ablation, few failure modes to document. Reported numbers are few-shot episode accuracies (VERIFIED: 89.62/96.36 % 1-/5-shot miniImageNet with ViT-B/32; 93.48/98.37 % with ViT-L/14), a different evaluation frame from the K-way protocol the course assumes |

## 3. Score summary

| Candidate | A complexity | B hardware | C implementation | D novelty | E scoring | Total |
| --- | --- | --- | --- | --- | --- | --- |
| **C1 F-EDL (NeurIPS 2025)** | 4 | 5 | 4 | 5 | 5 | **23** |
| **C2 ICLA (WACV 2025)** | 5 | 4 | 5 | 4 | 4 | **22** |
| C4 ProbaCLIP (MDPI 2025) | 5 | 5 | 5 | 2 | 2 | 19 |
| C3 HENN (ICLR 2024) | 2 | 3 | 2 | 4 | 4 | 15 |

## 4. Options for Gate 1

**Option A (recommended): C1 as the algorithm, C2 as the Bayesian comparator inside the same
experiment.**
One backbone budget buys both: train a compact CNN once with the flexible-Dirichlet head (C1), and
train the same backbone with cross-entropy, then attach last-layer Laplace with identity curvature
(C2) post-hoc, plus a cross-entropy + temperature-scaling baseline that the Mirage critique says is
hard to beat. Four configurations, one shared data pipeline, one shared metric harness. This is the
only option that lets us *test* the field's central criticism with our own measurements instead of
citing it, which is the strongest possible material for report component (5).
Estimated total wall-clock `[estimate]`: 3–4 hours, i.e. comfortably inside the 2026-10-08 pilot and
2026-10-09 full-run slots. Not measured.

**Option B: C2 alone.** Lowest implementation risk, most legible Bayes mathematics, weakest novelty,
and the published task is OOD detection rather than recognition.

**Option C: C4 alone.** Near-zero risk of a late failure because nothing is trained, but it forfeits
the novelty requirement and produces a thin implementation narrative. Only sensible if the supervisor
prefers certainty over marks.

**Option D: C1 alone.** Simpler than Option A but throws away the free second Bayesian method, and
leaves us without an independent implementation to cross-check our own.

## 5. Dataset position (to be confirmed at Gate 2, not Gate 1)

Per the supervisor's Gate-0 answer, the dataset follows the algorithm. For C1/C2 the natural choice is
**CIFAR-10** because it is the verified headline setting of both papers (F-EDL Table 1: 91.19 %;
ICLA: 100 epochs, ResNet-18), so our numbers are directly comparable to published ones — the single
most useful property for a report that must show the AI reproduced a real algorithm. Official CIFAR-10
python archive measured at 163 MB by a subagent. A second small in-distribution set (CIFAR-100, or the
locally cached EuroSAT copy from Assignment 1) plus SVHN and Tiny-ImageNet as out-of-distribution
controls would satisfy the outline's cross-dataset comparison. No dataset has been downloaded in this
stage.

## 6. What approval at Gate 1 does and does not permit

Approving Option A authorises: writing the design and experiment plan (Gate 2), and preparing
read-only code skeletons for review. It does **not** authorise: downloading datasets or weights,
installing `torch-uncertainty` / `laplace-torch` (both approved as *candidates*; the actual `pip
install` waits for Gate 2), starting any training run, or creating the public GitHub repository.
