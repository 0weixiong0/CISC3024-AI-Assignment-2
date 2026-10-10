# Deep Research: Recent Bayesian and Probabilistic Classification Algorithms for Computer Vision

> Generated 2026-10-06 | Depth: standard | Sources: 34 | Subagents: 3 retrieval + 2 gap-fill + 1 verification

*Artifact language: English, per the project's supervision contract (research artifacts in English,
supervisor briefings in Chinese). Search terms were English because the citation backbone of this
assignment must be English-language literature.*

---

## TL;DR

Within 2024–2026 the laptop-feasible frontier of Bayesian image classification has moved decisively
away from posteriors over all weights and towards **cheap probabilistic structure attached to an
ordinary trained network** — either a posterior over class probabilities (evidential/Dirichlet heads)
or a Gaussian posterior over the last layer only (post-hoc Laplace). The recommended choice for this
assignment is **F-EDL (NeurIPS 2025)** as the algorithm under study, with **ICLA (WACV 2025)** as a
second Bayesian comparator and plain cross-entropy plus temperature scaling as the baseline, because
that combination shares one backbone-training budget and lets the experiment test the field's own
strongest criticism rather than merely citing it.

## Executive Summary

Four families were screened across 25 ledger records: Bayesian deep models with weight-space
posteriors, generative class-conditional density classifiers, closed-form probabilistic classifiers
with modern regularisation, and uncertainty-aware recognition built on Bayesian decision theory.

Three findings drive the recommendation.

First, **recency and feasibility pull in the same direction only for post-hoc and head-level
methods.** The 2024–2026 literature that is both peer-reviewed at a strong venue and cheap enough for
an 8 GB laptop concentrates on the classification head: F-EDL decouples the Dirichlet concentration,
mode and precision into three learned heads [1][2]; hypernetwork evidential models enumerate belief
over subsets [6][7]; and the Laplace line replaced the Hessian with an identity matrix and optimised
only the prior precision [3][4]. Full weight-space posteriors, MCMC and low-rank Bayesian fine-tuning
all sit outside a six-day window [36].

Second, **the closed-form option is real but venue-weak.** ProbaCLIP is a genuinely training-free
Bayes-rule classifier — Gaussian-kernel class-conditional densities over frozen CLIP embeddings,
priors from class frequencies, argmax posterior — and it reports 96.36 % five-shot accuracy on
miniImageNet [8][9]. But a systematic title sweep of CVPR 2025 (2,871 titles), ICCV 2025, WACV 2025
(929 titles) and BMVC 2025 found no in-window top-venue paper in which a closed-form class-conditional
density plus Bayes rule *is* the classifier; the top-venue papers use Gaussian and mixture densities
only as components inside trained systems [30][33][34][35]. A student choosing this route satisfies
"recent" but not "state of the art", and forfeits most of the implementation narrative the report has
to document.

Third, and most important for honest experiment design, **the evidential family has a strong internal
critique that must be reproduced rather than ignored.** A NeurIPS 2024 paper argues that evidential
networks deliver useful behaviour "despite their poor uncertainty quantification capabilities" and
function mainly as energy-based out-of-distribution detectors [10][11], and the Laplace family's own
authors report that Hessian fitting is a bottleneck that does not improve calibration at all [4][5].
Both critiques name cheap deterministic baselines as the competitor. Any assignment experiment that
reports only accuracy would therefore be measuring the wrong thing; the metric set has to include
calibration and selective-prediction scores, which a maintained library now provides for free
[12][13].

## 1. Status Quo — where Bayesian classification for vision actually sits

**Confidence: High** (Tier-1 peer-reviewed sources, multiple independent confirmations).

The classical textbook picture — estimate class-conditional densities, multiply by priors, take the
argmax posterior — survives in computer vision mainly in forms where the density is estimated over a
*frozen pretrained representation* rather than over pixels. This is the single most consistent pattern
found across the 2024–2026 retrieval waves: the expensive part (the visual encoder) is rented from a
pretrained model, and the Bayesian contribution is confined to a small, well-understood probabilistic
head [8][9][18][28][32]. Three variants of this pattern were confirmed in the window: kernel-density
class-conditional models over CLIP embeddings [8][9], Gaussian-process layers inserted at the
activations of an already-trained network [14], and Gaussian heads fitted to aggregated feature
statistics [35].

Where the classifier must actually be *trained* rather than estimated, the 2024–2026 literature
overwhelmingly prefers **approximate posteriors restricted to a small parameter subset**. Last-layer
Laplace — a Gaussian posterior over only the final linear layer, with the prior precision chosen by
marginal likelihood — is the canonical example, introduced in 2021 [16][17] and refined through 2024
and 2025 by variational linearisation [15] and by the identity-curvature variant [3][4]. The reason
for this restriction is empirical rather than theoretical: the full-curvature machinery that the
original Laplace literature emphasised was shown to be actively harmful for out-of-distribution
behaviour while contributing nothing to calibration [4][5].

The second dominant pattern is the **evidential head**, which abandons weight-space posteriors
altogether and puts a distribution over the probability simplex itself. The classical formulation maps
non-negative network evidence to Dirichlet parameters via `α = e + 1` and adds a KL regulariser
towards a uniform Dirichlet [21]. The 2024–2026 line of work modifies exactly this coupling: F-EDL
replaces the tied parameterisation with three independent heads producing concentration, mode and
precision [1][2]; the hypernetwork family replaces the single Dirichlet with a belief structure over
class subsets, so epistemic uncertainty becomes Jøsang vagueness summed over subset masses [6][7]; and
a NeurIPS 2024 variant moves from Dirichlet to a subjective-logic opinion carrying an explicit
uncertainty mass [20]. A 2026 IEEE TPAMI survey confirms that this is now a recognised subfield with
its own taxonomy rather than a collection of tricks [19].

For a course assignment, the practical consequence is that the interesting Bayesian quantities are
now cheap enough to compute twice — once by hand and once with a library — which is precisely the
verification posture that makes the "how the AI implemented it" section auditable.

## 2. Emerging Trends — what changed in the last eighteen months

**Confidence: Medium-High** (recent Tier-1 sources; some trend inference from title-level evidence).

**Post-hoc Bayesianisation is becoming a post-training step, not a training regime.** The clearest
signal is the tooling: `laplace-torch` shipped version 0.2.3 on 2026-10-03 under MIT and documents ten
combinations of weight-subset and curvature structure [22][23], while `pytorch-ood` pushed to its dev
branch on 2026-10-05 with a family of post-hoc detectors that require no retraining at all [24].
Training-Free Bayesianization goes further and converts an already-trained low-rank adaptation into a
Bayesian one by searching for the maximal acceptable variance under a low-rank isotropic Gaussian
weight posterior [36]. The direction of travel is unambiguous: the posterior is being pushed to the
smallest parameter set that still yields useful uncertainty.

**Metric standardisation has arrived.** Torch-Uncertainty, accepted to the NeurIPS 2025 Datasets and
Benchmarks track, exposes Brier score, categorical NLL, AUROC/AUPR/FPR95, classwise and adaptive ECE
variants, and the selective-classification family (AURC, AUGRC, coverage at 5 % risk, risk at 80 %
coverage) as a single API surface, alongside evidential classification, Dirichlet calibration,
conformal prediction and Laplace integration [12][13]. Two years ago a student assembling this battery
would have written most of it by hand; today it is a dependency, which changes what is worth
implementing for its own sake.

**Bayesian decision rules with rejection are displacing pure classification.** CLIPScope applies an
explicit Bayesian posterior update to normalise confidence for zero-shot out-of-distribution detection
[27]; Sammani and Deligiannis re-derive CLIP's zero-shot rule through Bayes' theorem and show that the
class-prior factor biases the decision [26]; open-set work examines how rejection cost scales with
label-space size [29]. The recognition problem is increasingly stated as "classify or abstain", which
suits Bayes decision theory far better than the plain argmax framing.

**A cautionary trend line.** The 2026 TPAMI survey's own framing is "open challenges" [19], and 2026
arXiv activity concentrates on repairing post-hoc calibration rather than celebrating it [31]. Novelty
claims in this area should be read as incremental.

## 3. Critical Assessment — where these methods demonstrably fail

**Confidence: High** (direct quotations from the papers that report the failures).

**Evidential deep learning may be solving the wrong problem.** The most serious critique found is that
EDL's uncertainty is *spurious* because the formulation assumes no model randomness, and that the
dominant factor is the fixed target distribution imposed by the KL regulariser, which means epistemic
uncertainty cannot shrink to zero even with abundant data — a result the paper attributes to the Bengs
et al. line and then sharpens [10][11]. Its headline empirical claim is that EDL methods behave as
"distributed OOD detection algorithms based on energy-based models" and that deterministic
distillation baselines match or outperform classical EDL on out-of-distribution and
selective-classification tasks [10]. This matters for the choice at hand: F-EDL's squared-error
objective does not use the KL-to-uniform term that the critique attacks [1][2], which is a genuine
design argument in its favour — but it also means the published baselines our experiment must beat are
the cheap deterministic ones, not other Bayesian models.

**Reported confidence can be visibly inconsistent with reported accuracy.** In the F-EDL table that was
verified line by line, CIFAR-10 top-1 accuracy is 91.19 ± 0.2 while the accompanying mean confidence
figure is 99.10 [1][2]. Whether that column measures confidence on correct predictions or something
else could not be established from the retrieved text, but on its face it is the over-confidence
signature the critique predicts, and it is exactly the quantity our own experiment must measure rather
than inherit.

**The Laplace family's own machinery was found counterproductive.** Hessian fitting "creates a
bottleneck" in the gradients and "fails to widen the uncertainty surface for outliers"; prior-only
last-layer Laplace reached 92.50 % far-OOD AUROC on CIFAR-10 against 91.54 % for the GGN-curvature
variant, and "calibration scores remain similar across methods", leading the authors to recommend
discarding full Hessian computation for out-of-distribution tasks [4][5]. The WACV 2025 successor
implements exactly that recommendation: identity curvature, prior precision optimised by marginal
likelihood, ResNet-18 with SGD for 100 epochs on CIFAR-10 and CIFAR-100 and 90 on ImageNet-200,
winning five of six reported AUROC columns [3]. Note the metric: this is detection performance, not
recognition accuracy.

**Expensive Bayesian deep learning is not reliably better.** The NeurIPS 2023 large-scale evaluation
under shift remains the standard reference for that statement [18]. It predates the search window and
is cited here as foundational context, not as a recent result.

**Two evidence gaps that constrain any claim of completeness.** First, wall-clock and GPU-hours are
stated in none of the candidate papers; epoch counts were verified only for ICLA, and F-EDL defers its
hyperparameters to an appendix that no retrieval route returned [1][2][3]. Second, one promising
closed-form candidate — a von Mises-Fisher density head for CLIP — could not be verified through any
of seven routes (OpenReview verification wall, PDF 403, API challenge, no archive snapshot, no arXiv,
Semantic Scholar or Crossref record) [37], so it is excluded rather than cited. An absence of evidence
produced by title sweeps is strong screening evidence, not proof; no PRISMA-compliant claim is made.

## 4. Action Plan

- [ ] Supervisor approves Gate 1 (recommended Option A: F-EDL as the algorithm, ICLA as the Bayesian
      comparator, cross-entropy plus temperature scaling as the baseline).
- [ ] Write the Gate 2 design document: prior/likelihood/posterior specification for both Bayesian
      models, the frozen CIFAR-10 train/validation/test split with seed, the metric battery, and the
      regularisation list (label smoothing, weight decay, KL variants, early stopping on validation
      NLL, temperature and Dirichlet calibration).
- [ ] Request the two dependency installs explicitly (`torch-uncertainty` Apache-2.0, `laplace-torch`
      MIT) and record their licences in the report's reproducibility appendix.
- [ ] Gate 3a pilot on a 10 % CIFAR-10 subset: measure real wall-clock, peak VRAM and epoch budget for
      one ResNet-18 run, replacing every `[estimate]` in this document with a measurement.
- [ ] Implement the flexible-Dirichlet head by hand and cross-check it against the library
      implementation, reporting the agreement numerically rather than asserting correctness.
- [ ] Gate 3b full run: four configurations (CE, CE+temperature, F-EDL, CE-backbone+ICLA), one shared
      metric harness, plus one out-of-distribution control set (SVHN or Tiny-ImageNet).
- [ ] Gate 4: build the HTML visualization around the Bayesian quantities that actually exist —
      per-class evidence, posterior mean and variance, risk-coverage curve, calibration diagram — and
      publish the repository plus Pages link that PDF component (6) requires.
- [ ] Gate 5: draft the report against `Report Template.docx` with APA 7 references, keeping the
      critique material in Section 3 above as the honest basis for component (5).

## 5. Open Questions & Caveats

1. **Does F-EDL's decoupling survive measurement?** The mechanism is verified but its published
   evaluation reports AUPR rather than AUROC and no ECE at all in the table read [1][2]. Whether the
   flexible parameterisation improves calibration relative to classical EDL and to temperature-scaled
   cross-entropy is precisely what our own experiment has to establish.
2. **F-EDL's training schedule is unknown.** Epochs, optimiser and learning rate sit in Appendix E,
   which was unreachable through every route tried; the schedule must be set from our own validation
   curves and reported as ours, not as the paper's.
3. **Both official repositories are unlicensed** (`license: null` via the GitHub API for F-EDL, and the
   same for ProbaCLIP and HyperEvidentialNN) [2][9][7]. Reimplementation from the paper is therefore
   not merely convenient but necessary, which happens to align with the assignment's rule that the AI
   writes all code.
4. **ICLA and the pathologies paper are the same arXiv record.** arXiv:2312.10464 v1 is titled
   "Unveiling Empirical Pathologies of Laplace Approximation for Uncertainty Estimation" and v2 is the
   WACV 2025 "Identity Curvature Laplace Approximation" paper [4][5]. Any citation must pin the
   version, or the bibliography will appear to double-count one study.
5. **Hyper-opinion EDL remains unscored.** Its NeurIPS 2024 acceptance and authorship are confirmed but
   the method equations, datasets, results and code status could not be retrieved after two attempts
   [20]. It is recorded as a gap, not as a rejection.
6. **The Mirage critique's numbers were not read.** Its venue, dataset list and qualitative conclusion
   are verified; the per-method accuracy, NLL, ECE and AUROC values are not, so no figure from it may
   enter the final report until the tables are read verbatim [10][11].
7. **Hardware scores are unmeasured.** Every statement about fitting inside 8 GB rests on published
   backbone choices, not on a local benchmark. The Gate 3a pilot is the falsification test.

## Methodology

Depth: **standard**, per the `deep-research` skill, chosen because the deliverable is a scored
shortlist under a six-day deadline rather than an exhaustive review. Pipeline: Phase 0 clarification
was satisfied by the supervisor's Gate-0 answers (widest admissible algorithm scope; dataset to follow
the algorithm; approval to run `deep-research`; public repository plus Pages deferred to 2026-10-10).
Phase 1 scoped six areas. Phase 2 Wave 1 ran three parallel retrieval subagents (8 WebSearch and 5–6
WebFetch each; two reported hitting their fetch ceiling). Phase 3 triangulation identified three gaps;
Wave 2 ran two gap-fill subagents. Phase 3.1 delegated citation verification to one subagent covering
eight high-impact claims, which produced seven corrections recorded in
`research_records/stage_01_search_protocol.md` §6 — including one resolved venue conflict (HENN is
ICLR 2024, not venue-less), one narrowed dataset-scope claim (F-EDL's in-distribution options are
MNIST, CIFAR-10 and CIFAR-100 only), one unconfirmed technical detail (ProbaCLIP's bandwidth rule), and
one reattribution (the non-vanishing-uncertainty result belongs to the Bengs et al. line). Phase 3.5
left the outline unchanged; Phase 4 red-teaming surfaced the missing wall-clock evidence, which is
reported as an open question rather than filled with an estimate presented as a fact.

No PRISMA-style systematic-review claim is made. Papers and their repositories were deduplicated by
URL. Tier-3 sources were used only for negative sweeps and bibliographic metadata. Nothing was
downloaded into the project except one reference PDF of the ProbaCLIP article, saved by a subagent for
verification; no dataset, weight or package was installed.

## Bibliography

[1] Yoon, T., Kim, H., et al. — *Uncertainty Estimation by Flexible Evidential Deep Learning* (arXiv
Comments: NeurIPS 2025) — https://arxiv.org/abs/2510.18322 — Accessed 2026-10-06 — Tier: 1
[2] TaeseongYoon — F-EDL official repository (no license) — https://github.com/TaeseongYoon/F-EDL —
Accessed 2026-10-06 — Tier: 1
[3] Zhdanov, M., et al. — *Identity Curvature Laplace Approximation for Improved Out-of-Distribution
Detection* — WACV 2025, https://openaccess.thecvf.com/content/WACV2025/html/Zhdanov_Identity_Curvature_Laplace_Approximation_for_Improved_Out-of-Distribution_Detection_WACV_2025_paper.html
— Accessed 2026-10-06 — Tier: 1
[4] Zhdanov, M., Dereka, S., Kolesnikov, S. — arXiv:2312.10464v2 (WACV version) —
https://arxiv.org/html/2312.10464v2 — Accessed 2026-10-06 — Tier: 1
[5] Zhdanov, M., Dereka, S., Kolesnikov, S. — *Unveiling Empirical Pathologies of Laplace
Approximation for Uncertainty Estimation* — arXiv:2312.10464v1 — https://arxiv.org/abs/2312.10464 —
Accessed 2026-10-06 — Tier: 1 [foundational for the identity-curvature design]
[6] Li, C., Li, K., Ou, Y., Kaplan, L. M., Jøsang, A., Cho, J.-H., Jeong, D. H., Chen, F. — *Hyper
Evidential Deep Learning to Quantify Composite Classification Uncertainty* — ICLR 2024 —
https://arxiv.org/abs/2404.10980 — Accessed 2026-10-06 — Tier: 1
[7] Hugo101 — HyperEvidentialNN repository — https://github.com/Hugo101/HyperEvidentialNN — Accessed
2026-10-06 — Tier: 1
[8] Lima Junior, M. S. P. d. S., Ortiz-de-Lazcano-Lobato, J. M., López-Rubio, E. — *Training-Free
Few-Shot Image Classification via Kernel Density Estimation with CLIP Embeddings* — Mathematics
13(22):3615, 2025 — https://doi.org/10.3390/math13223615 — Accessed 2026-10-06 — Tier: 2
[9] icai-uma — ProbaCLIP repository (no license) — https://github.com/icai-uma/ProbaCLIP — Accessed
2026-10-06 — Tier: 1
[10] Shen, M., Ryu, J. J., Ghosh, S., Bu, Y., Sattigeri, P., Das, S., Wornell, G. W. — *Are
Uncertainty Quantification Capabilities of Evidential Deep Learning a Mirage?* — NeurIPS 2024 —
https://arxiv.org/abs/2402.06160 — Accessed 2026-10-06 — Tier: 1
[11] NeurIPS 2024 proceedings record for [10] —
https://proceedings.neurips.cc/paper_files/paper/2024/file/c3177be226ee12e34d6ba3b5e6fe6a5b-Paper-Conference.pdf
— Accessed 2026-10-06 — Tier: 1
[12] Lafage, E., Tangel, A., Dierckx, L., Hartmann, O., de Bézenac, E., Pajot, A. — *Torch-Uncertainty:
A Deep Learning Framework for Uncertainty Quantification* — NeurIPS 2025 Datasets & Benchmarks —
https://arxiv.org/abs/2511.10282 — Accessed 2026-10-06 — Tier: 1
[13] Torch-Uncertainty documentation — https://torch-uncertainty.github.io/ — Accessed 2026-10-06 —
Tier: 1
[14] Bergna, R., Depeweg, S., Calvo Ordonez, S., Plenk, J., Cartea, A., Hernandez-Lobato, J. M. —
*Post-Hoc Uncertainty Quantification in Pre-Trained Neural Networks via Activation-Level Gaussian
Processes* — arXiv:2502.20966 (arXiv-only) — https://arxiv.org/abs/2502.20966 — Accessed 2026-10-06 —
Tier: 1
[15] Authors not resolved — *Variational Linearized Laplace Approximation for Bayesian Neural
Networks* — ICLR 2024 — https://iclr.cc/virtual/2024/23396 — Accessed 2026-10-06 — Tier: 1
[16] Daxberger, E., Kristiadi, A., D'Angelo, A., van der Wilk, M., Hennig, P., Grosse, R. — *Laplace
Redux — Effortless Bayesian Deep Learning* — NeurIPS 2021 — https://github.com/aleximmer/Laplace —
Accessed 2026-10-06 — Tier: 1 [foundational]
[17] Kristiadi, A., Hein, M., Bauerschmidt, S. — *Learnable Uncertainty under Laplace
Approximations* — AISTATS 2021 — https://proceedings.mlr.press/v161/kristiadi21a/kristiadi21a.pdf —
Accessed 2026-10-06 — Tier: 1 [foundational]
[18] Müller, A., et al. — *A Large-Scale Evaluation of Bayesian Deep Learning under Shift* — NeurIPS
2023 Datasets & Benchmarks —
https://proceedings.neurips.cc/paper_files/paper/2023/file/5d97b7e62022c859347397f6c1e8d0f9-Paper-Conference.pdf
— Accessed 2026-10-06 — Tier: 1 [foundational]
[19] IEEE TPAMI 48(3), 2026 — *A Comprehensive Survey on Evidential Deep Learning for Uncertainty
Quantification* — https://www.computer.org/csdl/journal/tp/2026/03/11217233/2b4UDI712yk — Accessed
2026-10-06 — Tier: 1 (paywalled; abstract-level only)
[20] Qu, J., Chen, Y., Yue, X., Fu, W., Huang, Q. — *Hyper-opinion Evidential Deep Learning for
Out-of-Distribution Detection* — NeurIPS 2024 (abstract page only; PDF unreadable) —
https://proceedings.neurips.cc/paper_files/paper/2024/hash/99d4ceebdf75b64e8ed608a245b63416-Abstract-Conference.html
— Accessed 2026-10-06 — Tier: 1
[21] Sensoy, M., Kaplan, L., Kandemir, M. — *Evidential Deep Learning to Quantify Classification
Uncertainty* — NeurIPS 2018 (the KL-to-uniform regulariser the critique in [10] targets; identified
via [19][21] lineage) — https://arxiv.org/abs/1806.01768 — Accessed 2026-10-06 — Tier: 1 [foundational]
[22] PyPI — `laplace-torch` 0.2.3, MIT, uploaded 2026-10-03T19:37:38 —
https://pypi.org/pypi/laplace-torch/json — Accessed 2026-10-06 — Tier: 1
[23] Laplace documentation — Calibration tutorial (WideResNet-16-4, CIFAR-10, `last_layer`, `kron`) —
https://aleximmer.github.io/Laplace/ — Accessed 2026-10-06 — Tier: 1
[24] kkirchheim — pytorch-ood v0.4.0, Apache-2.0, pushed 2026-10-05 —
https://github.com/kkirchheim/pytorch-ood — Accessed 2026-10-06 — Tier: 1
[25] scikit-learn documentation — Discriminant Analysis (Ledoit-Wolf/OAS shrinkage, regularized QDA)
and Gaussian Mixture Models / Kernel Density — https://scikit-learn.org/stable/modules/lda_qda.html ,
https://scikit-learn.org/stable/modules/mixture.html — Accessed 2026-10-06 — Tier: 1
[26] Sammani, F., Deligiannis, N. — *Interpreting and Analysing CLIP's Zero-Shot Image Classification
via Mutual Knowledge* — arXiv:2410.13016 (NeurIPS 2024 attribution reported by fetch, not
independently confirmed) — https://arxiv.org/abs/2410.13016 — Accessed 2026-10-06 — Tier: 1
[27] Fu, et al. — *CLIPScope: Enhancing Zero-Shot OOD Detection with Bayesian Scoring* — WACV 2025 —
https://openaccess.thecvf.com/content/WACV2025/papers/Fu_CLIPScope_Enhancing_Zero-Shot_OOD_Detection_with_Bayesian_Scoring_WACV_2025_paper.pdf
— Accessed 2026-10-06 — Tier: 1
[28] Oquab, M., et al. — DINOv2 (code and weights) — https://github.com/facebookresearch/dinov2 —
Accessed 2026-10-06 — Tier: 1 [component, not a candidate]
[29] CEUR-WS Vol-4082 paper 6 — *Open-Set Recognition with Scalable Rejection under Large Label
Spaces* — https://ceur-ws.org/Vol-4082/paper6.pdf — Accessed 2026-10-06 — Tier: 1 (workshop level)
[30] CVF — CVPR 2025 and WACV 2025 open-access title listings used for the negative sweep (2,871 and
929 titles respectively) — https://openaccess.thecvf.com/CVPR2025?day=all ,
https://openaccess.thecvf.com/WACV2025?day=all — Accessed 2026-10-06 — Tier: 1
[31] arXiv 2026 preprints on post-hoc calibration repair: *Instance-Level Post Hoc Uncertainty
Quantification* https://arxiv.org/pdf/2606.04656 and *Confidence–Uncertainty Boundary Calibration*
https://arxiv.org/html/2602.11973v2 — Accessed 2026-10-06 — Tier: 1 (**title-level only; content not
read — cited as evidence of activity, not for any claim**)
[32] Authors not resolved — *Generative Classifiers Avoid Shortcut Solutions* — ICLR 2025 —
https://arxiv.org/abs/2512.25034 — Accessed 2026-10-06 — Tier: 1
[33] Authors not resolved — *Hierarchical Gaussian Mixture Normalizing Flow Modeling for Unified
Anomaly Detection* — ECCV 2024 — https://eccv.ecva.net/virtual/2024/poster/435 — Accessed 2026-10-06 —
Tier: 1
[34] Lee, et al. — *FlowCLAS: Enhancing Normalizing Flow-Based Anomaly Segmentation Via Contrastive
Learning* — WACV 2026 —
https://www.openaccess.thecvf.com/content/WACV2026/papers/Lee_FlowCLAS_Enhancing_Normalizing_Flow-Based_Anomaly_Segmentation_Via_Contrastive_Learning_WACV_2026_paper.pdf
— Accessed 2026-10-06 — Tier: 1 (title/venue only)
[35] Authors not resolved — *Gaussian-Head OFL* (ICLR 2026 per author comment) —
https://arxiv.org/abs/2602.01186 and *DPDL* (CVPR 2025 per author comment) —
https://arxiv.org/abs/2502.20981 — Accessed 2026-10-06 — Tier: 1 (venue from author-reported comments,
not independently confirmed)
[36] Training-Free Bayesianization https://arxiv.org/abs/2412.05723 , BLoB
https://arxiv.org/abs/2406.11675 , ScalaBL https://arxiv.org/abs/2506.21408 — Accessed 2026-10-06 —
Tier: 1 (low-rank Bayesian fine-tuning family; excluded on fit, not on quality)
[37] Anonymous — *Improving CLIP Zero-Shot Classification via von Mises-Fisher …* —
https://openreview.net/forum?id=KyeyEFPxJX — Accessed 2026-10-06 — **UNREACHABLE through seven routes;
no claim rests on it**
[38] Krizhevsky, A. — CIFAR-10/100 technical report listing (python archive 163 MB) —
https://www.cs.toronto.edu/~kriz/cifar.html — Accessed 2026-10-06 — Tier: 1
[39] Wald, N., et al. — MedMNIST (CC-licensed, 28/64/128/224 px variants) —
https://github.com/MedMNIST/MedMNIST — Accessed 2026-10-06 — Tier: 1
[40] Chen, M., et al. — *R-EDL: Relaxing Nonessential Settings of Evidential Deep Learning* — ICLR
2024 — https://openreview.net/forum?id=Si3YFA641c — Accessed 2026-10-06 — Tier: 1 (lead only, not read)
[41] Apple Machine Learning Research — *Normalizing Flows are Capable Generative Models* —
https://machinelearning.apple.com/research/normalizing-flows — 2025-06-30 — Accessed 2026-10-06 —
Tier: 2
[42] Lambda Labs — *ResNet9: train to 94% CIFAR-10 accuracy in 100 seconds* —
https://lambda.ai/blog/resnet9-train-to-94-cifar10-accuracy-in-100-seconds — Accessed 2026-10-06 —
Tier: 2 (search snippet only; **not** used as a Bayesian-method timing claim)

## Source Extracts

### [1][2] F-EDL — NeurIPS 2025
- **Summary:** Flexible Dirichlet head with three learned outputs; decouples concentration, modal
  probability and precision that classical EDL ties together. CIFAR-10/100 and Dirty-MNIST evaluated
  with VGG-16 and ResNet-18.
- **Verified quotes/figures:** `α_k = exp(g_φ1(z))`; `p = softmax(g_φ2(z))`; `τ = softplus(g_φ3(z))`;
  `L = E_{π~FD} ‖y − π‖² + y − p‖²`; Table 1 CIFAR-10 Test.Acc. `91.19 ± 0.2`, confidence `99.10`,
  OOD AUPR SVHN `91.20` / CIFAR-100 `88.37`; `main.py --ID_dataset` choices `["MNIST","CIFAR-10",
  "CIFAR-100"]`; GitHub API `"license": null`; files `README.md datasets.py main.py models.py train.py
  uq.py`.
- **Source type:** academic + code repository. **Tier:** 1.

### [3][4][5] ICLA and its predecessor
- **Summary:** Last-layer Laplace with curvature replaced by the identity matrix; only the prior
  precision is optimised, by marginal likelihood. ResNet-18 + SGD, 100 epochs on CIFAR-10/100, 90 on
  ImageNet-200, OpenOOD v1.5 protocol.
- **Verified figures (AUROC, CIFAR-10 near/far):** ICLA `90.01±0.21 / 92.50±0.38`; LLLA(GGN)
  `88.94 / 91.54±0.36`; LLLA(EF) `89.54 / 92.09`; LLLA(K-FAC) `88.02 / 90.73`; CIFAR-100 ICLA
  `81.45±0.10 / 80.79±0.46`; ImageNet-200 ICLA `81.88±0.10 / 89.45±0.14`; tuned λ = 2.76 (C10), 3.06
  (C100). Pathologies paper: "Prior LLLA achieves 92.50% AUROC on CIFAR-10 Far OOD vs 91.54% for
  GGN-based LLLA"; "Calibration scores remain similar across methods; neglecting Hessian does not hurt
  calibration".
- **Source type:** academic. **Tier:** 1.

### [8][9] ProbaCLIP
- **Summary:** Training-free Bayes-rule classifier; Gaussian-kernel class-conditional KDE over frozen
  CLIP visual embeddings after PCA to 99 % variance; cross-validation consensus voting; one-shot
  handled with distance measures.
- **Verified figures:** ViT-B/32 (512-d) and ViT-L/14@336px (768-d); miniImageNet 1-/5-shot
  `89.62±0.46 / 96.36±0.21` (B/32) and `93.48±0.36 / 98.37±0.11` (L/14); tieredImageNet
  `85.65±0.59 / 93.86±0.33` and `90.60±0.48 / 96.84±0.22`; CUB `82.40±0.58 / 93.62±0.31` and
  `93.69±0.40 / 98.16±0.17`; ablation `95.26 → 95.34 → 95.57 → 97.80 → 98.07 → 98.37`; 99.80 % at
  16-shot (L/14). Code selects bandwidth by ANLL/MANLL over `--iterations_kde 500`; "AMISE" wording
  unconfirmed.
- **Source type:** academic + code. **Tier:** 2 (journal), 1 (repository).

### [10][11] EDL Mirage
- **Summary:** Argues EDL's uncertainty is spurious because model randomness is assumed absent; the KL
  target distribution dominates the learned uncertainty; EDL behaves as an energy-based OOD detector.
- **Verified:** six datasets (CIFAR-10, CIFAR-100, SVHN, FMNIST, Tiny-ImageNet, corrupted-ID);
  compared methods include RPriorNet, Belief Matching, PostNet, NatPN, EnD2 and Bootstrap
  Distillation; λ swept over {1e-4, 1e-1}; VGG-16 and ResNet backbones; conclusion quote "EDL methods
  are essentially OOD detectors… incorporating model uncertainty can help EDL methods faithfully
  quantify uncertainties". **Per-method numbers NOT read.**
- **Source type:** academic. **Tier:** 1.

### [12][13][22][23][24] Tooling
- **Summary:** Torch-Uncertainty (NeurIPS 2025 D&B, Apache-2.0, v0.13.0 uploaded 2026-07-22, Python
  ≥3.10 <3.14) provides evidential classification, Dirichlet calibration, conformal APS/RAPS,
  MC-BatchNorm, DEUP and Laplace integration with the full metric battery; laplace-torch (MIT, v0.2.3
  uploaded 2026-10-03) documents ten weight-subset × curvature combinations including a CIFAR-10
  WideResNet-16-4 last-layer K-FAC calibration example; pytorch-ood (Apache-2.0, v0.4.0, pushed
  2026-10-05) supplies post-hoc detectors.
- **Verified gaps:** no memory or runtime figures in the Laplace docs; marginalized-loss families
  (DPU/PFL/WDL/MoBL) absent from Torch-Uncertainty's API.
- **Source type:** documentation + repositories. **Tier:** 1.

### [30] Negative sweep
- **Summary:** Title sweeps of CVPR 2025 (2,871 titles), ICCV 2025, WACV 2025 (929 titles) and BMVC
  2025 returned no closed-form class-conditional-density Bayes-rule image classifier; "density" hits
  were Gaussian-splatting or unrelated, WACV's only Laplace hit was ICLA, and BMVC's Bayes hits were
  BNN-based.
- **Source type:** proceedings listings. **Tier:** 1.
