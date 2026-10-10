# Stage 01 — Literature Search Protocol, Executed Queries and Verification Record

Project: CISC3024 AI Assignment #2 · Date: 2026-10-06 · Language: English (working-stage report)
Skill used: `deep-research` (installed locally; supervisor approval recorded in
`conversation_log.md` as U002). Depth: **standard**.

Status: retrieval, triangulation, gap-fill and citation verification are complete. **The supervisor
has not yet approved a candidate.** Nothing below has been implemented, downloaded or trained.

---

## 1. Research question

Which recent (2024–2026) classification or recognition algorithm grounded in probability or Bayes
theory is genuinely applied to computer vision, and offers the best defensible combination of
algorithmic transparency, local feasibility on an RTX 4060 8 GB laptop, ease of AI reimplementation,
novelty, and coverage of the six mandatory report components — within a window that ends
2026-10-12?

## 2. Constraints fixed before searching

- Recency is a hard filter, not a preference: the PDF says "a recent … algorithm". Primary window
  2024–2026; pre-2024 work may appear only as a foundational reference, a tool, or a rejection
  example, and is labelled as such.
- The algorithm must be a classifier or recognizer for image data, not a generative model, not a
  segmentation or detection system, and not an LLM adaptation technique.
- The student may not write code; all implementation is AI-generated. Therefore an algorithm whose
  paper is unreadable or whose equations cannot be recovered from public sources is unusable, no
  matter how novel.
- No dataset, weight, package or repository was downloaded during screening. Web research read
  public evidence only. One exception is recorded in Section 7 (a PDF saved by a subagent into
  `research_records/`).
- Local resources measured on 2026-10-05: RTX 4060 Laptop 8188 MiB, 31.6 GiB RAM, Python 3.12.10,
  torch 2.7.1+cu128 with CUDA available, ~53 GiB free on D:. No candidate has been benchmarked
  locally, so every cost figure below is a literature value or an explicitly labelled estimate.

## 3. Search space (six areas)

| # | Area | Sub-questions |
| --- | --- | --- |
| 1 | Bayesian deep classification | Posterior over weights versus posterior over class probabilities; post-hoc Laplace family; variational and MC-dropout successors; evidential heads; what is closed-form versus iteratively fitted |
| 2 | Generative density classifiers on images | Normalizing flows, VAE/latent-density, Gaussian-mixture and subspace densities over deep features; the frozen-encoder + probabilistic-head pattern |
| 3 | Uncertainty-aware and open-set recognition | Bayesian decision rules with rejection, Dirichlet/evidential heads, selective prediction, and the standard metric battery |
| 4 | Closed-form / lightweight probabilistic classifiers | Modern shrinkage LDA/QDA, high-dimensional covariance regularization, naive-Bayes variants — do they exist in-window at good venues? |
| 5 | Implementation and artifact feasibility | Official code, license, maintenance status, dataset size and license, consumer-GPU cost, maintained libraries |
| 6 | Critical and opposing evidence | Documented failure modes; studies where cheap deterministic baselines match or beat Bayesian methods |

## 4. Queries actually issued

Term sets handed to the three Wave-1 retrieval subagents (each was told to adapt freely and add year
qualifiers). These are the issued sets, not a claim about every string each agent typed.

Bayesian deep / post-hoc: `post-hoc Laplace approximation image classification 2025`,
`last laplace deep ensembles 2024 2025`, `Bayesian neural network efficient variational inference
2025 computer vision`, `evidential deep learning classification 2025`, `calibration Bayesian deep
learning critique 2025`, `are Bayesian neural networks worth it benchmark 2024`, `uncertainty
quantification image classification failure modes 2025`.

Generative density / closed-form: `normalizing flow classifier image 2025`, `class conditional
density estimation Bayes classifier 2024`, `frozen CLIP features probabilistic classifier head Bayes
2025`, `Gaussian mixture classifier deep features recognition 2025`, `regularized LDA high
dimensional image classification 2024 2025`, `shrinkage QDA image recognition 2025`, `Bayes decision
rule classifier computer vision 2026`, `generative classifier versus discriminative image benchmark
2025`.

Uncertainty / feasibility: `open-set recognition 2025 Bayesian decision theory`,
`out-of-distribution detection survey 2025`, `Dirichlet evidential classification image 2025`,
`selective prediction abstention calibration 2025`, `laplace-torch library 2025`,
`distributional-pytorch DPU evidential library`, `deep-od toolbox OOD detection`, `CIFAR-10 dataset
size license`, `Tiny-ImageNet classes images`, `MedMNIST 2025 datasets`, `training time single
consumer GPU Bayesian deep learning 2025`.

Wave-2 gap-fill targets: F-EDL method equations and Table 1; hyper-opinion EDL internals; the
Hyper-EDL venue conflict; the EDL "Mirage" critique with numbers; wall-clock and epoch counts;
ProbaCLIP full text through a non-403 route; the von Mises-Fisher CLIP paper through seven routes; an
in-window top-venue closed-form density classifier; ICLA numbers; `laplace-torch` capability matrix;
Bayesian-LoRA family.

## 5. Coverage accounting

- Wave 1: 3 retrieval subagents, 8 WebSearch + 5–6 WebFetch each (two agents hit their fetch ceiling;
  one reported blocked pages explicitly).
- Wave 2: 2 gap-fill subagents, 6 WebSearch + 6 WebFetch each.
- Verification: 1 citation-verification subagent, 8 claim–source pairs.
- Distinct records in `stage_01_screening_ledger.csv`: **25** (4 recommended-tier, 6 reserve,
  7 rejected, 3 critical-evidence, 3 tooling, 2 foundational, plus 1 systematic negative finding).
  Papers and their repositories are counted once; mirrors were deduplicated.
- This is a screening sweep, not a PRISMA-compliant systematic review, and no "absolute latest"
  claim is made.

## 6. Verification outcomes that changed a claim

Every correction below came from the Phase 3.1 spot-check, not from the first pass. They are kept
visible because the final report's component (1) is "how AI finds the algorithm".

| Ledger id | Claim before verification | Verified outcome |
| --- | --- | --- |
| B-P05 | "arXiv page states no venue" | **ICLR 2024 confirmed** (ICLR virtual poster 19273; arXiv Comments names ICLR 2024). Exact title is "…Composite **Classification** Uncertainty". Code `Hugo101/HyperEvidentialNN`, no license |
| B-P01 | README covers "CIFAR-10/100, MNIST, SVHN, TinyImageNet" as training sets | **PARTIAL** — `main.py --ID_dataset` accepts only MNIST, CIFAR-10, CIFAR-100; SVHN and Tiny-ImageNet appear in `datasets.py` as OOD sets. License absence confirmed (`license: null`, LICENSE 404) |
| B-P03 | "AMISE-optimal bandwidth" | **UNCONFIRMED** — released code tunes bandwidth by ANLL/MANLL over iterations; the AMISE wording could not be read. Do not assert it |
| B-P08 | "EDL's epistemic uncertainty never vanishes even with infinite data" | **Reattributed** — that result belongs to the Bengs et al. line, which this paper sharpens. NeurIPS 2024 venue confirmed via proceedings hash `c3177be2…`; the "distillation baselines win" sentence rests on one summarised fetch and needs a verbatim re-read of v3 before any number is printed |
| B-P11 | "a page titled CIFAR-10 in the docs" | **Naming corrected** — the WideResNet-16-4 / `last_layer` / `kron` example is the *Calibration* tutorial. MIT licence and v0.2.3 uploaded 2026-10-03 confirmed. No memory or runtime figure exists anywhere in the docs |
| B-P02 | CIFAR-10 far-OOD AUROC 92.50 | **SUPPORTED** with a caveat: "trained from scratch" is inferred from the setup wording, not verbatim; 100 epochs on CIFAR-10/100 and 90 on ImageNet-200 are verbatim |
| B-P16 | "scored by negative log-likelihood" | ECCV 2024 venue and anomaly-detection framing confirmed; the NLL-scoring wording was not returned verbatim |

Unreachable and therefore excluded rather than guessed: the von Mises-Fisher CLIP paper
(OpenReview `KyeyEFPxJX`, seven routes blocked), the hyper-opinion EDL PDF (unreadable binary twice),
the MDPI HTML page (403; substituted by the DOAJ record and the code repository), and the ACM CSUR
OOD survey (403).

## 7. Artifact inventory for Stage 01

| Absolute path | Content |
| --- | --- |
| `D:\claude\workspace\CISC3024 AI Assignment 2\research_records\stage_01_search_protocol.md` | This file |
| `D:\claude\workspace\CISC3024 AI Assignment 2\research_records\stage_01_screening_ledger.csv` | 25 screened records with dispositions, verification corrections and source URLs |
| `D:\claude\workspace\CISC3024 AI Assignment 2\research_records\stage_01_candidate_scoring.md` | Scored shortlist and recommendation for supervisor approval (Gate 1) |
| `D:\claude\workspace\CISC3024 AI Assignment 2\DEEP_RESEARCH_BAYESIAN_ALGORITHMS.md` | Full citation-backed research report (4,261 words, 42 bibliography entries) |
| `D:\claude\workspace\CISC3024 AI Assignment 2\research_records\retrieval_cache\` | Raw evidence fetched by the subagents: `CVPR2025.html`, `ICCV2025.html`, `WACV2025.html`, `bmvc2025.html`, `ecva.html` (the proceedings listings behind the negative sweep in §6, so the title counts are re-checkable), plus `icla.html`, `icla_text.txt`, `laplace_docs.html`, `lap_curv.html`, `lap_index.md`, `mdpi.txt`, `mdpi_math13223615.pdf` (ProbaCLIP article), and two small scratch notes `_d.md`, `_t.md`, `_ss_blla.json` |

The `retrieval_cache/` folder is working evidence for audit only. It is excluded from any submission
package and from the public repository, both because it contains third-party material and because it
adds 35 MB.

## 8. Known limits of this stage

1. Wall-clock and GPU-hours are **not stated** in any of the candidate papers; the epoch counts were
   verified for ICLA only. All timing figures in the shortlist are therefore labelled as estimates
   until the approved pilot measures them.
2. F-EDL's epoch and optimizer settings sit in its Appendix E, which was not returned by any fetch.
   If F-EDL is chosen, its schedule must be set from our own validation curve rather than copied.
3. The evidential family's headline numbers are reported as AUPR in the table we read, with no ECE;
   calibration metrics would have to be computed by us.
4. The negative finding in B-P25 is an absence of evidence produced by title sweeps of four
   proceedings lists plus arXiv filters. It is strong screening evidence, not proof that no such
   paper exists.
