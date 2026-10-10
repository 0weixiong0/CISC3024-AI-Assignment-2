# Stage 02b — M0 Mathematical Recovery, Reference Consultation Record and Plan Corrections

Project: CISC3024 AI Assignment #2 · Date: 2026-10-07 · Language: English (working-stage report)
Milestone: M0 of `stage_02_design_and_plan.md` §9.
Authorisation: supervisor decision **D-1(b)** — read the unlicensed official repository **only** to
establish the missing mathematical definition and hyperparameter values, record which file answered
which question, write our implementation from scratch, and disclose the consultation in the report.

Status: M0 is **complete and successful**. Every item that Stage 01 recorded as "not recovered" is now
recovered, and four earlier claims turned out to be wrong. The wrong claims are retracted here rather
than quietly fixed.

---

## 1. What was read, and what each file answered

Files fetched on 2026-10-07 into
`D:\claude\workspace\CISC3024 AI Assignment 2\research_records\reference_consulted\`
(machine-readable audit trail in `manifest.json`, which stores the URL, byte count and timestamp of
every fetch, plus the authorisation text):

| File consulted | Question it answered | What was taken from it |
| --- | --- | --- |
| `fedl_uq.py` (8,449 B) | What *is* the flexible Dirichlet? Its mean, variance and uncertainty decomposition | The three formulas in §2 — mathematical facts, re-derived and re-implemented by us |
| `fedl_train.py` (4,430 B) | What is the actual training objective, and is the expectation sampled or closed-form | The loss structure in §3 — closed form, no sampling |
| `fedl_main.py` (3,805 B) | Appendix E hyperparameters (epochs, optimiser, learning rate, schedule, batch, validation fraction) | The values in §4, re-stated as our own configuration with the source noted |
| `fedl_models.py` (14,688 B) | Which backbone belongs to which dataset, and how the three heads are wired | The architecture assignment in §5 |
| `arxiv_v2_abs.html`, `arxiv_v2_html.html` | Whether an appendix version exists publicly | The 943 KB HTML was retrieved; the hyperparameters we needed were already in `main.py`, so the appendix was not required |

Nothing from these files was copied into `src/`. Our implementation is written from the formulas, not
from their code, and will be reviewed against the plan's file layout in `stage_02` §8.

## 2. Recovered: the flexible Dirichlet mean and variance

With per-class evidence-derived concentration `α_k > 0` (note: `α_k = exp(g_α(z)_k)`, so it is *not*
the classical `e + 1`), a modal probability vector `p` with `Σ_k p_k = 1`, and a scalar precision
`τ = softplus(g_τ(z)) > 0`, define `α₀ = Σ_k α_k`. Then

```
mu  = (alpha + tau * p) / (alpha0 + tau)

var = mu * (1 - mu) / (alpha0 + tau + 1)
      + tau**2 * p * (1 - p) / ((alpha0 + tau) * (alpha0 + tau + 1))
```

`mu` is the predictive mean and `var` the predictive variance of the flexible Dirichlet, both in closed
form. The interpretation that makes this a *Bayesian* object rather than a trick is immediate from the
first term: when `τ → 0`, `mu → α/α₀`, which is exactly the classical Dirichlet mean used by standard
evidential deep learning; when `α → 0`, `mu → p`, the ordinary network output. So `τ` is a learned
per-sample mixing weight between "trust the accumulated evidence" and "trust the point estimate", and
the variance is the sum of a Dirichlet term and a `p`-dispersion term. This is the mechanism the paper's
title calls *flexible*, and it is now fully specified for our implementation.

Uncertainty decomposition (also recovered, and the quantities our webpage will visualise):

```
EU = sum_k var_k            # epistemic  (model) uncertainty
TU = 1 - sum_k mu_k**2      # total       uncertainty (Gini impurity of the mean)
AU = TU - EU                # aleatoric   (data) uncertainty
```

## 3. Recovered: the objective is a closed-form Bayes risk, not a Monte-Carlo estimate

Stage 01 recorded the objective as `L = E_{π~FD}‖y − π‖² + y − p‖²` and flagged "the sampling estimator
is not recovered". The resolution is better than expected: **there is no sampling**. For a squared loss
under a distribution with mean `mu` and covariance diagonal `var`,

```
E_{pi~FD} ||y - pi||^2  =  ||y - mu||^2  +  sum_k var_k
```

so the objective is `loss = Σ(y_onehot − mu)² + Σ var + Σ(y_onehot − p)²`, computed analytically. The
first two terms are the exact Bayes risk under squared loss; the third pins the modal probability to the
label and acts as the regulariser. Consequences for the plan: training cost is that of an ordinary
network with three heads, inference is **one** forward pass, and no random-number reproducibility issue
arises in the loss.

Two implementation details that the recovered source pins down and that we must match or deliberately
change, with the choice written down:

1. The reference sums the squared errors over the whole batch rather than averaging, so its loss value
   scales with batch size. We will use the batch **mean** (equivalent up to a constant factor on the
   gradient at fixed batch size, and it makes the learning rate transfer sensibly across our subset
   runs). Declared as an adaptation.
2. The reference hard-codes `num_classes = 10` inside the training loop, which is correct for CIFAR-10
   and MNIST but inconsistent with its own CIFAR-100 path. We derive the class count from the dataset.
   Recorded as a defect in the consulted source, not copied.

## 4. Recovered: the Appendix E hyperparameters

From `fedl_main.py` argument defaults (the values the released code actually runs with):

| Hyperparameter | Value |
| --- | --- |
| Optimiser | Adam |
| Learning rate | 5e-4 |
| Weight decay | 0 |
| LR schedule | `StepLR(step_size=20, gamma=0.1)` |
| Epochs | 100 (early stopping, see below) |
| Batch size | 64 |
| Validation fraction of train | 0.05 |
| Dropout rate | 0 |
| Head MLP width / depth | hidden 256, 2 layers (Dropout→Linear→ReLU→LayerNorm→Linear) |
| Mixed precision | enabled (`GradScaler` + `autocast`) |
| Early stopping | patience 3, evaluated every 10 epochs, monitored on **validation accuracy** |

Two notes. First, the released `main.py` has **no** `--spect_norm` flag even though the README's example
commands pass one, and `FEDL.__init__` sets `spect_norm = True` unconditionally — so spectral
normalisation of the backbone is always on in the shipped code. My Stage 01 plan proposed an ablation
"with and without spectral norm"; that ablation is still valid but must be described as a deviation we
introduce, because the reference does not offer the switch. Second, early stopping on validation
*accuracy* rather than on NLL contradicts our plan §6 rule 1 (model selection on validation NLL). We
keep our own rule and record the difference, because for a probabilistic model NLL is the quantity being
claimed.

## 5. Correction: the backbone for CIFAR-10 is VGG-16-BN, not ResNet-18

`fedl_models.py` assigns, per in-distribution dataset:

| Dataset | Backbone (from the consulted source) | Feature dim into the heads |
| --- | --- | --- |
| MNIST | small conv-net, spectral-normalised | 576 |
| **CIFAR-10** | **VGG-16-BN**, all convs and classifier linear layers spectral-normalised (`k_lipschitz = 1`) | 512 |
| CIFAR-100 | ResNet-18 (spectral-normalised, LeakyReLU, average-pool shortcut) | 512 |

`g_alpha` **is** the backbone's own classifier head, while `g_p` and `g_tau` are two separate MLPs on the
same 512-d pooled feature. So `α` and `p` do not come from three symmetric siblings: `α` comes from the
classification trunk, and `p`, `τ` from auxiliary MLPs.

**This invalidates two sentences in my earlier documents.** Stage 01 §C1 and Stage 02 §2.3 both said the
verified headline setting was "ResNet-18 on CIFAR-10". The verified fact is *VGG-16-BN on CIFAR-10* and
*ResNet-18 on CIFAR-100*; the paper text names both backbones and I combined them incorrectly. The
accuracy figure 91.19 ± 0.2 for CIFAR-10 therefore belongs to a spectral-normalised VGG-16-BN.

Design consequence, adopted: our CIFAR-10 configuration uses VGG-16-BN with spectral normalisation to
stay comparable with the published 91.19 %. VGG-16-BN at 32×32 is roughly 15 M parameters, which trains
comfortably inside 8 GB at batch 64 — but this is now a **measurable risk item for the M2 pilot**,
because spectral-norm power iteration adds per-step cost and ICLR-style timing claims do not exist for
this combination. If the pilot shows the VGG path is too slow for the schedule, the declared fallback is
ResNet-18 on CIFAR-10 with the mismatch stated in the report, and the 91.19 % comparison moved to the
CIFAR-100 column where ResNet-18 is the published backbone.

## 6. Correction: the "99.10 % confidence" reading was wrong; H2 is retracted

Stage 01 §C1 and Stage 02 §1 treated the published table's 99.10 figure as a *mean confidence* that
contradicts 91.19 % accuracy, and built hypothesis H2 on that apparent over-confidence. The consulted
`uq.py` shows what that column actually is: `conf_calibration` returns AUROC and AUPR for how well the
aleatoric (AU) and epistemic (EU) scores separate **correct from incorrect test predictions**. A value
near 99 % is therefore "the uncertainty score almost perfectly predicts whether the model was right",
which is a *positive* result, not an over-confidence symptom.

**H2 as originally written is withdrawn.** Replacement, stated before the experiment runs:

- **H2′.** Because the published table reports error-detection AUROC/AUPR and Brier but **not ECE**, the
  calibration of F-EDL under a standard ECE measurement is an open question in the literature. We will
  compute ECE (plus its adaptive and classwise variants) ourselves for all five configurations, and
  report whether F-EDL's confidence is better or worse calibrated than temperature-scaled
  cross-entropy. Whatever the direction, this fills a metric the paper does not report.

H1 and H3 stand unchanged, except that H1's accuracy band must now be compared against a
spectral-normalised VGG-16-BN rather than a ResNet-18.

## 7. Other defects observed in the consulted source (recorded, not copied)

Reported factually with line references, because "the AI read the reference critically" is exactly the
process evidence report component (3) needs:

1. `train.py` line 37: `num_classes = 10` hard-coded inside the per-batch loop, inconsistent with the
   CIFAR-100 path that `models.py` provides.
2. `uq.py` line 211: `dist_shift_detection_fedl_mnist(model, testloader, fix_tau, fix_p, device)` is
   called with five arguments but defined at line 133 to take three, so the MNIST corruption path cannot
   run as shipped.
3. `uq.py` lines 151–153: `auc_list[0].append(...)` and `aupr_list[1].append(...)` index a two-element
   tuple by position, so AUROC values are appended to the AUROC list only when the tuple slot happens to
   match — the corruption-loop bookkeeping is inconsistent.
4. None of these three are reachable from `main.py`, which only calls `train`, `eval`,
   `conf_calibration` and `ood_detection`. So the headline pipeline is unaffected; the defects are in the
   unused distribution-shift utilities.

## 8. Net effect on the plan

| Plan item | Before M0 | After M0 |
| --- | --- | --- |
| CIFAR-10 backbone | ResNet-18 (our choice, "paper uses ResNet-18") | **VGG-16-BN + spectral norm**, matching the reference; ResNet-18 kept as the declared fallback |
| F-EDL training cost | unknown, sampling assumed | closed-form moments, one forward pass, Adam 5e-4, StepLR-20, 100 epochs, batch 64 |
| H2 | over-confidence gap predicted | **withdrawn**, replaced by H2′ (measure ECE, which the paper does not report) |
| Validation split | 5,000 images (10 %) | 2,500 images (0.05) to match the reference default, test 10,000 unchanged, seed `20261006` |
| Spectral-norm ablation | assumed switchable in the reference | ours to introduce; the reference always enables it |
| Model-selection metric | validation NLL (ours) vs validation accuracy (theirs) | keep validation NLL, record the deviation and the reason |
| Loss reduction | unspecified | batch **mean**, declared as an adaptation from their batch sum |

Nothing in the approved scope changed: same five configurations, same two backbone training runs, same
metrics, same milestones. The dataset download and the dependency installs remain as authorised in
Stage 02 §11.

## 9. Compliance statement for the disclosure section of the report

Under decision D-1(b), the consultation was limited to establishing (i) the flexible-Dirichlet mean,
variance and uncertainty decomposition, (ii) the training objective and whether it is sampled,
(iii) the hyperparameter values, and (iv) the backbone-to-dataset assignment. All four are mathematical
and factual statements. No source code was copied into this project; the files listed in §1 are stored
outside `src/`, in `research_records/reference_consulted/`, and will be excluded from the public
repository. This record is the basis for the Acknowledgements disclosure.
