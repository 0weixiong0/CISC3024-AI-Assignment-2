# Stage 00 — Assignment #2 Preflight, Supervision Contract and Environment Verification

Project: CISC3024 Pattern Recognition, AI Assignment #2 (University of Macau, Fall 2026)
Workspace: `D:\claude\workspace\CISC3024 AI Assignment 2`
Date: 2026-10-05
Author: AI researcher (Qoder); supervisor: assignment owner (research director role)
Status: Preflight complete. Authoritative brief read and archived; supervision guidance written to
persistent memory; local compute re-verified; reusable assets from AI Assignment #1 inventoried.
**No candidate algorithm has been searched, selected, implemented or trained yet.** Literature
screening starts only after the supervisor answers the Gate-0 clarifications in Section 9.

This is an internal working-stage report in English. It is not the assessed final report and does
not follow `Report Template.docx`.

---

## 1. Task framing

AI Assignment #2 asks the student to locate a **recent classification or recognition algorithm
grounded in probability or Bayes theory** for computer vision and pattern recognition, and to have
an AI tool or agent perform all of the work: algorithm searching, programming and report writing.
The student is explicitly forbidden from writing code. The graded artifact is a Word or PDF report,
submitted through UMMoodle and screened by Turnitin, that documents six components, including a
public webpage link to the source code.

The delivered workflow therefore has to produce two things at once: (a) a working, measurable
Bayesian classifier experiment, and (b) an auditable process record showing how the AI searched,
designed, implemented, tested, corrected and reported.

## 2. Supervision contract (from the supervisor's opening prompt, U001)

| # | Contract item | Operational consequence |
| --- | --- | --- |
| 1 | Five outline stages: search → design/plan → staged implementation → HTML visualization → report | Each stage ends in a supervisor decision, not an automatic advance |
| 2 | Approval gates | G1 algorithm shortlist; G2 algorithm + experiment plan; G3 feasibility at each implementation scale; G4 backend-ready before frontend; G5 report structure before drafting |
| 3 | Chinese conversation, English artifacts | Chat replies in Chinese; code, comments, logs, figures, stage reports and final report in English |
| 4 | Full dialogue record, translated | `research_records/conversation_log.md` keeps original supervisor prompts plus complete English translations |
| 5 | Stage summaries anytime, with absolute paths | Stage files are written incrementally, not only at stage end; every created file is reported by full path |
| 6 | Skills/extensions must be found per stage and approved before download, install or use | Section 8 proposes a per-stage skill plan for approval |
| 7 | External software needs (e.g. GitHub) to be raised with the supervisor | Section 7 item 5: public repository is required by the brief and needs the supervisor's account action |
| 8 | Deadline: finish before 2026-10-12 | Section 6 schedule with one buffer day |
| 9 | `/grilling` may be used on the supervisor | Held in reserve; Gate-0 questions in Section 9 are the lighter-weight substitute for now |
| 10 | `AIassign2.pdf` wording outranks the supervisor's paraphrase | Differences reported first — Section 3 |
| 11 | AI Assignment #1 is the structural baseline | Reused conventions listed in Section 5 |

Persistent memory records created this session:

- `C:\Users\weixiong\.qoder\projects\D--claude-workspace-CISC3024-AI-Assignment-2\memory\project-cisc3024-ai-assignment-2.md`
- `C:\Users\weixiong\.qoder\memory\feedback-ai-assignment-supervision-workflow.md`
- Indexes updated: `C:\Users\weixiong\.qoder\projects\D--claude-workspace-CISC3024-AI-Assignment-2\memory\MEMORY.md`,
  `C:\Users\weixiong\.qoder\memory\MEMORY.md`

## 3. Authoritative requirements versus the supervisor's outline

Source: `AIassign2.pdf`, 1 page; archived extraction at
`D:\claude\workspace\CISC3024 AI Assignment 2\research_records\AIassign2_extracted.txt`.
The page contains no embedded raster images; the text layer was extracted with PyMuPDF and the page
was additionally rendered at 150 dpi and visually inspected, so the wording below is confirmed
against both the text layer and the rendered page.

Verbatim requirements: *"AI Assignment # 2 aims to find a recent classification/recognition
algorithm based on probability or Bayes theory for computer vision and pattern recognition. You are
expected to use an AI tool or agent to do everything for you, including, but not limited to,
algorithm searching, programming, and even report writing. A report in Word or PDF format should be
submitted online via UMMoodle next week and will go through a Turnitin similarity check. Your report
should contain detailed steps on how you did this assignment, including (1) how AI finds the
algorithm, (2) algorithm description, (3) how AI implements the algorithm, (4) experiment settings
and results, (5) what you have learnt from this AI assignment, and (6) a webpage link of your source
codes. Note: You are NOT allowed to write any code by yourself!"*

Differences found between the PDF and the supervisor's outline, reported here first per contract
item 10:

| # | Topic | PDF wording | Supervisor outline | Handling decision |
| --- | --- | --- | --- | --- |
| D1 | Recency | "**a recent** classification/recognition algorithm" | Recency appears only as one scoring dimension ("新颖程度") | Treat recency as a hard filter (2024–2026 window), not a tie-breaker |
| D2 | Is a network required? | "algorithm based on probability or Bayes theory" — no network required | Outline says "该网络", i.e. "this network", and asks to swap the neural network for a Bayesian algorithm | Allow non-neural Bayesian classifiers and Bayesian deep models alike; state the admissible space explicitly in the shortlist |
| D3 | Source-code webpage | Component (6) requires a **webpage link** to the source code | Not mentioned in the outline | A public GitHub repository (plus optional Pages demo) becomes a required deliverable, not optional |
| D4 | Authorship rule | "You are **NOT** allowed to write any code by yourself!" | Implied by "AI 完成全流程" | Keep AI authorship evidence; supervisor edits no source |
| D5 | Submission channel | UMMoodle, then Turnitin similarity check | Not mentioned | Add a submission-readiness step; no similarity-percentage promises can be made in advance |
| D6 | Deadline | "next week", no date in the PDF | Before 2026-10-12 | Plan to 2026-10-11 with 2026-10-12 as buffer; supervisor to confirm the exact UMMoodle cutoff and time zone |
| D7 | Training vocabulary | Silent | Outline speaks of training sets, fine-tuning and overfitting control | Map NN terms onto Bayesian ones — see Section 4 |
| D8 | Report template rules | Silent | Cover unchanged, identical TOC/heading formatting, APA 7 | Template constraints are additive requirements, kept in force |

## 4. "Does a Bayesian algorithm need training?" — the mapping used from now on

The supervisor flagged this as an open question. The answer depends on which Bayesian family is
chosen, and the mapping below is what the experiment plan will use so the outline's stage names stay
meaningful:

| Outline (neural-network) term | Bayesian-algorithm equivalent |
| --- | --- |
| Training set | Data used to estimate likelihood parameters and/or the prior; for posterior-fitting methods, the same split that SGD sees |
| Training | Two regimes: **closed-form estimation** (maximum-likelihood / MAP counts, means and covariances — no epochs, seconds to run) versus **iterative posterior fitting** (variational inference, Monte-Carlo dropout, Bayesian neural networks, EM — gradient steps or iterations, hours) |
| Fine-tuning from pretrained weights | Only meaningful for the iterative regime; a pretrained probabilistic encoder can be re-estimated on the target classes |
| Validation set | Model/prior selection: hyper-parameters, smoothing strength, covariance regularizer, number of components or MC samples |
| Test set | Frozen, never used for fitting or selection; report accuracy plus calibrated probabilistic metrics |
| Overfitting control | Conjugate/MAP priors, Laplace or additive smoothing, shrinkage of covariance matrices, low-rank or diagonal covariance, evidence lower bound monitoring, weight-space priors, early stopping on validation NLL |
| Loss | Not only cross-entropy: negative log-likelihood / ELBO, plus **calibration** metrics that a Bayesian method is expected to win on (ECE, Brier score, NLL) and uncertainty quality (risk-coverage / AUROC of selective prediction) |

Consequence for grading: a closed-form-only Bayesian classifier would be trivially cheap to run but
weak on "recent" and on novelty; an iterative Bayesian deep model is recent and novel but reintroduces
training cost. The shortlist must therefore span both regimes so the supervisor can choose the
trade-off knowingly.

## 5. Reusable assets from AI Assignment #1

Baseline folder: `D:\claude\workspace\CISC3024 AI Assignment 1` (git branch pushed to
`https://github.com/0weixiong0/CISC3024-AI-Assignment-1.git`, GitHub Pages demo live).

Directly reusable:

1. `research_records/` stage-log conventions — `stage_00_preflight.md`, `stage_01_search_protocol.md`,
   `stage_01_screening_ledger.csv`, `stage_02_experiment_design.md`, `stage_03_experiment_log.md`,
   `stage_04_visualization_log.md`, `stage_05_report_outline.md`, `conversation_log.md`.
2. Screening method: three parallel retrieval agents + a citation-verification agent, target
   ~20–30 screened papers → 6–8 closer reads → 3–4 final candidates, with a CSV disposition ledger.
3. `Report Template.docx` is byte-identical in both folders (200,767 bytes, same timestamp), so the
   A1 report build pipeline (`research_records/report_assets/build_report.py`, `make_figures.py`) is
   a known-good starting point.
4. `Agent_coding.pptx` is likewise identical (63,902 bytes) — see Section 12.
5. Local EuroSAT copy, ~234 MB, at `D:\claude\workspace\CISC3024 AI Assignment 1\data\EuroSAT\`
   (27,000 images, 10 classes) — available with zero download time if the supervisor approves reuse.

Deliberate differences for Assignment #2: the algorithm family changes from a deterministic CNN
(StarNet-S2) to a probability/Bayes classifier, so the metric set, the visualization story
(posterior, prior, uncertainty instead of Grad-CAM feature maps) and the "what was learnt"
reflections must not be copies of A1. Turnitin exposure is a real constraint here.

## 6. Schedule to the 2026-10-12 deadline

Today is 2026-10-05. Working days are counted as calendar days because the supervisor is available
in-session; each row ends at a gate.

| Day | Date | Gate | Output the supervisor reviews |
| --- | --- | --- | --- |
| D0 | 2026-10-05 (done) | G0 | This preflight + Gate-0 clarifications (Section 9) + skill plan (Section 8) |
| D1 | 2026-10-06 | G1 | Literature search protocol, screening ledger, 3–4 candidate Bayesian algorithms scored on the five requested dimensions |
| D2 | 2026-10-07 | G2 | Algorithm description, mathematical formulation, experiment design (splits, metrics, budget, expected results), milestone plan with MVP |
| D3 | 2026-10-08 | G3a | Small-scale pilot: smoke test, 10–20 % subset run, runtime and memory measurements, feasibility report |
| D4 | 2026-10-09 | G3b | Full-scale run + regularization/ablation comparison + frozen-test metrics |
| D5 | 2026-10-10 | G4 | HTML visualization interface (posterior inspection, intermediate-state checks) + public repository and webpage link |
| D6 | 2026-10-11 | G5 | Report structure approval, then full English draft in Word with APA 7 references |
| D7 | 2026-10-12 | — | Buffer: layout check, figure re-render, PDF export, submission readiness |

Risk note: D1 and D4 are the two days most likely to slip (search depth, and full-run runtime).
The mitigation is that closed-form Bayesian estimation runs in seconds on CPU, so the critical path
does not depend on GPU availability — see Section 7.

## 7. Local environment verification (measured 2026-10-05)

| Item | Measured value | Method |
| --- | --- | --- |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB | `nvidia-smi --query-gpu=...` (driver 592.00) |
| RAM | 31.6 GiB | `psutil.virtual_memory()` |
| Python | 3.12.10 | `sys.version` |
| PyTorch | 2.7.1+cu128, `cuda.is_available() == True`, device resolved to the RTX 4060 | import probe |
| Free storage on D: | ~53 GiB (154 GiB used of 206 GiB) | `df -h /d` |
| git | 2.52.0.windows.1 | `git --version` |
| GitHub CLI (`gh`) | **Not found in PATH** | `which gh` |
| Document tooling | PyMuPDF, python-docx present; PDF text extraction and 150-dpi rendering verified | used in this stage |

Feasibility reading: 8 GiB VRAM caps batch size for any Bayesian deep model with sampled weights;
MCMC-style inference is impractical at image scale, whereas variational inference, Monte-Carlo
dropout, Laplace approximation and closed-form density estimation are all realistic. CPU-only
classical Bayesian estimation (naive Bayes, LDA/QDA with shrinkage, Gaussian mixture, discriminative
Bayes) will finish in minutes, which is why the schedule does not put GPU on the critical path.
No candidate has been benchmarked locally yet — the numbers above are machine facts, not
algorithm-cost claims.

## 8. Per-stage skill and tool plan (awaiting approval)

Already installed and readable in this session; **none installed or run this turn**:

| Stage | Skill / tool | Status | Why |
| --- | --- | --- | --- |
| G1 search | `deep-research` | installed (used in A1) — **needs approval to run** | Multi-agent retrieval with source verification and citation-backed output |
| G1 search | `research` | installed | Lighter fallback if the supervisor wants a narrower pass |
| G2 design | `grilling` | installed, optional | Stress-tests the chosen design before implementation; the supervisor offered it explicitly |
| G3 implementation | `tdd`, `code-review`, `diagnosing-bugs` | installed | Small failing-test-first harness for the estimator, then review and debug loops |
| G4 frontend | `impeccable` (supervisor's standing preference), `ui-ux-pro-max`, `design-system` | installed | HTML visualization build and polish |
| G5 report | `slides` / document tooling via python-docx | installed | Word output against the template; APA 7 references |
| Any | `qoder-find-extensions` | installed | Only if a capability gap appears; new installs return for approval first |

Recommendation: approve `deep-research` for G1 now (it is already on disk, so no download is needed),
and hold `grilling` unless the Gate-1 choice ends up contested.

## 9. Gate-0 clarifications needed before literature screening

These three answers change what gets searched, so they are asked before retrieval rather than after.

1. **Algorithm family scope** — should screening keep the widest admissible space (closed-form
   probabilistic classifiers, generative density models, Bayesian deep networks / variational
   inference, and uncertainty-focused recognition methods), or restrict it? A wide net maximizes
   novelty options but lengthens the shortlist; a narrow net reaches a decision faster.
2. **Dataset** — reuse the already-downloaded EuroSAT copy (zero download time, but the same dataset
   as A1, which affects how distinct the two submissions look), or switch to a different image
   benchmark (extra download and preparation time, better separation from A1)?
3. **Source-code webpage** — confirm the supervisor will create and keep public a GitHub repository
   under the existing account, since `gh` is absent and repository creation plus Pages activation may
   need their browser action. A1 precedent shows this works over HTTPS.

## 10. Six required report components → planned evidence

| PDF component | Evidence to be produced | Stage file |
| --- | --- | --- |
| (1) How AI finds the algorithm | Search protocol, query log, screening ledger with dispositions, supervisor selection prompts | `stage_01_*` |
| (2) Algorithm description | Paper-verified formulation, prior/likelihood/posterior structure, declared adaptations | `stage_02_*` |
| (3) How AI implements the algorithm | Implementation steps, failing tests, corrections, supervisor interventions | `stage_03_*` |
| (4) Experiment settings and results | Splits, seeds, configs, runtime/memory measurements, metrics incl. calibration, ablations | `stage_03_*` |
| (5) What was learnt | Actual observed lessons and supervisor reflections, not invented feelings | `stage_03_*`, `stage_06_*` |
| (6) Webpage link of source codes | Public repository URL + runnable visualization page + reproduction instructions | `stage_04_*`, `stage_07_*` |

## 11. Artifact inventory created in Stage 00

| Absolute path | Content |
| --- | --- |
| `D:\claude\workspace\CISC3024 AI Assignment 2\research_records\AIassign2_extracted.txt` | Verbatim text layer of `AIassign2.pdf` (1 page), extraction evidence |
| `D:\claude\workspace\CISC3024 AI Assignment 2\research_records\stage_00_preflight.md` | This report |
| `D:\claude\workspace\CISC3024 AI Assignment 2\research_records\conversation_log.md` | Supervisor prompt U001 with full English translation, plus assistant reply record A001 |
| `C:\Users\weixiong\.qoder\projects\D--claude-workspace-CISC3024-AI-Assignment-2\memory\project-cisc3024-ai-assignment-2.md` | Persistent project brief memory |
| `C:\Users\weixiong\.qoder\memory\feedback-ai-assignment-supervision-workflow.md` | Persistent collaboration-contract memory |

## 12. Agent_coding.pptx decomposition actually used

Source: `Agent_coding.pptx`, 15 slides, text extracted from the slide XML.

The deck defines the agent-coding loop as **Decomposition → Task Description → Preparation (Plan) →
Execution → Supervision → Verification → (Result) Presentation → Evaluation → Iteration**, with
these specifics: Task Description is "the most important step" and must state Goal, Scope,
Constraints, Input, Output and Functional Requirements; descriptions must be clear and precise
because "AI might make mistakes or misunderstand your intention" (slide 8 contrasts two different
"LeNet" readings: 28×28 input with ReLU and max pooling versus 32×32 with scaled tanh and average
pooling, and asks which one, and whether 1998 LeNet-5 is meant); efficient description uses existing
references, web search, paper/documentation, expected outputs, and an agent-restates / human-refines
loop; Preparation may be AI-generated and human-refined, and should set milestones and a minimum
viable product. Slides 12–13 show a Codex demo of MNIST classification with LeNet as reference only.

Applied to this assignment: the Gate-2 plan must restate the algorithm specification in the
supervisor's terms before implementation (the slide-8 lesson), and the MVP milestone is the D3 pilot
rather than the full run.

## 13. Next actions after approval

1. Run the G1 literature search with `deep-research` under the protocol written at Gate 1.
2. Score 3–4 surviving candidates on the five requested dimensions: algorithm complexity, local
   hardware requirement, implementation difficulty, novelty, and how directly it demonstrates the
   assignment's expectations.
3. Deliver the shortlist with a recommendation and explicit risks for supervisor approval (G1).
