# CISC3024 AI Assignment #2 — Uncertainty Estimation by Flexible Evidential Deep Learning

Pattern Recognition, Department of Computer Science, University of Macau, Fall 2026.

**Live demo (GitHub Pages):** <https://0weixiong0.github.io/CISC3024-AI-Assignment-2/>
**Report:** `CISC3024_AI_Assignment_2_Report.pdf`

An AI agent (Qoder) searched the literature, selected a Bayesian classification algorithm, designed
and implemented the experiments, built the web interface, and wrote this repository. **The student
wrote no code**, as the assignment brief requires, and supervised every stage. The dialogue record
and the stage-by-stage feasibility reports are described in the report; the code, comments and
documentation here are in English.

---

## What was done

The assignment asks for a recent classification algorithm based on probability or Bayes theory,
applied to computer vision and pattern recognition, reproduced on local hardware.

| Stage | Outcome |
| --- | --- |
| Search | 25 candidate records screened on five axes; two shortlisted |
| Algorithm | **F-EDL** — *Uncertainty Estimation by Flexible Evidential Deep Learning*, NeurIPS 2025 (arXiv:2510.18322), with **ICLA** (WACV 2025, arXiv:2312.10464) as the in-experiment Bayesian comparator |
| Design | 5 configurations, 3 seeds, one frozen split, CIFAR-10 with SVHN and Tiny-ImageNet as out-of-distribution controls |
| Implementation | ~4.6k lines of Python, 19 tests, three of them against independent authorities |
| Experiments | 2.7 h full matrix on an RTX 4060 Laptop GPU (8 GB), plus curvature and Lipschitz-budget ablations |
| Interface | A browser bench running both trained networks in WebGPU/WASM, and a ten-entry measurement register |

Headline result: the cross-entropy baseline reaches **91.19 %** top-1 against **90.77 %** for F-EDL,
while F-EDL and ICLA provide an epistemic term the baseline cannot express — error-detection AUROC
0.9076 and 0.9057 against an undefined quantity for the baseline. Temperature scaling remains the
strongest calibration method at this scale (ECE 0.0080). Details, including the two findings that
contradicted the plan's expectation, are in the report.

## Repository layout

```
src/
  prepare_data.py        download CIFAR-10, SVHN, Tiny-ImageNet
  data.py                the frozen split (seed 20261006, 5 % validation) and loaders
  fd_distribution.py     flexible-Dirichlet moments, closed-form Bayes risk, uncertainty split
  models.py              spectral-normalised VGG-16-BN and ResNet-18, evidential and softmax heads
  losses.py              the F-EDL objective
  train.py               AMP training loop with early stopping on validation NLL
  laplace_icla.py        last-layer Laplace with identity or Kronecker curvature
  calibrate.py           temperature and Dirichlet (vector) scaling
  metrics.py             NLL, Brier, ECE family, AUROC/AUPR/FPR95, risk-coverage
  evaluate.py            predictive collection for in-distribution and OOD splits
  run_matrix.py          the 5-configuration x 3-seed experiment and its ablations
  export_webapp_data.py  the register's JSON and the browser data bundle
  export_onnx.py         the two ONNX graphs, with a parity gate against PyTorch
  export_test_wall.py    all 10,000 test images and their per-configuration posteriors
  make_figures.py        the report's figures
tests/                   19 tests: python -m pytest tests -q
webapp/                  the published site (see below)
report_assets/           figures
research_records/        stage reports, screening ledger, extracted brief
```

## Reproducing

```bash
pip install -r requirements.txt
python src/prepare_data.py
python -m pytest tests -q                 # 19 passed
python src/run_matrix.py --seeds 3        # ~2.7 h on an 8 GB GPU
python src/export_webapp_data.py --results "runs/full_*/matrix_results.json"
python src/export_onnx.py
python src/export_test_wall.py
python src/make_figures.py
```

`--seeds` takes either a run count (`3`) or an explicit comma-separated list
(`20261006,20261007,20261008`).

## The web interface

`webapp/` is a static site: no build step, no framework, no chart library, no server-side inference.

- **`index.html` — the bench.** Both trained networks run in the browser tab from the ONNX graphs in
  `webapp/model/`. Draw a 32×32 specimen, upload a photograph, or pick any of the 10,000 test images
  from the class-filtered, uncertainty-sortable picker. The baseline and the evidential network answer
  the same image side by side, and the evidential one refuses to answer when its total uncertainty
  passes the 5 %-risk line. All five configurations are recomputed from the exported parameters, so
  the page and the report cannot disagree.
- **`register.html` — the measurements.** Ten entries: the five-configuration reduction with all 36
  recorded quantities, reliability, risk-coverage, out-of-distribution controls, uncertainty
  decomposition, per-image evidence anatomy, seed spread, ablations, post-hoc instrument readings, and
  measured cost with provenance and limits.

`register.html` opens directly from disk. The bench needs the models over HTTP:

```bash
python -m http.server 8801 --directory webapp
```

Quantities the page does not have for a configuration — the epistemic term of a deterministic model,
the error-detection AUROC of a constant score — are drawn struck rather than as zero, and the
`none`-curvature ablation is labelled as reproducing the main run because both settings take the same
code branch.

## Data and third-party provenance

- **CIFAR-10, SVHN, Tiny-ImageNet** are third-party datasets. CIFAR-10's official page states no
  licence; the raw archives are therefore not redistributed here and `src/prepare_data.py` fetches
  them from their published locations. The `gh-pages` branch carries the 10,000 test thumbnails used
  by the bench, following the precedent set by Assignment 1.
- **F-EDL** (`github.com/TaeseongYoon/F-EDL`) declares no licence. Its mathematics was recovered from
  the paper's method section and appendix and re-implemented from scratch, with the consultation
  recorded as decision D-1(b) in the report; no source file was copied.
- **laplace-torch** (Apache 2.0) is used only as an independent cross-check of the Kronecker
  curvature in `tests/test_laplace_icla.py`, and is not required to run the experiments.
- **Archivo** and **JetBrains Mono** are variable fonts under the SIL Open Font License 1.1, subset
  and self-hosted in `webapp/fonts/` with their licence texts.
- No licence is claimed over the third-party material above, and none is granted by this repository.

## Verification artefacts

| File | What it records |
| --- | --- |
| `webapp/model/verification.json` | ONNX-versus-PyTorch parity on real test images and on blank, scribbled and noise inputs |
| `webapp/data/instruments.json` | head weights, fitted temperature and Dirichlet vector, Laplace MAP and posterior standard deviations, abstention thresholds |
| `webapp/data/test-index.json` | per-image posteriors for all 10,000 test images, from the same cache the report tables were aggregated from |
| `research_records/` | the stage reports, the 25-record screening ledger, and the extracted assignment brief |
