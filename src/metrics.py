"""Evaluation metrics for probabilistic classification, out-of-distribution detection and
selective prediction.

Project : CISC3024 AI Assignment #2 (Pattern Recognition, University of Macau)
Author  : AI-generated implementation (Qoder). The student wrote no code.

Why this file exists
--------------------
The reference implementation of the algorithm under study reports error-detection AUROC/AUPR and
Brier score but **not** expected calibration error, so the calibration comparison our hypothesis H2'
needs has to be measured here. Everything is implemented from first definitions and then cross-checked
against scikit-learn for the two metrics scikit-learn also provides (ROC AUC and average precision).

All functions take NumPy arrays unless stated otherwise. ``probs`` is always ``(N, K)`` row-stochastic.
"""

from __future__ import annotations

import numpy as np
from sklearn import metrics as skm

EPS = 1e-12


# --------------------------------------------------------------------------------------------
# Proper scoring rules
# --------------------------------------------------------------------------------------------
def negative_log_likelihood(probs: np.ndarray, labels: np.ndarray) -> float:
    """Mean categorical negative log-likelihood. Lower is better."""
    p = np.clip(probs[np.arange(len(labels)), labels], EPS, 1.0)
    return float(-np.log(p).mean())


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Multiclass Brier score: mean over samples of the squared distance to the one-hot target."""
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(labels)), labels] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


# --------------------------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------------------------
def _confidence_and_correctness(probs: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(float)
    return conf, correct


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> float:
    """Equal-width ECE (Guo et al., 2017 style): weighted mean of |accuracy - confidence| per bin."""
    conf, correct = _confidence_and_correctness(probs, labels)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi) if lo > 0 else conf <= hi
        if not mask.any():
            continue
        ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def adaptive_expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> float:
    """Adaptive ECE: equal-mass bins, so empty and sparse confidence regions cannot dominate."""
    conf, correct = _confidence_and_correctness(probs, labels)
    order = np.argsort(conf)
    chunks = np.array_split(order, n_bins)
    aece = 0.0
    for idx in chunks:
        if idx.size == 0:
            continue
        aece += (idx.size / len(conf)) * abs(correct[idx].mean() - conf[idx].mean())
    return float(aece)


def classwise_expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> float:
    """Classwise ECE: average of per-class equal-width ECE, which exposes miscalibration that the
    overall average hides when one class dominates the confidence distribution."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    per_class = []
    for k in range(probs.shape[1]):
        conf = probs[:, k]
        correct = (labels == k).astype(float)
        ece = 0.0
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (conf > lo) & (conf <= hi) if lo > 0 else conf <= hi
            if not mask.any():
                continue
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
        per_class.append(ece)
    return float(np.mean(per_class))


def reliability_bins(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> list[dict[str, float]]:
    """Per-bin statistics for the reliability diagram in the report and the web page."""
    conf, correct = _confidence_and_correctness(probs, labels)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi) if lo > 0 else conf <= hi
        out.append(
            {
                "bin_lo": float(lo),
                "bin_hi": float(hi),
                "count": int(mask.sum()),
                "confidence": float(conf[mask].mean()) if mask.any() else float("nan"),
                "accuracy": float(correct[mask].mean()) if mask.any() else float("nan"),
            }
        )
    return out


# --------------------------------------------------------------------------------------------
# Discrimination of correct vs incorrect predictions (the metric family the paper reports)
# --------------------------------------------------------------------------------------------
def error_detection_scores(
    probs: np.ndarray, labels: np.ndarray, uncertainty: np.ndarray
) -> dict[str, float]:
    """AUROC / AUPR for how well an uncertainty score separates wrong from right predictions.

    ``uncertainty`` is higher-for-more-uncertain; a wrong prediction is the positive class, so the
    score is used with a sign flip (larger uncertainty should imply larger error risk).
    """
    correct = (probs.argmax(axis=1) == labels).astype(int)
    wrong = 1 - correct
    unc = np.nan_to_num(uncertainty, nan=0.0, posinf=np.finfo(np.float64).max)
    return {
        "error_auroc": float(skm.roc_auc_score(wrong, unc)),
        "error_aupr": float(skm.average_precision_score(wrong, unc)),
    }


# --------------------------------------------------------------------------------------------
# Out-of-distribution detection
# --------------------------------------------------------------------------------------------
def ood_detection_scores(
    id_unc: np.ndarray, ood_unc: np.ndarray
) -> dict[str, float]:
    """AUROC, AUPR and FPR@95 %TPR for OOD detection from an uncertainty score.

    Convention: in-distribution labelled 0, out-of-distribution labelled 1, and a *larger* score must
    mean more out-of-distribution.
    """
    id_unc = np.nan_to_num(id_unc, nan=0.0)
    ood_unc = np.nan_to_num(ood_unc, nan=0.0)
    truth = np.concatenate([np.zeros(len(id_unc)), np.ones(len(ood_unc))])
    score = np.concatenate([id_unc, ood_unc])
    fpr, tpr, _ = skm.roc_curve(truth, score)
    # FPR at the first threshold reaching 95 % true-positive rate.
    idx = int(np.searchsorted(tpr, 0.95, side="left"))
    fpr95 = float(fpr[min(idx, len(fpr) - 1)])
    return {
        "auroc": float(skm.roc_auc_score(truth, score)),
        "aupr": float(skm.average_precision_score(truth, score)),
        "fpr95": fpr95,
    }


# --------------------------------------------------------------------------------------------
# Selective prediction
# --------------------------------------------------------------------------------------------
def risk_coverage_curve(
    probs: np.ndarray, labels: np.ndarray, n_points: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """Coverage ``r`` versus risk, abstaining on the most uncertain samples first.

    Returns ``(coverage, risk)`` arrays of length ``n_points``; risk is the error rate among the
    retained fraction.
    """
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels)
    order = np.argsort(-conf)  # keep the most confident first
    correct_sorted = correct[order]
    coverages, risks = [], []
    for i in np.linspace(1, len(correct_sorted), n_points).astype(int):
        coverages.append(i / len(correct_sorted))
        risks.append(1.0 - correct_sorted[:i].mean())
    return np.asarray(coverages), np.asarray(risks)


def selective_prediction_metrics(
    probs: np.ndarray, labels: np.ndarray, n_points: int = 100
) -> dict[str, float]:
    """AURC, coverage at 5 % risk, and risk at 80 % coverage."""
    cov, risk = risk_coverage_curve(probs, labels, n_points)
    aurc = float(np.trapezoid(risk, cov) / (cov[-1] - cov[0] + EPS))
    at_risk5 = cov[risk <= 0.05]
    at_cov80 = risk[cov >= 0.80]
    return {
        "aurc": aurc,
        "cov_at_risk5": float(at_risk5.max()) if at_risk5.size else 0.0,
        "risk_at_cov80": float(at_cov80.min()) if at_cov80.size else float(risk[-1]),
    }


# --------------------------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------------------------
def classification_metrics(probs: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Top-1 accuracy, top-2 accuracy and macro-F1."""
    top1 = float((probs.argmax(axis=1) == labels).mean())
    k = min(2, probs.shape[1])
    top2_idx = np.argsort(-probs, axis=1)[:, :k]
    top2 = float(np.mean([labels[i] in top2_idx[i] for i in range(len(labels))]))
    f1 = float(skm.f1_score(labels, probs.argmax(axis=1), average="macro", zero_division=0))
    return {"top1": top1, "top2": top2, "macro_f1": f1}


def full_report(
    probs: np.ndarray,
    labels: np.ndarray,
    epistemic: np.ndarray | None = None,
    aleatoric: np.ndarray | None = None,
) -> dict[str, float]:
    """One-call metric battery for a single configuration.

    ``epistemic`` and ``aleatoric`` are optional per-sample scores; when supplied, the OOD-free
    error-detection metrics are added.
    """
    out: dict[str, float] = {}
    out.update(classification_metrics(probs, labels))
    out["nll"] = negative_log_likelihood(probs, labels)
    out["brier"] = brier_score(probs, labels)
    out["ece"] = expected_calibration_error(probs, labels)
    out["adaptive_ece"] = adaptive_expected_calibration_error(probs, labels)
    out["classwise_ece"] = classwise_expected_calibration_error(probs, labels)
    out.update(selective_prediction_metrics(probs, labels))
    if epistemic is not None:
        out.update({f"err_{k}": v for k, v in error_detection_scores(probs, labels, epistemic).items()})
        out["epistemic_mean"] = float(np.mean(epistemic))
    if aleatoric is not None:
        out.update({f"ale_{k}": v for k, v in error_detection_scores(probs, labels, aleatoric).items()})
        out["aleatoric_mean"] = float(np.mean(aleatoric))
    return out
