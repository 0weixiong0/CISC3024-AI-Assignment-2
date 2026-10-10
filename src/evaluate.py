"""Run a trained model over a loader and turn its outputs into the full metric battery.

Project : CISC3024 AI Assignment #2 (Pattern Recognition, University of Macau)
Author  : AI-generated implementation (Qoder). The student wrote no code.

Both model families expose ``predictive(x)``, so one collection routine serves the evidential model,
the cross-entropy baseline and the post-hoc Laplace wrapper. For the deterministic models the
predictive variance is zero, which is the correct statement of "no epistemic term available" rather
than a missing value.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

import metrics as M


@torch.no_grad()
def collect_predictive(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_mc: int = 0,
) -> dict[str, np.ndarray]:
    """Return predictive means, variances, uncertainty scores, features and labels as NumPy arrays.

    ``n_mc`` larger than zero additionally concatenates the Monte-Carlo / posterior samples that
    ``predictive`` produces, shape ``(N, n_mc, K)``; it is only used by the visualisation layer.
    """
    model.eval()
    means, vars_, epis, aless, feats, labels, logits, samples = [], [], [], [], [], [], [], []

    for batch in loader:
        x, y = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
        out = model.predictive(x, n_mc=n_mc)

        means.append(out["mean"].float().cpu().numpy())
        vars_.append(out["variance"].float().cpu().numpy())
        epis.append(out["epistemic"].float().cpu().numpy())
        aless.append(out["aleatoric"].float().cpu().numpy())
        feats.append(out.get("features", out["mean"]).float().cpu().numpy())
        labels.append(y.cpu().numpy())
        if "logits" in out:
            logits.append(out["logits"].float().cpu().numpy())
        if n_mc > 0 and "samples" in out:
            samples.append(out["samples"].float().cpu().numpy())

    result: dict[str, np.ndarray] = {
        "mean": np.concatenate(means),
        "variance": np.concatenate(vars_),
        "epistemic": np.concatenate(epis),
        "aleatoric": np.concatenate(aless),
        "features": np.concatenate(feats),
        "labels": np.concatenate(labels),
    }
    if logits:
        result["logits"] = np.concatenate(logits)
    if samples:
        result["samples"] = np.concatenate(samples, axis=0)
    return result


def score_in_distribution(collected: dict[str, np.ndarray]) -> dict[str, float]:
    """Accuracy, NLL, Brier, the three ECE variants and the selective-prediction family."""
    return M.full_report(
        collected["mean"], collected["labels"],
        epistemic=collected["epistemic"], aleatoric=collected["aleatoric"],
    )


def score_ood(id_collected: dict[str, np.ndarray],
              ood_collected: dict[str, dict[str, np.ndarray]]) -> dict[str, float]:
    """AUROC / AUPR / FPR95 for every OOD control, under both uncertainty estimators.

    Both the epistemic score and the aleatoric score are evaluated, because the evidential model
    supplies two and the deterministic baselines supply only the total-uncertainty proxy; reporting
    them side by side is what makes the comparison honest.
    """
    out: dict[str, float] = {}
    id_ep = id_collected["epistemic"]
    id_al = id_collected["aleatoric"]
    id_to = id_collected["variance"].sum(1) * 0 + (1.0 - (id_collected["mean"] ** 2).sum(1))
    for name, coll in ood_collected.items():
        ep = coll["epistemic"]
        al = coll["aleatoric"]
        to = 1.0 - (coll["mean"] ** 2).sum(1)
        for label, s_id, s_ood in (("epistemic", id_ep, ep), ("aleatoric", id_al, al),
                                   ("total", id_to, to)):
            for k, v in M.ood_detection_scores(s_id, s_ood).items():
                out[f"{name}_{label}_{k}"] = v
    return out


def max_softmax_confidence(collected: dict[str, np.ndarray]) -> np.ndarray:
    """The energy-free baseline uncertainty score, used to check the deterministic models fairly."""
    return 1.0 - collected["mean"].max(axis=1)


def mean_confidence_gap(collected: dict[str, np.ndarray]) -> float:
    """mean(max prob) minus top-1 accuracy: the raw over-confidence number, reported for completeness.

    Recorded because an earlier draft of this project mis-read the paper's 99.10 figure as this
    quantity; computing it explicitly means the distinction is measured rather than argued.
    """
    conf = collected["mean"].max(axis=1).mean()
    acc = (collected["mean"].argmax(axis=1) == collected["labels"]).mean()
    return float(conf - acc)
