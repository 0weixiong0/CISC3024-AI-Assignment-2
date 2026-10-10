"""Collect the final per-image predictions and draw every figure used by the report.

Project : CISC3024 AI Assignment #2 (Pattern Recognition, University of Macau)
Author  : AI-generated implementation (Qoder). The student wrote no code.

Run after ``run_matrix.py`` has finished::

    python src/make_figures.py --results runs/full_*/matrix_results.json

Outputs
    analysis/test_predictions.npz    per-image mean/variance/epistemic/aleatoric, all configs
    analysis/summary.json            scalars the report quotes directly
    report_assets/figure_NN_*.png    300 dpi figures, English labels, print-ready

Design note: the figures are generated from the same code path that produced the reported numbers, so
a figure cannot silently disagree with its table. Curves that need per-image quantities are recomputed
here from the stored checkpoints rather than re-derived from aggregate statistics.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import gridspec
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data as D  # noqa: E402
from evaluate import collect_predictive  # noqa: E402
from metrics import (  # noqa: E402
    expected_calibration_error, ood_detection_scores, reliability_bins, risk_coverage_curve,
    selective_prediction_metrics,
)
from run_matrix import DEVICE, MODEL_NAMES, _head, build  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ANALYSIS = ROOT / "analysis"
ASSETS = ROOT / "report_assets"

CLASS_NAMES = np.array(["airplane", "automobile", "bird", "cat", "deer", "dog", "frog",
                        "horse", "ship", "truck"])

CONFIG_LABEL = {
    "B1_ce": "B1  cross-entropy",
    "B2_ce_temperature": "B2  CE + temperature",
    "A1_fedl": "A1  F-EDL (flexible Dirichlet)",
    "A2_fedl_dirichlet": "A2  F-EDL + Dirichlet scaling",
    "A3_laplace_icla": "A3  CE + last-layer Laplace (ICLA)",
}
CONFIG_COLOR = {
    "B1_ce": "#7f8c8d", "B2_ce_temperature": "#2980b9", "A1_fedl": "#c0392b",
    "A2_fedl_dirichlet": "#e67e22", "A3_laplace_icla": "#27ae60",
}

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 300, "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9, "legend.fontsize": 8, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False, "figure.autolayout": False,
})


# ================================================================================================
# Collection
# ================================================================================================
def load_models(payload: dict, out_root: Path):
    """Rebuild the B1 and A1 backbones for every seed from their stored checkpoints."""
    from calibrate import CalibratedModel, DirichletScaler, TemperatureScaler
    from laplace_icla import LaplaceWrapper, LastLayerLaplace, cache_features

    per_seed = {}
    for seed in payload["seeds"]:
        b1_ckpt = torch.load(out_root / f"b1_ce_s{seed}" / "best.pt", map_location=DEVICE,
                             weights_only=False)
        b1 = build(__import__("train").Config(**b1_ckpt["config"])); b1.load_state_dict(b1_ckpt["model"])
        b1.eval()
        a1_ckpt = torch.load(out_root / f"a1_fedl_s{seed}" / "best.pt", map_location=DEVICE,
                             weights_only=False)
        a1 = build(__import__("train").Config(**a1_ckpt["config"])); a1.load_state_dict(a1_ckpt["model"])
        a1.eval()

        loaders = D.cifar10_loaders(batch_size=256, num_workers=0, pin_memory=False)
        phi_tr, y_tr = cache_features(b1, loaders["train"], DEVICE)
        val_phi, val_y = cache_features(b1, loaders["val"], DEVICE)
        a1_val = collect_predictive(a1, loaders["val"], DEVICE)
        temp = TemperatureScaler().fit(_head(b1)(val_phi), val_y)
        dscal = DirichletScaler().fit(
            torch.tensor(np.log(a1_val["mean"].clip(1e-8, 1.0)), device=DEVICE),
            torch.tensor(a1_val["labels"], device=DEVICE))
        lap = LastLayerLaplace(curvature=payload["curvature"], n_samples=200).fit(
            _head(b1), phi_tr, y_tr)
        models = {
            "B1_ce": b1, "B2_ce_temperature": CalibratedModel(b1, temp), "A1_fedl": a1,
            "A2_fedl_dirichlet": CalibratedModel(a1, dscal),
            "A3_laplace_icla": LaplaceWrapper(b1, lap),
        }
        per_seed[seed] = {"models": models, "loaders": loaders, "lambda": lap.lam,
                          "temperature": temp.temperature, "dirichlet": dscal.describe()}
        print(f"[figures] seed {seed} rebuilt: lambda={lap.lam:.4g} T={temp.temperature:.3f}",
              flush=True)
    return per_seed


def collect_all(per_seed: dict) -> dict:
    """Per-image predictive quantities on the test split and on both OOD controls."""
    store: dict[str, dict] = {}
    for seed, ctx in per_seed.items():
        loaders = ctx["loaders"]
        store[str(seed)] = {}
        for name, model in ctx["models"].items():
            sets = {"test": loaders["test"], "ood_near_svhn": None, "ood_far_tiny": None}
            ood = D.ood_loaders(batch_size=256, num_workers=0)
            sets["ood_near_svhn"] = ood.get("near_svhn")
            sets["ood_far_tiny"] = ood.get("far_tiny_imagenet")
            for split, loader in sets.items():
                if loader is None:
                    continue
                coll = collect_predictive(model, loader, DEVICE)
                for key in ("mean", "variance", "epistemic", "aleatoric", "labels"):
                    if key in coll:
                        store[str(seed)][f"{name}__{split}__{key}"] = coll[key]
                store[str(seed)][f"{name}__{split}__total"] = (
                    1.0 - (coll["mean"] ** 2).sum(1))
        print(f"[figures] seed {seed} predictions collected", flush=True)
    return store


# ================================================================================================
# Figures
# ================================================================================================
def _save(fig, name: str, suffix: str = "") -> Path:
    if suffix:
        stem, _, ext = name.partition(".")
        name = f"{stem}_{suffix}.{ext}"
    fig.savefig(ASSETS / name, bbox_inches="tight")
    plt.close(fig)
    print(f"[figures] wrote {ASSETS / name}", flush=True)
    return ASSETS / name


def fig_training_curves(out_root: Path, seeds: list[int]) -> Path:
    """Loss and validation NLL versus epoch for both backbones, with the early-stop point marked."""
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
    for name, colour, style in (("b1_ce", CONFIG_COLOR["B1_ce"], "-"),
                                ("a1_fedl", CONFIG_COLOR["A1_fedl"], "-")):
        for i, seed in enumerate(seeds):
            path = out_root / f"{name}_s{seed}" / "history.json"
            if not path.exists():
                continue
            h = json.loads(path.read_text(encoding="utf-8"))["history"]
            ep = [r["epoch"] for r in h]
            label = name.replace("_", " ") if i == 0 else None
            axes[0].plot(ep, [r["train_loss"] for r in h], style, color=colour, label=label)
            vn = [(r["epoch"], r["val_nll"]) for r in h if "val_nll" in r]
            if vn:
                axes[1].plot([a for a, _ in vn], [b for _, b in vn], style, color=colour, label=label)
                best = min(vn, key=lambda t: t[1])
                axes[1].scatter(*best, zorder=5, color=colour, marker="*", s=70)
    axes[0].set(xlabel="epoch", ylabel="training loss (batch mean)")
    axes[0].set_title("Training objective")
    axes[1].set(xlabel="epoch", ylabel="validation NLL")
    axes[1].set_title("Model-selection signal (star = best, then early stop)")
    axes[1].legend(frameon=False)
    fig.suptitle("Figure 3. Optimisation and validation NLL for the two trained backbones", y=1.04)
    return _save(fig, "figure_03_training_curves.png")


def fig_reliability(store: dict, seed: str) -> Path:
    """One reliability panel per configuration: the calibration story in a single image."""
    labels = np.asarray(store[seed]["A1_fedl__test__labels"])
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.1), sharex=True, sharey=True)
    for ax, name in zip(axes, MODEL_NAMES):
        probs = store[seed][f"{name}__test__mean"]
        bins = reliability_bins(probs, labels)
        conf = [b["confidence"] for b in bins if b["count"] > 0]
        acc = [b["accuracy"] for b in bins if b["count"] > 0]
        cnt = [b["count"] for b in bins if b["count"] > 0]
        width = [1.0 / 15 * 0.92] * len(conf)
        ax.bar(conf, np.asarray(acc) - np.asarray(conf), width=width, bottom=conf,
               color=CONFIG_COLOR[name], edgecolor="white", linewidth=0.5, label="|acc − conf|")
        ax.plot([0, 1], [0, 1], "k--", lw=0.9, label="perfect")
        ax.set_title(f"{CONFIG_LABEL[name]}\nECE = {expected_calibration_error(probs, labels):.4f}",
                     fontsize=8.5)
        ax.set_xlabel("confidence")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    axes[0].set_ylabel("accuracy")
    axes[-1].legend(frameon=False, loc="upper left")
    fig.suptitle("Figure 4. Reliability diagrams on the frozen test split "
                 f"(bar height = miscalibration; seed {seed})", y=1.06)
    return _save(fig, "figure_04_reliability.png", f"s{seed}")


def fig_risk_coverage(store: dict, seed: str) -> Path:
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    labels = np.asarray(store[seed]["A1_fedl__test__labels"])
    for name in MODEL_NAMES:
        probs = store[seed][f"{name}__test__mean"]
        cov, risk = risk_coverage_curve(probs, labels, n_points=200)
        m = selective_prediction_metrics(probs, labels)
        ax.plot(cov, risk, color=CONFIG_COLOR[name], lw=1.6,
                label=f"{CONFIG_LABEL[name]}  AURC {m['aurc']:.4f}")
    ax.set(xlabel="coverage (fraction kept)", ylabel="risk (error rate)")
    ax.set_title(f"Figure 5. Selective prediction, seed {seed}\n"
                 "(abstain on the most uncertain first)")
    ax.legend(frameon=False, fontsize=7.5)
    return _save(fig, "figure_05_risk_coverage.png", f"s{seed}")


def fig_ood(store: dict, seeds: list[str]) -> Path:
    """Near and far OOD AUROC per configuration, averaged over seeds with error bars."""
    controls = [c for c in ("ood_near_svhn", "ood_far_tiny")
                if f"A1_fedl__{c}__labels" in store[seeds[0]]]
    score_names = ("epistemic", "total")
    fig, axes = plt.subplots(1, len(controls), figsize=(5.0 * len(controls), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, control in zip(axes, controls):
        width, xs = 0.36, np.arange(len(MODEL_NAMES))
        for j, score in enumerate(score_names):
            means, stds = [], []
            for name in MODEL_NAMES:
                vals = []
                for seed in seeds:
                    id_u = store[seed][f"{name}__test__{score}"]
                    od_u = store[seed][f"{name}__{control}__{score}"]
                    if len(id_u) == 0 or np.allclose(id_u, od_u) or (id_u.std() == 0 and
                                                                     od_u.std() == 0):
                        vals.append(0.5)
                        continue
                    vals.append(ood_detection_scores(id_u, od_u)["auroc"])
                means.append(np.mean(vals)); stds.append(np.std(vals))
            ax.bar(xs + (j - 0.5) * width, means, width * 0.9, yerr=stds, capsize=2,
                   color=["#95a5a6", "#34495e"][j], label=f"{score} score")
        ax.axhline(0.5, color="k", ls=":", lw=0.8)
        ax.set_xticks(xs)
        ax.set_xticklabels([n.split("_")[0] + n.split("_")[1][:3] for n in MODEL_NAMES], fontsize=8)
        ax.set_title(f"{control.replace('ood_', '').replace('_', ' ')}")
        ax.set_ylim(0.4, 1.02)
    axes[0].set_ylabel("AUROC (OOD detected as more uncertain)")
    axes[-1].legend(frameon=False)
    fig.suptitle("Figure 6. Out-of-distribution detection, mean and spread over seeds", y=1.03)
    return _save(fig, "figure_06_ood_detection.png")


def fig_uncertainty_decomposition(store: dict, seed: str) -> Path:
    """Do the epistemic and aleatoric terms actually mean anything? Compare correct vs wrong,
    in-distribution vs out-of-distribution."""
    labels = np.asarray(store[seed]["A1_fedl__test__labels"])
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    for name in ("A1_fedl", "A3_laplace_icla"):
        e = store[seed][f"{name}__test__epistemic"]
        a = store[seed][f"{name}__test__aleatoric"]
        pred = store[seed][f"{name}__test__mean"].argmax(1)
        axes[0].hist(e[pred == labels], bins=40, alpha=0.55, density=True,
                     color=CONFIG_COLOR[name], label=f"{name.split('_')[0]} correct")
        axes[0].hist(e[pred != labels], bins=40, alpha=0.55, density=True, histtype="step",
                     lw=1.6, color=CONFIG_COLOR[name], label=f"{name.split('_')[0]} wrong")
        if f"{name}__ood_far_tiny__epistemic" in store[seed]:
            od = store[seed][f"{name}__ood_far_tiny__epistemic"]
            axes[1].hist(od, bins=40, alpha=0.5, density=True, color=CONFIG_COLOR[name],
                         histtype="step", lw=1.8, label=f"{name.split('_')[0]} far-OOD")
    axes[1].hist(store[seed]["A1_fedl__test__epistemic"], bins=40, alpha=0.45, density=True,
                 color="#c0392b", histtype="step", lw=1.2, label="ID (reference)")
    axes[0].set(xlabel="epistemic uncertainty $\\sum_k \\mathrm{Var}[\\pi_k]$", ylabel="density")
    axes[0].set_title("Epistemic score: correct vs wrong predictions")
    axes[1].set(xlabel="epistemic uncertainty", ylabel="density")
    axes[1].set_title("In-distribution vs far out-of-distribution")
    for ax in axes:
        ax.legend(frameon=False, fontsize=7.5)
    fig.suptitle(f"Figure 7. What the two Bayesian uncertainty terms separate (seed {seed})", y=1.04)
    return _save(fig, "figure_07_uncertainty_decomposition.png", f"s{seed}")


def fig_confusion(store: dict, seed: str, name: str = "A1_fedl") -> Path:
    labels = np.asarray(store[seed][f"{name}__test__labels"])
    pred = store[seed][f"{name}__test__mean"].argmax(1)
    cm = np.zeros((10, 10), dtype=int)
    for t, p in zip(labels, pred):
        cm[int(t), int(p)] += 1
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    im = ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(10), CLASS_NAMES, rotation=90, fontsize=7.5)
    ax.set_yticks(range(10), CLASS_NAMES, fontsize=7.5)
    for i in range(10):
        for j in range(10):
            if cm[i, j] > 0:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=6.5,
                        color="white" if cm[i, j] / max(cm[i].sum(), 1) > 0.5 else "black")
    ax.set(xlabel="predicted", ylabel="true")
    ax.set_title(f"Figure 8. Confusion matrix, {CONFIG_LABEL[name]} (seed {seed})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="row-normalised count")
    return _save(fig, "figure_08_confusion_matrix.png", f"s{seed}_{name.split('_')[0]}")


def fig_evidence_anatomy(store: dict, seed: str, n_show: int = 6) -> Path:
    """The figure that makes a Bayesian classifier legible: the predictive distribution, the
    posterior spread and the uncertainty split for a handful of contrasting test images.

    Only the images that ``export_webapp_data.py`` actually wrote to disk can be shown, so the
    selection is restricted to the indices present in ``webapp/data/samples.json``. Picking an
    arbitrary argmax over the whole test split would reference files that are not there.
    """
    meta_path = WEB_SAMPLES.parent / "samples.json"
    if not meta_path.exists():
        print("[figures] skipped figure 9: run src/export_webapp_data.py first", flush=True)
        return None

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    available = {int(r["index"]): r["id"] for r in meta["images"]}

    probs = store[seed]["A1_fedl__test__mean"]
    var = store[seed]["A1_fedl__test__variance"]
    epi = store[seed]["A1_fedl__test__epistemic"]
    ale = store[seed]["A1_fedl__test__aleatoric"]
    labels = np.asarray(store[seed]["A1_fedl__test__labels"])
    pred = probs.argmax(1)
    conf = probs.max(1)

    pool = np.array(sorted(available), dtype=int)
    if len(pool) < n_show:
        print(f"[figures] skipped figure 9: only {len(pool)} exported samples", flush=True)
        return None
    correct_mask = pred[pool] == labels[pool]
    pool_conf, pool_epi, pool_ale = conf[pool], epi[pool], ale[pool]

    def pick(scores, best=max):
        j = best(range(len(pool)), key=lambda i: scores[i])
        return int(pool[j])

    picks = []
    for scores, best in (
        (np.where(correct_mask, pool_conf, -1), max),        # confident and correct
        (np.where(correct_mask, -pool_conf, -1), max),       # hesitant but correct
        (np.where(~correct_mask, pool_conf, -1), max),       # confidently wrong
        (np.where(~correct_mask, pool_epi, -1), max),        # wrong and maximally epistemic
        (pool_epi, max),                                     # maximum epistemic overall
        (pool_ale, max),                                     # maximum aleatoric overall
    ):
        idx = pick(scores, best)
        if idx not in picks:
            picks.append(idx)
    picks = picks[:n_show]

    fig = plt.figure(figsize=(12.6, 2.35 * max(len(picks), 1)))
    gs = gridspec.GridSpec(len(picks), 3, width_ratios=[1.0, 2.3, 1.35],
                           hspace=0.72, wspace=0.34)
    for row, idx in enumerate(picks):
        ax_img = fig.add_subplot(gs[row, 0])
        ax_img.imshow(Image.open(WEB_SAMPLES / f"{available[idx]}.png"))
        ax_img.set_axis_off()
        truth, got = CLASS_NAMES[labels[idx]], CLASS_NAMES[pred[idx]]
        verdict = "correct" if labels[idx] == pred[idx] else "WRONG"
        ax_img.set_title(f"{truth} -> predicted {got}  ({verdict})", fontsize=8.5, pad=4)

        ax_bar = fig.add_subplot(gs[row, 1])
        ax_bar.barh(range(10), probs[idx], color=CONFIG_COLOR["A1_fedl"], alpha=0.8,
                    label="predictive mean $\\mu_k$")
        ax_bar.errorbar(probs[idx], range(10), xerr=np.sqrt(var[idx]), fmt="none",
                        ecolor="#2c3e50", elinewidth=0.9, capsize=2,
                        label="posterior std $\\sqrt{\\mathrm{Var}[\\pi_k]}$")
        ax_bar.set_yticks(range(10), CLASS_NAMES, fontsize=7)
        ax_bar.invert_yaxis(); ax_bar.set_xlim(0, 1)
        ax_bar.set_xlabel("class probability" if row == len(picks) - 1 else "", fontsize=8)
        if row == 0:
            ax_bar.set_title("Dirichlet predictive mean with posterior spread", fontsize=8.5,
                             pad=4)
            ax_bar.legend(frameon=False, fontsize=6.5, loc="lower right")

        ax_txt = fig.add_subplot(gs[row, 2]); ax_txt.set_axis_off()
        ax_txt.text(0, 1.0,
                    f"confidence   {conf[idx]:.3f}\n"
                    f"epistemic    {epi[idx]:.4f}\n"
                    f"aleatoric    {ale[idx]:.4f}\n"
                    f"total        {epi[idx] + ale[idx]:.4f}\n"
                    f"mean var/k   {var[idx].mean():.5f}",
                    family="monospace", fontsize=8, va="top")
        if row == 0:
            ax_txt.set_title("Uncertainty decomposition", fontsize=8.5, pad=4)
    fig.suptitle(f"Figure 9. Anatomy of {len(picks)} contrasting test predictions under F-EDL "
                 f"(seed {seed})", y=1.005)
    return _save(fig, "figure_09_evidence_anatomy.png", f"s{seed}")


def fig_lambda_evidence(out_root: Path, seeds: list[int]) -> Path:
    """The Bayesian evidence as a function of the prior precision, with the fitted optimum marked."""
    from laplace_icla import LastLayerLaplace, cache_features
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    loaders = D.cifar10_loaders(batch_size=256, num_workers=0, pin_memory=False)
    for i, seed in enumerate(seeds):
        ck = torch.load(out_root / f"b1_ce_s{seed}" / "best.pt", map_location=DEVICE,
                        weights_only=False)
        b1 = build(__import__("train").Config(**ck["config"])); b1.load_state_dict(ck["model"])
        b1.eval()
        phi, y = cache_features(b1, loaders["train"], DEVICE)
        lap = LastLayerLaplace(curvature="identity").fit(_head(b1), phi, y)
        grid = np.logspace(np.log10(lap.lam / 50), np.log10(lap.lam * 50), 120)
        ax.plot(grid, [lap.evidence(float(g)) for g in grid], color=CONFIG_COLOR["A3_laplace_icla"],
                alpha=0.55 + 0.45 * i / max(len(seeds) - 1, 1), lw=1.4,
                label=f"seed {seed}  $\\lambda^*$={lap.lam:.3g}")
        ax.axvline(lap.lam, color=CONFIG_COLOR["A3_laplace_icla"], ls=":", lw=0.9)
    ax.set_xscale("log")
    ax.set(xlabel="prior precision  $\\lambda$", ylabel="Laplace log-evidence")
    ax.set_title("Figure 10. Evidence maximisation selects the prior precision\n"
                 "(posterior N(W_MAP, lambda^-1 I), identity curvature)")
    ax.legend(frameon=False, fontsize=7.5)
    return _save(fig, "figure_10_evidence_lambda.png")


def fig_dataset_samples() -> Path:
    """CIFAR-10 classes plus the two OOD controls, so the reader can see what is being rejected."""
    from PIL import Image
    loaders = D.cifar10_loaders(batch_size=256, num_workers=0, pin_memory=False)
    test = loaders["test"].dataset
    rng = np.random.default_rng(20261006)
    fig, axes = plt.subplots(3, 10, figsize=(13, 4.3))
    for k in range(10):
        idxs = np.where(np.asarray(test.targets) == k)[0]
        pick = rng.choice(idxs, 3, replace=False)
        for row, i in enumerate(pick):
            img, _ = test[int(i)]
            axes[row, k].imshow(img.permute(1, 2, 0).numpy())
            axes[row, k].set_axis_off()
            if row == 2:
                axes[row, k].set_title(CLASS_NAMES[k], fontsize=7.5)
    fig.suptitle("Figure 2 (top). CIFAR-10 test images used by the classifier", y=0.99)
    return _save(fig, "figure_02a_cifar_samples.png")


def fig_ood_examples() -> Path:
    from PIL import Image
    ood = D.ood_loaders(batch_size=256, num_workers=0, limit=26)
    fig, axes = plt.subplots(2, 10, figsize=(13, 2.9))
    titles = {"near_svhn": "near-OOD: SVHN", "far_tiny_imagenet": "far-OOD: Tiny-ImageNet-200"}
    for row, (key, loader) in enumerate(ood.items()):
        mean = torch.tensor(D.CIFAR10_MEAN).view(3, 1, 1)
        std = torch.tensor(D.CIFAR10_STD).view(3, 1, 1)
        for col, (x, _) in enumerate(loader):
            if col >= 10:
                break
            axes[row, col].imshow(((x[col] * std + mean).permute(1, 2, 0).clamp(0, 1)).numpy())
            axes[row, col].set_axis_off()
        axes[row, 0].set_ylabel(titles.get(key, key), fontsize=8)
    fig.suptitle("Figure 2 (bottom). The two out-of-distribution control sets", y=1.0)
    return _save(fig, "figure_02b_ood_examples.png")


# ================================================================================================
WEB_SAMPLES = ROOT / "webapp" / "data" / "samples"


def load_store_from_cache(payload: dict) -> dict:
    """Rebuild the per-seed prediction store from the cache written by an earlier run."""
    cache = ANALYSIS / "test_predictions.npz"
    npz = np.load(cache)
    store: dict[str, dict] = {}
    for key in npz.files:
        seed, name = key.split("::", 1)
        store.setdefault(seed, {})[name] = npz[key]
    wanted = [str(s) for s in payload["seeds"]]
    if not all(s in store for s in wanted):
        raise FileNotFoundError(f"cache {cache} is missing seeds {wanted}")
    print(f"[figures] reusing cached predictions from {cache}", flush=True)
    return store


# ---------------------------------------------------------------- report figures, second pass
#
# These four answer questions the first pass could not: how the algorithm was found, what the
# architecture actually does, what the Lipschitz budget bought, and whether uncertainty separates
# right answers from wrong ones. All of them read files that already exist -- the screening ledger,
# the matrix results and the per-image cache -- so none of them can disagree with a table.

LEDGER = ROOT / "research_records" / "stage_01_screening_ledger.csv"
MAIN_RESULTS = ROOT / "runs" / "full_1791561662" / "matrix_results.json"
NOSN_RESULTS = ROOT / "runs" / "full_1791636055" / "matrix_results_identity_nosn.json"

DISPOSITION_ORDER = [
    ("recommended_primary", "recommended as primary"),
    ("recommended_comparator", "recommended as comparator"),
    ("tooling_selected", "selected as tooling"),
    ("critical_evidence", "critical evidence against"),
    ("negative_finding", "negative finding"),
    ("foundational_background", "foundational background"),
    ("reserve", "reserve"),
    ("reserve_alternative", "reserve alternative"),
    ("tooling_reserve", "tooling reserve"),
    ("hold", "held pending verification"),
    ("rejected", "rejected"),
]


def fig_search_funnel() -> Path | None:
    """Where the 25 screened records came from and how each was disposed of."""
    import csv

    if not LEDGER.exists():
        print(f"[figures] no ledger at {LEDGER}", file=sys.stderr)
        return None
    rows = list(csv.DictReader(LEDGER.open(encoding="utf-8")))
    disp = Counter(r["screening_disposition"] for r in rows)
    fam = Counter(r["family"] for r in rows)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.1))
    left, right = axes

    labels = [name for key, name in DISPOSITION_ORDER if key in disp]
    counts = [disp[key] for key, name in DISPOSITION_ORDER if key in disp]
    keep = {"recommended as primary", "recommended as comparator", "selected as tooling"}
    colours = ["#0d6ba6" if name in keep else ("#ab2a1e" if name == "rejected" else "#4a5765")
               for name in labels]
    y = np.arange(len(labels))
    left.barh(y, counts, color=colours, height=0.62)
    left.set_yticks(y, labels, fontsize=7.6)
    left.invert_yaxis()
    left.set_xlabel("records", fontsize=8.5)
    left.set_title(f"disposition of {len(rows)} screened records", fontsize=9.5)
    for yi, c in zip(y, counts):
        left.text(c + 0.08, yi, str(c), va="center", fontsize=7.6)
    left.set_xlim(0, max(counts) + 1.2)

    fams = sorted(fam.items(), key=lambda kv: (-kv[1], kv[0]))
    y2 = np.arange(len(fams))
    right.barh(y2, [n for _, n in fams], color="#33414f", height=0.62)
    right.set_yticks(y2, [k.replace("-", " ") for k, _ in fams], fontsize=7.6)
    right.invert_yaxis()
    right.set_xlabel("records", fontsize=8.5)
    right.set_title("probability families represented in the screen", fontsize=9.5)
    for axis in (left, right):
        axis.tick_params(labelsize=8)
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _save(fig, "figure_13_search_funnel.png")


def fig_architecture() -> Path:
    """The inference path both models share, and what each reduction adds on top of it."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(9.8, 4.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 56)
    ax.axis("off")

    def box(x, y, w, h, title, sub, colour="#111a24"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=0.8",
                                    linewidth=1.0, edgecolor=colour, facecolor="#f4f6f3"))
        ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", fontsize=8.2,
                color="#111a24")
        ax.text(x + w / 2, y + h * 0.26, sub, ha="center", va="center", fontsize=6.6,
                color="#4a5765")

    def arrow(x1, y1, x2, y2, colour="#111a24"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8,
                                     linewidth=0.9, color=colour))

    box(0.5, 24, 15, 11, "input", "32 x 32 x 3, normalised")
    box(19, 22, 18, 15, "VGG-16-BN trunk", "spectral normalisation\non every layer")
    box(40.5, 26, 9, 7, "z", "512-d")

    box(53, 43, 17, 10.5, "alpha head", "exp(g1(z))", "#0d6ba6")
    box(53, 30.5, 17, 10.5, "p head", "softmax(g2(z))", "#0d6ba6")
    box(53, 18, 17, 10.5, "tau head", "softplus(g3(z))", "#0d6ba6")
    box(53, 3, 17, 10.5, "last-layer Laplace", "ICLA, identity curvature", "#1c4fbe")

    box(73.5, 25, 14.5, 17, "flexible\nDirichlet", "moments mu, Var", "#0d6ba6")
    box(73.5, 4.5, 14.5, 11.5, "MC predictive", "mean over S draws", "#1c4fbe")

    box(91, 16, 8.5, 18, "report", "EU\nTU\nAU", "#111a24")

    arrow(15.5, 29.5, 18.8, 29.5)
    arrow(37, 29.5, 40.3, 29.5)
    arrow(49.5, 30.5, 52.8, 48)
    arrow(49.5, 29.5, 52.8, 35.7)
    arrow(49.5, 28.5, 52.8, 23)
    arrow(49.5, 27, 52.8, 8, colour="#1c4fbe")
    arrow(70, 48, 74.5, 38)
    arrow(70, 35.7, 73.3, 33.5)
    arrow(70, 23, 73.3, 28)
    arrow(70, 8, 73.3, 10)
    arrow(88, 33.5, 90.8, 27)
    arrow(88, 10, 90.8, 17)

    ax.text(0.5, 53.5, "shared by all five configurations", fontsize=7.4, color="#4a5765")
    ax.text(53, 55.2, "evidential reduction  A1, A2", fontsize=7.4, color="#0d6ba6")
    ax.text(53, 0.2, "Laplace reduction  A3        B1, B2 use the MAP softmax of the same trunk",
            fontsize=7.4, color="#1c4fbe")
    fig.tight_layout()
    return _save(fig, "figure_01_architecture.png")


def fig_spectral_norm_ablation() -> Path | None:
    """What the per-layer Lipschitz budget actually cost, on identical data and schedules."""
    if not (MAIN_RESULTS.exists() and NOSN_RESULTS.exists()):
        print("[figures] spectral-norm ablation results missing", file=sys.stderr)
        return None
    on = json.loads(MAIN_RESULTS.read_text(encoding="utf-8"))["aggregate"]
    off = json.loads(NOSN_RESULTS.read_text(encoding="utf-8"))["aggregate"]

    metrics = [("top1", "Top-1 accuracy", 4), ("nll", "negative log-likelihood", 4),
               ("ece", "expected calibration error", 4)]
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.3), sharex=True)
    short = {m: m.split("_")[0] for m in MODEL_NAMES}
    x = np.arange(len(MODEL_NAMES))
    for axis, (key, title, dp) in zip(axes, metrics):
        a = np.array([on[m][key]["mean"] for m in MODEL_NAMES])
        b = np.array([off[m][key]["mean"] for m in MODEL_NAMES])
        ea = np.array([on[m][key]["std"] for m in MODEL_NAMES])
        eb = np.array([off[m][key]["std"] for m in MODEL_NAMES])
        axis.bar(x - 0.19, a, width=0.36, yerr=ea, capsize=2, color="#33414f",
                 label="with spectral norm")
        axis.bar(x + 0.19, b, width=0.36, yerr=eb, capsize=2, color="#c8cfc9",
                 edgecolor="#33414f", hatch="///", label="without")
        for xi, (va, vb) in enumerate(zip(a, b)):
            axis.text(xi + 0.19, vb, f"{vb - va:+.{dp - 1}f}" if key != "top1"
                      else f"{(vb - va) * 100:+.2f}", ha="center",
                      va="bottom" if vb >= va else "top", fontsize=6.6, color="#ab2a1e")
        axis.set_title(title, fontsize=9)
        axis.tick_params(labelsize=8)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_xticks(x, [short[m] for m in MODEL_NAMES])
    axes[0].set_xlabel("configuration", fontsize=8.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return _save(fig, "figure_14_spectral_norm_ablation.png")


def fig_calibration_families(store: dict, seed: str) -> Path:
    """Uncalibrated, temperature-scaled, and Dirichlet-scaled, on the four scores that matter."""
    agg = json.loads(MAIN_RESULTS.read_text(encoding="utf-8"))["aggregate"]
    picks = ["B1_ce", "B2_ce_temperature", "A1_fedl", "A2_fedl_dirichlet"]
    metrics = [("ece", "ECE"), ("adaptive_ece", "adaptive ECE"), ("brier", "Brier"),
               ("nll", "NLL")]
    fig, axes = plt.subplots(1, 4, figsize=(9.8, 3.0), sharey=False)
    labels = ["B1\nCE", "B2\n+ temp", "A1\nF-EDL", "A2\n+ Dirichlet"]
    for axis, (key, title) in zip(axes, metrics):
        means = np.array([agg[m][key]["mean"] for m in picks])
        stds = np.array([agg[m][key]["std"] for m in picks])
        best = int(np.argmin(means))
        axis.bar(range(len(picks)), means, yerr=stds, capsize=2.5, width=0.62,
                 color=["#0d6ba6" if i == best else "#c8cfc9" for i in range(len(picks))],
                 edgecolor="#33414f", linewidth=0.7)
        for i, v in enumerate(means):
            axis.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=6.8)
        axis.set_title(title, fontsize=9)
        axis.set_xticks(range(len(picks)), labels, fontsize=7.4)
        axis.tick_params(labelsize=8)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("lower is better", fontsize=8)
    fig.tight_layout()
    return _save(fig, "figure_15_calibration_families.png", suffix=seed)


def fig_uncertainty_vs_correctness(store: dict, seed: str) -> Path:
    """Do the uncertainty quantities separate the images the model gets right from those it does not?

    Step histograms on a logarithmic count axis, because on a linear one every panel is a single spike
    at zero and the separation the report claims is invisible. Each panel carries the AUROC of its own
    score against the binary error event, computed from the same arrays being plotted.
    """
    models = ["B1_ce", "A1_fedl", "A3_laplace_icla"]
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.2))
    labels = np.asarray(store[seed]["A1_fedl__test__labels"])
    for axis, model in zip(axes, models):
        d = store[seed]
        total = np.asarray(d[f"{model}__test__total"], dtype=np.float64)
        correct = np.asarray(d[f"{model}__test__mean"]).argmax(1) == labels
        bins = np.linspace(0.0, 0.85, 34)
        axis.hist(total[correct], bins=bins, histtype="step", linewidth=1.1, color="#33414f",
                  label=f"correct ({correct.sum():,})")
        axis.hist(total[~correct], bins=bins, histtype="step", linewidth=1.1, color="#ab2a1e",
                  label=f"wrong ({(~correct).sum():,})")
        # the same abstention rule the browser uses: retain the most confident until risk passes 5 %
        order = np.argsort(total)
        risk = 1.0 - np.cumsum(correct[order]) / np.arange(1, correct.size + 1)
        kept = np.where(risk <= 0.05)[0]
        thr = float(total[order][kept[-1] + 1]) if kept.size else float(total[order][0])
        axis.axvline(thr, linestyle="--", linewidth=1.0, color="#111a24")
        axis.text(thr, 0.62, f" abstain\n beyond {thr:.3f}", fontsize=6.6,
                  va="center", transform=axis.get_xaxis_transform())
        axis.set_yscale("log")
        axis.set_xlim(0, 0.85)
        axis.set_ylim(0.8, 2e4)
        axis.set_title(f"{model.split('_')[0]}  error AUROC, this seed: "
                     f"{ood_auc(total, ~correct):.4f}", fontsize=9)
        axis.set_xlabel("total uncertainty", fontsize=8)
        axis.tick_params(labelsize=8)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("test images", fontsize=8)
    axes[0].legend(fontsize=7, frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.86))
    fig.tight_layout()
    return _save(fig, "figure_16_uncertainty_vs_correctness.png", suffix=seed)


def ood_auc(score: np.ndarray, positive: np.ndarray) -> float:
    """Ranking quality of ``score`` against the boolean ``positive``, by rank sum.

    Written here rather than imported so the number printed in the title is computed from exactly the
    arrays the panel draws, not from the aggregate table.
    """
    n_pos = int(positive.sum())
    n_neg = int(positive.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(score.size, dtype=np.float64)
    ranks[order] = np.arange(1, score.size + 1)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))

def main(results_glob: str, reuse: bool = False) -> int:
    files = sorted(glob.glob(results_glob))
    if not files:
        print(f"[figures] no results matching {results_glob!r}", file=sys.stderr)
        return 1
    path = Path(max(files, key=os.path.getmtime))
    payload = json.loads(path.read_text(encoding="utf-8"))
    out_root = Path(payload["out_dir"])
    ANALYSIS.mkdir(exist_ok=True); ASSETS.mkdir(exist_ok=True)

    seeds = [str(s) for s in payload["seeds"]]
    cache = ANALYSIS / "test_predictions.npz"
    if reuse and cache.exists():
        store = load_store_from_cache(payload)
    else:
        per_seed = load_models(payload, out_root)
        store = collect_all(per_seed)
        flat = {f"{seed}::{k}": v for seed, d in store.items() for k, v in d.items()}
        np.savez_compressed(cache, **flat)
        print(f"[figures] wrote {cache}", flush=True)

    made = [fig_dataset_samples(), fig_ood_examples(), fig_training_curves(out_root, payload["seeds"])]
    for seed in seeds:
        made += [fig_reliability(store, seed), fig_risk_coverage(store, seed),
                 fig_uncertainty_decomposition(store, seed), fig_confusion(store, seed)]
    made += [fig_ood(store, seeds), fig_evidence_anatomy(store, seeds[0]),
             fig_lambda_evidence(out_root, payload["seeds"])]
    made += [fig_architecture(), fig_search_funnel(), fig_spectral_norm_ablation()]
    made += [fig_calibration_families(store, seeds[0]),
             fig_uncertainty_vs_correctness(store, seeds[0])]
    made = [m for m in made if m is not None]

    summary = {"results_file": str(path), "seeds": payload["seeds"],
               "figures": [str(p.name) for p in made],
               "aggregate": payload.get("aggregate", {})}
    (ANALYSIS / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"[figures] {len(made)} figures written to {ASSETS}", flush=True)
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default="runs/full_*/matrix_results.json")
    p.add_argument("--reuse", action="store_true",
                   help="reuse analysis/test_predictions.npz instead of re-running inference")
    a = p.parse_args()
    raise SystemExit(main(a.results, reuse=a.reuse))
