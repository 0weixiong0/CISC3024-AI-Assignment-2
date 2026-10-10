"""Export the trained models and their evaluation artefacts into the JSON the web app reads.

Project : CISC3024 AI Assignment #2 (Pattern Recognition, University of Macau)
Author  : AI-generated implementation (Qoder). The student wrote no code.

This is the bridge between the experiment and the visualisation required by assignment component (6).
It writes plain JSON plus PNG crops so the browser page needs no Python backend to show:

    webapp/data/manifest.json          configuration table, curves, dataset facts, provenance
    webapp/data/samples/<id>.png       the actual images the predictions refer to
    webapp/data/samples.json           per-sample posterior quantities for every configuration

Run it after ``run_matrix.py`` has produced a ``matrix_results.json``::

    python src/export_webapp_data.py --results runs/full_*/matrix_results.json --n-samples 24
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data as D  # noqa: E402
from evaluate import collect_predictive  # noqa: E402
from fd_distribution import flexible_dirichlet_moments, uncertainty_decomposition  # noqa: E402
from metrics import (  # noqa: E402
    adaptive_expected_calibration_error, expected_calibration_error, reliability_bins,
    risk_coverage_curve,
)
from run_matrix import DEVICE, MODEL_NAMES, build, _head  # noqa: E402
from train import Config  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
WEBAPP = ROOT / "webapp" / "data"

CLASS_NAMES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse",
               "ship", "truck")

# The ablation panel only needs these; carrying all 36 metrics per variant would triple the payload.
VARIANT_METRICS = ("top1", "nll", "brier", "ece", "adaptive_ece", "aurc", "err_error_auroc",
                   "near_svhn_total_auroc", "far_tiny_imagenet_total_auroc",
                   "epistemic_mean", "aleatoric_mean")

# Result files are named matrix_results[_<curvature>][_nosn].json, so a variant key can be a
# combination; each maps to the one thing that differs from the main run.
VARIANT_NAMES = {"": "main run", "identity": "identity curvature", "kron": "kron curvature",
                 "none": "no curvature", "nosn": "no spectral normalisation",
                 "identity_nosn": "no spectral normalisation"}


def _json_safe(value):
    """Replace the non-finite floats json.dumps would emit as bare NaN/Infinity tokens.

    ``JSON.parse`` rejects those tokens, so a payload containing them is readable by Python but not
    by a browser. Empty reliability bins are the usual source.
    """
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return _json_safe(value.item())
    return value


def variants_of(out_root: Path) -> list[dict]:
    """Every completed run variant under runs/full_*, with the settings that distinguish it."""
    variants = []
    for path in sorted(ROOT.glob("runs/full_*/matrix_results*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[export] skipping unreadable {path}: {exc}", file=sys.stderr)
            continue
        key = path.stem.removeprefix("matrix_results").removeprefix("_") or "identity"
        variants.append({
            "key": key,
            "label": VARIANT_NAMES.get(key, key),
            "source": str(path.relative_to(ROOT)).replace("\\", "/"),
            "curvature": payload["curvature"],
            "spectral_norm": payload.get("spectral_norm", True),
            "backbone": payload["backbone"],
            "seeds": payload["seeds"],
            "is_main": path.parent.resolve() == out_root.resolve()
                       and path.name == "matrix_results.json",
            "metrics": {name: {m: stats for m, stats in sorted(entry.items())
                               if m in VARIANT_METRICS}
                        for name, entry in sorted(payload["aggregate"].items())},
        })
    return variants


def cost_of(payload: dict) -> list[dict]:
    """Measured training cost per seed, straight from the matrix run record."""
    return [{"seed": s["seed"],
             "training_seconds": s.get("training_seconds"),
             "mean_epoch_seconds": s.get("mean_epoch_seconds"),
             "epochs_run": s.get("epochs_run"),
             "peak_vram_mb": s.get("peak_vram_mb"),
             "lambda_closed_form": s.get("laplace_lambda_closed_form")}
            for s in payload["per_seed"]]


def trainable_parameters(out_root: Path, seed: int) -> int | None:
    """The parameter count the run record itself reports, so the page never re-types it."""
    summary = out_root / f"b1_ce_s{seed}" / "summary.json"
    if not summary.exists():
        return None
    return json.loads(summary.read_text(encoding="utf-8")).get("trainable_parameters")


def _denormalise(tensor: torch.Tensor) -> torch.Tensor:
    """Undo CIFAR normalisation so the saved PNG looks like the image the model actually saw."""
    mean = torch.tensor(D.CIFAR10_MEAN, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(D.CIFAR10_STD, device=tensor.device).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


def interesting_indices(a1: torch.nn.Module, loader, n_random: int, seed: int) -> list[int]:
    """Rank the whole test split by the evidential quantities, then keep a stratified shortlist.

    A purely random gallery cannot be relied on to contain the cases the figure and the web page
    need to explain: with a 91 % accurate model, 24 random draws yield about two misclassifications,
    so the "confidently wrong" and "maximally epistemic" panels would have nothing to show. The
    interesting indices are therefore found by scoring all 10,000 test images first, and random
    draws only top the list up.
    """
    coll = collect_predictive(a1, loader, DEVICE)
    probs, labels = coll["mean"], coll["labels"]
    pred = probs.argmax(1)
    conf, epi, ale = probs.max(1), coll["epistemic"], coll["aleatoric"]
    correct = pred == labels

    def argwhere(mask, scores, want_max=True, k=2):
        idx = np.where(mask)[0]
        if idx.size == 0:
            return []
        order = idx[np.argsort(scores[idx] if want_max else -scores[idx])][:k]
        return order.tolist()

    curated: list[int] = []
    for group in (
        argwhere(correct, conf, True),                          # confident and correct
        argwhere(correct, conf, False),                         # hesitant but correct
        argwhere(~correct, conf, True),                         # confidently wrong
        argwhere(~correct, epi, True),                          # wrong with high epistemic
        argwhere(np.ones_like(correct), epi, True),             # maximum epistemic overall
        argwhere(np.ones_like(correct), ale, True),             # maximum aleatoric overall
        argwhere(np.ones_like(correct), np.abs(probs[:, 0] - probs[:, 1]), True),  # near-tie
    ):
        curated.extend(group)

    seen, ordered = set(), []
    for i in curated:
        if i not in seen:
            seen.add(int(i)); ordered.append(int(i))

    rng = np.random.default_rng(seed)
    filler = [int(i) for i in rng.choice(len(labels), size=max(n_random * 3, 30), replace=False)
              if int(i) not in seen]
    ordered.extend(filler[:max(0, n_random - len(ordered))])
    return sorted(ordered[:max(n_random, len(ordered))])


def pick_samples(test_loader, indices: list[int]) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Gather the images at the given dataset indices, in index order."""
    wanted = set(indices)
    images, labels, found = [], [], []
    offset = 0
    for x, y in test_loader:
        for i in range(x.size(0)):
            if offset + i in wanted:
                images.append(x[i])
                labels.append(int(y[i]))
                found.append(offset + i)
        offset += x.size(0)
        if offset >= max(wanted):
            break
    order = np.argsort(found)
    return (torch.stack([images[j] for j in order]),
            torch.tensor([labels[j] for j in order], device=DEVICE),
            [found[j] for j in order])


@torch.no_grad()
def model_payload(name: str, model, batch: torch.Tensor, n_mc: int) -> dict:
    """Per-sample posterior quantities for one configuration."""
    out = model.predictive(batch.to(DEVICE), n_mc=n_mc)
    mean = out["mean"].float().cpu().numpy()
    var = out["variance"].float().cpu().numpy()
    epistemic = out["epistemic"].float().cpu().numpy()
    aleatoric = out["aleatoric"].float().cpu().numpy()
    total = out["total"].float().cpu().numpy()
    return {
        "mean": mean.tolist(),
        "variance": var.tolist(),
        "epistemic": epistemic.tolist(),
        "aleatoric": aleatoric.tolist(),
        "total": total.tolist(),
        "predicted_class": mean.argmax(1).tolist(),
        "confidence": mean.max(1).tolist(),
        "n_monte_carlo": n_mc,
    }


def evidential_internals(a1, batch: torch.Tensor) -> dict:
    """The raw Dirichlet quantities, exposed only for the evidential model."""
    with torch.no_grad():
        out = a1.forward(batch.to(DEVICE))
    alpha = out["alpha"].cpu().numpy()
    p = out["p"].cpu().numpy()
    tau = out["tau"].cpu().numpy()
    mu, var = flexible_dirichlet_moments(out["alpha"], out["p"], out["tau"])
    unc = uncertainty_decomposition(mu, var)
    return {
        "alpha": alpha.tolist(), "p_mode": p.tolist(), "tau": tau.reshape(-1).tolist(),
        "alpha0": alpha.sum(1).tolist(),
        "evidence": np.maximum(alpha - 1.0, 0.0).tolist(),
        "mean": mu.cpu().numpy().tolist(),
        "variance": var.cpu().numpy().tolist(),
        "epistemic": unc["epistemic"].cpu().numpy().tolist(),
        "aleatoric": unc["aleatoric"].cpu().numpy().tolist(),
    }


def curves_for(probs: np.ndarray, labels: np.ndarray) -> dict:
    """Reliability bins and the risk-coverage curve, recomputed here so the page draws them live."""
    cov, risk = risk_coverage_curve(probs, labels)
    return {
        "ece": expected_calibration_error(probs, labels),
        "adaptive_ece": adaptive_expected_calibration_error(probs, labels),
        "reliability_bins": reliability_bins(probs, labels),
        "risk_coverage": {"coverage": cov.tolist(), "risk": risk.tolist()},
    }


NOTES = (
    "Probabilities come from the frozen test split; nothing here was used for model selection.",
    "B1 and B2 have no epistemic term by construction, so the register strikes that quantity "
    "instead of plotting it as zero.",
    "A2 rescales probabilities only; its Dirichlet variance is not carried through calibration.",
    "Every configuration in the main table reads the same data/splits.json split and the same "
    "evaluation code path, so the comparison is internally consistent.",
    "Run variants recorded with fewer seeds than the main table are partial and are drawn as "
    "half-height rules.",
)


def write_outputs(manifest: dict, samples: dict) -> None:
    """Write the two JSON files plus the JavaScript bundle the page falls back to under file://."""
    manifest = _json_safe(manifest)
    samples = _json_safe(samples)
    WEBAPP.mkdir(parents=True, exist_ok=True)
    (WEBAPP / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (WEBAPP / "samples.json").write_text(json.dumps(samples), encoding="utf-8")
    bundle = ("/* Generated by src/export_webapp_data.py. Do not edit; regenerate instead.\n"
              "   Opened from file:// a browser refuses fetch() of local JSON, so the page reads\n"
              "   this bundle instead. Serve the folder over HTTP and the JSON files are used. */\n"
              "window.BAYES_DATA = ")
    bundle += json.dumps({"manifest": manifest, "samples": samples}, separators=(",", ":"))
    bundle += ";\n"
    (WEBAPP / "dataset.js").write_text(bundle, encoding="utf-8")


def main(results_path: str, n_samples: int, n_mc: int) -> int:
    results_file = sorted(glob.glob(results_path))
    if not results_file:
        print(f"[export] no results file matching {results_path!r}", file=sys.stderr)
        return 1
    results_file = Path(max(results_file, key=os.path.getmtime))
    payload = json.loads(results_file.read_text(encoding="utf-8"))
    out_root = Path(payload["out_dir"])

    WEBAPP.mkdir(parents=True, exist_ok=True)
    (WEBAPP / "samples").mkdir(exist_ok=True)

    loaders = D.cifar10_loaders(batch_size=256, num_workers=0, pin_memory=False)
    test_set = loaders["test"].dataset

    # Rebuild both backbones first: the gallery selection needs the evidential model's scores over
    # the entire test split, and the post-hoc layers need the same weights the experiment used.
    first_seed = payload["seeds"][0]
    b1_ckpt = torch.load(out_root / f"b1_ce_s{first_seed}" / "best.pt", map_location=DEVICE,
                         weights_only=False)
    b1 = build(Config(**b1_ckpt["config"])); b1.load_state_dict(b1_ckpt["model"]); b1.eval()
    a1_ckpt = torch.load(out_root / f"a1_fedl_s{first_seed}" / "best.pt", map_location=DEVICE,
                         weights_only=False)
    a1 = build(Config(**a1_ckpt["config"])); a1.load_state_dict(a1_ckpt["model"]); a1.eval()

    indices = interesting_indices(a1, loaders["test"], n_samples, seed=payload["split"]["seed"])
    images, labels, found = pick_samples(loaders["test"], indices)
    curated = set(indices[:14])

    for i, idx in enumerate(found):
        arr = (_denormalise(images[i]).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        Image.fromarray(arr).save(WEBAPP / "samples" / f"test_{idx:05d}.png")
    saved = [{"id": f"test_{idx:05d}", "index": idx, "label": int(labels[j]),
              "label_name": CLASS_NAMES[int(labels[j])],
              "selection": "curated" if idx in curated else "random"}
             for j, idx in enumerate(found)]

    samples: dict[str, object] = {"images": saved, "class_names": CLASS_NAMES}
    samples["A1_fedl"] = model_payload("A1_fedl", a1, images, n_mc)
    samples["A1_evidential_internals"] = evidential_internals(a1, images)
    samples["B1_ce"] = model_payload("B1_ce", b1, images, n_mc)

    # Post-hoc layers refitted exactly as run_matrix did, so the page shows the same numbers.
    from calibrate import CalibratedModel, DirichletScaler, TemperatureScaler
    from laplace_icla import LaplaceWrapper, LastLayerLaplace, cache_features

    phi_tr, y_tr = cache_features(b1, loaders["train"], DEVICE)
    val_phi, val_y = cache_features(b1, loaders["val"], DEVICE)
    a1_val = collect_predictive(a1, loaders["val"], DEVICE)

    temp = TemperatureScaler().fit(_head(b1)(val_phi), val_y)
    dscal = DirichletScaler().fit(
        torch.tensor(np.log(a1_val["mean"].clip(1e-8, 1.0)), device=DEVICE),
        torch.tensor(a1_val["labels"], device=DEVICE))
    lap = LastLayerLaplace(curvature=payload["curvature"], n_samples=n_mc).fit(_head(b1), phi_tr, y_tr)

    samples["B2_ce_temperature"] = model_payload("B2", CalibratedModel(b1, temp), images, n_mc)
    samples["A2_fedl_dirichlet"] = model_payload("A2", CalibratedModel(a1, dscal), images, n_mc)
    samples["A3_laplace_icla"] = model_payload("A3", LaplaceWrapper(b1, lap), images, n_mc)

    # Curves for the full test split, per configuration, from the cached predictive means.
    npz = np.load(out_root / "predictives.npz")
    curves: dict[str, dict] = {}
    for key in npz.files:
        if not key.startswith(f"s{payload['seeds'][0]}_test_"):
            continue
        name = key.split("test_", 1)[1]
        curves[name] = curves_for(npz[key], np.asarray(test_set.targets))

    manifest = {
        "source_results": str(results_file),
        "generated_from_out_dir": str(out_root),
        "dataset": {"name": "CIFAR-10", "classes": list(CLASS_NAMES),
                    "split": payload["split"], "test_images": int(len(test_set))},
        "experiment": {**{k: payload[k] for k in ("epochs", "seeds", "curvature", "backbone",
                                                  "total_seconds", "device")},
                       "trainable_parameters": trainable_parameters(
                           out_root, payload["seeds"][0])},
        "models": payload["aggregate"],
        "per_seed": [{"seed": s["seed"], "models": s["models"],
                      "laplace": s.get("laplace"), "calibrators": s.get("calibrators")}
                     for s in payload["per_seed"]],
        "cost": cost_of(payload),
        "variants": variants_of(out_root),
        "curves": curves,
        "n_monte_carlo": n_mc,
        "notes": NOTES,
    }

    write_outputs(manifest, samples)
    print(f"[export] wrote {WEBAPP / 'manifest.json'}")
    print(f"[export] wrote {WEBAPP / 'samples.json'} and {len(saved)} sample PNGs")
    return 0


def refresh_without_training() -> int:
    """Rebuild the derived manifest blocks and the JavaScript bundle from the JSON already on disk.

    The page reads derived quantities (training cost, run variants) that do not need a GPU, so this
    path lets them be corrected after ``main()`` has run without re-running inference.
    """
    manifest = json.loads((WEBAPP / "manifest.json").read_text(encoding="utf-8"))
    samples = json.loads((WEBAPP / "samples.json").read_text(encoding="utf-8"))
    payload = json.loads(Path(manifest["source_results"]).read_text(encoding="utf-8"))
    out_root = Path(manifest["generated_from_out_dir"])
    manifest["cost"] = cost_of(payload)
    manifest["variants"] = variants_of(out_root)
    manifest["experiment"]["trainable_parameters"] = trainable_parameters(
        out_root, payload["seeds"][0])
    manifest["notes"] = NOTES
    write_outputs(manifest, samples)
    print(f"[export] refreshed derived blocks from disk; variants: "
          f"{[v['key'] for v in manifest['variants']]}")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default="runs/full_*/matrix_results.json")
    p.add_argument("--n-samples", type=int, default=24)
    p.add_argument("--n-mc", type=int, default=64)
    p.add_argument("--derived-only", action="store_true",
                   help="skip inference; rewrite the derived blocks and dataset.js from disk")
    a = p.parse_args()
    raise SystemExit(refresh_without_training() if a.derived_only
                     else main(a.results, a.n_samples, a.n_mc))
