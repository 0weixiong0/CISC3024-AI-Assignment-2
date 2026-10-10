"""Run the configuration matrix, over one or more seeds, and dump one comparison table.

Project : CISC3024 AI Assignment #2 (Pattern Recognition, University of Macau)
Author  : AI-generated implementation (Qoder). The student wrote no code.

Configurations (plan section 4). Only two of them need a backbone training run; the rest are post-hoc
fits on stored weights and cached features, which is what makes the five-way comparison affordable.

    B1  cross-entropy baseline                     trained
    B2  B1 + temperature scaling                   post-hoc
    A1  flexible-Dirichlet evidential model        trained
    A2  A1 + Dirichlet scaling                     post-hoc
    A3  B1 + last-layer Laplace (identity or kron) post-hoc

Seeds. Each seed re-trains B1 and A1 and re-fits every post-hoc layer on those weights, so a seed is a
complete independent repetition. The frozen split is deliberately *not* re-drawn per seed -- it is fixed
by ``data.SEED`` -- so what varies between seeds is initialisation and shuffling order only. The test
split is opened exactly once per (configuration, seed) pair, which is the plan's M5 rule honoured at
multi-seed scale; nothing that feeds model selection ever sees it.

Usage:
    python src/run_matrix.py --pilot                       # 10 % of train, 20 epochs, 1 seed
    python src/run_matrix.py --seeds 3                     # full data, 100 epochs, 3 seeds
    python src/run_matrix.py --seeds 3 --skip-training     # reuse existing checkpoints
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data as D  # noqa: E402
from calibrate import CalibratedModel, DirichletScaler, TemperatureScaler  # noqa: E402
from evaluate import (  # noqa: E402
    collect_predictive, mean_confidence_gap, score_in_distribution, score_ood,
)
from laplace_icla import LaplaceWrapper, LastLayerLaplace, cache_features  # noqa: E402
from models import DeterministicNet, EvidentialNet  # noqa: E402
from train import Config, train  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAMES = ("B1_ce", "B2_ce_temperature", "A1_fedl", "A2_fedl_dirichlet", "A3_laplace_icla")

# Largest value `--seeds N` may be read as a count of runs.
MAX_SEED_COUNT = 20


def build(cfg: Config) -> torch.nn.Module:
    cls = EvidentialNet if cfg.is_evidential else DeterministicNet
    kwargs = dict(spectral_norm=cfg.spectral_norm, dropout=cfg.dropout)
    if cfg.is_evidential:
        kwargs.update(hidden=cfg.hidden, head_layers=cfg.head_layers)
    return cls(cfg.backbone, 10, **kwargs).to(DEVICE)


def _head(model: torch.nn.Module) -> torch.nn.Module:
    """The final linear layer that the Laplace treatment replaces."""
    return model.classifier if hasattr(model, "classifier") else model.alpha_head[-1]


def load_or_train(cfg: Config, out_dir: Path, skip_training: bool) -> tuple[torch.nn.Module, dict]:
    """Return the best-validation-NLL model plus its training summary.

    Reuses ``best.pt`` when ``skip_training`` is set, so an interrupted multi-seed run resumes without
    re-training the seeds that already finished.
    """
    ckpt = out_dir / "best.pt"
    summary_path = out_dir / "summary.json"
    if skip_training and ckpt.exists():
        payload = torch.load(ckpt, map_location=DEVICE, weights_only=False)
        model = build(Config(**payload["config"]))
        model.load_state_dict(payload["model"])
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        print(f"[matrix] reusing {out_dir.name} epoch {payload.get('epoch')}", flush=True)
        return model.eval(), summary

    summary = train(cfg, out_dir)
    summary_path.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    payload = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    model = build(cfg)
    model.load_state_dict(payload["model"])
    return model.eval(), summary


def evaluate_pair(b1: torch.nn.Module, a1: torch.nn.Module, loaders: dict, ood: dict,
                  curvature: str) -> tuple[dict, dict, dict]:
    """Fit every post-hoc layer on the two backbones, then score all five configurations."""
    phi_tr, y_tr = cache_features(b1, loaders["train"], DEVICE)
    val_phi, val_y = cache_features(b1, loaders["val"], DEVICE)
    head = _head(b1)

    a1_val = collect_predictive(a1, loaders["val"], DEVICE)
    a1_val_logp = torch.tensor(np.log(a1_val["mean"].clip(1e-8, 1.0)), device=DEVICE)
    a1_val_targets = torch.tensor(a1_val["labels"], device=DEVICE)

    temp = TemperatureScaler().fit(head(val_phi), val_y)
    dscal = DirichletScaler().fit(a1_val_logp, a1_val_targets)
    lap = LastLayerLaplace(curvature=curvature, n_samples=120).fit(head, phi_tr, y_tr)

    models = {
        "B1_ce": b1,
        "B2_ce_temperature": CalibratedModel(b1, temp),
        "A1_fedl": a1,
        "A2_fedl_dirichlet": CalibratedModel(a1, dscal),
        "A3_laplace_icla": LaplaceWrapper(b1, lap),
    }

    extras = {
        "calibrators": {"b2_temperature": temp.describe(), "a2_dirichlet": dscal.describe()},
        "laplace": {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                    for k, v in lap.state.items()},
        "laplace_lambda_closed_form": float(lap.P / float((lap.W_map.reshape(-1) ** 2).sum())),
    }

    rows: dict[str, dict] = {}
    means: dict[str, np.ndarray] = {}
    for name, model in models.items():
        coll = collect_predictive(model, loaders["test"], DEVICE)
        means[f"test_{name}"] = coll["mean"]
        row = score_in_distribution(coll)
        row["mean_confidence_minus_accuracy"] = mean_confidence_gap(coll)
        for oname, oloader in ood.items():
            row.update(score_ood(coll, {oname: collect_predictive(model, oloader, DEVICE)}))
        rows[name] = {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                      for k, v in row.items()}
        print(f"[matrix] {name:22s} acc={row['top1']:.4f} nll={row['nll']:.4f} "
              f"ece={row['ece']:.4f} brier={row['brier']:.4f}", flush=True)
    return rows, means, extras


def aggregate(per_seed: list[dict]) -> dict[str, dict]:
    """Mean and sample standard deviation of every metric across seeds."""
    out: dict[str, dict] = {}
    for name in MODEL_NAMES:
        rows = [s["models"][name] for s in per_seed if name in s["models"]]
        if not rows:
            continue
        numeric = [k for k, v in rows[0].items() if isinstance(v, (int, float))]
        out[name] = {k: {"mean": statistics.fmean(r[k] for r in rows),
                         "std": statistics.stdev([r[k] for r in rows]) if len(rows) > 1 else 0.0,
                         "n": len(rows)} for k in numeric}
    return out


def run(pilot: bool, skip_training: bool, epochs: int, curvature: str, workers: int,
        seeds: list[int], backbone: str = "vgg16_bn", out_dir: str | None = None,
        spectral_norm: bool = True) -> dict:
    tag = "pilot" if pilot else "full"
    frac = 0.10 if pilot else None
    n_epochs = epochs or (20 if pilot else 100)
    # An explicit --out-dir is what makes --skip-training usable: without it every invocation writes a
    # fresh timestamped directory and can therefore never find the checkpoints it is told to reuse.
    out_root = Path(out_dir) if out_dir else ROOT / "runs" / f"{tag}_{int(time.time())}"
    out_root.mkdir(parents=True, exist_ok=True)

    results: dict[str, object] = {
        "tag": tag, "epochs": n_epochs, "train_fraction": frac, "device": str(DEVICE),
        "curvature": curvature, "workers": workers, "seeds": seeds, "backbone": backbone,
        "spectral_norm": spectral_norm,
        "split": {"seed": D.SEED, "val_fraction": D.VAL_FRACTION}, "per_seed": [],
    }
    per_seed: list[dict] = []
    all_means: dict[str, np.ndarray] = {}
    t_all = time.perf_counter()

    for seed in seeds:
        t0 = time.perf_counter()
        b1, sum_b1 = load_or_train(
            Config(name="b1_ce", backbone=backbone, epochs=n_epochs, train_fraction=frac,
                   num_workers=workers, seed=seed, spectral_norm=spectral_norm),
            out_root / f"b1_ce_s{seed}", skip_training)
        a1, sum_a1 = load_or_train(
            Config(name="a1_fedl", backbone=backbone, epochs=n_epochs, train_fraction=frac,
                   num_workers=workers, seed=seed, spectral_norm=spectral_norm),
            out_root / f"a1_fedl_s{seed}", skip_training)
        training_seconds = round(time.perf_counter() - t0, 1)

        # Loaders are built after training on purpose: constructing them earlier leaves two
        # persistent Windows worker pools alive at once, which deadlocked the first pilot run.
        loaders = D.cifar10_loaders(batch_size=64, num_workers=workers, train_subset_fraction=frac,
                                    pin_memory=DEVICE.type == "cuda" and workers > 0)
        ood = D.ood_loaders(batch_size=256, num_workers=workers, limit=2500 if pilot else 10000)

        rows, means, extras = evaluate_pair(b1, a1, loaders, ood, curvature)
        all_means.update({f"s{seed}_{k}": v for k, v in means.items()})

        entry = {"seed": seed, "models": rows, "ood_controls_used": list(ood.keys()),
                 "training_seconds": training_seconds,
                 "peak_vram_mb": {"b1": sum_b1.get("peak_vram_mb"), "a1": sum_a1.get("peak_vram_mb")},
                 "mean_epoch_seconds": {"b1": sum_b1.get("mean_epoch_seconds"),
                                        "a1": sum_a1.get("mean_epoch_seconds")},
                 "epochs_run": {"b1": len(sum_b1.get("history", [])),
                                "a1": len(sum_a1.get("history", []))},
                 **extras}
        per_seed.append(entry)
        results["per_seed"].append(entry)
        print(f"[matrix] seed {seed} complete: training {training_seconds}s, "
              f"total {round(time.perf_counter() - t0, 1)}s", flush=True)

        # Release this seed's worker pools before the next iteration creates its own. Two persistent
        # pools coexisting in one process is what deadlocked on Windows, and it is also why the pilot
        # had to run with workers=0; clearing them here is what makes --workers > 0 safe again.
        del loaders, ood, b1, a1
        gc.collect()

    results["aggregate"] = aggregate(per_seed)
    results["total_seconds"] = round(time.perf_counter() - t_all, 1)
    results["out_dir"] = str(out_root)
    # Reusing an existing run directory must not destroy the results already stored there.
    suffix = f"_{curvature}" if out_dir else ""
    if not spectral_norm:
        suffix += "_nosn"
    results_path = out_root / f"matrix_results{suffix}.json"
    results_path.write_text(json.dumps(results, indent=1), encoding="utf-8")
    np.savez_compressed(out_root / f"predictives{suffix}.npz", **all_means)
    results["results_file"] = str(results_path)
    print(f"[matrix] wrote {results_path} in {results['total_seconds']}s", flush=True)
    return results


def parse_seeds(text: str) -> list[int]:
    """Accept either a count ('3' -> three consecutive seeds) or an explicit list ('42,43,44').

    A bare number larger than the maximum count is a single seed written without its trailing comma,
    which as a count would mean millions of runs. It is rejected rather than obeyed: this CLI already
    once spent an afternoon training seed 20261009 because of exactly that typo.
    """
    if "," in text:
        return [int(t) for t in text.split(",") if t.strip()]
    value = int(text)
    if value > MAX_SEED_COUNT:
        raise SystemExit(
            f"--seeds {text!r} would be read as a count of {text} runs. To mean one specific seed, "
            f"write it as a list: --seeds {text},")
    return [D.SEED + i for i in range(value)]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pilot", action="store_true", help="10 %% of train, 20 epochs")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--skip-training", action="store_true", help="reuse existing best.pt checkpoints")
    p.add_argument("--curvature", default="identity", choices=["identity", "kron", "none"])
    p.add_argument("--workers", type=int, default=0,
                   help="DataLoader worker processes; 0 avoids the Windows spawn deadlock")
    p.add_argument("--seeds", default="1", help="run count (max 20), or a comma-separated list of explicit seeds")
    p.add_argument("--backbone", default="vgg16_bn", choices=["vgg16_bn", "resnet18"])
    p.add_argument("--out-dir", default=None,
                   help="write into an existing run directory (enables --skip-training reuse)")
    p.add_argument("--no-spectral-norm", action="store_true",
                   help="ablate the architecture-level Lipschitz regulariser on both backbones")
    a = p.parse_args()
    run(pilot=a.pilot, skip_training=a.skip_training, epochs=a.epochs, curvature=a.curvature,
        workers=a.workers, seeds=parse_seeds(a.seeds), backbone=a.backbone, out_dir=a.out_dir,
        spectral_norm=not a.no_spectral_norm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
