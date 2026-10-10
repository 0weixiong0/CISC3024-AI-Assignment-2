"""Write every CIFAR-10 test image plus its per-configuration posterior, for the web wall.

Project : CISC3024 AI Assignment #2 (Pattern Recognition, University of Macau)
Author  : AI-generated implementation (Qoder). The student wrote no code.

The bench page lets a visitor browse the whole station rather than a curated handful: ten thousand
thumbnails, filterable by class and sortable by any uncertainty quantity. This script writes the two
files that makes possible.

    webapp/data/test/<index>.png    one PNG per test image, at the 32x32 the model saw
    webapp/data/test-index.json     label, class name, and per-configuration predicted class,
                                    confidence, epistemic, aleatoric and total uncertainty

The numbers are read from ``analysis/test_predictions.npz``, the same cache the report's tables were
aggregated from, so a cell on the wall and a row in the register cannot disagree. Only the seed the
browser reproduces is exported: the bench runs seed 20261006's checkpoints, and quoting a different
seed's uncertainty next to a live inference would be a mismatch waiting to happen.

Images are PNG rather than JPEG because the wall's thumbnails are also the input the browser feeds to
the networks: a lossy recompression would silently change the posteriors the page then displays.

Licensing note for whoever packages this for publication: CIFAR-10 is Krizhevsky's dataset and its
official page states no licence. Assignment 1 shipped its whole EuroSAT test set on the ``gh-pages``
branch under CC BY 4.0; this folder follows that precedent, and the raw archive stays out of the
source repository so ``src/prepare_data.py`` remains the way to regenerate it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data as D  # noqa: E402
from run_matrix import MODEL_NAMES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "webapp" / "data" / "test"
INDEX = ROOT / "webapp" / "data" / "test-index.json"

CLASS_NAMES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse",
               "ship", "truck")


def denormalise(tensor: torch.Tensor) -> np.ndarray:
    """Undo the CIFAR normalisation so the PNG is the image a human would recognise."""
    mean = torch.tensor(D.CIFAR10_MEAN).view(3, 1, 1)
    std = torch.tensor(D.CIFAR10_STD).view(3, 1, 1)
    return ((tensor * std + mean).clamp(0, 1) * 255).permute(1, 2, 0).numpy().astype(np.uint8)


def main(seed: int, write_images: bool) -> int:
    npz_path = ROOT / "analysis" / "test_predictions.npz"
    if not npz_path.exists():
        print(f"[wall] {npz_path} not found; run src/evaluate.py first", file=sys.stderr)
        return 1
    npz = np.load(npz_path)

    missing = [m for m in MODEL_NAMES if f"{seed}::{m}__test__mean" not in npz]
    if missing:
        print(f"[wall] seed {seed} is not cached for {missing}", file=sys.stderr)
        return 1

    labels = np.asarray(npz[f"{seed}::B1_ce__test__labels"], dtype=np.int64)
    n = len(labels)

    records: list[dict] = []
    for k in range(n):
        entry: dict[str, object] = {"i": k, "y": int(labels[k]), "c": CLASS_NAMES[int(labels[k])]}
        models: dict[str, list[float]] = {}
        for model in MODEL_NAMES:
            mean = np.asarray(npz[f"{seed}::{model}__test__mean"])[k]
            models[model] = [int(mean.argmax()), round(float(mean.max()), 4),
                             round(float(npz[f"{seed}::{model}__test__epistemic"][k]), 6),
                             round(float(npz[f"{seed}::{model}__test__aleatoric"][k]), 5),
                             round(float(npz[f"{seed}::{model}__test__total"][k]), 5)]
        entry["m"] = models
        records.append(entry)

    if write_images:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        test_set = D.cifar10_loaders(batch_size=512, num_workers=0, pin_memory=False)["test"].dataset
        for i in range(len(test_set)):
            array = denormalise(test_set[i][0].float().clone())
            Image.fromarray(array).save(OUT_DIR / f"{i:05d}.png", optimize=False)
        print(f"[wall] wrote {n} PNGs to {OUT_DIR.relative_to(ROOT)}")

    INDEX.write_text(json.dumps({
        "seed": seed,
        "classes": list(CLASS_NAMES),
        "models": list(MODEL_NAMES),
        "columns": ["predicted", "confidence", "epistemic", "aleatoric", "total"],
        "count": n,
        "images": "data/test/",
        "provenance": "analysis/test_predictions.npz",
        "note": ("Uncertainty quantities are the values the report aggregated. The browser recomputes "
                 "them from the same seed's checkpoints, so a wall cell and a live inference agree."),
        "records": records,
    }, separators=(",", ":")), encoding="utf-8")
    print(f"[wall] wrote {INDEX.relative_to(ROOT)} "
          f"({INDEX.stat().st_size / 1e6:.2f} MB, {n} records)")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=20261006)
    p.add_argument("--no-images", action="store_true", help="rewrite the index only")
    a = p.parse_args()
    raise SystemExit(main(a.seed, not a.no_images))
