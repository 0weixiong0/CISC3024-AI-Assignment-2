"""Download and prepare the datasets declared in stage_02_design_and_plan.md.

Author: AI (Qoder). Project: CISC3024 AI Assignment #2.
Writes only into ./data . Nothing is uploaded or published by this script.

Datasets:
  * CIFAR-10        (torchvision, 163 MB)          -> data/cifar10
  * SVHN            (torchvision, ~182 MB)         -> data/svhn        (near-OOD control)
  * Tiny-ImageNet-200 (cs231n zip, 248 MB)         -> data/tiny-imagenet-200 (far-OOD control)

Tiny-ImageNet needs a manual fix: the official archive ships val/val_annotations.txt
instead of per-class folders, which torchvision.ImageFolder cannot read. This script
relocates validation images into val/class/<class> directories and records the mapping.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)

TINY_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
TINY_ZIP = DATA / "tiny-imagenet-200.zip"

REPORT: dict[str, object] = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fetch_torchvision() -> None:
    """CIFAR-10 and SVHN through torchvision (train + test splits).

    Note the API difference that broke the first run: CIFAR10 takes ``train=bool``
    while SVHN takes ``split='train'|'test'``. They are therefore handled separately
    instead of through a shared loop.
    """
    import torchvision as tv

    transforms = tv.transforms.Compose([tv.transforms.ToTensor()])

    jobs = [
        ("cifar10", "train", lambda: tv.datasets.CIFAR10(root=str(DATA), train=True, download=True, transform=transforms)),
        ("cifar10", "test", lambda: tv.datasets.CIFAR10(root=str(DATA), train=False, download=True, transform=transforms)),
        ("svhn", "train", lambda: tv.datasets.SVHN(root=str(DATA), split="train", download=True, transform=transforms)),
        ("svhn", "test", lambda: tv.datasets.SVHN(root=str(DATA), split="test", download=True, transform=transforms)),
    ]

    for name, split, build in jobs:
        t0 = time.perf_counter()
        ds = build()
        n = len(ds)  # type: ignore[arg-type]
        dt = time.perf_counter() - t0
        log(f"{name}/{split}: {n} images in {dt:.1f}s")
        REPORT[f"{name}_{split}"] = {"n": n, "seconds": round(dt, 1)}


def dir_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 2**20


def tiny_is_prepared(target: Path) -> bool:
    """True when val/ has class folders and no longer holds val_annotations.txt."""
    val = target / "val"
    if not val.is_dir():
        return False
    if (val / "val_annotations.txt").exists():
        return False
    if (val / "images").exists():
        return False
    return any(p.is_dir() for p in val.iterdir())


def fetch_tiny_imagenet() -> None:
    """Download and reorganise Tiny-ImageNet-200 into an ImageFolder-compatible tree."""
    target = DATA / "tiny-imagenet-200"
    if tiny_is_prepared(target):
        log("tiny-imagenet-200 already prepared, skipping")
        return

    if not TINY_ZIP.exists():
        t0 = time.perf_counter()
        log(f"downloading {TINY_URL}")
        req = urllib.request.Request(TINY_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=300) as resp, open(TINY_ZIP, "wb") as fh:
            total = int(resp.headers.get("Content-Length", "0"))
            REPORT["tiny_imagenet_zip_bytes_declared"] = total
            got = 0
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
                got += len(chunk)
                if total and got % (25 << 20) < (1 << 20):
                    log(f"  {got / 2**20:.0f}/{total / 2**20:.0f} MB")
        log(f"downloaded {got / 2**20:.1f} MB in {time.perf_counter() - t0:.0f}s")
        REPORT["tiny_imagenet_zip_bytes_actual"] = got

    if not target.exists():
        log("extracting")
        with zipfile.ZipFile(TINY_ZIP) as zf:
            zf.extractall(DATA)

    ann = target / "val" / "val_annotations.txt"
    if ann.exists():
        log("relocating validation images into per-class folders")
        mapping: dict[str, str] = {}
        with open(ann, encoding="utf-8") as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2:
                    continue
                fname, cls = parts[0], parts[1]
                src = target / "val" / "images" / fname
                if not src.exists():
                    continue
                dst_dir = target / "val" / cls
                dst_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst_dir / fname))
                mapping[fname] = cls
        shutil.rmtree(target / "val" / "images", ignore_errors=True)
        ann.write_text(
            "\n".join(f"{k}\t{v}" for k, v in sorted(mapping.items())), encoding="utf-8"
        )
        REPORT["tiny_val_moved"] = len(mapping)
        log(f"moved {len(mapping)} validation images into {len(set(mapping.values()))} classes")

    REPORT["tiny_imagenet_mb_on_disk"] = round(dir_size_mb(target), 1)


def main() -> int:
    try:
        fetch_torchvision()
    except Exception as exc:  # noqa: BLE001 - report and continue with the remaining dataset
        REPORT["torchvision_error"] = f"{type(exc).__name__}: {exc}"
        log(f"ERROR torchvision stage: {type(exc).__name__}: {exc}")

    try:
        fetch_tiny_imagenet()
    except Exception as exc:  # noqa: BLE001
        REPORT["tiny_imagenet_error"] = f"{type(exc).__name__}: {exc}"
        log(f"ERROR tiny-imagenet stage: {type(exc).__name__}: {exc}")

    REPORT["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out = ROOT / "data" / "download_report.json"
    out.write_text(json.dumps(REPORT, indent=2), encoding="utf-8")
    log(f"wrote {out}")
    return 0 if "error" not in json.dumps(REPORT).lower() else 1


if __name__ == "__main__":
    sys.exit(main())
