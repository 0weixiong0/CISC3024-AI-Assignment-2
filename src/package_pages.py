"""Assemble the GitHub Pages branch from the working copy of webapp/.

Project : CISC3024 AI Assignment #2 (Pattern Recognition, University of Macau)
Author  : AI-generated implementation (Qoder). The student wrote no code.

`main` carries source only. The published site also needs three things that are deliberately not in
that branch -- the two ONNX graphs, the onnxruntime-web wasm, and the 10,000 test thumbnails -- so
this script copies a complete `webapp/` into a build folder that can be committed to `gh-pages`.

    python src/package_pages.py                     # writes pages_build/
    python src/package_pages.py --check             # report sizes, write nothing

The copy is a straight mirror rather than a filtered one: the branch must be servable as it stands,
and a page that loads locally but 404s on Pages is worse than a slightly larger repository.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEBAPP = ROOT / "webapp"
BUILD = ROOT / "pages_build"

README = """# CISC3024 AI Assignment #2 — published site

This branch is the GitHub Pages deployment. Source, experiments and the report live on `main`.

- **`index.html`** — the bench. Both trained networks run in your browser tab; draw a 32×32
  specimen, upload a photograph, or pick any of the 10,000 CIFAR-10 test images.
- **`register.html`** — the ten-entry measurement register. It also opens from a local checkout with
  no server.

Contents beyond the source on `main`: `model/*.onnx` (114 MB, the two exported graphs), `ort/`
(33 MB, onnxruntime-web), and `data/test/` (23 MB, the test thumbnails). Regenerate all three with
`python src/export_onnx.py` and `python src/export_test_wall.py`, then rebuild this branch with
`python src/package_pages.py`.

CIFAR-10's official page states no licence; the thumbnails are published here for course
demonstration only, and the raw archive is fetched from its published location by
`src/prepare_data.py`.
"""


def tree_size(root: Path) -> tuple[float, int]:
    total, count = 0, 0
    for path in root.rglob("*"):
        if path.is_file():
            total += path.stat().st_size
            count += 1
    return total / 1e6, count


def main(check_only: bool) -> int:
    if not WEBAPP.exists():
        print(f"[pages] {WEBAPP} not found", file=sys.stderr)
        return 1

    required = ["index.html", "register.html", "styles.css", "bench.js", "register.js",
                "data/manifest.json", "data/samples.json", "data/instruments.json",
                "data/test-index.json", "data/dataset.js"]
    for rel in ["model/baseline.onnx", "model/evidential.onnx"]:
        required.append(rel)
    missing = [r for r in required if not (WEBAPP / r).exists()]
    if missing:
        print(f"[pages] missing from webapp/: {missing}", file=sys.stderr)
        return 1

    size, count = tree_size(WEBAPP)
    images = len(list((WEBAPP / "data" / "test").glob("*.png")))
    print(f"[pages] webapp/ is {size:.1f} MB across {count} files, {images} test thumbnails")
    if images < 10000:
        print("[pages] fewer than 10,000 thumbnails; run src/export_test_wall.py first",
              file=sys.stderr)
        return 1
    if check_only:
        return 0

    # The build folder doubles as the working tree of the gh-pages branch, so its .git directory
    # must survive a rebuild; copying over an existing tree keeps it in place.
    BUILD.mkdir(exist_ok=True)
    shutil.copytree(WEBAPP, BUILD, dirs_exist_ok=True)
    (BUILD / "README.md").write_text(README, encoding="utf-8")
    built, built_count = tree_size(BUILD)
    print(f"[pages] wrote {BUILD.relative_to(ROOT)}: {built:.1f} MB, {built_count} files")
    print("[pages] publish with: git checkout --orphan gh-pages (in a clean clone), commit this "
          "folder, push, then enable Pages for the branch")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true", help="verify sizes without writing")
    raise SystemExit(main(p.parse_args().check))
