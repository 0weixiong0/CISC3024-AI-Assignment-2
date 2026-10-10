"""Capture the published interface for the report's figures.

Project : CISC3024 AI Assignment #2 (Pattern Recognition, University of Macau)
Author  : AI-generated implementation (Qoder). The student wrote no code.

The bench page needs ~114 MB of ONNX and a session compile before anything is visible, so a plain
screenshot tool captures an empty page. Playwright waits for the page to report that it is ready and
for the first inference to paint, then shoots the two views the report cites.

    python src/capture_webapp.py [--base http://127.0.0.1:8801]

Writes into report_assets/ and prints the measured state of each capture, so a figure that shows a
loading spinner cannot be mistaken for a finished one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "report_assets"
WIDTH, HEIGHT = 1440, 1000


def wait_for_instruments(page, timeout_ms: int) -> str:
    page.wait_for_function(
        "() => /Ready/.test(document.getElementById('bench-status').textContent)",
        timeout=timeout_ms)
    return page.evaluate("() => document.getElementById('bench-status').textContent.trim()")


def bench_shot(base: str, specimen: int, name: str) -> dict:
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", args=["--use-gl=angle"])
        page = browser.new_page(viewport={"width": WIDTH, "height": HEIGHT})
        page.goto(f"{base}/index.html?specimen={specimen}", wait_until="domcontentloaded")
        status = wait_for_instruments(page, 180_000)
        # wait until the evidential plate has actually drawn its ten rules
        page.wait_for_function(
            "() => document.querySelectorAll('#bars-a1 path.series').length >= 10",
            timeout=60_000)
        page.wait_for_timeout(900)
        page.screenshot(path=str(OUT / name), full_page=True)
        facts = page.evaluate("""() => ({
            runtime: document.getElementById('field-runtime').textContent,
            specimen: document.getElementById('field-specimen').textContent,
            verdict: document.getElementById('verdict-a1').textContent.trim().slice(0, 90),
            a1: Object.fromEntries([...document.querySelectorAll('#read-a1 div')]
              .map(d => [d.querySelector('dt').textContent, d.querySelector('dd').textContent])),
            cells: document.querySelectorAll('#pick-grid button').length,
            pickerOpen: document.querySelector('details.picker').open
        })""")
        browser.close()
    facts["status"] = status
    return facts


def register_shot(base: str, name: str) -> dict:
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome")
        page = browser.new_page(viewport={"width": WIDTH, "height": HEIGHT})
        page.goto(f"{base}/register.html", wait_until="domcontentloaded")
        page.wait_for_function("() => document.querySelectorAll('table').length >= 2",
                               timeout=30_000)
        page.screenshot(path=str(OUT / name), full_page=False)
        facts = page.evaluate("""() => ({
            entries: document.querySelectorAll('.entry').length,
            struck: document.querySelectorAll('td.struck').length,
            tables: document.querySelectorAll('table').length
        })""")
        # the whole register, with every entry open, as a second view
        page.evaluate("() => document.querySelectorAll('.entry')"
                      ".forEach(e => e.setAttribute('data-open','true'))")
        page.wait_for_timeout(400)
        page.screenshot(path=str(OUT / name.replace("first_viewport", "all_entries")),
                        full_page=True)
        browser.close()
    return facts


def main(base: str) -> int:
    OUT.mkdir(exist_ok=True)
    if not OUT.exists():
        print(f"[capture] cannot write to {OUT}", file=sys.stderr)
        return 1
    facts = bench_shot(base, 9370, "figure_11_bench_specimen_9370.png")
    print("[capture] bench:", facts)
    facts2 = register_shot(base, "figure_12_register_first_viewport.png")
    print("[capture] register:", facts2)
    for path in sorted(OUT.glob("figure_1[12]_*.png")):
        print(f"[capture] {path.name} {path.stat().st_size / 1024:.0f} kB")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", default="http://127.0.0.1:8801")
    raise SystemExit(main(p.parse_args().base))
