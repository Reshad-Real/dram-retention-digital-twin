#!/usr/bin/env python
"""Compile the CircuiTikz circuit schematics and export them as figures.

Sources live in MANUSCRIPT/tikz/*.tex (standalone circuitikz, American source
models). Each is compiled to PDF with Tectonic and rasterized to a high-resolution
PNG. Outputs go to results/figures/ and MANUSCRIPT/figures/.

    python scripts/build_circuits.py
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1]
TIKZ = ROOT / "MANUSCRIPT" / "tikz"
TECTONIC = ROOT / "tools" / "tex" / "tectonic.exe"
OUT_DIRS = [ROOT / "results" / "figures", ROOT / "MANUSCRIPT" / "figures"]

JOBS = {
    "fig02a_circuit": ("fig02_circuit", 4.0),
    "fig02b_replica": ("fig02_replica", 5.0),
}


def main() -> int:
    for out in OUT_DIRS:
        out.mkdir(parents=True, exist_ok=True)
    for src, (dst, zoom) in JOBS.items():
        # compile in a throwaway temp dir so the tikz/ folder never holds a
        # lock-prone output PDF (Windows Search can hold new PDFs open)
        tmp = Path(tempfile.mkdtemp(prefix="dramckt_"))
        (tmp / f"{src}.tex").write_text((TIKZ / f"{src}.tex").read_text(encoding="utf-8"),
                                        encoding="utf-8")
        r = subprocess.run([str(TECTONIC), "-X", "compile", f"{src}.tex"],
                           cwd=str(tmp), capture_output=True, text=True)
        if r.returncode != 0:
            print(f"FAILED {src}:\n{r.stderr[-800:]}")
            shutil.rmtree(tmp, ignore_errors=True)
            return 1
        pdf = tmp / f"{src}.pdf"
        doc = fitz.open(pdf)
        pix = doc[0].get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        data = pdf.read_bytes()
        doc.close()

        def _write(target: Path, payload) -> None:
            for attempt in range(8):
                try:
                    if isinstance(payload, bytes):
                        target.write_bytes(payload)
                    else:
                        payload.save(str(target))
                    return
                except (PermissionError, OSError):
                    time.sleep(0.6)          # transient lock (indexer); retry
            raise RuntimeError(f"could not write {target}")

        for out in OUT_DIRS:
            _write(out / f"{dst}.png", pix)
        _write(ROOT / "results" / "figures" / f"{dst}.pdf", data)
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"  {src} -> {dst}.png  ({pix.width}x{pix.height})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
