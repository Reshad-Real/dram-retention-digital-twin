#!/usr/bin/env python
"""Stage 0 -- provision NGSpice and the PTM device models, then self-check.

    python scripts/00_setup_environment.py
    python scripts/00_setup_environment.py --check-only

What it does
    1. locates (or downloads) an NGSpice build into ``tools/``
    2. downloads the PTM BSIM4 model cards into ``data/spice_models/``
    3. synthesises the process-corner cards declared in the configuration
    4. runs a smoke simulation through both execution backends
    5. writes ``logs/manifest_setup.json`` and
       ``supplementary/environment_lock.txt``

Every download is verified by size and content before being accepted, so a
captive-portal HTML page is never silently mistaken for an archive.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dramdt.config import load_config                                # noqa: E402
from dramdt.env import NgSpiceNotFound, resolve_environment          # noqa: E402
from dramdt.logging_utils import (RunManifest, package_versions,     # noqa: E402
                                  setup_logging, stage)
from dramdt.models.technology import Technology                      # noqa: E402
from dramdt.paths import (LOGS_DIR, SPICE_MODEL_DIR, SUPPLEMENTARY_DIR,
                          TOOLS_DIR, ensure_directories)             # noqa: E402

# PTM bulk-CMOS BSIM4 cards, hosted by the PTM project (University of Minnesota).
PTM_MODELS = {
    "45nm_HP.pm": "1H5eUrlxDpi2Sdmf5W9rCsjBRjttYPFZs",
    "45nm_LP.pm": "1l_4DKHzqwFFLugqTWzVWdWruB7eJL4mK",
    "32nm_HP.pm": "1Wr835xhQDQwXfHIA1k1_fPD1z3af1iAr",
    "32nm_LP.pm": "1irl52pj95lruVrSEXkKPgQxvjTDH9uJQ",
    "22nm_HP.pm": "1rXi_b-YINlmufzJa-VWFiyihG41ANKYy",
    "22nm_LP.pm": "1YH7vUTEpGnez_R603BfGvVXIxSXaaHoI",
}
PTM_URL = "https://drive.usercontent.google.com/download?id={fid}&export=download"

NGSPICE_VERSION = "46"
NGSPICE_URLS = [
    "https://master.dl.sourceforge.net/project/ngspice/ng-spice-rework/"
    f"{NGSPICE_VERSION}/ngspice-{NGSPICE_VERSION}_64.7z?viasf=1",
    "https://downloads.sourceforge.net/project/ngspice/ng-spice-rework/"
    f"{NGSPICE_VERSION}/ngspice-{NGSPICE_VERSION}_64.7z",
]


def _download(url: str, target: Path, min_bytes: int, expect_text: str | None = None,
              log=None) -> bool:
    """Download and sanity-check a file.  Returns True on success."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=600) as r, target.open("wb") as fh:
            shutil.copyfileobj(r, fh)
    except Exception as exc:
        if log:
            log.warning("download failed (%s): %s", url.split("/")[2], exc)
        return False

    size = target.stat().st_size if target.exists() else 0
    if size < min_bytes:
        if log:
            log.warning("%s is only %d bytes -- rejected (likely an error page)",
                        target.name, size)
        target.unlink(missing_ok=True)
        return False
    if expect_text is not None:
        head = target.read_bytes()[:400].decode("utf-8", errors="replace")
        if expect_text not in head or head.lstrip().lower().startswith("<!doctype"):
            if log:
                log.warning("%s does not look like the expected content -- rejected",
                            target.name)
            target.unlink(missing_ok=True)
            return False
    return True


def fetch_ptm_models(log, force: bool = False) -> dict[str, str]:
    SPICE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    status: dict[str, str] = {}
    for name, fid in PTM_MODELS.items():
        target = SPICE_MODEL_DIR / name
        if target.exists() and target.stat().st_size > 5000 and not force:
            status[name] = "present"
            continue
        ok = _download(PTM_URL.format(fid=fid), target, min_bytes=5000,
                       expect_text="PTM", log=log)
        status[name] = "downloaded" if ok else "FAILED"
        log.info("PTM %-12s %s", name, status[name])
    return status


def fetch_ngspice(log, force: bool = False) -> str:
    """Vendor an NGSpice build under ``tools/ng`` (Windows archive)."""
    try:
        env = resolve_environment(probe=False)
        if env.has_exe and not force:
            log.info("NGSpice already available: %s", env.exe)
            return "present"
    except NgSpiceNotFound:
        pass

    if platform.system() != "Windows":
        log.warning("Automatic NGSpice provisioning is implemented for Windows only. "
                    "On Linux/macOS install it with your package manager, e.g. "
                    "`apt install ngspice` or `brew install ngspice`.")
        return "manual-install-required"

    TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    archive = TOOLS_DIR / f"ngspice-{NGSPICE_VERSION}_64.7z"
    got = any(_download(u, archive, min_bytes=2_000_000, log=log) for u in NGSPICE_URLS)
    if not got:
        log.error("Could not download NGSpice; install it manually and set "
                  "DRAMDT_NGSPICE_EXE.")
        return "FAILED"

    dest = TOOLS_DIR / "ng"
    dest.mkdir(parents=True, exist_ok=True)
    # Windows ships bsdtar, which reads 7-Zip archives.
    r = subprocess.run(["tar", "-xf", str(archive), "-C", str(dest)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        log.error("Extraction failed: %s", r.stderr[:400])
        return "FAILED"
    log.info("NGSpice extracted to %s", dest)
    return "downloaded"


def smoke_test(log) -> dict[str, object]:
    """Run a trivial circuit through both backends and compare the results."""
    from dramdt.simulation.runner import SpiceRunner

    deck = (
        "* dramdt smoke test\n"
        "V1 a 0 DC 1.0\n"
        "R1 a b 1k\n"
        "R2 b 0 1k\n"
        ".control\n"
        "set noaskquit\n"
        "op\n"
        "echo ---DRAMDT-MEAS-BEGIN---\n"
        "print v(b)\n"
        "echo ---DRAMDT-MEAS-END---\n"
        ".endc\n"
        ".end\n"
    )
    env = resolve_environment()
    out: dict[str, object] = {"environment": env.describe()}

    with SpiceRunner(env, backend="subprocess") as runner:
        res = runner.run(deck, tag="smoke")
        v = res.get("v_b")
        out["subprocess"] = {"ok": res.ok, "v_b": v, "runtime_s": round(res.runtime_s, 4)}
        log.info("subprocess backend: ok=%s v(b)=%s (expected 0.5)", res.ok, v)
        if res.ok and v is not None and abs(v - 0.5) > 1e-6:
            log.error("subprocess backend returned an unexpected value")

    if env.has_dll:
        try:
            with SpiceRunner(env, backend="pyspice-shared") as runner:
                res = runner.run(deck, tag="smoke_shared")
                v2 = res.get("v_b")
                out["pyspice_shared"] = {"ok": res.ok, "v_b": v2,
                                         "runtime_s": round(res.runtime_s, 4)}
                log.info("pyspice-shared backend: ok=%s v(b)=%s", res.ok, v2)
        except Exception as exc:
            log.warning("pyspice-shared backend unavailable: %s", exc)
            out["pyspice_shared"] = {"ok": False, "error": str(exc)}
    else:
        out["pyspice_shared"] = {"ok": False, "error": "shared library not found"}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", nargs="+", default=["dram_1t1c_45nm_lp.yaml"])
    ap.add_argument("--check-only", action="store_true",
                    help="verify the environment without downloading anything")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()

    ensure_directories()
    log = setup_logging("setup", filename="00_setup_environment.log")
    manifest = RunManifest(stage="setup")

    with stage("setup", log, manifest, LOGS_DIR / "manifest_setup.json"):
        if not args.check_only:
            manifest.outputs["ngspice"] = fetch_ngspice(log, force=args.force)
            manifest.outputs["ptm_models"] = fetch_ptm_models(log, force=args.force)

        cfg = load_config(*args.config)
        log.info("Configuration: %s", cfg.summary())
        tech = Technology.from_config(cfg, generate=True)
        log.info("Technology: %s", json.dumps(tech.describe(), indent=2))
        manifest.outputs["technology"] = tech.describe()
        manifest.outputs["corner_cards"] = {k: str(v) for k, v in tech.corner_cards.items()}

        manifest.outputs["smoke_test"] = smoke_test(log)

        lock = SUPPLEMENTARY_DIR / "environment_lock.txt"
        lock.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# dramdt environment lock",
            f"# platform: {platform.platform()}",
            f"# python:   {sys.version.splitlines()[0]}",
            "",
            "## package versions",
        ]
        lines += [f"{k}=={v}" for k, v in package_versions().items()]
        lines += ["", "## native tools",
                  f"ngspice_exe={manifest.outputs['smoke_test']['environment']['ngspice_exe']}",
                  f"ngspice_version={manifest.outputs['smoke_test']['environment']['ngspice_version']}",
                  "", "## device models",
                  f"model_fingerprint={tech.fingerprint()}"]
        lock.write_text("\n".join(lines) + "\n", encoding="utf-8")
        log.info("Environment lock -> %s", lock)
        manifest.outputs["environment_lock"] = str(lock)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
