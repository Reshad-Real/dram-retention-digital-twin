#!/usr/bin/env python
"""Run the whole pipeline end to end.

    python scripts/run_all.py                       # full study
    python scripts/run_all.py --quick               # small, fast smoke run
    python scripts/run_all.py --from 04 --to 10     # resume a range of stages
    python scripts/run_all.py --skip 03 07          # omit specific stages

Each stage is a separate process, so a failure is isolated and the pipeline can
be resumed from the stage that failed.  A summary of stage status and timings is
written to ``logs/pipeline_summary_<tag>.json``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dramdt.logging_utils import setup_logging                          # noqa: E402
from dramdt.paths import LOGS_DIR, ensure_directories                   # noqa: E402

ARCH = "dram_1t1c_45nm_lp.yaml"
SURR = "surrogate.yaml"
OPTI = "optimization.yaml"

#: (id, script, description, config files, extra args, quick-mode overrides)
STAGES = [
    ("00", "00_setup_environment.py", "Provision NGSpice + PTM models",
     [ARCH], [], []),
    ("01", "01_validate_spice.py", "Validate the simulation setup",
     [ARCH], [], ["--n", "40", "--n-backend", "4"]),
    ("02", "02_generate_dataset.py", "SPICE sampling campaign + dataset export",
     [ARCH], [], ["--n-samples", "1500"]),
    ("03", "03_assist_study.py", "DRAM assist-technique comparison",
     [ARCH], [], ["--n-designs", "60"]),
    ("04", "04_train_surrogates.py", "Train and compare surrogate models",
     [ARCH, SURR], [], ["--no-tuning", "--models", "ridge", "random_forest",
                        "xgboost", "lightgbm", "mlp"]),
    ("05", "05_optimize.py", "Multi-objective optimisation",
     [ARCH, SURR, OPTI], [], ["--runs", "3", "--generations", "25",
                              "--population", "40"]),
    ("06", "06_explain.py", "Explainability (SHAP / PDP / ALE)",
     [ARCH, SURR], [], []),
    ("07", "07_robustness.py", "Robustness, yield and sensitivity",
     [ARCH, SURR, OPTI], [], ["--max-designs", "2", "--sobol-base", "128",
                              "--skip-spice-check"]),
    ("08", "08_statistics.py", "Statistical validation",
     [ARCH, SURR, OPTI], [], []),
    ("09", "09_figures_tables.py", "Publication figures",
     [ARCH, SURR, OPTI], [], []),
    ("10", "10_report.py", "Assemble the technical report",
     [ARCH, SURR, OPTI], [], []),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--quick", action="store_true",
                    help="small sample counts and budgets, for a smoke run")
    ap.add_argument("--from", dest="from_stage", default="00")
    ap.add_argument("--to", dest="to_stage", default="10")
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--continue-on-error", action="store_true")
    args = ap.parse_args()

    ensure_directories()
    log = setup_logging("pipeline", filename=f"run_all_{args.tag}.log")
    python = sys.executable

    results: list[dict] = []
    t_start = time.perf_counter()

    for sid, script, description, configs, extra, quick_args in STAGES:
        if sid < args.from_stage or sid > args.to_stage or sid in args.skip:
            log.info("[%s] skipped -- %s", sid, description)
            results.append({"stage": sid, "script": script, "status": "skipped"})
            continue

        cmd = [python, str(ROOT / "scripts" / script), "--config", *configs]
        if script != "00_setup_environment.py":
            cmd += ["--tag", args.tag]
        cmd += extra
        if args.quick:
            cmd += quick_args

        log.info("=" * 78)
        log.info("[%s] %s", sid, description)
        log.info("      %s", " ".join(cmd[1:]))
        log.info("=" * 78)

        t0 = time.perf_counter()
        proc = subprocess.run(cmd, cwd=str(ROOT))
        dt = time.perf_counter() - t0
        status = "ok" if proc.returncode == 0 else f"failed (rc={proc.returncode})"
        log.info("[%s] %s in %.1f s", sid, status, dt)
        results.append({"stage": sid, "script": script, "status": status,
                        "duration_s": round(dt, 2)})

        if proc.returncode != 0 and not args.continue_on_error:
            log.error("Stopping: stage %s failed. Fix it and resume with "
                      "--from %s", sid, sid)
            break

    total = time.perf_counter() - t_start
    summary = {
        "tag": args.tag,
        "quick": args.quick,
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "total_duration_s": round(total, 2),
        "stages": results,
    }
    out = LOGS_DIR / f"pipeline_summary_{args.tag}.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    log.info("=" * 78)
    for r in results:
        log.info("  %-4s %-34s %s", r["stage"], r["script"], r["status"])
    log.info("Total: %.1f s -- summary at %s", total, out.name)
    return 0 if all(r["status"] in ("ok", "skipped") for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
