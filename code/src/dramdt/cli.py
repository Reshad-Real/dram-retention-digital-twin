"""Command-line entry point (``dramdt ...``).

A thin dispatcher over the pipeline scripts plus a few inspection commands that
are useful without running anything:

    dramdt info                       show the resolved environment and config
    dramdt config <name>              summarise a configuration and its design space
    dramdt validate-palette           re-run the figure-palette accessibility checks
    dramdt algorithms <out_dir>       write the IEEE pseudocode
    dramdt run <stage> [args...]      run a pipeline stage
    dramdt run-all [args...]          run the whole pipeline
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .paths import PROJECT_ROOT

_STAGES = {
    "setup": "00_setup_environment.py",
    "validate": "01_validate_spice.py",
    "dataset": "02_generate_dataset.py",
    "assist": "03_assist_study.py",
    "surrogates": "04_train_surrogates.py",
    "optimize": "05_optimize.py",
    "explain": "06_explain.py",
    "robustness": "07_robustness.py",
    "statistics": "08_statistics.py",
    "figures": "09_figures_tables.py",
    "report": "10_report.py",
}


def _cmd_info(_args: argparse.Namespace) -> int:
    from .env import NgSpiceNotFound, resolve_environment
    from .logging_utils import package_versions

    print("dramdt environment")
    print("=" * 60)
    print(f"project root : {PROJECT_ROOT}")
    try:
        env = resolve_environment()
        for k, v in env.describe().items():
            print(f"{k:14s} : {v}")
    except NgSpiceNotFound as exc:
        print(f"ngspice        : NOT FOUND ({exc})")
    print("\npackages")
    print("-" * 60)
    for k, v in package_versions().items():
        print(f"  {k:14s} {v}")
    return 0


def _cmd_config(args: argparse.Namespace) -> int:
    from .config import load_config

    cfg = load_config(*args.name)
    print(cfg.summary())
    print()
    ds = cfg.design_space
    print(f"{'variable':18s} {'group':12s} {'kind':12s} {'low':>12s} {'high':>12s}  unit")
    print("-" * 84)
    for row in ds.describe():
        lo = f"{row['low']:.4g}" if row["low"] != "" else row["categories"][:22]
        hi = f"{row['high']:.4g}" if row["high"] != "" else ""
        print(f"{row['variable']:18s} {row['group']:12s} {row['kind']:12s} "
              f"{lo:>12s} {hi:>12s}  {row['unit']}")
    print("\nderived:")
    for k, v in ds.derived.items():
        print(f"  {k:14s} = {v}")
    print("\nobjectives:")
    for o in cfg.get("objectives", []):
        print(f"  {o['direction']:8s} {o['name']}")
    return 0


def _cmd_validate_palette(_args: argparse.Namespace) -> int:
    from .viz.palette import CATEGORICAL, validate_palette

    report = validate_palette(CATEGORICAL)
    print(report.summary())
    df = report.to_frame()
    print(f"\nminimum normal-vision distance : {df['normal'].min():.2f}  (floor 15)")
    print(f"minimum worst-case CVD distance: {df['cvd_worst'].min():.2f}  (floor  8)")
    print(f"minimum greyscale separation   : {df['greyscale'].min():.2f}")
    return 0 if report.passed else 1


def _cmd_algorithms(args: argparse.Namespace) -> int:
    from .reporting.algorithms import write_all_algorithms

    written = write_all_algorithms(args.out_dir)
    for fmt, paths in written.items():
        print(f"{fmt}: {len(paths)} files -> {Path(paths[0]).parent}")
    return 0


def _run_script(script: str, extra: list[str]) -> int:
    path = PROJECT_ROOT / "scripts" / script
    if not path.exists():
        print(f"stage script not found: {path}", file=sys.stderr)
        return 2
    return subprocess.run([sys.executable, str(path), *extra],
                          cwd=str(PROJECT_ROOT)).returncode


def _cmd_run(args: argparse.Namespace) -> int:
    if args.stage not in _STAGES:
        print(f"unknown stage {args.stage!r}; choose from: "
              f"{', '.join(_STAGES)}", file=sys.stderr)
        return 2
    return _run_script(_STAGES[args.stage], args.extra)


def _cmd_run_all(args: argparse.Namespace) -> int:
    return _run_script("run_all.py", args.extra)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dramdt", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="show the environment").set_defaults(fn=_cmd_info)

    p = sub.add_parser("config", help="summarise a configuration")
    p.add_argument("name", nargs="*", default=["dram_1t1c_45nm_lp.yaml"])
    p.set_defaults(fn=_cmd_config)

    sub.add_parser("validate-palette", help="re-run the palette checks") \
        .set_defaults(fn=_cmd_validate_palette)

    p = sub.add_parser("algorithms", help="write the IEEE pseudocode")
    p.add_argument("out_dir", nargs="?", default=str(PROJECT_ROOT / "reports" / "algorithms"))
    p.set_defaults(fn=_cmd_algorithms)

    p = sub.add_parser("run", help="run one pipeline stage")
    p.add_argument("stage", choices=list(_STAGES))
    p.add_argument("extra", nargs=argparse.REMAINDER)
    p.set_defaults(fn=_cmd_run)

    p = sub.add_parser("run-all", help="run the whole pipeline")
    p.add_argument("extra", nargs=argparse.REMAINDER)
    p.set_defaults(fn=_cmd_run_all)

    args = ap.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
