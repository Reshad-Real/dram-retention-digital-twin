#!/usr/bin/env python
"""Stage 2/3 -- run the SPICE sampling campaign and build the published dataset.

    python scripts/02_generate_dataset.py --config dram_1t1c_45nm_lp.yaml
    python scripts/02_generate_dataset.py --n-samples 2000 --tag pilot
    python scripts/02_generate_dataset.py --no-resume          # start clean

Outputs
    data/raw/<tag>_raw.parquet                full campaign output
    data/processed/<tag>_processed.parquet    cleaned + engineered + split
    data/processed/dram_dataset_<tag>.xlsx    repository-ready workbook
    data/processed/dram_dataset_<tag>.csv.gz  CSV mirror
    results/tables/preprocess_report_<tag>.csv
    logs/manifest_dataset_<tag>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dramdt.config import load_config                                   # noqa: E402
from dramdt.dataset import Preprocessor, SimulationCampaign, export_dataset  # noqa: E402
from dramdt.env import resolve_environment                              # noqa: E402
from dramdt.logging_utils import RunManifest, setup_logging, stage      # noqa: E402
from dramdt.paths import (INTERIM_DIR, LOGS_DIR, PROCESSED_DIR, RAW_DIR,
                          TABLES_DIR, ensure_directories)               # noqa: E402
from dramdt.seeds import seed_everything                                # noqa: E402
from dramdt.simulation.retention import validate_against_transient      # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", nargs="+", default=["dram_1t1c_45nm_lp.yaml"])
    ap.add_argument("--n-samples", type=int, default=None,
                    help="override experiment.n_samples")
    ap.add_argument("--tag", default=None, help="output name tag (default: config name)")
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore completed chunks and re-simulate everything")
    ap.add_argument("--skip-retention-validation", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--scratch", default=None, help="scratch directory for SPICE decks")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    ensure_directories()

    cfg = load_config(*args.config)
    tag = args.tag or cfg.name
    log = setup_logging("dataset", filename=f"02_generate_dataset_{tag}.log")

    if args.n_samples is not None:
        cfg["experiment"]["n_samples"] = int(args.n_samples)
    if args.workers is not None:
        cfg["experiment"]["n_workers"] = int(args.workers)

    n_samples = int(cfg["experiment"]["n_samples"])
    seed = int(cfg["experiment"]["seed"])
    seeds = seed_everything(seed)

    manifest = RunManifest(stage=f"dataset_{tag}", config_hash=cfg.hash, seeds=seeds,
                           parameters={"config": [str(s) for s in cfg.sources],
                                       "n_samples": n_samples, "tag": tag})

    with stage(f"dataset[{tag}]", log, manifest,
               LOGS_DIR / f"manifest_dataset_{tag}.json"):
        env = resolve_environment()
        log.info("Environment: %s", json.dumps(env.describe(), indent=2))
        manifest.environment["ngspice"] = env.describe()

        # ---------------- 1. campaign --------------------------------
        campaign = SimulationCampaign(
            cfg, environment=env,
            chunk_dir=INTERIM_DIR / f"chunks_{tag}",
            scratch_root=args.scratch)
        log.info("Technology: %s", json.dumps(campaign.tech.describe(), indent=2))

        result = campaign.run(n_samples=n_samples, resume=not args.no_resume)
        raw = result.frame
        manifest.outputs["campaign"] = result.describe()

        raw_path = RAW_DIR / f"{tag}_raw.parquet"
        raw.to_parquet(raw_path, index=False)
        log.info("Raw campaign output -> %s (%d rows)", raw_path.name, len(raw))
        manifest.outputs["raw_dataset"] = str(raw_path)

        # ---------------- 2. retention model validation ---------------
        validation = pd.DataFrame()
        agreement: dict[str, float] = {}
        if not args.skip_retention_validation:
            validation = campaign.run_retention_validation(raw)
            if not validation.empty:
                agreement = validate_against_transient(
                    validation["retention_quasistatic_s"],
                    validation["retention_transient_s"])
                vpath = TABLES_DIR / f"retention_validation_{tag}.csv"
                validation.to_csv(vpath, index=False)
                log.info("Retention model agreement vs direct transient: %s",
                         json.dumps({k: round(v, 5) for k, v in agreement.items()}))
                manifest.outputs["retention_validation"] = {
                    "table": str(vpath), **agreement}

        # ---------------- 3. preprocessing ---------------------------
        pre = Preprocessor(cfg)
        processed, report = pre.run(raw)
        proc_path = PROCESSED_DIR / f"{tag}_processed.parquet"
        processed.to_parquet(proc_path, index=False)
        manifest.outputs["processed_dataset"] = str(proc_path)
        manifest.outputs["preprocess_report"] = report.to_dict()

        rep_rows = [{"item": k, "value": json.dumps(v) if isinstance(v, (dict, list)) else v}
                    for k, v in report.to_dict().items()]
        pd.DataFrame(rep_rows).to_csv(
            TABLES_DIR / f"preprocess_report_{tag}.csv", index=False)

        # ---------------- 4. publication export -----------------------
        ret_note = (
            "retention_time_s is obtained by integrating the charge-loss ODE "
            "dV/dt = -I_leak(V)/C_node over the storage-node leakage characteristic "
            "I_leak(V), which is MEASURED by a DC sweep of a leakage-replica probe "
            "in the same NGSpice run (so it contains subthreshold, GIDL, junction and "
            "gate-tunnelling components at that sample's own temperature and corner). "
            "The model was cross-checked against direct long-window transient "
            "simulations")
        if agreement:
            ret_note += (f": n={int(agreement.get('n_pairs', 0))} design points, "
                         f"Pearson r(log10)={agreement.get('pearson_r_log10', float('nan')):.5f}, "
                         f"median |log10 error|={agreement.get('median_abs_log10_error', float('nan')):.4f}, "
                         f"{agreement.get('within_factor_2_pct', float('nan')):.1f} % within a factor of 2.")
        else:
            ret_note += " (see results/tables/retention_validation_*.csv)."

        meta = {
            "title": f"SPICE-characterised 1T1C DRAM design dataset ({cfg.name})",
            "description": str(cfg.get_path("meta.description", "")),
            "simulator": env.describe().get("ngspice_version", "ngspice"),
            "device_models": str(cfg.get_path("technology.citation", "")),
            "model_fingerprint": campaign.tech.fingerprint(),
            "sampling": (f"{result.design.sampler.upper()} design, "
                         f"{result.design.n} points in {result.design.d} dimensions, "
                         f"centred-L2 discrepancy "
                         f"{result.design.diagnostics.get('centered_l2_discrepancy', float('nan')):.6g}"),
            "seed": seed,
            "config_hash": cfg.hash,
            "framework": "dramdt v1.0.0",
            "retention_note": ret_note,
            "licence": "CC BY 4.0",
            "citation": "Please cite the accompanying article and this dataset record.",
        }

        extra = {}
        if not validation.empty:
            extra["Retention_Validation"] = validation
        if agreement:
            extra["Retention_Agreement"] = pd.DataFrame(
                [{"statistic": k, "value": v} for k, v in agreement.items()])

        export = export_dataset(
            processed,
            excel_path=PROCESSED_DIR / f"dram_dataset_{tag}.xlsx",
            csv_path=PROCESSED_DIR / f"dram_dataset_{tag}.csv.gz",
            meta=meta,
            design_space_rows=cfg.design_space.describe(),
            provenance={
                "config_sources": ", ".join(str(s.name) for s in cfg.sources),
                "config_hash": cfg.hash,
                "master_seed": seed,
                "ngspice": env.describe().get("ngspice_exe", ""),
                "model_card": str(campaign.tech.base_card),
                "model_fingerprint": campaign.tech.fingerprint(),
                "n_workers": result.diagnostics.get("n_workers"),
                "campaign_duration_s": round(result.duration_s, 1),
                "sampler": result.design.sampler,
                **{f"design::{k}": v for k, v in result.design.diagnostics.items()},
            },
            extra_sheets=extra)
        manifest.outputs["export"] = {
            "excel": str(export.excel_path), "csv": str(export.csv_path),
            "rows": export.n_rows, "columns": export.n_columns,
            "size_mb": export.size_mb}

        log.info("Dataset ready: %d rows x %d columns (%.1f MB)",
                 export.n_rows, export.n_columns, export.size_mb)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
