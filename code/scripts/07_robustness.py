#!/usr/bin/env python
"""Stage 7b -- robustness, yield and sensitivity analysis of the selected designs.

    python scripts/07_robustness.py --tag main

Outputs
    results/robustness/monte_carlo_<design>.parquet
    results/robustness/yield_summary_<tag>.csv
    results/robustness/corner_sweep_<tag>.parquet
    results/robustness/worst_case_<tag>.csv
    results/robustness/sobol_<tag>.csv
    results/robustness/spice_yield_check_<tag>.csv
    results/tables/robustness_*.{csv,tex,md}
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dramdt.config import load_config                                   # noqa: E402
from dramdt.digital_twin import DigitalTwin                             # noqa: E402
from dramdt.env import resolve_environment                              # noqa: E402
from dramdt.logging_utils import RunManifest, setup_logging, stage      # noqa: E402
from dramdt.models import DramCellBuilder, Technology                   # noqa: E402
from dramdt.optimization import DramDesignProblem                       # noqa: E402
from dramdt.paths import (LOGS_DIR, MODELS_DIR, OPT_DIR, ROBUSTNESS_DIR,
                          TABLES_DIR, ensure_directories)               # noqa: E402
from dramdt.reporting.tables import TableSpec, export_table             # noqa: E402
from dramdt.robustness import RobustnessAnalyzer                        # noqa: E402
from dramdt.seeds import seed_everything                                # noqa: E402
from dramdt.simulation import MetricExtractor, SpiceRunner              # noqa: E402


def make_spice_evaluator(cfg, log):
    """Return a callable that simulates a list of design dicts in NGSpice."""
    env = resolve_environment(probe=False)
    tech = Technology.from_config(cfg, generate=False)
    builder = DramCellBuilder(cfg, tech)
    extractor = MetricExtractor(cfg)
    ds = cfg.design_space

    def evaluate(designs: list[dict]) -> pd.DataFrame:
        rows = []
        with SpiceRunner(env, timeout_s=600) as runner:
            for d in designs:
                full = ds.apply_derived({**ds.constants, **d})
                p = builder.parameters_from_sample(full, sample_id=-1)
                res = runner.run(builder.build_characterization_netlist(p), tag="mc")
                rows.append(extractor.extract(res, p).to_record())
        return pd.DataFrame(rows)

    return evaluate


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", nargs="+",
                    default=["dram_1t1c_45nm_lp.yaml", "surrogate.yaml",
                             "optimization.yaml"])
    ap.add_argument("--tag", default="main")
    ap.add_argument("--max-designs", type=int, default=6)
    ap.add_argument("--skip-spice-check", action="store_true")
    ap.add_argument("--sobol-base", type=int, default=512)
    args = ap.parse_args()

    ensure_directories()
    log = setup_logging("robustness", filename=f"07_robustness_{args.tag}.log")
    cfg = load_config(*args.config)
    seeds = seed_everything(int(cfg["experiment"]["seed"]))
    manifest = RunManifest(stage=f"robustness_{args.tag}", config_hash=cfg.hash,
                           seeds=seeds)

    with stage(f"robustness[{args.tag}]", log, manifest,
               LOGS_DIR / f"manifest_robustness_{args.tag}.json"):
        twin_path = MODELS_DIR / f"digital_twin_{args.tag}.joblib"
        sel_path = OPT_DIR / f"selected_designs_{args.tag}.csv"
        if not twin_path.exists() or not sel_path.exists():
            log.error("Need %s and %s -- run stages 04 and 05 first",
                      twin_path.name, sel_path.name)
            return 2

        twin = DigitalTwin.load(twin_path, strict=False)
        spec = DramDesignProblem(cfg, twin)
        selected = pd.read_csv(sel_path)
        if selected.empty:
            log.error("No selected designs to analyse")
            return 3
        selected = selected.head(args.max_designs)

        evaluator = None if args.skip_spice_check else make_spice_evaluator(cfg, log)
        analyzer = RobustnessAnalyzer(cfg, twin, spice_evaluator=evaluator)
        ROBUSTNESS_DIR.mkdir(parents=True, exist_ok=True)

        yield_rows, worst_rows, corner_frames, spice_rows = [], [], [], []

        for i, (_, row) in enumerate(selected.iterrows()):
            design = {v: float(row[v]) for v in spec.var_names if v in row}
            design_id = f"{row.get('selection', 'design')}_{i}"
            log.info("--- %s ---", design_id)

            # nominal condition = the first configured operating condition
            cond = dict(spec.conditions[0]) if spec.conditions else {}
            mc = analyzer.monte_carlo(design, design_id, condition=cond)
            mc.samples.to_parquet(
                ROBUSTNESS_DIR / f"monte_carlo_{design_id}.parquet", index=False)
            yield_rows.append({"design_id": design_id,
                               "selection": row.get("selection", ""),
                               **mc.yields, **mc.summary})
            log.info("  twin yield: %.2f %% over %d trials",
                     mc.yields.get("yield_joint", float("nan")),
                     int(mc.yields.get("n_trials", 0)))
            if mc.spice_yields:
                spice_rows.append({
                    "design_id": design_id,
                    "twin_yield_pct": mc.yields.get("yield_joint"),
                    "spice_yield_pct": mc.spice_yields.get("yield_joint"),
                    "n_spice_trials": mc.spice_yields.get("n_trials"),
                    "absolute_difference_pp": abs(
                        (mc.yields.get("yield_joint") or 0)
                        - (mc.spice_yields.get("yield_joint") or 0)),
                })

            cs = analyzer.corner_sweep(design, design_id)
            corner_frames.append(cs.grid)
            worst_rows.append(cs.worst_case)

        pd.DataFrame(yield_rows).to_csv(
            ROBUSTNESS_DIR / f"yield_summary_{args.tag}.csv", index=False)
        if corner_frames:
            pd.concat(corner_frames, ignore_index=True).to_parquet(
                ROBUSTNESS_DIR / f"corner_sweep_{args.tag}.parquet", index=False)
        pd.DataFrame(worst_rows).to_csv(
            ROBUSTNESS_DIR / f"worst_case_{args.tag}.csv", index=False)
        if spice_rows:
            sf = pd.DataFrame(spice_rows)
            sf.to_csv(ROBUSTNESS_DIR / f"spice_yield_check_{args.tag}.csv", index=False)
            export_table(sf, TableSpec(
                name=f"robustness_spice_yield_check_{args.tag}",
                caption="Twin-predicted Monte-Carlo yield against NGSpice "
                        "re-simulation of a random subset of the same trials.",
                notes="Differences are in percentage points."), TABLES_DIR)
            manifest.outputs["max_yield_gap_pp"] = float(
                sf["absolute_difference_pp"].max())

        # ---------------- Sobol' sensitivity --------------------------
        sob = []
        for obj in spec.objectives:
            log.info("Sobol' indices for %s ...", obj.name)
            sob.append(analyzer.sensitivity(spec, obj.name, n_base=args.sobol_base))
        if sob:
            sobol = pd.concat(sob, ignore_index=True)
            sobol.to_csv(ROBUSTNESS_DIR / f"sobol_{args.tag}.csv", index=False)
            export_table(sobol, TableSpec(
                name=f"robustness_sobol_{args.tag}",
                caption="Sobol' first-order and total-effect sensitivity indices.",
                notes="S_T - S_1 is the share of variance a variable contributes "
                      "through interactions rather than on its own.",
                columns=["response", "variable", "S1", "ST", "interaction"],
                max_rows=60), TABLES_DIR)

        # ---------------- tables --------------------------------------
        yf = pd.DataFrame(yield_rows)
        ycols = ["design_id", "selection"] + \
                [c for c in yf.columns if c.startswith("yield_")]
        export_table(yf[ycols], TableSpec(
            name=f"robustness_yield_{args.tag}",
            caption="Monte-Carlo yield of the selected designs under process "
                    "variation.",
            notes="yield_joint is the fraction of trials meeting every "
                  "specification simultaneously, not the product of the "
                  "individual yields.",
            highlight_best="yield_joint"), TABLES_DIR)

        wf = pd.DataFrame(worst_rows)
        export_table(wf, TableSpec(
            name=f"robustness_worst_case_{args.tag}",
            caption="Worst-case performance of the selected designs across the "
                    "PVT corner grid, with the corner at which it occurs."),
            TABLES_DIR)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
