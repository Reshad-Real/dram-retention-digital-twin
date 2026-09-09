#!/usr/bin/env python
"""Stage 8 -- statistical validation of the surrogate and optimiser comparisons.

    python scripts/08_statistics.py --tag main

Two comparisons are tested, each with the full protocol (descriptives,
normality, Friedman omnibus, Nemenyi post-hoc with a critical-difference
diagram, pairwise Wilcoxon with Holm correction, Cliff's delta effect sizes):

* **Surrogate models** blocked by cross-validation fold, per response.
* **Optimisation algorithms** blocked by independent run, per indicator.

Outputs
    results/statistics/*.csv
    results/tables/statistics_*.{csv,tex,md}
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
from dramdt.logging_utils import RunManifest, setup_logging, stage      # noqa: E402
from dramdt.paths import (LOGS_DIR, OPT_DIR, STATS_DIR, SURROGATE_DIR,
                          TABLES_DIR, ensure_directories)               # noqa: E402
from dramdt.reporting.tables import TableSpec, export_table             # noqa: E402
from dramdt.statistics import compare_algorithms                        # noqa: E402


#: indicator -> is a larger value better?
INDICATOR_SENSE = {"hypervolume": True, "igd": False, "igd_plus": False,
                   "spacing": False, "spread": True, "epsilon": False,
                   "n_solutions": True}


def _dump(report, prefix: str, out_dir: Path, log) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, frame in (("descriptives", report.descriptives),
                        ("normality", report.normality),
                        ("posthoc", report.posthoc),
                        ("pairwise", report.pairwise),
                        ("ranks", report.ranks)):
        if frame is not None and not frame.empty:
            p = out_dir / f"{prefix}_{name}.csv"
            frame.to_csv(p, index=False)
            written[name] = str(p)
    if report.omnibus:
        (out_dir / f"{prefix}_omnibus.json").write_text(
            json.dumps(report.omnibus, indent=2, default=str), encoding="utf-8")
        written["omnibus"] = report.omnibus
    written["critical_difference"] = report.critical_difference
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", nargs="+",
                    default=["dram_1t1c_45nm_lp.yaml", "surrogate.yaml",
                             "optimization.yaml"])
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()

    ensure_directories()
    log = setup_logging("statistics", filename=f"08_statistics_{args.tag}.log")
    cfg = load_config(*args.config)
    alpha = float(cfg.get_path("statistics.alpha", 0.05))
    posthoc = str(cfg.get_path("statistics.posthoc", "nemenyi"))
    manifest = RunManifest(stage=f"statistics_{args.tag}", config_hash=cfg.hash)

    with stage(f"statistics[{args.tag}]", log, manifest,
               LOGS_DIR / f"manifest_statistics_{args.tag}.json"):
        STATS_DIR.mkdir(parents=True, exist_ok=True)

        # ============ 1. surrogate models, blocked by CV fold ==========
        folds_path = SURROGATE_DIR / f"fold_scores_{args.tag}.csv"
        if folds_path.exists():
            folds = pd.read_csv(folds_path)
            log.info("Surrogate comparison: %d fold scores over %d models, %d targets",
                     len(folds), folds["model"].nunique(), folds["target"].nunique())
            summaries = []
            for target in folds["target"].unique():
                sub = folds[folds["target"] == target]
                report = compare_algorithms(sub, indicator="r2", group_col="model",
                                            block_col="fold", higher_is_better=True,
                                            alpha=alpha, posthoc=posthoc)
                stem = f"surrogate_{target.replace('/', '_')}"
                manifest.outputs[stem] = _dump(report, stem, STATS_DIR, log)
                if not report.ranks.empty:
                    r = report.ranks.copy()
                    r.insert(0, "response", target)
                    r["critical_difference"] = report.critical_difference
                    summaries.append(r)
                    export_table(report.ranks, TableSpec(
                        name=f"statistics_surrogate_ranks_{stem}",
                        caption=f"Mean Friedman ranks of the surrogate models for "
                                f"{target} (lower is better).",
                        notes=f"Friedman p = "
                              f"{report.omnibus.get('p_value', float('nan')):.4g}; "
                              f"Nemenyi critical difference = "
                              f"{report.critical_difference:.3f} at alpha = {alpha}."),
                        TABLES_DIR)
            if summaries:
                allranks = pd.concat(summaries, ignore_index=True)
                allranks.to_csv(STATS_DIR / f"surrogate_ranks_all_{args.tag}.csv",
                                index=False)
                overall = (allranks.groupby("group")["mean_rank"].mean()
                           .sort_values().reset_index()
                           .rename(columns={"group": "model",
                                            "mean_rank": "mean_rank_across_responses"}))
                export_table(overall, TableSpec(
                    name=f"statistics_surrogate_overall_{args.tag}",
                    caption="Surrogate models ranked by mean Friedman rank across "
                            "every modelled response (lower is better)."), TABLES_DIR)
        else:
            log.warning("No surrogate fold scores at %s -- skipping", folds_path)

        # ============ 2. optimisers, blocked by run ====================
        ind_path = OPT_DIR / f"indicators_{args.tag}.csv"
        if ind_path.exists():
            ind = pd.read_csv(ind_path)
            log.info("Optimiser comparison: %d runs over %d algorithms",
                     len(ind), ind["algorithm"].nunique())
            rank_frames = []
            for indicator, higher in INDICATOR_SENSE.items():
                if indicator not in ind.columns:
                    continue
                if ind[indicator].notna().sum() < 6:
                    continue
                report = compare_algorithms(ind, indicator=indicator,
                                            group_col="algorithm", block_col="run",
                                            higher_is_better=higher, alpha=alpha,
                                            posthoc=posthoc)
                stem = f"optimizer_{indicator}"
                manifest.outputs[stem] = _dump(report, stem, STATS_DIR, log)
                if not report.ranks.empty:
                    r = report.ranks.copy()
                    r.insert(0, "indicator", indicator)
                    r["critical_difference"] = report.critical_difference
                    rank_frames.append(r)

                if not report.descriptives.empty:
                    d = report.descriptives.copy()
                    d.insert(0, "indicator", indicator)
                    export_table(d, TableSpec(
                        name=f"statistics_optimizer_{indicator}_{args.tag}",
                        caption=f"Distribution of the {indicator} indicator across "
                                f"independent optimisation runs.",
                        notes=f"Friedman p = "
                              f"{report.omnibus.get('p_value', float('nan')):.4g}; "
                              f"Kendall's W = "
                              f"{report.omnibus.get('kendalls_w', float('nan')):.4f}. "
                              f"{'Higher' if higher else 'Lower'} is better. "
                              "mean_ci_* are 95 % percentile-bootstrap intervals."),
                        TABLES_DIR)
                if not report.pairwise.empty:
                    export_table(report.pairwise, TableSpec(
                        name=f"statistics_optimizer_pairwise_{indicator}_{args.tag}",
                        caption=f"Pairwise Wilcoxon signed-rank tests on {indicator} "
                                "with Holm correction and Cliff's delta.",
                        columns=["group_1", "group_2", "n_pairs", "p_value",
                                 "p_value_holm", "significant", "cliffs_delta",
                                 "effect_magnitude", "a12"]), TABLES_DIR)

            if rank_frames:
                allr = pd.concat(rank_frames, ignore_index=True)
                allr.to_csv(STATS_DIR / f"optimizer_ranks_all_{args.tag}.csv",
                            index=False)
                export_table(allr, TableSpec(
                    name=f"statistics_optimizer_ranks_{args.tag}",
                    caption="Mean Friedman ranks of the optimisers per indicator "
                            "(lower is better).",
                    notes="Ranks are computed per run after orienting each "
                          "indicator so that a smaller value is better."), TABLES_DIR)
        else:
            log.warning("No optimiser indicators at %s -- skipping", ind_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
