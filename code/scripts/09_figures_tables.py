#!/usr/bin/env python
"""Stage 9 -- render every publication figure from the saved artefacts.

    python scripts/09_figures_tables.py --tag main
    python scripts/09_figures_tables.py --rebuild-palette

Figures are written to ``results/figures/`` as PNG (300 dpi), PDF and SVG.
The script is tolerant: any figure whose input artefact is missing is skipped
with a warning, and the manifest records exactly which figures were produced.
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
from dramdt.paths import (FIGURES_DIR, LOGS_DIR, MODELS_DIR, OPT_DIR,
                          PROCESSED_DIR, ROBUSTNESS_DIR, STATS_DIR,
                          SURROGATE_DIR, TABLES_DIR, XAI_DIR,
                          ensure_directories)                           # noqa: E402
from dramdt.reporting.tables import TableSpec, export_table             # noqa: E402
from dramdt.viz import apply_style                                      # noqa: E402
from dramdt.viz import figures as F                                     # noqa: E402
from dramdt.viz.palette import (CATEGORICAL, design_palette,            # noqa: E402
                                validate_palette)

RESPONSES = ["retention_time_s", "read_delay_ns", "write_delay_ns",
             "read_margin_mv", "total_power_nw", "energy_per_access_fj",
             "leakage_current_fa", "v_sn_written_v"]
LOG_RESPONSES = ["retention_time_s", "total_power_nw", "leakage_current_fa"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", nargs="+",
                    default=["dram_1t1c_45nm_lp.yaml", "surrogate.yaml",
                             "optimization.yaml"])
    ap.add_argument("--tag", default="main")
    ap.add_argument("--rebuild-palette", action="store_true",
                    help="recompute and re-validate the categorical palette")
    args = ap.parse_args()

    ensure_directories()
    log = setup_logging("figures", filename=f"09_figures_{args.tag}.log")
    cfg = load_config(*args.config)
    apply_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    manifest = RunManifest(stage=f"figures_{args.tag}", config_hash=cfg.hash)
    produced: dict[str, list[str]] = {}

    def record(name: str, paths) -> None:
        if paths:
            produced[name] = [str(p) for p in paths]
            log.info("  figure %-38s -> %d files", name, len(paths))
        else:
            log.warning("  figure %-38s skipped (no data)", name)

    with stage(f"figures[{args.tag}]", log, manifest,
               LOGS_DIR / f"manifest_figures_{args.tag}.json"):

        # ---------------- palette audit -------------------------------
        if args.rebuild_palette:
            computed = design_palette(8)
            log.info("Recomputed palette: %s", computed)
        report = validate_palette(CATEGORICAL)
        log.info("Palette validation: %s", report.summary())
        export_table(report.to_frame(), TableSpec(
            name="figure_palette_validation",
            caption="Accessibility validation of the categorical figure palette.",
            notes="Distances are OKLab Euclidean x100. 'cvd_worst' is the minimum "
                  "over simulated protanopia, deuteranopia and tritanopia; the "
                  "acceptance floors are 15 (normal vision) and 8 (CVD). "
                  "'greyscale' is the luminance separation after greyscale "
                  "conversion."), TABLES_DIR)
        manifest.outputs["palette_passed"] = report.passed

        # ---------------- stage 3: dataset ----------------------------
        proc_path = PROCESSED_DIR / f"{args.tag}_processed.parquet"
        df = pd.read_parquet(proc_path) if proc_path.exists() else pd.DataFrame()
        if not df.empty:
            ds = cfg.design_space
            variables = [v for v in ds.variables if v.kind != "categorical"][:5]
            names = [v.name for v in variables]
            if all(n in df for n in names):
                # Map back through each variable's OWN encoding, so a log-scaled
                # variable is shown on its sampling scale.  Min-max normalising
                # the physical values instead would make a perfectly uniform
                # log-scaled design look strongly skewed.
                U = np.column_stack([
                    np.array([v.encode(x) for x in df[v.name].to_numpy(dtype=float)])
                    for v in variables])
                record("design_space_coverage",
                       F.fig_design_space_coverage(U, names,
                                                   FIGURES_DIR / "fig01_design_space_coverage"))
            record("response_distributions",
                   F.fig_response_distributions(df, RESPONSES,
                                                FIGURES_DIR / "fig02_response_distributions",
                                                log_scale=LOG_RESPONSES))
            corr_cols = [v.name for v in cfg.design_space.variables
                         if v.kind != "categorical"] + \
                        [c for c in RESPONSES if c in df.columns]
            record("correlation_matrix",
                   F.fig_correlation_matrix(df, corr_cols,
                                            FIGURES_DIR / "fig03_correlation_matrix"))
        else:
            log.warning("Processed dataset not found: %s", proc_path)

        val_path = TABLES_DIR / f"retention_validation_{args.tag}.csv"
        if not val_path.exists():
            val_path = TABLES_DIR / "validation_retention_model.csv"
        if val_path.exists():
            v = pd.read_csv(val_path)
            # Compute the agreement statistics from the very frame being
            # plotted.  Reading them from another stage's table would annotate
            # this scatter with a different study's sample size.
            from dramdt.simulation.retention import validate_against_transient
            stats = validate_against_transient(v["retention_quasistatic_s"],
                                               v["retention_transient_s"])
            record("retention_validation",
                   F.fig_retention_validation(v, FIGURES_DIR / "fig04_retention_validation",
                                              stats))

        # ---------------- stage 4: surrogates -------------------------
        lb_path = SURROGATE_DIR / f"leaderboard_{args.tag}.csv"
        if lb_path.exists():
            lb = pd.read_csv(lb_path)
            record("surrogate_comparison",
                   F.fig_surrogate_comparison(lb, FIGURES_DIR / "fig05_surrogate_comparison"))
        fold_path = SURROGATE_DIR / f"fold_scores_{args.tag}.csv"
        if fold_path.exists():
            folds = pd.read_csv(fold_path)
            for i, target in enumerate(folds["target"].unique()):
                record(f"cv_boxplot_{target}",
                       F.fig_cv_boxplots(folds, FIGURES_DIR /
                                         f"fig06{chr(97+i)}_cv_{target}", target=target))
        pred_path = SURROGATE_DIR / f"predictions_{args.tag}.parquet"
        if pred_path.exists():
            preds = pd.read_parquet(pred_path)
            pairs, metrics = {}, {}
            from dramdt.surrogate.metrics import regression_metrics
            for target, sub in preds.groupby("target"):
                pairs[target] = (sub["y_true"].to_numpy(), sub["y_pred"].to_numpy())
                metrics[target] = regression_metrics(sub["y_true"], sub["y_pred"])
            record("predicted_vs_actual",
                   F.fig_predicted_vs_actual(pairs, FIGURES_DIR / "fig07_parity",
                                             metrics))
            record("residuals", F.fig_residuals(pairs, FIGURES_DIR / "fig08_residuals"))
        bench_path = SURROGATE_DIR / f"twin_benchmark_{args.tag}.csv"
        if bench_path.exists():
            record("twin_speedup",
                   F.fig_twin_speedup(pd.read_csv(bench_path),
                                      FIGURES_DIR / "fig09_twin_speedup"))

        # ---------------- stage 4b: XAI -------------------------------
        for i, npz in enumerate(sorted(XAI_DIR.glob("shap_values_*.npz"))):
            target = npz.stem.replace("shap_values_", "")
            data = np.load(npz, allow_pickle=True)
            record(f"shap_beeswarm_{target}",
                   F.fig_shap_beeswarm(data["shap_values"], data["data"],
                                       list(data["features"]),
                                       FIGURES_DIR / f"fig10{chr(97+i)}_shap_beeswarm_{target}",
                                       target=target))
        for i, csv in enumerate(sorted(XAI_DIR.glob("shap_importance_*.csv"))):
            if csv.stem.startswith("shap_importance_all"):
                continue
            target = csv.stem.replace("shap_importance_", "")
            record(f"shap_importance_{target}",
                   F.fig_shap_importance(pd.read_csv(csv),
                                         FIGURES_DIR / f"fig11{chr(97+i)}_shap_importance_{target}",
                                         target=target))
        for i, csv in enumerate(sorted(XAI_DIR.glob("shap_interactions_*.csv"))):
            target = csv.stem.replace("shap_interactions_", "")
            record(f"shap_interactions_{target}",
                   F.fig_shap_interactions(pd.read_csv(csv),
                                           FIGURES_DIR / f"fig12{chr(97+i)}_shap_interactions_{target}",
                                           target=target))
        for i, csv in enumerate(sorted(XAI_DIR.glob("partial_dependence_[!2]*.csv"))):
            target = csv.stem.replace("partial_dependence_", "")
            ale_path = XAI_DIR / f"ale_{target}.csv"
            ale = pd.read_csv(ale_path) if ale_path.exists() else None
            record(f"partial_dependence_{target}",
                   F.fig_partial_dependence(pd.read_csv(csv),
                                            FIGURES_DIR / f"fig13{chr(97+i)}_pdp_{target}",
                                            target=target, ale=ale))

        # ---------------- stage 5: optimisation -----------------------
        pareto_path = OPT_DIR / f"pareto_{args.tag}.csv"
        objectives = [o["name"] for o in cfg.get("objectives", [])]
        labels = {o["name"]: o.get("label", o["name"]) for o in cfg.get("objectives", [])}
        if pareto_path.exists():
            front = pd.read_csv(pareto_path)
            record("pareto_matrix",
                   F.fig_pareto_matrix(front, objectives,
                                       FIGURES_DIR / "fig14_pareto_matrix",
                                       colour_by="found_by", labels=labels))
            record("parallel_coordinates",
                   F.fig_parallel_coordinates(front, objectives,
                                              FIGURES_DIR / "fig15_parallel_coordinates",
                                              colour_by="found_by"))
            sel_path = OPT_DIR / f"selected_designs_{args.tag}.csv"
            sel = pd.read_csv(sel_path) if sel_path.exists() else None
            if len(objectives) >= 2:
                record("tradeoff",
                       F.fig_tradeoff_scatter(front, objectives[2] if len(objectives) > 2
                                              else objectives[0], objectives[0],
                                              FIGURES_DIR / "fig16_tradeoff",
                                              colour_by=objectives[1] if len(objectives) > 1 else None,
                                              highlight=sel))
        conv_path = OPT_DIR / f"convergence_{args.tag}.csv"
        if conv_path.exists():
            record("hypervolume_convergence",
                   F.fig_hypervolume_convergence(pd.read_csv(conv_path),
                                                 FIGURES_DIR / "fig17_convergence"))
        ind_path = OPT_DIR / f"indicators_{args.tag}.csv"
        if ind_path.exists():
            record("indicator_boxplots",
                   F.fig_indicator_boxplots(pd.read_csv(ind_path),
                                            FIGURES_DIR / "fig18_indicators"))

        # ---------------- stage 8: critical-difference diagrams -------
        for i, csv in enumerate(sorted(STATS_DIR.glob("*_ranks.csv"))):
            ranks = pd.read_csv(csv)
            if ranks.empty:
                continue
            cd_source = csv.with_name(csv.name.replace("_ranks", "_posthoc"))
            cd = float("nan")
            if cd_source.exists():
                ph = pd.read_csv(cd_source)
                if "critical_difference" in ph.columns and len(ph):
                    cd = float(ph["critical_difference"].iloc[0])
            name = csv.stem.replace("_ranks", "")
            record(f"critical_difference_{name}",
                   F.fig_critical_difference(ranks, cd,
                                             FIGURES_DIR / f"fig19{chr(97+i)}_cd_{name}",
                                             title=f"Critical-difference diagram: {name}"))

        # ---------------- stage 7: robustness -------------------------
        corner_path = ROBUSTNESS_DIR / f"corner_sweep_{args.tag}.parquet"
        if corner_path.exists():
            grid = pd.read_parquet(corner_path)
            for i, design_id in enumerate(grid["design_id"].unique()[:3]):
                sub = grid[grid["design_id"] == design_id]
                for j, resp in enumerate(["retention_time_s", "read_margin_mv"]):
                    if resp in sub.columns:
                        record(f"corner_{design_id}_{resp}",
                               F.fig_corner_heatmap(sub, resp,
                                                    FIGURES_DIR /
                                                    f"fig20{chr(97+i)}{j}_corner_{design_id}_{resp}",
                                                    design_id=str(design_id)))
        specs = dict(cfg.get_path("robustness.yield_thresholds", {}) or {})
        directions = {"read_margin_mv": "min", "write_efficiency": "min",
                      "retention_time_s": "min", "read_delay_ns": "max"}
        for i, mc in enumerate(sorted(ROBUSTNESS_DIR.glob("monte_carlo_*.parquet"))[:3]):
            design_id = mc.stem.replace("monte_carlo_", "")
            record(f"monte_carlo_{design_id}",
                   F.fig_monte_carlo(pd.read_parquet(mc), specs,
                                     FIGURES_DIR / f"fig21{chr(97+i)}_monte_carlo_{design_id}",
                                     design_id=design_id, directions=directions))
        sobol_path = ROBUSTNESS_DIR / f"sobol_{args.tag}.csv"
        if sobol_path.exists():
            record("sobol_indices",
                   F.fig_sobol_indices(pd.read_csv(sobol_path),
                                       FIGURES_DIR / "fig22_sobol"))

        # ---------------- stage 7: assist techniques -------------------
        assist_path = PROCESSED_DIR / f"assist_study_{args.tag}.parquet"
        if assist_path.exists():
            assist = pd.read_parquet(assist_path)
            record("assist_comparison",
                   F.fig_assist_comparison(
                       assist,
                       ["retention_time_s", "read_delay_ns", "read_margin_mv",
                        "total_power_nw", "energy_per_access_fj"],
                       FIGURES_DIR / "fig23_assist_comparison"))

        manifest.outputs["figures"] = produced
        log.info("Produced %d figures (%d files)", len(produced),
                 sum(len(v) for v in produced.values()))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
