#!/usr/bin/env python
"""Stage 5 -- multi-objective design optimisation through the digital twin.

    python scripts/05_optimize.py
    python scripts/05_optimize.py --algorithms nsga2 nsga3 --runs 5

Outputs
    results/optimization/pareto_<tag>.csv          global Pareto set (physical units)
    results/optimization/indicators_<tag>.csv      per-run indicator values
    results/optimization/convergence_<tag>.csv     hypervolume traces
    results/optimization/selected_designs_<tag>.csv  knee/weighted picks
    results/optimization/spice_verification_<tag>.csv  selected designs re-simulated
    results/tables/optimization_*.{csv,tex,md}
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
from dramdt.digital_twin import DigitalTwin                             # noqa: E402
from dramdt.env import resolve_environment                              # noqa: E402
from dramdt.logging_utils import RunManifest, setup_logging, stage      # noqa: E402
from dramdt.models import DramCellBuilder, Technology                   # noqa: E402
from dramdt.optimization import (DramDesignProblem, OptimizationRunner)  # noqa: E402
from dramdt.paths import (LOGS_DIR, MODELS_DIR, OPT_DIR, TABLES_DIR,
                          ensure_directories)                           # noqa: E402
from dramdt.reporting.tables import TableSpec, export_table             # noqa: E402
from dramdt.seeds import seed_everything                                # noqa: E402
from dramdt.simulation import MetricExtractor, SpiceRunner              # noqa: E402


# --------------------------------------------------------------------------
def select_designs(front: pd.DataFrame, spec, cfg, log) -> pd.DataFrame:
    """Pick representative designs: knee point, weighted best, per-objective best."""
    sel_cfg = dict(cfg.get_path("optimization.selection", {}) or {})
    n_report = int(sel_cfg.get("n_report", 12))
    weights = np.asarray(sel_cfg.get("weights") or
                         [1.0 / spec.n_obj] * spec.n_obj, dtype=float)
    weights = weights / weights.sum()

    feasible = front[front["feasible"]] if "feasible" in front.columns else front
    if feasible.empty:
        log.warning("No feasible Pareto solutions; selecting from the full set")
        feasible = front
    if feasible.empty:
        return pd.DataFrame()

    F = np.column_stack([obj.sign * pd.to_numeric(feasible[obj.name], errors="coerce")
                         for obj in spec.objectives]).astype(float)
    ok = np.all(np.isfinite(F), axis=1)
    F, sub = F[ok], feasible[ok]
    if len(F) == 0:
        return pd.DataFrame()

    lo, hi = F.min(axis=0), F.max(axis=0)
    N = (F - lo) / np.where(hi - lo > 0, hi - lo, 1.0)

    picks: list[tuple[str, int]] = []
    # knee point: minimum distance to the ideal corner of the normalised space
    picks.append(("knee", int(np.argmin(np.linalg.norm(N, axis=1)))))
    # weighted scalarisation
    picks.append(("weighted", int(np.argmin(N @ weights))))
    # best single objective
    for j, obj in enumerate(spec.objectives):
        picks.append((f"best_{obj.name}", int(np.argmin(N[:, j]))))
    # a spread of additional compromise solutions
    remaining = n_report - len(picks)
    if remaining > 0 and len(N) > len(picks):
        order = np.argsort(N @ weights)
        for idx in order:
            if remaining <= 0:
                break
            if idx not in [p[1] for p in picks]:
                picks.append(("compromise", int(idx)))
                remaining -= 1

    rows = []
    for label, idx in picks:
        r = sub.iloc[idx].to_dict()
        r["selection"] = label
        rows.append(r)
    out = pd.DataFrame(rows).drop_duplicates(
        subset=[v for v in spec.var_names if v in sub.columns], keep="first")
    log.info("Selected %d representative designs", len(out))
    return out


def verify_in_spice(designs: pd.DataFrame, cfg, spec, log) -> pd.DataFrame:
    """Re-simulate the selected designs in NGSpice at every condition."""
    env = resolve_environment(probe=False)
    tech = Technology.from_config(cfg, generate=False)
    builder = DramCellBuilder(cfg, tech)
    extractor = MetricExtractor(cfg)

    rows = []
    with SpiceRunner(env, timeout_s=600) as runner:
        for _, row in designs.iterrows():
            for cond in spec.conditions:
                sample = {v: row[v] for v in spec.var_names if v in row}
                sample.update(cond)
                full = cfg.design_space.apply_derived(
                    {**cfg.design_space.constants, **sample})
                p = builder.parameters_from_sample(full, sample_id=-1)
                res = runner.run(builder.build_characterization_netlist(p), tag="verify")
                met = extractor.extract(res, p).to_record()
                rec = {"selection": row.get("selection", ""),
                       "found_by": row.get("found_by", ""), **cond}
                for obj in spec.objectives:
                    rec[f"{obj.name}_predicted"] = row.get(f"{obj.name}@" +
                                                           "_".join(f"{k}{v}" for k, v in cond.items()),
                                                           np.nan)
                    rec[f"{obj.name}_simulated"] = met.get(obj.name, np.nan)
                for c in spec.constraints:
                    rec[f"{c.name}_simulated"] = met.get(c.name, np.nan)
                rec["read_success"] = met.get("read_success")
                rec["write_success"] = met.get("write_success")
                rows.append(rec)
    frame = pd.DataFrame(rows)

    for obj in spec.objectives:
        pc, sc = f"{obj.name}_predicted", f"{obj.name}_simulated"
        if pc in frame and sc in frame:
            a = pd.to_numeric(frame[pc], errors="coerce").to_numpy(float)
            b = pd.to_numeric(frame[sc], errors="coerce").to_numpy(float)
            m = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 1e-30)
            if m.any():
                err = 100 * np.abs(a[m] - b[m]) / np.abs(b[m])
                log.info("  %-22s twin vs SPICE: median %.2f %%, p90 %.2f %% (n=%d)",
                         obj.name, float(np.median(err)),
                         float(np.percentile(err, 90)), int(m.sum()))
                frame[f"{obj.name}_rel_error_pct"] = np.where(
                    np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 1e-30),
                    100 * np.abs(a - b) / np.where(np.abs(b) > 1e-30, np.abs(b), np.nan),
                    np.nan)
    return frame


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", nargs="+",
                    default=["dram_1t1c_45nm_lp.yaml", "surrogate.yaml",
                             "optimization.yaml"])
    ap.add_argument("--tag", default="main")
    ap.add_argument("--twin", default=None)
    ap.add_argument("--algorithms", nargs="*", default=None)
    ap.add_argument("--runs", type=int, default=None)
    ap.add_argument("--generations", type=int, default=None)
    ap.add_argument("--population", type=int, default=None)
    ap.add_argument("--skip-verification", action="store_true")
    args = ap.parse_args()

    ensure_directories()
    log = setup_logging("optimize", filename=f"05_optimize_{args.tag}.log")
    cfg = load_config(*args.config)
    seeds = seed_everything(int(cfg["experiment"]["seed"]))

    for key, val in (("algorithms", args.algorithms), ("n_runs", args.runs),
                     ("n_generations", args.generations),
                     ("population_size", args.population)):
        if val:
            cfg["optimization"][key] = val

    manifest = RunManifest(stage=f"optimize_{args.tag}", config_hash=cfg.hash, seeds=seeds)

    with stage(f"optimize[{args.tag}]", log, manifest,
               LOGS_DIR / f"manifest_optimize_{args.tag}.json"):
        twin_path = Path(args.twin) if args.twin else \
            MODELS_DIR / f"digital_twin_{args.tag}.joblib"
        if not twin_path.exists():
            log.error("Digital twin not found: %s -- run scripts/04_train_surrogates.py",
                      twin_path)
            return 2
        twin = DigitalTwin.load(twin_path, strict=False)
        log.info("Twin: %s", json.dumps(twin.describe(), indent=2, default=str))

        spec = DramDesignProblem(cfg, twin)
        log.info("Problem: %s", json.dumps(spec.describe(), indent=2, default=str))
        manifest.parameters["problem"] = spec.describe()

        runner = OptimizationRunner(spec, cfg)
        log.info("Budget: %d evaluations per run, %d runs, algorithms %s",
                 runner.budget, runner.n_runs, runner.names)
        result = runner.run()
        if not result.runs:
            return 3

        OPT_DIR.mkdir(parents=True, exist_ok=True)
        result.pareto_solutions.to_csv(OPT_DIR / f"pareto_{args.tag}.csv", index=False)
        result.indicator_table.to_csv(OPT_DIR / f"indicators_{args.tag}.csv", index=False)
        result.convergence.to_csv(OPT_DIR / f"convergence_{args.tag}.csv", index=False)
        np.savetxt(OPT_DIR / f"reference_front_{args.tag}.csv",
                   result.reference_front, delimiter=",")
        manifest.outputs["settings"] = result.settings
        manifest.outputs["best_algorithm"] = result.best_algorithm
        manifest.outputs["n_pareto"] = int(len(result.pareto_solutions))

        selected = select_designs(result.pareto_solutions, spec, cfg, log)
        selected.to_csv(OPT_DIR / f"selected_designs_{args.tag}.csv", index=False)

        if not args.skip_verification and not selected.empty:
            log.info("Re-simulating the selected designs in NGSpice ...")
            verification = verify_in_spice(selected, cfg, spec, log)
            verification.to_csv(OPT_DIR / f"spice_verification_{args.tag}.csv", index=False)
            export_table(verification, TableSpec(
                name=f"optimization_spice_verification_{args.tag}",
                caption="Twin-predicted versus SPICE-simulated performance of the "
                        "selected Pareto designs, at every operating condition.",
                max_rows=40), TABLES_DIR)
            err_cols = [c for c in verification.columns if c.endswith("_rel_error_pct")]
            if err_cols:
                manifest.outputs["verification_median_error_pct"] = {
                    c: float(np.nanmedian(verification[c])) for c in err_cols}

        # ---------------- tables --------------------------------------
        summary = (result.indicator_table
                   .groupby(["algorithm", "label"])
                   .agg(runs=("run", "count"),
                        hv_median=("hypervolume", "median"),
                        hv_mean=("hypervolume", "mean"),
                        hv_std=("hypervolume", "std"),
                        igd_plus_median=("igd_plus", "median"),
                        spacing_median=("spacing", "median"),
                        spread_median=("spread", "median"),
                        n_solutions_median=("n_solutions", "median"),
                        runtime_s_median=("runtime_s", "median"))
                   .reset_index()
                   .sort_values("hv_median", ascending=False))
        export_table(summary, TableSpec(
            name=f"optimization_algorithm_summary_{args.tag}",
            caption="Multi-objective optimiser comparison over independent runs "
                    "under an identical evaluation budget.",
            notes=f"Budget {result.settings['evaluation_budget']} surrogate "
                  f"evaluations per run, {result.settings['n_runs']} runs per "
                  "algorithm. Hypervolume and IGD+ use a reference front built "
                  "from the non-dominated union of every run.",
            highlight_best="hv_median"), TABLES_DIR)

        cols = [v for v in spec.var_names] + [o.name for o in spec.objectives]
        cols = ["selection", "found_by"] + [c for c in cols if c in selected.columns]
        export_table(selected[cols] if not selected.empty else selected,
                     TableSpec(name=f"optimization_selected_designs_{args.tag}",
                               caption="Representative Pareto-optimal DRAM designs.",
                               notes="Objective values are worst case over the "
                                     "configured PVT conditions."), TABLES_DIR)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
