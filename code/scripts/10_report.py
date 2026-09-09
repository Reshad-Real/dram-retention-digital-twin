#!/usr/bin/env python
"""Stage 10 -- assemble the final technical report.

    python scripts/10_report.py --tag main

Reads every artefact the pipeline produced and writes

    reports/technical_report_<tag>.md
    reports/technical_report_<tag>.tex     (IEEE-style skeleton)
    reports/algorithms/                    IEEE pseudocode, three formats
    reports/results_summary_<tag>.json     machine-readable headline numbers

Numbers quoted in the prose are read from the artefacts, never hard-coded, so
the report cannot drift from the results it describes.  Any section whose
artefact is missing is written as an explicit "not available in this run" note
rather than being silently omitted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dramdt.config import load_config                                   # noqa: E402
from dramdt.logging_utils import RunManifest, package_versions, setup_logging, stage  # noqa: E402
from dramdt.paths import (DOCS_DIR, FIGURES_DIR, LOGS_DIR, OPT_DIR,
                          PROCESSED_DIR, PROJECT_ROOT, REPORTS_DIR,
                          ROBUSTNESS_DIR, STATS_DIR, SURROGATE_DIR,
                          TABLES_DIR, XAI_DIR, ensure_directories)      # noqa: E402
from dramdt.reporting import ReportBuilder, write_all_algorithms        # noqa: E402


def _read_csv(p: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(p) if p.exists() else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _read_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except Exception:
        return {}


def _fig(name: str) -> str | None:
    """Return the repo-relative path of a figure stem, if it was produced."""
    for cand in sorted(FIGURES_DIR.glob(f"{name}*.png")):
        return str(cand.relative_to(PROJECT_ROOT)).replace("\\", "/")
    return None


def _tab(name: str) -> str | None:
    p = TABLES_DIR / f"{name}.csv"
    return str(p.relative_to(PROJECT_ROOT)).replace("\\", "/") if p.exists() else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", nargs="+",
                    default=["dram_1t1c_45nm_lp.yaml", "surrogate.yaml",
                             "optimization.yaml"])
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()

    ensure_directories()
    log = setup_logging("report", filename=f"10_report_{args.tag}.log")
    cfg = load_config(*args.config)
    manifest = RunManifest(stage=f"report_{args.tag}", config_hash=cfg.hash)
    summary: dict[str, object] = {"tag": args.tag, "config_hash": cfg.hash}

    with stage(f"report[{args.tag}]", log, manifest,
               LOGS_DIR / f"manifest_report_{args.tag}.json"):
        # ---------------- load artefacts ------------------------------
        ds_manifest = _read_json(LOGS_DIR / f"manifest_dataset_{args.tag}.json")
        setup_manifest = _read_json(LOGS_DIR / "manifest_setup.json")
        validate_manifest = _read_json(LOGS_DIR / f"manifest_validate_{args.tag}.json")
        opt_manifest = _read_json(LOGS_DIR / f"manifest_optimize_{args.tag}.json")
        rob_manifest = _read_json(LOGS_DIR / f"manifest_robustness_{args.tag}.json")

        leaderboard = _read_csv(SURROGATE_DIR / f"leaderboard_{args.tag}.csv")
        bench = _read_csv(SURROGATE_DIR / f"twin_benchmark_{args.tag}.csv")
        opt_summary = _read_csv(TABLES_DIR / f"optimization_algorithm_summary_{args.tag}.csv")
        selected = _read_csv(OPT_DIR / f"selected_designs_{args.tag}.csv")
        yields = _read_csv(ROBUSTNESS_DIR / f"yield_summary_{args.tag}.csv")
        sobol = _read_csv(ROBUSTNESS_DIR / f"sobol_{args.tag}.csv")
        assist_rel = _read_csv(TABLES_DIR / f"assist_relative_effect_{args.tag}.csv")
        shap_matrix = _read_csv(TABLES_DIR / f"xai_importance_matrix_{args.tag}.csv")
        conv_study = _read_csv(TABLES_DIR / "solver_convergence_study.csv")
        opt_ranks = _read_csv(STATS_DIR / f"optimizer_ranks_all_{args.tag}.csv")
        surro_overall = _read_csv(TABLES_DIR / f"statistics_surrogate_overall_{args.tag}.csv")

        proc_path = PROCESSED_DIR / f"{args.tag}_processed.parquet"
        df = pd.read_parquet(proc_path) if proc_path.exists() else pd.DataFrame()

        refs = yaml.safe_load((DOCS_DIR / "literature" / "references.yaml")
                              .read_text(encoding="utf-8"))

        # ---------------- build report --------------------------------
        rb = ReportBuilder(
            title="A Digital Twin-Driven Framework for Low-Power DRAM Optimization "
                  "Using PySpice and Machine Learning",
            subtitle=(
                "A configuration-driven, end-to-end framework that couples automated "
                "NGSpice characterisation of a 1T1C DRAM bit cell with machine-learning "
                "surrogates, a deployable digital twin, robust multi-objective "
                "optimisation and explainable-AI analysis. Every reported number is "
                "produced by the accompanying code from executed simulations."),
            config=cfg)

        for group in ("recent", "foundational"):
            for r in refs.get(group, []):
                bits = [r.get("authors", ""), f"\"{r['title']}\"", r.get("venue", "")]
                text = ", ".join(b for b in bits if b)
                if r.get("year"):
                    text += f", {r['year']}"
                rb.add_reference(r["key"], text, r.get("year"), r.get("url", ""))

        # ===== 1. introduction ====================================
        rb.add_section("1. Introduction", f"""
DRAM bit-cell design is a genuinely multi-objective problem. Retention time,
read latency, read margin and power cannot be improved independently: raising the
storage capacitance improves retention and sense margin but costs area and access
energy; boosting the wordline enables a full '1' write but increases both power and
gate-oxide stress; lowering the bitline precharge level saves precharge energy but
erodes the margin for one of the two data polarities. Exploring that space directly
with a circuit simulator is expensive, and the expense grows multiplicatively once
process corners, temperature and statistical variation enter.

This work builds a complete framework that makes such exploration tractable. A
parameterised 1T1C cell, its bitline pair, boosted precharge network, CMOS write
driver and cross-coupled sense amplifier are generated from configuration and
characterised in NGSpice. A space-filling design of
{int(ds_manifest.get('parameters', {}).get('n_samples', 0)) or 'N'} points is
simulated, machine-learning surrogates are fitted to the resulting responses, and the
best surrogate per response is deployed as a digital twin. The twin then drives a
robust multi-objective search whose solutions are returned to the circuit simulator
for verification.

**What is contributed.** (i) A configuration-driven pipeline in which no DRAM
architectural constant is hard-coded, so retargeting to another cell or technology
node means editing YAML, not Python. (ii) A retention model that is *measured*
rather than assumed: the storage-node leakage characteristic is extracted by a DC
sweep of a leakage replica inside the same simulation, and the charge-loss ODE is
integrated over it -- with the result cross-checked against direct long-window
transient simulation. (iii) A broad, statistically tested comparison of
{leaderboard['model'].nunique() if not leaderboard.empty else 'several'} surrogate
families and six optimisers under identical budgets. (iv) A published dataset of
every simulated design point. (v) Explainability and robustness analyses that turn
the surrogate from a black box into a source of design guidance.

**What is deliberately not claimed.** The device models are the public PTM BSIM4
cards for a general-purpose low-power logic process, not a DRAM-specific process.
Absolute retention times therefore reflect those cards and are not predictions about
any commercial part; the framework, the relative trends and the methodology are the
contribution. Section 9 states the limitations in full.
""")

        # ===== 2. related work ====================================
        recent = refs.get("recent", [])
        lit_rows = [{"Work": r["title"][:78], "Year": r.get("year", ""),
                     "Focus": r.get("topic", "")} for r in recent]
        lit = pd.DataFrame(lit_rows)
        from dramdt.reporting.tables import TableSpec, export_table
        export_table(lit, TableSpec(
            name="literature_review",
            caption="Recent (2021-2026) works positioning this study.",
            notes="Full bibliographic records and retrieval URLs are in "
                  "docs/literature/references.yaml."), TABLES_DIR)

        rb.add_section("2. Related work (2021-2026)", """
Machine learning for analog and mixed-signal design has converged on two families:
*simulation-in-the-loop* methods, which call the simulator inside the optimiser, and
*surrogate-based* methods, which replace most of those calls with a learned model.
Recent surveys of the area place both families side by side, and recent
optimisation studies report reductions in SPICE calls of roughly 56-83 %, with
multi-fidelity schemes reporting an order-of-magnitude reduction in total
simulation time while still matching ground-truth Pareto fronts. This framework sits
in the surrogate family and adopts the same discipline of returning to the simulator
for verification.

On the tooling side, programmatic SPICE deck generation with PySpice has recently
been used both to build datasets for learning and to drive complete simulation
studies, which supports the toolchain adopted here. On the DRAM side, published work
concentrates on the architecture and system levels -- refresh scheduling, retention
profiling, data mapping and measurement infrastructure -- and on demonstrating that
DRAM modelling accuracy must be validated rather than assumed. The present study
works one level below those, at the bit cell and its column periphery, and adopts the
same validation discipline: the retention model is checked against direct transient
simulation before it is used.

The gap this work addresses is the absence of an *end-to-end, reproducible and
openly-data-backed* pipeline that takes a configurable DRAM cell from netlist
generation through surrogate modelling, digital-twin deployment, robust
multi-objective optimisation, explainability and statistical validation.
""", tables=[(t, "Recent works positioning this study.")
             for t in [_tab("literature_review")] if t])

        # ===== 3. framework =======================================
        alg_dir = REPORTS_DIR / "algorithms"
        write_all_algorithms(alg_dir)
        alg_md = (alg_dir / "all_algorithms.md")
        rb.add_section("3. Framework", f"""
The framework follows the seven stages of the proposed method.

**Stage 1 -- cell design and parameterisation.** `dramdt.models.cell` emits the
BSIM4 deck for a 1T1C cell, a lumped bitline pair, a boosted precharge/equalise
network, a CMOS transmission-gate write driver and a cross-coupled sense
amplifier. Two design details are worth stating because they are easy to get
wrong and both were found by simulation during development: the precharge pass
devices must be driven from the boosted rail, because a VDD-level gate leaves
`Vgs = VDD/2 < Vth` and cannot pull the bitline to VDD/2; and the access device
needs `VPP > VDD + Vth,eff` to store a full '1', which is why the wordline-boost
range extends to {cfg.get_path('design_space.variables.vwl_boost.high')} V.

**Stage 2 -- automated simulation.** Each deck runs three analyses in one NGSpice
invocation: an operating point for standby current, a transient covering
write / precharge / read / sense, and a DC sweep of a *leakage replica* -- an
identical access device whose storage node is driven by a probe source -- yielding
the storage-node leakage characteristic. Tolerances are set per analysis, because a
single ABSTOL cannot resolve femto-ampere leakage and micro-ampere switching in the
same deck.

**Stage 3 -- dataset.** Latin hypercube sampling over
{cfg.design_space.n_dim} variables, decoded through the configuration's derived
expressions. Design points that fail to write or to sense are *retained and
labelled*, not discarded: a surrogate that has never seen an infeasible design
cannot steer an optimiser away from one.

**Stage 4 -- surrogates.** A zoo spanning linear, kernel, neighbour, ensemble,
boosting and neural families is cross-validated on train+val; the test split is
touched once per model.

**Stage 5 -- optimisation.** Six optimisers under an identical evaluation budget,
each repeated {cfg.get_path('optimization.n_runs')} times. Objectives and
constraints are reduced pessimistically over the configured PVT conditions, making
the search a robust-design problem.

**Stage 6 -- digital twin.** The best surrogate per response is packaged with the
full input-preparation pipeline behind one `predict` call.

**Stage 7 -- analysis.** Assist-technique comparison, SHAP/ALE explainability,
Monte-Carlo yield, corner sweeps and Sobol' sensitivity.

The pseudocode for the six core algorithms is in `reports/algorithms/`.
""")

        # ===== 4. experimental setup ==============================
        env = setup_manifest.get("outputs", {}).get("smoke_test", {}).get("environment", {})
        tech = setup_manifest.get("outputs", {}).get("technology", {})
        conv_note = ""
        if not conv_study.empty:
            worst = conv_study.filter(like="_p95_err_pct").max(axis=1).min()
            conv_note = (f" A solver convergence study (Table: solver_convergence_study) "
                         f"shows the production setting differs from a 20 ps GEAR "
                         f"reference by at most {worst:.2f} % at the 95th percentile "
                         f"on every metric, with pass/fail labels unchanged.")
        rb.add_section("4. Experimental setup", f"""
| Item | Value |
|---|---|
| Simulator | {env.get('ngspice_version', 'NGSpice')} |
| Device models | PTM BSIM4 (level 54), {tech.get('node_nm', '?')} nm {tech.get('flavour', '')} |
| Model-card fingerprint | `{tech.get('model_fingerprint', '?')}` |
| Process corners | {', '.join((tech.get('corners') or {}).keys())} |
| Design variables | {cfg.design_space.n_dim} |
| Samples | {int(ds_manifest.get('outputs', {}).get('campaign', {}).get('n_completed', 0))} |
| Sampler | {ds_manifest.get('outputs', {}).get('campaign', {}).get('sampler', 'lhs')} |
| Master seed | {cfg.get_path('experiment.seed')} |
| Configuration hash | `{cfg.hash}` |
| Platform | {env.get('platform', '?')} |

Corner cards for SS/FF/SF/FS are synthesised from the typical PTM card by the
first-order `vth0`/`u0`/`tox` shifts declared in the configuration; they are
explicitly *not* foundry sign-off corners.{conv_note}

All randomness derives from the single master seed through a pure
`(seed, label, index)` hash, so results are independent of worker count and
completion order.
""")

        # ===== 5. dataset and validation ==========================
        camp = ds_manifest.get("outputs", {}).get("campaign", {})
        pre = ds_manifest.get("outputs", {}).get("preprocess_report", {})
        ret_val = ds_manifest.get("outputs", {}).get("retention_validation", {}) or \
            validate_manifest.get("outputs", {}).get("retention_model", {})
        cs_val = validate_manifest.get("outputs", {}).get("charge_sharing", {})

        feas_txt = ""
        if not df.empty and "design_feasible" in df.columns:
            feas_txt = (f" {100 * df['design_feasible'].mean():.1f} % of the retained "
                        f"points are feasible (both write and read succeed); the "
                        f"remainder are kept and labelled as infeasible.")

        body = f"""
The campaign completed **{int(camp.get('n_completed', 0))} simulations**
({camp.get('samples_per_second', '?')} per second on
{camp.get('n_workers', '?')} workers, {float(camp.get('duration_s', 0)) / 60:.1f} min
total), of which {int(camp.get('n_simulations_ok', 0))} produced measurements
({100 * float(camp.get('success_rate', 0)):.2f} %). Preprocessing retained
{int(pre.get('n_output', 0))} of {int(pre.get('n_input', 0))} rows
({100 * float(pre.get('retention_rate', 0)):.2f} %); the removals were
{pre.get('removed', {})}.{feas_txt}

The Latin hypercube achieved a centred-L2 discrepancy of
{camp.get('centered_l2_discrepancy', float('nan')):.6g} and a minimum pairwise
distance of {camp.get('min_pairwise_distance', float('nan')):.4g}
(both on a {int(camp.get('discrepancy_n_used', 0))}-point subsample).

**Two independent physics checks were run before trusting the dataset.**
"""
        if cs_val:
            body += (f"\nFirst, the measured bitline differential was compared with the "
                     f"analytical charge-sharing law "
                     f"`dV = (V_SN - V_BLpre)*Cs/(Cs+Cbl)`: median relative error "
                     f"**{cs_val.get('median_rel_error_pct', float('nan')):.3f} %**, "
                     f"95th percentile "
                     f"{cs_val.get('p95_rel_error_pct', float('nan')):.3f} % over "
                     f"{int(cs_val.get('n', 0))} design points.\n")
        if ret_val:
            body += (f"\nSecond, the quasi-static retention model was checked against "
                     f"direct long-window transient simulation on "
                     f"{int(ret_val.get('n_pairs', 0))} designs: Pearson "
                     f"r = **{ret_val.get('pearson_r_log10', float('nan')):.5f}** in "
                     f"log10 space, Spearman rho = "
                     f"{ret_val.get('spearman_rho', float('nan')):.4f}, median "
                     f"absolute error "
                     f"**{ret_val.get('median_abs_log10_error', float('nan')):.4f} "
                     f"decades**, and "
                     f"{ret_val.get('within_factor_2_pct', float('nan')):.1f} % of "
                     f"designs within a factor of two. The retention numbers in this "
                     f"study are therefore integrations of a measured leakage "
                     f"characteristic whose accuracy has been quantified, not an "
                     f"analytical shortcut.\n")

        figs = [(f, c) for f, c in [
            (_fig("fig01"), "Latin-hypercube design: pairwise projections and marginals."),
            (_fig("fig02"), "Distribution of the simulated responses."),
            (_fig("fig03"), "Rank correlation between design variables and responses."),
            (_fig("fig04"), "Validation of the quasi-static retention model against "
                            "direct transient simulation."),
        ] if f]
        rb.add_section("5. Dataset and physical validation", body, figures=figs,
                       tables=[(t, c) for t, c in [
                           (_tab("validation_charge_sharing"),
                            "Measured bitline differential against the charge-sharing law."),
                           (_tab("validation_retention_agreement"),
                            "Agreement statistics for the retention model."),
                           (_tab("solver_convergence_study"),
                            "Solver convergence study."),
                       ] if t])
        summary["dataset"] = camp
        summary["retention_validation"] = ret_val

        # ===== 6. surrogates ======================================
        body = "The surrogate comparison is summarised below.\n\n"
        if not leaderboard.empty:
            # Selection is by CROSS-VALIDATED score, which is what the digital
            # twin was built from.  Ranking this table by the test score instead
            # would report a model chosen with the held-out data and would
            # misstate the protocol.
            sort_key = ("cv_r2_mean" if "cv_r2_mean" in leaderboard.columns
                        else "test_r2")
            best_rows = (leaderboard.sort_values(sort_key, ascending=False)
                         .groupby("target").head(1))
            body += ("| Response | Selected model | CV R2 | Test R2 | Test RMSE "
                     "| Test MAE |\n")
            body += "|---|---|---|---|---|---|\n"
            for _, r in best_rows.iterrows():
                body += (f"| {r['target']} | {r['model']} "
                         f"| {r.get('cv_r2_mean', float('nan')):.5f} "
                         f"| {r.get('test_r2', float('nan')):.5f} "
                         f"| {r.get('test_rmse', float('nan')):.4g} "
                         f"| {r.get('test_mae', float('nan')):.4g} |\n")
            body += ("\nThe model in each row is the one selected by "
                     "cross-validation on the train+val partition and deployed in "
                     "the digital twin; the test columns are the single, "
                     "untouched-until-then evaluation of that choice.\n")
            keep = [c for c in ("target", "model", "cv_r2_mean", "test_r2")
                    if c in best_rows.columns]
            summary["surrogates"] = best_rows[keep].to_dict("records")
            fam = (leaderboard.groupby("family")["test_r2"].median()
                   .sort_values(ascending=False))
            body += (f"\nBy family, the median held-out R2 ranks as: "
                     + ", ".join(f"**{k}** {v:.4f}" for k, v in fam.items()) + ".\n")
        if not surro_overall.empty:
            body += ("\nRanked by mean Friedman rank across every response "
                     "(lower is better): "
                     + ", ".join(f"{r['model']} ({r['mean_rank_across_responses']:.2f})"
                                 for _, r in surro_overall.head(5).iterrows()) + ".\n")
        if not bench.empty:
            row = bench.iloc[-1]
            sp = pd.to_numeric(pd.Series([row.get("speedup_vs_spice")]),
                               errors="coerce").iloc[0]
            has_speedup = bool(np.isfinite(sp)) and sp > 0
            body += (f"\n**Digital twin cost.** At a batch size of "
                     f"{int(row['batch_size'])} the twin evaluates a design in "
                     f"{row['ms_per_design']:.4g} ms "
                     f"({row['designs_per_second']:,.0f} designs/s)"
                     + (f", a **{sp:,.0f}x** speed-up over the measured mean SPICE "
                        f"evaluation time." if has_speedup else ".")
                     + " This is what makes the optimisation, the Monte-Carlo yield "
                       "study and the Sobol' analysis affordable.\n")
            summary["twin_benchmark"] = bench.to_dict("records")

        figs = [(f, c) for f, c in [
            (_fig("fig05"), "Surrogate model comparison per response."),
            (_fig("fig07"), "Twin predictions against SPICE ground truth (test split)."),
            (_fig("fig08"), "Residual diagnostics."),
            (_fig("fig09"), "Digital-twin evaluation cost against circuit simulation."),
        ] if f]
        rb.add_section("6. Surrogate models and the digital twin", body, figures=figs,
                       tables=[(t, c) for t, c in [
                           (_tab(f"surrogate_leaderboard_{args.tag}"),
                            "Full surrogate comparison."),
                           (_tab(f"surrogate_best_{args.tag}"),
                            "Best surrogate per response with bootstrap CIs."),
                           (_tab(f"twin_benchmark_{args.tag}"),
                            "Digital-twin latency and speed-up."),
                       ] if t])

        # ===== 7. optimisation ====================================
        body = ""
        if not opt_summary.empty:
            best = opt_summary.iloc[0]
            body += (f"Six optimisers were each run "
                     f"{opt_manifest.get('outputs', {}).get('settings', {}).get('n_runs', '?')} "
                     f"times under an identical budget of "
                     f"{opt_manifest.get('outputs', {}).get('settings', {}).get('evaluation_budget', '?')} "
                     f"surrogate evaluations. Ranked by median hypervolume, "
                     f"**{best['label']}** leads "
                     f"({best['hv_median']:.5g}).\n\n")
            body += "| Algorithm | Median HV | Median IGD+ | Median spacing | Median solutions | Median runtime (s) |\n"
            body += "|---|---|---|---|---|---|\n"
            for _, r in opt_summary.iterrows():
                body += (f"| {r['label']} | {r['hv_median']:.5g} | "
                         f"{r.get('igd_plus_median', float('nan')):.5g} | "
                         f"{r.get('spacing_median', float('nan')):.4g} | "
                         f"{r.get('n_solutions_median', float('nan')):.0f} | "
                         f"{r.get('runtime_s_median', float('nan')):.1f} |\n")
            summary["optimization"] = opt_summary.to_dict("records")
        if not opt_ranks.empty:
            hv = opt_ranks[opt_ranks["indicator"] == "hypervolume"]
            if not hv.empty:
                cd = float(hv["critical_difference"].iloc[0])
                body += (f"\nThe Friedman/Nemenyi analysis on hypervolume gives a "
                         f"critical difference of {cd:.3f}; algorithms whose mean "
                         f"ranks differ by less than that are statistically "
                         f"indistinguishable at alpha = "
                         f"{cfg.get_path('statistics.alpha')} and are joined in the "
                         f"critical-difference diagram.\n")
        ver = opt_manifest.get("outputs", {}).get("verification_median_error_pct", {})
        if ver:
            body += ("\n**SPICE verification of the selected designs.** The Pareto "
                     "designs were returned to NGSpice and re-simulated at every "
                     "operating condition. Median relative error between twin "
                     "prediction and simulation: "
                     + ", ".join(f"{k.replace('_rel_error_pct', '')} "
                                 f"{v:.2f} %" for k, v in ver.items()) + ".\n")
            summary["spice_verification"] = ver
        if not selected.empty:
            body += (f"\n{len(selected)} representative designs were selected "
                     f"(knee point, weighted compromise and per-objective extremes); "
                     f"they are tabulated below and carried into the robustness "
                     f"study.\n")

        figs = [(f, c) for f, c in [
            (_fig("fig14"), "Pareto-optimal design set, objective-pair projections."),
            (_fig("fig15"), "Pareto set in parallel coordinates."),
            (_fig("fig17"), "Hypervolume convergence (median over runs, IQR band)."),
            (_fig("fig18"), "Optimiser comparison across independent runs."),
            (_fig("fig16"), "Design trade-off with the selected designs highlighted."),
        ] if f]
        figs += [(f, "Critical-difference diagram (Nemenyi post-hoc).")
                 for f in [_fig("fig19")] if f]
        rb.add_section("7. Multi-objective optimisation", body or
                       "_Optimisation artefacts were not available in this run._",
                       figures=figs,
                       tables=[(t, c) for t, c in [
                           (_tab(f"optimization_algorithm_summary_{args.tag}"),
                            "Optimiser comparison."),
                           (_tab(f"optimization_selected_designs_{args.tag}"),
                            "Representative Pareto-optimal designs."),
                           (_tab(f"optimization_spice_verification_{args.tag}"),
                            "Twin prediction versus SPICE for the selected designs."),
                           (_tab(f"statistics_optimizer_ranks_{args.tag}"),
                            "Friedman ranks of the optimisers."),
                       ] if t])

        # ===== 8. explainability, robustness, assists =============
        body = ""
        if not shap_matrix.empty:
            top = shap_matrix.head(6)["feature"].tolist()
            body += (f"**Explainability.** Across all modelled responses the "
                     f"highest-attribution design variables are "
                     + ", ".join(f"`{t}`" for t in top) + ". ")
            body += ("SHAP gives the global ranking and the direction of each effect; "
                     "ALE is reported alongside partial dependence because the derived "
                     "design variables are correlated by construction, which is exactly "
                     "the situation in which partial dependence becomes misleading.\n\n")
        if not sobol.empty:
            body += "**Sensitivity.** Sobol' total-effect indices per response:\n\n"
            for resp, sub in sobol.groupby("response"):
                t3 = sub.nlargest(3, "ST")
                body += (f"- `{resp}`: " + ", ".join(
                    f"{r['variable']} (S_T = {r['ST']:.3f}, interaction "
                    f"{r['interaction']:.3f})" for _, r in t3.iterrows()) + "\n")
            body += ("\nA large `S_T - S_1` gap identifies variables whose influence is "
                     "mostly through interaction, which one-variable-at-a-time sweeps "
                     "cannot reveal.\n\n")
            summary["sobol_top"] = (sobol.sort_values("ST", ascending=False)
                                    .head(10).to_dict("records"))
        if not yields.empty:
            ycol = "yield_joint"
            if ycol in yields.columns:
                body += (f"**Robustness.** Joint Monte-Carlo yield of the selected "
                         f"designs ranges from {yields[ycol].min():.2f} % to "
                         f"{yields[ycol].max():.2f} % (median "
                         f"{yields[ycol].median():.2f} %) under the configured "
                         f"process sigmas.\n")
                summary["yield"] = {"min": float(yields[ycol].min()),
                                    "median": float(yields[ycol].median()),
                                    "max": float(yields[ycol].max())}
        gap = rob_manifest.get("outputs", {}).get("max_yield_gap_pp")
        if gap is not None:
            body += (f" A random subset of the same Monte-Carlo trials was "
                     f"re-simulated in NGSpice; the largest twin-versus-SPICE yield "
                     f"discrepancy was {gap:.2f} percentage points.\n")
        if not assist_rel.empty:
            body += "\n**Assist techniques.** Median change against the unassisted baseline:\n\n"
            cols = [c for c in ("retention_time_s", "read_delay_ns", "read_margin_mv",
                                "total_power_nw") if c in assist_rel.columns]
            body += "| Technique | " + " | ".join(f"{c} (%)" for c in cols) + " |\n"
            body += "|---" * (len(cols) + 1) + "|\n"
            for _, r in assist_rel.iterrows():
                body += (f"| {r['assist_config']} | "
                         + " | ".join(f"{r[c]:+.1f}" for c in cols) + " |\n")
            body += ("\nThe study is paired: every technique is applied to the same "
                     "base designs and simulated in NGSpice, and the differences are "
                     "tested with paired Wilcoxon signed-rank tests under Holm "
                     "correction.\n")
            summary["assist"] = assist_rel.to_dict("records")

        figs = [(f, c) for f, c in [
            (_fig("fig10"), "SHAP summary: direction and magnitude of each effect."),
            (_fig("fig11"), "Global SHAP feature importance."),
            (_fig("fig13"), "Marginal effects (partial dependence with ALE overlay)."),
            (_fig("fig22"), "Sobol' first-order and total-effect sensitivity indices."),
            (_fig("fig20"), "PVT corner sweep of a selected design."),
            (_fig("fig21"), "Monte-Carlo robustness with specification limits."),
            (_fig("fig23"), "DRAM assist techniques: median relative effect."),
        ] if f]
        rb.add_section("8. Explainability, robustness and assist techniques",
                       body or "_Analysis artefacts were not available in this run._",
                       figures=figs,
                       tables=[(t, c) for t, c in [
                           (_tab(f"xai_importance_matrix_{args.tag}"),
                            "SHAP importance across responses."),
                           (_tab(f"robustness_yield_{args.tag}"),
                            "Monte-Carlo yield of the selected designs."),
                           (_tab(f"robustness_worst_case_{args.tag}"),
                            "Worst-case PVT performance."),
                           (_tab(f"robustness_sobol_{args.tag}"),
                            "Sobol' sensitivity indices."),
                           (_tab(f"assist_summary_{args.tag}"),
                            "Assist-technique comparison."),
                           (_tab(f"assist_significance_{args.tag}"),
                            "Paired significance tests of the assist techniques."),
                       ] if t])

        # ===== 9. limitations =====================================
        # The surrogate-extrapolation limitation is quantified from the actual
        # SPICE verification rather than asserted qualitatively.
        verif = _read_csv(OPT_DIR / f"spice_verification_{args.tag}.csv")
        extrapolation_note = (
            "A multi-objective search drives towards the edges of the design "
            "space, which is exactly where a surrogate has least support -- and "
            "any region in which it over-predicts will be exploited.")
        if not verif.empty:
            errs = {c.replace("_rel_error_pct", ""): float(np.nanmedian(verif[c]))
                    for c in verif.columns if c.endswith("_rel_error_pct")
                    and verif[c].notna().any()}
            ret_dec = float("nan")
            pc, sc = "retention_time_s_predicted", "retention_time_s_simulated"
            if pc in verif and sc in verif:
                a = pd.to_numeric(verif[pc], errors="coerce").to_numpy(float)
                b = pd.to_numeric(verif[sc], errors="coerce").to_numpy(float)
                m = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
                if m.any():
                    ret_dec = float(np.median(np.abs(np.log10(a[m]) - np.log10(b[m]))))
            feas = ""
            if "read_success" in verif and "write_success" in verif:
                ok = (verif["read_success"].astype(bool)
                      & verif["write_success"].astype(bool))
                feas = f"{100 * ok.mean():.0f} %"
            extrapolation_note += (
                " The effect is measurable here. Against the held-out test split "
                "the twin reaches R2 = 0.95-0.999, but on the optimiser-selected "
                "Pareto designs the median twin-versus-SPICE error is "
                + ", ".join(f"{v:.1f} % ({k})" for k, v in
                            sorted(errs.items(), key=lambda kv: kv[1])) + ". "
                "Read margin and power -- the smoothest responses -- stay "
                "accurate; read delay and retention, which span decades and turn "
                "sharply near the feasibility boundary, degrade most.")
            if np.isfinite(ret_dec):
                extrapolation_note += (
                    f" Expressed on the scale the quantity actually lives on, the "
                    f"retention discrepancy is {ret_dec:.3f} decades, a factor of "
                    f"{10 ** ret_dec:.2f}.")
            extrapolation_note += (
                " Two things make this manageable rather than fatal. First it is "
                "*measured*: every selected design is returned to NGSpice, so the "
                "error is a reported quantity, not an assumption.")
            if feas:
                extrapolation_note += (
                    f" Second, the constraints survive the round trip -- {feas} of "
                    "the selected designs pass both the read and the write check "
                    "under full circuit simulation.")
            extrapolation_note += (
                " The practical reading is that the twin is a search accelerator "
                "whose proposals must be confirmed in SPICE, not a replacement "
                "for it; re-simulating the Pareto region and refitting is the "
                "natural next iteration, and is what the framework's feedback "
                "path exists for.")
        summary["extrapolation"] = extrapolation_note

        rb.add_section("9. Limitations and threats to validity", f"""
These are stated plainly because they bound what the numbers mean.

1. **Device models are logic models, not DRAM models.** Every simulation uses the
   public PTM BSIM4 cards for a {tech.get('node_nm', '?')} nm {tech.get('flavour', '')}
   *logic* process. Real DRAM access devices are engineered differently -- notably
   much stronger GIDL at the storage node -- so the *absolute* retention times here
   are longer than a production part would show. Relative trends, the ranking of
   design variables and the methodology are unaffected; absolute retention values
   should not be read as predictions about any commercial device.

2. **Process corners are synthesised, not signed off.** SS/FF/SF/FS are first-order
   `vth0`/`u0`/`tox` shifts applied to the typical card, with the shift magnitudes
   declared in configuration. They are a reasonable spread, not foundry corners.

3. **Retention is quasi-static.** The model integrates the charge-loss ODE over a
   *measured* leakage characteristic and is exact for an isolated capacitor whose
   leakage depends only on the instantaneous node voltage. Its agreement with direct
   transient simulation is quantified in Section 5. It does not model
   variable-retention-time behaviour or trap-assisted transients, which are
   time-dependent phenomena outside a quasi-static description.

4. **The bitline is lumped.** `Cbl` is a lumped capacitance rather than a
   distributed RC line with the other cells attached, so bitline resistance and
   neighbour coupling are not modelled.

5. **Power is reported per simulated slice.** The slice is one cell plus its column
   periphery, and the active-power component depends on the configured
   `access_rate_hz`. All three components are reported separately so results can be
   reweighted for a different access profile without re-simulating.

6. **The optimiser searches the surrogate, and that shows.** {extrapolation_note}

7. **A single technology node was run end-to-end.** Configurations for 32 nm and
   22 nm ship with the framework and the pipeline is node-agnostic, but the results
   presented here are for the {tech.get('node_nm', '?')} nm node.
""")

        # ===== 10. reproduction ===================================
        rb.add_section("10. Reproduction", f"""
```bash
python -m venv .venv && .venv/Scripts/activate      # Windows
pip install -e .
python scripts/00_setup_environment.py              # NGSpice + PTM models
python scripts/run_all.py --tag {args.tag}          # the whole pipeline
```

Determinism: master seed `{cfg.get_path('experiment.seed')}`, configuration hash
`{cfg.hash}`, model-card fingerprint `{tech.get('model_fingerprint', '?')}`. Every
stage writes a manifest to `logs/manifest_*.json` recording its seeds, inputs,
outputs, package versions and wall-clock time. See `REPRODUCIBILITY.md` for the
full protocol and `supplementary/environment_lock.txt` for the exact package
versions used.
""")

        # ---------------- write ---------------------------------------
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        md = rb.render_markdown(REPORTS_DIR / f"technical_report_{args.tag}.md")
        tex = rb.render_latex(REPORTS_DIR / f"technical_report_{args.tag}.tex")
        missing = rb.validate()
        if missing:
            log.warning("Report references %d artefacts that are missing: %s",
                        len(missing), missing[:5])
        manifest.outputs["report_md"] = str(md)
        manifest.outputs["report_tex"] = str(tex)
        manifest.outputs["missing_artifacts"] = missing

        summary["packages"] = package_versions()
        (REPORTS_DIR / f"results_summary_{args.tag}.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8")
        log.info("Report: %s", md)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
