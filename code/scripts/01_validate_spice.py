#!/usr/bin/env python
"""Stage 1 -- validate the simulation setup before spending a campaign on it.

Four independent checks, each producing a table in ``results/tables/``:

1. **Charge-sharing law** -- the measured bitline differential is compared with
   the analytical ``dV = (V_SN - V_BLpre) * Cs/(Cs+Cbl)``.  Agreement confirms
   the cell, bitline and precharge network behave as intended.
2. **Solver convergence study** -- extracted metrics at several transient print
   steps and integration methods, relative to the finest setting, justifying the
   production solver settings instead of asserting them.
3. **Retention model validation** -- quasi-static integration against direct
   long-window transient simulation.
4. **Backend equivalence** -- the same deck through the ``subprocess`` and
   ``pyspice-shared`` backends.

    python scripts/01_validate_spice.py --n 120
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
from dramdt.doe.sampling import generate_design                         # noqa: E402
from dramdt.env import resolve_environment                              # noqa: E402
from dramdt.logging_utils import RunManifest, setup_logging, stage      # noqa: E402
from dramdt.models import DramCellBuilder, Technology                   # noqa: E402
from dramdt.paths import LOGS_DIR, TABLES_DIR, ensure_directories       # noqa: E402
from dramdt.reporting.tables import TableSpec, export_table             # noqa: E402
from dramdt.simulation import MetricExtractor, SpiceRunner              # noqa: E402
from dramdt.simulation.retention import validate_against_transient      # noqa: E402
from dramdt.simulation.runner import parse_measurements                 # noqa: E402


def _rel_error(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 1e-30)
    return np.abs(a[m] - b[m]) / np.abs(b[m])


def check_charge_sharing(cfg, builder, extractor, runner, points, log) -> pd.DataFrame:
    """Measured bitline differential against the charge-sharing law."""
    ds = cfg.design_space
    rows = []
    for i, u in enumerate(points):
        p = builder.parameters_from_sample(ds.decode_row(u), i)
        res = runner.run(builder.build_characterization_netlist(p), tag="cs")
        met = extractor.extract(res, p)
        if not (np.isfinite(met.read_margin_v) and np.isfinite(met.v_sn_written)):
            continue
        predicted = (met.v_sn_written - p.vblpre) * p.cs / (p.cs + p.cbl)
        rows.append({
            "sample": i, "corner": p.corner, "temperature_c": p.temperature_c,
            "cs_ff": p.cs * 1e15, "cbl_ff": p.cbl * 1e15,
            "v_sn_written_v": met.v_sn_written,
            "measured_dv_mv": met.read_margin_v * 1e3,
            "analytical_dv_mv": predicted * 1e3,
            "relative_error_pct": 100.0 * abs(met.read_margin_v - predicted)
            / max(abs(predicted), 1e-12),
        })
    frame = pd.DataFrame(rows)
    if not frame.empty:
        log.info("Charge sharing: median relative error %.3f %%, p95 %.3f %% (n=%d)",
                 frame["relative_error_pct"].median(),
                 frame["relative_error_pct"].quantile(0.95), len(frame))
    return frame


def check_solver_convergence(cfg_specs, points, log, env) -> pd.DataFrame:
    """Metric sensitivity to the transient print step and integration method."""
    settings = [("20 ps / GEAR", 20e-12, "GEAR"),
                ("50 ps / GEAR", 50e-12, "GEAR"),
                ("50 ps / TRAP", 50e-12, "TRAP"),
                ("100 ps / GEAR", 100e-12, "GEAR")]
    metrics = ["read_margin_mv", "read_delay_ns", "write_delay_ns",
               "v_sn_written_v", "retention_time_s", "energy_per_access_fj",
               "total_power_nw"]
    results: dict[str, pd.DataFrame] = {}
    timing: dict[str, float] = {}

    import time
    for label, step, method in settings:
        cfg = load_config(*cfg_specs)
        cfg["simulation"]["tran_step_s"] = step
        cfg["simulation"]["options"]["METHOD"] = method
        tech = Technology.from_config(cfg)
        builder = DramCellBuilder(cfg, tech)
        extractor = MetricExtractor(cfg)
        ds = cfg.design_space
        rows = []
        t0 = time.perf_counter()
        with SpiceRunner(env, timeout_s=300) as runner:
            for i, u in enumerate(points):
                p = builder.parameters_from_sample(ds.decode_row(u), i)
                res = runner.run(builder.build_characterization_netlist(p), tag="conv")
                rec = extractor.extract(res, p).to_record()
                rec["sample"] = i
                rows.append(rec)
        timing[label] = (time.perf_counter() - t0) / max(len(points), 1)
        results[label] = pd.DataFrame(rows).set_index("sample")
        log.info("  %-14s %6.1f ms/sample", label, timing[label] * 1e3)

    ref_label = settings[0][0]
    ref = results[ref_label]
    out_rows = []
    for label in results:
        if label == ref_label:
            continue
        cur = results[label]
        row = {"setting": label,
               "ms_per_sample": round(timing[label] * 1e3, 1),
               "speedup_vs_reference": round(timing[ref_label] / timing[label], 3)}
        for m in metrics:
            if m in ref.columns and m in cur.columns:
                e = _rel_error(cur[m].to_numpy(float), ref[m].to_numpy(float))
                row[f"{m}_median_err_pct"] = round(100 * float(np.median(e)), 4) if e.size else np.nan
                row[f"{m}_p95_err_pct"] = round(100 * float(np.percentile(e, 95)), 4) if e.size else np.nan
        for flag in ("read_success", "write_success"):
            if flag in ref.columns and flag in cur.columns:
                row[f"{flag}_agreement_pct"] = round(
                    100 * float((ref[flag].astype(bool) == cur[flag].astype(bool)).mean()), 2)
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def check_retention(cfg, builder, extractor, runner, points, log) -> tuple[pd.DataFrame, dict]:
    """Quasi-static retention model against direct transient simulation."""
    ds = cfg.design_space
    rows = []
    for i, u in enumerate(points):
        p = builder.parameters_from_sample(ds.decode_row(u), i)
        res = runner.run(builder.build_characterization_netlist(p), tag="ret")
        met = extractor.extract(res, p)
        if (not np.isfinite(met.retention_s) or met.retention_censored
                or met.retention_s <= 0 or met.retention_s > 0.5):
            continue
        if not (np.isfinite(met.v_sn_written) and met.v_sn_written > met.v_fail):
            continue
        deck = builder.build_retention_netlist(
            p, t_stop=met.retention_s * 2.5, v_fail=met.v_fail,
            n_points=4000, v_init=met.v_sn_written)
        r = runner.run(deck, tag="rettran")
        vals, _ = parse_measurements(r.log_excerpt)
        t_tran = vals.get("t_ret_tran", float("nan"))
        if np.isfinite(t_tran) and t_tran > 0:
            rows.append({
                "sample": i, "corner": p.corner, "temperature_c": p.temperature_c,
                "cs_ff": p.cs * 1e15,
                "retention_quasistatic_s": met.retention_s,
                "retention_transient_s": t_tran,
                "ratio": met.retention_s / t_tran,
            })
    frame = pd.DataFrame(rows)
    stats = {}
    if len(frame) >= 3:
        stats = validate_against_transient(frame["retention_quasistatic_s"],
                                           frame["retention_transient_s"])
        log.info("Retention model: r(log10)=%.5f, median |err|=%.4f decades, "
                 "%.1f %% within 2x (n=%d)",
                 stats.get("pearson_r_log10", float("nan")),
                 stats.get("median_abs_log10_error", float("nan")),
                 stats.get("within_factor_2_pct", float("nan")),
                 int(stats.get("n_pairs", 0)))
    return frame, stats


#: Metrics derived purely from the operating point and the transient.  The
#: leakage DC sweep -- and therefore retention and the leakage current -- is not
#: available through the shared-library backend (PySpice 1.5 + ngspice 46 reject
#: a `dc` command issued through the shared command interface), so comparing
#: those columns across backends would compare a number against nothing.
_BACKEND_COMPARABLE = ("read_margin_mv", "read_delay_ns", "write_delay_ns",
                       "v_sn_written_v", "read_energy_fj")


def check_backends(cfg, builder, env, points, log) -> pd.DataFrame:
    """Same decks through both execution backends."""
    ds = cfg.design_space
    extractor = MetricExtractor(cfg)
    rows = []
    try:
        sub = SpiceRunner(env, backend="subprocess")
        sh = SpiceRunner(env, backend="pyspice-shared")
    except Exception as exc:
        log.warning("Backend comparison skipped: %s", exc)
        return pd.DataFrame()

    for i, u in enumerate(points):
        p = builder.parameters_from_sample(ds.decode_row(u), i)
        deck = builder.build_characterization_netlist(p)
        a = extractor.extract(sub.run(deck, tag="ba"), p).to_record()
        b = extractor.extract(sh.run(deck, tag="bb"), p).to_record()
        row = {"sample": i}
        for m in _BACKEND_COMPARABLE:
            va, vb = a.get(m, np.nan), b.get(m, np.nan)
            row[f"{m}_subprocess"] = va
            row[f"{m}_shared"] = vb
            row[f"{m}_rel_diff_pct"] = (100 * abs(va - vb) / abs(va)
                                        if np.isfinite(va) and np.isfinite(vb)
                                        and abs(va) > 1e-30 else np.nan)
        rows.append(row)
    sub.close()
    sh.close()
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", nargs="+", default=["dram_1t1c_45nm_lp.yaml"])
    ap.add_argument("--tag", default="main",
                    help="run label; used for the log and manifest names. The "
                         "validation tables themselves are untagged because they "
                         "validate the simulation setup, not one dataset.")
    ap.add_argument("--n", type=int, default=120, help="design points per check")
    ap.add_argument("--n-backend", type=int, default=8)
    ap.add_argument("--skip-convergence", action="store_true")
    ap.add_argument("--skip-backend", action="store_true")
    args = ap.parse_args()

    ensure_directories()
    log = setup_logging("validate", filename=f"01_validate_spice_{args.tag}.log")
    # Load the configuration before building the manifest so its hash is
    # recorded: a manifest without one cannot be tied back to the experiment
    # whose simulation setup it validated.
    cfg = load_config(*args.config)
    manifest = RunManifest(stage=f"validate_spice_{args.tag}", config_hash=cfg.hash,
                           parameters={"config": [str(s) for s in cfg.sources],
                                       "n_points": args.n})

    with stage("validate_spice", log, manifest, LOGS_DIR / f"manifest_validate_{args.tag}.json"):
        # (configuration already loaded above so the manifest carries its hash)
        env = resolve_environment()
        tech = Technology.from_config(cfg)
        builder = DramCellBuilder(cfg, tech)
        extractor = MetricExtractor(cfg)
        ds = cfg.design_space
        design = generate_design(args.n, ds.n_dim, "lhs", 4242, "validation",
                                 compute_diagnostics=False)
        points = design.points

        with SpiceRunner(env, timeout_s=600) as runner:
            log.info("--- check 1: charge-sharing law ---")
            cs = check_charge_sharing(cfg, builder, extractor, runner, points, log)
            export_table(cs, TableSpec(
                name="validation_charge_sharing",
                caption="Measured bitline differential against the analytical "
                        "charge-sharing law.",
                notes="Relative error between the simulated V_BL - V_BLB at the "
                      "sense-amplifier firing instant and (V_SN - V_BLpre)*Cs/(Cs+Cbl).",
                max_rows=40), TABLES_DIR)
            if not cs.empty:
                manifest.outputs["charge_sharing"] = {
                    "n": int(len(cs)),
                    "median_rel_error_pct": float(cs["relative_error_pct"].median()),
                    "p95_rel_error_pct": float(cs["relative_error_pct"].quantile(0.95)),
                }

            log.info("--- check 3: retention model ---")
            ret, ret_stats = check_retention(cfg, builder, extractor, runner,
                                             points[:min(len(points), 60)], log)
            export_table(ret, TableSpec(
                name="validation_retention_model",
                caption="Quasi-static retention model against direct transient "
                        "simulation.",
                max_rows=40), TABLES_DIR)
            if ret_stats:
                export_table(pd.DataFrame([{"statistic": k, "value": v}
                                           for k, v in ret_stats.items()]),
                             TableSpec(name="validation_retention_agreement",
                                       caption="Agreement statistics for the "
                                               "quasi-static retention model."),
                             TABLES_DIR)
                manifest.outputs["retention_model"] = ret_stats

        if not args.skip_convergence:
            log.info("--- check 2: solver convergence study ---")
            conv = check_solver_convergence(args.config, points[:min(len(points), 60)],
                                            log, env)
            export_table(conv, TableSpec(
                name="solver_convergence_study",
                caption="Sensitivity of the extracted metrics to the transient print "
                        "step and integration method, relative to a 20 ps GEAR "
                        "reference.",
                notes="Percentages are relative errors against the 20 ps GEAR "
                      "setting over the same design points; agreement columns give "
                      "the share of points whose pass/fail label is unchanged."),
                TABLES_DIR)
            manifest.outputs["solver_convergence"] = conv.to_dict(orient="records")

        if not args.skip_backend:
            log.info("--- check 4: backend equivalence ---")
            be = check_backends(cfg, builder, env, points[:args.n_backend], log)
            export_table(be, TableSpec(
                name="validation_backend_equivalence",
                caption="Equivalence of the subprocess and PySpice shared-library "
                        "execution backends."), TABLES_DIR)
            if not be.empty:
                cols = [c for c in be.columns if c.endswith("_rel_diff_pct")]
                worst = float(np.nanmax(be[cols].to_numpy(dtype=float))) if cols else float("nan")
                log.info("Backend equivalence: worst relative difference %.4g %%", worst)
                manifest.outputs["backend_equivalence"] = {
                    "n": int(len(be)), "worst_rel_diff_pct": worst}

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
