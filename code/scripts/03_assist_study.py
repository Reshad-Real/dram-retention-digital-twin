#!/usr/bin/env python
"""Stage 7 -- controlled comparison of the DRAM assist techniques.

Each assist technique is a declarative transform of a design point (see
``configs/base.yaml``).  The study is *paired*: the same set of randomly drawn
base designs is simulated once per technique, so every technique is measured
against the identical population and the differences are attributable to the
technique rather than to sampling noise.  Every point is a real NGSpice run.

    python scripts/03_assist_study.py --n-designs 300

Outputs
    data/processed/assist_study_<tag>.parquet
    results/tables/assist_*.{csv,tex,md}
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dramdt.config import load_config                                   # noqa: E402
from dramdt.dataset.campaign import SimulationCampaign                  # noqa: E402
from dramdt.doe.sampling import generate_design                         # noqa: E402
from dramdt.env import resolve_environment                              # noqa: E402
from dramdt.logging_utils import RunManifest, setup_logging, stage      # noqa: E402
from dramdt.dataset.preprocess import sanitize_for_parquet              # noqa: E402
from dramdt.models.assist import AssistLibrary                          # noqa: E402
from dramdt.paths import (INTERIM_DIR, LOGS_DIR, PROCESSED_DIR, TABLES_DIR,
                          ensure_directories)                           # noqa: E402
from dramdt.reporting.tables import TableSpec, export_table             # noqa: E402
from dramdt.seeds import seed_everything                                # noqa: E402
from dramdt.statistics import wilcoxon_holm                             # noqa: E402

METRICS = ["retention_time_s", "read_delay_ns", "write_delay_ns",
           "read_margin_mv", "total_power_nw", "energy_per_access_fj",
           "leakage_current_fa", "refresh_interval_ms"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", nargs="+", default=["dram_1t1c_45nm_lp.yaml"])
    ap.add_argument("--tag", default="main")
    ap.add_argument("--n-designs", type=int, default=300,
                    help="base design points, each simulated under every technique")
    ap.add_argument("--techniques", nargs="*", default=None)
    args = ap.parse_args()

    ensure_directories()
    log = setup_logging("assist", filename=f"03_assist_study_{args.tag}.log")
    cfg = load_config(*args.config)
    seeds = seed_everything(int(cfg["experiment"]["seed"]))
    manifest = RunManifest(stage=f"assist_{args.tag}", config_hash=cfg.hash, seeds=seeds)

    with stage(f"assist[{args.tag}]", log, manifest,
               LOGS_DIR / f"manifest_assist_{args.tag}.json"):
        library = AssistLibrary.from_config(cfg)
        names = args.techniques or library.names
        log.info("Assist techniques: %s", names)

        ds = cfg.design_space
        design = generate_design(args.n_designs, ds.n_dim, "lhs",
                                 int(cfg["experiment"]["seed"]), "assist_study",
                                 compute_diagnostics=False)

        # Clip assisted values back into the declared design space so no
        # technique is evaluated outside the region the models were fitted on.
        clip = {v.name: (v.low, v.high) for v in ds.variables if v.kind != "categorical"}

        base_samples = [ds.decode_row(u) for u in design.points]
        env = resolve_environment(probe=False)
        campaign = SimulationCampaign(cfg, environment=env,
                                      chunk_dir=INTERIM_DIR / f"assist_{args.tag}")

        # The comparison must be *controlled*: the baseline pins the wordline
        # boost, the precharge ratio, the wordline low level and the dielectric,
        # so every technique is composed on top of the baseline and differs from
        # it only by its own knobs.  Applying a technique directly to the raw
        # sampled design would let it inherit whatever the Latin hypercube drew
        # for the knobs the baseline pins -- a technique that changes nothing
        # electrical would then still appear to shift the metrics.
        baseline_technique = library.get("baseline")

        frames = []
        for name in names:
            technique = library[name]
            log.info("--- technique '%s' ---", name)
            samples = []
            for k, base in enumerate(base_samples):
                start = (baseline_technique.apply(base, clip=clip)
                         if baseline_technique is not None else dict(base))
                # Apply the technique to the *design variables*, then rebuild the
                # derived quantities so VPP, Cbl, VBLpre etc. stay consistent.
                modified = technique.apply(start, clip=clip)
                modified = ds.apply_derived(modified)
                modified["assist_config"] = name
                modified["base_design_id"] = k
                samples.append(modified)

            # `assist_config` and `base_design_id` travel inside each sample dict
            # and are copied into the record by simulate_one, so they survive a
            # failed row -- re-attaching them positionally here would misalign
            # the moment any row dropped out.
            result = _run_samples(campaign, samples, log)
            if "assist_config" not in result.columns:
                result["assist_config"] = name
            frames.append(result)

        frame = pd.concat(frames, ignore_index=True)
        frame, coerced = sanitize_for_parquet(frame)
        if coerced:
            log.info("Coerced mixed-type columns for storage: %s", coerced)
            manifest.outputs["coerced_columns"] = coerced
        out = PROCESSED_DIR / f"assist_study_{args.tag}.parquet"
        frame.to_parquet(out, index=False)
        manifest.outputs["dataset"] = str(out)
        log.info("Assist study: %d simulations across %d techniques",
                 len(frame), len(names))

        # ---------------- summary -------------------------------------
        ok = frame[frame["simulation_ok"].astype(bool)] if "simulation_ok" in frame else frame
        cols = [m for m in METRICS if m in ok.columns]
        summary = ok.groupby("assist_config")[cols].median().reset_index()
        feas = (ok.assign(feasible=ok.get("read_success", False).astype(bool)
                          & ok.get("write_success", False).astype(bool))
                .groupby("assist_config")["feasible"].mean().mul(100).reset_index()
                .rename(columns={"feasible": "feasible_pct"}))
        summary = summary.merge(feas, on="assist_config")

        baseline = "baseline"
        if baseline in summary["assist_config"].values:
            base_row = summary[summary["assist_config"] == baseline].iloc[0]
            rel = summary.copy()
            for c in cols:
                rel[c] = 100.0 * (summary[c] / base_row[c] - 1.0)
            rel = rel[rel["assist_config"] != baseline]
            export_table(rel, TableSpec(
                name=f"assist_relative_effect_{args.tag}",
                caption="Median relative effect of each assist technique against "
                        "the unassisted baseline.",
                notes="Percentage change of the median over the same "
                      f"{args.n_designs} paired base designs; positive means the "
                      "metric increased."), TABLES_DIR)

        export_table(summary, TableSpec(
            name=f"assist_summary_{args.tag}",
            caption="Median performance of each DRAM assist technique.",
            notes=f"Paired study: every technique is applied to the identical set "
                  f"of {args.n_designs} base designs and simulated in NGSpice."),
            TABLES_DIR)

        # ---------------- paired significance testing ------------------
        stats_rows = []
        for metric in cols:
            pivot = ok.pivot_table(index="base_design_id", columns="assist_config",
                                   values=metric, aggfunc="mean").dropna()
            if pivot.empty or baseline not in pivot.columns or len(pivot) < 5:
                continue
            groups = {c: pivot[c].to_numpy() for c in pivot.columns}
            res = wilcoxon_holm(groups, alpha=0.05, control=baseline)
            res.insert(0, "metric", metric)
            stats_rows.append(res)
        if stats_rows:
            stats = pd.concat(stats_rows, ignore_index=True)
            export_table(stats, TableSpec(
                name=f"assist_significance_{args.tag}",
                caption="Paired Wilcoxon signed-rank tests of each assist technique "
                        "against the baseline, with Holm correction and Cliff's "
                        "delta effect size."), TABLES_DIR)
            manifest.outputs["n_significant"] = int(stats["significant"].sum())

        export_table(pd.DataFrame(library.describe()), TableSpec(
            name="assist_technique_definitions",
            caption="Definition of each assist technique evaluated."), TABLES_DIR)

    return 0


def _run_samples(campaign: SimulationCampaign, samples: list[dict], log) -> pd.DataFrame:
    """Simulate an explicit list of design points through the campaign pool."""
    import math
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from dramdt.dataset.campaign import _simulate_chunk, _worker_init

    chunk = max(int(campaign.exp.get("chunk_size", 125)), 1)
    n_chunks = math.ceil(len(samples) / chunk)
    env_desc = {
        "ngspice_exe": str(campaign.env.exe) if campaign.env.exe else "",
        "ngspice_dll": str(campaign.env.dll) if campaign.env.dll else "",
        "corner_cards": {k: str(v) for k, v in campaign.tech.corner_cards.items()},
    }
    work = []
    for c in range(n_chunks):
        lo, hi = c * chunk, min((c + 1) * chunk, len(samples))
        work.append((c, [(i, samples[i], None) for i in range(lo, hi)]))

    records: list[dict] = []
    with ProcessPoolExecutor(max_workers=campaign.n_workers,
                             initializer=_worker_init,
                             initargs=(dict(campaign.cfg), env_desc, False, None)) as pool:
        futures = [pool.submit(_simulate_chunk, w) for w in work]
        for k, fut in enumerate(as_completed(futures), start=1):
            _cid, recs = fut.result()
            records.extend(recs)
            if k % 5 == 0 or k == len(futures):
                log.info("    %d/%d chunks", k, len(futures))
    return (pd.DataFrame.from_records(records)
            .sort_values("sample_id").reset_index(drop=True))


if __name__ == "__main__":
    raise SystemExit(main())
