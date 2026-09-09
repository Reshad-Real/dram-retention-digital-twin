#!/usr/bin/env python
"""Stage 4/6 -- train and compare the surrogate zoo, then build the digital twin.

    python scripts/04_train_surrogates.py
    python scripts/04_train_surrogates.py --no-tuning --models xgboost lightgbm

Outputs
    models/surrogates_<tag>.joblib          every fitted surrogate
    models/digital_twin_<tag>.joblib        the deployed twin
    results/surrogate/leaderboard_<tag>.csv comparison table
    results/surrogate/fold_scores_<tag>.csv per-fold CV scores (for the stats stage)
    results/surrogate/predictions_<tag>.parquet  test-split predictions
    results/tables/surrogate_*.{csv,tex,md}
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
from dramdt.logging_utils import RunManifest, setup_logging, stage      # noqa: E402
from dramdt.paths import (LOGS_DIR, MODELS_DIR, PROCESSED_DIR, SURROGATE_DIR,
                          TABLES_DIR, ensure_directories)               # noqa: E402
from dramdt.reporting.tables import TableSpec, export_table             # noqa: E402
from dramdt.seeds import seed_everything                                # noqa: E402
from dramdt.surrogate import SurrogateTrainer                           # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", nargs="+",
                    default=["dram_1t1c_45nm_lp.yaml", "surrogate.yaml"])
    ap.add_argument("--tag", default="main")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--no-tuning", action="store_true")
    ap.add_argument("--max-rows", type=int, default=None)
    args = ap.parse_args()

    ensure_directories()
    log = setup_logging("surrogate", filename=f"04_train_surrogates_{args.tag}.log")
    cfg = load_config(*args.config)
    seeds = seed_everything(int(cfg["experiment"]["seed"]))

    manifest = RunManifest(stage=f"surrogate_{args.tag}", config_hash=cfg.hash, seeds=seeds)

    with stage(f"surrogate[{args.tag}]", log, manifest,
               LOGS_DIR / f"manifest_surrogate_{args.tag}.json"):
        path = Path(args.dataset) if args.dataset else \
            PROCESSED_DIR / f"{args.tag}_processed.parquet"
        if not path.exists():
            log.error("Processed dataset not found: %s -- run scripts/02_generate_dataset.py",
                      path)
            return 2
        df = pd.read_parquet(path)
        log.info("Dataset: %s (%d rows x %d columns)", path.name, len(df), len(df.columns))
        manifest.parameters["dataset"] = str(path)

        if args.models:
            cfg["surrogate"]["models"] = list(args.models)
        if args.max_rows:
            cfg["surrogate"]["max_train_rows"] = int(args.max_rows)

        trainer = SurrogateTrainer(cfg)
        # extra responses the optimiser's constraints need
        for extra in cfg.get_path("surrogate.extra_targets", []) or []:
            if extra in df.columns and extra not in trainer.targets:
                trainer.targets.append(extra)
        log.info("Targets: %s", trainer.targets)
        log.info("Models:  %s", trainer.model_names)

        comparison = trainer.fit_all(df)
        if not comparison.models:
            log.error("No surrogates were trained")
            return 3

        SURROGATE_DIR.mkdir(parents=True, exist_ok=True)
        comparison.leaderboard.to_csv(
            SURROGATE_DIR / f"leaderboard_{args.tag}.csv", index=False)
        comparison.fold_scores.to_csv(
            SURROGATE_DIR / f"fold_scores_{args.tag}.csv", index=False)
        manifest.outputs["best_per_target"] = comparison.best

        # ---------------- optional Optuna tuning ----------------------
        tuning_results: dict[str, dict] = {}
        tune_cfg = cfg.get_path("surrogate.tuning", {}) or {}
        if tune_cfg.get("enabled", False) and not args.no_tuning:
            for target in list(comparison.models):
                best_name = comparison.best.get(target)
                if best_name not in (tune_cfg.get("models") or []):
                    continue
                res = trainer.tune(df, target, best_name)
                if res:
                    tuning_results[f"{target}::{best_name}"] = res
                    # refit the tuned model and keep it if it genuinely improves
                    tuned, _folds = trainer.fit_target(df, target, [best_name])
                    old = comparison.models[target][best_name]
                    trainer.hyperparameters.setdefault(best_name, {}).update(
                        res["best_params"])
                    retuned, _ = trainer.fit_target(df, target, [best_name])
                    new = retuned.get(best_name)
                    key = f"{trainer.primary_metric}_mean"
                    if new and new.cv_summary.get(key, -np.inf) > \
                            old.cv_summary.get(key, -np.inf):
                        comparison.models[target][best_name] = new
                        log.info("Tuned %s/%s improved CV %s: %.5f -> %.5f",
                                 target, best_name, trainer.primary_metric,
                                 old.cv_summary.get(key, float("nan")),
                                 new.cv_summary.get(key, float("nan")))
                    else:
                        log.info("Tuning did not improve %s/%s; keeping the "
                                 "default configuration", target, best_name)
            if tuning_results:
                (SURROGATE_DIR / f"tuning_{args.tag}.json").write_text(
                    json.dumps(tuning_results, indent=2, default=str), encoding="utf-8")
                manifest.outputs["tuning"] = {
                    k: v["best_value"] for k, v in tuning_results.items()}

        # ---------------- test-split predictions ----------------------
        preds = {}
        test = df[df["split"] == "test"] if "split" in df.columns else df
        for target, models in comparison.models.items():
            best = models[comparison.best[target]]
            X, y = SurrogateTrainer._frame_for(test, target, best.features)
            if len(X) == 0:
                continue
            preds[target] = pd.DataFrame({
                "y_true": y, "y_pred": best.predict(X), "target": target,
                "model": best.model_name})
        if preds:
            pd.concat(preds.values(), ignore_index=True).to_parquet(
                SURROGATE_DIR / f"predictions_{args.tag}.parquet", index=False)

        # ---------------- persist ------------------------------------
        import copy
        import joblib
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        surro_path = MODELS_DIR / f"surrogates_{args.tag}.joblib"

        # Persist the fitted pipeline only for the winner of each response.  The
        # losing pipelines add gigabytes (k-NN alone stores its whole training
        # set, and a 400-tree forest is ~100 MB) while adding nothing that the
        # leaderboard and the per-fold scores do not already record; their
        # metrics, hyper-parameters and notes are kept so the comparison stays
        # fully auditable.
        archive: dict[str, dict[str, Any]] = {}
        for target, models in comparison.models.items():
            archive[target] = {}
            for name, surrogate in models.items():
                if name == comparison.best.get(target):
                    archive[target][name] = surrogate
                else:
                    lean = copy.copy(surrogate)
                    lean.pipeline = None
                    lean.notes = list(surrogate.notes) + [
                        "fitted pipeline not persisted (non-winning model); "
                        "refit from the recorded hyperparameters to reproduce"]
                    archive[target][name] = lean
        joblib.dump({"models": archive, "best": comparison.best,
                     "config_hash": cfg.hash}, surro_path, compress=3)
        log.info("Surrogate archive: %s (%.1f MB, pipelines kept for the %d winners)",
                 surro_path.name, surro_path.stat().st_size / 1024 / 1024,
                 len(comparison.best))
        manifest.outputs["surrogates"] = str(surro_path)

        best_models = {t: comparison.models[t][comparison.best[t]]
                       for t in comparison.models}
        spice_runtime = float(pd.to_numeric(df.get("sim_runtime_s"),
                                            errors="coerce").mean()) \
            if "sim_runtime_s" in df.columns else float("nan")
        twin = DigitalTwin(cfg, best_models, metadata={
            "config_hash": cfg.hash,
            "dataset": str(path),
            "n_training_rows": int(len(df)),
            "mean_spice_runtime_s": spice_runtime,
            "best_models": {t: m.model_name for t, m in best_models.items()},
        })
        twin_path = twin.save(MODELS_DIR / f"digital_twin_{args.tag}.joblib")
        manifest.outputs["digital_twin"] = str(twin_path)

        bench = twin.benchmark(seed=int(cfg["experiment"]["seed"]))
        bench.to_csv(SURROGATE_DIR / f"twin_benchmark_{args.tag}.csv", index=False)
        manifest.outputs["twin_benchmark"] = bench.to_dict(orient="records")
        log.info("Twin latency: %s", bench.to_string(index=False))

        # ---------------- publication tables --------------------------
        lb = comparison.leaderboard.copy()
        cols = ["target", "model", "family", "n_compare", "n_train",
                "cv_r2_mean", "cv_r2_std", "test_r2", "test_rmse", "test_mae",
                "test_mape", "fit_time_s", "predict_us_per_sample"]
        cols = [c for c in cols if c in lb.columns]
        n_folds = int(cfg.get_path("surrogate.cv_folds", 5))
        export_table(lb[cols].sort_values(["target", "test_r2"], ascending=[True, False]),
                     TableSpec(name=f"surrogate_leaderboard_{args.tag}",
                               caption="Surrogate model comparison across all responses.",
                               notes=f"cv_* are {n_folds}-fold cross-validated means over "
                                     "n_compare rows of the train+val partition; the "
                                     "model is then refitted on all n_train rows and "
                                     "scored once on the held-out test partition. "
                                     "Kernel methods carry a further per-model cap "
                                     "(see the notes column)."),
                     TABLES_DIR)

        rows = []
        for target, models in comparison.models.items():
            best = models[comparison.best[target]]
            r = {"response": target, "best model": best.model_name,
                 "family": best.family, "n train": best.n_train, "n test": best.n_test}
            r.update({f"test {k}": v for k, v in best.test_metrics.items()
                      if k in ("r2", "rmse", "mae", "mape", "max_error")})
            for metric, (pt, lo, hi) in best.test_ci.items():
                r[f"test {metric} 95% CI"] = f"[{lo:.4f}, {hi:.4f}]"
            rows.append(r)
        export_table(pd.DataFrame(rows),
                     TableSpec(name=f"surrogate_best_{args.tag}",
                               caption="Best surrogate per response with bootstrap "
                                       "confidence intervals.",
                               highlight_best="test r2"),
                     TABLES_DIR)
        export_table(bench, TableSpec(
            name=f"twin_benchmark_{args.tag}",
            caption="Digital-twin evaluation latency and speed-up over SPICE.",
            notes="The SPICE reference is the mean simulation time actually "
                  "recorded while generating the training dataset."), TABLES_DIR)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
