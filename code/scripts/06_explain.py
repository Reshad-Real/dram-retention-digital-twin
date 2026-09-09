#!/usr/bin/env python
"""Stage 4b -- explainability of the surrogate models.

    python scripts/06_explain.py --tag main

Outputs (per response)
    results/xai/shap_importance_<target>.csv
    results/xai/shap_values_<target>.npz
    results/xai/shap_interactions_<target>.csv
    results/xai/permutation_importance_<target>.csv
    results/xai/partial_dependence_<target>.csv
    results/xai/ale_<target>.csv
    results/tables/xai_*.{csv,tex,md}
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dramdt.config import load_config                                   # noqa: E402
from dramdt.logging_utils import RunManifest, setup_logging, stage      # noqa: E402
from dramdt.paths import (LOGS_DIR, MODELS_DIR, PROCESSED_DIR, TABLES_DIR,
                          XAI_DIR, ensure_directories)                  # noqa: E402
from dramdt.reporting.tables import TableSpec, export_table             # noqa: E402
from dramdt.seeds import seed_everything                                # noqa: E402
from dramdt.surrogate.train import SurrogateTrainer                     # noqa: E402
from dramdt.xai import SurrogateExplainer                               # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", nargs="+",
                    default=["dram_1t1c_45nm_lp.yaml", "surrogate.yaml"])
    ap.add_argument("--tag", default="main")
    ap.add_argument("--targets", nargs="*", default=None)
    args = ap.parse_args()

    ensure_directories()
    log = setup_logging("xai", filename=f"06_explain_{args.tag}.log")
    cfg = load_config(*args.config)
    seeds = seed_everything(int(cfg["experiment"]["seed"]))
    manifest = RunManifest(stage=f"xai_{args.tag}", config_hash=cfg.hash, seeds=seeds)

    with stage(f"xai[{args.tag}]", log, manifest,
               LOGS_DIR / f"manifest_xai_{args.tag}.json"):
        import joblib
        surro_path = MODELS_DIR / f"surrogates_{args.tag}.joblib"
        if not surro_path.exists():
            log.error("Surrogates not found: %s", surro_path)
            return 2
        blob = joblib.load(surro_path)
        models, best = blob["models"], blob["best"]

        df = pd.read_parquet(PROCESSED_DIR / f"{args.tag}_processed.parquet")
        explainer = SurrogateExplainer(cfg)
        XAI_DIR.mkdir(parents=True, exist_ok=True)

        targets = args.targets or list(models)
        rankings: list[pd.DataFrame] = []

        for target in targets:
            if target not in models:
                log.warning("No surrogate for %r", target)
                continue
            surrogate = models[target][best[target]]
            dev = df[df["split"].isin(["train", "val"])] if "split" in df else df
            X, y = SurrogateTrainer._frame_for(dev, target, surrogate.features)
            if len(X) < 50:
                log.warning("Too few rows to explain %r", target)
                continue
            log.info("Explaining %s (%s) on %d rows", target,
                     surrogate.model_name, len(X))
            bundle = explainer.explain(surrogate, X, y)

            stem = target.replace("/", "_")
            if not bundle.shap_importance.empty:
                bundle.shap_importance.to_csv(
                    XAI_DIR / f"shap_importance_{stem}.csv", index=False)
                r = bundle.shap_importance.head(15).copy()
                r.insert(0, "response", target)
                rankings.append(r)
            if bundle.shap_values is not None:
                np.savez_compressed(
                    XAI_DIR / f"shap_values_{stem}.npz",
                    shap_values=bundle.shap_values,
                    data=bundle.shap_data,
                    features=np.array(bundle.shap_features, dtype=object))
            for name, frame in (("shap_interactions", bundle.shap_interactions),
                                ("permutation_importance", bundle.permutation_importance),
                                ("partial_dependence", bundle.partial_dependence),
                                ("partial_dependence_2d", bundle.partial_dependence_2d),
                                ("ale", bundle.ale)):
                if frame is not None and not frame.empty:
                    frame.to_csv(XAI_DIR / f"{name}_{stem}.csv", index=False)

            manifest.outputs[target] = {
                "model": surrogate.model_name,
                "explainer": bundle.explainer_kind,
                "elapsed_s": round(bundle.elapsed_s, 2),
                "top_features": (bundle.shap_importance["feature"].head(5).tolist()
                                 if not bundle.shap_importance.empty else []),
                "notes": bundle.notes,
            }

            if not bundle.shap_importance.empty:
                export_table(bundle.shap_importance.head(15), TableSpec(
                    name=f"xai_shap_importance_{stem}",
                    caption=f"Global SHAP feature importance for {target}.",
                    columns=["feature", "mean_abs_shap", "mean_shap",
                             "importance_pct"],
                    notes=f"Explainer: {bundle.explainer_kind}."), TABLES_DIR)

        if rankings:
            combined = pd.concat(rankings, ignore_index=True)
            combined.to_csv(XAI_DIR / f"shap_importance_all_{args.tag}.csv", index=False)
            pivot = combined.pivot_table(index="feature", columns="response",
                                         values="importance_pct", aggfunc="first")
            pivot["mean_importance_pct"] = pivot.mean(axis=1)
            pivot = pivot.sort_values("mean_importance_pct", ascending=False).reset_index()
            export_table(pivot.head(20), TableSpec(
                name=f"xai_importance_matrix_{args.tag}",
                caption="SHAP importance (% of total attribution) of each design "
                        "variable across all modelled responses.",
                notes="Blank entries mean the feature did not enter that response's "
                      "top-15 ranking."), TABLES_DIR)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
