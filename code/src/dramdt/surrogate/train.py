"""Cross-validated surrogate training, comparison, tuning and persistence.

Protocol
--------
The dataset carries a deterministic ``split`` column (train / val / test).

* **Model comparison** uses K-fold cross-validation over ``train + val``.  Every
  fold's held-out score is retained, because the statistical tests in
  :mod:`dramdt.statistics` need per-fold observations, not just their mean.
* **Reported generalisation** is measured once on the untouched ``test`` split
  by a model refitted on all of ``train + val``.
* **Hyper-parameter tuning** (Optuna) runs *inside* the same CV loop on
  ``train + val`` only, so the test split is never used for selection.

Rows whose target is missing are dropped per target, not globally: a design
whose sense amplifier never resolves has no read delay, but it still has a
retention time and must remain in the retention model's training set.
"""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from ..logging_utils import get_logger
from ..seeds import derive_seed
from .metrics import bootstrap_ci, regression_metrics
from .zoo import (MODEL_REGISTRY, SUBSAMPLE_LIMITS, build_model,
                  build_preprocessor, model_families, needs_scaling)

__all__ = ["SurrogateTrainer", "TrainedSurrogate", "SurrogateComparison"]

log = get_logger(__name__)


# --------------------------------------------------------------------------
@dataclass
class TrainedSurrogate:
    """A fitted pipeline plus everything needed to reproduce and audit it."""

    target: str
    model_name: str
    family: str
    pipeline: Any = None
    features_numeric: list[str] = field(default_factory=list)
    features_categorical: list[str] = field(default_factory=list)
    cv_scores: dict[str, list[float]] = field(default_factory=dict)
    cv_summary: dict[str, float] = field(default_factory=dict)
    test_metrics: dict[str, float] = field(default_factory=dict)
    test_ci: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    fit_time_s: float = 0.0
    predict_time_us_per_sample: float = 0.0
    n_train: int = 0
    n_compare: int = 0
    n_test: int = 0
    subsampled_to: int | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def features(self) -> list[str]:
        return list(self.features_numeric) + list(self.features_categorical)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("surrogate has no fitted pipeline")
        return np.asarray(self.pipeline.predict(X[self.features]))

    def to_record(self) -> dict[str, Any]:
        rec: dict[str, Any] = {
            "target": self.target, "model": self.model_name, "family": self.family,
            "n_train": self.n_train, "n_compare": self.n_compare,
            "n_test": self.n_test,
            "fit_time_s": round(self.fit_time_s, 4),
            "predict_us_per_sample": round(self.predict_time_us_per_sample, 4),
            "subsampled_to": self.subsampled_to or "",
        }
        for k, v in self.cv_summary.items():
            rec[f"cv_{k}"] = v
        for k, v in self.test_metrics.items():
            rec[f"test_{k}"] = v
        for k, (pt, lo, hi) in self.test_ci.items():
            rec[f"test_{k}_ci_low"] = lo
            rec[f"test_{k}_ci_high"] = hi
        rec["notes"] = "; ".join(self.notes)
        return rec


@dataclass
class SurrogateComparison:
    """Everything the surrogate stage produced."""

    models: dict[str, dict[str, TrainedSurrogate]] = field(default_factory=dict)
    leaderboard: pd.DataFrame = field(default_factory=pd.DataFrame)
    fold_scores: pd.DataFrame = field(default_factory=pd.DataFrame)
    best: dict[str, str] = field(default_factory=dict)
    tuning: dict[str, Any] = field(default_factory=dict)

    def best_model(self, target: str) -> TrainedSurrogate:
        return self.models[target][self.best[target]]


# --------------------------------------------------------------------------
class SurrogateTrainer:
    """Trains and compares the model zoo for every configured target."""

    def __init__(self, config: Mapping[str, Any]):
        sc = dict(config.get("surrogate", {}))
        self.cfg = config
        self.seed = int(config.get("experiment", {}).get("seed", 0))
        self.n_splits = int(sc.get("cv_folds", 5))
        self.model_names: list[str] = list(sc.get("models") or list(MODEL_REGISTRY))
        self.explicit_features: list[str] | None = sc.get("features")
        self.categorical: list[str] = list(sc.get("categorical_features", ["corner"]))
        self.exclude: set[str] = set(sc.get("exclude_features", []))
        self.hyperparameters: dict[str, dict[str, Any]] = dict(sc.get("hyperparameters", {}))
        self.bootstrap_n = int(sc.get("bootstrap_iterations", 500))
        self.tune_cfg = dict(sc.get("tuning", {}))
        self.primary_metric = str(sc.get("primary_metric", "r2"))
        self.max_train_rows = int(sc.get("max_train_rows", 0) or 0)
        # Cross-validating 14 model families on the full development set is
        # dominated by a few super-linear learners.  The *comparison* therefore
        # runs on a capped, deterministic subsample, while the model finally
        # kept for the digital twin is refitted on the entire development set.
        # Both sizes are recorded per model (n_compare / n_train) so the
        # protocol is visible in the leaderboard rather than implied.
        self.comparison_max_rows = int(sc.get("comparison_max_rows", 0) or 0)

        # targets: prefer the transformed column when a transform was declared
        self.targets: list[str] = []
        self.target_labels: dict[str, str] = {}
        for obj in config.get("objectives", []):
            name = obj["name"]
            col = f"log10_{name}" if obj.get("transform") == "log10" else name
            self.targets.append(col)
            self.target_labels[col] = obj.get("label", name)
        if sc.get("targets"):
            self.targets = list(sc["targets"])

    # ------------------------------------------------------------------
    def resolve_features(self, df: pd.DataFrame) -> tuple[list[str], list[str]]:
        """Split the modelling features into numeric and categorical lists."""
        if self.explicit_features:
            feats = [f for f in self.explicit_features if f in df.columns]
        else:
            ds_names = list(self.cfg.design_space.names) if hasattr(self.cfg, "design_space") \
                else []
            engineered = ["charge_share_ratio", "cbl_over_cs", "signal_charge_c",
                          "stored_charge_c", "vwl_overdrive", "vgs_retention",
                          "access_aspect", "access_area", "temperature_k",
                          "inv_thermal_v", "sa_drive_ratio", "sa_aspect"]
            feats = [f for f in (ds_names + engineered) if f in df.columns]
        feats = [f for f in feats if f not in self.exclude]

        cats = [f for f in feats if f in self.categorical or df[f].dtype == object]
        nums = [f for f in feats if f not in cats]
        return nums, cats

    # ------------------------------------------------------------------
    def _make_pipeline(self, model_name: str, nums: list[str], cats: list[str],
                       params: Mapping[str, Any] | None = None):
        from sklearn.pipeline import Pipeline
        pre = build_preprocessor(nums, cats, scale=needs_scaling(model_name))
        est = build_model(model_name, seed=derive_seed(self.seed, f"model:{model_name}"),
                          **dict(params or {}))
        return Pipeline([("pre", pre), ("model", est)])

    @staticmethod
    def _frame_for(df: pd.DataFrame, target: str, features: Sequence[str]
                   ) -> tuple[pd.DataFrame, np.ndarray]:
        cols = list(features) + [target]
        sub = df[cols].copy()
        y = pd.to_numeric(sub[target], errors="coerce").to_numpy(dtype=float)
        keep = np.isfinite(y)
        return sub.loc[keep, list(features)], y[keep]

    def _subsample(self, X: pd.DataFrame, y: np.ndarray, limit: int, tag: str
                   ) -> tuple[pd.DataFrame, np.ndarray, int | None]:
        if limit <= 0 or len(X) <= limit:
            return X, y, None
        rng = np.random.default_rng(derive_seed(self.seed, f"subsample:{tag}"))
        idx = rng.choice(len(X), limit, replace=False)
        return X.iloc[idx], y[idx], limit

    # ------------------------------------------------------------------
    def fit_target(self, df: pd.DataFrame, target: str,
                   model_names: Sequence[str] | None = None
                   ) -> tuple[dict[str, TrainedSurrogate], list[dict[str, Any]]]:
        """Cross-validate and test every model for one target."""
        from sklearn.model_selection import KFold

        nums, cats = self.resolve_features(df)
        features = nums + cats
        names = list(model_names or self.model_names)

        dev_mask = df["split"].isin(["train", "val"]) if "split" in df else np.ones(len(df), bool)
        X_dev, y_dev = self._frame_for(df[dev_mask], target, features)
        X_test, y_test = self._frame_for(df[~dev_mask], target, features) \
            if "split" in df else (X_dev.iloc[:0], y_dev[:0])

        if len(X_dev) < self.n_splits * 5:
            log.warning("Target %r has only %d usable rows; skipping", target, len(X_dev))
            return {}, []

        if self.max_train_rows:
            X_dev, y_dev, _ = self._subsample(X_dev, y_dev, self.max_train_rows,
                                              f"dev:{target}")

        # Comparison set (cross-validated) vs. full development set (final refit).
        X_cmp, y_cmp, cmp_cap = self._subsample(
            X_dev, y_dev, self.comparison_max_rows, f"compare:{target}")

        log.info("Target %-28s | %d dev rows (%d used for CV), %d test rows, "
                 "%d features", target, len(X_dev), len(X_cmp), len(X_test),
                 len(features))

        kf = KFold(n_splits=self.n_splits, shuffle=True,
                   random_state=derive_seed(self.seed, f"cv:{target}"))
        folds = list(kf.split(X_cmp))

        trained: dict[str, TrainedSurrogate] = {}
        fold_rows: list[dict[str, Any]] = []

        for name in names:
            if name not in MODEL_REGISTRY:
                log.warning("Skipping unknown model %r", name)
                continue
            family = model_families()[name]
            params = self.hyperparameters.get(name, {})
            limit = SUBSAMPLE_LIMITS.get(name, 0)

            try:
                surrogate = TrainedSurrogate(
                    target=target, model_name=name, family=family,
                    features_numeric=nums, features_categorical=cats,
                    hyperparameters=dict(params))

                # ---------------- cross-validation --------------------
                per_fold: dict[str, list[float]] = {}
                t_cv = time.perf_counter()
                for k, (tr, te) in enumerate(folds):
                    Xtr, ytr = X_cmp.iloc[tr], y_cmp[tr]
                    Xte, yte = X_cmp.iloc[te], y_cmp[te]
                    Xtr, ytr, sub = self._subsample(Xtr, ytr, limit, f"{name}:{target}:{k}")
                    pipe = self._make_pipeline(name, nums, cats, params)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        pipe.fit(Xtr, ytr)
                        pred = pipe.predict(Xte)
                    scores = regression_metrics(yte, pred)
                    for mk, mv in scores.items():
                        per_fold.setdefault(mk, []).append(mv)
                    fold_rows.append({"target": target, "model": name, "family": family,
                                      "fold": k, **scores})
                    if sub:
                        surrogate.subsampled_to = sub
                cv_time = time.perf_counter() - t_cv

                surrogate.cv_scores = per_fold
                surrogate.cv_summary = {
                    f"{k}_mean": float(np.nanmean(v)) for k, v in per_fold.items()}
                surrogate.cv_summary.update({
                    f"{k}_std": float(np.nanstd(v)) for k, v in per_fold.items()})

                # ---------------- refit on all dev, score on test ------
                # refit on the entire development set, not the comparison subsample
                Xfit, yfit, sub = self._subsample(X_dev, y_dev, limit,
                                                  f"{name}:{target}:final")
                pipe = self._make_pipeline(name, nums, cats, params)
                t0 = time.perf_counter()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pipe.fit(Xfit, yfit)
                surrogate.fit_time_s = time.perf_counter() - t0
                surrogate.pipeline = pipe
                surrogate.n_train = len(Xfit)
                surrogate.n_compare = len(X_cmp)
                surrogate.n_test = len(X_test)
                if sub:
                    surrogate.subsampled_to = sub
                    surrogate.notes.append(
                        f"fitted on a {sub}-row subsample (model cost is super-linear in n)")
                if cmp_cap:
                    surrogate.notes.append(
                        f"cross-validated on a {cmp_cap}-row subsample; final model "
                        f"refitted on all {len(X_dev)} development rows")

                if len(X_test):
                    t0 = time.perf_counter()
                    pred = pipe.predict(X_test)
                    dt = time.perf_counter() - t0
                    surrogate.predict_time_us_per_sample = 1e6 * dt / max(len(X_test), 1)
                    surrogate.test_metrics = regression_metrics(y_test, pred)
                    for metric in ("r2", "rmse", "mae"):
                        surrogate.test_ci[metric] = bootstrap_ci(
                            y_test, pred, metric=metric, n_boot=self.bootstrap_n,
                            seed=derive_seed(self.seed, f"boot:{name}:{target}:{metric}"))

                trained[name] = surrogate
                log.info("  %-14s cv_%s=%.4f+-%.4f  test_%s=%.4f  (%.1fs cv, %.1fs fit)",
                         name, self.primary_metric,
                         surrogate.cv_summary.get(f"{self.primary_metric}_mean", float("nan")),
                         surrogate.cv_summary.get(f"{self.primary_metric}_std", float("nan")),
                         self.primary_metric,
                         surrogate.test_metrics.get(self.primary_metric, float("nan")),
                         cv_time, surrogate.fit_time_s)

            except ImportError as exc:
                log.warning("  %-14s unavailable: %s", name, exc)
            except Exception as exc:
                log.exception("  %-14s FAILED: %s", name, exc)

        return trained, fold_rows

    # ------------------------------------------------------------------
    def fit_all(self, df: pd.DataFrame) -> SurrogateComparison:
        """Run the full comparison across every target."""
        comparison = SurrogateComparison()
        records: list[dict[str, Any]] = []
        all_folds: list[dict[str, Any]] = []

        for target in self.targets:
            if target not in df.columns:
                log.warning("Target %r not in the dataset; skipping", target)
                continue
            trained, folds = self.fit_target(df, target)
            if not trained:
                continue
            comparison.models[target] = trained
            all_folds.extend(folds)
            records.extend(s.to_record() for s in trained.values())

            key = f"{self.primary_metric}_mean"
            best = max(trained.values(),
                       key=lambda s: s.cv_summary.get(key, -np.inf))
            comparison.best[target] = best.model_name
            log.info("Best surrogate for %s: %s (cv %s = %.4f)",
                     target, best.model_name, self.primary_metric,
                     best.cv_summary.get(key, float("nan")))

        comparison.leaderboard = pd.DataFrame.from_records(records)
        comparison.fold_scores = pd.DataFrame.from_records(all_folds)
        return comparison

    # ------------------------------------------------------------------
    def tune(self, df: pd.DataFrame, target: str, model_name: str,
             n_trials: int | None = None) -> dict[str, Any]:
        """Optuna hyper-parameter search, cross-validated on train+val only."""
        import optuna
        from sklearn.model_selection import KFold

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        n_trials = int(n_trials or self.tune_cfg.get("n_trials", 40))
        space = self.tune_cfg.get("spaces", {}).get(model_name)
        if not space:
            log.info("No tuning space configured for %r; skipping", model_name)
            return {}

        nums, cats = self.resolve_features(df)
        features = nums + cats
        dev = df[df["split"].isin(["train", "val"])] if "split" in df else df
        X, y = self._frame_for(dev, target, features)
        limit = SUBSAMPLE_LIMITS.get(model_name, 0)
        tune_rows = int(self.tune_cfg.get("max_rows", 15000))
        X, y, _ = self._subsample(X, y, tune_rows, f"tune:{model_name}:{target}")

        kf = KFold(n_splits=max(self.n_splits - 2, 3), shuffle=True,
                   random_state=derive_seed(self.seed, f"tunecv:{target}"))
        folds = list(kf.split(X))

        def objective(trial: "optuna.Trial") -> float:
            params: dict[str, Any] = {}
            for pname, spec in space.items():
                kind = spec.get("type", "float")
                if kind == "int":
                    params[pname] = trial.suggest_int(pname, int(spec["low"]), int(spec["high"]),
                                                      step=int(spec.get("step", 1)))
                elif kind == "categorical":
                    params[pname] = trial.suggest_categorical(pname, spec["choices"])
                else:
                    params[pname] = trial.suggest_float(pname, float(spec["low"]),
                                                        float(spec["high"]),
                                                        log=bool(spec.get("log", False)))
            scores = []
            for tr, te in folds:
                Xtr, ytr = X.iloc[tr], y[tr]
                Xtr, ytr, _ = self._subsample(Xtr, ytr, limit, f"tune:{model_name}")
                pipe = self._make_pipeline(model_name, nums, cats, params)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pipe.fit(Xtr, ytr)
                    pred = pipe.predict(X.iloc[te])
                scores.append(regression_metrics(y[te], pred).get(self.primary_metric, np.nan))
            return float(np.nanmean(scores))

        sampler = optuna.samplers.TPESampler(
            seed=derive_seed(self.seed, f"optuna:{model_name}:{target}"))
        study = optuna.create_study(direction="maximize", sampler=sampler,
                                    study_name=f"{model_name}::{target}")
        t0 = time.perf_counter()
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        elapsed = time.perf_counter() - t0

        log.info("Optuna %s/%s: best %s=%.5f after %d trials (%.1fs)",
                 model_name, target, self.primary_metric, study.best_value, n_trials, elapsed)
        return {
            "model": model_name, "target": target,
            "best_params": study.best_params, "best_value": float(study.best_value),
            "n_trials": n_trials, "elapsed_s": round(elapsed, 2),
            "history": [{"trial": t.number, "value": t.value} for t in study.trials
                        if t.value is not None],
        }
