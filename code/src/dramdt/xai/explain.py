"""Explainability for the DRAM surrogates.

Four complementary views are produced, because no single one is sufficient:

``SHAP``                  additive, per-prediction attributions with a solid
                          game-theoretic grounding; gives both global ranking
                          and local explanations.
``permutation importance``model-agnostic, measured on held-out data, so it
                          reflects predictive value rather than split counts.
``partial dependence``    the average marginal effect of a feature -- readable
                          as a design trend ("raise VPP by 100 mV and ...").
``ALE``                   accumulated local effects, which stay valid when the
                          design variables are correlated (partial dependence
                          does not, and the derived variables here *are*
                          correlated by construction).

Explanations are computed on the *transformed* feature matrix that the final
estimator actually sees, so one-hot corner columns appear individually and
nothing is attributed to a preprocessing step.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from ..logging_utils import get_logger
from ..seeds import derive_seed

__all__ = ["SurrogateExplainer", "ExplanationBundle",
           "permutation_importance_frame", "partial_dependence_frame",
           "ale_frame"]

log = get_logger(__name__)

_TREE_MODELS = {"random_forest", "extra_trees", "hist_gbr",
                "xgboost", "lightgbm", "catboost"}


@dataclass
class ExplanationBundle:
    """Everything the explainer produced for one (target, model)."""

    target: str
    model_name: str
    shap_values: np.ndarray | None = None
    shap_features: list[str] = field(default_factory=list)
    shap_data: np.ndarray | None = None
    shap_importance: pd.DataFrame = field(default_factory=pd.DataFrame)
    shap_interactions: pd.DataFrame = field(default_factory=pd.DataFrame)
    permutation_importance: pd.DataFrame = field(default_factory=pd.DataFrame)
    partial_dependence: pd.DataFrame = field(default_factory=pd.DataFrame)
    partial_dependence_2d: pd.DataFrame = field(default_factory=pd.DataFrame)
    ale: pd.DataFrame = field(default_factory=pd.DataFrame)
    explainer_kind: str = ""
    elapsed_s: float = 0.0
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
def _split_pipeline(pipeline) -> tuple[Any, Any]:
    """Return ``(preprocessor, final_estimator)`` from a fitted Pipeline."""
    try:
        return pipeline.named_steps["pre"], pipeline.named_steps["model"]
    except Exception:
        return None, pipeline


def _transformed(pipeline, X: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    pre, _ = _split_pipeline(pipeline)
    if pre is None:
        return np.asarray(X, dtype=float), list(X.columns)
    Z = pre.transform(X)
    Z = np.asarray(Z.todense()) if hasattr(Z, "todense") else np.asarray(Z)
    try:
        names = [str(n) for n in pre.get_feature_names_out()]
    except Exception:                                       # pragma: no cover
        names = [f"f{i}" for i in range(Z.shape[1])]
    return Z, names


# --------------------------------------------------------------------------
def permutation_importance_frame(pipeline, X: pd.DataFrame, y: np.ndarray,
                                 n_repeats: int = 15, seed: int = 0,
                                 max_rows: int = 5000) -> pd.DataFrame:
    """Model-agnostic permutation importance on held-out data."""
    from sklearn.inspection import permutation_importance

    if len(X) > max_rows:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), max_rows, replace=False)
        X, y = X.iloc[idx], y[idx]
    res = permutation_importance(pipeline, X, y, n_repeats=n_repeats,
                                 random_state=seed, n_jobs=1,
                                 scoring="r2")
    return (pd.DataFrame({
        "feature": list(X.columns),
        "importance_mean": res.importances_mean,
        "importance_std": res.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True))


def partial_dependence_frame(pipeline, X: pd.DataFrame, features: Sequence[str],
                             grid_resolution: int = 40,
                             max_rows: int = 2000, seed: int = 0) -> pd.DataFrame:
    """1-D partial dependence for a list of features."""
    from sklearn.inspection import partial_dependence

    if len(X) > max_rows:
        rng = np.random.default_rng(seed)
        X = X.iloc[rng.choice(len(X), max_rows, replace=False)]

    rows: list[dict[str, Any]] = []
    for feat in features:
        if feat not in X.columns:
            continue
        pd_res = None
        # Heavily skewed engineered features (products of two design variables)
        # can have identical 5th/95th percentiles, which makes sklearn refuse to
        # build a grid.  Widen the percentile window before giving up.
        for pct in ((0.05, 0.95), (0.01, 0.99), (0.0, 1.0)):
            try:
                pd_res = partial_dependence(pipeline, X, [feat],
                                            grid_resolution=grid_resolution,
                                            percentiles=pct, kind="average")
                break
            except Exception as exc:
                last = exc
        if pd_res is None:
            log.warning("Partial dependence failed for %r: %s", feat, last)
            continue
        grid = np.asarray(pd_res["grid_values"][0], dtype=float)
        avg = np.asarray(pd_res["average"][0], dtype=float)
        for g, a in zip(grid, avg):
            rows.append({"feature": feat, "value": float(g), "partial_dependence": float(a)})
    return pd.DataFrame(rows)


def partial_dependence_2d(pipeline, X: pd.DataFrame,
                          pairs: Sequence[tuple[str, str]],
                          grid_resolution: int = 20,
                          max_rows: int = 1500, seed: int = 0) -> pd.DataFrame:
    """Two-way partial dependence surfaces (interaction views)."""
    from sklearn.inspection import partial_dependence

    if len(X) > max_rows:
        rng = np.random.default_rng(seed)
        X = X.iloc[rng.choice(len(X), max_rows, replace=False)]

    rows: list[dict[str, Any]] = []
    for f1, f2 in pairs:
        if f1 not in X.columns or f2 not in X.columns:
            continue
        # sklearn's `partial_dependence` takes a *flat* list of the interacting
        # features ([f1, f2]); the nested form [(f1, f2)] belongs to
        # PartialDependenceDisplay and is read here as one unknown column name.
        res = None
        for pct in ((0.05, 0.95), (0.01, 0.99), (0.0, 1.0)):
            try:
                res = partial_dependence(pipeline, X, [f1, f2],
                                         grid_resolution=grid_resolution,
                                         percentiles=pct, kind="average")
                break
            except Exception as exc:
                last = exc
        if res is None:
            log.warning("2-D partial dependence failed for (%s, %s): %s", f1, f2, last)
            continue
        g1 = np.asarray(res["grid_values"][0], dtype=float)
        g2 = np.asarray(res["grid_values"][1], dtype=float)
        Z = np.asarray(res["average"][0], dtype=float)
        for i, a in enumerate(g1):
            for j, b in enumerate(g2):
                rows.append({"feature_1": f1, "feature_2": f2,
                             "value_1": float(a), "value_2": float(b),
                             "partial_dependence": float(Z[i, j])})
    return pd.DataFrame(rows)


def ale_frame(pipeline, X: pd.DataFrame, features: Sequence[str],
              n_bins: int = 25, max_rows: int = 5000, seed: int = 0) -> pd.DataFrame:
    """First-order accumulated local effects.

    ALE differences are evaluated *within* each quantile bin, so -- unlike
    partial dependence -- the model is never queried at feature combinations
    that the design space cannot produce.
    """
    if len(X) > max_rows:
        rng = np.random.default_rng(seed)
        X = X.iloc[rng.choice(len(X), max_rows, replace=False)]

    rows: list[dict[str, Any]] = []
    for feat in features:
        if feat not in X.columns or not pd.api.types.is_numeric_dtype(X[feat]):
            continue
        vals = X[feat].to_numpy(dtype=float)
        edges = np.unique(np.quantile(vals[np.isfinite(vals)],
                                      np.linspace(0, 1, n_bins + 1)))
        if edges.size < 3:
            continue
        idx = np.clip(np.searchsorted(edges, vals, side="left") - 1, 0, edges.size - 2)

        local: list[float] = []
        for b in range(edges.size - 1):
            m = idx == b
            if m.sum() < 2:
                local.append(0.0)
                continue
            lo, hi = X[m].copy(), X[m].copy()
            lo[feat] = edges[b]
            hi[feat] = edges[b + 1]
            try:
                d = pipeline.predict(hi) - pipeline.predict(lo)
            except Exception as exc:                        # pragma: no cover
                log.warning("ALE failed for %r: %s", feat, exc)
                local.append(0.0)
                continue
            local.append(float(np.mean(d)))

        acc = np.concatenate([[0.0], np.cumsum(local)])
        acc = acc - np.mean(acc)                # centre so the effect is zero-mean
        for e, a in zip(edges, acc):
            rows.append({"feature": feat, "value": float(e), "ale": float(a)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
class SurrogateExplainer:
    """Runs the full explanation suite for one fitted surrogate."""

    def __init__(self, config: Mapping[str, Any]):
        x = dict(config.get("xai", {}))
        self.shap_cfg = dict(x.get("shap", {"enabled": True}))
        self.perm_cfg = dict(x.get("permutation_importance", {"enabled": True}))
        self.pdp_cfg = dict(x.get("partial_dependence", {"enabled": True}))
        self.ale_cfg = dict(x.get("ale", {"enabled": True}))
        self.seed = int(config.get("experiment", {}).get("seed", 0))

    # ------------------------------------------------------------------
    def _shap(self, surrogate, X: pd.DataFrame,
              bundle: ExplanationBundle) -> None:
        import shap

        pipeline = surrogate.pipeline
        _pre, model = _split_pipeline(pipeline)
        n_explain = min(int(self.shap_cfg.get("max_explain", 2000)), len(X))
        n_bg = int(self.shap_cfg.get("max_background", 200))
        rng = np.random.default_rng(derive_seed(self.seed, f"shap:{bundle.target}"))

        X_exp = X.iloc[rng.choice(len(X), n_explain, replace=False)]
        Z_exp, names = _transformed(pipeline, X_exp)
        bundle.shap_features = names
        bundle.shap_data = Z_exp

        if surrogate.model_name in _TREE_MODELS:
            try:
                explainer = shap.TreeExplainer(model)
                sv = explainer.shap_values(Z_exp, check_additivity=False)
                bundle.explainer_kind = "TreeExplainer (exact)"
            except Exception as exc:
                log.warning("TreeExplainer failed (%s); falling back to Permutation", exc)
                sv, bundle.explainer_kind = self._model_agnostic_shap(
                    pipeline, X, X_exp, n_bg, rng)
        else:
            sv, bundle.explainer_kind = self._model_agnostic_shap(
                pipeline, X, X_exp, n_bg, rng)

        if sv is None:
            bundle.notes.append("SHAP unavailable for this model")
            return
        sv = np.asarray(sv)
        if sv.ndim == 3:                     # (n, features, outputs)
            sv = sv[..., 0]
        # The model-agnostic explainer scores fewer rows than were prepared, so
        # the retained feature matrix must be trimmed to match: a beeswarm plot
        # colours each SHAP value by its own feature value and needs the two
        # arrays to line up row for row.
        if len(sv) != len(bundle.shap_data):
            bundle.shap_data = bundle.shap_data[:len(sv)]
        bundle.shap_values = sv

        imp = np.abs(sv).mean(axis=0)
        bundle.shap_importance = (pd.DataFrame({
            "feature": names[:sv.shape[1]],
            "mean_abs_shap": imp,
            "mean_shap": sv.mean(axis=0),
            "std_shap": sv.std(axis=0),
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))
        bundle.shap_importance["importance_pct"] = (
            100.0 * bundle.shap_importance["mean_abs_shap"]
            / max(bundle.shap_importance["mean_abs_shap"].sum(), 1e-12))

        # pairwise interaction strength for tree models
        if surrogate.model_name in _TREE_MODELS:
            n_int = min(int(self.shap_cfg.get("interaction_max", 400)), len(Z_exp))
            try:
                inter = shap.TreeExplainer(model).shap_interaction_values(Z_exp[:n_int])
                inter = np.asarray(inter)
                if inter.ndim == 4:
                    inter = inter[..., 0]
                strength = np.abs(inter).mean(axis=0)
                np.fill_diagonal(strength, 0.0)
                rows = []
                k = strength.shape[0]
                for i in range(k):
                    for j in range(i + 1, k):
                        rows.append({"feature_1": names[i], "feature_2": names[j],
                                     "interaction_strength": float(strength[i, j])})
                bundle.shap_interactions = (pd.DataFrame(rows)
                                            .sort_values("interaction_strength",
                                                         ascending=False)
                                            .reset_index(drop=True))
            except Exception as exc:
                bundle.notes.append(f"SHAP interaction values unavailable: {exc}")

    @staticmethod
    def _model_agnostic_shap(pipeline, X_full: pd.DataFrame, X_exp: pd.DataFrame,
                             n_bg: int, rng) -> tuple[np.ndarray | None, str]:
        import shap
        Z_full, _ = _transformed(pipeline, X_full)
        Z_exp, _ = _transformed(pipeline, X_exp)
        _pre, model = _split_pipeline(pipeline)
        bg = shap.kmeans(Z_full, min(n_bg, max(len(Z_full) // 10, 5)))
        try:
            explainer = shap.PermutationExplainer(model.predict, bg.data)
            sv = explainer(Z_exp[:min(len(Z_exp), 400)]).values
            return sv, "PermutationExplainer (model-agnostic)"
        except Exception as exc:
            log.warning("Model-agnostic SHAP failed: %s", exc)
            return None, "unavailable"

    # ------------------------------------------------------------------
    def explain(self, surrogate, X: pd.DataFrame, y: np.ndarray) -> ExplanationBundle:
        """Compute every enabled explanation for one surrogate."""
        bundle = ExplanationBundle(target=surrogate.target,
                                   model_name=surrogate.model_name)
        t0 = time.perf_counter()

        if self.shap_cfg.get("enabled", True):
            try:
                self._shap(surrogate, X, bundle)
            except Exception as exc:
                log.exception("SHAP failed for %s/%s: %s",
                              surrogate.target, surrogate.model_name, exc)
                bundle.notes.append(f"SHAP failed: {exc}")

        if self.perm_cfg.get("enabled", True):
            try:
                bundle.permutation_importance = permutation_importance_frame(
                    surrogate.pipeline, X, y,
                    n_repeats=int(self.perm_cfg.get("n_repeats", 15)),
                    seed=derive_seed(self.seed, f"perm:{surrogate.target}"),
                    max_rows=int(self.perm_cfg.get("max_rows", 5000)))
            except Exception as exc:
                log.exception("Permutation importance failed: %s", exc)
                bundle.notes.append(f"permutation importance failed: {exc}")

        # rank features for the dependence plots, preferring SHAP
        if not bundle.shap_importance.empty:
            ranked = [f for f in bundle.shap_importance["feature"].tolist()
                      if f.replace("num__", "").replace("cat__", "") in X.columns]
            ranked = [f.replace("num__", "").replace("cat__", "") for f in ranked]
        elif not bundle.permutation_importance.empty:
            ranked = bundle.permutation_importance["feature"].tolist()
        else:
            ranked = list(X.columns)
        top_n = int(self.pdp_cfg.get("n_features", 8))
        seen: list[str] = []
        for f in ranked:
            if f in X.columns and f not in seen:
                seen.append(f)
            if len(seen) >= top_n:
                break

        if self.pdp_cfg.get("enabled", True) and seen:
            try:
                bundle.partial_dependence = partial_dependence_frame(
                    surrogate.pipeline, X, seen,
                    grid_resolution=int(self.pdp_cfg.get("grid_resolution", 40)),
                    seed=derive_seed(self.seed, f"pdp:{surrogate.target}"))
                n_pairs = int(self.pdp_cfg.get("two_way_pairs", 4))
                pairs = [(seen[i], seen[j])
                         for i in range(min(len(seen), 4))
                         for j in range(i + 1, min(len(seen), 4))][:n_pairs]
                if pairs:
                    bundle.partial_dependence_2d = partial_dependence_2d(
                        surrogate.pipeline, X, pairs,
                        seed=derive_seed(self.seed, f"pdp2:{surrogate.target}"))
            except Exception as exc:
                log.exception("Partial dependence failed: %s", exc)
                bundle.notes.append(f"partial dependence failed: {exc}")

        if self.ale_cfg.get("enabled", True) and seen:
            try:
                bundle.ale = ale_frame(
                    surrogate.pipeline, X, seen,
                    n_bins=int(self.ale_cfg.get("n_bins", 25)),
                    seed=derive_seed(self.seed, f"ale:{surrogate.target}"))
            except Exception as exc:
                log.exception("ALE failed: %s", exc)
                bundle.notes.append(f"ALE failed: {exc}")

        bundle.elapsed_s = time.perf_counter() - t0
        log.info("Explained %s/%s in %.1f s (%s)", surrogate.target,
                 surrogate.model_name, bundle.elapsed_s, bundle.explainer_kind)
        return bundle
