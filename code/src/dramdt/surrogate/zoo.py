"""The surrogate model registry.

Every entry is a factory returning a scikit-learn compatible estimator, so the
whole comparison, the cross-validation and the digital twin can treat models
uniformly.  Families covered:

``linear``     ridge / elastic-net / polynomial ridge  -- interpretable baselines
``kernel``     support-vector regression, Gaussian-process regression
``neighbour``  k-nearest neighbours
``ensemble``   random forest, extra trees, gradient boosting
``boosting``   XGBoost, LightGBM, CatBoost
``neural``     multi-layer perceptron (the "ANN" of the proposed method)

Optional third-party libraries are imported lazily: if XGBoost is not installed
the registry simply does not offer it, rather than failing at import time.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Mapping

import numpy as np

from ..logging_utils import get_logger

__all__ = ["MODEL_REGISTRY", "build_model", "build_preprocessor",
           "model_families", "available_models", "needs_scaling",
           "SUBSAMPLE_LIMITS"]

log = get_logger(__name__)

#: Models whose cost is super-linear in n are fitted on a capped subsample.
#: The cap is recorded in the results so the comparison stays honest.
SUBSAMPLE_LIMITS: dict[str, int] = {
    "gpr": 2000,
    "svr_rbf": 5000,
}

#: Models that require standardised inputs.
_SCALE_REQUIRED = {"ridge", "elasticnet", "poly2_ridge", "svr_rbf", "gpr",
                   "knn", "mlp", "mlp_deep"}


def needs_scaling(name: str) -> bool:
    return name in _SCALE_REQUIRED


# --------------------------------------------------------------------------
# factories
# --------------------------------------------------------------------------
def _ridge(seed: int, **kw: Any):
    from sklearn.linear_model import Ridge
    return Ridge(alpha=kw.pop("alpha", 1.0), random_state=seed, **kw)


def _elasticnet(seed: int, **kw: Any):
    from sklearn.linear_model import ElasticNet
    return ElasticNet(alpha=kw.pop("alpha", 0.001), l1_ratio=kw.pop("l1_ratio", 0.5),
                      max_iter=kw.pop("max_iter", 5000), random_state=seed, **kw)


def _poly2_ridge(seed: int, **kw: Any):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures
    return make_pipeline(
        PolynomialFeatures(degree=kw.pop("degree", 2), include_bias=False),
        Ridge(alpha=kw.pop("alpha", 10.0), random_state=seed))


def _svr_rbf(seed: int, **kw: Any):
    from sklearn.svm import SVR
    return SVR(C=kw.pop("C", 10.0), epsilon=kw.pop("epsilon", 0.05),
               gamma=kw.pop("gamma", "scale"), **kw)


def _gpr(seed: int, **kw: Any):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import (ConstantKernel, Matern,
                                                  WhiteKernel)
    kernel = (ConstantKernel(1.0, (1e-3, 1e3))
              * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e3),
                       nu=kw.pop("nu", 2.5))
              + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-8, 1e1)))
    return GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, random_state=seed,
        n_restarts_optimizer=kw.pop("n_restarts_optimizer", 1), **kw)


def _knn(seed: int, **kw: Any):
    from sklearn.neighbors import KNeighborsRegressor
    return KNeighborsRegressor(n_neighbors=kw.pop("n_neighbors", 10),
                               weights=kw.pop("weights", "distance"), **kw)


def _random_forest(seed: int, **kw: Any):
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(
        n_estimators=kw.pop("n_estimators", 400),
        max_depth=kw.pop("max_depth", None),
        min_samples_leaf=kw.pop("min_samples_leaf", 2),
        max_features=kw.pop("max_features", 0.5),
        n_jobs=kw.pop("n_jobs", -1), random_state=seed, **kw)


def _extra_trees(seed: int, **kw: Any):
    from sklearn.ensemble import ExtraTreesRegressor
    return ExtraTreesRegressor(
        n_estimators=kw.pop("n_estimators", 400),
        min_samples_leaf=kw.pop("min_samples_leaf", 2),
        max_features=kw.pop("max_features", 0.6),
        n_jobs=kw.pop("n_jobs", -1), random_state=seed, **kw)


def _gbr(seed: int, **kw: Any):
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(
        max_iter=kw.pop("max_iter", 500),
        learning_rate=kw.pop("learning_rate", 0.06),
        max_depth=kw.pop("max_depth", None),
        min_samples_leaf=kw.pop("min_samples_leaf", 20),
        l2_regularization=kw.pop("l2_regularization", 0.0),
        early_stopping=kw.pop("early_stopping", False),
        random_state=seed, **kw)


def _xgboost(seed: int, **kw: Any):
    from xgboost import XGBRegressor
    return XGBRegressor(
        n_estimators=kw.pop("n_estimators", 700),
        learning_rate=kw.pop("learning_rate", 0.05),
        max_depth=kw.pop("max_depth", 8),
        subsample=kw.pop("subsample", 0.85),
        colsample_bytree=kw.pop("colsample_bytree", 0.85),
        min_child_weight=kw.pop("min_child_weight", 3),
        reg_lambda=kw.pop("reg_lambda", 1.0),
        reg_alpha=kw.pop("reg_alpha", 0.0),
        tree_method=kw.pop("tree_method", "hist"),
        n_jobs=kw.pop("n_jobs", -1), random_state=seed,
        verbosity=0, **kw)


def _lightgbm(seed: int, **kw: Any):
    from lightgbm import LGBMRegressor
    return LGBMRegressor(
        n_estimators=kw.pop("n_estimators", 900),
        learning_rate=kw.pop("learning_rate", 0.05),
        num_leaves=kw.pop("num_leaves", 63),
        max_depth=kw.pop("max_depth", -1),
        subsample=kw.pop("subsample", 0.85),
        subsample_freq=kw.pop("subsample_freq", 1),
        colsample_bytree=kw.pop("colsample_bytree", 0.85),
        min_child_samples=kw.pop("min_child_samples", 20),
        reg_lambda=kw.pop("reg_lambda", 0.0),
        n_jobs=kw.pop("n_jobs", -1), random_state=seed,
        verbose=-1, **kw)


def _catboost(seed: int, **kw: Any):
    from catboost import CatBoostRegressor
    return CatBoostRegressor(
        iterations=kw.pop("iterations", 900),
        learning_rate=kw.pop("learning_rate", 0.06),
        depth=kw.pop("depth", 8),
        l2_leaf_reg=kw.pop("l2_leaf_reg", 3.0),
        random_seed=seed, verbose=0, allow_writing_files=False, **kw)


def _mlp(seed: int, **kw: Any):
    from sklearn.neural_network import MLPRegressor
    return MLPRegressor(
        hidden_layer_sizes=kw.pop("hidden_layer_sizes", (128, 64)),
        activation=kw.pop("activation", "relu"),
        alpha=kw.pop("alpha", 1e-4),
        learning_rate_init=kw.pop("learning_rate_init", 1e-3),
        batch_size=kw.pop("batch_size", 256),
        max_iter=kw.pop("max_iter", 400),
        early_stopping=kw.pop("early_stopping", True),
        n_iter_no_change=kw.pop("n_iter_no_change", 15),
        random_state=seed, **kw)


def _mlp_deep(seed: int, **kw: Any):
    kw.setdefault("hidden_layer_sizes", (256, 128, 64))
    kw.setdefault("max_iter", 500)
    return _mlp(seed, **kw)


#: name -> (factory, family, human label)
MODEL_REGISTRY: dict[str, tuple[Callable[..., Any], str, str]] = {
    "ridge":         (_ridge, "linear", "Ridge regression"),
    "elasticnet":    (_elasticnet, "linear", "Elastic net"),
    "poly2_ridge":   (_poly2_ridge, "linear", "Quadratic polynomial ridge"),
    "svr_rbf":       (_svr_rbf, "kernel", "Support-vector regression (RBF)"),
    "gpr":           (_gpr, "kernel", "Gaussian-process regression (Matern 5/2)"),
    "knn":           (_knn, "neighbour", "k-nearest neighbours"),
    "random_forest": (_random_forest, "ensemble", "Random forest"),
    "extra_trees":   (_extra_trees, "ensemble", "Extremely randomised trees"),
    "hist_gbr":      (_gbr, "ensemble", "Histogram gradient boosting"),
    "xgboost":       (_xgboost, "boosting", "XGBoost"),
    "lightgbm":      (_lightgbm, "boosting", "LightGBM"),
    "catboost":      (_catboost, "boosting", "CatBoost"),
    "mlp":           (_mlp, "neural", "Artificial neural network (128-64)"),
    "mlp_deep":      (_mlp_deep, "neural", "Artificial neural network (256-128-64)"),
}


def model_families() -> dict[str, str]:
    return {name: fam for name, (_f, fam, _l) in MODEL_REGISTRY.items()}


def available_models() -> list[str]:
    """Registry entries whose backing library can actually be imported."""
    ok: list[str] = []
    for name in MODEL_REGISTRY:
        try:
            build_model(name, seed=0)
            ok.append(name)
        except ImportError as exc:
            log.warning("Model %r unavailable: %s", name, exc)
        except Exception as exc:                            # pragma: no cover
            log.warning("Model %r failed to construct: %s", name, exc)
    return ok


def build_model(name: str, seed: int = 0, **params: Any):
    """Instantiate a registry model with optional hyper-parameter overrides."""
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown surrogate {name!r}; available: {sorted(MODEL_REGISTRY)}")
    factory, _family, _label = MODEL_REGISTRY[name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return factory(seed, **params)


# --------------------------------------------------------------------------
def build_preprocessor(numeric_features: list[str],
                       categorical_features: list[str],
                       scale: bool) -> Any:
    """Column transformer: optional standardisation + one-hot for categoricals."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    num_steps: list[tuple[str, Any]] = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    numeric_pipe = Pipeline(num_steps)

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:                                       # sklearn < 1.2
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                         ("onehot", ohe)])

    transformers = []
    if numeric_features:
        transformers.append(("num", numeric_pipe, numeric_features))
    if categorical_features:
        transformers.append(("cat", cat_pipe, categorical_features))
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)
