"""Regression scoring for the surrogate comparison.

All scores are computed on the *original* response scale unless a target was
declared with a transform, in which case they are reported in both spaces --
an R^2 of 0.99 in log space can still be a factor-of-two error in the physical
units the designer cares about.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = ["regression_metrics", "bootstrap_ci", "METRIC_NAMES",
           "smape", "mape", "nrmse"]

METRIC_NAMES = ("r2", "rmse", "mae", "medae", "mape", "smape", "nrmse", "max_error",
                "pearson_r", "spearman_rho", "explained_variance")


def _clean(y_true: Sequence[float], y_pred: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(y_pred, dtype=float).ravel()
    m = np.isfinite(t) & np.isfinite(p)
    return t[m], p[m]


def mape(y_true: Sequence[float], y_pred: Sequence[float], eps: float = 1e-12) -> float:
    """Mean absolute percentage error, guarded against near-zero truths."""
    t, p = _clean(y_true, y_pred)
    m = np.abs(t) > eps
    if not m.any():
        return float("nan")
    return float(np.mean(np.abs((t[m] - p[m]) / t[m])) * 100.0)


def smape(y_true: Sequence[float], y_pred: Sequence[float], eps: float = 1e-12) -> float:
    """Symmetric MAPE -- bounded at 200 %, safe when the truth crosses zero."""
    t, p = _clean(y_true, y_pred)
    denom = (np.abs(t) + np.abs(p)) / 2.0
    m = denom > eps
    if not m.any():
        return float("nan")
    return float(np.mean(np.abs(t[m] - p[m]) / denom[m]) * 100.0)


def nrmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """RMSE normalised by the range of the truth."""
    t, p = _clean(y_true, y_pred)
    if t.size < 2:
        return float("nan")
    rng = float(np.max(t) - np.min(t))
    if rng <= 0:
        return float("nan")
    return float(np.sqrt(np.mean((t - p) ** 2)) / rng)


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, float]:
    """Full metric bundle.  Returns NaNs rather than raising on degenerate input."""
    from scipy import stats
    from sklearn.metrics import (explained_variance_score, max_error,
                                 mean_absolute_error, median_absolute_error,
                                 mean_squared_error, r2_score)

    t, p = _clean(y_true, y_pred)
    n = t.size
    if n < 2:
        return {k: float("nan") for k in METRIC_NAMES} | {"n": float(n)}

    out: dict[str, float] = {"n": float(n)}
    out["r2"] = float(r2_score(t, p))
    out["rmse"] = float(math.sqrt(mean_squared_error(t, p)))
    out["mae"] = float(mean_absolute_error(t, p))
    out["medae"] = float(median_absolute_error(t, p))
    out["mape"] = mape(t, p)
    out["smape"] = smape(t, p)
    out["nrmse"] = nrmse(t, p)
    out["max_error"] = float(max_error(t, p))
    out["explained_variance"] = float(explained_variance_score(t, p))
    try:
        out["pearson_r"] = float(stats.pearsonr(t, p)[0])
    except Exception:
        out["pearson_r"] = float("nan")
    try:
        out["spearman_rho"] = float(stats.spearmanr(t, p)[0])
    except Exception:
        out["spearman_rho"] = float("nan")
    return out


def bootstrap_ci(y_true: Sequence[float], y_pred: Sequence[float],
                 metric: str = "r2", n_boot: int = 1000,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Percentile bootstrap confidence interval for one metric.

    Returns ``(point_estimate, lower, upper)``.
    """
    t, p = _clean(y_true, y_pred)
    if t.size < 5:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    point = regression_metrics(t, p).get(metric, float("nan"))
    draws = np.empty(n_boot, dtype=float)
    n = t.size
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        draws[b] = regression_metrics(t[idx], p[idx]).get(metric, float("nan"))
    draws = draws[np.isfinite(draws)]
    if draws.size < 10:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)
