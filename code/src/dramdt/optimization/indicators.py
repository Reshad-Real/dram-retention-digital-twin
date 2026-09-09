"""Multi-objective performance indicators.

All indicators operate on *minimisation* objective matrices; the problem class
already flips maximised objectives, so nothing here needs to know the sense of
each objective.

Because the true Pareto front of this problem is unknown, IGD/IGD+ are computed
against a **reference front** built as the non-dominated subset of the union of
every algorithm's output across every repetition -- the standard protocol for
real-world benchmarks.  Hypervolume uses a reference point derived from the
worst value observed for each objective in that same union, inflated by a
configurable margin so that no algorithm sits exactly on the boundary.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

__all__ = ["non_dominated", "hypervolume", "igd", "igd_plus", "spacing",
           "spread", "additive_epsilon", "indicator_bundle",
           "build_reference_front", "build_reference_point"]


# --------------------------------------------------------------------------
def non_dominated(F: np.ndarray) -> np.ndarray:
    """Boolean mask of the non-dominated rows of a minimisation matrix."""
    F = np.asarray(F, dtype=float)
    n = F.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)
    keep = np.ones(n, dtype=bool)
    order = np.argsort(F[:, 0], kind="stable")
    for i in order:
        if not keep[i]:
            continue
        # Point i dominates j when i <= j in every objective and i < j in at
        # least one.  Those j are the ones to eliminate -- note the direction:
        # testing `F <= F[i]` instead would select i's *dominators* and delete
        # exactly the wrong half of the front.
        dominated_by_i = np.all(F >= F[i], axis=1) & np.any(F > F[i], axis=1)
        keep[dominated_by_i] = False
    return keep


def build_reference_front(fronts: Sequence[np.ndarray], max_points: int = 5000,
                          seed: int = 0) -> np.ndarray:
    """Non-dominated union of every supplied front."""
    stacked = [np.asarray(f, dtype=float) for f in fronts if f is not None and len(f)]
    if not stacked:
        return np.empty((0, 0))
    U = np.vstack(stacked)
    U = U[np.all(np.isfinite(U), axis=1)]
    if len(U) == 0:
        return np.empty((0, U.shape[1] if U.ndim > 1 else 0))
    if len(U) > max_points:
        rng = np.random.default_rng(seed)
        U = U[rng.choice(len(U), max_points, replace=False)]
    return U[non_dominated(U)]


def build_reference_point(fronts: Sequence[np.ndarray], margin: float = 0.1
                          ) -> np.ndarray:
    """Nadir-based hypervolume reference point, inflated by ``margin``."""
    stacked = [np.asarray(f, dtype=float) for f in fronts if f is not None and len(f)]
    if not stacked:
        return np.empty(0)
    U = np.vstack(stacked)
    U = U[np.all(np.isfinite(U), axis=1)]
    worst = U.max(axis=0)
    best = U.min(axis=0)
    span = np.where(worst - best > 0, worst - best, 1.0)
    return worst + margin * span


# --------------------------------------------------------------------------
def hypervolume(F: np.ndarray, ref_point: Sequence[float]) -> float:
    """Hypervolume dominated by ``F`` with respect to ``ref_point``."""
    F = np.asarray(F, dtype=float)
    ref = np.asarray(ref_point, dtype=float)
    if F.size == 0 or ref.size == 0:
        return float("nan")
    F = F[np.all(np.isfinite(F), axis=1)]
    F = F[np.all(F <= ref, axis=1)]              # points outside contribute nothing
    if len(F) == 0:
        return 0.0
    try:
        from pymoo.indicators.hv import HV
        return float(HV(ref_point=ref)(F))
    except Exception:                                       # pragma: no cover
        return float("nan")


def igd(F: np.ndarray, reference: np.ndarray) -> float:
    """Inverted generational distance (lower is better)."""
    F, R = np.asarray(F, float), np.asarray(reference, float)
    if F.size == 0 or R.size == 0:
        return float("nan")
    try:
        from pymoo.indicators.igd import IGD
        return float(IGD(R)(F))
    except Exception:                                       # pragma: no cover
        d = np.sqrt(((R[:, None, :] - F[None, :, :]) ** 2).sum(-1)).min(axis=1)
        return float(d.mean())


def igd_plus(F: np.ndarray, reference: np.ndarray) -> float:
    """IGD+ -- the Pareto-compliant variant of IGD."""
    F, R = np.asarray(F, float), np.asarray(reference, float)
    if F.size == 0 or R.size == 0:
        return float("nan")
    try:
        from pymoo.indicators.igd_plus import IGDPlus
        return float(IGDPlus(R)(F))
    except Exception:                                       # pragma: no cover
        diff = np.maximum(F[None, :, :] - R[:, None, :], 0.0)
        return float(np.sqrt((diff ** 2).sum(-1)).min(axis=1).mean())


def spacing(F: np.ndarray) -> float:
    """Schott's spacing metric: uniformity of the solution distribution."""
    F = np.asarray(F, dtype=float)
    F = F[np.all(np.isfinite(F), axis=1)]
    if len(F) < 3:
        return float("nan")
    d = np.abs(F[:, None, :] - F[None, :, :]).sum(-1)
    np.fill_diagonal(d, np.inf)
    di = d.min(axis=1)
    return float(np.sqrt(np.sum((di.mean() - di) ** 2) / (len(di) - 1)))


def spread(F: np.ndarray) -> float:
    """Maximum spread (diagonal of the front's bounding box)."""
    F = np.asarray(F, dtype=float)
    F = F[np.all(np.isfinite(F), axis=1)]
    if len(F) < 2:
        return float("nan")
    return float(np.sqrt(((F.max(axis=0) - F.min(axis=0)) ** 2).sum()))


def additive_epsilon(F: np.ndarray, reference: np.ndarray) -> float:
    """Additive epsilon indicator relative to a reference front."""
    F, R = np.asarray(F, float), np.asarray(reference, float)
    if F.size == 0 or R.size == 0:
        return float("nan")
    # smallest eps such that every reference point is eps-dominated by some F
    eps = (R[:, None, :] - F[None, :, :]).max(axis=2).min(axis=1)
    return float(eps.max())


# --------------------------------------------------------------------------
def indicator_bundle(F: np.ndarray, reference_front: np.ndarray,
                     ref_point: Sequence[float]) -> dict[str, float]:
    """All indicators for one front, in one call."""
    F = np.asarray(F, dtype=float)
    F = F[np.all(np.isfinite(F), axis=1)] if F.ndim == 2 else F
    return {
        "n_solutions": float(len(F)),
        "hypervolume": hypervolume(F, ref_point),
        "igd": igd(F, reference_front),
        "igd_plus": igd_plus(F, reference_front),
        "spacing": spacing(F),
        "spread": spread(F),
        "epsilon": additive_epsilon(F, reference_front),
    }
