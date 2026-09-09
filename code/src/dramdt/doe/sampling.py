"""Space-filling designs over the unit hypercube.

Latin Hypercube Sampling (LHS) is the default sampler for the main campaign: it
guarantees one sample per stratum in every one-dimensional projection, which is
what makes a 50k-point design informative for 15 design variables without the
clustering a pseudo-random design suffers from.

Quality is *measured*, not assumed -- :func:`discrepancy` and
:func:`min_pairwise_distance` are recorded into the dataset manifest and
reported in the supplementary material.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np
from scipy.stats import qmc

from ..logging_utils import get_logger
from ..seeds import derive_seed

__all__ = ["DesignMatrix", "generate_design", "lhs", "sobol", "random_uniform",
           "grid", "discrepancy", "min_pairwise_distance"]

log = get_logger(__name__)

# Above this many points scipy's iterative LHS optimisation costs more than it
# is worth; a scrambled LHS is already excellent and the switch is logged.
_OPTIMISE_LIMIT = 20000


@dataclass
class DesignMatrix:
    """A unit-hypercube design plus the diagnostics describing its quality."""

    points: np.ndarray                      # (n, d) in [0, 1]
    sampler: str
    seed: int
    diagnostics: dict[str, float] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.points.shape[0])

    @property
    def d(self) -> int:
        return int(self.points.shape[1])

    def describe(self) -> dict[str, Any]:
        return {"sampler": self.sampler, "n_samples": self.n,
                "n_dimensions": self.d, "seed": self.seed, **self.diagnostics}


# --------------------------------------------------------------------------
# quality metrics
# --------------------------------------------------------------------------
def discrepancy(points: np.ndarray, method: str = "CD",
                max_points: int = 4000, seed: int = 0) -> float:
    """Centred (or wrap-around) L2 discrepancy -- lower is more uniform.

    The exact computation is O(n^2 d), which is minutes for a 50k design, so
    above ``max_points`` the statistic is estimated on a random subsample.  The
    subsample size is reported alongside the value (``discrepancy_n_used``)
    because the discrepancy of a uniform design shrinks with n and the numbers
    are therefore only comparable at equal n.
    """
    p = np.asarray(points, dtype=float)
    if p.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        p = p[rng.choice(p.shape[0], max_points, replace=False)]
    try:
        return float(qmc.discrepancy(p, method=method, workers=-1))
    except TypeError:                                       # older scipy
        try:
            return float(qmc.discrepancy(p, method=method))
        except Exception:                                   # pragma: no cover
            return float("nan")
    except Exception:                                       # pragma: no cover
        return float("nan")


def min_pairwise_distance(points: np.ndarray, max_points: int = 4000,
                          seed: int = 0) -> float:
    """Minimum pairwise Euclidean distance (the maximin criterion).

    Sub-sampled above ``max_points`` because the exact computation is O(n^2);
    the sub-sample size is reported alongside the value.
    """
    from scipy.spatial.distance import pdist
    p = np.asarray(points, dtype=float)
    if p.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        p = p[rng.choice(p.shape[0], max_points, replace=False)]
    if p.shape[0] < 2:
        return float("nan")
    return float(pdist(p).min())


# --------------------------------------------------------------------------
# samplers
# --------------------------------------------------------------------------
def lhs(n: int, d: int, seed: int,
        criterion: Literal["classic", "centered", "maximin"] = "maximin",
        iterations: int = 20) -> np.ndarray:
    """Latin Hypercube design in [0, 1]^d."""
    optimization = None
    if criterion == "maximin":
        if n <= _OPTIMISE_LIMIT:
            optimization = "random-cd"
        else:
            log.info("LHS: n=%d exceeds the %d-point optimisation limit; using a "
                     "scrambled (non-iterated) Latin Hypercube.", n, _OPTIMISE_LIMIT)
    sampler = qmc.LatinHypercube(
        d=d, seed=seed, optimization=optimization,
        **({"scramble": True} if criterion != "centered" else {"scramble": False}))
    if optimization is not None:
        try:
            sampler = qmc.LatinHypercube(d=d, seed=seed, optimization=optimization)
        except TypeError:                                   # pragma: no cover
            sampler = qmc.LatinHypercube(d=d, seed=seed)
    pts = sampler.random(n)
    if criterion == "centered":
        # snap every sample to its stratum centre
        strata = np.floor(pts * n).astype(int)
        pts = (strata + 0.5) / n
    return np.clip(pts, 0.0, 1.0)


def sobol(n: int, d: int, seed: int, scramble: bool = True) -> np.ndarray:
    """Scrambled Sobol' sequence.  ``n`` is rounded up to a power of two."""
    m = max(int(math.ceil(math.log2(max(n, 2)))), 1)
    pts = qmc.Sobol(d=d, scramble=scramble, seed=seed).random_base2(m=m)
    return np.clip(pts[:n], 0.0, 1.0)


def random_uniform(n: int, d: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).random((n, d))


def grid(n: int, d: int, seed: int = 0) -> np.ndarray:
    """Full-factorial grid with as many levels per axis as fit inside ``n``."""
    levels = max(int(round(n ** (1.0 / d))), 2)
    axes = [(np.arange(levels) + 0.5) / levels] * d
    mesh = np.meshgrid(*axes, indexing="ij")
    pts = np.stack([m.ravel() for m in mesh], axis=1)
    return pts[:n]


# --------------------------------------------------------------------------
def generate_design(n: int, d: int, sampler: str = "lhs",
                    seed: int = 0, label: str = "design",
                    criterion: str = "maximin", iterations: int = 20,
                    compute_diagnostics: bool = True) -> DesignMatrix:
    """Build a design matrix of ``n`` points in ``d`` dimensions."""
    if n < 1 or d < 1:
        raise ValueError("n and d must be positive")
    child = derive_seed(seed, label)

    sampler = sampler.lower()
    if sampler == "lhs":
        pts = lhs(n, d, child, criterion=criterion, iterations=iterations)  # type: ignore[arg-type]
    elif sampler == "sobol":
        pts = sobol(n, d, child)
    elif sampler == "random":
        pts = random_uniform(n, d, child)
    elif sampler == "grid":
        pts = grid(n, d, child)
    else:
        raise ValueError(f"Unknown sampler {sampler!r}")

    diag: dict[str, float] = {}
    if compute_diagnostics:
        n_used = min(n, 4000)
        diag["centered_l2_discrepancy"] = discrepancy(pts, "CD", seed=child)
        diag["discrepancy_n_used"] = float(n_used)
        diag["min_pairwise_distance"] = min_pairwise_distance(pts, seed=child)
        diag["min_distance_n_used"] = float(n_used)
        diag["mean_coordinate"] = float(pts.mean())
        diag["min_coordinate"] = float(pts.min())
        diag["max_coordinate"] = float(pts.max())

    log.info("Design: %s n=%d d=%d seed=%d  CD-discrepancy=%.6g  min-dist=%.4g",
             sampler, n, d, child, diag.get("centered_l2_discrepancy", float("nan")),
             diag.get("min_pairwise_distance", float("nan")))
    return DesignMatrix(points=pts, sampler=sampler, seed=child, diagnostics=diag)
