"""Non-parametric statistical validation.

Protocol (Demsar's recommendations for comparing algorithms over multiple
problems/repetitions, adapted to repeated runs of one problem):

1. **Normality** is tested (Shapiro-Wilk) but *not* relied upon -- indicator
   distributions from evolutionary runs are routinely non-normal, so the
   headline tests are rank based.
2. **Omnibus**: the Friedman test asks whether any algorithm differs.
3. **Post-hoc**: if the omnibus test rejects, Nemenyi (all-pairs, with a
   critical-difference diagram) or Wilcoxon signed-rank with Holm correction
   (pairwise against a control) localises the difference.
4. **Effect size**: Cliff's delta and the Vargha-Delaney A12 statistic report
   *how much* better, because a significant p-value on 11 runs can still be a
   negligible difference.
5. **Uncertainty**: BCa-free percentile bootstrap intervals on the mean.

Nothing here decides significance silently: every function returns the
statistic, the p-value, the effect size and the decision at the configured
alpha, and the report tabulates all of them.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ..logging_utils import get_logger

__all__ = [
    "StatisticalReport", "descriptive_statistics", "normality", "friedman_test",
    "nemenyi_posthoc", "wilcoxon_holm", "cliffs_delta", "vargha_delaney_a12",
    "bootstrap_mean_ci", "critical_difference", "compare_algorithms",
    "kruskal_wallis", "holm_correction",
]

log = get_logger(__name__)

#: Studentized-range critical values divided by sqrt(2), for the Nemenyi test
#: at alpha = 0.05, indexed by the number of compared algorithms (k = 2..20).
_NEMENYI_Q05 = {
    2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031,
    9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354,
    15: 3.391, 16: 3.426, 17: 3.458, 18: 3.489, 19: 3.517, 20: 3.544,
}
_NEMENYI_Q10 = {
    2: 1.645, 3: 2.052, 4: 2.291, 5: 2.459, 6: 2.589, 7: 2.693, 8: 2.780,
    9: 2.855, 10: 2.920, 11: 2.978, 12: 3.030, 13: 3.077, 14: 3.120,
    15: 3.159, 16: 3.196, 17: 3.230, 18: 3.261, 19: 3.291, 20: 3.319,
}


@dataclass
class StatisticalReport:
    """Everything the statistical stage produced for one indicator."""

    indicator: str
    higher_is_better: bool
    descriptives: pd.DataFrame = field(default_factory=pd.DataFrame)
    normality: pd.DataFrame = field(default_factory=pd.DataFrame)
    omnibus: dict[str, Any] = field(default_factory=dict)
    posthoc: pd.DataFrame = field(default_factory=pd.DataFrame)
    pairwise: pd.DataFrame = field(default_factory=pd.DataFrame)
    ranks: pd.DataFrame = field(default_factory=pd.DataFrame)
    critical_difference: float = float("nan")
    alpha: float = 0.05
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
def descriptive_statistics(groups: Mapping[str, Sequence[float]]) -> pd.DataFrame:
    rows = []
    for name, values in groups.items():
        v = np.asarray(values, dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            rows.append({"group": name, "n": 0})
            continue
        lo, hi = bootstrap_mean_ci(v)
        rows.append({
            "group": name, "n": int(v.size),
            "mean": float(v.mean()), "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
            "median": float(np.median(v)),
            "iqr": float(np.percentile(v, 75) - np.percentile(v, 25)),
            "min": float(v.min()), "max": float(v.max()),
            "mean_ci_low": lo, "mean_ci_high": hi,
        })
    return pd.DataFrame(rows)


def normality(groups: Mapping[str, Sequence[float]], alpha: float = 0.05) -> pd.DataFrame:
    from scipy import stats
    rows = []
    for name, values in groups.items():
        v = np.asarray(values, dtype=float)
        v = v[np.isfinite(v)]
        if v.size < 3 or np.allclose(v, v[0]):
            rows.append({"group": name, "n": int(v.size), "statistic": float("nan"),
                         "p_value": float("nan"), "normal_at_alpha": ""})
            continue
        s, p = stats.shapiro(v)
        rows.append({"group": name, "n": int(v.size), "statistic": float(s),
                     "p_value": float(p), "normal_at_alpha": bool(p > alpha)})
    return pd.DataFrame(rows)


def bootstrap_mean_ci(values: Sequence[float], n_boot: int = 10000,
                      alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(v, (n_boot, v.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


# --------------------------------------------------------------------------
def friedman_test(matrix: np.ndarray) -> dict[str, Any]:
    """Friedman test over a ``(n_blocks, k_treatments)`` matrix."""
    from scipy import stats
    M = np.asarray(matrix, dtype=float)
    if M.ndim != 2 or M.shape[1] < 3:
        return {"test": "friedman", "statistic": float("nan"), "p_value": float("nan"),
                "note": "Friedman requires at least 3 treatments"}
    keep = np.all(np.isfinite(M), axis=1)
    M = M[keep]
    if M.shape[0] < 3:
        return {"test": "friedman", "statistic": float("nan"), "p_value": float("nan"),
                "note": "fewer than 3 complete blocks"}
    stat, p = stats.friedmanchisquare(*[M[:, j] for j in range(M.shape[1])])
    n, k = M.shape
    # Kendall's W as the effect size for the omnibus test
    w = float(stat / (n * (k - 1))) if n * (k - 1) > 0 else float("nan")
    return {"test": "friedman", "statistic": float(stat), "p_value": float(p),
            "n_blocks": int(n), "k_treatments": int(k), "kendalls_w": w}


def kruskal_wallis(groups: Mapping[str, Sequence[float]]) -> dict[str, Any]:
    from scipy import stats
    arrays = [np.asarray(v, float)[np.isfinite(np.asarray(v, float))]
              for v in groups.values()]
    arrays = [a for a in arrays if a.size > 0]
    if len(arrays) < 2:
        return {"test": "kruskal", "statistic": float("nan"), "p_value": float("nan")}
    stat, p = stats.kruskal(*arrays)
    return {"test": "kruskal", "statistic": float(stat), "p_value": float(p)}


def critical_difference(k: int, n: int, alpha: float = 0.05) -> float:
    """Nemenyi critical difference for ``k`` treatments over ``n`` blocks."""
    table = _NEMENYI_Q05 if abs(alpha - 0.05) < 1e-9 else _NEMENYI_Q10
    q = table.get(int(k))
    if q is None or n <= 0:
        return float("nan")
    return float(q * math.sqrt(k * (k + 1) / (6.0 * n)))


def nemenyi_posthoc(matrix: np.ndarray, labels: Sequence[str],
                    alpha: float = 0.05) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """All-pairs Nemenyi test.

    Returns ``(pairwise_frame, mean_rank_frame, critical_difference)``.
    Ranks are assigned per block with rank 1 = best, so a *lower* mean rank is
    better regardless of the indicator's own direction (the caller is expected
    to have oriented the matrix so that smaller is better).
    """
    from scipy import stats

    M = np.asarray(matrix, dtype=float)
    keep = np.all(np.isfinite(M), axis=1)
    M = M[keep]
    n, k = M.shape
    if n < 2 or k < 2:
        return pd.DataFrame(), pd.DataFrame(), float("nan")

    ranks = np.apply_along_axis(stats.rankdata, 1, M)
    mean_ranks = ranks.mean(axis=0)
    cd = critical_difference(k, n, alpha)
    se = math.sqrt(k * (k + 1) / (6.0 * n))

    rows = []
    for i, j in itertools.combinations(range(k), 2):
        diff = abs(mean_ranks[i] - mean_ranks[j])
        z = diff / se if se > 0 else float("nan")
        # two-sided p from the studentized range approximation
        p = float(min(1.0, 2.0 * (1.0 - stats.norm.cdf(abs(z) / math.sqrt(2.0))) * k * (k - 1) / 2))
        rows.append({
            "group_1": labels[i], "group_2": labels[j],
            "mean_rank_1": float(mean_ranks[i]), "mean_rank_2": float(mean_ranks[j]),
            "rank_difference": float(diff),
            "critical_difference": cd,
            "significant": bool(diff > cd) if np.isfinite(cd) else "",
            "z": float(z), "p_value_adjusted": p,
        })

    rank_frame = (pd.DataFrame({"group": list(labels), "mean_rank": mean_ranks})
                  .sort_values("mean_rank").reset_index(drop=True))
    rank_frame["rank_position"] = np.arange(1, len(rank_frame) + 1)
    return pd.DataFrame(rows), rank_frame, cd


def holm_correction(p_values: Sequence[float], alpha: float = 0.05
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Holm step-down correction.  Returns ``(adjusted_p, reject)``."""
    p = np.asarray(p_values, dtype=float)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj, adj <= alpha


def wilcoxon_holm(groups: Mapping[str, Sequence[float]], alpha: float = 0.05,
                  control: str | None = None) -> pd.DataFrame:
    """Pairwise Wilcoxon signed-rank tests with Holm-corrected p-values."""
    from scipy import stats

    names = list(groups)
    pairs = ([(control, n) for n in names if n != control] if control
             else list(itertools.combinations(names, 2)))
    rows, raw_p = [], []
    for a, b in pairs:
        x = np.asarray(groups[a], dtype=float)
        y = np.asarray(groups[b], dtype=float)
        n = min(x.size, y.size)
        x, y = x[:n], y[:n]
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        if x.size < 3 or np.allclose(x, y):
            stat, p = float("nan"), 1.0
        else:
            try:
                stat, p = stats.wilcoxon(x, y)
            except ValueError:
                stat, p = float("nan"), 1.0
        d = cliffs_delta(x, y)
        rows.append({
            "group_1": a, "group_2": b, "n_pairs": int(x.size),
            "statistic": float(stat), "p_value": float(p),
            "median_1": float(np.median(x)) if x.size else float("nan"),
            "median_2": float(np.median(y)) if y.size else float("nan"),
            "cliffs_delta": d["delta"], "effect_magnitude": d["magnitude"],
            "a12": vargha_delaney_a12(x, y),
        })
        raw_p.append(p)

    frame = pd.DataFrame(rows)
    if not frame.empty:
        adj, reject = holm_correction(raw_p, alpha)
        frame["p_value_holm"] = adj
        frame["significant"] = reject
    return frame


# --------------------------------------------------------------------------
def cliffs_delta(x: Sequence[float], y: Sequence[float]) -> dict[str, Any]:
    """Cliff's delta with the conventional magnitude thresholds."""
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return {"delta": float("nan"), "magnitude": "undefined"}
    gt = int((a[:, None] > b[None, :]).sum())
    lt = int((a[:, None] < b[None, :]).sum())
    delta = (gt - lt) / (a.size * b.size)
    ad = abs(delta)
    magnitude = ("negligible" if ad < 0.147 else
                 "small" if ad < 0.33 else
                 "medium" if ad < 0.474 else "large")
    return {"delta": float(delta), "magnitude": magnitude}


def vargha_delaney_a12(x: Sequence[float], y: Sequence[float]) -> float:
    """Probability that a random draw from ``x`` exceeds one from ``y``."""
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    gt = (a[:, None] > b[None, :]).sum()
    eq = (a[:, None] == b[None, :]).sum()
    return float((gt + 0.5 * eq) / (a.size * b.size))


# --------------------------------------------------------------------------
def compare_algorithms(table: pd.DataFrame,
                       indicator: str,
                       group_col: str = "algorithm",
                       block_col: str = "run",
                       higher_is_better: bool = True,
                       alpha: float = 0.05,
                       posthoc: str = "nemenyi",
                       control: str | None = None) -> StatisticalReport:
    """Full statistical comparison of algorithms on one indicator.

    ``table`` is long format: one row per (algorithm, run) with the indicator
    value in column ``indicator``.
    """
    report = StatisticalReport(indicator=indicator,
                               higher_is_better=higher_is_better, alpha=alpha)

    pivot = table.pivot_table(index=block_col, columns=group_col,
                              values=indicator, aggfunc="mean")
    pivot = pivot.dropna(axis=0, how="any")
    labels = list(pivot.columns)
    if pivot.empty or len(labels) < 2:
        report.notes.append("insufficient data for a comparison")
        return report

    groups = {name: pivot[name].to_numpy(dtype=float) for name in labels}
    report.descriptives = descriptive_statistics(groups)
    report.normality = normality(groups, alpha)

    M = pivot.to_numpy(dtype=float)
    # Orient so that smaller == better, which is what the rank machinery expects.
    M_ranked = -M if higher_is_better else M

    report.omnibus = friedman_test(M_ranked)
    if len(labels) == 2:
        report.omnibus = {"test": "wilcoxon (2 groups)",
                          **wilcoxon_holm(groups, alpha).iloc[0].to_dict()}

    if posthoc == "nemenyi" and len(labels) >= 3:
        pw, ranks, cd = nemenyi_posthoc(M_ranked, labels, alpha)
        report.posthoc, report.ranks, report.critical_difference = pw, ranks, cd
    report.pairwise = wilcoxon_holm(groups, alpha, control=control)

    p = report.omnibus.get("p_value", float("nan"))
    if not np.isfinite(p):
        # A NaN p-value means the test could not be run (too few blocks or
        # treatments) -- reporting that as "no significant difference" would be
        # an unsupported claim.
        verdict = f"NOT TESTABLE ({report.omnibus.get('note', 'insufficient data')})"
        report.notes.append(f"omnibus test not applicable: {verdict}")
    elif p < alpha:
        verdict = "differences detected"
    else:
        verdict = "no significant difference"
    log.info("[%s] %s: omnibus p=%.4g -> %s", indicator,
             report.omnibus.get("test", "?"), p, verdict)
    return report
