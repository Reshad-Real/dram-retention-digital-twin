"""Every figure in the study, one function per figure.

Each function takes already-computed data plus an output stem and returns the
list of files written (PNG + PDF + SVG at 300 dpi).  No function performs
analysis: figures render results, they never generate them.

Design rules applied throughout (see :mod:`dramdt.viz.style`):
single measure per axis (never a second y-scale), sequential single-hue ramps
for magnitude, a diverging ramp with a neutral midpoint only for signed
quantities, categorical hues assigned in fixed order and always paired with a
distinct marker/line style so identity survives greyscale printing.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from ..logging_utils import get_logger
from .palette import CATEGORICAL, DIVERGING_PAIR, REFERENCE_INK, SEQUENTIAL_HUE, STATUS_COLORS
from .style import (HATCHES, MARKERS, annotate_panel, figure_size, legend_below,
                    save_figure, series_style)

log = get_logger(__name__)

__all__ = [
    "sequential_cmap", "diverging_cmap",
    "fig_design_space_coverage", "fig_response_distributions",
    "fig_correlation_matrix", "fig_retention_validation",
    "fig_surrogate_comparison", "fig_predicted_vs_actual", "fig_residuals",
    "fig_cv_boxplots", "fig_critical_difference", "fig_shap_importance",
    "fig_shap_beeswarm", "fig_shap_interactions", "fig_partial_dependence",
    "fig_ale", "fig_pareto_matrix", "fig_parallel_coordinates",
    "fig_hypervolume_convergence", "fig_indicator_boxplots",
    "fig_corner_heatmap", "fig_monte_carlo", "fig_sobol_indices",
    "fig_assist_comparison", "fig_twin_speedup", "fig_learning_curve",
    "fig_tradeoff_scatter",
]


def _plt():
    import matplotlib.pyplot as plt
    return plt


def sequential_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("dramdt_seq", list(SEQUENTIAL_HUE))


def diverging_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("dramdt_div", list(DIVERGING_PAIR))


def _pretty(name: str) -> str:
    return (name.replace("_", " ").replace(" nw", " (nW)").replace(" ns", " (ns)")
            .replace(" mv", " (mV)").replace(" fj", " (fJ)").replace(" fa", " (fA)")
            .replace(" ms", " (ms)").replace(" s", " (s)").strip().capitalize())


# ==========================================================================
# Stage 3 -- dataset
# ==========================================================================
def fig_design_space_coverage(points: np.ndarray, names: Sequence[str],
                              path: str | Path, max_vars: int = 5) -> list[Path]:
    """Pairwise projections of the sampling design, with 1-D marginals."""
    plt = _plt()
    k = min(len(names), max_vars)
    P = np.asarray(points)[:, :k]
    n_show = min(len(P), 3000)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(P), n_show, replace=False)

    fig, axes = plt.subplots(k, k, figsize=figure_size("page", height=6.4))
    for i in range(k):
        for j in range(k):
            ax = axes[i, j]
            if i == j:
                ax.hist(P[:, i], bins=40, color=CATEGORICAL[3], alpha=0.85,
                        edgecolor="white", linewidth=0.2)
                ax.set_yticks([])
            else:
                ax.scatter(P[idx, j], P[idx, i], s=0.7, alpha=0.25,
                           color=CATEGORICAL[0], linewidths=0, rasterized=True)
            if i == k - 1:
                ax.set_xlabel(names[j], fontsize=6)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(names[i], fontsize=6)
            else:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=5)
    fig.suptitle("Latin-hypercube design: pairwise projections and marginals")
    return save_figure(fig, path)


def fig_response_distributions(df: pd.DataFrame, columns: Sequence[str],
                               path: str | Path, log_scale: Sequence[str] = ()
                               ) -> list[Path]:
    """Marginal distribution of each simulated response."""
    plt = _plt()
    cols = [c for c in columns if c in df.columns]
    n = len(cols)
    ncol = min(3, n)
    nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=figure_size("page", height=2.1 * nrow))
    axes = np.atleast_1d(axes).ravel()

    for a, col in enumerate(cols):
        ax = axes[a]
        v = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if col in log_scale:
            v = v[v > 0]
            if v.size:
                ax.hist(np.log10(v), bins=60, color=CATEGORICAL[a % len(CATEGORICAL)],
                        alpha=0.85, edgecolor="white", linewidth=0.2)
                ax.set_xlabel(f"log$_{{10}}$({_pretty(col)})")
        else:
            ax.hist(v, bins=60, color=CATEGORICAL[a % len(CATEGORICAL)],
                    alpha=0.85, edgecolor="white", linewidth=0.2)
            ax.set_xlabel(_pretty(col))
        if v.size:
            ax.axvline(np.median(np.log10(v) if col in log_scale else v),
                       color=REFERENCE_INK, linewidth=0.9, linestyle="--")
        ax.set_ylabel("Count" if a % ncol == 0 else "")
    for a in range(len(cols), len(axes)):
        axes[a].set_visible(False)
    fig.suptitle("Distribution of simulated responses (dashed line: median)")
    return save_figure(fig, path)


def fig_correlation_matrix(df: pd.DataFrame, columns: Sequence[str],
                           path: str | Path, method: str = "spearman") -> list[Path]:
    """Signed correlation heatmap -- diverging ramp with a neutral midpoint."""
    plt = _plt()
    cols = [c for c in columns if c in df.columns]
    C = df[cols].apply(pd.to_numeric, errors="coerce").corr(method=method)

    fig, ax = plt.subplots(figsize=figure_size("page", height=6.0))
    im = ax.imshow(C.to_numpy(), cmap=diverging_cmap(), vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=90, fontsize=5.5)
    ax.set_yticklabels(cols, fontsize=5.5)
    ax.grid(False)
    if len(cols) <= 18:
        for i in range(len(cols)):
            for j in range(len(cols)):
                v = C.iloc[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=4.2,
                            color="white" if abs(v) > 0.62 else REFERENCE_INK)
    cb = fig.colorbar(im, ax=ax, shrink=0.75)
    cb.set_label(f"{method.capitalize()} rank correlation")
    ax.set_title("Correlation between design variables and responses")
    return save_figure(fig, path)


def fig_retention_validation(frame: pd.DataFrame, path: str | Path,
                             stats: Mapping[str, float] | None = None) -> list[Path]:
    """Quasi-static retention model against direct transient simulation."""
    plt = _plt()
    q = pd.to_numeric(frame["retention_quasistatic_s"], errors="coerce").to_numpy(float)
    t = pd.to_numeric(frame["retention_transient_s"], errors="coerce").to_numpy(float)
    m = np.isfinite(q) & np.isfinite(t) & (q > 0) & (t > 0)
    q, t = q[m], t[m]

    fig, axes = plt.subplots(1, 2, figsize=figure_size("page", height=2.7))
    ax = axes[0]
    ax.scatter(t, q, s=8, alpha=0.7, color=CATEGORICAL[0], edgecolor="white",
               linewidth=0.3, zorder=3)
    if q.size:
        lo = min(q.min(), t.min()) * 0.7
        hi = max(q.max(), t.max()) * 1.4
        ax.plot([lo, hi], [lo, hi], color=REFERENCE_INK, linewidth=0.9,
                linestyle="--", zorder=2, label="1:1")
        ax.fill_between([lo, hi], [lo / 2, hi / 2], [lo * 2, hi * 2],
                        color=CATEGORICAL[0], alpha=0.10, zorder=1,
                        label="factor of 2")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Direct transient retention (s)")
    ax.set_ylabel("Quasi-static model (s)")
    ax.legend(loc="upper left")
    annotate_panel(ax, "(a)")

    ax = axes[1]
    if q.size:
        err = np.log10(q / t)
        ax.hist(err, bins=30, color=CATEGORICAL[1], alpha=0.85,
                edgecolor="white", linewidth=0.2)
        ax.axvline(0.0, color=REFERENCE_INK, linewidth=0.9, linestyle="--")
    ax.set_xlabel(r"$\log_{10}$(model / transient)")
    ax.set_ylabel("Count")
    annotate_panel(ax, "(b)")
    if stats:
        txt = "\n".join([
            f"n = {int(stats.get('n_pairs', len(q)))}",
            f"r (log$_{{10}}$) = {stats.get('pearson_r_log10', float('nan')):.4f}",
            f"median |err| = {stats.get('median_abs_log10_error', float('nan')):.4f} dec",
            f"within 2x = {stats.get('within_factor_2_pct', float('nan')):.1f} %",
        ])
        ax.text(0.97, 0.95, txt, transform=ax.transAxes, ha="right", va="top",
                fontsize=6, bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                      edgecolor="#D9D9D9", alpha=0.9))
    fig.suptitle("Validation of the quasi-static retention model")
    return save_figure(fig, path)


# ==========================================================================
# Stage 4 -- surrogates
# ==========================================================================
def fig_surrogate_comparison(leaderboard: pd.DataFrame, path: str | Path,
                             metric: str = "test_r2") -> list[Path]:
    """Ranked model performance per target -- one panel per response."""
    plt = _plt()
    if leaderboard.empty or metric not in leaderboard.columns:
        return []
    targets = list(leaderboard["target"].unique())
    n = len(targets)
    fig, axes = plt.subplots(n, 1, figsize=figure_size("page", height=1.9 * n + 0.6),
                             sharex=False)
    axes = np.atleast_1d(axes)

    for a, target in enumerate(targets):
        ax = axes[a]
        sub = (leaderboard[leaderboard["target"] == target]
               .sort_values(metric, ascending=True))
        y = np.arange(len(sub))
        vals = sub[metric].to_numpy(dtype=float)
        # magnitude -> single hue, shaded by value (never a categorical hue)
        cmap = sequential_cmap()
        norm = (vals - np.nanmin(vals)) / max(np.nanmax(vals) - np.nanmin(vals), 1e-9)
        colors = [cmap(0.25 + 0.7 * v) for v in np.nan_to_num(norm)]
        ax.barh(y, vals, color=colors, height=0.72, edgecolor="white", linewidth=0.4)

        lo_col, hi_col = f"{metric}_ci_low", f"{metric}_ci_high"
        if lo_col in sub.columns and hi_col in sub.columns:
            lo = sub[lo_col].to_numpy(dtype=float)
            hi = sub[hi_col].to_numpy(dtype=float)
            ok = np.isfinite(lo) & np.isfinite(hi)
            ax.errorbar(vals[ok], y[ok],
                        xerr=np.vstack([vals[ok] - lo[ok], hi[ok] - vals[ok]]),
                        fmt="none", ecolor=REFERENCE_INK, elinewidth=0.7, capsize=1.5)

        ax.set_yticks(y)
        ax.set_yticklabels(sub["model"], fontsize=6)
        ax.set_xlabel(metric.replace("test_", "test ").upper())
        ax.set_title(_pretty(target), fontsize=7, loc="left")
        if np.isfinite(vals).any():
            ax.set_xlim(max(min(0.0, np.nanmin(vals)), -0.2),
                        min(1.02, np.nanmax(vals) * 1.06 + 0.02))
        for yi, v in zip(y, vals):
            if np.isfinite(v):
                ax.text(v, yi, f" {v:.4f}", va="center", ha="left", fontsize=5.2,
                        color=REFERENCE_INK)
    fig.suptitle("Surrogate model comparison (bars: held-out test score, "
                 "whiskers: 95 % bootstrap CI)")
    return save_figure(fig, path)


def fig_predicted_vs_actual(pairs: Mapping[str, tuple[np.ndarray, np.ndarray]],
                            path: str | Path,
                            metrics: Mapping[str, Mapping[str, float]] | None = None
                            ) -> list[Path]:
    """Parity plots for the best surrogate of each response."""
    plt = _plt()
    items = list(pairs.items())
    n = len(items)
    ncol = min(2, n)
    nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=figure_size("page", height=2.7 * nrow))
    axes = np.atleast_1d(axes).ravel()

    for a, (target, (y_true, y_pred)) in enumerate(items):
        ax = axes[a]
        yt = np.asarray(y_true, dtype=float)
        yp = np.asarray(y_pred, dtype=float)
        m = np.isfinite(yt) & np.isfinite(yp)
        yt, yp = yt[m], yp[m]
        ax.scatter(yt, yp, s=3, alpha=0.25, color=CATEGORICAL[a % len(CATEGORICAL)],
                   linewidths=0, rasterized=True)
        if yt.size:
            lo, hi = float(min(yt.min(), yp.min())), float(max(yt.max(), yp.max()))
            pad = 0.03 * (hi - lo + 1e-12)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                    color=REFERENCE_INK, linewidth=0.9, linestyle="--")
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel(f"Simulated {_pretty(target)}")
        ax.set_ylabel(f"Predicted {_pretty(target)}")
        if metrics and target in metrics:
            mm = metrics[target]
            ax.text(0.04, 0.95,
                    f"$R^2$ = {mm.get('r2', float('nan')):.4f}\n"
                    f"RMSE = {mm.get('rmse', float('nan')):.4g}\n"
                    f"MAE = {mm.get('mae', float('nan')):.4g}",
                    transform=ax.transAxes, va="top", ha="left", fontsize=6,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor="#D9D9D9", alpha=0.9))
        annotate_panel(ax, f"({chr(97 + a)})")
    for a in range(len(items), len(axes)):
        axes[a].set_visible(False)
    fig.suptitle("Digital-twin predictions against SPICE ground truth (test split)")
    return save_figure(fig, path)


def fig_residuals(pairs: Mapping[str, tuple[np.ndarray, np.ndarray]],
                  path: str | Path) -> list[Path]:
    """Residual-versus-fitted diagnostics."""
    plt = _plt()
    items = list(pairs.items())
    n = len(items)
    fig, axes = plt.subplots(1, n, figsize=figure_size("page", height=2.2))
    axes = np.atleast_1d(axes).ravel()
    for a, (target, (y_true, y_pred)) in enumerate(items):
        ax = axes[a]
        yt, yp = np.asarray(y_true, float), np.asarray(y_pred, float)
        m = np.isfinite(yt) & np.isfinite(yp)
        r = yp[m] - yt[m]
        ax.scatter(yp[m], r, s=2.5, alpha=0.22,
                   color=CATEGORICAL[a % len(CATEGORICAL)], linewidths=0,
                   rasterized=True)
        ax.axhline(0.0, color=REFERENCE_INK, linewidth=0.9, linestyle="--")
        ax.set_xlabel(f"Predicted {_pretty(target)}")
        ax.set_ylabel("Residual" if a == 0 else "")
        annotate_panel(ax, f"({chr(97 + a)})")
    fig.suptitle("Residual diagnostics")
    return save_figure(fig, path)


def fig_cv_boxplots(fold_scores: pd.DataFrame, path: str | Path,
                    metric: str = "r2", target: str | None = None) -> list[Path]:
    """Per-fold cross-validation score distributions by model."""
    plt = _plt()
    df = fold_scores if target is None else fold_scores[fold_scores["target"] == target]
    if df.empty:
        return []
    order = (df.groupby("model")[metric].median().sort_values(ascending=False).index.tolist())
    data = [df[df["model"] == m][metric].dropna().to_numpy() for m in order]

    fig, ax = plt.subplots(figsize=figure_size("page", height=2.8))
    bp = ax.boxplot(data, patch_artist=True, widths=0.62,
                    medianprops=dict(color=REFERENCE_INK, linewidth=1.1),
                    flierprops=dict(markersize=2, markerfacecolor="none",
                                    markeredgecolor=REFERENCE_INK, markeredgewidth=0.4))
    cmap = sequential_cmap()
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(cmap(0.30 + 0.6 * (1 - i / max(len(order) - 1, 1))))
        patch.set_edgecolor("white")
        patch.set_linewidth(0.5)
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(order, rotation=38, ha="right", fontsize=6)
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Cross-validated {metric.upper()}"
                 + (f" -- {_pretty(target)}" if target else ""), loc="left")
    return save_figure(fig, path)


def fig_learning_curve(curve: pd.DataFrame, path: str | Path,
                       target: str = "") -> list[Path]:
    """Score against training-set size (how much data the surrogate needs)."""
    plt = _plt()
    if curve.empty:
        return []
    fig, ax = plt.subplots(figsize=figure_size("column", height=2.3))
    for i, (model, sub) in enumerate(curve.groupby("model")):
        st = series_style(i)
        sub = sub.sort_values("n_train")
        ax.plot(sub["n_train"], sub["score_mean"], label=model,
                color=st["color"], marker=st["marker"], linestyle=st["linestyle"],
                markersize=3)
        if "score_std" in sub:
            ax.fill_between(sub["n_train"],
                            sub["score_mean"] - sub["score_std"],
                            sub["score_mean"] + sub["score_std"],
                            color=st["color"], alpha=0.12, linewidth=0)
    ax.set_xscale("log")
    ax.set_xlabel("Training samples")
    ax.set_ylabel("Cross-validated $R^2$")
    ax.set_title(f"Learning curve{(' -- ' + _pretty(target)) if target else ''}", loc="left")
    ax.legend(fontsize=5.5)
    return save_figure(fig, path)


# ==========================================================================
# Stage 4b -- explainability
# ==========================================================================
def fig_shap_importance(importance: pd.DataFrame, path: str | Path,
                        top_n: int = 15, target: str = "") -> list[Path]:
    """Global SHAP importance ranking."""
    plt = _plt()
    if importance.empty:
        return []
    sub = importance.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=figure_size("column", height=0.17 * len(sub) + 1.0))
    cmap = sequential_cmap()
    vals = sub["mean_abs_shap"].to_numpy(dtype=float)
    norm = vals / max(vals.max(), 1e-12)
    ax.barh(range(len(sub)), vals, height=0.72, edgecolor="white", linewidth=0.4,
            color=[cmap(0.28 + 0.65 * v) for v in norm])
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels([str(f).replace("num__", "").replace("cat__", "")
                        for f in sub["feature"]], fontsize=6)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Feature importance{(' -- ' + _pretty(target)) if target else ''}",
                 loc="left")
    if "importance_pct" in sub.columns:
        for i, (v, p) in enumerate(zip(vals, sub["importance_pct"])):
            ax.text(v, i, f" {p:.1f}%", va="center", ha="left", fontsize=5.2,
                    color=REFERENCE_INK)
    return save_figure(fig, path)


def fig_shap_beeswarm(shap_values: np.ndarray, data: np.ndarray,
                      features: Sequence[str], path: str | Path,
                      top_n: int = 12, target: str = "") -> list[Path]:
    """Beeswarm of SHAP values: direction and magnitude per feature."""
    plt = _plt()
    if shap_values is None or len(shap_values) == 0:
        return []
    sv = np.asarray(shap_values)
    X = np.asarray(data)
    # Defensive: an explainer may have scored fewer rows than were prepared.
    n = min(len(sv), len(X))
    sv, X = sv[:n], X[:n]
    order = np.argsort(np.abs(sv).mean(axis=0))[::-1][:top_n][::-1]

    fig, ax = plt.subplots(figsize=figure_size("column", height=0.20 * len(order) + 1.1))
    cmap = sequential_cmap()
    rng = np.random.default_rng(0)
    for row, j in enumerate(order):
        vals = sv[:, j]
        feat = X[:, j].astype(float)
        finite = np.isfinite(feat)
        if finite.sum() > 1:
            lo, hi = np.nanpercentile(feat[finite], [1, 99])
            norm = np.clip((feat - lo) / max(hi - lo, 1e-12), 0, 1)
        else:
            norm = np.full_like(feat, 0.5, dtype=float)
        jitter = rng.uniform(-0.16, 0.16, size=len(vals))
        ax.scatter(vals, row + jitter, s=2.2, c=[cmap(0.15 + 0.75 * v) for v in norm],
                   linewidths=0, alpha=0.6, rasterized=True)
    ax.axvline(0.0, color=REFERENCE_INK, linewidth=0.7, linestyle="--")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([str(features[j]).replace("num__", "").replace("cat__", "")
                        for j in order], fontsize=6)
    ax.set_xlabel("SHAP value (effect on the prediction)")
    ax.set_title(f"SHAP summary{(' -- ' + _pretty(target)) if target else ''}", loc="left")

    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, pad=0.02, shrink=0.8)
    cb.set_label("Feature value (low $\\rightarrow$ high)", fontsize=6)
    cb.ax.tick_params(labelsize=5)
    return save_figure(fig, path)


def fig_shap_interactions(interactions: pd.DataFrame, path: str | Path,
                          top_n: int = 12, target: str = "") -> list[Path]:
    """Strongest pairwise SHAP interactions."""
    plt = _plt()
    if interactions.empty:
        return []
    sub = interactions.head(top_n).iloc[::-1]
    labels = [f"{str(a).replace('num__','')} x {str(b).replace('num__','')}"
              for a, b in zip(sub["feature_1"], sub["feature_2"])]
    fig, ax = plt.subplots(figsize=figure_size("column", height=0.18 * len(sub) + 1.0))
    cmap = sequential_cmap()
    vals = sub["interaction_strength"].to_numpy(dtype=float)
    norm = vals / max(vals.max(), 1e-12)
    ax.barh(range(len(sub)), vals, height=0.7, edgecolor="white", linewidth=0.4,
            color=[cmap(0.28 + 0.65 * v) for v in norm])
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels(labels, fontsize=5.5)
    ax.set_xlabel("Mean |SHAP interaction value|")
    ax.set_title(f"Feature interactions{(' -- ' + _pretty(target)) if target else ''}",
                 loc="left")
    return save_figure(fig, path)


def fig_partial_dependence(pdp: pd.DataFrame, path: str | Path,
                           target: str = "", ale: pd.DataFrame | None = None
                           ) -> list[Path]:
    """Marginal effect curves; ALE overlaid where available."""
    plt = _plt()
    if pdp.empty:
        return []
    feats = list(dict.fromkeys(pdp["feature"]))[:8]
    ncol = min(4, len(feats))
    nrow = math.ceil(len(feats) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=figure_size("page", height=1.9 * nrow))
    axes = np.atleast_1d(axes).ravel()

    for a, feat in enumerate(feats):
        ax = axes[a]
        sub = pdp[pdp["feature"] == feat].sort_values("value")
        ax.plot(sub["value"], sub["partial_dependence"], color=CATEGORICAL[0],
                linewidth=1.3, linestyle="-", label="PDP")
        if ale is not None and not ale.empty:
            sa = ale[ale["feature"] == feat].sort_values("value")
            if not sa.empty:
                centre = sub["partial_dependence"].mean()
                ax.plot(sa["value"], sa["ale"] + centre, color=CATEGORICAL[1],
                        linewidth=1.1, linestyle="--", label="ALE")
        ax.set_xlabel(feat, fontsize=6)
        ax.set_ylabel("Effect" if a % ncol == 0 else "")
        ax.tick_params(labelsize=5.5)
        if a == 0:
            ax.legend(fontsize=5.5, loc="best")
    for a in range(len(feats), len(axes)):
        axes[a].set_visible(False)
    fig.suptitle(f"Marginal effects{(' -- ' + _pretty(target)) if target else ''}")
    return save_figure(fig, path)


def fig_ale(ale: pd.DataFrame, path: str | Path, target: str = "") -> list[Path]:
    """Accumulated local effects alone (valid under correlated inputs)."""
    plt = _plt()
    if ale.empty:
        return []
    feats = list(dict.fromkeys(ale["feature"]))[:8]
    ncol = min(4, len(feats))
    nrow = math.ceil(len(feats) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=figure_size("page", height=1.9 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for a, feat in enumerate(feats):
        ax = axes[a]
        sub = ale[ale["feature"] == feat].sort_values("value")
        ax.plot(sub["value"], sub["ale"], color=CATEGORICAL[a % len(CATEGORICAL)],
                linewidth=1.3)
        ax.axhline(0.0, color=REFERENCE_INK, linewidth=0.6, linestyle=":")
        ax.set_xlabel(feat, fontsize=6)
        ax.set_ylabel("ALE" if a % ncol == 0 else "")
        ax.tick_params(labelsize=5.5)
    for a in range(len(feats), len(axes)):
        axes[a].set_visible(False)
    fig.suptitle(f"Accumulated local effects{(' -- ' + _pretty(target)) if target else ''}")
    return save_figure(fig, path)


# ==========================================================================
# Stage 5 -- optimisation
# ==========================================================================
def fig_pareto_matrix(front: pd.DataFrame, objectives: Sequence[str],
                      path: str | Path, colour_by: str | None = None,
                      labels: Mapping[str, str] | None = None) -> list[Path]:
    """Pairwise projections of the Pareto set."""
    plt = _plt()
    labels = dict(labels or {})
    objs = [o for o in objectives if o in front.columns]
    k = len(objs)
    if k < 2:
        return []
    pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    ncol = min(3, len(pairs))
    nrow = math.ceil(len(pairs) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=figure_size("page", height=2.3 * nrow))
    axes = np.atleast_1d(axes).ravel()

    groups = (list(front.groupby(colour_by)) if colour_by and colour_by in front.columns
              else [(None, front)])
    for a, (i, j) in enumerate(pairs):
        ax = axes[a]
        for g, (name, sub) in enumerate(groups):
            st = series_style(g)
            ax.scatter(sub[objs[i]], sub[objs[j]], s=11, alpha=0.75,
                       color=st["color"], marker=st["marker"],
                       edgecolor="white", linewidth=0.3,
                       label=str(name) if name is not None else None)
        ax.set_xlabel(labels.get(objs[i], _pretty(objs[i])), fontsize=6)
        ax.set_ylabel(labels.get(objs[j], _pretty(objs[j])), fontsize=6)
        ax.tick_params(labelsize=5.5)
        if objs[i].startswith("retention") or "power" in objs[i]:
            ax.set_xscale("log")
        if objs[j].startswith("retention") or "power" in objs[j]:
            ax.set_yscale("log")
        annotate_panel(ax, f"({chr(97 + a)})")
    for a in range(len(pairs), len(axes)):
        axes[a].set_visible(False)
    if groups[0][0] is not None:
        handles, lab = axes[0].get_legend_handles_labels()
        fig.legend(handles, lab, loc="lower center", ncol=min(len(groups), 6),
                   frameon=False, fontsize=6, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Pareto-optimal design set: objective-pair projections")
    return save_figure(fig, path)


def fig_parallel_coordinates(front: pd.DataFrame, columns: Sequence[str],
                             path: str | Path, colour_by: str | None = None,
                             max_lines: int = 250) -> list[Path]:
    """Parallel-coordinates view of the Pareto set (all objectives at once)."""
    plt = _plt()
    cols = [c for c in columns if c in front.columns]
    if len(cols) < 2 or front.empty:
        return []
    df = front
    if len(df) > max_lines:
        df = df.sample(max_lines, random_state=0)
    M = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    lo = np.nanmin(M, axis=0)
    hi = np.nanmax(M, axis=0)
    N = (M - lo) / np.where(hi - lo > 0, hi - lo, 1.0)

    fig, ax = plt.subplots(figsize=figure_size("page", height=2.9))
    if colour_by and colour_by in df.columns:
        keys = list(dict.fromkeys(df[colour_by]))
        cmap = {k: CATEGORICAL[i % len(CATEGORICAL)] for i, k in enumerate(keys)}
        colors = [cmap[k] for k in df[colour_by]]
    else:
        keys, colors = [], [CATEGORICAL[0]] * len(df)
    for row, c in zip(N, colors):
        ax.plot(range(len(cols)), row, color=c, alpha=0.28, linewidth=0.7)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([_pretty(c) for c in cols], rotation=22, ha="right", fontsize=6)
    ax.set_ylabel("Normalised value")
    ax.set_ylim(-0.03, 1.03)
    for x in range(len(cols)):
        ax.axvline(x, color="#D9D9D9", linewidth=0.5, zorder=0)
    if keys:
        from matplotlib.lines import Line2D
        ax.legend(handles=[Line2D([0], [0], color=cmap[k], lw=1.4, label=str(k))
                           for k in keys], fontsize=5.5, ncol=min(len(keys), 6),
                  loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False)
    ax.set_title("Pareto set in parallel coordinates", loc="left")
    return save_figure(fig, path)


def fig_hypervolume_convergence(convergence: pd.DataFrame, path: str | Path
                                ) -> list[Path]:
    """Median hypervolume against evaluation count, with an IQR band."""
    plt = _plt()
    if convergence.empty:
        return []
    fig, ax = plt.subplots(figsize=figure_size("page", height=2.6))
    for i, (alg, sub) in enumerate(convergence.groupby("label")):
        st = series_style(i)
        g = sub.groupby("n_evaluations")["hypervolume"]
        med, q1, q3 = g.median(), g.quantile(0.25), g.quantile(0.75)
        x = med.index.to_numpy(dtype=float)
        ax.plot(x, med.to_numpy(), label=str(alg), color=st["color"],
                linestyle=st["linestyle"], linewidth=1.3,
                marker=st["marker"], markersize=2.6, markevery=max(len(x) // 12, 1))
        ax.fill_between(x, q1.to_numpy(), q3.to_numpy(), color=st["color"],
                        alpha=0.13, linewidth=0)
    ax.set_xlabel("Surrogate evaluations")
    ax.set_ylabel("Hypervolume")
    ax.set_title("Convergence (median over independent runs, band: IQR)", loc="left")
    ax.legend(fontsize=5.8, ncol=2)
    return save_figure(fig, path)


def fig_indicator_boxplots(table: pd.DataFrame, path: str | Path,
                           indicators: Sequence[str] = ("hypervolume", "igd_plus",
                                                        "spacing", "spread")
                           ) -> list[Path]:
    """Per-run indicator distributions by algorithm."""
    plt = _plt()
    inds = [i for i in indicators if i in table.columns]
    if table.empty or not inds:
        return []
    algs = list(dict.fromkeys(table["label"]))
    ncol = min(2, len(inds))
    nrow = math.ceil(len(inds) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=figure_size("page", height=2.3 * nrow))
    axes = np.atleast_1d(axes).ravel()

    for a, ind in enumerate(inds):
        ax = axes[a]
        data = [table[table["label"] == alg][ind].dropna().to_numpy() for alg in algs]
        bp = ax.boxplot(data, patch_artist=True, widths=0.6,
                        medianprops=dict(color=REFERENCE_INK, linewidth=1.1),
                        flierprops=dict(markersize=2, markerfacecolor="none",
                                        markeredgecolor=REFERENCE_INK,
                                        markeredgewidth=0.4))
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(CATEGORICAL[i % len(CATEGORICAL)])
            patch.set_alpha(0.72)
            patch.set_edgecolor("white")
            patch.set_linewidth(0.5)
        ax.set_xticks(range(1, len(algs) + 1))
        ax.set_xticklabels(algs, rotation=28, ha="right", fontsize=5.6)
        ax.set_ylabel(_pretty(ind))
        annotate_panel(ax, f"({chr(97 + a)})")
    for a in range(len(inds), len(axes)):
        axes[a].set_visible(False)
    fig.suptitle("Optimiser comparison across independent runs")
    return save_figure(fig, path)


def fig_critical_difference(ranks: pd.DataFrame, cd: float, path: str | Path,
                            title: str = "") -> list[Path]:
    """Demsar critical-difference diagram."""
    plt = _plt()
    if ranks.empty or not np.isfinite(cd):
        return []
    r = ranks.sort_values("mean_rank").reset_index(drop=True)
    k = len(r)
    vals = r["mean_rank"].to_numpy(dtype=float)
    lo = max(math.floor(vals.min()) - 0.2, 0.5)
    hi = math.ceil(vals.max()) + 0.2
    span = hi - lo

    # Labels live in margins *outside* the rank axis, so a connector can never
    # run underneath its own text.
    margin = 0.55 * span
    left, right = k // 2, k - k // 2
    row_h = 0.42

    fig, ax = plt.subplots(figsize=figure_size("page",
                                               height=0.34 * max(left, right) + 1.5))
    n_cliques_max = max(k - 1, 1)
    ax.set_xlim(lo - margin, hi + margin)
    ax.set_ylim(-0.34 - 0.16 * n_cliques_max, max(left, right) * row_h + 0.95)
    ax.axis("off")
    ax.grid(False)

    y0 = 0.0                                    # the rank axis sits at y = 0
    ax.plot([lo, hi], [y0, y0], color=REFERENCE_INK, linewidth=0.9,
            solid_capstyle="butt")
    # Ticks hang *below* the axis: the method connectors occupy the space above
    # it, and tick labels placed there collide with the lowest-ranked method.
    for t in np.arange(math.ceil(lo), math.floor(hi) + 1):
        ax.plot([t, t], [y0, y0 - 0.07], color=REFERENCE_INK, linewidth=0.7)
        ax.text(t, y0 - 0.10, f"{int(t)}", ha="center", va="top", fontsize=6)

    # methods branch upward: best ranks to the left margin, worst to the right
    for i, row in r.iterrows():
        is_left = i < left
        slot = i if is_left else (k - 1 - i)
        y = y0 + row_h * (slot + 1)
        x = float(row["mean_rank"])
        edge = lo if is_left else hi
        ax.plot([x, x], [y0, y], color=REFERENCE_INK, linewidth=0.7)
        ax.plot([x, edge], [y, y], color=REFERENCE_INK, linewidth=0.7)
        ax.text(edge - 0.02 * span if is_left else edge + 0.02 * span, y,
                f"{row['group']} ({x:.2f})",
                va="center", ha="right" if is_left else "left", fontsize=6)

    # cliques: maximal groups whose mean ranks differ by less than CD, drawn
    # below the axis so they never collide with the method connectors
    drawn = 0
    for i in range(k):
        j = i
        while j + 1 < k and vals[j + 1] - vals[i] <= cd:
            j += 1
        if j > i and not (i > 0 and vals[j] - vals[i - 1] <= cd):
            y = y0 - 0.26 - 0.16 * drawn
            ax.plot([vals[i] - 0.02 * span, vals[j] + 0.02 * span], [y, y],
                    color=STATUS_COLORS["serious"], linewidth=2.6,
                    solid_capstyle="round")
            drawn += 1

    # the CD ruler
    y_cd = y0 + max(left, right) * row_h + 0.34
    ax.plot([lo, lo + cd], [y_cd, y_cd], color=STATUS_COLORS["serious"],
            linewidth=2.2, solid_capstyle="butt")
    for xt in (lo, lo + cd):
        ax.plot([xt, xt], [y_cd - 0.06, y_cd + 0.06],
                color=STATUS_COLORS["serious"], linewidth=1.0)
    ax.text(lo + cd / 2, y_cd + 0.09, f"CD = {cd:.2f}", ha="center", va="bottom",
            fontsize=6, color=STATUS_COLORS["serious"])

    ax.set_title(title or "Critical-difference diagram (Nemenyi, alpha = 0.05)",
                 loc="left", fontsize=7)
    ax.text(0.5, -0.02, "mean Friedman rank (lower is better); methods joined by a "
                        "bar are not significantly different",
            transform=ax.transAxes, ha="center", va="top", fontsize=5.5,
            color="#4D4D4D")
    return save_figure(fig, path)


# ==========================================================================
# Stage 7 -- robustness and assist techniques
# ==========================================================================
def fig_corner_heatmap(grid: pd.DataFrame, response: str, path: str | Path,
                       design_id: str = "") -> list[Path]:
    """Response across the corner x temperature grid (magnitude -> single hue)."""
    plt = _plt()
    if grid.empty or response not in grid.columns:
        return []
    pivot = grid.pivot_table(index="corner", columns="temperature_c",
                             values=response, aggfunc="mean")
    fig, ax = plt.subplots(figsize=figure_size("column", height=2.1))
    im = ax.imshow(pivot.to_numpy(), cmap=sequential_cmap(), aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c:g}" for c in pivot.columns], fontsize=6)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=6)
    ax.set_xlabel("Temperature (degC)")
    ax.grid(False)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.iloc[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.3g}", ha="center", va="center", fontsize=5,
                        color="white" if v > np.nanmedian(pivot.to_numpy()) else REFERENCE_INK)
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label(_pretty(response), fontsize=6)
    ax.set_title(f"PVT corner sweep{(' -- ' + design_id) if design_id else ''}",
                 loc="left", fontsize=7)
    return save_figure(fig, path)


def fig_monte_carlo(samples: pd.DataFrame, specs: Mapping[str, float],
                    path: str | Path, design_id: str = "",
                    directions: Mapping[str, str] | None = None) -> list[Path]:
    """Monte-Carlo response distributions with the specification limits marked."""
    plt = _plt()
    directions = dict(directions or {})
    cols = [c for c in specs if c in samples.columns]
    if not cols:
        return []
    ncol = min(2, len(cols))
    nrow = math.ceil(len(cols) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=figure_size("page", height=2.2 * nrow))
    axes = np.atleast_1d(axes).ravel()

    for a, col in enumerate(cols):
        ax = axes[a]
        v = pd.to_numeric(samples[col], errors="coerce").to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        limit = float(specs[col])
        direction = directions.get(col, "min")
        passed = (v >= limit) if direction == "min" else (v <= limit)
        ax.hist(v[passed], bins=45, color=STATUS_COLORS["good"], alpha=0.75,
                edgecolor="white", linewidth=0.2, label="pass")
        if (~passed).any():
            ax.hist(v[~passed], bins=45, color=STATUS_COLORS["critical"], alpha=0.8,
                    edgecolor="white", linewidth=0.2, label="fail")
        ax.axvline(limit, color=REFERENCE_INK, linewidth=1.0, linestyle="--")
        ax.set_xlabel(_pretty(col))
        ax.set_ylabel("Count" if a % ncol == 0 else "")
        ax.text(0.97, 0.94, f"yield {100 * passed.mean():.2f} %",
                transform=ax.transAxes, ha="right", va="top", fontsize=6,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor="#D9D9D9", alpha=0.9))
        if a == 0:
            ax.legend(fontsize=5.5, loc="upper left")
        annotate_panel(ax, f"({chr(97 + a)})")
    for a in range(len(cols), len(axes)):
        axes[a].set_visible(False)
    fig.suptitle(f"Monte-Carlo robustness{(' -- ' + design_id) if design_id else ''}"
                 " (dashed line: specification limit)")
    return save_figure(fig, path)


def fig_sobol_indices(sobol: pd.DataFrame, path: str | Path,
                      top_n: int = 12) -> list[Path]:
    """First-order versus total-effect sensitivity indices."""
    plt = _plt()
    if sobol.empty:
        return []
    responses = list(dict.fromkeys(sobol["response"])) if "response" in sobol else [""]
    n = len(responses)
    fig, axes = plt.subplots(n, 1, figsize=figure_size("page", height=2.1 * n))
    axes = np.atleast_1d(axes).ravel()
    for a, resp in enumerate(responses):
        ax = axes[a]
        sub = (sobol[sobol["response"] == resp] if resp else sobol).head(top_n)
        y = np.arange(len(sub))
        ax.barh(y - 0.19, sub["S1"], height=0.36, color=CATEGORICAL[0],
                edgecolor="white", linewidth=0.4, label="First order $S_1$")
        ax.barh(y + 0.19, sub["ST"], height=0.36, color=CATEGORICAL[3],
                edgecolor="white", linewidth=0.4, hatch=HATCHES[1],
                label="Total effect $S_T$")
        ax.set_yticks(y)
        ax.set_yticklabels(sub["variable"], fontsize=6)
        ax.invert_yaxis()
        ax.set_xlabel("Sensitivity index")
        ax.set_title(_pretty(resp) if resp else "", fontsize=7, loc="left")
        if a == 0:
            ax.legend(fontsize=5.8, loc="lower right")
    fig.suptitle("Sobol' variance-based sensitivity "
                 "($S_T - S_1$ is the interaction share)")
    return save_figure(fig, path)


def fig_assist_comparison(frame: pd.DataFrame, metrics: Sequence[str],
                          path: str | Path, baseline: str = "baseline") -> list[Path]:
    """Assist techniques against the baseline, as relative change per metric."""
    plt = _plt()
    cols = [m for m in metrics if m in frame.columns]
    if frame.empty or not cols:
        return []
    if "assist_config" not in frame.columns:
        return []
    agg = frame.groupby("assist_config")[cols].median()
    if baseline not in agg.index:
        return []
    rel = 100.0 * (agg / agg.loc[baseline] - 1.0)
    rel = rel.drop(index=baseline)

    fig, ax = plt.subplots(figsize=figure_size("page", height=0.40 * len(rel) + 1.6))
    y = np.arange(len(rel))
    width = 0.8 / max(len(cols), 1)
    for i, col in enumerate(cols):
        st = series_style(i)
        ax.barh(y + (i - (len(cols) - 1) / 2) * width, rel[col], height=width * 0.92,
                color=st["color"], edgecolor="white", linewidth=0.4,
                hatch=HATCHES[i % len(HATCHES)], label=_pretty(col))
    ax.axvline(0.0, color=REFERENCE_INK, linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(rel.index, fontsize=6)
    ax.set_xlabel(f"Change relative to '{baseline}' (%)")
    ax.legend(fontsize=5.6, ncol=min(len(cols), 4), loc="upper center",
              bbox_to_anchor=(0.5, -0.16), frameon=False)
    ax.set_title("DRAM assist techniques: median relative effect", loc="left")
    return save_figure(fig, path)


def fig_twin_speedup(benchmark: pd.DataFrame, path: str | Path) -> list[Path]:
    """Digital-twin evaluation latency and the resulting speed-up over SPICE."""
    plt = _plt()
    if benchmark.empty:
        return []
    fig, axes = plt.subplots(1, 2, figsize=figure_size("page", height=2.3))
    ax = axes[0]
    ax.plot(benchmark["batch_size"], benchmark["ms_per_design"],
            color=CATEGORICAL[0], marker=MARKERS[0], linewidth=1.3, markersize=4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Batch size (designs)")
    ax.set_ylabel("Latency per design (ms)")
    annotate_panel(ax, "(a)")

    ax = axes[1]
    sp = pd.to_numeric(benchmark.get("speedup_vs_spice"), errors="coerce")
    ok = sp.notna()
    if ok.any():
        ax.bar(np.arange(ok.sum()), sp[ok], color=CATEGORICAL[1],
               edgecolor="white", linewidth=0.4, width=0.7)
        ax.set_xticks(np.arange(ok.sum()))
        ax.set_xticklabels(benchmark["batch_size"][ok], fontsize=6)
        ax.set_yscale("log")
        ax.set_xlabel("Batch size (designs)")
        ax.set_ylabel(r"Speed-up over SPICE ($\times$)")
        for i, v in enumerate(sp[ok]):
            ax.text(i, v, f"{v:,.0f}x", ha="center", va="bottom", fontsize=5.4)
    annotate_panel(ax, "(b)")
    fig.suptitle("Digital-twin evaluation cost against circuit simulation")
    return save_figure(fig, path)


def fig_tradeoff_scatter(frame: pd.DataFrame, x: str, y: str, path: str | Path,
                         size_by: str | None = None, colour_by: str | None = None,
                         highlight: pd.DataFrame | None = None) -> list[Path]:
    """Two-objective trade-off with the selected designs highlighted."""
    plt = _plt()
    if frame.empty or x not in frame.columns or y not in frame.columns:
        return []
    fig, ax = plt.subplots(figsize=figure_size("column", height=2.5))
    s = 14.0
    if size_by and size_by in frame.columns:
        v = pd.to_numeric(frame[size_by], errors="coerce").to_numpy(dtype=float)
        rng = np.nanmax(v) - np.nanmin(v)
        s = 5 + 28 * (v - np.nanmin(v)) / (rng if rng > 0 else 1.0)
    if colour_by and colour_by in frame.columns:
        c = pd.to_numeric(frame[colour_by], errors="coerce")
        sc = ax.scatter(frame[x], frame[y], c=c, s=s, cmap=sequential_cmap(),
                        alpha=0.85, edgecolor="white", linewidth=0.3)
        cb = fig.colorbar(sc, ax=ax, shrink=0.85)
        cb.set_label(_pretty(colour_by), fontsize=6)
    else:
        ax.scatter(frame[x], frame[y], s=s, color=CATEGORICAL[0], alpha=0.8,
                   edgecolor="white", linewidth=0.3)
    if highlight is not None and not highlight.empty:
        ax.scatter(highlight[x], highlight[y], s=48, facecolor="none",
                   edgecolor=STATUS_COLORS["critical"], linewidth=1.1,
                   label="selected", zorder=5)
        ax.legend(fontsize=5.8)
    ax.set_xlabel(_pretty(x))
    ax.set_ylabel(_pretty(y))
    if "power" in x or "retention" in x:
        ax.set_xscale("log")
    if "power" in y or "retention" in y:
        ax.set_yscale("log")
    ax.set_title("Design trade-off", loc="left")
    return save_figure(fig, path)
