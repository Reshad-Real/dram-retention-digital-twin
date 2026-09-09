#!/usr/bin/env python
"""Stage 13 -- consolidated, large-font manuscript figures.

Regenerates a curated, multi-panel figure set from the saved analysis artefacts
(no model reloading, no re-running of SHAP), plus the two live-SPICE figures
(waveforms, leakage).  Every panel carries a bold "(x) Name" title and uses a
large font so the figures stay legible at page width.

    python scripts/13_manuscript_figures.py            # all figures
    python scripts/13_manuscript_figures.py --no-spice # skip the two live ones
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = RES / "figures"
XAI = RES / "xai"
OPT = RES / "optimization"
ROB = RES / "robustness"
STAT = RES / "statistics"
SUR = RES / "surrogate"
TAB = RES / "tables"
PROC = ROOT / "data" / "processed"
FIG.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib.lines import Line2D      # noqa: E402

from dramdt.viz.palette import CATEGORICAL, REFERENCE_INK   # noqa: E402

CAT = list(CATEGORICAL)
INK = REFERENCE_INK

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 14,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "xtick.labelsize": 12.5,
    "ytick.labelsize": 12.5,
    "legend.fontsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.9,
    "axes.grid": True,
    "grid.color": "#D9D9D9",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "lines.linewidth": 2.0,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "figure.constrained_layout.use": True,
})

FORMATS = ("png", "pdf", "svg")
NICE = {
    "log10_retention_time_s": "Retention  $\\log_{10} t_{ret}$ (s)",
    "retention_time_s": "Retention time (s)",
    "read_delay_ns": "Read delay (ns)",
    "read_margin_mv": "Read margin (mV)",
    "log10_total_power_nw": "Power  $\\log_{10} P$ (nW)",
    "total_power_nw": "Total power (nW)",
    "write_efficiency": "Write efficiency",
    "energy_per_access_fj": "Energy/access (fJ)",
}

#: raw feature / variable identifier -> paper notation (F8, L7).
FEAT = {
    "vdd": "$V_{DD}$", "vwl_boost": "$V_{boost}$", "vwl_low": "$V_{WL,low}$",
    "vblpre_ratio": r"$\rho$", "cs": "$C_s$", "cbl_ratio": r"$\kappa$",
    "cbl_over_cs": r"$\kappa$", "cbl": "$C_{bl}$",
    "wacc": "$W_{acc}$", "lacc": "$L_{acc}$", "wsan": "$W_{san}$",
    "wsap_ratio": "$W_{sap}/W_{san}$", "wen_ratio": "$W_{en}/W_{san}$",
    "t_write_pulse": "$t_{wr}$", "t_sense_delay": "$t_{se}$", "t_sense": "$t_{se}$",
    "temperature_c": "$T$", "temperature_k": "$T$", "corner": "corner",
    "vwl_overdrive": "$V_{ov}$", "charge_share_ratio": r"$C_s/(C_s{+}C_{bl})$",
    "vgs_retention": "$V_{gs,ret}$", "vplate_ratio": r"$\rho_{pl}$",
}


def flab(raw):
    """Map a raw feature/variable identifier to the paper's notation."""
    s = str(raw).replace("num__", "").replace("cat__", "")
    if s in FEAT:
        return FEAT[s]
    for pre in ("corner_", "corner="):          # one-hot corner columns -> corner code
        if s.startswith(pre):
            return s[len(pre):]
    return s.replace("_", " ")


#: canonical per-algorithm colour, applied across every optimisation panel (F5).
ALGO_COLORS = {
    "nsga2": CAT[0], "NSGA-II": CAT[0],
    "nsga3": CAT[3], "NSGA-III": CAT[3],
    "moead": CAT[5], "MOEA/D": CAT[5],
    "de": CAT[1], "Differential evolution": CAT[1],
    "pso": CAT[2], "Particle-swarm optimization": CAT[2],
    "bayesian": CAT[6], "Bayesian optimization (multi-objective TPE)": CAT[6],
}
#: short display label for optimisation panels.
ALGO_SHORT = {
    "nsga2": "NSGA-II", "nsga3": "NSGA-III", "moead": "MOEA/D",
    "de": "DE", "pso": "PSO", "bayesian": "Bayesian (TPE)",
    "NSGA-II": "NSGA-II", "NSGA-III": "NSGA-III", "MOEA/D": "MOEA/D",
    "Differential evolution": "DE", "Particle-swarm optimization": "PSO",
    "Bayesian optimization (multi-objective TPE)": "Bayesian (TPE)",
}


def acolor(key):
    return ALGO_COLORS.get(str(key), CAT[7])


def ashort(key):
    return ALGO_SHORT.get(str(key), str(key))


def save(fig, name):
    for ext in FORMATS:
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {name}")


def plabel(ax, text):
    ax.set_title(text, loc="left", fontweight="bold", fontsize=15, pad=6)


# ======================================================================
def fig_dataset(name="fig05_dataset"):
    """(a) design-space coverage, (b) response distributions, (c) correlation."""
    df = pd.read_parquet(PROC / "main_processed.parquet")
    fig = plt.figure(figsize=(15, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.15, 1.25])

    # (a) coverage: two representative design variables, hexbin
    ax = fig.add_subplot(gs[0, 0])
    x = pd.to_numeric(df["vdd"])
    y = pd.to_numeric(df["cs"]) * 1e15
    hb = ax.hexbin(x, y, gridsize=34, cmap="viridis", mincnt=1, yscale="log")
    ax.set_xlabel("$V_{DD}$ (V)")
    ax.set_ylabel("$C_s$ (fF)")
    cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("count", fontsize=11)
    plabel(ax, "(a) Design-space coverage")

    # (b) response distributions -- three stacked panels, each in its OWN unit,
    # because overlaying log10(s), log10(ns) and log10(nW) on one axis makes the
    # relative positions meaningless.
    specs = [("retention_time_s", CAT[0], "$\\log_{10}$ retention (s)"),
             ("read_delay_ns", CAT[3], "$\\log_{10}$ read delay (ns)"),
             ("total_power_nw", CAT[5], "$\\log_{10}$ power (nW)")]
    inner = gs[0, 1].subgridspec(3, 1, hspace=0.55)
    for row, (col, c, lab) in enumerate(specs):
        axr = fig.add_subplot(inner[row, 0])
        v = pd.to_numeric(df[col], errors="coerce").dropna()
        v = v[v > 0]
        axr.hist(np.log10(v), bins=50, histtype="stepfilled", alpha=0.55, color=c,
                 density=True)
        axr.set_ylabel("density", fontsize=9)
        axr.tick_params(labelsize=9)
        axr.set_xlabel(lab, fontsize=10)
        if row == 0:
            plabel(axr, "(b) Response distributions")

    # (c) Spearman correlation heatmap. Temperature is included as the first row:
    # without it the near-zero retention correlations of the design variables are
    # an artefact of leaving out the variable that moves leakage by three decades.
    ax = fig.add_subplot(gs[0, 2])
    rowvars = ["temperature_c", "vdd", "vwl_boost", "vwl_low", "vblpre_ratio",
               "cs", "cbl_ratio", "wacc", "lacc", "wsan", "wsap_ratio",
               "wen_ratio", "t_write_pulse", "t_sense_delay"]
    rowvars = [v for v in rowvars if v in df.columns]
    resps = ["retention_time_s", "read_delay_ns", "read_margin_mv",
             "total_power_nw", "write_efficiency"]
    sub = df[rowvars + resps].apply(pd.to_numeric, errors="coerce")
    corr = sub.corr(method="spearman").loc[rowvars, resps].values
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(resps)))
    ax.set_xticklabels(["retention", "read delay", "read margin", "power",
                        "write eff."], rotation=35, ha="right", fontsize=10)
    ax.set_yticks(range(len(rowvars)))
    ax.set_yticklabels([flab(v) for v in rowvars], fontsize=9.5)
    ax.axhline(0.5, color=INK, lw=1.0)          # separate temperature from design vars
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Spearman $\\rho$", fontsize=11)
    plabel(ax, "(c) Correlation (with temperature)")

    save(fig, name)


# ======================================================================
def fig_parity(name="fig06_parity"):
    """Six-panel digital-twin parity plots on the held-out test partition."""
    pred = pd.read_parquet(SUR / "predictions_main.parquet")
    lb = pd.read_csv(SUR / "leaderboard_main.csv")
    order = ["log10_retention_time_s", "read_delay_ns", "read_margin_mv",
             "log10_total_power_nw", "energy_per_access_fj", "write_efficiency"]
    letters = "abcdef"
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.2))
    for k, tgt in enumerate(order):
        ax = axes.flat[k]
        sub = pred[pred["target"] == tgt]
        yt = sub["y_true"].to_numpy()
        yp = sub["y_pred"].to_numpy()
        lo, hi = np.nanpercentile(np.concatenate([yt, yp]), [0.3, 99.7])
        ax.scatter(yt, yp, s=7, alpha=0.28, color=CAT[k % len(CAT)],
                   edgecolors="none", rasterized=True)
        ax.plot([lo, hi], [lo, hi], color=INK, lw=1.6, ls="--")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        row = lb[lb["target"] == tgt].sort_values("cv_r2_mean").iloc[-1]
        ax.text(0.05, 0.93, f"$R^2$ = {row['test_r2']:.3f}\nRMSE = {row['test_rmse']:.3g}",
                transform=ax.transAxes, va="top", fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCCCCC"))
        ax.set_xlabel("SPICE")
        ax.set_ylabel("digital twin")
        plabel(ax, f"({letters[k]}) {NICE[tgt]}")
    save(fig, name)


# ======================================================================
def fig_surrogate(name="fig07_surrogate"):
    """(a) model ranking, (b) selected-model R^2 per response, (c) twin speedup."""
    lb = pd.read_csv(SUR / "leaderboard_main.csv")
    bench = pd.read_csv(SUR / "twin_benchmark_main.csv")
    fig = plt.figure(figsize=(15, 4.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.15, 1.0])

    # (a) model median test R2 across responses
    ax = fig.add_subplot(gs[0, 0])
    med = lb.groupby("model")["test_r2"].median().sort_values()
    names = {"mlp_deep": "MLP (deep)", "mlp": "MLP", "catboost": "CatBoost",
             "lightgbm": "LightGBM", "xgboost": "XGBoost", "hist_gbr": "Hist-GBR",
             "poly2_ridge": "Poly2-ridge", "gpr": "GPR", "extra_trees": "Extra-trees",
             "svr_rbf": "SVR-RBF", "random_forest": "Rand-forest", "knn": "$k$-NN",
             "ridge": "Ridge", "elasticnet": "Elastic-net"}
    cols = [CAT[1] if m in ("mlp_deep", "mlp") else CAT[7] for m in med.index]
    ax.barh(range(len(med)), med.values, color=cols)
    ax.set_yticks(range(len(med)))
    ax.set_yticklabels([names.get(m, m) for m in med.index], fontsize=11)
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("median test $R^2$ across responses")
    plabel(ax, "(a) Model ranking")

    # (b) selected model test R2 per response with CI
    ax = fig.add_subplot(gs[0, 1])
    order = ["log10_retention_time_s", "read_delay_ns", "read_margin_mv",
             "log10_total_power_nw", "energy_per_access_fj", "write_efficiency"]
    sel = lb.sort_values("cv_r2_mean").groupby("target").tail(1).set_index("target")
    vals, los, his, labs = [], [], [], []
    for t in order:
        r = sel.loc[t]
        vals.append(r["test_r2"])
        los.append(r["test_r2"] - r.get("test_r2_ci_low", r["test_r2"]))
        his.append(r.get("test_r2_ci_high", r["test_r2"]) - r["test_r2"])
        labs.append(NICE[t].split("  ")[0])
    ax.barh(range(len(order)), vals, xerr=[los, his], color=CAT[3],
            error_kw=dict(ecolor=INK, lw=1.2, capsize=3))
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(labs, fontsize=11)
    ax.set_xlim(0.9, 1.0)
    for i, v in enumerate(vals):                     # exact value so the truncation misleads no one
        ax.text(min(v, 0.998), i, f" {v:.3f}", va="center", ha="right",
                color="white", fontsize=9.5, fontweight="bold")
    ax.text(0.02, 0.02, "$R^2$ axis truncated at 0.90", transform=ax.transAxes,
            fontsize=9, style="italic", color=INK)
    ax.set_xlabel("test $R^2$ (95% CI)")
    ax.invert_yaxis()
    plabel(ax, "(b) Selected twin per response")

    # (c) twin speedup vs batch size
    ax = fig.add_subplot(gs[0, 2])
    ax.loglog(bench["batch_size"], bench["speedup_vs_spice"], "-o", color=CAT[5],
              ms=8, lw=2.2)
    for _, r in bench.iterrows():
        ax.annotate(f"{r['speedup_vs_spice']:.0f}$\\times$",
                    (r["batch_size"], r["speedup_vs_spice"]),
                    textcoords="offset points", xytext=(0, 9), fontsize=10.5,
                    ha="center")
    ax.set_xlabel("batch size")
    ax.set_ylabel("speed-up over SPICE")
    plabel(ax, "(c) Twin evaluation cost")

    save(fig, name)


# ======================================================================
def _beeswarm(ax, shap, data, feats, topn=10):
    order = np.argsort(np.abs(shap).mean(0))[::-1][:topn]
    for row, j in enumerate(order):
        s = shap[:, j]
        v = data[:, j]
        vn = (v - np.nanmin(v)) / (np.nanmax(v) - np.nanmin(v) + 1e-12)
        yj = row + (np.random.RandomState(row).rand(len(s)) - 0.5) * 0.6
        sc = ax.scatter(s, yj, c=vn, cmap="coolwarm", s=10, alpha=0.6,
                        edgecolors="none", rasterized=True)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([flab(feats[j]) for j in order], fontsize=10.5)
    ax.invert_yaxis()
    return sc, order


def fig_xai(name="fig08_xai"):
    """(a) SHAP beeswarm for retention, (b) importance heatmap, (c) PDP+ALE."""
    npz = np.load(XAI / "shap_values_log10_retention_time_s.npz", allow_pickle=True)
    imp = pd.read_csv(XAI / "shap_importance_all_main.csv")
    fig = plt.figure(figsize=(15, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.25, 1.0])

    # (a) beeswarm
    ax = fig.add_subplot(gs[0, 0])
    sc, _ = _beeswarm(ax, npz["shap_values"], npz["data"], npz["features"])
    ax.set_xlabel("SHAP value (effect on $\\log_{10} t_{ret}$)")
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("feature value", fontsize=11)
    plabel(ax, "(a) SHAP attribution: retention")

    # (b) importance heatmap: responses x top features (by mean importance)
    ax = fig.add_subplot(gs[0, 1])
    piv = imp.pivot_table(index="feature", columns="response",
                          values="importance_pct", aggfunc="mean").fillna(0)
    piv["__m"] = piv.mean(1)
    piv = piv.sort_values("__m", ascending=False).drop(columns="__m").head(12)
    rcols = ["log10_retention_time_s", "read_delay_ns", "read_margin_mv",
             "log10_total_power_nw", "energy_per_access_fj", "write_efficiency"]
    rcols = [c for c in rcols if c in piv.columns]
    M = piv[rcols].values
    im = ax.imshow(M, cmap="magma", aspect="auto")
    ax.set_xticks(range(len(rcols)))
    ax.set_xticklabels([NICE[c].split("  ")[0] for c in rcols], rotation=35,
                       ha="right", fontsize=10)
    ax.set_yticks(range(len(piv)))
    ax.set_yticklabels([flab(f) for f in piv.index], fontsize=10)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("importance (%)", fontsize=11)
    plabel(ax, "(b) Attribution across responses")

    # (c) PDP + ALE for the top retention driver
    ax = fig.add_subplot(gs[0, 2])
    pdp = pd.read_csv(XAI / "partial_dependence_log10_retention_time_s.csv")
    ale = pd.read_csv(XAI / "ale_log10_retention_time_s.csv")
    feat = pdp["feature"].value_counts().index[0]
    p = pdp[pdp["feature"] == feat]
    a = ale[ale["feature"] == feat]
    # both curves centred on their own mean so they share a baseline and the
    # divergence that is read off is real, not an artefact of an offset (F7).
    ax.plot(p["value"], p["partial_dependence"] - p["partial_dependence"].mean(),
            color=CAT[3], lw=2.4, label="PDP (centred)")
    ax.plot(a["value"], a["ale"] - a["ale"].mean(), color=CAT[0], lw=2.4,
            ls="--", label="ALE (centred)")
    ax.set_xlabel(flab(feat))
    ax.set_ylabel("centred effect on $\\log_{10} t_{ret}$")
    ax.legend()
    plabel(ax, "(c) Marginal effect (PDP vs ALE)")

    save(fig, name)


# ======================================================================
def _cliques(rvals, cd):
    """Maximal runs of consecutive (rank-sorted) methods spanning <= CD.

    A clique bar connects methods that are NOT significantly different. Returns
    (i, j) index pairs into the rank-sorted order, dropping any run contained in
    another so no redundant bar is drawn.
    """
    n = len(rvals)
    runs = []
    for i in range(n):
        j = i
        while j + 1 < n and (rvals[j + 1] - rvals[i]) <= cd + 1e-9:
            j += 1
        if j > i:
            runs.append((i, j))
    maximal = [(i, j) for (i, j) in runs
               if not any(a <= i and j <= b and (a, b) != (i, j) for (a, b) in runs)]
    return maximal


def _cd_diagram(ax, ranks, cd, title):
    """Critical-difference diagram (lower rank = better) with clique bars (M10)."""
    ranks = ranks.sort_values("mean_rank")
    names = [ashort(g) for g in ranks["group"].tolist()]
    rvals = ranks["mean_rank"].tolist()
    keys = ranks["group"].tolist()
    n = len(names)
    lo, hi = 1, n
    ax.set_xlim(lo - 0.5, hi + 0.7)
    ax.set_ylim(-0.5, n + 2.2)
    ax.axis("off")
    top = n + 1
    ax.plot([lo, hi], [top, top], color=INK, lw=1.5)
    for r in range(lo, hi + 1):
        ax.plot([r, r], [top, top + 0.15], color=INK, lw=1.2)
        ax.text(r, top + 0.32, str(r), ha="center", fontsize=11)
    for i, (nm, rv, ky) in enumerate(zip(names, rvals, keys)):
        y = n - i
        ax.plot([rv, rv], [y, top], color=acolor(ky), lw=1.3)
        ax.plot(rv, y, "o", color=acolor(ky), ms=9)
        ax.text(rv + 0.12, y, f"{nm} ({rv:.2f})", va="center", fontsize=11)
    # clique bars: groups NOT significantly different (span <= CD)
    cliqs = _cliques(rvals, cd)
    for b, (i, j) in enumerate(cliqs):
        yb = 0.35 + 0.32 * b
        ax.plot([rvals[i] - 0.06, rvals[j] + 0.06], [yb, yb], color=INK, lw=3.5,
                solid_capstyle="round")
    # CD reference length
    ax.plot([lo, lo + cd], [top + 0.75, top + 0.75], color=INK, lw=2.5)
    ax.text(lo + cd / 2, top + 0.92, f"CD = {cd:.2f}", ha="center", fontsize=11)
    ax.set_title(title, loc="left", fontweight="bold", fontsize=15)


def fig_optimization(name="fig09_optimization"):
    """(a) tradeoff, (b) convergence, (c) HV boxplot, (d) CD diagram."""
    pf = pd.read_csv(OPT / "pareto_main.csv")
    conv = pd.read_csv(OPT / "convergence_main.csv")
    ind = pd.read_csv(OPT / "indicators_main.csv")
    ranks = pd.read_csv(STAT / "optimizer_hypervolume_ranks.csv")
    post = pd.read_csv(STAT / "optimizer_hypervolume_posthoc.csv")
    cd = float(post["critical_difference"].iloc[0])

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.5))

    # (a) retention-delay projection of the 4-D front, colour PER ALGORITHM (F5),
    # with the 2-D non-dominated subset drawn so a real trade-off is visible (F3).
    ax = axes[0, 0]
    for a in sorted(pf["found_by"].unique()):
        s = pf[pf["found_by"] == a]
        ax.scatter(s["retention_time_s"] * 1e3, s["read_delay_ns"], s=20,
                   alpha=0.5, color=acolor(a), label=ashort(a), edgecolors="none")
    xy = pf[["retention_time_s", "read_delay_ns"]].apply(pd.to_numeric, errors="coerce").dropna()
    xy = xy.to_numpy()
    ret, dly = xy[:, 0], xy[:, 1]
    nd = np.ones(len(xy), bool)                 # maximise retention, minimise delay
    for i in range(len(xy)):
        nd[i] = not np.any((ret >= ret[i]) & (dly <= dly[i]) &
                           ((ret > ret[i]) | (dly < dly[i])))
    f = xy[nd]
    f = f[np.argsort(f[:, 0])]
    ax.plot(f[:, 0] * 1e3, f[:, 1], "-", color=INK, lw=2.4, zorder=5,
            label="2-D non-dominated")
    ax.scatter(f[:, 0] * 1e3, f[:, 1], s=42, facecolor="white", edgecolor=INK,
               linewidths=1.4, zorder=6)
    ax.set_xscale("log")
    ax.set_xlabel("retention time (ms)")
    ax.set_ylabel("read delay (ns)")
    ax.legend(fontsize=9, ncol=2, loc="upper right")
    plabel(ax, "(a) Retention--delay projection")

    # (b) convergence: median HV vs evaluations, colour PER ALGORITHM, legend OUT
    ax = axes[0, 1]
    for lab in sorted(conv["label"].unique()):
        s = conv[conv["label"] == lab]
        g = s.groupby("n_evaluations")["hypervolume"].median()
        ax.plot(g.index, g.values, lw=2.2, color=acolor(lab), label=ashort(lab))
    ax.set_xlabel("design evaluations")
    ax.set_ylabel("normalized hypervolume (median)")
    ax.legend(fontsize=9, ncol=1, loc="center left", bbox_to_anchor=(1.02, 0.5),
              frameon=False)
    plabel(ax, "(b) Hypervolume convergence")

    # (c) HV boxplot by algorithm, colour PER ALGORITHM
    ax = axes[1, 0]
    labs = (ind.groupby("label")["hypervolume"].median().sort_values(ascending=False)
            .index.tolist())
    data = [ind[ind["label"] == l]["hypervolume"].values for l in labs]
    bp = ax.boxplot(data, vert=True, patch_artist=True, widths=0.6)
    for box, l in zip(bp["boxes"], labs):
        box.set(facecolor=acolor(l), alpha=0.6)
    for med in bp["medians"]:
        med.set(color=INK, lw=1.5)
    ax.set_xticklabels([ashort(l) for l in labs], rotation=30, ha="right",
                       fontsize=10)
    ax.set_ylabel("normalized hypervolume")
    plabel(ax, "(c) Hypervolume by algorithm")

    # (d) CD diagram with clique bars
    _cd_diagram(axes[1, 1], ranks, cd, "(d) Critical-difference (hypervolume)")

    save(fig, name)


# ======================================================================
def fig_robustness(name="fig10_robustness"):
    """(a) corner heatmap, (b) Monte-Carlo distribution, (c) Sobol indices."""
    cs = pd.read_parquet(ROB / "corner_sweep_main.parquet")
    mc = pd.read_parquet(ROB / "monte_carlo_best_retention_time_s_2.parquet")
    sob = pd.read_csv(ROB / "sobol_main.csv")
    fig = plt.figure(figsize=(15, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.05, 1.2])

    # (a) corner heatmap of retention (s) for knee design, log colour scale so
    # the hot-corner worst case (tens of ms) stays visible next to the cold-
    # corner best case (tens of s)
    from matplotlib.colors import LogNorm
    ax = fig.add_subplot(gs[0, 0])
    k = cs[cs["design_id"] == "knee_0"].copy()
    k["ret_s"] = pd.to_numeric(k["retention_time_s"])
    piv = k.pivot_table(index="corner", columns="temperature_c", values="ret_s")
    corners = ["FF", "FS", "TT", "SF", "SS"]
    piv = piv.reindex([c for c in corners if c in piv.index])
    vals = np.clip(piv.values, 1e-3, None)
    im = ax.imshow(vals, cmap="viridis", aspect="auto",
                   norm=LogNorm(vmin=np.nanmin(vals), vmax=np.nanmax(vals)))
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([f"{c:.0f}" for c in piv.columns], fontsize=9)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index)
    # annotate each cell so the reader sees that corner is second-order while
    # temperature dominates (M1) -- the colour alone hides the corner spread.
    for r in range(vals.shape[0]):
        for c in range(vals.shape[1]):
            x = vals[r, c]
            txt = f"{x:.1f}" if x >= 1 else (f"{x*1e3:.0f}m" if x >= 1e-3 else "-")
            ax.text(c, r, txt, ha="center", va="center", fontsize=7,
                    color="white" if x < np.nanmax(vals) * 0.15 else "black")
    ax.set_xlabel("temperature ($^\\circ$C)")
    ax.set_ylabel("process corner")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("retention (s, log)", fontsize=11)
    plabel(ax, "(a) Corner sweep: knee design")

    # (b) Monte-Carlo retention distribution, with the training range marked so
    # out-of-range extrapolation is visible (M7).
    ax = fig.add_subplot(gs[0, 1])
    v = pd.to_numeric(mc["retention_time_s"])
    ax.hist(v, bins=45, color=CAT[0], alpha=0.55, density=True)
    ax.axvline(v.median(), color=INK, lw=1.8, ls="--",
               label=f"median {v.median():.1f} s")
    TRAIN_MAX = 25.83                                # Table: max characterized retention
    if v.max() > TRAIN_MAX:
        ax.axvspan(TRAIN_MAX, v.max() * 1.02, color="#8E0049", alpha=0.12)
        ax.axvline(TRAIN_MAX, color="#8E0049", lw=1.6, ls=":",
                   label=f"training max {TRAIN_MAX:.1f} s")
    ax.set_xlabel("retention time (s)")
    ax.set_ylabel("density")
    ax.legend(fontsize=9.5)
    plabel(ax, "(b) Monte-Carlo: best-retention design")

    # (c) Sobol total-effect indices for retention with bootstrap CIs (B3, M14).
    ax = fig.add_subplot(gs[0, 2])
    resp = "retention_time_s"
    s = sob[sob["response"] == resp].nlargest(6, "ST").reset_index(drop=True)
    yy = np.arange(len(s))
    st_err = None
    if {"ST_ci_low", "ST_ci_high"}.issubset(s.columns):
        st_err = [np.clip(s["ST"].values - s["ST_ci_low"].values, 0, None),
                  np.clip(s["ST_ci_high"].values - s["ST"].values, 0, None)]
    s1_err = None
    if {"S1_ci_low", "S1_ci_high"}.issubset(s.columns):
        s1_err = [np.clip(s["S1"].values - s["S1_ci_low"].values, 0, None),
                  np.clip(s["S1_ci_high"].values - s["S1"].values, 0, None)]
    ax.barh(yy + 0.19, s["ST"].values, height=0.36, color=CAT[3],
            xerr=st_err, error_kw=dict(ecolor=INK, lw=1.0, capsize=2),
            label="$S_T$ total")
    ax.barh(yy - 0.19, s["S1"].values, height=0.36, color=CAT[1],
            xerr=s1_err, error_kw=dict(ecolor=INK, lw=1.0, capsize=2),
            label="$S_1$ first-order")
    ax.set_yticks(list(yy))
    ax.set_yticklabels([flab(x) for x in s["variable"].tolist()], fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("Sobol' index (retention)")
    ax.legend(loc="lower right")
    plabel(ax, "(c) Sensitivity (Sobol')")

    save(fig, name)


# ======================================================================
def fig_assist(name="fig11_assist"):
    """Assist-technique relative effect against the common baseline."""
    rel = pd.read_csv(TAB / "assist_relative_effect_main.csv")
    # leakage is included (M12): the negative-wordline row raises it strongly, and
    # that is a headline claim in the text that had no series before.
    metrics = [("retention_time_s", "retention"), ("read_margin_mv", "read margin"),
               ("read_delay_ns", "read delay"), ("total_power_nw", "power"),
               ("leakage_current_fa", "leakage")]
    rel = rel.set_index("assist_config")
    order = ["wordline_boost", "bitline_precharge_opt", "sense_amp_upsize",
             "negative_wordline", "high_k_capacitor", "combined_low_power",
             "refresh_interval_opt"]
    order = [o for o in order if o in rel.index]
    nice = {"wordline_boost": "wordline boost",
            "bitline_precharge_opt": "bitline precharge opt.",
            "sense_amp_upsize": "sense-amp upsize",
            "negative_wordline": "negative wordline",
            "high_k_capacitor": "high-$k$ capacitor",
            "combined_low_power": "combined low-power",
            "refresh_interval_opt": "refresh opt. (control)"}
    fig, ax = plt.subplots(figsize=(13.0, 5.8))
    n = len(metrics)
    yb = np.arange(len(order))
    h = 0.82 / n
    CLIP = 400
    for i, (col, lab) in enumerate(metrics):
        raw = rel.loc[order, col].values.astype(float)
        vals = np.clip(raw, -100, CLIP)
        yy = yb + (i - (n - 1) / 2) * h
        ax.barh(yy, vals, height=h, color=CAT[i], label=lab)
        for y, rv, cv in zip(yy, raw, vals):        # annotate control zeros + clipped bars (M12)
            if abs(rv) < 0.05:
                ax.text(1.5, y, "0.0%", va="center", ha="left", fontsize=6.5,
                        color=INK)
            elif rv > CLIP:
                ax.text(CLIP - 3, y, f"{rv:.0f}%", va="center", ha="right",
                        fontsize=6.5, color="white", fontweight="bold")
    ax.axvline(0, color=INK, lw=1.0)
    ax.set_yticks(yb)
    ax.set_yticklabels([nice[o] for o in order], fontsize=12)
    ax.invert_yaxis()
    ax.set_xlabel("median relative change vs baseline (%), clipped to [-100, 400]")
    ax.legend(ncol=5, loc="lower right", fontsize=10)
    plabel(ax, "Assist techniques: paired effect against a common baseline")
    save(fig, name)


# ======================================================================
def _spice_setup():
    from dramdt.config import load_config
    from dramdt.env import resolve_environment
    from dramdt.models import DramCellBuilder, Technology
    from dramdt.simulation import MetricExtractor
    cfg = load_config("dram_1t1c_45nm_lp.yaml")
    env = resolve_environment(probe=False)
    tech = Technology.from_config(cfg, generate=False)
    builder = DramCellBuilder(cfg, tech)
    extractor = MetricExtractor(cfg)
    return cfg, env, builder, extractor


def fig_waveforms(name="fig03_waveforms"):
    """The characterisation sequence captured from a live NGSpice transient."""
    cfg, env, builder, _ = _spice_setup()
    ds = cfg.design_space
    sample = ds.decode_row([0.5] * ds.n_dim)
    sample.update({"vdd": 1.1, "vwl_boost": 0.95, "vwl_low": 0.0,
                   "vblpre_ratio": 0.5, "cs": 20e-15, "cbl_ratio": 5.0,
                   "temperature_c": 27.0, "corner": "TT",
                   "t_write_pulse": 6e-9, "t_sense_delay": 6e-9})
    p = builder.parameters_from_sample(ds.apply_derived(sample), 0)
    t = builder.schedule(p)
    deck = builder.build_characterization_netlist(p)
    deck = deck.replace("wrdata leak.csv i(Vsnp)",
                        "wrdata leak.csv i(Vsnp)\nsetplot tran1\n"
                        "wrdata wave.csv v(wl) v(bl) v(blb) v(sn) v(saen) v(preg)")
    wd = Path(tempfile.mkdtemp(prefix="dramdt_wave_"))
    (wd / "d.cir").write_text(deck, encoding="utf-8")
    subprocess.run([str(env.exe), "-b", "-o", "d.log", "d.cir"], cwd=str(wd),
                   capture_output=True, text=True, timeout=600,
                   stdin=subprocess.DEVNULL)
    rows = []
    for line in (wd / "wave.csv").read_text().splitlines():
        parts = line.split()
        if len(parts) >= 12:
            try:
                rows.append([float(x) for x in parts])
            except ValueError:
                pass
    A = np.asarray(rows)
    tt = A[:, 0] * 1e9
    wl, bl, blb, sn, saen, preg = (A[:, 1], A[:, 3], A[:, 5], A[:, 7],
                                   A[:, 9], A[:, 11])
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(13, 8.2))
    ax = axes[0]
    ax.plot(tt, wl, color=CAT[0], lw=2.4, label="WL")
    ax.plot(tt, preg, color=CAT[3], lw=2.0, ls="--", label="PRE")
    ax.plot(tt, saen, color=CAT[1], lw=2.0, ls="-.", label="SAE")
    ax.set_ylabel("control (V)")
    ax.set_ylim(-0.15, max(wl.max(), preg.max()) * 1.5)
    ax.legend(ncol=3, loc="upper right")
    plabel(ax, "(a) Control signals: wordline, precharge, sense enable")
    ax = axes[1]
    ax.plot(tt, sn, color=CAT[4], lw=2.6, label="$V_{SN}$")
    ax.axhline(p.vplate, color=INK, lw=1.0, ls=":")
    ax.set_ylabel("storage node (V)")
    ax.legend(loc="lower right")
    plabel(ax, "(b) Storage node: write, hold, charge-share, restore")
    ax = axes[2]
    ax.plot(tt, bl, color=CAT[0], lw=2.4, label="BL")
    ax.plot(tt, blb, color=CAT[5], lw=2.0, ls="--", label=r"$\overline{BL}$")
    ax.set_ylabel("bitlines (V)")
    ax.set_xlabel("time (ns)")
    ax.legend(loc="center right")
    plabel(ax, "(c) Bitline pair: charge-share differential then latch")
    phases = [(0, t.pre1_end, "precharge"), (t.wr_on, t.wr_off, "write '1'"),
              (t.pre2_on, t.pre2_end, "precharge"),
              (t.rd_wl, t.sa, "charge share"), (t.sa, t.stop, "sense+restore")]
    for a in axes:
        for kk, (t0, t1, _l) in enumerate(phases):
            a.axvspan(t0 * 1e9, t1 * 1e9, color=INK,
                      alpha=0.05 if kk % 2 == 0 else 0.10, lw=0)
    for t0, t1, lab in phases:
        axes[0].text((t0 + t1) / 2 * 1e9, axes[0].get_ylim()[1] * 0.82, lab,
                     fontsize=10.5, ha="center", color=INK)
    save(fig, name)
    shutil.rmtree(wd, ignore_errors=True)


def fig_retention(name="fig04_retention"):
    """(a-c) measured leakage + integrand; (d-e) quasi-static model validation."""
    from dramdt.simulation import SpiceRunner
    cfg, env, builder, extractor = _spice_setup()
    ds = cfg.design_space
    base = ds.decode_row([0.5] * ds.n_dim)
    base.update({"vdd": 1.1, "vwl_boost": 0.95, "vwl_low": 0.0,
                 "vblpre_ratio": 0.5, "cs": 20e-15, "cbl_ratio": 5.0,
                 "corner": "TT"})
    temps = [-40.0, -10.0, 27.0, 55.0, 85.0, 125.0]
    arr_T, arr_I = [], []

    fig = plt.figure(figsize=(15, 9.0))
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 1])
    axa = fig.add_subplot(gs[0, 0:2])
    axb = fig.add_subplot(gs[0, 2:4])
    axc = fig.add_subplot(gs[0, 4:6])
    axd = fig.add_subplot(gs[1, 0:3])
    axe = fig.add_subplot(gs[1, 3:6])

    with SpiceRunner(env, timeout_s=600) as runner:
        for i, T in enumerate(temps):
            s = dict(base)
            s["temperature_c"] = T
            p = builder.parameters_from_sample(ds.apply_derived(s), i)
            res = runner.run(builder.build_characterization_netlist(p), tag="lk")
            met = extractor.extract(res, p)
            if not res.has_leakage_curve:
                continue
            v, cur = res.leakage_v, res.leakage_i
            m = cur > 0
            axa.semilogy(v[m], cur[m] * 1e15, lw=2.2, color=CAT[i % len(CAT)],
                         marker="o", ms=3.5, markevery=6, label=f"{T:.0f}")
            arr_T.append(T + 273.15)
            arr_I.append(float(np.interp(met.v_sn_written, v, cur)) * 1e15)
            lo, hi = met.v_fail, met.v_sn_written
            g = np.linspace(lo, hi, 400)
            ig = np.interp(g, v, cur)
            ok = ig > 0
            axc.semilogy(g[ok], p.cs / ig[ok], lw=2.2, color=CAT[i % len(CAT)])

    axa.set_xlabel("storage-node voltage $V_{SN}$ (V)")
    axa.set_ylabel("leakage leaving node (fA)")
    axa.legend(title="$T$ ($^\\circ$C)", ncol=2, fontsize=10, title_fontsize=10.5)
    plabel(axa, "(a) Measured $I_{leak}(V_{SN})$")

    kB = 8.617333262e-5
    x = 1.0 / (kB * np.asarray(arr_T))
    y = np.log10(np.asarray(arr_I))
    axb.plot(x, arr_I, color=CAT[0], marker="o", ms=7, lw=2.2)
    axb.set_yscale("log")
    axb.invert_xaxis()
    axb.set_xlabel("$1/k_BT$ (eV$^{-1}$)")
    axb.set_ylabel("$I_{leak}$ at written level (fA)")
    if len(arr_T) >= 2:
        Ea = (y[-1] - y[-2]) * np.log(10) / (x[-2] - x[-1])
        axb.annotate(f"thermally activated\n$E_a \\approx$ {Ea:.2f} eV",
                     xy=(x[-1], arr_I[-1]), xytext=(0.36, 0.55),
                     textcoords="axes fraction", fontsize=11, color=CAT[0],
                     arrowprops=dict(arrowstyle="->", color=CAT[0], lw=1.2))
        axb.annotate("tunnelling limited", xy=(x[0], arr_I[0]),
                     xytext=(0.06, 0.16), textcoords="axes fraction",
                     fontsize=11, color=CAT[3],
                     arrowprops=dict(arrowstyle="->", color=CAT[3], lw=1.2))
    plabel(axb, "(b) Arrhenius behaviour")

    axc.set_xlabel("storage-node voltage $V_{SN}$ (V)")
    axc.set_ylabel("$C_{node}/I_{leak}$ (s V$^{-1}$)")
    plabel(axc, "(c) Retention integrand")

    # (d,e) quasi-static vs transient validation
    val = pd.read_csv(TAB / "retention_validation_main.csv")
    q = pd.to_numeric(val["retention_quasistatic_s"])
    tr = pd.to_numeric(val["retention_transient_s"])
    ok = (q > 0) & (tr > 0)
    q, tr = q[ok], tr[ok]
    lo = min(q.min(), tr.min()) * 0.6
    hi = max(q.max(), tr.max()) * 1.6
    axd.loglog([lo, hi], [lo, hi], color=INK, lw=1.6, ls="--")
    axd.fill_between([lo, hi], [lo * 0.5, hi * 0.5], [lo * 2, hi * 2],
                     color=CAT[1], alpha=0.15, label="factor of 2")
    axd.scatter(tr, q, s=26, color=CAT[0], alpha=0.6, edgecolors="none")
    axd.set_xlim(lo, hi)
    axd.set_ylim(lo, hi)
    r = np.corrcoef(np.log10(tr), np.log10(q))[0, 1]
    within2 = float((np.abs(np.log10(q / tr)) <= np.log10(2)).mean() * 100)
    axd.text(0.05, 0.93, f"$r$ = {r:.5f}\nwithin 2$\\times$: {within2:.0f}%",
             transform=axd.transAxes, va="top", fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCCCCC"))
    axd.set_xlabel("direct transient retention (s)")
    axd.set_ylabel("quasi-static model (s)")
    axd.legend(loc="lower right")
    plabel(axd, "(d) Model vs direct transient simulation")

    err = np.log10(q / tr)
    axe.hist(err, bins=35, color=CAT[3], alpha=0.6, density=True)
    axe.axvline(0, color=INK, lw=1.2)
    axe.axvline(np.median(err), color=CAT[0], lw=2.0, ls="--",
                label=f"median {np.median(err):+.4f} dec")
    axe.set_xlabel("$\\log_{10}$(model / transient)  (decades)")
    axe.set_ylabel("density")
    axe.legend()
    plabel(axe, "(e) Error distribution")
    save(fig, name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-spice", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    funcs = {"dataset": fig_dataset, "parity": fig_parity,
             "surrogate": fig_surrogate, "xai": fig_xai,
             "optimization": fig_optimization, "robustness": fig_robustness,
             "assist": fig_assist}
    print("data-driven figures ...")
    for k, f in funcs.items():
        if args.only and k not in args.only:
            continue
        f()
    if not args.no_spice and (not args.only or "waveforms" in args.only
                              or "retention" in args.only):
        print("live-SPICE figures ...")
        if not args.only or "waveforms" in args.only:
            fig_waveforms()
        if not args.only or "retention" in args.only:
            fig_retention()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
