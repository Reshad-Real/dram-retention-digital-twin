#!/usr/bin/env python
"""Stage 12 -- figures that exist only to support the manuscript.

Three explanatory figures the analysis pipeline does not produce:

``figM1``  block schematic of the simulated 1T1C cell, its column periphery and
           the leakage replica
``figM2``  the characterisation waveform sequence, captured from a real NGSpice
           transient rather than drawn by hand
``figM3``  the measured storage-node leakage characteristic, its thermal
           behaviour, and the retention integrand derived from it

    python scripts/12_manuscript_figures.py
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dramdt.config import load_config                                   # noqa: E402
from dramdt.env import resolve_environment                              # noqa: E402
from dramdt.logging_utils import setup_logging                          # noqa: E402
from dramdt.models import DramCellBuilder, Technology                   # noqa: E402
from dramdt.paths import FIGURES_DIR, ensure_directories                # noqa: E402
from dramdt.simulation import MetricExtractor, SpiceRunner              # noqa: E402
from dramdt.viz import apply_style, figure_size, save_figure            # noqa: E402
from dramdt.viz.palette import CATEGORICAL, REFERENCE_INK               # noqa: E402


# ==========================================================================
def figure_schematic(path: Path):
    """Block schematic of the simulated slice.

    Only the access device and the replica are drawn at transistor level,
    because those are what the study varies and measures; the periphery is
    shown as labelled blocks on the bitline pair.  A full transistor-level
    rendering does not lay out on this canvas without wires and labels
    colliding, and the netlist construction in Section 3 specifies it exactly.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Rectangle

    fig, ax = plt.subplots(figsize=figure_size("page", height=3.6))
    ax.set_xlim(0, 104)
    ax.set_ylim(0, 70)
    ax.axis("off")
    ax.grid(False)
    K, LW = REFERENCE_INK, 1.2
    Y_BL, Y_BLB = 40.0, 16.0

    def wire(pts, color=K, lw=LW, ls="-"):
        p = np.asarray(pts, float)
        ax.plot(p[:, 0], p[:, 1], color=color, lw=lw, ls=ls,
                solid_capstyle="round", zorder=2)

    def dot(x, y):
        ax.plot([x], [y], marker="o", ms=3.0, color=K, zorder=5)

    def gnd(x, y):
        for i, w in enumerate((2.6, 1.7, 0.8)):
            ax.plot([x - w, x + w], [y - i * 1.2, y - i * 1.2], color=K, lw=LW)

    def mos(x, y, half=4.0, gate_left=True):
        """Draw a MOS symbol; returns the gate terminal coordinate."""
        s = -1.0 if gate_left else 1.0
        ax.plot([x, x], [y - half, y + half], color=K, lw=LW + 0.7, zorder=3)
        ax.plot([x + s * 2.0, x + s * 2.0], [y - half - 0.5, y + half + 0.5],
                color=K, lw=LW + 0.3, zorder=3)
        return (x + s * 2.0, y)

    def block(x, y, w, h, title, sub, colour):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4",
                                    fc="white", ec=colour, lw=1.1, zorder=3))
        ax.text(x + w / 2, y + h - 3.4, title, fontsize=6.8, ha="center",
                color=colour, fontweight="bold", zorder=4)
        for k, line in enumerate(sub):
            ax.text(x + w / 2, y + h - 7.2 - 3.4 * k, line, fontsize=5.9,
                    ha="center", zorder=4)

    # ---------------- bitline pair -----------------------------------
    X_SA = 78.0
    wire([[16, Y_BL], [X_SA, Y_BL]])
    wire([[46, Y_BLB], [X_SA, Y_BLB]])
    ax.text(14.5, Y_BL, "BL", fontsize=7.4, ha="right", va="center",
            fontweight="bold")
    ax.text(44.5, Y_BLB, r"$\overline{\mathrm{BL}}$", fontsize=7.4, ha="right",
            va="center", fontweight="bold")

    # ---------------- the 1T1C cell (transistor level) ---------------
    xc = 24.0
    g = mos(xc, Y_BL + 10)
    wire([[g[0], g[1]], [xc - 10, g[1]]])
    ax.text(xc - 10.8, g[1], "WL", fontsize=7.0, ha="right", va="center",
            fontweight="bold")
    wire([[xc, Y_BL + 6], [xc, Y_BL]])
    dot(xc, Y_BL)
    wire([[xc, Y_BL + 14], [xc, Y_BL + 18]])
    dot(xc, Y_BL + 18)
    ax.text(xc + 2.4, Y_BL + 17.4, "SN", fontsize=6.8, va="center",
            fontweight="bold")
    ax.text(xc + 3.0, Y_BL + 10, "$M_{acc}$", fontsize=6.6, va="center",
            style="italic")
    ax.plot([xc - 4.0, xc + 4.0], [Y_BL + 21, Y_BL + 21], color=K, lw=LW + 0.8)
    ax.plot([xc - 4.0, xc + 4.0], [Y_BL + 23, Y_BL + 23], color=K, lw=LW + 0.8)
    wire([[xc, Y_BL + 18], [xc, Y_BL + 21]])
    ax.text(xc + 5.4, Y_BL + 22, "$C_s$", fontsize=6.8, va="center",
            style="italic")
    wire([[xc, Y_BL + 23], [xc, Y_BL + 26]])
    ax.text(xc, Y_BL + 27.6, "$V_{plate}$", fontsize=6.4, ha="center")
    ax.add_patch(Rectangle((xc - 12, Y_BL + 3), 25, 23.5, fill=False,
                           ec=CATEGORICAL[0], lw=0.8, ls=(0, (4, 2)), zorder=1))
    ax.text(xc - 12, Y_BL + 27.6, "1T1C cell", fontsize=6.8,
            color=CATEGORICAL[0], fontweight="bold")

    # ---------------- write driver (left of the BLB rail) -------------
    block(26, Y_BL - 22, 18, 14, "write driver",
          [r"$M_{wn}\,\|\,M_{wp}$", r"WRT / $\overline{\mathrm{WRT}}$"],
          CATEGORICAL[5])
    wire([[35, Y_BL - 8], [35, Y_BL]])
    dot(35, Y_BL)
    wire([[26, Y_BL - 15], [20, Y_BL - 15]])
    ax.text(19, Y_BL - 15, "$V_{data}$", fontsize=6.4, ha="right", va="center")

    # ---------------- precharge / equalise (between the rails) --------
    x_pc = 59.0
    block(50, Y_BLB + 4, 18, 12, "precharge / equalise",
          ["$M_{pc1}, M_{pc2}, M_{eq}$", "gates at VPP"], CATEGORICAL[3])
    wire([[x_pc, Y_BLB + 16], [x_pc, Y_BL]])
    dot(x_pc, Y_BL)
    wire([[x_pc, Y_BLB + 4], [x_pc, Y_BLB]])
    dot(x_pc, Y_BLB)
    wire([[50, Y_BLB + 10], [44, Y_BLB + 10]], color=CATEGORICAL[3], lw=0.9)
    ax.text(43, Y_BLB + 10, "$V_{BLpre}$", fontsize=6.2, ha="right",
            va="center", color=CATEGORICAL[3])

    # ---------------- sense amplifier --------------------------------
    block(X_SA, Y_BLB - 1, 20, 30, "sense amplifier",
          ["cross-coupled", "NMOS / PMOS latch", "$M_{n1},M_{n2}$",
           "$M_{p1},M_{p2}$", r"enables SAE, $\overline{\mathrm{SAE}}$"],
          CATEGORICAL[1])
    dot(X_SA, Y_BL)
    dot(X_SA, Y_BLB)
    wire([[X_SA + 10, Y_BLB + 29], [X_SA + 10, Y_BLB + 34]])
    ax.text(X_SA + 10, Y_BLB + 35.8, "$V_{DD}$", fontsize=6.4, ha="center")
    wire([[X_SA + 10, Y_BLB - 1], [X_SA + 10, Y_BLB - 5]])
    gnd(X_SA + 10, Y_BLB - 5)

    # ---------------- bitline capacitance (one per rail) --------------
    for y, sgn in ((Y_BL, +1), (Y_BLB, -1)):
        xcap = 72.0
        ax.plot([xcap - 2.6, xcap + 2.6], [y + sgn * 6.5, y + sgn * 6.5],
                color=K, lw=LW + 0.6)
        ax.plot([xcap - 2.6, xcap + 2.6], [y + sgn * 8.2, y + sgn * 8.2],
                color=K, lw=LW + 0.6)
        wire([[xcap, y], [xcap, y + sgn * 6.5]])
        dot(xcap, y)
        wire([[xcap, y + sgn * 8.2], [xcap, y + sgn * 11]])
        if sgn > 0:
            for i, w in enumerate((2.6, 1.7, 0.8)):
                ax.plot([xcap - w, xcap + w], [y + 11 + i * 1.2, y + 11 + i * 1.2],
                        color=K, lw=LW)
        else:
            gnd(xcap, y - 11)
        ax.text(xcap + 3.4, y + sgn * 7.3, "$C_{BL}$", fontsize=6.4, va="center")

    # ---------------- leakage replica --------------------------------
    bx, by = 4, 1
    ax.add_patch(Rectangle((bx, by), 64, 13, fill=False, ec=CATEGORICAL[4],
                           lw=1.0, ls=(0, (4, 2))))
    ax.text(bx + 1.5, by + 10.2, "leakage replica: an access device identical to "
            "$M_{acc}$, storage node driven by a probe source",
            fontsize=6.3, color=CATEGORICAL[4], fontweight="bold")
    xr = bx + 15
    gr = mos(xr, by + 4.6, half=3.0)
    wire([[gr[0], gr[1]], [bx + 7, gr[1]]])
    ax.text(bx + 6, gr[1], "$V_{WL,low}$", fontsize=6.2, ha="right", va="center")
    wire([[xr, by + 7.6], [xr + 8, by + 7.6]])
    ax.text(xr + 9, by + 7.6, "$V_{BLpre}$", fontsize=6.2, va="center")
    wire([[xr, by + 1.6], [xr + 17, by + 1.6]])
    ax.add_patch(plt.Circle((xr + 20, by + 1.6), 2.6, fill=False,
                            ec=CATEGORICAL[4], lw=1.0))
    ax.text(xr + 20, by + 1.6, "$V_{snp}$", fontsize=5.3, ha="center",
            va="center", color=CATEGORICAL[4])
    wire([[xr + 22.6, by + 1.6], [xr + 27, by + 1.6]])
    gnd(xr + 27, by + 1.6)
    ax.text(xr + 31, by + 5.0, r"DC sweep $0\to V_{DD}$" "\n"
            r"gives $I_{leak}(V_{SN})$", fontsize=6.2, color=CATEGORICAL[4],
            va="center")

    ax.set_title("Simulated DRAM slice: one 1T1C cell with its column periphery, "
                 "and the leakage replica", fontsize=8, loc="left")
    return save_figure(fig, path)


# ==========================================================================
def figure_waveforms(cfg, builder, extractor, env, path: Path, log):
    """The characterisation sequence, captured from a real transient."""
    import matplotlib.pyplot as plt

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
    wf = wd / "wave.csv"
    if not wf.exists():
        log.error("waveform capture failed")
        shutil.rmtree(wd, ignore_errors=True)
        return []

    rows = []
    for line in wf.read_text().splitlines():
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

    fig, axes = plt.subplots(3, 1, sharex=True,
                             figsize=figure_size("page", height=5.0))

    ax = axes[0]
    ax.plot(tt, wl, color=CATEGORICAL[0], lw=1.3, label="WL")
    ax.plot(tt, preg, color=CATEGORICAL[3], lw=1.1, ls="--", label="PRE")
    ax.plot(tt, saen, color=CATEGORICAL[1], lw=1.1, ls="-.", label="SAE")
    ax.set_ylabel("Control (V)")
    ax.set_ylim(-0.15, max(wl.max(), preg.max()) * 1.55)
    ax.legend(fontsize=6, ncol=3, loc="upper right")
    ax.text(0.006, 0.94, "(a)", transform=ax.transAxes, fontweight="bold",
            va="top")

    ax = axes[1]
    ax.plot(tt, sn, color=CATEGORICAL[4], lw=1.4, label="$V_{SN}$")
    ax.axhline(p.vplate, color=REFERENCE_INK, lw=0.7, ls=":")
    ax.set_ylabel("Storage node (V)")
    ax.legend(fontsize=6, loc="lower right")
    ax.text(0.006, 0.94, "(b)", transform=ax.transAxes, fontweight="bold",
            va="top")

    ax = axes[2]
    ax.plot(tt, bl, color=CATEGORICAL[0], lw=1.3, label="BL")
    ax.plot(tt, blb, color=CATEGORICAL[5], lw=1.1, ls="--",
            label=r"$\overline{BL}$")
    ax.set_ylabel("Bitlines (V)")
    ax.set_xlabel("Time (ns)")
    ax.legend(fontsize=6, loc="center right")
    ax.text(0.006, 0.94, "(c)", transform=ax.transAxes, fontweight="bold",
            va="top")

    phases = [(0, t.pre1_end, "precharge"), (t.wr_on, t.wr_off, "write '1'"),
              (t.pre2_on, t.pre2_end, "precharge"),
              (t.rd_wl, t.sa, "charge share"), (t.sa, t.stop, "sense + restore")]
    for a in axes:
        for k, (t0, t1, _lab) in enumerate(phases):
            a.axvspan(t0 * 1e9, t1 * 1e9, color=REFERENCE_INK,
                      alpha=0.05 if k % 2 == 0 else 0.10, lw=0)
    for t0, t1, lab in phases:
        axes[0].text((t0 + t1) / 2 * 1e9, axes[0].get_ylim()[1] * 0.80, lab,
                     fontsize=5.6, ha="center", color=REFERENCE_INK)

    fig.suptitle("Characterisation sequence captured from the NGSpice transient")
    shutil.rmtree(wd, ignore_errors=True)
    return save_figure(fig, path)


# ==========================================================================
def figure_leakage(cfg, builder, extractor, env, path: Path, log):
    """Measured I_leak(V_SN), its thermal behaviour, and the retention integrand."""
    import matplotlib.pyplot as plt

    ds = cfg.design_space
    base = ds.decode_row([0.5] * ds.n_dim)
    base.update({"vdd": 1.1, "vwl_boost": 0.95, "vwl_low": 0.0,
                 "vblpre_ratio": 0.5, "cs": 20e-15, "cbl_ratio": 5.0,
                 "corner": "TT"})

    fig, axes = plt.subplots(1, 3, figsize=figure_size("page", height=2.5))
    temps = [-40.0, -10.0, 27.0, 55.0, 85.0, 125.0]
    arr_T, arr_I = [], []

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
            axes[0].semilogy(v[m], cur[m] * 1e15, lw=1.2, color=CATEGORICAL[i],
                             marker="o", ms=2.2, markevery=5, label=f"{T:.0f}")
            arr_T.append(T + 273.15)
            arr_I.append(float(np.interp(met.v_sn_written, v, cur)) * 1e15)
            lo, hi = met.v_fail, met.v_sn_written
            g = np.linspace(lo, hi, 400)
            ig = np.interp(g, v, cur)
            ok = ig > 0
            axes[2].semilogy(g[ok], p.cs / ig[ok], lw=1.2, color=CATEGORICAL[i])

    axes[0].set_xlabel("Storage-node voltage $V_{SN}$ (V)")
    axes[0].set_ylabel("Leakage leaving the node (fA)")
    axes[0].legend(fontsize=5.6, title=r"$T$ ($^\circ$C)", title_fontsize=5.8,
                   ncol=2, loc="lower right")
    axes[0].text(0.03, 0.96, "(a)", transform=axes[0].transAxes,
                 fontweight="bold", va="top")

    kB = 8.617333262e-5
    x = 1.0 / (kB * np.asarray(arr_T))
    y = np.log10(np.asarray(arr_I))
    ax = axes[1]
    ax.plot(x, arr_I, color=CATEGORICAL[0], marker="o", ms=4, lw=1.3)
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("$1/k_BT$ (eV$^{-1}$)")
    ax.set_ylabel("$I_{leak}$ at the written level (fA)")
    ax.text(0.03, 0.96, "(b)", transform=ax.transAxes, fontweight="bold",
            va="top")
    if len(arr_T) >= 2:
        Ea = (y[-1] - y[-2]) * np.log(10) / (x[-2] - x[-1])
        ax.annotate(f"thermally activated\n$E_a \\approx$ {Ea:.2f} eV",
                    xy=(x[-1], arr_I[-1]), xytext=(0.40, 0.58),
                    textcoords="axes fraction", fontsize=5.8,
                    color=CATEGORICAL[0], ha="left",
                    arrowprops=dict(arrowstyle="->", color=CATEGORICAL[0], lw=0.7))
        ax.annotate("tunnelling limited\n(weakly $T$-dependent)",
                    xy=(x[0], arr_I[0]), xytext=(0.05, 0.14),
                    textcoords="axes fraction", fontsize=5.8,
                    color=CATEGORICAL[3], ha="left",
                    arrowprops=dict(arrowstyle="->", color=CATEGORICAL[3], lw=0.7))

    axes[2].set_xlabel("Storage-node voltage $V_{SN}$ (V)")
    axes[2].set_ylabel("$C_{node}/I_{leak}$ (s V$^{-1}$)")
    axes[2].text(0.03, 0.96, "(c)", transform=axes[2].transAxes,
                 fontweight="bold", va="top")

    fig.suptitle("Measured storage-node leakage, its thermal behaviour, and the "
                 "retention integrand derived from it")
    return save_figure(fig, path)


# ==========================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", nargs="+", default=["dram_1t1c_45nm_lp.yaml"])
    args = ap.parse_args()

    ensure_directories()
    log = setup_logging("mfigures", filename="12_manuscript_figures.log")
    apply_style()
    cfg = load_config(*args.config)
    env = resolve_environment(probe=False)
    tech = Technology.from_config(cfg, generate=False)
    builder = DramCellBuilder(cfg, tech)
    extractor = MetricExtractor(cfg)

    log.info("figM1 schematic ...")
    log.info("  -> %d files", len(figure_schematic(FIGURES_DIR / "figM1_schematic")))
    log.info("figM2 waveforms ...")
    log.info("  -> %d files", len(figure_waveforms(cfg, builder, extractor, env,
                                                   FIGURES_DIR / "figM2_waveforms", log)))
    log.info("figM3 leakage ...")
    log.info("  -> %d files", len(figure_leakage(cfg, builder, extractor, env,
                                                 FIGURES_DIR / "figM3_leakage", log)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
