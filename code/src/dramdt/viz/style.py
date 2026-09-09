"""Matplotlib configuration for IEEE-style publication figures.

Conventions enforced here:

* **Sizing** -- figures are authored at their final printed width (3.5 in for a
  single column, 7.16 in for a double-column span) so fonts are never rescaled
  by the typesetter.
* **Type** -- a serif family matching journal body text, with a minimum
  effective size of 8 pt at print scale.
* **Recessive chrome** -- light grid, no top/right spines, axes below the data.
* **Identity is never colour-alone** -- :func:`series_style` pairs each colour
  with a distinct marker *and* line style, so the figures survive greyscale
  printing and colour-vision deficiency.
* **Output** -- every figure is written as PNG (300 dpi raster), PDF and SVG
  (vector), with fonts embedded as Type 42 so they remain editable/searchable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt

from .palette import CATEGORICAL, REFERENCE_INK

__all__ = ["apply_style", "figure_size", "save_figure", "series_style",
           "COLUMN_WIDTH_IN", "PAGE_WIDTH_IN", "FORMATS", "MARKERS",
           "LINESTYLES", "HATCHES", "legend_below", "annotate_panel"]

COLUMN_WIDTH_IN = 3.5           # IEEE single column
PAGE_WIDTH_IN = 7.16            # IEEE double column
GOLDEN = 0.618

FORMATS = ("png", "pdf", "svg")

#: Secondary encodings, cycled in lockstep with the categorical colours.
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")
LINESTYLES = ("-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)),
              (0, (1, 1)), (0, (7, 2, 1, 2)))
HATCHES = ("", "///", "...", "\\\\\\", "xxx", "---", "+++", "ooo")

_TEXT_PRIMARY = "#1A1A1A"
_TEXT_SECONDARY = "#4D4D4D"
_GRID = "#D9D9D9"


def apply_style(scale: float = 1.0, grid: bool = True) -> None:
    """Install the publication rcParams.  Call once before plotting."""
    base = 8.0 * scale
    mpl.rcParams.update({
        # --- type -------------------------------------------------------
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Nimbus Roman",
                       "Liberation Serif", "serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": base,
        "axes.titlesize": base + 1,
        "axes.labelsize": base,
        "xtick.labelsize": base - 0.5,
        "ytick.labelsize": base - 0.5,
        "legend.fontsize": base - 0.5,
        "figure.titlesize": base + 2,

        # --- colour -----------------------------------------------------
        "axes.prop_cycle": mpl.cycler(color=list(CATEGORICAL)),
        "text.color": _TEXT_PRIMARY,
        "axes.labelcolor": _TEXT_PRIMARY,
        "axes.edgecolor": _TEXT_SECONDARY,
        "xtick.color": _TEXT_SECONDARY,
        "ytick.color": _TEXT_SECONDARY,

        # --- chrome -----------------------------------------------------
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "axes.grid": grid,
        "axes.axisbelow": True,
        "grid.color": _GRID,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.9,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,

        # --- marks ------------------------------------------------------
        "lines.linewidth": 1.2,
        "lines.markersize": 3.5,
        "lines.markeredgewidth": 0.6,
        "patch.linewidth": 0.5,
        "boxplot.flierprops.markersize": 2.5,

        # --- legend -----------------------------------------------------
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": _GRID,
        "legend.borderpad": 0.35,
        "legend.handlelength": 1.8,
        "legend.columnspacing": 1.0,
        "legend.labelspacing": 0.3,

        # --- output -----------------------------------------------------
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "savefig.transparent": False,
        "pdf.fonttype": 42,          # embed TrueType so text stays editable
        "ps.fonttype": 42,
        "svg.fonttype": "none",      # keep SVG text as text
        "figure.autolayout": False,
        "figure.constrained_layout.use": True,
    })


def figure_size(width: str | float = "column", aspect: float = GOLDEN,
                height: float | None = None) -> tuple[float, float]:
    """Figure size in inches at final print width."""
    if isinstance(width, str):
        w = {"column": COLUMN_WIDTH_IN, "page": PAGE_WIDTH_IN,
             "wide": PAGE_WIDTH_IN, "half": PAGE_WIDTH_IN / 2}.get(width, COLUMN_WIDTH_IN)
    else:
        w = float(width)
    return (w, height if height is not None else w * aspect)


def series_style(index: int) -> dict[str, Any]:
    """Colour + marker + line style for series ``index`` (fixed order)."""
    i = int(index)
    return {
        "color": CATEGORICAL[i % len(CATEGORICAL)],
        "marker": MARKERS[i % len(MARKERS)],
        "linestyle": LINESTYLES[i % len(LINESTYLES)],
        "hatch": HATCHES[i % len(HATCHES)],
    }


def legend_below(ax, ncol: int = 3, **kw: Any):
    """Place a legend under the axes -- keeps the plot area uncluttered."""
    return ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
                     ncol=ncol, frameon=False, **kw)


def annotate_panel(ax, label: str, dx: float = -0.16, dy: float = 1.04) -> None:
    """Panel letter for multi-part figures, e.g. ``(a)``."""
    ax.text(dx, dy, label, transform=ax.transAxes, fontweight="bold",
            va="bottom", ha="left")


def save_figure(fig, path: str | Path, formats: Sequence[str] = FORMATS,
                close: bool = True, dpi: int = 300) -> list[Path]:
    """Write one figure in every requested format; returns the paths written."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for fmt in formats:
        target = p.with_suffix(f".{fmt}")
        try:
            fig.savefig(target, format=fmt, dpi=dpi)
            written.append(target)
        except Exception as exc:                            # pragma: no cover
            import logging
            logging.getLogger(__name__).warning(
                "Could not write %s: %s", target.name, exc)
    if close:
        plt.close(fig)
    return written
