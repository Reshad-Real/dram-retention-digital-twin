"""Colour palette definition and an executable accessibility validator.

The categorical palette used by every figure is checked *computationally*
rather than by eye, against five criteria:

1. **Lightness band** -- every step sits inside a usable OKLab lightness window,
   so no swatch disappears against the surface.
2. **Chroma floor** -- every step carries enough chroma to read as a distinct hue
   rather than as grey.
3. **Colour-vision-deficient separation** -- the OKLab distance (x100) between
   every pair, simulated under protanopia, deuteranopia and tritanopia, must
   reach 8.  This is the check that eyeballing always gets wrong.
4. **Normal-vision floor** -- pairs must be at least 15 apart for full-colour
   readers.
5. **Contrast** -- each step against the figure surface, for label legibility.

A sixth, print-specific criterion is added here because the figures target a
print journal: **greyscale separation**, the luminance distance between steps
after a greyscale conversion.  Together with the per-series marker and linestyle
cycles in :mod:`dramdt.viz.style`, this keeps figures readable when a reader
prints the paper in black and white.

The default palette is Okabe-Ito, the standard colour-vision-safe qualitative
set for scientific figures.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

__all__ = ["OKABE_ITO", "SEQUENTIAL_HUE", "DIVERGING_PAIR", "STATUS_COLORS",
           "validate_palette", "PaletteReport", "hex_to_rgb", "srgb_to_oklab",
           "simulate_cvd", "oklab_distance"]

#: Okabe-Ito, kept for reference and for the palette-comparison test.
#: NOTE: under the all-pairs criterion used here it FAILS -- the vermillion and
#: reddish-purple steps collapse to an OKLab distance of 0.6 under simulated
#: protanopia.  It is safe only when series are assigned in its intended order
#: and few of them are used, which is why the framework does not use it.
OKABE_ITO: tuple[str, ...] = (
    "#0072B2", "#D55E00", "#009E73", "#CC79A7",
    "#E69F00", "#56B4E9", "#F0E442", "#000000",
)

#: The categorical palette every figure uses.  Computed by :func:`design_palette`
#: (greedy max-min over an OKLCh lattice, optimising whichever of the CVD and
#: normal-vision separations is binding) and verified by :func:`validate_palette`
#: with zero failures on the all-pairs criterion:
#:
#:     minimum normal-vision OKLab distance   16.9   (floor 15)
#:     minimum worst-case CVD distance         9.0   (floor  8)
#:
#: The selection is *nested*: the first six entries are exactly the six-colour
#: solution (minimum distances 24.8 / 12.1), so figures with <= 6 series get the
#: widest margins automatically.  Assign in this fixed order and never cycle.
#: Greyscale separation for some later pairs is small, so every categorical
#: figure also varies marker and line style -- identity is never colour-alone.
#:
#: Reproduce with:  python scripts/09_figures_tables.py --rebuild-palette
CATEGORICAL: tuple[str, ...] = (
    "#8E0049",   # deep magenta   -- series 1
    "#16D894",   # green-cyan     -- series 2
    "#CC87F7",   # light violet   -- series 3
    "#1D61C5",   # strong blue    -- series 4
    "#657E22",   # olive          -- series 5
    "#FE9F0B",   # amber          -- series 6
    "#AD6782",   # dusty rose     -- series 7 (use sparingly)
    "#4699BC",   # steel blue     -- series 8 (use sparingly)
)

#: Neutral ink for the reference / baseline series (never a categorical hue).
REFERENCE_INK = "#2B2B2B"

#: Single-hue sequential ramp for magnitude (light -> dark).
SEQUENTIAL_HUE: tuple[str, ...] = (
    "#F7FBFF", "#DEEBF7", "#C6DBEF", "#9ECAE1", "#6BAED6",
    "#4292C6", "#2171B5", "#08519C", "#08306B",
)

#: Diverging pair with a neutral grey midpoint (never a hue at the midpoint).
DIVERGING_PAIR: tuple[str, ...] = (
    "#8C3800", "#C1600D", "#E9A25C", "#F5D4B0",
    "#F0F0F0",
    "#B8D6E8", "#6FA8CC", "#2E7BAA", "#0B4C74",
)

#: Reserved status colours -- never reused as a data series.
STATUS_COLORS = {
    "good": "#1B7837",
    "warning": "#E69F00",
    "serious": "#D55E00",
    "critical": "#A50026",
}

LIGHT_SURFACE = "#FFFFFF"
DARK_SURFACE = "#111418"


# --------------------------------------------------------------------------
# colour-space machinery
# --------------------------------------------------------------------------
def hex_to_rgb(color: str) -> np.ndarray:
    """Hex string -> gamma-encoded sRGB in [0, 1]."""
    c = color.lstrip("#")
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    return np.array([int(c[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], dtype=float)


def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    return np.clip(np.where(rgb <= 0.0031308, rgb * 12.92,
                            1.055 * np.maximum(rgb, 0) ** (1 / 2.4) - 0.055), 0.0, 1.0)


def srgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """Gamma-encoded sRGB -> OKLab (Ottosson)."""
    lin = _srgb_to_linear(np.asarray(rgb, dtype=float))
    m1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                   [0.2119034982, 0.6806995451, 0.1073969566],
                   [0.0883024619, 0.2817188376, 0.6299787005]])
    lms = m1 @ lin
    lms = np.cbrt(lms)
    m2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                   [1.9779984951, -2.4285922050, 0.4505937099],
                   [0.0259040371, 0.7827717662, -0.8086757660]])
    return m2 @ lms


def oklab_to_srgb(lab: np.ndarray) -> np.ndarray:
    """OKLab -> gamma-encoded sRGB (inverse of :func:`srgb_to_oklab`)."""
    m2_inv = np.array([[1.0, 0.3963377774, 0.2158037573],
                       [1.0, -0.1055613458, -0.0638541728],
                       [1.0, -0.0894841775, -1.2914855480]])
    lms = m2_inv @ np.asarray(lab, dtype=float)
    lms = lms ** 3
    m1_inv = np.array([[4.0767416621, -3.3077115913, 0.2309699292],
                       [-1.2684380046, 2.6097574011, -0.3413193965],
                       [-0.0041960863, -0.7034186147, 1.7076147010]])
    return _linear_to_srgb(m1_inv @ lms)


def oklch_to_hex(L: float, C: float, H_deg: float) -> tuple[str, bool]:
    """OKLCh -> hex, plus a flag saying whether the colour was in sRGB gamut."""
    h = math.radians(H_deg)
    lab = np.array([L, C * math.cos(h), C * math.sin(h)])
    m2_inv = np.array([[1.0, 0.3963377774, 0.2158037573],
                       [1.0, -0.1055613458, -0.0638541728],
                       [1.0, -0.0894841775, -1.2914855480]])
    lms = (m2_inv @ lab) ** 3
    m1_inv = np.array([[4.0767416621, -3.3077115913, 0.2309699292],
                       [-1.2684380046, 2.6097574011, -0.3413193965],
                       [-0.0041960863, -0.7034186147, 1.7076147010]])
    lin = m1_inv @ lms
    in_gamut = bool(np.all(lin >= -1e-4) and np.all(lin <= 1 + 1e-4))
    rgb = _linear_to_srgb(lin)
    return "#{:02X}{:02X}{:02X}".format(*(np.round(rgb * 255).astype(int))), in_gamut


def design_palette(n: int,
                   surface: str = LIGHT_SURFACE,
                   lightness_range: tuple[float, float] = (0.42, 0.78),
                   chroma_range: tuple[float, float] = (0.07, 0.17),
                   n_hue: int = 96, n_light: int = 9, n_chroma: int = 5,
                   greyscale_weight: float = 0.20,
                   cvd_min: float = 8.0, normal_min: float = 15.0,
                   seed_color: str | None = None) -> list[str]:
    """Greedily select ``n`` maximally-separable colours from an OKLCh lattice.

    Selection maximises, at each step, the minimum over already-chosen colours of

        min(CVD distance over protan/deutan/tritan)  +  w * greyscale distance

    so the palette is simultaneously colour-vision-safe and print-legible.  This
    is the "snap to passing" step: the palette is *computed* against the same
    checks :func:`validate_palette` enforces, never chosen by eye.
    """
    pool: list[str] = []
    for L in np.linspace(*lightness_range, n_light):
        for C in np.linspace(*chroma_range, n_chroma):
            for H in np.linspace(0, 360, n_hue, endpoint=False):
                hexv, ok = oklch_to_hex(float(L), float(C), float(H))
                if ok:
                    pool.append(hexv)
    pool = sorted(set(pool))

    # Pre-compute the CVD and greyscale representations once.
    lab_cvd = {c: {k: srgb_to_oklab(hex_to_rgb(simulate_cvd(c, k)))
                   for k in _CVD_MATRICES} for c in pool}
    lum = {c: relative_luminance(c) for c in pool}

    lab_normal = {c: srgb_to_oklab(hex_to_rgb(c)) for c in pool}

    def score(a: str, b: str) -> float:
        # Optimise the *binding* constraint: both the CVD distance and the
        # normal-vision distance are normalised by their own thresholds and the
        # smaller ratio is what limits the pair.  Maximising CVD separation
        # alone drives the search into a single hue family and breaks
        # normal-vision separability instead.
        worst_cvd = min(float(np.linalg.norm(lab_cvd[a][k] - lab_cvd[b][k]) * 100.0)
                        for k in _CVD_MATRICES)
        normal = float(np.linalg.norm(lab_normal[a] - lab_normal[b]) * 100.0)
        grey = abs(lum[a] - lum[b]) * 100.0
        return min(worst_cvd / cvd_min, normal / normal_min) + greyscale_weight * grey / 100.0

    chosen = [seed_color] if seed_color and seed_color in pool else [
        max(pool, key=lambda c: contrast_ratio(c, surface))]
    while len(chosen) < n:
        best, best_val = None, -math.inf
        for cand in pool:
            if cand in chosen:
                continue
            val = min(score(cand, c) for c in chosen)
            if val > best_val:
                best, best_val = cand, val
        if best is None:                                    # pragma: no cover
            break
        chosen.append(best)
    return chosen


def oklab_distance(c1: str, c2: str) -> float:
    """OKLab Euclidean distance, scaled by 100 (the units the checks use)."""
    a = srgb_to_oklab(hex_to_rgb(c1))
    b = srgb_to_oklab(hex_to_rgb(c2))
    return float(np.linalg.norm(a - b) * 100.0)


# Viénot, Brettel & Mollon (1999) dichromat simulation on linear RGB.
_RGB2LMS = np.array([[17.8824, 43.5161, 4.11935],
                     [3.45565, 27.1554, 3.86714],
                     [0.0299566, 0.184309, 1.46709]])
_LMS2RGB = np.linalg.inv(_RGB2LMS)

_CVD_MATRICES = {
    "protanopia": np.array([[0.0, 2.02344, -2.52581],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0]]),
    "deuteranopia": np.array([[1.0, 0.0, 0.0],
                              [0.494207, 0.0, 1.24827],
                              [0.0, 0.0, 1.0]]),
    "tritanopia": np.array([[1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [-0.395913, 0.801109, 0.0]]),
}


def simulate_cvd(color: str, kind: str) -> str:
    """Simulate a dichromatic view of one colour; returns a hex string."""
    if kind not in _CVD_MATRICES:
        raise ValueError(f"unknown CVD type {kind!r}")
    lin = _srgb_to_linear(hex_to_rgb(color))
    lms = _RGB2LMS @ lin
    sim = _CVD_MATRICES[kind] @ lms
    out = _linear_to_srgb(_LMS2RGB @ sim)
    return "#{:02X}{:02X}{:02X}".format(*(np.round(out * 255).astype(int)))


def relative_luminance(color: str) -> float:
    lin = _srgb_to_linear(hex_to_rgb(color))
    return float(np.dot([0.2126, 0.7152, 0.0722], lin))


def contrast_ratio(c1: str, c2: str) -> float:
    l1, l2 = relative_luminance(c1), relative_luminance(c2)
    hi, lo = max(l1, l2), min(l1, l2)
    return float((hi + 0.05) / (lo + 0.05))


def greyscale_distance(c1: str, c2: str) -> float:
    """Luminance separation after greyscale conversion, scaled to 0-100."""
    return abs(relative_luminance(c1) - relative_luminance(c2)) * 100.0


# --------------------------------------------------------------------------
@dataclass
class PaletteReport:
    """Outcome of the palette validation."""

    colors: list[str]
    surface: str
    lightness: list[dict] = field(default_factory=list)
    chroma: list[dict] = field(default_factory=list)
    pairs: list[dict] = field(default_factory=list)
    contrast: list[dict] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures

    def summary(self) -> str:
        lines = [f"Palette validation ({len(self.colors)} colours, "
                 f"surface {self.surface}): "
                 f"{'PASS' if self.passed else 'FAIL'}"]
        for f in self.failures:
            lines.append(f"  FAIL  {f}")
        for w in self.warnings:
            lines.append(f"  WARN  {w}")
        return "\n".join(lines)

    def to_frame(self):
        import pandas as pd
        return pd.DataFrame(self.pairs)


def validate_palette(colors: Sequence[str] = OKABE_ITO,
                     surface: str = LIGHT_SURFACE,
                     adjacent_only: bool = False,
                     cvd_min: float = 8.0,
                     normal_min: float = 15.0,
                     lightness_band: tuple[float, float] = (0.10, 0.95),
                     chroma_floor: float = 0.02,
                     contrast_min: float = 1.6,
                     greyscale_min: float = 3.0) -> PaletteReport:
    """Run the six checks over a categorical palette.

    ``adjacent_only`` restricts the pair checks to consecutive entries (the
    assignment order), which is the relevant case when series are always
    allocated in a fixed order; the default checks *all* pairs, which is the
    stricter requirement needed when a figure may show any subset.
    """
    report = PaletteReport(colors=list(colors), surface=surface)

    # -- 1. lightness band, 2. chroma floor ---------------------------
    for c in colors:
        lab = srgb_to_oklab(hex_to_rgb(c))
        L = float(lab[0])
        C = float(math.hypot(lab[1], lab[2]))
        report.lightness.append({"color": c, "oklab_L": round(L, 4)})
        report.chroma.append({"color": c, "oklab_C": round(C, 4)})
        if not (lightness_band[0] <= L <= lightness_band[1]):
            report.failures.append(
                f"{c}: OKLab lightness {L:.3f} outside band {lightness_band}")
        if C < chroma_floor and c not in ("#000000", "#FFFFFF"):
            report.warnings.append(
                f"{c}: chroma {C:.3f} below floor {chroma_floor} (reads as grey)")

    # -- 3./4. pair separation, normal vision and CVD -------------------
    pairs: Iterable[tuple[str, str]]
    pairs = (list(zip(colors, colors[1:])) if adjacent_only
             else list(itertools.combinations(colors, 2)))
    for c1, c2 in pairs:
        row = {"color_1": c1, "color_2": c2,
               "normal": round(oklab_distance(c1, c2), 2),
               "greyscale": round(greyscale_distance(c1, c2), 2)}
        worst = math.inf
        for kind in _CVD_MATRICES:
            d = oklab_distance(simulate_cvd(c1, kind), simulate_cvd(c2, kind))
            row[kind] = round(d, 2)
            worst = min(worst, d)
        row["cvd_worst"] = round(worst, 2)
        report.pairs.append(row)

        if row["normal"] < normal_min:
            report.failures.append(
                f"{c1} vs {c2}: normal-vision distance {row['normal']:.1f} "
                f"below the hard floor {normal_min}")
        if worst < cvd_min:
            report.failures.append(
                f"{c1} vs {c2}: worst-case CVD distance {worst:.1f} below {cvd_min}")
        if row["greyscale"] < greyscale_min:
            report.warnings.append(
                f"{c1} vs {c2}: greyscale separation {row['greyscale']:.1f} is low -- "
                "these two series must also differ by marker or line style")

    # -- 5. contrast against the surface --------------------------------
    for c in colors:
        cr = contrast_ratio(c, surface)
        report.contrast.append({"color": c, "contrast_vs_surface": round(cr, 2)})
        if cr < contrast_min:
            report.warnings.append(
                f"{c}: contrast {cr:.2f} against {surface} -- needs a visible label "
                "or a table view")
    return report
