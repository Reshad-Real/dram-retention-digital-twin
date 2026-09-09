"""Publication-ready table export.

Every table is written in three forms so it can be dropped straight into a
manuscript, a repository or a README:

``.csv``   the machine-readable record
``.tex``   a ``booktabs`` table with caption, label and sensible column types
``.md``    a GitHub-flavoured Markdown rendering

Numbers are formatted once, centrally, so significant figures are consistent
across the whole paper: values are rendered to a fixed number of significant
digits and switched to scientific notation only outside a readable magnitude
window.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from ..logging_utils import get_logger

__all__ = ["TableSpec", "format_number", "dataframe_to_latex",
           "dataframe_to_markdown", "export_table"]

log = get_logger(__name__)

_LATEX_ESCAPES = {
    "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
    "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
}


def _escape_latex(text: str) -> str:
    """Escape LaTeX specials.

    Backslashes are handled first so that the escape sequences introduced for
    the other characters are not themselves re-escaped.
    """
    out = []
    for ch in str(text):
        if ch == "\\":
            out.append(r"\textbackslash{}")
        else:
            out.append(_LATEX_ESCAPES.get(ch, ch))
    return "".join(out)


def format_number(value: Any, sig: int = 4,
                  sci_low: float = 1e-3, sci_high: float = 1e5) -> str:
    """Consistent numeric formatting across every table."""
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, str):
        return value
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(v):
        return "--" if math.isnan(v) else ("$\\infty$" if v > 0 else "$-\\infty$")
    if v == 0:
        return "0"
    a = abs(v)
    if a < sci_low or a >= sci_high:
        mant, exp = f"{v:.{sig - 1}e}".split("e")
        return f"{mant}e{int(exp)}"
    if float(v).is_integer() and a < 1e5:
        return f"{int(v)}"
    decimals = max(sig - int(math.floor(math.log10(a))) - 1, 0)
    return f"{v:.{min(decimals, 10)}f}"


@dataclass
class TableSpec:
    """Presentation metadata for one table."""

    name: str
    caption: str = ""
    label: str | None = None
    columns: Sequence[str] | None = None
    rename: Mapping[str, str] = field(default_factory=dict)
    sig: int = 4
    index: bool = False
    notes: str = ""
    highlight_best: str | None = None      # column whose best value is bolded
    best_is_max: bool = True
    max_rows: int | None = None

    @property
    def tex_label(self) -> str:
        return self.label or f"tab:{re.sub(r'[^a-z0-9]+', '_', self.name.lower())}"


def _prepare(df: pd.DataFrame, spec: TableSpec) -> pd.DataFrame:
    out = df.copy()
    if spec.columns:
        keep = [c for c in spec.columns if c in out.columns]
        out = out[keep]
    if spec.max_rows:
        out = out.head(spec.max_rows)
    if spec.rename:
        out = out.rename(columns=dict(spec.rename))
    return out


def dataframe_to_latex(df: pd.DataFrame, spec: TableSpec) -> str:
    """Render a ``booktabs`` LaTeX table."""
    out = _prepare(df, spec)
    cols = list(out.columns)
    numeric = [pd.api.types.is_numeric_dtype(out[c]) for c in cols]
    align = "".join("r" if n else "l" for n in numeric)
    if spec.index:
        align = "l" + align

    best_idx = None
    if spec.highlight_best and spec.highlight_best in out.columns:
        series = pd.to_numeric(out[spec.highlight_best], errors="coerce")
        if series.notna().any():
            best_idx = int(series.idxmax() if spec.best_is_max else series.idxmin())

    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        rf"\caption{{{spec.caption or spec.name}}}",
        rf"\label{{{spec.tex_label}}}",
        r"\footnotesize",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
    ]
    header = [_escape_latex(str(c)) for c in cols]
    if spec.index:
        header = [""] + header
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")

    for ridx, (row_label, row) in enumerate(out.iterrows()):
        is_best = best_idx is not None and row_label == best_idx
        cells = []
        for c in cols:
            txt = (format_number(row[c], spec.sig)
                   if pd.api.types.is_number(row[c]) else _escape_latex(row[c]))
            if is_best:
                txt = rf"\textbf{{{txt}}}"
            cells.append(txt)
        if spec.index:
            cells = [_escape_latex(str(out.index[ridx]))] + cells
        lines.append(" & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}"]
    if spec.notes:
        lines.append(rf"\\[2pt] \footnotesize {_escape_latex(spec.notes)}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def dataframe_to_markdown(df: pd.DataFrame, spec: TableSpec) -> str:
    """Render a GitHub-flavoured Markdown table."""
    out = _prepare(df, spec)
    cols = list(out.columns)
    lines = []
    if spec.caption:
        lines.append(f"**{spec.caption}**")
        lines.append("")
    lines.append("| " + " | ".join(str(c) for c in cols) + " |")
    lines.append("|" + "|".join("---" for _ in cols) + "|")
    for _, row in out.iterrows():
        cells = [format_number(row[c], spec.sig) if pd.api.types.is_number(row[c])
                 else str(row[c]) for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    if spec.notes:
        lines.append("")
        lines.append(f"_{spec.notes}_")
    return "\n".join(lines) + "\n"


def export_table(df: pd.DataFrame, spec: TableSpec, out_dir: str | Path,
                 formats: Sequence[str] = ("csv", "tex", "md")) -> dict[str, Path]:
    """Write one table in every requested format; returns the paths."""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", spec.name)
    written: dict[str, Path] = {}

    if df is None or df.empty:
        log.warning("Table %r is empty -- nothing written", spec.name)
        return written

    if "csv" in formats:
        p = d / f"{stem}.csv"
        _prepare(df, spec).to_csv(p, index=spec.index)
        written["csv"] = p
    if "tex" in formats:
        p = d / f"{stem}.tex"
        p.write_text(dataframe_to_latex(df, spec), encoding="utf-8")
        written["tex"] = p
    if "md" in formats:
        p = d / f"{stem}.md"
        p.write_text(dataframe_to_markdown(df, spec), encoding="utf-8")
        written["md"] = p
    log.info("Table '%s' -> %s", spec.name, ", ".join(p.name for p in written.values()))
    return written
