"""Publication-ready dataset export.

Produces a self-describing Excel workbook suitable for deposit in a data
repository (Mendeley Data, Zenodo, figshare), plus a compressed CSV mirror for
programmatic reuse.

Workbook structure
------------------
``README``            what the dataset is, how it was produced, how to cite
``Data_Dictionary``   one row per column: units, role, description
``Dataset``           the full table (one row per simulated design point)
``Design_Space``      the sampled variables and their ranges
``Summary_Statistics``  descriptive statistics of every numeric column
``Provenance``        tool versions, seeds, config hash, model-card fingerprint

Excel's hard limit is 1,048,576 rows per sheet; if the dataset exceeds that it
is split across ``Dataset_1``, ``Dataset_2``, ... and the split is recorded in
the README sheet.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ..logging_utils import get_logger, package_versions

__all__ = ["export_dataset", "DatasetExportReport", "build_data_dictionary"]

log = get_logger(__name__)

EXCEL_MAX_ROWS = 1_048_575          # minus the header row


# --------------------------------------------------------------------------
@dataclass
class DatasetExportReport:
    excel_path: Path | None = None
    csv_path: Path | None = None
    n_rows: int = 0
    n_columns: int = 0
    n_sheets: int = 0
    size_mb: float = 0.0
    columns: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
_UNIT_PATTERNS: tuple[tuple[str, str, str], ...] = (
    # suffix,        unit,     description fragment
    ("_ns", "ns", "time"),
    ("_ms", "ms", "time"),
    ("_fj", "fJ", "energy"),
    ("_nw", "nW", "power"),
    ("_uw", "uW", "power"),
    ("_mv", "mV", "voltage"),
    ("_fa", "fA", "current"),
    ("_v", "V", "voltage"),
    ("_s", "s", "time"),
    ("_c", "C", "charge"),
    ("_k", "K", "temperature"),
    ("_m", "m", "length"),
)

_ROLE_HINTS: dict[str, str] = {
    "sample_id": "identifier",
    "split": "partition",
    "corner": "process condition",
    "assist_config": "experimental condition",
    "config_hash": "provenance",
    "model_fingerprint": "provenance",
    "backend": "provenance",
    "worker_pid": "provenance",
    "notes": "diagnostic",
}


def _infer_unit(column: str, explicit: Mapping[str, str]) -> str:
    if column in explicit:
        return explicit[column]
    low = column.lower()
    for suffix, unit, _ in _UNIT_PATTERNS:
        if low.endswith(suffix):
            return unit
    if low.startswith(("w", "l")) and low not in ("write_success",):
        return ""
    return ""


def build_data_dictionary(df: pd.DataFrame,
                          design_space_rows: Sequence[Mapping[str, Any]] = (),
                          descriptions: Mapping[str, str] | None = None,
                          units: Mapping[str, str] | None = None) -> pd.DataFrame:
    """One row per column describing units, dtype, role and meaning."""
    descriptions = dict(descriptions or {})
    units = dict(units or {})

    for row in design_space_rows:
        name = str(row.get("variable", ""))
        if name:
            descriptions.setdefault(name, str(row.get("description", "")))
            if row.get("unit"):
                units.setdefault(name, str(row["unit"]))

    records = []
    for col in df.columns:
        series = df[col]
        numeric = pd.api.types.is_numeric_dtype(series)
        role = _ROLE_HINTS.get(col, "")
        if not role:
            if col.startswith(("outlier_", "n_")) or col in ("converged", "simulation_ok"):
                role = "diagnostic"
            elif col.endswith("_success") or col in ("design_feasible", "retention_censored"):
                role = "label"
            elif any(col.startswith(p) for p in ("retention_", "read_", "write_", "total_",
                                                 "standby_", "active_", "refresh_",
                                                 "energy_", "leakage_")):
                role = "response"
            else:
                role = "predictor"
        records.append({
            "column": col,
            "dtype": str(series.dtype),
            "unit": _infer_unit(col, units),
            "role": role,
            "n_missing": int(series.isna().sum()),
            "n_unique": int(series.nunique(dropna=True)) if not numeric else "",
            "minimum": (float(np.nanmin(series.to_numpy(dtype=float)))
                        if numeric and series.notna().any() else ""),
            "maximum": (float(np.nanmax(series.to_numpy(dtype=float)))
                        if numeric and series.notna().any() else ""),
            "description": descriptions.get(col, ""),
        })
    return pd.DataFrame.from_records(records)


# --------------------------------------------------------------------------
def _readme_frame(meta: Mapping[str, Any], n_rows: int, n_cols: int,
                  sheets: Sequence[str]) -> pd.DataFrame:
    rows = [
        ("Title", meta.get("title", "DRAM 1T1C SPICE characterisation dataset")),
        ("Description", meta.get("description", "")),
        ("Generated (UTC)", datetime.now(timezone.utc).isoformat(timespec="seconds")),
        ("Rows", n_rows),
        ("Columns", n_cols),
        ("Sheets", ", ".join(sheets)),
        ("", ""),
        ("HOW IT WAS PRODUCED", ""),
        ("Simulator", meta.get("simulator", "")),
        ("Device models", meta.get("device_models", "")),
        ("Model-card fingerprint", meta.get("model_fingerprint", "")),
        ("Sampling", meta.get("sampling", "")),
        ("Master seed", meta.get("seed", "")),
        ("Configuration hash", meta.get("config_hash", "")),
        ("Framework", meta.get("framework", "dramdt")),
        ("", ""),
        ("IMPORTANT NOTES", ""),
        ("Retention model", meta.get("retention_note", "")),
        ("Failed designs", "Design points that fail to write or to sense are RETAINED and "
                           "flagged via write_success / read_success / design_feasible, so "
                           "the dataset covers the infeasible region as well."),
        ("Censoring", "retention_censored=TRUE marks points whose storage node equilibrates "
                      "above the failure level; retention_time_s then holds the censoring "
                      "cap, not a measured value."),
        ("Corner models", "SS/FF/SF/FS cards are first-order syntheses from the typical PTM "
                          "card (vth0/u0/tox shifts declared in the configuration). They are "
                          "not foundry sign-off corners."),
        ("", ""),
        ("LICENCE", meta.get("licence", "CC BY 4.0")),
        ("CITATION", meta.get("citation", "")),
    ]
    return pd.DataFrame(rows, columns=["Field", "Value"])


def _summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include=[np.number])
    if num.empty:
        return pd.DataFrame()
    desc = num.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
    desc.insert(0, "column", desc.index)
    desc["n_missing"] = [int(num[c].isna().sum()) for c in desc["column"]]
    desc["skew"] = [float(num[c].skew()) for c in desc["column"]]
    desc["kurtosis"] = [float(num[c].kurtosis()) for c in desc["column"]]
    return desc.reset_index(drop=True)


def export_dataset(df: pd.DataFrame,
                   excel_path: str | Path,
                   csv_path: str | Path | None = None,
                   meta: Mapping[str, Any] | None = None,
                   design_space_rows: Sequence[Mapping[str, Any]] = (),
                   provenance: Mapping[str, Any] | None = None,
                   extra_sheets: Mapping[str, pd.DataFrame] | None = None
                   ) -> DatasetExportReport:
    """Write the repository-ready workbook (and optional CSV mirror)."""
    meta = dict(meta or {})
    excel_path = Path(excel_path)
    excel_path.parent.mkdir(parents=True, exist_ok=True)

    dictionary = build_data_dictionary(df, design_space_rows)
    summary = _summary_statistics(df)
    ds_frame = pd.DataFrame(list(design_space_rows)) if design_space_rows else pd.DataFrame()

    prov_rows = [{"key": k, "value": str(v)} for k, v in (provenance or {}).items()]
    prov_rows += [{"key": f"package::{k}", "value": v}
                  for k, v in package_versions().items()]
    prov = pd.DataFrame(prov_rows)

    n_chunks = max(math.ceil(len(df) / EXCEL_MAX_ROWS), 1)
    data_sheets = ["Dataset"] if n_chunks == 1 else [f"Dataset_{i+1}" for i in range(n_chunks)]
    sheet_names = ["README", "Data_Dictionary", *data_sheets, "Design_Space",
                   "Summary_Statistics", "Provenance", *list((extra_sheets or {}).keys())]

    readme = _readme_frame(meta, len(df), len(df.columns), sheet_names)

    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as xl:
        readme.to_excel(xl, sheet_name="README", index=False)
        dictionary.to_excel(xl, sheet_name="Data_Dictionary", index=False)
        for i, name in enumerate(data_sheets):
            lo, hi = i * EXCEL_MAX_ROWS, min((i + 1) * EXCEL_MAX_ROWS, len(df))
            df.iloc[lo:hi].to_excel(xl, sheet_name=name, index=False)
        if not ds_frame.empty:
            ds_frame.to_excel(xl, sheet_name="Design_Space", index=False)
        if not summary.empty:
            summary.to_excel(xl, sheet_name="Summary_Statistics", index=False)
        if not prov.empty:
            prov.to_excel(xl, sheet_name="Provenance", index=False)
        for name, frame in (extra_sheets or {}).items():
            frame.to_excel(xl, sheet_name=name[:31], index=False)

        # cosmetic: freeze headers and widen the first column of text sheets
        book = xl.book
        header = book.add_format({"bold": True, "bg_color": "#DDE7F0", "border": 1})
        for sheet_name in xl.sheets:
            ws = xl.sheets[sheet_name]
            ws.freeze_panes(1, 0)
            ws.set_column(0, 0, 34)
            if sheet_name in ("README",):
                ws.set_column(1, 1, 110)
            elif sheet_name in ("Data_Dictionary", "Design_Space", "Provenance"):
                ws.set_column(1, 8, 22)
        for sheet_name, frame in (("README", readme), ("Data_Dictionary", dictionary)):
            ws = xl.sheets[sheet_name]
            for c, col in enumerate(frame.columns):
                ws.write(0, c, str(col), header)

    size_mb = excel_path.stat().st_size / (1024 * 1024)
    report = DatasetExportReport(excel_path=excel_path, n_rows=len(df),
                                 n_columns=len(df.columns), n_sheets=len(sheet_names),
                                 size_mb=round(size_mb, 2), columns=list(df.columns))

    if csv_path is not None:
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False,
                  compression="gzip" if str(csv_path).endswith(".gz") else None)
        report.csv_path = csv_path

    log.info("Dataset exported: %s (%d rows x %d cols, %.1f MB, %d sheets)",
             excel_path.name, len(df), len(df.columns), size_mb, len(sheet_names))
    return report
