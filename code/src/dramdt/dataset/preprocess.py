"""Stage 3b -- dataset cleaning, feature engineering and splitting.

The pipeline is deliberately conservative: a simulated point is only removed
when it is *not a valid observation of the physics* (the simulator failed, did
not converge, or a required measurement never triggered).  Points that
represent genuine design failures -- a write that cannot charge the cell, a
sense amplifier that cannot resolve -- are **kept and labelled**, because a
surrogate that has never seen a failing design cannot warn an optimiser away
from one.

Every removal is counted and reported in :class:`PreprocessReport`, which is
written next to the processed dataset.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from ..logging_utils import get_logger
from ..seeds import derive_seed

__all__ = ["Preprocessor", "PreprocessReport", "engineer_features",
           "sanitize_for_parquet"]

log = get_logger(__name__)

KELVIN = 273.15
BOLTZMANN_EV = 8.617333262e-5      # eV/K


# --------------------------------------------------------------------------
def sanitize_for_parquet(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Make every column writable by Arrow, reporting what had to change.

    Concatenating frames from different experimental conditions can leave an
    object column holding more than one Python type (for example a policy name
    present for some techniques and absent for others).  Arrow rejects those.
    Rather than silently dropping the column, each mixed column is coerced --
    to numeric where that is lossless, otherwise to string -- and the coercion
    is returned so it can be logged.
    """
    out = df.copy()
    coerced: dict[str, str] = {}
    for col in out.columns:
        if out[col].dtype != object:
            continue
        non_null = out[col].dropna()
        if non_null.empty:
            out[col] = out[col].astype("float64")
            coerced[col] = "all-null -> float64"
            continue
        types = {type(v) for v in non_null.head(5000)}
        if len(types) <= 1:
            continue
        numeric = pd.to_numeric(out[col], errors="coerce")
        if numeric.notna().sum() == non_null.shape[0]:
            out[col] = numeric
            coerced[col] = f"{sorted(t.__name__ for t in types)} -> numeric"
        else:
            out[col] = out[col].astype(str).where(out[col].notna(), None)
            coerced[col] = f"{sorted(t.__name__ for t in types)} -> string"
    return out, coerced


@dataclass
class PreprocessReport:
    """Audit trail of what the preprocessing stage did."""

    n_input: int = 0
    n_output: int = 0
    removed: dict[str, int] = field(default_factory=dict)
    labelled: dict[str, int] = field(default_factory=dict)
    engineered_features: list[str] = field(default_factory=list)
    target_transforms: dict[str, str] = field(default_factory=dict)
    outliers_flagged: dict[str, int] = field(default_factory=dict)
    split_sizes: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["retention_rate"] = round(self.n_output / max(self.n_input, 1), 6)
        return d


# --------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add physically-motivated derived features.

    Each feature encodes a known first-order relationship, so the surrogate has
    to learn less and the SHAP attributions stay interpretable:

    ``charge_share_ratio``  Cs/(Cs+Cbl) -- the charge-sharing gain
    ``signal_charge_c``     Cs*(VDD-VBLpre) -- charge available to the SA
    ``vwl_overdrive``       VPP-VDD -- headroom for a full '1' write
    ``vgs_write``           VPP-VDD -- access-device drive at the end of a write
    ``vgs_retention``       VWL,low-VBLpre -- reverse bias that gates leakage
    ``access_aspect``       W/L of the access device
    ``inv_thermal_v``       1/(kT/q) -- the Arrhenius/subthreshold axis
    ``sa_drive_ratio``      SA latch width over bitline capacitance
    """
    out = df.copy()
    created: list[str] = []

    def add(name: str, series: pd.Series) -> None:
        out[name] = series
        created.append(name)

    cs = out.get("cs")
    cbl = out.get("cbl")
    if cs is not None and cbl is not None:
        add("charge_share_ratio", cs / (cs + cbl))
        add("cbl_over_cs", cbl / cs)
    if cs is not None and "vdd" in out and "vblpre" in out:
        add("signal_charge_c", cs * (out["vdd"] - out["vblpre"]))
        add("stored_charge_c", cs * out["vdd"])
    if "vwl_high" in out and "vdd" in out:
        add("vwl_overdrive", out["vwl_high"] - out["vdd"])
        add("vgs_write", out["vwl_high"] - out["vdd"])
    if "vwl_low" in out and "vblpre" in out:
        add("vgs_retention", out["vwl_low"] - out["vblpre"])
    if "wacc" in out and "lacc" in out:
        add("access_aspect", out["wacc"] / out["lacc"])
        add("access_area", out["wacc"] * out["lacc"])
    if "temperature_c" in out:
        tk = out["temperature_c"] + KELVIN
        add("temperature_k", tk)
        add("inv_thermal_v", 1.0 / (BOLTZMANN_EV * tk))
    if "wsan" in out and cbl is not None:
        add("sa_drive_ratio", out["wsan"] / cbl)
    if "wsan" in out and "lsa" in out:
        add("sa_aspect", out["wsan"] / out["lsa"])

    # DEBUG, not INFO: this runs on every digital-twin prediction, so at INFO it
    # would emit millions of lines during an optimisation run.
    log.debug("Feature engineering: added %d features", len(created))
    return out, created


# --------------------------------------------------------------------------
class Preprocessor:
    """Cleans a raw campaign frame into a modelling-ready dataset."""

    #: measurements that must be present for a row to be a valid observation
    REQUIRED_FINITE = ("read_margin_mv", "v_sn_written_v", "retention_time_s",
                       "energy_per_access_fj", "total_power_nw")

    def __init__(self, config: Mapping[str, Any]):
        pp = dict(config.get("preprocessing", {}))
        self.drop_failed_sim = bool(pp.get("drop_failed_simulations", True))
        self.drop_nonconverged = bool(pp.get("drop_nonconverged", True))
        self.drop_incomplete = bool(pp.get("drop_incomplete_measurements", True))
        self.outlier_method = str(pp.get("outlier_method", "iqr"))
        self.outlier_factor = float(pp.get("outlier_factor", 3.0))
        self.outlier_action = str(pp.get("outlier_action", "flag"))   # flag | clip | drop
        self.feature_engineering = bool(pp.get("feature_engineering", True))
        self.log_floor = float(pp.get("log_floor", 1e-12))
        self.split = dict(pp.get("split", {"train": 0.7, "val": 0.15, "test": 0.15}))
        self.stratify_by = pp.get("stratify_by", "corner")
        self.seed = int(config.get("experiment", {}).get("seed", 0))
        self.targets = [o["name"] for o in config.get("objectives", [])]
        self.transforms = {o["name"]: o.get("transform")
                           for o in config.get("objectives", []) if o.get("transform")}

    # ------------------------------------------------------------------
    def run(self, df: pd.DataFrame) -> tuple[pd.DataFrame, PreprocessReport]:
        rep = PreprocessReport(n_input=len(df))
        work = df.copy()

        # ---- 1. validity filtering ------------------------------------
        if self.drop_failed_sim and "simulation_ok" in work:
            mask = work["simulation_ok"].astype(bool)
            rep.removed["simulation_failed"] = int((~mask).sum())
            work = work[mask]
        if self.drop_nonconverged and "converged" in work:
            mask = work["converged"].astype(bool)
            rep.removed["not_converged"] = int((~mask).sum())
            work = work[mask]
        if self.drop_incomplete:
            present = [c for c in self.REQUIRED_FINITE if c in work.columns]
            if present:
                finite = np.isfinite(work[present].to_numpy(dtype=float)).all(axis=1)
                rep.removed["incomplete_measurements"] = int((~finite).sum())
                work = work[finite]

        work = work.reset_index(drop=True)

        # ---- 2. label (do not remove) genuine design failures ----------
        for flag in ("read_success", "write_success", "retention_censored"):
            if flag in work:
                rep.labelled[flag] = int(work[flag].astype(bool).sum())
        if "read_success" in work and "write_success" in work:
            work["design_feasible"] = (work["read_success"].astype(bool)
                                       & work["write_success"].astype(bool))
            rep.labelled["design_feasible"] = int(work["design_feasible"].sum())

        # ---- 3. feature engineering -----------------------------------
        if self.feature_engineering:
            work, created = engineer_features(work)
            rep.engineered_features = created

        # ---- 4. target transforms -------------------------------------
        for name, tf in self.transforms.items():
            if name not in work:
                continue
            if tf == "log10":
                col = f"log10_{name}"
                vals = work[name].to_numpy(dtype=float)
                # A zero response is a real observation (e.g. a cell with no
                # retention at all), so it is floored rather than discarded.
                # The floor is a configuration value and is reported.
                with np.errstate(divide="ignore", invalid="ignore"):
                    work[col] = np.where(np.isfinite(vals),
                                         np.log10(np.maximum(vals, self.log_floor)),
                                         np.nan)
                n_floored = int(np.sum(np.isfinite(vals) & (vals < self.log_floor)))
                rep.target_transforms[name] = col
                if n_floored:
                    rep.notes.append(
                        f"{name}: {n_floored} values floored at {self.log_floor:g} "
                        "before the log10 transform")

        # ---- 5. outliers ----------------------------------------------
        numeric_targets = [t for t in self.targets if t in work.columns]
        for col in numeric_targets:
            vals = work[col].to_numpy(dtype=float)
            finite = np.isfinite(vals)
            if finite.sum() < 20:
                continue
            if self.outlier_method == "iqr":
                q1, q3 = np.percentile(vals[finite], [25, 75])
                iqr = q3 - q1
                lo, hi = q1 - self.outlier_factor * iqr, q3 + self.outlier_factor * iqr
            else:                       # robust z-score (MAD)
                med = np.median(vals[finite])
                mad = np.median(np.abs(vals[finite] - med)) or 1e-12
                lo, hi = med - self.outlier_factor * 1.4826 * mad, \
                    med + self.outlier_factor * 1.4826 * mad
            flag = finite & ((vals < lo) | (vals > hi))
            rep.outliers_flagged[col] = int(flag.sum())
            work[f"outlier_{col}"] = flag
            if self.outlier_action == "clip":
                work[col] = np.clip(vals, lo, hi)
            elif self.outlier_action == "drop":
                work = work[~flag].reset_index(drop=True)

        # ---- 6. split --------------------------------------------------
        work = self._assign_split(work, rep)

        rep.n_output = len(work)
        log.info("Preprocessing: %d -> %d rows (removed %s)",
                 rep.n_input, rep.n_output, rep.removed)
        return work.reset_index(drop=True), rep

    # ------------------------------------------------------------------
    def _assign_split(self, df: pd.DataFrame, rep: PreprocessReport) -> pd.DataFrame:
        """Deterministic train/val/test assignment, stratified when possible."""
        n = len(df)
        if n == 0:
            df["split"] = pd.Series(dtype=object)
            return df
        fr_tr = float(self.split.get("train", 0.7))
        fr_va = float(self.split.get("val", 0.15))
        rng = np.random.default_rng(derive_seed(self.seed, "split"))

        labels = np.empty(n, dtype=object)
        strata_col = self.stratify_by if (self.stratify_by in df.columns) else None
        groups = ([(k, np.flatnonzero(df[strata_col].to_numpy() == k))
                   for k in pd.unique(df[strata_col])]
                  if strata_col else [(None, np.arange(n))])

        for _key, idx in groups:
            idx = idx.copy()
            rng.shuffle(idx)
            n_tr = int(round(len(idx) * fr_tr))
            n_va = int(round(len(idx) * fr_va))
            labels[idx[:n_tr]] = "train"
            labels[idx[n_tr:n_tr + n_va]] = "val"
            labels[idx[n_tr + n_va:]] = "test"

        df = df.copy()
        df["split"] = labels
        rep.split_sizes = {k: int((labels == k).sum()) for k in ("train", "val", "test")}
        if strata_col:
            rep.notes.append(f"split stratified by '{strata_col}'")
        return df
