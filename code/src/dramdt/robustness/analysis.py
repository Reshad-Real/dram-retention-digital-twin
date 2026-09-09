"""Robustness, yield and sensitivity analysis of candidate designs.

Three complementary studies:

**Monte-Carlo yield** -- physical parameters are perturbed by their configured
process sigmas and the twin evaluates thousands of perturbed instances per
design.  Yield is the fraction meeting *every* specification simultaneously,
not the product of per-spec marginals (which would ignore correlation between
failures).

**PVT corner sweep** -- each design is evaluated over the full corner x
temperature grid, giving the worst-case operating condition per specification.

**Sobol' sensitivity** -- variance-based first-order (``S1``) and total-effect
(``ST``) indices computed with the Saltelli estimator.  ``ST - S1`` quantifies
how much of a variable's influence comes from interactions, which is exactly
the question a designer asks when a one-at-a-time sweep disagrees with reality.

Because all three are surrogate-driven, a configurable subset of the
Monte-Carlo trials is **re-verified in NGSpice**, and the agreement between the
predicted and simulated yield is reported.  A robustness claim that has never
been checked against the simulator is not a result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from ..logging_utils import get_logger
from ..seeds import derive_seed

__all__ = ["RobustnessAnalyzer", "MonteCarloResult", "CornerSweepResult",
           "sobol_indices", "yield_from_frame"]

log = get_logger(__name__)


# --------------------------------------------------------------------------
def yield_from_frame(frame: pd.DataFrame,
                     thresholds: Mapping[str, float],
                     directions: Mapping[str, str] | None = None
                     ) -> dict[str, float]:
    """Per-specification and joint yield of a set of perturbed instances."""
    directions = dict(directions or {})
    ok = np.ones(len(frame), dtype=bool)
    out: dict[str, float] = {}
    for name, threshold in thresholds.items():
        if name not in frame.columns:
            out[f"yield_{name}"] = float("nan")
            continue
        vals = frame[name].to_numpy(dtype=float)
        direction = directions.get(name, "min")
        passed = (vals >= threshold) if direction == "min" else (vals <= threshold)
        passed &= np.isfinite(vals)
        out[f"yield_{name}"] = float(100.0 * passed.mean())
        ok &= passed
    out["yield_joint"] = float(100.0 * ok.mean()) if len(frame) else float("nan")
    out["n_trials"] = float(len(frame))
    return out


# --------------------------------------------------------------------------
@dataclass
class MonteCarloResult:
    design_id: str
    samples: pd.DataFrame
    summary: dict[str, Any]
    yields: dict[str, float]
    spice_check: pd.DataFrame = field(default_factory=pd.DataFrame)
    spice_yields: dict[str, float] = field(default_factory=dict)


@dataclass
class CornerSweepResult:
    design_id: str
    grid: pd.DataFrame
    worst_case: dict[str, Any]


# --------------------------------------------------------------------------
def sobol_indices(evaluate, bounds: Sequence[tuple[float, float]],
                  names: Sequence[str], n_base: int = 1024,
                  seed: int = 0, n_boot: int = 500) -> pd.DataFrame:
    """Variance-based sensitivity via the Saltelli estimator.

    ``evaluate`` maps an ``(n, d)`` array to an ``(n,)`` response.  The total
    cost is ``n_base * (d + 2)`` evaluations, which is negligible against the
    twin but impossible against SPICE -- this analysis is one of the concrete
    payoffs of having a surrogate.
    """
    from scipy.stats import qmc

    d = len(bounds)
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)

    m = int(math.ceil(math.log2(max(n_base, 2))))
    sample = qmc.Sobol(d=2 * d, scramble=True, seed=seed).random_base2(m=m)
    n = sample.shape[0]
    A = lo + sample[:, :d] * (hi - lo)
    B = lo + sample[:, d:] * (hi - lo)

    yA = np.asarray(evaluate(A), dtype=float)
    yB = np.asarray(evaluate(B), dtype=float)
    var = float(np.nanvar(np.concatenate([yA, yB])))
    if not np.isfinite(var) or var <= 0:
        return pd.DataFrame({"variable": list(names), "S1": np.nan, "ST": np.nan})

    yAB_all = np.empty((d, n), dtype=float)
    for i in range(d):
        AB = A.copy()
        AB[:, i] = B[:, i]
        yAB_all[i] = np.asarray(evaluate(AB), dtype=float)

    def _indices(idx):
        vb = float(np.nanvar(np.concatenate([yA[idx], yB[idx]])))
        vb = vb if (np.isfinite(vb) and vb > 0) else var
        s1 = np.array([np.nanmean(yB[idx] * (yAB_all[i, idx] - yA[idx])) / vb
                       for i in range(d)])
        st = np.array([np.nanmean((yA[idx] - yAB_all[i, idx]) ** 2) / (2.0 * vb)
                       for i in range(d)])
        return s1, st

    s1, st = _indices(np.arange(n))
    # bootstrap CIs over the Saltelli sample rows (B3: report uncertainty)
    rng = np.random.default_rng(seed)
    boot_s1 = np.empty((n_boot, d)); boot_st = np.empty((n_boot, d))
    for bi in range(n_boot):
        idx = rng.integers(0, n, n)
        boot_s1[bi], boot_st[bi] = _indices(idx)
    rows = []
    for i in range(d):
        rows.append({"variable": names[i], "S1": float(s1[i]), "ST": float(st[i]),
                     "interaction": max(float(st[i] - s1[i]), 0.0),
                     "S1_ci_low": float(np.nanpercentile(boot_s1[:, i], 2.5)),
                     "S1_ci_high": float(np.nanpercentile(boot_s1[:, i], 97.5)),
                     "ST_ci_low": float(np.nanpercentile(boot_st[:, i], 2.5)),
                     "ST_ci_high": float(np.nanpercentile(boot_st[:, i], 97.5))})
    frame = pd.DataFrame(rows)
    frame["S1_normalised"] = frame["S1"].clip(lower=0) / max(frame["S1"].clip(lower=0).sum(), 1e-12)
    frame["n_evaluations"] = n * (d + 2)
    frame["n_base"] = n
    return frame.sort_values("ST", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
class RobustnessAnalyzer:
    """Monte-Carlo, corner and sensitivity studies driven by the digital twin."""

    def __init__(self, config: Mapping[str, Any], twin, spice_evaluator=None):
        self.cfg = config
        self.twin = twin
        self.spice_evaluator = spice_evaluator      # callable(list[dict]) -> DataFrame
        rb = dict(config.get("robustness", {}))
        self.mc_cfg = dict(rb.get("monte_carlo", {}))
        self.corner_cfg = dict(rb.get("corner_sweep", {}))
        self.thresholds = dict(rb.get("yield_thresholds", {}))
        self.seed = int(config.get("experiment", {}).get("seed", 0))

        # direction of each specification, taken from the objective declarations
        self.directions: dict[str, str] = {}
        for obj in config.get("objectives", []):
            self.directions[obj["name"]] = "min" if obj.get("direction") == "maximize" else "max"
        self.directions.setdefault("write_efficiency", "min")
        self.directions.setdefault("retention_time_s", "min")
        self.directions.setdefault("read_margin_mv", "min")
        self.directions.setdefault("read_delay_ns", "max")

    # ------------------------------------------------------------------
    def monte_carlo(self, design: Mapping[str, Any], design_id: str,
                    n_trials: int | None = None,
                    condition: Mapping[str, Any] | None = None,
                    verify_in_spice: bool = True) -> MonteCarloResult:
        """Perturb the design by its process sigmas and evaluate the population."""
        n = int(n_trials or self.mc_cfg.get("n_trials", 1000))
        sigmas: dict[str, float] = dict(self.mc_cfg.get("parameter_sigma", {}))
        rng = np.random.default_rng(derive_seed(self.seed, f"mc:{design_id}"))

        base = dict(design)
        if condition:
            base.update(condition)

        designs: list[dict[str, Any]] = []
        for _ in range(n):
            d = dict(base)
            for key, rel in sigmas.items():
                if key in d and isinstance(d[key], (int, float)):
                    d[key] = float(d[key]) * float(rng.normal(1.0, rel))
            designs.append(d)

        frame = self.twin.predict(designs).frame
        yields = yield_from_frame(frame, self.thresholds, self.directions)

        summary: dict[str, Any] = {"design_id": design_id, "n_trials": n}
        for spec in self.thresholds:
            if spec in frame.columns:
                v = frame[spec].to_numpy(dtype=float)
                v = v[np.isfinite(v)]
                if v.size:
                    summary[f"{spec}_mean"] = float(v.mean())
                    summary[f"{spec}_std"] = float(v.std(ddof=1)) if v.size > 1 else 0.0
                    summary[f"{spec}_p1"] = float(np.percentile(v, 1))
                    summary[f"{spec}_p99"] = float(np.percentile(v, 99))
                    # process capability against a one-sided limit
                    t = float(self.thresholds[spec])
                    s = summary[f"{spec}_std"] or 1e-12
                    summary[f"{spec}_cpk"] = float(
                        (v.mean() - t) / (3 * s) if self.directions.get(spec) == "min"
                        else (t - v.mean()) / (3 * s))

        result = MonteCarloResult(design_id=design_id, samples=frame,
                                  summary=summary, yields=yields)

        # ---- SPICE re-verification of a random subset -------------------
        n_check = int(self.mc_cfg.get("spice_verification_trials", 0))
        if verify_in_spice and self.spice_evaluator is not None and n_check > 0:
            pick = rng.choice(len(designs), min(n_check, len(designs)), replace=False)
            subset = [designs[i] for i in pick]
            try:
                sim = self.spice_evaluator(subset)
                result.spice_check = sim
                result.spice_yields = yield_from_frame(sim, self.thresholds, self.directions)
                log.info("  [%s] yield: twin %.2f %% vs SPICE %.2f %% (n=%d)",
                         design_id, yields.get("yield_joint", float("nan")),
                         result.spice_yields.get("yield_joint", float("nan")), len(sim))
            except Exception as exc:
                log.exception("SPICE verification failed for %s: %s", design_id, exc)
        return result

    # ------------------------------------------------------------------
    def corner_sweep(self, design: Mapping[str, Any], design_id: str
                     ) -> CornerSweepResult:
        """Evaluate one design across the full PVT grid."""
        corners = list(self.corner_cfg.get("corners", ["TT"]))
        temps = list(self.corner_cfg.get("temperatures_c", [27]))

        designs, index = [], []
        for c in corners:
            for t in temps:
                d = dict(design)
                d["corner"] = c
                d["temperature_c"] = float(t)
                designs.append(d)
                index.append((c, float(t)))

        frame = self.twin.predict(designs).frame.copy()
        # `predict` carries the prepared inputs through, so `corner` and
        # `temperature_c` are usually already present: assign, then reorder.
        frame["corner"] = [c for c, _ in index]
        frame["temperature_c"] = [t for _, t in index]
        frame["design_id"] = design_id
        lead = ["design_id", "corner", "temperature_c"]
        frame = frame[lead + [c for c in frame.columns if c not in lead]]

        worst: dict[str, Any] = {"design_id": design_id}
        for spec, threshold in self.thresholds.items():
            if spec not in frame.columns:
                continue
            v = frame[spec].to_numpy(dtype=float)
            if not np.isfinite(v).any():
                continue
            if self.directions.get(spec, "min") == "min":
                k = int(np.nanargmin(v))
                worst[f"{spec}_worst"] = float(v[k])
                worst[f"{spec}_margin"] = float(v[k] - threshold)
            else:
                k = int(np.nanargmax(v))
                worst[f"{spec}_worst"] = float(v[k])
                worst[f"{spec}_margin"] = float(threshold - v[k])
            worst[f"{spec}_worst_corner"] = frame["corner"].iloc[k]
            worst[f"{spec}_worst_temp_c"] = float(frame["temperature_c"].iloc[k])
        worst["passes_all_corners"] = all(
            v >= 0 for k, v in worst.items() if k.endswith("_margin"))
        return CornerSweepResult(design_id=design_id, grid=frame, worst_case=worst)

    # ------------------------------------------------------------------
    def sensitivity(self, spec, target: str, n_base: int = 512,
                    condition: Mapping[str, Any] | None = None) -> pd.DataFrame:
        """Sobol' indices of one response over the optimisation design space."""
        cond = dict(condition or (spec.conditions[0] if spec.conditions else {}))
        names = spec.var_names
        bounds = [(v.low, v.high) for v in spec.variables]

        def evaluate(X: np.ndarray) -> np.ndarray:
            frame = self.twin.predict(spec.decode(X, cond)).frame
            if target not in frame.columns:
                return np.full(len(X), np.nan)
            return frame[target].to_numpy(dtype=float)

        frame = sobol_indices(evaluate, bounds, names, n_base=n_base,
                              seed=derive_seed(self.seed, f"sobol:{target}"))
        frame.insert(0, "response", target)
        return frame
