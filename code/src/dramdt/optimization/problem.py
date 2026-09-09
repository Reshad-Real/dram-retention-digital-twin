"""The pymoo problem that the digital twin solves.

Design vector
-------------
Only the *controllable* variables enter the search vector.  Temperature and
process corner are operating **conditions**, not design choices, so they are
supplied separately:

``condition_mode: nominal``
    Objectives are evaluated at a single nominal condition.
``condition_mode: worst_case``
    Objectives and constraints are evaluated at every configured condition and
    reduced with the pessimistic operator (minimum of a maximised objective,
    maximum of a minimised one).  This turns the search into a *robust* design
    problem: a solution is only as good as its worst corner.

Objective signs
---------------
pymoo minimises, so maximised objectives are negated once here.  Everything
downstream (indicators, plots, tables) therefore works in a consistent
minimisation space, and only the reporting layer flips signs back.

Constraints
-----------
pymoo's convention is ``g(x) <= 0`` for feasibility.  A ``min`` constraint on a
response ``r`` with threshold ``t`` becomes ``g = t - r``; a ``max`` constraint
becomes ``g = r - t``.  Constraints are normalised by the threshold magnitude so
that constraints of very different units contribute comparably to the
constraint-violation measure pymoo uses for tournament selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ..config import Config, ConfigError
from ..digital_twin.twin import DigitalTwin
from ..logging_utils import get_logger

__all__ = ["ObjectiveSpec", "ConstraintSpec", "DramDesignProblem"]

log = get_logger(__name__)


@dataclass(frozen=True)
class ObjectiveSpec:
    name: str                       # physical response name
    direction: str                  # 'minimize' | 'maximize'
    label: str = ""

    @property
    def sign(self) -> float:
        """+1 keeps a minimised objective, -1 flips a maximised one."""
        return 1.0 if self.direction == "minimize" else -1.0


@dataclass(frozen=True)
class ConstraintSpec:
    name: str
    kind: str                       # 'min' | 'max'
    threshold: float
    description: str = ""

    def violation(self, values: np.ndarray) -> np.ndarray:
        """``g(x) <= 0`` when satisfied, normalised by the threshold scale."""
        scale = max(abs(self.threshold), 1e-9)
        if self.kind == "min":
            return (self.threshold - values) / scale
        return (values - self.threshold) / scale


class DramDesignProblem:
    """Surrogate-backed design problem, exposed as a pymoo ``Problem``.

    Instantiating this class builds and returns a real pymoo ``Problem``
    subclass instance via :meth:`build`; the class itself holds the
    configuration logic so it can also be used to decode solutions afterwards.
    """

    def __init__(self, config: Config, twin: DigitalTwin):
        self.cfg = config
        self.twin = twin
        opt = dict(config.get("optimization", {}))

        ds = config.design_space
        names = opt.get("variables")
        if not names:
            names = [v.name for v in ds.variables
                     if v.kind != "categorical" and v.group == "design"]
        missing = [n for n in names if n not in ds.names]
        if missing:
            raise ConfigError(f"optimization.variables not in the design space: {missing}")
        self.var_names: list[str] = list(names)
        self.variables = [ds[n] for n in self.var_names]

        self.objectives = [
            ObjectiveSpec(name=o["name"], direction=o.get("direction", "minimize"),
                          label=o.get("label", o["name"]))
            for o in config.get("objectives", [])]
        if not self.objectives:
            raise ConfigError("No objectives configured")

        self.constraints = [
            ConstraintSpec(name=name,
                           kind=("min" if "min" in spec else "max"),
                           threshold=float(spec.get("min", spec.get("max"))),
                           description=str(spec.get("description", "")))
            for name, spec in (opt.get("constraints") or {}).items()]

        self.condition_mode = str(opt.get("condition_mode", "worst_case"))
        self.conditions: list[dict[str, Any]] = list(
            opt.get("conditions") or [{"temperature_c": 85.0, "corner": "TT"}])
        if self.condition_mode == "nominal":
            self.conditions = self.conditions[:1]

        self.fixed: dict[str, Any] = dict(opt.get("fixed_variables", {}))
        #: candidate *designs* assessed -- the currency the budget is denominated in
        self.n_evaluations = 0
        #: underlying digital-twin calls (designs x operating conditions)
        self.n_twin_calls = 0

    # ------------------------------------------------------------------
    @property
    def n_var(self) -> int:
        return len(self.var_names)

    @property
    def n_obj(self) -> int:
        return len(self.objectives)

    @property
    def n_constr(self) -> int:
        return len(self.constraints)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return (np.array([v.low for v in self.variables], dtype=float),
                np.array([v.high for v in self.variables], dtype=float))

    # ------------------------------------------------------------------
    def decode(self, X: np.ndarray, condition: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Search vectors + one condition -> design dictionaries for the twin."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        rows: list[dict[str, Any]] = []
        for x in X:
            d: dict[str, Any] = dict(self.fixed)
            for name, value in zip(self.var_names, x):
                d[name] = float(value)
            d.update(condition)
            rows.append(d)
        return rows

    def responses(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Worst-case (or nominal) response values for each search vector."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        n, n_cond = len(X), len(self.conditions)
        needed = {o.name for o in self.objectives} | {c.name for c in self.constraints}

        # All (design x condition) combinations go through the twin in ONE call.
        # Predicting per condition instead costs n_cond times as many scikit-learn
        # invocations, and their fixed per-call overhead dominates the runtime of
        # the serially-evaluating optimisers (MOEA/D, Bayesian) by an order of
        # magnitude.
        stacked: list[dict[str, Any]] = []
        for cond in self.conditions:
            stacked.extend(self.decode(X, cond))
        frame = self.twin.predict(stacked).frame

        per_condition: dict[str, list[np.ndarray]] = {k: [] for k in needed}
        for key in needed:
            if key in frame.columns:
                col = frame[key].to_numpy(dtype=float)
                for c in range(n_cond):
                    per_condition[key].append(col[c * n:(c + 1) * n])
            else:
                for _ in range(n_cond):
                    per_condition[key].append(np.full(n, np.nan))

        out: dict[str, np.ndarray] = {}
        direction = {o.name: o.direction for o in self.objectives}
        for key, stack in per_condition.items():
            arr = np.vstack(stack)
            if len(self.conditions) == 1:
                out[key] = arr[0]
                continue
            all_nan = np.all(~np.isfinite(arr), axis=0)
            reduced = np.full(arr.shape[1], np.nan)
            if (~all_nan).any():
                usable = arr[:, ~all_nan]
                if direction.get(key, "minimize") == "maximize" or \
                        any(c.name == key and c.kind == "min" for c in self.constraints):
                    # pessimistic for "bigger is better"
                    reduced[~all_nan] = np.nanmin(usable, axis=0)
                else:
                    # pessimistic for "smaller is better"
                    reduced[~all_nan] = np.nanmax(usable, axis=0)
            out[key] = reduced
        # The budget counts candidate designs, so that an algorithm is not
        # penalised for the number of operating conditions the *problem*
        # declares.  Twin calls are tracked separately for cost reporting.
        self.n_evaluations += len(X)
        self.n_twin_calls += len(X) * len(self.conditions)
        return out

    # ------------------------------------------------------------------
    def evaluate_raw(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(F, G)`` in pymoo minimisation / ``g<=0`` convention."""
        resp = self.responses(X)
        n = np.atleast_2d(X).shape[0]

        F = np.empty((n, self.n_obj), dtype=float)
        for j, obj in enumerate(self.objectives):
            vals = resp.get(obj.name, np.full(n, np.nan))
            F[:, j] = obj.sign * vals
        # A design the surrogate cannot score is treated as maximally bad
        # rather than silently dropped, so the optimiser learns to avoid it.
        bad = ~np.isfinite(F)
        if bad.any():
            finite = F[np.isfinite(F)]
            fill = (np.max(finite) + 1.0) if finite.size else 1e6
            F[bad] = fill

        if self.n_constr:
            G = np.empty((n, self.n_constr), dtype=float)
            for j, con in enumerate(self.constraints):
                vals = resp.get(con.name, np.full(n, np.nan))
                g = con.violation(vals)
                g[~np.isfinite(g)] = 1e3
                G[:, j] = g
        else:
            G = np.zeros((n, 0), dtype=float)
        return F, G

    # ------------------------------------------------------------------
    def build(self):
        """Return a concrete pymoo ``Problem`` bound to this specification."""
        from pymoo.core.problem import Problem

        outer = self
        xl, xu = self.bounds()

        class _Problem(Problem):
            def __init__(self):
                super().__init__(n_var=outer.n_var, n_obj=outer.n_obj,
                                 n_ieq_constr=outer.n_constr, xl=xl, xu=xu)
                self.spec = outer

            def _evaluate(self, X, out, *args, **kwargs):
                F, G = outer.evaluate_raw(X)
                out["F"] = F
                if outer.n_constr:
                    out["G"] = G

        return _Problem()

    # ------------------------------------------------------------------
    def solutions_frame(self, X: np.ndarray, F: np.ndarray | None = None) -> pd.DataFrame:
        """Tabulate a solution set with physical responses at every condition."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        frame = pd.DataFrame(X, columns=self.var_names)

        resp = self.responses(X)
        for obj in self.objectives:
            frame[obj.name] = resp.get(obj.name, np.full(len(X), np.nan))
        for con in self.constraints:
            if con.name not in frame.columns:
                frame[con.name] = resp.get(con.name, np.full(len(X), np.nan))
            frame[f"g_{con.name}"] = con.violation(frame[con.name].to_numpy(float))

        if self.n_constr:
            gcols = [f"g_{c.name}" for c in self.constraints]
            frame["feasible"] = (frame[gcols] <= 1e-9).all(axis=1)
        else:
            frame["feasible"] = True

        # per-condition detail, so a designer can see where the worst case bites
        for cond in self.conditions:
            tag = "_".join(f"{k}{v}" for k, v in cond.items())
            sub = self.twin.predict(self.decode(X, cond)).frame
            for obj in self.objectives:
                if obj.name in sub.columns:
                    frame[f"{obj.name}@{tag}"] = sub[obj.name].to_numpy()
        return frame

    def describe(self) -> dict[str, Any]:
        return {
            "n_var": self.n_var, "variables": self.var_names,
            "n_obj": self.n_obj,
            "objectives": [f"{o.direction} {o.name}" for o in self.objectives],
            "n_constr": self.n_constr,
            "constraints": [f"{c.name} {c.kind} {c.threshold}" for c in self.constraints],
            "condition_mode": self.condition_mode,
            "conditions": self.conditions,
        }
