"""Execution of the optimiser portfolio, with a fair and repeated protocol.

Fairness
--------
Every algorithm receives the **same evaluation budget**
``pop_size * n_generations`` surrogate evaluations, and every algorithm is run
``n_runs`` times from independent seeds.  Comparisons are therefore made on
distributions, not on single lucky runs, which is what the statistical tests in
:mod:`dramdt.statistics` consume.

Reference front
---------------
The true Pareto front is unknown, so IGD/IGD+ and the hypervolume reference
point are derived from the non-dominated union of *all* runs of *all*
algorithms.  That union is computed once, after every run has finished, and
reused for every indicator so the numbers are mutually comparable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ..logging_utils import get_logger
from ..seeds import derive_seed
from .algorithms import (ALGORITHMS, AlgorithmSpec, PenalizedProblem,
                         build_algorithm, tchebycheff, weight_vectors)
from .indicators import (build_reference_front, build_reference_point,
                         indicator_bundle, non_dominated)
from .problem import DramDesignProblem

__all__ = ["OptimizationRunner", "OptimizationResult", "SingleRun"]

log = get_logger(__name__)


@dataclass
class SingleRun:
    """One (algorithm, seed) execution."""

    algorithm: str
    label: str
    run_index: int
    seed: int
    F: np.ndarray                       # non-dominated front, minimisation space
    X: np.ndarray                       # matching decision vectors
    n_evaluations: int
    runtime_s: float
    history: list[dict[str, float]] = field(default_factory=list)
    indicators: dict[str, float] = field(default_factory=dict)
    n_feasible: int = 0
    notes: str = ""


@dataclass
class OptimizationResult:
    runs: list[SingleRun] = field(default_factory=list)
    reference_front: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    reference_point: np.ndarray = field(default_factory=lambda: np.empty(0))
    indicator_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    convergence: pd.DataFrame = field(default_factory=pd.DataFrame)
    pareto_solutions: pd.DataFrame = field(default_factory=pd.DataFrame)
    best_algorithm: str = ""
    settings: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
class OptimizationRunner:
    """Runs the whole portfolio against a :class:`DramDesignProblem`."""

    def __init__(self, spec: DramDesignProblem, config: Mapping[str, Any]):
        self.spec = spec
        opt = dict(config.get("optimization", {}))
        self.cfg = opt
        self.pop_size = int(opt.get("population_size", 120))
        self.n_gen = int(opt.get("n_generations", 150))
        self.n_runs = int(opt.get("n_runs", 11))
        self.names: list[str] = list(opt.get("algorithms") or list(ALGORITHMS))
        self.seed = int(config.get("experiment", {}).get("seed", 0))
        self.penalty = float(opt.get("constraint_penalty", 1e3))
        self.n_weights = int(opt.get("scalarized_weight_vectors", 24))
        self.hv_margin = float(opt.get("hv_reference_margin", 0.1))

    @property
    def budget(self) -> int:
        return self.pop_size * self.n_gen

    # ------------------------------------------------------------------
    @property
    def probe_size(self) -> int:
        """Designs spent estimating the ideal point for a scalarised run.

        Charged against the same budget as everything else, so the Tchebycheff
        methods do not get free information the population-based methods lack.
        """
        return int(self.cfg.get("ideal_point_probe", min(256, self.pop_size * 4)))

    def _ideal_point(self, n: int = 256, seed: int = 0) -> np.ndarray:
        """Estimate the ideal point from a space-filling probe of the domain."""
        rng = np.random.default_rng(seed)
        xl, xu = self.spec.bounds()
        X = xl + rng.random((n, self.spec.n_var)) * (xu - xl)
        F, _G = self.spec.evaluate_raw(X)
        F = F[np.all(np.isfinite(F), axis=1)]
        return F.min(axis=0) if len(F) else np.zeros(self.spec.n_obj)

    # ------------------------------------------------------------------
    def _run_moo(self, name: str, run_index: int, seed: int) -> SingleRun:
        from pymoo.optimize import minimize
        from pymoo.core.callback import Callback

        spec = build_algorithm(name, self.spec.n_obj, self.pop_size, seed, self.cfg)
        # MOEA/D has no native constrained form in pymoo: give it the penalised
        # objective vector instead, and say so in the run notes.
        if name == "moead" and self.spec.n_constr:
            problem = PenalizedProblem(self.spec, self.penalty).build_vector()
            note = f"constraints handled by penalty (coefficient {self.penalty:g})"
        else:
            problem = self.spec.build()
            note = ""

        class _Track(Callback):
            def __init__(self):
                super().__init__()
                self.rows: list[dict[str, float]] = []

            def notify(self, algorithm):
                F = algorithm.opt.get("F") if algorithm.opt is not None else None
                if F is None or len(F) == 0:
                    return
                F = np.atleast_2d(np.asarray(F, dtype=float))
                self.rows.append({
                    "generation": int(algorithm.n_gen),
                    "n_evaluations": int(algorithm.evaluator.n_eval),
                    "n_nondominated": int(non_dominated(F).sum()),
                    **{f"f{j}_min": float(F[:, j].min()) for j in range(F.shape[1])},
                    "front": F.copy(),
                })

        cb = _Track()
        before = self.spec.n_evaluations
        t0 = time.perf_counter()
        res = minimize(problem, spec.algorithm, ("n_gen", self.n_gen),
                       seed=seed, callback=cb, verbose=False,
                       save_history=False)
        runtime = time.perf_counter() - t0

        F = np.atleast_2d(np.asarray(res.F, dtype=float)) if res.F is not None \
            else np.empty((0, self.spec.n_obj))
        X = np.atleast_2d(np.asarray(res.X, dtype=float)) if res.X is not None \
            else np.empty((0, self.spec.n_var))
        return SingleRun(algorithm=name, label=ALGORITHMS[name][1], run_index=run_index,
                         seed=seed, F=F, X=X,
                         n_evaluations=self.spec.n_evaluations - before,
                         runtime_s=runtime, history=cb.rows, notes=note)

    # ------------------------------------------------------------------
    def _run_scalarized(self, name: str, run_index: int, seed: int) -> SingleRun:
        """DE / PSO applied to Tchebycheff subproblems (see algorithms.py)."""
        from pymoo.optimize import minimize

        # Split the SAME budget across the subproblems.  Using the configured
        # weight-vector count regardless of n_gen would silently multiply the
        # budget: with W subproblems each needing at least 2 generations, the
        # count must be capped so that W * gens * pop stays inside the budget.
        n_weights = max(min(self.n_weights, self.n_gen // 2), 2)
        W = weight_vectors(self.spec.n_obj, n_weights, seed=seed)
        ideal = self._ideal_point(n=self.probe_size, seed=derive_seed(seed, "ideal"))
        wrapper = PenalizedProblem(self.spec, self.penalty)

        remaining = max(self.budget - self.probe_size, n_weights * self.pop_size * 2)
        per_run_gen = max(int(remaining / (n_weights * self.pop_size)), 2)
        Xs: list[np.ndarray] = []
        before = self.spec.n_evaluations
        t0 = time.perf_counter()
        rows: list[dict[str, float]] = []
        for k, w in enumerate(W):
            alg = build_algorithm(name, 1, self.pop_size,
                                  derive_seed(seed, f"{name}:{k}"), self.cfg).algorithm
            sub = wrapper.build_scalar(w, ideal)
            res = minimize(sub, alg, ("n_gen", per_run_gen),
                           seed=derive_seed(seed, f"{name}:seed:{k}"), verbose=False)
            if res.X is not None:
                Xs.append(np.atleast_2d(np.asarray(res.X, dtype=float))[0])
            if Xs:
                Fk, _ = self.spec.evaluate_raw(np.vstack(Xs))
                nd = Fk[non_dominated(Fk)]
                rows.append({
                    "generation": k + 1,
                    "n_evaluations": self.spec.n_evaluations - before,
                    "n_nondominated": int(len(nd)),
                    **{f"f{j}_min": float(nd[:, j].min()) for j in range(nd.shape[1])},
                    "front": nd.copy(),
                })
        runtime = time.perf_counter() - t0

        if not Xs:
            return SingleRun(name, ALGORITHMS[name][1], run_index, seed,
                             np.empty((0, self.spec.n_obj)), np.empty((0, self.spec.n_var)),
                             0, runtime, notes="no solutions returned")

        X = np.vstack(Xs)
        F, G = self.spec.evaluate_raw(X)
        if G.shape[1]:
            feasible = (G <= 1e-9).all(axis=1)
            if feasible.any():
                X, F = X[feasible], F[feasible]
        mask = non_dominated(F)
        return SingleRun(
            algorithm=name, label=ALGORITHMS[name][1], run_index=run_index, seed=seed,
            F=F[mask], X=X[mask],
            n_evaluations=self.spec.n_evaluations - before, runtime_s=runtime,
            history=rows,
            notes=(f"Tchebycheff decomposition over {n_weights} weight vectors, "
                   f"{per_run_gen} generations each ({self.probe_size} designs "
                   f"spent on the ideal-point probe); constraints by penalty "
                   f"(coefficient {self.penalty:g})"))

    # ------------------------------------------------------------------
    def _run_bayesian(self, name: str, run_index: int, seed: int) -> SingleRun:
        """Multi-objective Bayesian optimisation via Optuna's TPE sampler."""
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        xl, xu = self.spec.bounds()
        names = self.spec.var_names
        directions = ["minimize"] * self.spec.n_obj
        # The same budget as every other algorithm, denominated in candidate
        # designs, so the comparison stays like-for-like.
        n_trials = min(int(self.cfg.get("bayesian_trials") or self.budget), self.budget)
        batch = int(self.cfg.get("bayesian_batch", 1))

        sampler = optuna.samplers.TPESampler(
            seed=seed, multivariate=True, group=True,
            n_startup_trials=int(self.cfg.get("bayesian_startup", 64)))
        study = optuna.create_study(directions=directions, sampler=sampler)

        before = self.spec.n_evaluations
        rows: list[dict[str, float]] = []
        t0 = time.perf_counter()

        def objective(trial: "optuna.Trial") -> Sequence[float]:
            x = np.array([[trial.suggest_float(n, float(lo), float(hi))
                           for n, lo, hi in zip(names, xl, xu)]])
            F, G = self.spec.evaluate_raw(x)
            pen = self.penalty * np.maximum(G, 0.0).sum() if G.shape[1] else 0.0
            return [float(v + pen) for v in F[0]]

        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        runtime = time.perf_counter() - t0

        best = study.best_trials
        if not best:
            return SingleRun(name, ALGORITHMS[name][1], run_index, seed,
                             np.empty((0, self.spec.n_obj)),
                             np.empty((0, self.spec.n_var)), n_trials, runtime,
                             notes="no Pareto trials")
        X = np.array([[t.params[n] for n in names] for t in best], dtype=float)
        F, G = self.spec.evaluate_raw(X)
        if G.shape[1]:
            feasible = (G <= 1e-9).all(axis=1)
            if feasible.any():
                X, F = X[feasible], F[feasible]
        mask = non_dominated(F)

        # convergence trace from the trial history
        seen: list[list[float]] = []
        for i, t in enumerate(study.trials, start=1):
            if t.values is None:
                continue
            seen.append(list(t.values))
            if i % max(n_trials // 40, 1) == 0:
                arr = np.asarray(seen, dtype=float)
                nd = arr[non_dominated(arr)]
                rows.append({"generation": i, "n_evaluations": i,
                             "n_nondominated": int(len(nd)),
                             **{f"f{j}_min": float(nd[:, j].min())
                                for j in range(nd.shape[1])},
                             "front": nd.copy()})

        return SingleRun(
            algorithm=name, label=ALGORITHMS[name][1], run_index=run_index, seed=seed,
            F=F[mask], X=X[mask], n_evaluations=self.spec.n_evaluations - before,
            runtime_s=runtime, history=rows,
            notes=f"Optuna multi-objective TPE, {n_trials} trials; constraints by penalty")

    # ------------------------------------------------------------------
    def run(self) -> OptimizationResult:
        """Execute every algorithm ``n_runs`` times and score the outcome."""
        result = OptimizationResult(settings={
            "population_size": self.pop_size, "n_generations": self.n_gen,
            "evaluation_budget": self.budget, "n_runs": self.n_runs,
            "algorithms": self.names, "constraint_penalty": self.penalty,
            "scalarized_weight_vectors": self.n_weights,
            **self.spec.describe(),
        })

        for name in self.names:
            if name not in ALGORITHMS:
                log.warning("Unknown algorithm %r -- skipping", name)
                continue
            kind = ALGORITHMS[name][0]
            for r in range(self.n_runs):
                seed = derive_seed(self.seed, f"opt:{name}", r)
                try:
                    if kind == "moo":
                        run = self._run_moo(name, r, seed)
                    elif kind == "scalarized":
                        run = self._run_scalarized(name, r, seed)
                    else:
                        run = self._run_bayesian(name, r, seed)
                except Exception as exc:
                    log.exception("%s run %d failed: %s", name, r, exc)
                    continue
                run.n_feasible = int(len(run.F))
                result.runs.append(run)
                log.info("  %-9s run %2d/%d | %3d solutions | %6.1f s | %d evaluations",
                         name, r + 1, self.n_runs, len(run.F), run.runtime_s,
                         run.n_evaluations)

        if not result.runs:
            log.error("No optimisation runs completed")
            return result

        # ---- reference front / point from the union of every run ------
        fronts = [r.F for r in result.runs if len(r.F)]
        result.reference_front = build_reference_front(fronts, seed=self.seed)

        # The four objectives are in incommensurate units (ms, ns, nW, mV), so
        # an unnormalised hypervolume / IGD+ is dominated by whichever objective
        # carries the largest raw magnitude.  Normalise every objective to [0, 1]
        # over the pooled front, so both indicators are unit-independent and the
        # hypervolume reference point is the fixed nadir 1 + margin in every
        # dimension.  The normalisation is recorded for reproducibility.
        pooled = np.vstack([np.asarray(f, float) for f in fronts])
        pooled = pooled[np.all(np.isfinite(pooled), axis=1)]
        obj_lo = pooled.min(axis=0)
        obj_hi = pooled.max(axis=0)
        obj_span = np.where(obj_hi - obj_lo > 0, obj_hi - obj_lo, 1.0)

        def _norm(F):
            F = np.asarray(F, float)
            return (F - obj_lo) / obj_span if F.size else F

        ref_front_n = _norm(result.reference_front)
        ref_point_n = np.full(len(obj_span), 1.0 + self.hv_margin)
        result.reference_point = ref_point_n
        result.settings["objective_normalization"] = {
            "objectives": self.spec.describe().get("objectives", []),
            "pooled_min": obj_lo.tolist(), "pooled_max": obj_hi.tolist(),
            "reference_point_normalized": ref_point_n.tolist(),
            "note": "HV and IGD+ are computed on objectives normalised to [0,1] "
                    "over the pooled front; reference point = nadir(1) + margin."}
        log.info("Reference front: %d points; normalised reference point: %s",
                 len(result.reference_front),
                 np.round(ref_point_n, 4).tolist())

        # ---- indicators (on normalised objectives) ---------------------
        rows = []
        for run in result.runs:
            run.indicators = indicator_bundle(_norm(run.F), ref_front_n,
                                              ref_point_n)
            rows.append({"algorithm": run.algorithm, "label": run.label,
                         "run": run.run_index, "seed": run.seed,
                         "runtime_s": round(run.runtime_s, 3),
                         "n_evaluations": run.n_evaluations,
                         **run.indicators})
        result.indicator_table = pd.DataFrame(rows)

        # ---- convergence traces ---------------------------------------
        conv_rows = []
        for run in result.runs:
            for row in run.history:
                front = row.get("front")
                hv = (indicator_bundle(_norm(front), ref_front_n,
                                       ref_point_n)["hypervolume"]
                      if front is not None and len(front) else float("nan"))
                conv_rows.append({
                    "algorithm": run.algorithm, "label": run.label, "run": run.run_index,
                    "generation": row.get("generation"),
                    "n_evaluations": row.get("n_evaluations"),
                    "n_nondominated": row.get("n_nondominated"),
                    "hypervolume": hv})
        result.convergence = pd.DataFrame(conv_rows)

        # ---- best algorithm by median hypervolume ----------------------
        if not result.indicator_table.empty:
            med = (result.indicator_table.groupby("algorithm")["hypervolume"]
                   .median().sort_values(ascending=False))
            result.best_algorithm = str(med.index[0])
            log.info("Median hypervolume ranking: %s",
                     ", ".join(f"{k}={v:.5g}" for k, v in med.items()))

        # ---- pooled Pareto solutions in physical units -----------------
        all_X = np.vstack([r.X for r in result.runs if len(r.X)])
        all_F = np.vstack([r.F for r in result.runs if len(r.F)])
        tags = np.concatenate([[r.algorithm] * len(r.F) for r in result.runs if len(r.F)])
        mask = non_dominated(all_F)
        frame = self.spec.solutions_frame(all_X[mask])
        frame.insert(0, "found_by", tags[mask])
        result.pareto_solutions = frame
        log.info("Global Pareto set: %d solutions (%d feasible)",
                 len(frame), int(frame["feasible"].sum()) if "feasible" in frame else -1)
        return result
