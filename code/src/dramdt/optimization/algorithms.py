"""The optimiser portfolio.

Native multi-objective algorithms
    ``nsga2``  NSGA-II -- fast non-dominated sort + crowding distance
    ``nsga3``  NSGA-III -- reference-direction niching (better for >3 objectives)
    ``moead``  MOEA/D  -- decomposition into scalar subproblems

Scalarised algorithms
    ``de``     differential evolution
    ``pso``    particle-swarm optimisation

    Both are single-objective methods.  They are applied here in the standard
    decomposition fashion: the objective vector is reduced with an augmented
    Tchebycheff scalarisation against a set of uniformly spread weight vectors,
    one independent run per weight vector, and the union of the run optima forms
    the front.  The per-run evaluation budget is the total budget divided by the
    number of weight vectors, so every algorithm in the comparison spends the
    same number of surrogate evaluations.

Bayesian optimisation
    ``bayesian``  Optuna's multi-objective Tree-structured Parzen Estimator --
    a sequential model-based (Bayesian) method, given the same evaluation budget.

Constraint handling: NSGA-II/III use pymoo's native feasibility-first
tournament.  MOEA/D, DE and PSO have no native constrained form here, so they
receive a penalised objective (documented in :class:`PenalizedProblem`), which
is the conventional treatment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ..logging_utils import get_logger

__all__ = ["ALGORITHMS", "build_algorithm", "algorithm_labels", "AlgorithmSpec",
           "PenalizedProblem", "tchebycheff", "weight_vectors"]

log = get_logger(__name__)

#: name -> (kind, human label).  ``kind`` drives how the runner executes it.
ALGORITHMS: dict[str, tuple[str, str]] = {
    "nsga2":    ("moo", "NSGA-II"),
    "nsga3":    ("moo", "NSGA-III"),
    "moead":    ("moo", "MOEA/D"),
    "de":       ("scalarized", "Differential evolution"),
    "pso":      ("scalarized", "Particle-swarm optimization"),
    "bayesian": ("bayesian", "Bayesian optimization (multi-objective TPE)"),
}


def algorithm_labels() -> dict[str, str]:
    return {k: v[1] for k, v in ALGORITHMS.items()}


@dataclass
class AlgorithmSpec:
    name: str
    kind: str
    label: str
    algorithm: Any = None
    n_gen: int = 100
    pop_size: int = 100
    extra: dict[str, Any] = None            # type: ignore[assignment]


# --------------------------------------------------------------------------
def weight_vectors(n_obj: int, n_vectors: int, seed: int = 0) -> np.ndarray:
    """Uniformly spread weight vectors on the unit simplex."""
    from pymoo.util.ref_dirs import get_reference_directions
    try:
        w = get_reference_directions("energy", n_obj, n_vectors, seed=seed)
    except Exception:                                       # pragma: no cover
        rng = np.random.default_rng(seed)
        w = rng.dirichlet(np.ones(n_obj), size=n_vectors)
    return np.clip(np.asarray(w, dtype=float), 1e-6, None)


def tchebycheff(F: np.ndarray, w: np.ndarray, ideal: np.ndarray,
                rho: float = 0.05) -> np.ndarray:
    """Augmented Tchebycheff scalarisation (lower is better).

    ``max_i w_i |f_i - z*_i| + rho * sum_i w_i |f_i - z*_i|`` -- the augmentation
    term removes weakly-Pareto-optimal solutions that the plain Tchebycheff
    function admits.
    """
    F = np.atleast_2d(np.asarray(F, dtype=float))
    d = np.abs(F - np.asarray(ideal, dtype=float)) * np.asarray(w, dtype=float)
    return d.max(axis=1) + rho * d.sum(axis=1)


class PenalizedProblem:
    """Wrap a constrained problem as an unconstrained scalar/vector one.

    ``f_penalised = f + penalty * sum(max(g, 0))``.  The penalty is applied on
    the *normalised* constraint violations produced by
    :meth:`ConstraintSpec.violation`, so a single coefficient is meaningful
    across constraints with different units.
    """

    def __init__(self, spec, penalty: float = 1e3):
        self.spec = spec
        self.penalty = float(penalty)

    def build_scalar(self, w: np.ndarray, ideal: np.ndarray):
        """A pymoo single-objective ``Problem`` for one weight vector."""
        from pymoo.core.problem import Problem

        spec, penalty = self.spec, self.penalty
        xl, xu = spec.bounds()

        class _Scalar(Problem):
            def __init__(self):
                super().__init__(n_var=spec.n_var, n_obj=1, n_ieq_constr=0,
                                 xl=xl, xu=xu)

            def _evaluate(self, X, out, *args, **kwargs):
                F, G = spec.evaluate_raw(X)
                s = tchebycheff(F, w, ideal)
                if G.shape[1]:
                    s = s + penalty * np.maximum(G, 0.0).sum(axis=1)
                out["F"] = s.reshape(-1, 1)

        return _Scalar()

    def build_vector(self):
        """A pymoo multi-objective ``Problem`` with penalised objectives."""
        from pymoo.core.problem import Problem

        spec, penalty = self.spec, self.penalty
        xl, xu = spec.bounds()

        class _Vector(Problem):
            def __init__(self):
                super().__init__(n_var=spec.n_var, n_obj=spec.n_obj,
                                 n_ieq_constr=0, xl=xl, xu=xu)

            def _evaluate(self, X, out, *args, **kwargs):
                F, G = spec.evaluate_raw(X)
                if G.shape[1]:
                    F = F + penalty * np.maximum(G, 0.0).sum(axis=1, keepdims=True)
                out["F"] = F

        return _Vector()


# --------------------------------------------------------------------------
def _operators(seed: int, cfg: Mapping[str, Any]):
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.lhs import LHS
    # `prob` must be omitted rather than passed as None: pymoo compares it
    # numerically and a None raises TypeError inside the operator.  Omitting it
    # selects pymoo's default of 1/n_var, which is the conventional choice.
    mutation_kwargs: dict[str, Any] = {"eta": float(cfg.get("mutation_eta", 20))}
    if cfg.get("mutation_prob") is not None:
        mutation_kwargs["prob"] = float(cfg["mutation_prob"])
    return dict(
        sampling=LHS(),
        crossover=SBX(prob=float(cfg.get("crossover_prob", 0.9)),
                      eta=float(cfg.get("crossover_eta", 15))),
        mutation=PM(**mutation_kwargs),
        eliminate_duplicates=True,
    )


def build_algorithm(name: str, n_obj: int, pop_size: int, seed: int,
                    cfg: Mapping[str, Any] | None = None) -> AlgorithmSpec:
    """Instantiate one optimiser from the portfolio."""
    cfg = dict(cfg or {})
    if name not in ALGORITHMS:
        raise KeyError(f"Unknown algorithm {name!r}; available: {sorted(ALGORITHMS)}")
    kind, label = ALGORITHMS[name]

    if name == "nsga2":
        from pymoo.algorithms.moo.nsga2 import NSGA2
        alg = NSGA2(pop_size=pop_size, **_operators(seed, cfg))

    elif name == "nsga3":
        from pymoo.algorithms.moo.nsga3 import NSGA3
        from pymoo.util.ref_dirs import get_reference_directions
        ref = get_reference_directions("energy", n_obj, pop_size, seed=seed)
        alg = NSGA3(ref_dirs=ref, pop_size=pop_size, **_operators(seed, cfg))

    elif name == "moead":
        from pymoo.algorithms.moo.moead import MOEAD
        from pymoo.util.ref_dirs import get_reference_directions
        ref = get_reference_directions("energy", n_obj, pop_size, seed=seed)
        ops = _operators(seed, cfg)
        ops.pop("eliminate_duplicates", None)
        ops.pop("sampling", None)
        alg = MOEAD(ref_dirs=ref,
                    n_neighbors=int(cfg.get("moead_neighbors", 20)),
                    prob_neighbor_mating=float(cfg.get("moead_prob_neighbor", 0.9)),
                    **ops)

    elif name == "de":
        from pymoo.algorithms.soo.nonconvex.de import DE
        from pymoo.operators.sampling.lhs import LHS
        alg = DE(pop_size=pop_size, sampling=LHS(),
                 variant=str(cfg.get("de_variant", "DE/rand/1/bin")),
                 CR=float(cfg.get("de_cr", 0.9)),
                 F=cfg.get("de_f", 0.8))

    elif name == "pso":
        from pymoo.algorithms.soo.nonconvex.pso import PSO
        from pymoo.operators.sampling.lhs import LHS
        alg = PSO(pop_size=pop_size, sampling=LHS(),
                  w=float(cfg.get("pso_w", 0.7)),
                  c1=float(cfg.get("pso_c1", 1.5)),
                  c2=float(cfg.get("pso_c2", 1.5)),
                  adaptive=bool(cfg.get("pso_adaptive", True)))

    elif name == "bayesian":
        alg = None                      # driven directly by Optuna in the runner

    else:                                                   # pragma: no cover
        raise KeyError(name)

    return AlgorithmSpec(name=name, kind=kind, label=label, algorithm=alg,
                         pop_size=pop_size, extra=dict(cfg))
