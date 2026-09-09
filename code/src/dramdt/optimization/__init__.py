"""Stage 5 -- multi-objective design optimisation driven by the digital twin.

``problem``     the pymoo problem that wraps the surrogate ensemble
``algorithms``  NSGA-II, NSGA-III, MOEA/D, differential evolution, PSO, Bayesian
``indicators``  hypervolume, IGD/IGD+, spacing, spread, epsilon
``runner``      repeated runs, Pareto extraction, indicator tables
"""

from .indicators import (hypervolume, igd, igd_plus, spacing, spread,
                         additive_epsilon, non_dominated, indicator_bundle)
from .problem import DramDesignProblem, ObjectiveSpec, ConstraintSpec
from .algorithms import ALGORITHMS, build_algorithm, algorithm_labels
from .runner import OptimizationRunner, OptimizationResult

__all__ = [
    "hypervolume", "igd", "igd_plus", "spacing", "spread", "additive_epsilon",
    "non_dominated", "indicator_bundle",
    "DramDesignProblem", "ObjectiveSpec", "ConstraintSpec",
    "ALGORITHMS", "build_algorithm", "algorithm_labels",
    "OptimizationRunner", "OptimizationResult",
]
