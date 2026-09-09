"""Stage 3a -- design of experiments.

``sampling``  Latin Hypercube / Sobol / random / grid designs over the unit cube
"""

from .sampling import (generate_design, lhs, sobol, random_uniform, grid,
                       DesignMatrix, discrepancy, min_pairwise_distance)

__all__ = ["generate_design", "lhs", "sobol", "random_uniform", "grid",
           "DesignMatrix", "discrepancy", "min_pairwise_distance"]
