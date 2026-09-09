"""Stage 8 -- statistical validation of the comparisons.

``tests``  normality, omnibus, post-hoc, effect size, bootstrap intervals
"""

from .tests import (StatisticalReport, compare_algorithms, friedman_test,
                    nemenyi_posthoc, wilcoxon_holm, cliffs_delta,
                    vargha_delaney_a12, bootstrap_mean_ci, normality,
                    critical_difference, descriptive_statistics)

__all__ = [
    "StatisticalReport", "compare_algorithms", "friedman_test",
    "nemenyi_posthoc", "wilcoxon_holm", "cliffs_delta", "vargha_delaney_a12",
    "bootstrap_mean_ci", "normality", "critical_difference",
    "descriptive_statistics",
]
