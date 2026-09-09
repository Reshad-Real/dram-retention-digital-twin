"""Stage 2 -- automated PySpice / NGSpice simulation.

``runner``      deck execution (batch subprocess + PySpice shared library)
``measure``     raw NGSpice output -> physical DRAM metrics
``retention``   quasi-static and direct-transient retention models
``montecarlo``  device-mismatch Monte-Carlo campaigns
``corners``     process/voltage/temperature corner sweeps
"""

from .runner import (SpiceRunner, SimulationResult, SpiceError,
                     parse_measurements, parse_leakage_file)
from .measure import MetricExtractor, DramMetrics
from .retention import (quasistatic_retention, retention_fail_level,
                        RetentionResult)

__all__ = [
    "SpiceRunner", "SimulationResult", "SpiceError",
    "parse_measurements", "parse_leakage_file",
    "MetricExtractor", "DramMetrics",
    "quasistatic_retention", "retention_fail_level", "RetentionResult",
]
