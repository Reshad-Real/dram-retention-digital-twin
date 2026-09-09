"""
dramdt -- A Digital Twin-Driven Framework for Low-Power DRAM Optimization
using PySpice and Machine Learning.

The package is organised along the seven stages of the proposed method:

    1. models/        DRAM cell design & parameterisation
    2. simulation/    Automated PySpice / NGSpice simulation
    3. doe/, dataset/ Dataset generation & preprocessing
    4. surrogate/     Machine-learning surrogate models
    5. optimization/  Multi-objective optimisation
    6. digital_twin/  Digital twin for real-time prediction
    7. xai/, robustness/, statistics/  Analysis & comparison

Everything is configuration driven: no DRAM-architecture specific constant is
hard-coded in the source tree.  See ``configs/`` for the shipped architectures.
"""

from __future__ import annotations

__version__ = "1.0.0"
__author__ = "dramdt framework"
__license__ = "MIT"

__all__ = [
    "__version__",
    "config",
    "paths",
    "env",
    "seeds",
    "logging_utils",
]
