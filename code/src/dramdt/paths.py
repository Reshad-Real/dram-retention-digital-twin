"""Canonical project paths.

Every other module resolves file locations through this module so that the
whole framework can be relocated by setting the ``DRAMDT_ROOT`` environment
variable (used by the unit tests to run against a temporary tree).
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "PROJECT_ROOT", "CONFIG_DIR", "DATA_DIR", "RAW_DIR", "INTERIM_DIR",
    "PROCESSED_DIR", "EXTERNAL_DIR", "SPICE_MODEL_DIR", "MODELS_DIR",
    "RESULTS_DIR", "FIGURES_DIR", "TABLES_DIR", "OPT_DIR", "XAI_DIR",
    "ROBUSTNESS_DIR", "STATS_DIR", "SURROGATE_DIR", "REPORTS_DIR", "LOGS_DIR",
    "DOCS_DIR", "SUPPLEMENTARY_DIR", "TOOLS_DIR", "NOTEBOOKS_DIR",
    "ensure_directories", "rel",
]


def _root() -> Path:
    env = os.environ.get("DRAMDT_ROOT")
    if env:
        return Path(env).resolve()
    # src/dramdt/paths.py -> src/dramdt -> src -> <root>
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT: Path = _root()

CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"
SPICE_MODEL_DIR = DATA_DIR / "spice_models"

MODELS_DIR = PROJECT_ROOT / "models"

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
OPT_DIR = RESULTS_DIR / "optimization"
XAI_DIR = RESULTS_DIR / "xai"
ROBUSTNESS_DIR = RESULTS_DIR / "robustness"
STATS_DIR = RESULTS_DIR / "statistics"
SURROGATE_DIR = RESULTS_DIR / "surrogate"

REPORTS_DIR = PROJECT_ROOT / "reports"
LOGS_DIR = PROJECT_ROOT / "logs"
DOCS_DIR = PROJECT_ROOT / "docs"
SUPPLEMENTARY_DIR = PROJECT_ROOT / "supplementary"
TOOLS_DIR = PROJECT_ROOT / "tools"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

_ALL_DIRS = [
    CONFIG_DIR, DATA_DIR, RAW_DIR, INTERIM_DIR, PROCESSED_DIR, EXTERNAL_DIR,
    SPICE_MODEL_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR, TABLES_DIR, OPT_DIR,
    XAI_DIR, ROBUSTNESS_DIR, STATS_DIR, SURROGATE_DIR, REPORTS_DIR, LOGS_DIR,
    DOCS_DIR, SUPPLEMENTARY_DIR, TOOLS_DIR, NOTEBOOKS_DIR,
]


def ensure_directories() -> None:
    """Create the full output tree if it does not already exist."""
    for d in _ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


def rel(p: os.PathLike | str) -> str:
    """Return *p* relative to the project root (for tidy log/report output)."""
    p = Path(p)
    try:
        return str(p.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)
