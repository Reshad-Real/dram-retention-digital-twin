"""Publication-ready output: tables, IEEE-style algorithms and the report.

``tables``     CSV / LaTeX / Markdown export with caption and label handling
``algorithms`` the framework's algorithms as IEEE-style pseudocode
``report``     assembly of the final Markdown/LaTeX report
"""

from .tables import (export_table, TableSpec, format_number, dataframe_to_latex,
                     dataframe_to_markdown)
from .algorithms import ALGORITHM_SPECS, render_algorithm, write_all_algorithms
from .report import ReportBuilder

__all__ = [
    "export_table", "TableSpec", "format_number", "dataframe_to_latex",
    "dataframe_to_markdown", "ALGORITHM_SPECS", "render_algorithm",
    "write_all_algorithms", "ReportBuilder",
]
