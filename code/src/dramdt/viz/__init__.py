"""Publication-quality figure generation.

``palette``  the computed, CVD-validated colour system
``style``    matplotlib configuration, figure sizing and multi-format saving
``figures``  every figure in the paper, one function each
"""

from .palette import (CATEGORICAL, SEQUENTIAL_HUE, DIVERGING_PAIR,
                      STATUS_COLORS, validate_palette, design_palette)
from .style import (apply_style, figure_size, save_figure, series_style,
                    COLUMN_WIDTH_IN, PAGE_WIDTH_IN, FORMATS)

__all__ = [
    "CATEGORICAL", "SEQUENTIAL_HUE", "DIVERGING_PAIR", "STATUS_COLORS",
    "validate_palette", "design_palette",
    "apply_style", "figure_size", "save_figure", "series_style",
    "COLUMN_WIDTH_IN", "PAGE_WIDTH_IN", "FORMATS",
]
