"""Stage 4b -- explainable AI for the surrogate models.

``explain``  SHAP values, permutation importance, partial dependence and ALE
"""

from .explain import (SurrogateExplainer, ExplanationBundle,
                      permutation_importance_frame, partial_dependence_frame,
                      ale_frame)

__all__ = ["SurrogateExplainer", "ExplanationBundle",
           "permutation_importance_frame", "partial_dependence_frame",
           "ale_frame"]
