"""Stage 4 -- machine-learning surrogate models.

``metrics``  regression scores with uncertainty
``zoo``      the model registry (linear, kernel, ensemble, boosting, ANN, GPR)
``train``    cross-validated training, comparison, tuning and persistence
"""

from .metrics import regression_metrics, bootstrap_ci, METRIC_NAMES
from .zoo import MODEL_REGISTRY, build_model, build_preprocessor, model_families
from .train import SurrogateTrainer, TrainedSurrogate, SurrogateComparison

__all__ = [
    "regression_metrics", "bootstrap_ci", "METRIC_NAMES",
    "MODEL_REGISTRY", "build_model", "build_preprocessor", "model_families",
    "SurrogateTrainer", "TrainedSurrogate", "SurrogateComparison",
]
