"""Stage 3 -- dataset generation, preprocessing and publication export.

``campaign``    parallel NGSpice sampling campaign with checkpointing
``preprocess``  cleaning, outlier handling, feature engineering, splitting
``export``      Mendeley-Data-ready Excel/CSV release of the dataset
"""

from .campaign import (SimulationCampaign, CampaignResult, simulate_one,
                       RECORD_PROVENANCE_FIELDS)
from .preprocess import Preprocessor, PreprocessReport
from .export import export_dataset, DatasetExportReport

__all__ = [
    "SimulationCampaign", "CampaignResult", "simulate_one",
    "RECORD_PROVENANCE_FIELDS",
    "Preprocessor", "PreprocessReport",
    "export_dataset", "DatasetExportReport",
]
