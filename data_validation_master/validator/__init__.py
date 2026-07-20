"""
validator package — STEP-T1D Data Validation Master v2
=======================================================
Supports validation_schema_v2.yaml with:
  - map_values transform  (Spanish → English translation)
  - fuzzy + encode pipeline for institution
  - binary_indicator anchor for dichotomous 0/1/null columns
  - year_base / time_base anchors for date and duration variables

Quick start:
    from validator.validator import DataValidator

    v = DataValidator(
        data_path   = "data/raw.csv",
        schema_path = "validation_schema_v2.yaml",
    )
    v.validate().report().save()
"""

from .validator       import DataValidator
from .audit           import AuditLog, Reason
from .rules           import ValidationSchema, VariableRule, GlobalConfig
from .transformations import TRANSFORMATIONS, apply_map_values

__all__ = [
    "DataValidator",
    "AuditLog",
    "Reason",
    "ValidationSchema",
    "VariableRule",
    "GlobalConfig",
    "TRANSFORMATIONS",
    "apply_map_values",
]
