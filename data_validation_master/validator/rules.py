"""
rules.py
Pydantic v2 models that parse and validate validation_schema_v2.yaml.

Key changes vs previous version:
  - Added `map_values` field (dict) — used by the new map_values transform
  - `encoding_map` and `label_map` kept as documentation-only fields
  - `yn_translate_map`, `month_translation_map` are inline map_values dicts
  - Validators warn loudly on common configuration mistakes

If the YAML has a structural error, Pydantic raises here before
any data is touched.
"""

from __future__ import annotations
from typing import Any, Literal, Optional
from pydantic import BaseModel, model_validator


# ---------------------------------------------------------------------------
# Variable-level rule
# ---------------------------------------------------------------------------

class VariableRule(BaseModel):

    # ── Identity ─────────────────────────────────────────────────────────────
    label:               Optional[str]   = None  # snake_case output column name
    note:                Optional[str]   = None  # human documentation

    # ── Type & nullability ────────────────────────────────────────────────────
    dtype:               Literal[
                             "int", "float", "string",
                             "categorical", "dichotomous", "date",
                         ]
    nullable:            bool            = True

    # ── Value constraints ─────────────────────────────────────────────────────
    allowed:             Optional[list]  = None  # explicit allowed value list
    min:                 Optional[float] = None  # numeric lower bound (inclusive)
    max:                 Optional[float] = None  # numeric upper bound (inclusive)
    max_length:          Optional[int]   = None  # string max character length
    pattern:             Optional[str]   = None  # regex (uses `regex` lib for \p{L})

    # ── Sentinel / special values ─────────────────────────────────────────────
    special_null_values: list            = []    # these values → pd.NA silently
    negative_action:     Optional[str]   = None  # "null" → negatives → pd.NA

    # ── Value mapping (string → string or string → int) ───────────────────────
    # map_values: used by the `map_values` transform step.
    # Values NOT in the map pass through unchanged.
    # Example: {"si": "Yes", "sí": "Yes", "no": "No"}
    map_values:          Optional[dict]  = None

    # ── Encoding (text → int, used by `encode` transform) ─────────────────────
    encoding:            Optional[dict]  = None  # canonical text → int code
    encoding_map:        Optional[dict]  = None  # doc only (int → display text)
    label_map:           Optional[dict]  = None  # alias for encoding_map

    # ── Coerce (text → text or text → int, used by `coerce` transform) ────────
    coerce:              Optional[dict]  = None

    # ── Fuzzy matching ────────────────────────────────────────────────────────
    fuzzy_match:         bool            = False
    threshold:           int             = 88
    canonical:           list[str]       = []

    # ── Transformations ───────────────────────────────────────────────────────
    transformations:     list[str]       = []

    # ── Date handling ─────────────────────────────────────────────────────────
    raw_format:          Optional[str]   = None  # strptime format of raw date
    raw_dtype:           Optional[str]   = None  # e.g. "date_string"
    output_format:       Optional[str]   = None  # strftime format for output

    # ── Deduplication ─────────────────────────────────────────────────────────
    dedup_key:           bool            = False
    dedup_normalize:     bool            = False  # generates <label>_key column

    # ── Error handling ────────────────────────────────────────────────────────
    invalid_action:      Literal["flag", "drop", "raise"] = "flag"

    # allow extra fields from YAML anchors without crashing
    model_config = {"extra": "allow"}

    # ── Validators ───────────────────────────────────────────────────────────

    @model_validator(mode="after")
    def check_fuzzy_needs_canonical(self) -> "VariableRule":
        if self.fuzzy_match and not self.canonical:
            raise ValueError(
                f"Variable '{self.label}': fuzzy_match=true "
                f"but canonical list is empty."
            )
        return self

    @model_validator(mode="after")
    def check_special_null_needs_nullable(self) -> "VariableRule":
        if self.special_null_values and not self.nullable:
            raise ValueError(
                f"Variable '{self.label}': has special_null_values "
                f"but nullable=false. Set nullable=true."
            )
        return self

    @model_validator(mode="after")
    def check_encode_needs_encoding(self) -> "VariableRule":
        if "encode" in self.transformations and self.encoding is None:
            raise ValueError(
                f"Variable '{self.label}': uses 'encode' transform "
                f"but no 'encoding' map is defined."
            )
        return self

    @model_validator(mode="after")
    def check_map_values_needs_dict(self) -> "VariableRule":
        if "map_values" in self.transformations and self.map_values is None:
            raise ValueError(
                f"Variable '{self.label}': uses 'map_values' transform "
                f"but no 'map_values' dict is defined."
            )
        return self

    @model_validator(mode="after")
    def check_coerce_needs_dict(self) -> "VariableRule":
        if "coerce" in self.transformations and self.coerce is None:
            raise ValueError(
                f"Variable '{self.label}': uses 'coerce' transform "
                f"but no 'coerce' dict is defined."
            )
        return self


# ---------------------------------------------------------------------------
# Global config
# ---------------------------------------------------------------------------

class GlobalConfig(BaseModel):
    null_strings:   list[str] = []
    invalid_action: str       = "flag"

    model_config = {"extra": "allow"}

# ---------------------------------------------------------------------------
# Output variables
# ---------------------------------------------------------------------------
class OutputConfig(BaseModel):
    keep_columns: list[str] = []
    drop_columns: list[str] = []

    model_config = {"extra": "allow"}


# ---------------------------------------------------------------------------
# Top-level schema
# ---------------------------------------------------------------------------

class ValidationSchema(BaseModel):
    global_config: GlobalConfig
    variables:     dict[str, VariableRule]
    output:        OutputConfig = OutputConfig() 
    
    model_config = {"extra": "allow"}
