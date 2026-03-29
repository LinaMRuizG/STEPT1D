from pydantic import BaseModel, field_validator
from typing import Optional, Literal, Any

class VariableRule(BaseModel):
    label:          Optional[str]  = None
    note:           Optional[str]  = None
    dtype:          Literal["int", "float", "string", "categorical",
                            "dichotomous", "date"]
    nullable:       bool           = True
    fuzzy_match:    bool           = False
    threshold:      int            = 88
    canonical:      list[str]      = []
    encoding:       Optional[dict] = None
    coerce:         Optional[dict] = None
    transformations: list[str]     = []
    invalid_action: Literal["flag", "drop", "raise"] = "flag"
    min:            Optional[float] = None
    max:            Optional[float] = None
    max_length:     Optional[int]   = None
    pattern:        Optional[str]   = None
    raw_format:     Optional[str]   = None
    raw_dtype:      Optional[str]   = None
    output_format:  Optional[str]   = None
    negative_action: Optional[str]  = None
    dedup_key:      bool            = False
    dedup_normalize: bool           = False
    allowed:        Optional[list]  = None
    special_null_values: list            = [] 

class GlobalConfig(BaseModel):
    null_strings:   list[str] = []
    invalid_action: str       = "flag"

class ValidationSchema(BaseModel):
    global_config:  GlobalConfig
    variables:      dict[str, VariableRule]