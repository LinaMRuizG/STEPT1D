"""
audit.py
Accumulates every data quality event during validation and writes
them to audit_log.csv.

Each record captures:
    column      — variable label (snake_case, not raw header)
    row_index   — pandas row index where the issue occurred
    raw_value   — original value before any transformation
    reason      — why it was flagged (see Reason class)
    fixed_value — what it was changed to (None if just flagged)
    confidence  — fuzzy match score 0-100 (only for fuzzy_corrected)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Reason codes — single source of truth for all audit reasons
# ---------------------------------------------------------------------------
class Reason:
    # value is genuinely missing and nullable=false
    UNEXPECTED_NULL         = "unexpected_null"
    # numeric value below rule.min
    BELOW_MIN               = "below_min"
    # numeric value above rule.max
    ABOVE_MAX               = "above_max"
    # string longer than rule.max_length
    EXCEEDS_MAX_LENGTH      = "exceeds_max_length"
    # string does not match rule.pattern regex
    PATTERN_MISMATCH        = "pattern_mismatch"
    # value not in rule.allowed list
    VALUE_NOT_IN_ALLOWED    = "value_not_in_allowed"
    # value in rule.special_null_values → silently becomes pd.NA
    SENTINEL_TO_NULL        = "sentinel_value_set_to_null"
    # negative int/float → silently becomes pd.NA
    NEGATIVE_TO_NULL        = "negative_value_set_to_null"
    # raw text corrected to canonical via fuzzy matching
    FUZZY_CORRECTED         = "fuzzy_corrected"
    # no canonical match at or above threshold → becomes pd.NA
    FUZZY_NO_MATCH          = "fuzzy_no_match"
    # value not found in rule.encoding map
    ENCODE_NO_MATCH         = "encode_no_match"
    # value not found in rule.map_values dict
    MAP_VALUES_NO_MATCH     = "map_values_no_match"
    # schema column not present in the raw data
    COLUMN_MISSING          = "column_in_schema_not_in_data"


# ---------------------------------------------------------------------------
# AuditLog
# ---------------------------------------------------------------------------
@dataclass
class AuditLog:
    """Thread-safe accumulator for validation events."""

    records: list = field(default_factory=list)

    # ── Core logging ─────────────────────────────────────────────────────────

    def log(
        self,
        col:         str,
        index:       Any,
        raw_value:   Any,
        reason:      str,
        fixed_value: Any   = None,
        confidence:  float = None,
    ) -> None:
        """Append one audit record."""
        self.records.append(
            {
                "column":      col,
                "row_index":   index,
                "raw_value":   raw_value,
                "reason":      reason,
                "fixed_value": fixed_value,
                "confidence":  confidence,
            }
        )

    def log_series(
        self,
        col:    str,
        series: pd.Series,
        reason: str,
    ) -> None:
        """Log every non-null value in a filtered series."""
        for idx, val in series.items():
            self.log(col=col, index=idx, raw_value=val, reason=reason)

    # ── Output ───────────────────────────────────────────────────────────────

    def to_dataframe(self) -> pd.DataFrame:
        if self.records:
            return pd.DataFrame(self.records)
        return pd.DataFrame(
            columns=[
                "column", "row_index", "raw_value",
                "reason", "fixed_value", "confidence",
            ]
        )

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_csv(path, index=False, encoding="utf-8")
        print(f"  Audit log saved → {path}")

    def summary(self) -> pd.Series:
        """Count of events grouped by reason, sorted descending."""
        df = self.to_dataframe()
        if df.empty:
            return pd.Series(dtype=int)
        return df.groupby("reason").size().sort_values(ascending=False)

    def summary_by_column(self) -> pd.DataFrame:
        """Count of events grouped by column and reason."""
        df = self.to_dataframe()
        if df.empty:
            return pd.DataFrame()
        return (
            df.groupby(["column", "reason"])
            .size()
            .reset_index(name="count")
            .sort_values(["column", "count"], ascending=[True, False])
        )

    # ── Dunder helpers ────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.records)

    def __repr__(self) -> str:
        return f"AuditLog({len(self.records)} records)"
