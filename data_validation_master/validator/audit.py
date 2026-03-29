import pandas as pd
from dataclasses import dataclass, field

@dataclass
class AuditLog:
    records: list = field(default_factory=list)

    def log(self, col: str, index, raw_value, reason: str,
            fixed_value=None, confidence: float = None):
        self.records.append({
            "column":      col,
            "row_index":   index,
            "raw_value":   raw_value,
            "reason":      reason,
            "fixed_value": fixed_value,
            "confidence":  confidence,
        })

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)

    def save(self, path: str):
        self.to_dataframe().to_csv(path, index=False)