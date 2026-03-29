import yaml
import pandas as pd
from pathlib import Path
from rapidfuzz import process, fuzz

from .rules import ValidationSchema, VariableRule
from .transformations import TRANSFORMATIONS
from .audit import AuditLog


class DataValidator:

    def __init__(
        self,
        data_path: str,
        schema_path: str,
        csv_sep: str = ";",           # REDCap exports from Colombian locale use ";"
        csv_encoding: str = "utf-8",
        csv_skiprows: int = 0,
    ):
        self.df = pd.read_csv(
            data_path,
            sep=csv_sep,
            encoding=csv_encoding,
            skiprows=csv_skiprows if csv_skiprows > 0 else None,
            on_bad_lines="warn",      # warn instead of crash on malformed rows
            low_memory=False,         # avoids dtype guessing issues on large files
            keep_default_na=False
        )
        self.schema  = self._load_schema(schema_path)
        self.audit   = AuditLog()
        self.cleaned = self.df.copy()

        print(f"Loaded data    : {self.df.shape[0]} rows × {self.df.shape[1]} columns")
        print(f"Loaded schema  : {len(self.schema.variables)} variables defined")

    # -------------------------------------------------------------------------
    # Schema loading
    # -------------------------------------------------------------------------

    def _load_schema(self, path: str) -> ValidationSchema:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return ValidationSchema(
            global_config=raw.get("global", {}),
            variables={
                k: VariableRule(**v)
                for k, v in raw["variables"].items()
            }
        )

    # -------------------------------------------------------------------------
    # Global pre-processing
    # -------------------------------------------------------------------------

    def _global_preprocess(self):
        """Replace all null sentinel strings with pd.NA across every column."""
        null_strings = self.schema.global_config.null_strings
        self.cleaned.replace(null_strings, pd.NA, inplace=True)

    # -------------------------------------------------------------------------
    # Transformation pipeline
    # -------------------------------------------------------------------------

    def _apply_transformations(
        self, series: pd.Series, rule: VariableRule
    ) -> pd.Series:
        for t_name in rule.transformations:
            if t_name == "encode":
                series = series.map(rule.encoding)
            elif t_name == "coerce":
                series = series.map(rule.coerce)
            elif t_name in TRANSFORMATIONS:
                series = TRANSFORMATIONS[t_name](series)
            else:
                raise ValueError(
                    f"Unknown transformation '{t_name}' — "
                    f"add it to transformations.py"
                )
        return series

    # -------------------------------------------------------------------------
    # Fuzzy matching
    # -------------------------------------------------------------------------

    def _apply_fuzzy(
        self, series: pd.Series, rule: VariableRule, col: str
    ) -> pd.Series:
        """Map unique raw values to canonical list using fuzzy matching."""
        canonical = rule.canonical
        threshold = rule.threshold

        def _match(val):
            if pd.isna(val):
                return val
            result = process.extractOne(
                val,
                canonical,
                scorer=fuzz.WRatio,
                score_cutoff=threshold,
            )
            if result:
                self.audit.log(
                    col, None, val,
                    reason="fuzzy_corrected",
                    fixed_value=result[0],
                    confidence=result[1],
                )
                return result[0]
            else:
                self.audit.log(col, None, val, reason="fuzzy_no_match")
                return pd.NA

        return series.apply(_match)

    # -------------------------------------------------------------------------
    # Per-column validation
    # -------------------------------------------------------------------------

    def _validate_column(self, col: str, rule: VariableRule):
        series = self.cleaned[col].copy()

        # 1 — apply transformation chain
        series = self._apply_transformations(series, rule)

        # 2 — fuzzy matching for categoricals
        if rule.fuzzy_match and rule.canonical:
            series = self._apply_fuzzy(series, rule, col)

        # 3 — special null values (ADD THIS BLOCK — must run before min/max check)
        if rule.special_null_values:
            sentinel_mask = series.isin(rule.special_null_values)
            if sentinel_mask.any():
                #for idx, val in series[sentinel_mask].items():
                #    self.audit.log(
                #        col=col,
                #        index=idx,
                #        raw_value=val,
                #        reason="sentinel_value_set_to_null",
                #        fixed_value=None,
                #        confidence=None
                #    )
                series = series.where(~sentinel_mask, other=pd.NA)
            
        # 3 — nullability check
        if not rule.nullable:
            nulls = series[series.isna()]
            for idx, val in nulls.items():
                self.audit.log(col, idx, val, reason="unexpected_null")

        # 4 — range checks
        if rule.min is not None:
            out_of_range = series[series.notna() & (series < rule.min)]
            for idx, val in out_of_range.items():
                self.audit.log(col, idx, val, reason="below_min")

        if rule.max is not None:
            out_of_range = series[series.notna() & (series > rule.max)]
            for idx, val in out_of_range.items():
                self.audit.log(col, idx, val, reason="above_max")

        # 5 — max_length check for strings
        if rule.max_length is not None and series.dtype == object:
            too_long = series[series.notna() & (series.str.len() > rule.max_length)]
            for idx, val in too_long.items():
                self.audit.log(col, idx, val, reason="exceeds_max_length")

        # 6 — allowed values check (dichotomous)
        if rule.allowed is not None:
            allowed_with_na = [v for v in rule.allowed if v is not None]
            invalid = series[series.notna() & ~series.isin(allowed_with_na)]
            for idx, val in invalid.items():
                self.audit.log(col, idx, val, reason="invalid_allowed_value")

        # 7 — dedup shadow column (lowercase + no accents key)
        if rule.dedup_normalize:
            import unicodedata
            def _normalize(val):
                if pd.isna(val):
                    return val
                val = val.lower()
                val = unicodedata.normalize("NFD", val)
                val = "".join(c for c in val if unicodedata.category(c) != "Mn")
                return val
            self.cleaned[f"{col}_key"] = series.apply(_normalize)

        self.cleaned[col] = series

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def validate(self) -> "DataValidator":
        """Run global pre-processing then validate every column in the schema."""
        self._global_preprocess()

        missing_cols = []
        for col, rule in self.schema.variables.items():
            if col in self.cleaned.columns:
                self._validate_column(col, rule)
            else:
                missing_cols.append(col)

        if missing_cols:
            print(f"\nWARNING: {len(missing_cols)} schema column(s) not found in data:")
            for c in missing_cols:
                print(f"  - {c}")

        return self   # enables chaining: .validate().report().save()

    def save(
        self,
        output_path: str = "/Users/linaruiz/Documents/STEPT1D/data_validation_master/outputs/cleaned_db.csv",
        audit_path:  str = "/Users/linaruiz/Documents/STEPT1D/data_validation_master/outputs/audit_log.csv",
    ) -> "DataValidator":
        """Save cleaned dataset and audit log to CSV."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(audit_path).parent.mkdir(parents=True, exist_ok=True)

        self.cleaned.to_csv(output_path, index=False, encoding="utf-8")
        self.audit.save(audit_path)

        print(f"\nSaved cleaned data → {output_path}")
        print(f"Saved audit log    → {audit_path}")
        return self

    def report(self) -> "DataValidator":
        """Print a summary of all issues found during validation."""
        df_audit = self.audit.to_dataframe()

        print(f"\n{'='*50}")
        print(f"VALIDATION REPORT")
        print(f"{'='*50}")
        print(f"Rows in dataset  : {len(self.cleaned)}")
        print(f"Total issues     : {len(df_audit)}")

        if not df_audit.empty:
            print(f"\nIssues by reason:")
            summary = df_audit.groupby("reason").size().sort_values(ascending=False)
            for reason, count in summary.items():
                print(f"  {reason:<30} {count}")

            print(f"\nIssues by column (top 10):")
            by_col = df_audit.groupby("column").size().sort_values(ascending=False).head(10)
            for col, count in by_col.items():
                print(f"  {col:<30} {count}")
        else:
            print("No issues found.")

        print(f"{'='*50}\n")
        return self