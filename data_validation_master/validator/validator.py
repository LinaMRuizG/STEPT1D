"""
validator.py
Core DataValidator class for validation_schema_v2.yaml.

Usage:
    from validator.validator import DataValidator

    v = DataValidator(
        data_path   = "data/raw.csv",
        schema_path = "validation_schema_v2.yaml",
    )
    v.validate().report().save()

Per-column pipeline (executed in this order):
    1.  Global preprocess   — null sentinel strings → pd.NA (object cols only)
    2.  Pre-encode transforms — trim, lowercase, smart_title_case, etc.
    3.  Fuzzy matching      — correct typos against canonical list (categorical)
    4.  Post-encode transforms — encode (text → int) if present
    5.  Special null values — sentinel ints/strings → pd.NA (not errors)
    6.  Nullability check   — flag unexpected pd.NA when nullable=false
    7.  Allowed values      — flag values outside rule.allowed
    8.  Range check         — flag values outside rule.min / rule.max
    9.  Max-length check    — flag strings exceeding rule.max_length
    10. Pattern check       — flag strings failing rule.pattern regex
    11. Dedup shadow cols   — generate <label>_key for deduplication
    12. Column rename       — rename raw header to rule.label in output
"""

from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml
from rapidfuzz import fuzz, process

from .audit import AuditLog, Reason
from .rules import GlobalConfig, ValidationSchema, VariableRule
from .transformations import TRANSFORMATIONS, apply_map_values

# Project root = two levels above this file
# STEPT1D/data_validation_master/validator/validator.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = Path.cwd() / "outputs"


class DataValidator:
    """
    Reads a raw CSV and a validation schema YAML, applies all rules,
    and produces a cleaned CSV + full audit log.
    """

    def __init__(
        self,
        data_path:    str,
        schema_path:  str,
        csv_sep:      str = ";",
        csv_encoding: str = "utf-8",
    ) -> None:
        print(f"  Loading data   → {data_path}")
        print(f"  Loading schema → {schema_path}")

        self.df = pd.read_csv(
            data_path,
            sep=csv_sep,
            encoding=csv_encoding,
            on_bad_lines="warn",
            low_memory=False,
            keep_default_na=False,   # prevent pandas auto-NA; we do it ourselves
            dtype=str,          # ADD THIS — read everything as string first
                        # the validator then converts to the right typ
        )
        print(f"  Raw shape: {self.df.shape}")

        self.schema  = self._load_schema(schema_path)
        self.audit   = AuditLog()
        self.cleaned = self.df.copy()

    # =========================================================================
    # Schema loading
    # =========================================================================

    def _load_schema(self, path: str) -> ValidationSchema:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        global_cfg = GlobalConfig(**raw.get("global", {}))

        variables: dict[str, VariableRule] = {}
        errors = 0
        for key, cfg in raw.get("variables", {}).items():
            if not isinstance(cfg, dict):
                print(f"  WARNING: '{key}' has no config dict — skipped.")
                continue
            try:
                variables[key] = VariableRule(**cfg)
            except Exception as e:
                print(f"  ERROR parsing '{key}': {e}")
                errors += 1

        if errors:
            print(f"\n  {errors} schema variable(s) failed to parse — fix before running.\n")

        return ValidationSchema(global_config=global_cfg, variables=variables)

    # =========================================================================
    # Step 1 — Global preprocessor
    # =========================================================================


    def _global_preprocess(self) -> None:
        """
        Step 1a — replace null sentinel strings with pd.NA (object/string cols only).
        Step 1b — replace float NaN/None with pd.NA safely.
        """
        null_strings = self.schema.global_config.null_strings

        # Target string/object columns explicitly to prevent float promotion
        str_cols = self.cleaned.select_dtypes(include=["object", "string"]).columns

        # Step 1a: Replace null strings
        if null_strings and len(str_cols) > 0:
            self.cleaned[str_cols] = self.cleaned[str_cols].replace(null_strings, pd.NA)

        # Step 1b: Replace NaNs safely column by column without global float casting
        for col in self.cleaned.columns:
            self.cleaned[col] = self.cleaned[col].where(pd.notna(self.cleaned[col]), None)

    # =========================================================================
    # Steps 2 & 4 — Transformation engine
    # =========================================================================

    def _apply_single_transform(
        self,
        series: pd.Series,
        rule:   VariableRule,
        t_name: str,
    ) -> pd.Series:
        """
        Apply one named transformation to a Series.

        Three transforms require rule-level context and are handled here
        rather than in transformations.py:
            encode      — uses rule.encoding  (text → int)
            map_values  — uses rule.map_values (text → text or text → int)
            coerce      — uses rule.coerce     (text → text or text → int,
                                                values not in map pass through)
        All others are dispatched to the TRANSFORMATIONS registry.
        """
        label = rule.label or "unknown"

        # ── encode: text → int via rule.encoding ─────────────────────────────
        if t_name == "encode":
            encoded   = series.map(rule.encoding)
            no_match  = series[series.notna() & encoded.isna()]
            for idx, val in no_match.items():
                self.audit.log(
                    col=label, index=idx, raw_value=val,
                    reason=Reason.ENCODE_NO_MATCH,
                )
            return encoded

        # ── map_values: text → text via rule.map_values ──────────────────────
        if t_name == "map_values":
            mapping = rule.map_values or {}
            result  = apply_map_values(series, mapping)
            # log values that had no match in the map (they pass through
            # unchanged and will be caught by the allowed check later)
            if mapping:
                no_match = series[series.notna() & ~series.isin(mapping.keys())]
                for idx, val in no_match.items():
                    self.audit.log(
                        col=label, index=idx, raw_value=val,
                        reason=Reason.MAP_VALUES_NO_MATCH,
                    )
            return result

        # ── coerce: text → text, values not in map pass through ──────────────
        if t_name == "coerce":
            mapping  = rule.coerce or {}
            mapped   = series.map(mapping)
            in_map   = series.isin(mapping.keys())
            # keep original where not in map
            return series.where(~in_map, other=mapped)

        # ── standard transforms from registry ────────────────────────────────
        if t_name in TRANSFORMATIONS:
            return TRANSFORMATIONS[t_name](series)

        raise ValueError(
            f"Unknown transformation '{t_name}' on label='{label}'. "
            f"Add it to transformations.py and the TRANSFORMATIONS dict."
        )

    def _run_transforms(
        self,
        series:      pd.Series,
        rule:        VariableRule,
        stop_before: Optional[str] = None,
        start_at:    Optional[str] = None,
    ) -> pd.Series:
        """
        Run the rule's transformation list, optionally:
          stop_before — stop when this transform name is reached
          start_at    — skip transforms until this name is reached
        Used to split pre-fuzzy and post-fuzzy transforms cleanly.
        """
        found = (start_at is None)   # if no start_at, begin immediately

        for t_name in rule.transformations:
            if start_at and t_name == start_at:
                found = True
            if not found:
                continue
            if stop_before and t_name == stop_before:
                break
            series = self._apply_single_transform(series, rule, t_name)

        return series

    # =========================================================================
    # Step 3 — Fuzzy matching
    # =========================================================================

    def _apply_fuzzy(self, series, rule, col):
        import unicodedata

        def _strip(text: str) -> str:
            """Remove accents for comparison only."""
            nfkd = unicodedata.normalize("NFKD", str(text).lower())
            return "".join(c for c in nfkd if not unicodedata.combining(c))

        canonical      = rule.canonical
        # build a map: accent-stripped canonical → original canonical
        # e.g. "peru" → "Perú", "colombia" → "Colombia"
        stripped_map   = {_strip(c): c for c in canonical}
        stripped_canon = list(stripped_map.keys())

        threshold = rule.threshold
        label     = rule.label or col
        cache: dict = {}

        for val in series.dropna().unique():
            val_stripped = _strip(str(val))     # "Peru" → "peru"

            result = process.extractOne(
                val_stripped,
                stripped_canon,                 # compare against ["peru", "colombia", ...]
                scorer=fuzz.WRatio,
                score_cutoff=threshold,
            )
            if result:
                matched_stripped, score, _ = result
                matched_original = stripped_map[matched_stripped]  # "peru" → "Perú"
                cache[val] = matched_original
                if str(val) != matched_original:
                    self.audit.log(
                        col=label, index=None,
                        raw_value=val,
                        reason=Reason.FUZZY_CORRECTED,
                        fixed_value=matched_original,
                        confidence=round(score, 2),
                    )
            else:
                cache[val] = pd.NA
                self.audit.log(
                    col=label, index=None,
                    raw_value=val,
                    reason=Reason.FUZZY_NO_MATCH,
                )

        return series.map(cache)
    # =========================================================================
    # Step 5 — Special null values
    # =========================================================================

    def _apply_special_nulls(
        self,
        series: pd.Series,
        rule:   VariableRule,
        col:    str,
    ) -> pd.Series:
        """
        Convert known sentinel values (e.g. 1800, 99) to pd.NA.
        These are NOT data errors — they are agreed-upon codes for "unknown".
        Logged as sentinel_value_set_to_null for traceability.

        Must run AFTER encode/map_values so integer codes are already in place.
        """
        if not rule.special_null_values:
            return series

        label         = rule.label or col
        sentinel_mask = series.isin(rule.special_null_values)

        for idx, val in series[sentinel_mask].items():
            self.audit.log(
                col=label, index=idx,
                raw_value=val,
                reason=Reason.SENTINEL_TO_NULL,
            )

        return series.where(~sentinel_mask, other=pd.NA)

    # =========================================================================
    # Steps 6–10 — Validation checks (read-only, only log — don't mutate)
    # =========================================================================

    def _check_nullable(
        self, series: pd.Series, rule: VariableRule, col: str
    ) -> None:
        """Flag rows where the value is pd.NA but nullable=false."""
        if rule.nullable:
            return
        label = rule.label or col
        for idx, val in series[series.isna()].items():
            self.audit.log(
                col=label, index=idx,
                raw_value=val, reason=Reason.UNEXPECTED_NULL,
            )

    def _check_allowed(
        self, series: pd.Series, rule: VariableRule, col: str
    ) -> None:
        """Flag values not present in rule.allowed."""
        if not rule.allowed:
            return
        label        = rule.label or col
        allowed_vals = [v for v in rule.allowed if v is not None]
        invalid      = series[series.notna() & ~series.isin(allowed_vals)]
        for idx, val in invalid.items():
            self.audit.log(
                col=label, index=idx,
                raw_value=val, reason=Reason.VALUE_NOT_IN_ALLOWED,
            )

    def _check_range(
        self, series: pd.Series, rule: VariableRule, col: str
    ) -> None:
        """Flag numeric values outside [rule.min, rule.max] without modifying the series."""
        if rule.min is None and rule.max is None:
            return
        label = rule.label or col
        
        # Read-only numeric evaluation for validation logging
        numeric = pd.to_numeric(series, errors="coerce")

        if rule.min is not None:
            below = series[numeric.notna() & (numeric < rule.min)]
            for idx, val in below.items():
                self.audit.log(
                    col=label, index=idx,
                    raw_value=val, reason=Reason.BELOW_MIN,
                )

        if rule.max is not None:
            above = series[numeric.notna() & (numeric > rule.max)]
            for idx, val in above.items():
                self.audit.log(
                    col=label, index=idx,
                    raw_value=val, reason=Reason.ABOVE_MAX,
                )

    def _check_max_length(
        self, series: pd.Series, rule: VariableRule, col: str
    ) -> None:
        """Flag strings longer than rule.max_length."""
        if not rule.max_length or series.dtype != object:
            return
        label    = rule.label or col
        too_long = series[series.notna() & (series.str.len() > rule.max_length)]
        for idx, val in too_long.items():
            self.audit.log(
                col=label, index=idx,
                raw_value=val, reason=Reason.EXCEEDS_MAX_LENGTH,
            )

    def _check_pattern(
        self, series: pd.Series, rule: VariableRule, col: str
    ) -> None:
        """Flag strings that do not match rule.pattern regex."""
        if not rule.pattern or series.dtype != object:
            return
        label = rule.label or col

        try:
            import regex as _re
        except ImportError:
            import re as _re      # fallback: no \p{L} support

        def _fails(val) -> bool:
            if pd.isna(val):
                return False
            return not bool(_re.match(rule.pattern, str(val)))

        mismatched = series[series.apply(_fails)]
        for idx, val in mismatched.items():
            self.audit.log(
                col=label, index=idx,
                raw_value=val, reason=Reason.PATTERN_MISMATCH,
            )

    # =========================================================================
    # Step 11 — Deduplication shadow columns
    # =========================================================================

    def _generate_dedup_key(self, target_col: str, rule: VariableRule) -> None:
        """
        Generate a <label>_key shadow column for deduplication.
        The key is lowercase + accent-stripped so "María" == "Maria".
        Only generated when dedup_key=true AND dedup_normalize=true.
        """
        if not (rule.dedup_key and rule.dedup_normalize and rule.label):
            return

        def _normalize(val) -> str:
            if pd.isna(val):
                return val
            nfkd = unicodedata.normalize("NFKD", str(val).lower())
            return "".join(c for c in nfkd if not unicodedata.combining(c))

        key_col = f"{rule.label}_key"
        self.cleaned[key_col] = self.cleaned[target_col].apply(_normalize)

    # =========================================================================
    # Per-column orchestrator
    # =========================================================================

    def _validate_column(self, col: str, rule: VariableRule) -> None:
        """
        Run the full validation pipeline for one column.

        Transform order is split around fuzzy matching:
            pre-fuzzy:  everything in rule.transformations up to (not including) 'encode'
            fuzzy:      applied on cleaned text
            post-fuzzy: 'encode' and anything after it

        The cleaned result is written back to self.cleaned.
        If rule.label differs from col, the column is renamed.
        """
        series = self.cleaned[col].copy()

        # ── Step 2: pre-fuzzy transforms ─────────────────────────────────────
        # run everything before 'encode' (trim, lowercase, map_values, etc.)
        series = self._run_transforms(series, rule, stop_before="encode")

        # ── Step 3: fuzzy matching ────────────────────────────────────────────
        if rule.fuzzy_match and rule.canonical:
            series = self._apply_fuzzy(series, rule, col)

        # ── Step 4: post-fuzzy transforms ─────────────────────────────────────
        # run 'encode' and everything after it
        series = self._run_transforms(series, rule, start_at="encode")

        # ── Step 5: special null values ───────────────────────────────────────
        series = self._apply_special_nulls(series, rule, col)

        # ── Step 6: nullability ───────────────────────────────────────────────
        self._check_nullable(series, rule, col)

        # ── Step 7: allowed values ────────────────────────────────────────────
        self._check_allowed(series, rule, col)

        # ── Step 8: range ─────────────────────────────────────────────────────
        self._check_range(series, rule, col)

        # ── Step 9: max length ────────────────────────────────────────────────
        self._check_max_length(series, rule, col)

        # ── Step 10: pattern ──────────────────────────────────────────────────
        self._check_pattern(series, rule, col)

        # ── Step 12: write back + rename ──────────────────────────────────────
        target_col = rule.label if rule.label else col
        if target_col != col and target_col not in self.cleaned.columns:
            self.cleaned.rename(columns={col: target_col}, inplace=True)
        self.cleaned[target_col] = series

        # ── Step 11: dedup shadow column ──────────────────────────────────────
        self._generate_dedup_key(target_col, rule)

    # =========================================================================
    # Public API
    # =========================================================================

    def validate(self) -> "DataValidator":
        """
        Run the full validation pipeline.
        Returns self so calls can be chained: .validate().report().save()
        """
        print("\n── Validation started ──────────────────────────────────────")

        # Step 1: global null-string replacement
        self._global_preprocess()
        print("  ✓ Global preprocessing complete")

        # Steps 2–12: per-column
        missing_cols  = []
        processed     = 0

        for col, rule in self.schema.variables.items():
            if col in self.cleaned.columns:
                self._validate_column(col, rule)
                processed += 1
            else:
                missing_cols.append(col)
                self.audit.log(
                    col=col, index=None, raw_value=None,
                    reason=Reason.COLUMN_MISSING,
                )

        print(f"  ✓ {processed} column(s) validated")

        if missing_cols:
            print(f"\n  ⚠ {len(missing_cols)} schema column(s) not found in data:")
            for c in missing_cols:
                print(f"    · {c}")

        print(f"\n  ✓ Done — {len(self.audit)} audit event(s) logged")
        print("────────────────────────────────────────────────────────────\n")
        return self

    def report(self) -> "DataValidator":
        """Print a grouped summary of audit events to stdout."""
        summary = self.audit.summary()
        width   = 55

        print(f"\n{'='*width}")
        print(f"  AUDIT SUMMARY — {len(self.audit)} total event(s)")
        print(f"{'='*width}")

        if summary.empty:
            print("  No issues found — data looks clean!")
        else:
            print(f"  {'Reason':<38} {'Count':>6}")
            print(f"  {'-'*38} {'-'*6}")
            for reason, count in summary.items():
                print(f"  {reason:<38} {count:>6}")

        print(f"{'='*width}\n")
        return self

    def report_by_column(self) -> "DataValidator":
        """Print audit events grouped by column then reason."""
        df = self.audit.summary_by_column()
        if df.empty:
            print("No audit events.")
            return self

        current_col = None
        for _, row in df.iterrows():
            if row["column"] != current_col:
                current_col = row["column"]
                print(f"\n  {current_col}")
            print(f"    {row['reason']:<38} {row['count']:>4}")
        print()
        return self

    def save_old(
        self,
        output_path: Optional[str] = None,
        audit_path:  Optional[str] = None,
    ) -> "DataValidator":
        """Save the cleaned dataset and audit log as CSV files."""
        output_path = output_path or str(
            PROJECT_ROOT / "outputs" / "cleaned_db.csv"
        )
        audit_path = audit_path or str(
            PROJECT_ROOT / "outputs" / "audit_log.csv"
        )

        # create directories if they don't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(audit_path).parent.mkdir(parents=True, exist_ok=True)

        self.cleaned.to_csv(output_path, index=False, encoding="utf-8")
        self.audit.save(audit_path)

        print(f"  Cleaned data saved → {output_path}")
        print(f"  Cleaned shape:       {self.cleaned.shape}")
        return self
    
    def save(
        self,
        output_path: Optional[str] = None,
        audit_path: Optional[str] = None,
    ) -> "DataValidator":
        """Save the cleaned dataset and audit log as CSV files."""
        output_path = output_path or str(DEFAULT_OUTPUT_DIR / "cleaned_db.csv")
        audit_path = audit_path or str(DEFAULT_OUTPUT_DIR / "audit_log.csv")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(audit_path).parent.mkdir(parents=True, exist_ok=True)

        # Ensure integer-like strings (e.g. "2018") are saved cleanly as text without .0
        export_df = self.cleaned.copy()
        for col, rule in self.schema.variables.items():
            target_col = rule.label if rule.label else col
            if target_col in export_df.columns and rule.dtype == "string":
                # Format numbers safely as integers without decimals
                export_df[target_col] = export_df[target_col].apply(
                    lambda x: str(int(float(x))) if pd.notna(x) and str(x).replace('.0','').isdigit() else x
                )

        export_df.to_csv(output_path, index=False, encoding="utf-8")
        self.audit.save(audit_path)

        print(f" Cleaned data saved -> {output_path}")
        print(f" Cleaned shape: {self.cleaned.shape}")
        return self

    def get_cleaned(self) -> pd.DataFrame:
        """Return the cleaned DataFrame (copy)."""
        return self.cleaned.copy()

    def get_audit(self) -> pd.DataFrame:
        """Return the audit log as a DataFrame."""
        return self.audit.to_dataframe()
