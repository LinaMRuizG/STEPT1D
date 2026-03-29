import pandas as pd
import regex as re

def trim(series: pd.Series) -> pd.Series:
    return series.str.strip() if series.dtype == object else series

def lowercase(series: pd.Series) -> pd.Series:
    return series.str.lower() if series.dtype == object else series

def collapse_spaces(series: pd.Series) -> pd.Series:
    return series.str.replace(r"\s+", " ", regex=True)

def sanitize(series: pd.Series) -> pd.Series:
    return series.str.replace(r"[\n\r\t]", " ", regex=True)

def to_int(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")

def empty_to_null(series: pd.Series) -> pd.Series:
    return series.replace("", pd.NA)

def negative_to_null(series: pd.Series) -> pd.Series:
    return series.where(series >= 0, other=pd.NA)

def to_date(series: pd.Series, fmt: str = None) -> pd.Series:
    return pd.to_datetime(series, format=fmt, errors="coerce")

def extract_year(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.year.astype("Int64")

def smart_title_case(series: pd.Series) -> pd.Series:
    PARTICLES = {"de", "del", "la", "las", "los", "y", "e", "el", "van", "von"}
    def _title(val):
        if pd.isna(val): return val
        words = val.split()
        return " ".join(
            w.capitalize() if i == 0 or w.lower() not in PARTICLES
            else w.lower()
            for i, w in enumerate(words)
        )
    return series.apply(_title)

# Registry — maps YAML transformation name → function
TRANSFORMATIONS = {
    "trim":             trim,
    "lowercase":        lowercase,
    "uppercase":        lambda s: s.str.upper(),
    "collapse_spaces":  collapse_spaces,
    "sanitize":         sanitize,
    "to_int":           to_int,
    "empty_to_null":    empty_to_null,
    "negative_to_null": negative_to_null,
    "to_date":          to_date,
    "extract_year":     extract_year,
    "smart_title_case": smart_title_case,
}