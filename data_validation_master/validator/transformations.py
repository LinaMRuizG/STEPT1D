"""
transformations.py
Pure transformation functions: pd.Series → pd.Series.

Rules:
  - Every function takes a Series and returns a Series.
  - No side effects, no audit logging — that happens in validator.py.
  - All functions must handle pd.NA / NaN gracefully without crashing.

To add a new transformation:
  1. Write the function here.
  2. Register it in the TRANSFORMATIONS dict at the bottom.
  3. Use its key string in the YAML transformations list.
"""

from __future__ import annotations
import unicodedata
import pandas as pd
import re


# ---------------------------------------------------------------------------
# String cleaning
# ---------------------------------------------------------------------------

def to_str(series: pd.Series) -> pd.Series:
    """
    Convert numeric values to their string representation.
    Handles float → string before map_values can match them.

    0.0   → "0.0"
    1.0   → "1.0"
    0     → "0"
    1     → "1"
    pd.NA → pd.NA   (preserves nulls)
    NaN   → pd.NA   (converts float NaN to pd.NA)
    """
    def _convert(val):
        if pd.isna(val):
            return pd.NA
        return str(val)

    return series.apply(_convert).astype(object)

def trim(series: pd.Series) -> pd.Series:
    """Remove leading and trailing whitespace from string columns."""
    if series.dtype == object:
        return series.str.strip()
    return series


def uppercase(series: pd.Series) -> pd.Series:
    """Convert strings to UPPERCASE."""
    if series.dtype == object:
        return series.str.upper()
    return series


def lowercase(series: pd.Series) -> pd.Series:
    """Convert strings to lowercase."""
    if series.dtype == object:
        return series.str.lower()
    return series


def collapse_spaces(series: pd.Series) -> pd.Series:
    """Replace multiple consecutive whitespace characters with a single space."""
    if series.dtype == object:
        return series.str.replace(r"\s+", " ", regex=True)
    return series


def sanitize(series: pd.Series) -> pd.Series:
    """
    Remove control characters that corrupt text fields.
    Replaces \\n \\r \\t with a single space, then collapses runs of spaces.
    Must run BEFORE collapse_spaces.
    """
    if series.dtype == object:
        series = series.str.replace(r"[\n\r\t\x00-\x1f]", " ", regex=True)
        series = series.str.replace(r"\s+", " ", regex=True)
    return series


def empty_to_null(series: pd.Series) -> pd.Series:
    """
    Convert empty strings to pd.NA.

    Why needed even with global null_strings:
      The global preprocessor catches literal "" before transformations run.
      But trim() can produce new empty strings from values like "   ".
      This step catches those post-trim empties.
    """
    if series.dtype == object:
        return series.replace("", pd.NA)
    return series


def smart_title_case(series: pd.Series) -> pd.Series:
    """
    Capitalize the first letter of each word, respecting Spanish particles.

    Particles (de, del, la, las, los, y, e, el, van, von, a, al, en, con, sin)
    stay lowercase unless they are the first word in the string.

    Examples:
        "MARIA DE LOS ANGELES" → "Maria de los Angeles"
        "JUAN DEL CARMEN"      → "Juan del Carmen"
        "si"                   → "Si"
        "NO"                   → "No"
    """
    PARTICLES = {
        "de", "del", "la", "las", "los", "y", "e", "el",
        "van", "von", "a", "al", "en", "con", "sin",
    }

    def _title(val: str) -> str:
        if pd.isna(val):
            return val
        words = str(val).split()
        result = []
        for i, word in enumerate(words):
            lower_word = word.lower()
            if i == 0 or lower_word not in PARTICLES:
                result.append(word.capitalize())
            else:
                result.append(lower_word)
        return " ".join(result)

    if series.dtype == object:
        return series.apply(lambda v: _title(v) if not pd.isna(v) else v)
    return series


# ---------------------------------------------------------------------------
# Value mapping (string → string or string → int)
# ---------------------------------------------------------------------------

def apply_map_values(series: pd.Series, mapping: dict) -> pd.Series:
    """
    Replace values using a dictionary.
    Values NOT present in the mapping pass through unchanged.
    pd.NA passes through unchanged.

    Used by the validator when it sees the 'map_values' transform name.
    The mapping dict comes from rule.map_values.

    Examples (with yn_translate_map):
        "si"  → "Yes"
        "sí"  → "Yes"
        "no"  → "No"
        "xyz" → "xyz"  (not in map, passes through → caught by allowed check)
        pd.NA → pd.NA

    Examples (with month_translation_map):
        "Septiembre" → "September"
        "Enero"      → "January"
    """
    if series.dtype != object:
        return series

    mapped  = series.map(mapping)
    in_map  = series.isin(mapping.keys())

    # where the value was in the map → use mapped value
    # where it was not → keep original (pass-through)
    return series.where(~in_map, other=mapped)


# ---------------------------------------------------------------------------
# Numeric conversion
# ---------------------------------------------------------------------------

def to_int(series: pd.Series) -> pd.Series:
    """
    Convert to nullable integer (Int64).

    Handles:
        0.0   → 0
        1.0   → 1
        "1"   → 1
        nan   → pd.NA
        pd.NA → pd.NA

    Uses pd.to_numeric(errors='coerce') so unparseable strings → pd.NA.
    Int64 (capital I) supports pd.NA natively; plain int does not.
    """
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def to_float(series: pd.Series) -> pd.Series:
    """Convert to nullable float (Float64)."""
    return pd.to_numeric(series, errors="coerce").astype("Float64")


def negative_to_null(series: pd.Series) -> pd.Series:
    """
    Replace negative numeric values with pd.NA.

    Used for time-since variables where negative = data entry error.
    Must run AFTER to_int or to_float so the series is already numeric.

    Pipeline:
        raw "-5" → to_int → -5 → negative_to_null → pd.NA  (logged in audit)
        raw "3"  → to_int →  3 → negative_to_null →  3     ✓
    """
    numeric = pd.to_numeric(series, errors="coerce")
    result  = numeric.where(numeric >= 0, other=pd.NA)
    return result.astype("Int64")


# ---------------------------------------------------------------------------
# Date conversion
# ---------------------------------------------------------------------------

def to_date(series: pd.Series, fmt: str = None) -> pd.Series:
    """
    Parse string dates to datetime64[ns].

    errors='coerce' ensures unparseable values (including pd.NA, "nan")
    become NaT without crashing the pipeline.

    The fmt parameter is passed from rule.raw_format when the validator
    calls this function. If None, pandas infers the format (slower but safe).
    """
    return pd.to_datetime(series, format=fmt, errors="coerce")


def extract_year(series: pd.Series) -> pd.Series:
    """
    Extracts the 4-digit year as a clean string ("YYYY") from date strings.

    "2018-01-01" -> "2018"
    "1800-01-01" -> "1800" (sentinel, handled by special_null_values)
    "nan"        -> None / pd.NA
    """
    # 1. Parse dates safely
    parsed = pd.to_datetime(series, format="%Y-%m-%d", errors="coerce")
    
    # 2. Format valid dates to 4-digit YYYY strings, leaves NaT as NaN
    years_str = parsed.dt.strftime("%Y")
    
    # 3. Return as object series with Python None for missing values
    return years_str.astype("object").where(pd.notna(years_str), None)

def to_none(series: pd.Series) -> pd.Series:
    """
    Replaces pd.NA, np.nan, NaT, and string representations of nulls with Python None objects.
    Note: Converting to object dtype is necessary to hold Python None.
    """
    # Replace pd.NA / NaN values with None
    return series.astype("object").where(pd.notna(series), None)

# ---------------------------------------------------------------------------
# Domain-specific extraction
# ---------------------------------------------------------------------------

def extract_month_name(series: pd.Series) -> pd.Series:
    """
    Extract just the Spanish month name from strings like "Septiembre (09)".

    "Septiembre (09)" → "Septiembre"
    "Mayo (05)"       → "Mayo"
    "Enero"           → "Enero"      (already clean, no change)
    pd.NA             → pd.NA

    The number-in-parentheses pattern is removed with regex.
    Runs BEFORE map_values so the translation map can do "Septiembre" → "September".
    """
    if series.dtype == object:
        return (
            series
            .str.replace(r"\s*\(\d{1,2}\)\s*", "", regex=True)
            .str.strip()
        )
    return series


def extract_commune_name(series: pd.Series) -> pd.Series:
    """
    Extract just the commune name from strings like "Popular (1)".

    "Popular (1)"           → "Popular"
    "Laureles-Estadio (11)" → "Laureles-Estadio"
    pd.NA                   → pd.NA

    Same regex logic as extract_month_name.
    """
    if series.dtype == object:
        return (
            series
            .str.replace(r"\s*\(\d{1,2}\)\s*", "", regex=True)
            .str.strip()
        )
    return series


def normalize_accents_key(series: pd.Series) -> pd.Series:
    """
    Generate a deduplication-ready key: lowercase + accent-stripped.

    Used only for shadow _key columns (never stored as display value).

    "María García López" → "maria garcia lopez"
    "Ángel"              → "angel"
    pd.NA                → pd.NA
    """
    def _normalize(val: str) -> str:
        if pd.isna(val):
            return val
        nfkd = unicodedata.normalize("NFKD", str(val).lower())
        return "".join(c for c in nfkd if not unicodedata.combining(c))

    if series.dtype == object:
        return series.apply(lambda v: _normalize(v) if not pd.isna(v) else v)
    return series

# ---------------------------------------------------------------------------
# For the list of dates
# ---------------------------------------------------------------------------
def collapse_spaces_date(series: pd.Series) -> pd.Series:
    """
    Standardizes delimiters (converts ; to ,), collapses spaces, 
    and strips trailing/leading commas or dangling punctuation.
    """
    def _clean(val):
        if pd.isna(val) or str(val).lower() in ['nan', 'none', 'null', '']:
            return pd.NA
        
        # 1. STEP AT THE START: Replace all semicolons with commas
        cleaned = str(val).replace(';', ',')
        
        # 2. Replace all whitespaces/newlines/tabs with a single space
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # 3. Clean spaces around commas (e.g., "04/2018 , 05/2018" -> "04/2018, 05/2018")
        cleaned = re.sub(r'\s*,\s*', ', ', cleaned)
        
        # 4. Collapse multiple consecutive commas into a single comma
        cleaned = re.sub(r'(,\s*)+', ', ', cleaned)
        
        # 5. Strip leading and trailing commas and spaces
        cleaned = cleaned.strip(', ')
        
        # 6. Check if string became empty after cleaning
        if not cleaned:
            return pd.NA
            
        return cleaned

    return series.apply(_clean)


# ---------------------------------------------------------------------------
# TRANSFORMATIONS registry
# Maps the YAML transformation name string → function.
#
# NOTE: map_values, encode, and coerce are NOT in this registry because
# they require access to rule-level dicts (rule.map_values, rule.encoding,
# rule.coerce). They are handled directly in validator.py's
# _apply_single_transform() method.
# ---------------------------------------------------------------------------

TRANSFORMATIONS: dict[str, callable] = {
    # ── string cleaning ──────────────────────────────────────────────────────
    "trim":                  trim,
    "uppercase":             uppercase,
    "lowercase":             lowercase,
    "collapse_spaces":       collapse_spaces,
    "sanitize":              sanitize,
    "empty_to_null":         empty_to_null,
    "smart_title_case":      smart_title_case,
    "to_str":                to_str, 
    # ── numeric ──────────────────────────────────────────────────────────────
    "to_int":                to_int,
    "to_float":              to_float,
    "negative_to_null":      negative_to_null,
    # ── date ─────────────────────────────────────────────────────────────────
    "to_date":               to_date,
    "extract_year":          extract_year,
    "to_none":               to_none,
    # ── domain-specific ──────────────────────────────────────────────────────
    "extract_month_name":    extract_month_name,
    "extract_commune_name":  extract_commune_name,
    "normalize_accents_key": normalize_accents_key,
    #-- list of dates as string -----------------------------------------------
    'collapse_spaces_date': collapse_spaces,
    # ── handled in validator (need rule context) ──────────────────────────────
    # "map_values" → validator._apply_single_transform uses rule.map_values
    # "encode"     → validator._apply_single_transform uses rule.encoding
    # "coerce"     → validator._apply_single_transform uses rule.coerce
}
