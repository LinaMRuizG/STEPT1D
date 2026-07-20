"""
Run this cell in your notebook BEFORE running DataValidator.
It diagnoses exactly why year_diagnosis comes out as float
while birthdate_year comes out as int.
"""

import pandas as pd

PATH = "/Users/linaruiz/Documents/STEPT1D/data_confidential/STEP-T1D_Study_-_Spatiotemporal_Trends_in_the_Epidemiological_Profile_of_Type_1_Diabetes_-_all_versions_-_labels_-_2026-03-14-16-08-39.csv"

# ── 1. Read exactly as the validator does (dtype=str) ────────────────────────
df = pd.read_csv(PATH, sep=";", keep_default_na=False, dtype=str)

col_birth = "Año de nacimiento"
col_diag  = "Año de diagnóstico"

print("=== RAW dtype after read_csv ===")
print(f"  {col_birth}: {df[col_birth].dtype}")
print(f"  {col_diag}:  {df[col_diag].dtype}")

print("\n=== Unique raw values in year_diagnosis ===")
print(df[col_diag].value_counts(dropna=False).to_string())

print("\n=== Are there nulls? ===")
print(f"  birthdate_year nulls: {df[col_birth].isna().sum()}")
print(f"  year_diagnosis nulls: {df[col_diag].isna().sum()}")

print("\n=== After pd.to_datetime ===")
parsed_birth = pd.to_datetime(df[col_birth], format="%Y-%m-%d", errors="coerce")
parsed_diag  = pd.to_datetime(df[col_diag],  format="%Y-%m-%d", errors="coerce")
print(f"  birthdate NaT count: {parsed_birth.isna().sum()}")
print(f"  year_diag NaT count: {parsed_diag.isna().sum()}")

print("\n=== After .dt.year (before astype) ===")
year_birth_raw = parsed_birth.dt.year
year_diag_raw  = parsed_diag.dt.year
print(f"  birthdate_year dtype: {year_birth_raw.dtype}")
print(f"  year_diagnosis dtype: {year_diag_raw.dtype}")

print("\n=== After .astype('Int64') ===")
year_birth_int = year_birth_raw.astype("Int64")
year_diag_int  = year_diag_raw.astype("Int64")
print(f"  birthdate_year dtype: {year_birth_int.dtype}")
print(f"  year_diagnosis dtype: {year_diag_int.dtype}")
print(f"  year_diagnosis sample:\n{year_diag_int.value_counts(dropna=False).head(5)}")
