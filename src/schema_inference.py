"""
Dynamic Schema Inference
------------------------
Infers column type, role, sensitivity, and statistics.
Works on ANY dataset.
"""

import pandas as pd


# Known sensitive keywords
SENSITIVE_KEYWORDS = [
    "sex",
    "gender",
    "race",
    "ethnicity",
    "religion",
    "marital"
]


def is_sensitive(col_name: str) -> bool:
    name = col_name.lower()
    return any(k in name for k in SENSITIVE_KEYWORDS)


def is_binary(series: pd.Series) -> bool:
    vals = series.dropna().unique()
    return len(vals) == 2


def infer_role(col: str, series: pd.Series) -> str:
    """
    Determine semantic role of column
    """
    if is_binary(series):
        return "target"

    if is_sensitive(col):
        return "protected"

    if pd.api.types.is_numeric_dtype(series):
        return "feature"

    return "categorical"


def infer_type(series: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    else:
        return "categorical"


def infer_schema(df: pd.DataFrame) -> dict:
    schema = {}

    for col in df.columns:
        s = df[col]

        # Consider additional missing value symbols
        missing_mask = s.isna() | s.astype(str).isin(["-", "?", "."])
        missing_pct = round(missing_mask.mean() * 100, 2)

        inferred_type = infer_type(s)
        inferred_role = infer_role(col, s)
        sensitive = is_sensitive(col)

        # Sample values for schema warnings
        sample_values = s.dropna().head(10).tolist()

        # Confidence logic: flag unusual types
        confidence = 0.9
        if inferred_type == "categorical" and all(isinstance(v, (int, float)) for v in sample_values):
            confidence = 0.6
        if inferred_type == "numeric" and any(str(v).strip() in ["-", "?", "."] for v in sample_values):
            confidence = 0.6

        schema[col] = {
            "inferred_type": inferred_type,
            "inferred_role": inferred_role,
            "missing_percentage": missing_pct,
            "sensitive": sensitive,
            "confidence": confidence,
            "sample_values": sample_values
        }

    return schema