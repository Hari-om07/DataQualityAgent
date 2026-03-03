"""
Dynamic Data Quality Audit + Fairness
-------------------------------------
Uses inferred schema instead of hardcoded columns.
Works on ANY dataset.
Includes fairness disparity analysis (Step 6).
"""

import pandas as pd
import json
import sys
import os

from .dataset_loader import load_dataset
from .schema_inference import infer_schema
from .bias_fairness_check import compute_outcome_disparity

# -----------------------------
# Human-friendly schema warnings
# -----------------------------
MISSING_INDICATORS = ["-", "?", ".", "", "na", "n/a", "none", "null"]

def standardize_schema_warnings(schema: dict, df: pd.DataFrame, raw_warnings: list) -> list:
    """
    Convert technical schema warnings to plain language for non-IT users.
    Flags only columns with actual issues: missing values, unusual data types, or low confidence.
    """
    standardized = []

    # Handle raw warnings first
    for w in raw_warnings:
        w_lower = w.lower()
        if "could not identify outcome" in w_lower:
            msg = (
                "Target column or demographic/protected attributes not found; "
                "fairness analysis cannot be performed."
            )
            standardized.append(msg)
        elif "missing values detected in sensitive column" in w_lower:
            col = w.split("'")[1]
            msg = (
                f"There are missing values in the sensitive column '{col}', "
                "which may affect fairness analysis."
            )
            standardized.append(msg)
        # ignore generic "unusual type" warnings here; we handle them below

    # Check each column for actual potential issues
    for col, meta in schema.items():
        s = df[col]

        # 1️⃣ Missing or invalid values
        missing_count = s.isna().sum() + s.astype(str).str.strip().str.lower().isin(MISSING_INDICATORS).sum()
        missing_pct = (missing_count / len(df)) * 100
        if missing_count > 0:
            standardized.append(f"Column '{col}' has {round(missing_pct,2)}% missing or invalid values; please review.")

        # 2️⃣ Low confidence
        if meta.get("confidence", 1.0) < 0.7:
            standardized.append(
                f"The type or role of column '{col}' could not be determined reliably; please verify the data."
            )

        # 3️⃣ Type mismatches
        if meta["inferred_type"] == "categorical":
            sample_values = s.dropna().head(20)
            if any(isinstance(v, (int, float)) for v in sample_values):
                standardized.append(
                    f"Column '{col}' is expected to be text but contains numeric values; please check for data entry errors."
                )

        if meta["inferred_type"] == "numeric":
            sample_values = s.dropna().head(20)
            if any(str(v).strip().lower() in MISSING_INDICATORS for v in sample_values):
                standardized.append(
                    f"Column '{col}' has invalid numeric values like '-', '?', or empty cells; please review."
                )

    return standardized


# -----------------------------
# Quality thresholds
# -----------------------------
MISSING_VALUE_THRESHOLDS = {
    "LOW": 1,
    "MEDIUM": 5,
    "HIGH": 10
}


# -----------------------------
# Detect outcome + protected attrs
# -----------------------------
def detect_roles(schema: dict):
    outcome_col = None
    protected_attrs = []

    for col, meta in schema.items():
        role = meta["inferred_role"]

        if role == "target":
            outcome_col = col

        if meta["sensitive"]:
            protected_attrs.append(col)

    return outcome_col, protected_attrs
    
# -----------------------------
# Main audit function
# -----------------------------
def run_data_quality_audit(input_data) -> dict:
    """
    input_data: str (CSV path) OR pandas.DataFrame
    """
    # ---------- Load Data ----------
    if isinstance(input_data, (str, os.PathLike)):
        if not os.path.exists(input_data):
            raise FileNotFoundError(f"CSV file not found: {input_data}")
        try:
            df = pd.read_csv(input_data)
        except UnicodeDecodeError:
            # Try fallback encoding
            df = pd.read_csv(input_data, encoding="ISO-8859-1")

    elif isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    else:
        raise ValueError("Input must be CSV file path (str/PathLike) or pandas DataFrame")
    
    schema = infer_schema(df)

    report = {
        "missing_values": {},
        "schema_warnings": [],
        "severity": "LOW"
    }

    highest_severity = "LOW"

    # -----------------------------
    # Missing value analysis
    # -----------------------------
    MISSING_INDICATORS = ["-", "?", ".", "", "na", "n/a", "none", "null"]

    for col in df.columns:
        s = df[col]
        # Count NaNs and custom missing indicators
        missing_count = s.isna().sum() + s.astype(str).str.strip().str.lower().isin(MISSING_INDICATORS).sum()
        missing_pct = (missing_count / len(df)) * 100

        if missing_count > 0:
            report["missing_values"][col] = round(missing_pct, 2)

            # Flag if sensitive column has missing values
            if schema.get(col, {}).get("sensitive", False):
                report["schema_warnings"].append(
                    f"Missing values detected in sensitive column '{col}'"
                )

    # Determine overall severity from missing values
    missing_severity = 0
    for pct in report["missing_values"].values():
        if pct >= MISSING_VALUE_THRESHOLDS["HIGH"]:
            missing_severity = max(missing_severity, 3)
        elif pct >= MISSING_VALUE_THRESHOLDS["MEDIUM"]:
            missing_severity = max(missing_severity, 2)
        elif pct >= MISSING_VALUE_THRESHOLDS["LOW"]:
            missing_severity = max(missing_severity, 1)

    severity_map = {1: "LOW", 2: "MEDIUM", 3: "HIGH"}
    highest_severity = severity_map.get(missing_severity, highest_severity)

    # -----------------------------
    # Confidence-based warnings
    # -----------------------------
    for col, meta in schema.items():
        if meta["confidence"] < 0.7:
            report["schema_warnings"].append(
                f"Low confidence inference for column '{col}' "
                f"(type={meta['inferred_type']}, role={meta['inferred_role']})"
            )

        if meta["inferred_type"] not in ["int64", "float64", "object", "bool"]:
            report["schema_warnings"].append(
                f"Unusual inferred type '{meta['inferred_type']}' for column '{col}'"
            )

    
    # -----------------------------
    # Step 6 — Fairness analysis
    # -----------------------------
    outcome_col, protected_attrs = detect_roles(schema)
    if outcome_col in report["missing_values"]:
        report["schema_warnings"].append(
            f"Missing values detected in outcome column '{outcome_col}'"
        )
        highest_severity = "HIGH"

    if outcome_col and protected_attrs:
        try:
            # --- Create binary target safely ---
            unique_vals = df[outcome_col].dropna().unique()

            if len(unique_vals) == 2:
                mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
                df["_target_binary"] = df[outcome_col].map(mapping)
            else:
                raise ValueError(
                    f"Outcome column '{outcome_col}' is not binary"
                )
            

            fairness = compute_outcome_disparity(
                df,
                outcome_col=outcome_col,
                protected_attrs=protected_attrs
            )

            report["fairness_analysis"] = {
                "outcome_column": outcome_col,
                "protected_attributes": protected_attrs,
                "outcome_disparity": fairness
            }

        except Exception as e:
            report["schema_warnings"].append(
                f"Fairness analysis failed: {str(e)}"
            )

    else:
        report["schema_warnings"].append(
            "Could not identify outcome or protected attributes for fairness analysis"
        )

    
    report["schema_warnings"] = standardize_schema_warnings(schema, df, report["schema_warnings"])
    report["severity"] = highest_severity
    return report




# -----------------------------
# CLI execution
# -----------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/data_quality_check.py <dataset.csv>")
        sys.exit(1)

    dataset_path = sys.argv[1]

    report = run_data_quality_audit(dataset_path)

    print("✅ Data Quality Audit Complete")
    print(json.dumps(report, indent=4))