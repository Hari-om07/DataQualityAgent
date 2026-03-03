import pandas as pd
import numpy as np
import os

# -----------------------------
# Schema Inference Import
# -----------------------------
from .schema_inference import infer_schema

# -----------------------------
# Robust Target Conversion
# -----------------------------
def to_binary_target(series):
    """
    Convert common classification targets to 0/1 robustly.
    Handles Adult and arbitrary datasets.
    """

    # Case 1: already binary numeric
    if pd.api.types.is_numeric_dtype(series):
        unique_vals = set(series.dropna().unique())
        if unique_vals <= {0, 1}:
            return series.astype(float)

        # numeric but not binary → threshold at median
        return (series > series.median()).astype(float)

    # Case 2: categorical / string
    s = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(".", "", regex=False)
    )

    positive_labels = {
        "1", "yes", "y", "true",
        "approved", "positive",
        ">50k", "high", "success"
    }

    return s.isin(positive_labels).astype(float)
# -----------------------------
# Age Binning (Professional)
# -----------------------------
def bin_age(series: pd.Series) -> pd.Series:
    """
    Convert raw age into fairness-relevant bins.
    """
    bins = [0, 28, 37, 48, 100]
    labels = ["18-28", "29-37", "38-48", "49+"] 
    return pd.cut(series, bins=bins, labels=labels, include_lowest=True)


# -----------------------------
# Representation Bias
# -----------------------------
def compute_representation_bias(df, sensitive_cols):
    rep_bias = {}

    for col in sensitive_cols:
        if col not in df.columns:
            continue

        series = df[col]

        if col == "age" and pd.api.types.is_numeric_dtype(series):
            series = bin_age(series)

        dist = series.value_counts(normalize=True, dropna=False)
        rep_bias[col] = dist.round(3).to_dict()

    return rep_bias


# -----------------------------
# Outcome Disparity
# -----------------------------
def compute_outcome_disparity(df, outcome_col, protected_attrs):
    outcome_bias = {}

    if outcome_col not in df.columns:
        return outcome_bias

    target = df[outcome_col]

    # Encode outcome → binary
    if pd.api.types.is_numeric_dtype(target):
        binary = (target > target.median()).astype(float)
    else:
        top_vals = target.value_counts().index[:2]
        if len(top_vals) < 2:
            return outcome_bias
        positive = top_vals[1]
        binary = (target == positive).astype(float)

    df = df.copy()
    df["_outcome_binary"] = binary

    # Compute disparity
    for col in protected_attrs:
        if col not in df.columns:
            continue

        series = df[col]

        if col.lower() == "age" and pd.api.types.is_numeric_dtype(series):
            series = bin_age(series)

        grouped = df.groupby(series)["_outcome_binary"].mean()
        outcome_bias[col] = grouped.round(3).to_dict()

    return outcome_bias


# -----------------------------
# Ethical Risk Scoring
# -----------------------------
def compute_ethical_risk(rep_bias, outcome_bias):
    """
    Simple rule-based ethical risk assessment.
    """
    max_disparity = 0

    for col in outcome_bias:
        vals = list(outcome_bias[col].values())
        if len(vals) > 1:
            disparity = max(vals) - min(vals)
            max_disparity = max(max_disparity, disparity)

    if max_disparity > 0.25:
        return "HIGH"
    elif max_disparity > 0.15:
        return "MEDIUM"
    else:
        return "LOW"


# -----------------------------
# Main Bias & Fairness Audit
# -----------------------------
def run_bias_fairness_check(input_data):
    """
    input_data: CSV path (str/PathLike) or pandas DataFrame
    """

    # ---------- Load Data ----------
    if isinstance(input_data, (str, os.PathLike)):
        if not os.path.exists(input_data):
            raise FileNotFoundError(f"CSV file not found: {input_data}")
        try:
            df = pd.read_csv(input_data)
        except UnicodeDecodeError:
            df = pd.read_csv(input_data, encoding="ISO-8859-1")

    elif isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    else:
        raise ValueError(
            "Input must be either CSV file path (str/PathLike) or pandas DataFrame"
        )

    # ---------- Schema Inference ----------
    schema = infer_schema(df)
    sensitive_cols = [col for col, meta in schema.items() if meta.get("sensitive")]

    # ---------- Dataset-Agnostic Target Selection ----------
    target_candidates = [
        (col, meta["confidence"]) 
        for col, meta in schema.items() 
        if meta.get("inferred_role") == "target"
    ]

    if not target_candidates:
        return {
            "representation_bias": {},
            "outcome_disparity": {},
            "ethical_risk": "UNKNOWN",
            "fairness_assessable": False,
            "reason": "No target column found"
        }

    # Prefer non-sensitive targets
    non_sensitive_targets = [
        (col, conf) for col, conf in target_candidates 
        if col not in sensitive_cols
    ]
    if non_sensitive_targets:
        target_candidates = non_sensitive_targets

    # Sort by confidence
    target_candidates.sort(key=lambda x: x[1], reverse=True)
    target_col = target_candidates[0][0]

    # Ensure binary outcome
    if df[target_col].nunique() != 2:
        return {
            "representation_bias": {},
            "outcome_disparity": {},
            "ethical_risk": "UNKNOWN",
            "fairness_assessable": False,
            "reason": f"Target column '{target_col}' is not binary"
        }

    # ---------- Convert Target to Binary ----------
    df["_target_binary"] = to_binary_target(df[target_col])

    # ---------- Representation & Raw Outcome Bias ----------
    rep_bias = compute_representation_bias(df, sensitive_cols)
    outcome_bias = compute_outcome_disparity(
        df,
        outcome_col=target_col,
        protected_attrs=sensitive_cols
    )

    # ======================================================
    # NEW: Formal Fairness Metrics (SPD & DIR)
    # ======================================================
    fairness_metrics = {}

    for protected_col in sensitive_cols:

        groups = df[protected_col].dropna().unique()
        if len(groups) < 2:
            continue

        # Auto-detect privileged group (largest group)
        group_counts = df[protected_col].value_counts()
        privileged_group = group_counts.idxmax()

        positive_rates = {}
        for group in groups:
            group_df = df[df[protected_col] == group]
            positive_rate = group_df["_target_binary"].mean()
            positive_rates[group] = round(float(positive_rate), 4)

        # Select first non-privileged as unprivileged
        unprivileged_group = [g for g in groups if g != privileged_group][0]

        privileged_rate = positive_rates[privileged_group]
        unprivileged_rate = positive_rates[unprivileged_group]

        # Statistical Parity Difference
        spd = round(unprivileged_rate - privileged_rate, 4)

        # Disparate Impact Ratio
        dir_ratio = (
            round(unprivileged_rate / privileged_rate, 4)
            if privileged_rate != 0 else None
        )

        # Risk Flag (80% Rule)
        fairness_flag = "LOW"
        if dir_ratio is not None:
            if dir_ratio < 0.8 or dir_ratio > 1.25:
                fairness_flag = "HIGH"
            elif abs(spd) > 0.1:
                fairness_flag = "MEDIUM"

        fairness_metrics[protected_col] = {
            "privileged_group": privileged_group,
            "unprivileged_group": unprivileged_group,
            "positive_rates": positive_rates,
            "statistical_parity_difference": spd,
            "disparate_impact_ratio": dir_ratio,
            "fairness_risk": fairness_flag
        }

    # ======================================================

    ethical_risk = compute_ethical_risk(rep_bias, outcome_bias)

    return {
        "target_column": target_col,
        "representation_bias": rep_bias,
        "outcome_disparity": outcome_bias,
        "fairness_metrics": fairness_metrics,   # NEW OUTPUT
        "ethical_risk": ethical_risk,
        "fairness_assessable": True,
        "outcome_column": target_col,
        "protected_attributes": sensitive_cols
    }
# -----------------------------
# CLI Execution
# -----------------------------
if __name__ == "__main__":
    result = run_bias_fairness_check("data/adult.csv")

    print("⚖️ Bias & Fairness Audit Complete")
    import json
    print(json.dumps(result, indent=4))
