import json
import csv
from datetime import datetime
import os
from src.predictive_fairness import compute_equal_opportunity

# -----------------------------
# Extract Bias Risk
# -----------------------------
def extract_bias_risk(fairness_info):
    """
    Compute bias risk from a fairness analysis dictionary.
    Returns: "HIGH", "MEDIUM", "LOW", or "UNKNOWN"
    """
    if not isinstance(fairness_info, dict):
        return "UNKNOWN"

    disparities = fairness_info.get("outcome_disparity")
    if disparities:
        max_gap = 0
        for col in disparities:
            vals = list(disparities[col].values())
            if len(vals) > 1:
                gap = max(vals) - min(vals)
                max_gap = max(max_gap, gap)

        if max_gap > 0.25:
            return "HIGH"
        elif max_gap > 0.15:
            return "MEDIUM"
        elif max_gap > 0.05:
            return "LOW"
        else:
            return "LOW"

    # Fallback to ethical_risk if provided
    if "ethical_risk" in fairness_info:
        return fairness_info["ethical_risk"]

    return "UNKNOWN"


# -----------------------------
# Main Agent Reasoning
# -----------------------------
def reason_and_recommend(quality_report, bias_report=None):
    decisions = {
        "overall_risk": "LOW",
        "issues_detected": [],
        "recommended_actions": [],
        "human_review_required": False
    }

    # -----------------------------
    # 1️⃣ Data Quality Severity
    # -----------------------------
    severity = quality_report.get("severity", "LOW")
    if severity == "HIGH":
        decisions["overall_risk"] = "HIGH"
        decisions["issues_detected"].append("Significant data quality issues detected")
        decisions["recommended_actions"].append("Investigate missing, inconsistent, or noisy data")
        decisions["human_review_required"] = True
    elif severity == "MEDIUM":
        if decisions["overall_risk"] != "HIGH":
            decisions["overall_risk"] = "MEDIUM"
        decisions["issues_detected"].append("Moderate data quality issues detected")
        decisions["recommended_actions"].append("Review data preprocessing and imputation strategy")

    # -----------------------------
    # 2️⃣ Fairness / Bias Assessment
    # -----------------------------
    fairness_info = quality_report.get("fairness_analysis") or bias_report

    if fairness_info:
        bias_risk = extract_bias_risk(fairness_info)
    else:
        bias_risk = None

    if not bias_risk:
        # No fairness info → MEDIUM + human review
        decisions["issues_detected"].append(
            "Fairness analysis unavailable (no outcome or protected attributes detected)"
        )
        decisions["recommended_actions"].extend([
            "Specify prediction target column",
            "Provide demographic/protected attributes",
            "Add dataset metadata for fairness evaluation"
        ])
        if decisions["overall_risk"] == "LOW":
            decisions["overall_risk"] = "MEDIUM"
        decisions["human_review_required"] = True
        return decisions

    # Apply bias risk logic
    if bias_risk == "HIGH":
        decisions["overall_risk"] = "HIGH"
        decisions["issues_detected"].append(
            "Significant outcome disparity detected across sensitive groups"
        )
        decisions["recommended_actions"].extend([
            "Investigate potential discrimination in dataset",
            "Apply fairness-aware rebalancing or preprocessing",
            "Conduct fairness evaluation before model deployment"
        ])
        decisions["human_review_required"] = True
    elif bias_risk == "MEDIUM":
        if decisions["overall_risk"] != "HIGH":
            decisions["overall_risk"] = "MEDIUM"
        decisions["issues_detected"].append(
            "Moderate disparity detected across demographic groups"
        )
        decisions["recommended_actions"].extend([
            "Review group representation and outcome distribution",
            "Consider fairness constraints during modeling"
        ])
    # LOW → no additional action

    # -----------------------------
    # 3️⃣ Schema Warnings
    # -----------------------------
    schema_warnings = quality_report.get("schema_warnings", [])
    if schema_warnings and len(schema_warnings) > 0:
        decisions["issues_detected"].append("Schema inference uncertainty detected")
        if decisions["overall_risk"] == "LOW":
            decisions["overall_risk"] = "MEDIUM"
        decisions["human_review_required"] = True

    return decisions


# -----------------------------
# Run + Logging
# -----------------------------
def run_agent_reasoning(quality_report, bias_report=None, df=None, target_col=None, sensitive_cols=None):

    # ---------------------------------
    # Base reasoning from your engine
    # ---------------------------------
    final_decision = reason_and_recommend(quality_report, bias_report)

    overall_risk = final_decision.get("overall_risk", "LOW")
    human_review_required = final_decision.get("human_review_required", False)
    issues = final_decision.get("issues_detected", [])
    recommended_actions = final_decision.get("recommended_actions", [])

    # =====================================================
    # DATASET-LEVEL FAIRNESS ESCALATION (SPD / DIR)
    # =====================================================
    if bias_report:
        fairness_metrics = bias_report.get("fairness_metrics", {})

        for attr, metrics in fairness_metrics.items():
            risk = metrics.get("fairness_risk", "LOW")
            spd = metrics.get("statistical_parity_difference")
            dir_ratio = metrics.get("disparate_impact_ratio")

            if risk == "HIGH":
                human_review_required = True
                overall_risk = "HIGH"

                issues.append(
                    f"High dataset fairness risk in '{attr}' "
                    f"(DIR={dir_ratio}, SPD={spd})."
                )

                recommended_actions.append(
                    f"Apply bias mitigation techniques for '{attr}' "
                    f"(reweighing, sampling, or fairness-aware modeling)."
                )

            elif risk == "MEDIUM" and overall_risk != "HIGH":
                overall_risk = "MEDIUM"

                issues.append(
                    f"Moderate dataset fairness disparity in '{attr}' (SPD={spd})."
                )

                recommended_actions.append(
                    f"Monitor fairness metrics for '{attr}' during model training."
                )

    # =====================================================
    # NEW: PREDICTIVE FAIRNESS ESCALATION (EOD)
    # =====================================================
    predictive_results = []

    if df is not None and target_col and sensitive_cols:
        predictive_results = compute_equal_opportunity(
            df,
            target_col,
            sensitive_cols
        )

        for result in predictive_results:
            attr = result["protected_attribute"]
            eod = result["equal_opportunity_difference"]
            risk = result["risk_level"]

            if risk == "HIGH":
                human_review_required = True
                overall_risk = "HIGH"

                issues.append(
                    f"High predictive bias detected in '{attr}' "
                    f"(EOD={eod})."
                )

                recommended_actions.append(
                    f"Consider fairness-aware post-processing or "
                    f"equalized opportunity constraints for '{attr}'."
                )

            elif risk == "MEDIUM" and overall_risk != "HIGH":
                overall_risk = "MEDIUM"

                issues.append(
                    f"Moderate predictive disparity in '{attr}' (EOD={eod})."
                )

                recommended_actions.append(
                    f"Evaluate threshold adjustment or retraining for '{attr}'."
                )

    # =====================================================

    # Update final decision
    final_decision["overall_risk"] = overall_risk
    final_decision["human_review_required"] = human_review_required
    final_decision["issues_detected"] = issues
    final_decision["recommended_actions"] = recommended_actions
    final_decision["predictive_fairness"] = predictive_results

    # ---------------------------------
    # Logging & Report Saving
    # ---------------------------------
    os.makedirs("reports", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    with open("reports/agent_decision_report.json", "w") as f:
        json.dump(final_decision, f, indent=4)

    OUTPUT_FILE = "outputs/agent_log.csv"
    file_exists = os.path.isfile(OUTPUT_FILE)

    with open(OUTPUT_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "overall_risk",
                "human_review_required",
                "issues",
                "recommended_actions"
            ])

        writer.writerow([
            datetime.now().isoformat(),
            final_decision["overall_risk"],
            final_decision["human_review_required"],
            "; ".join(final_decision["issues_detected"]),
            "; ".join(final_decision["recommended_actions"])
        ])

    return final_decision


# -----------------------------
# CLI Test
# -----------------------------
if __name__ == "__main__":
    # Example: load a dataset's quality report JSON
    with open("reports/data_quality_report.json") as f:
        quality_report = json.load(f)

    decision = run_agent_reasoning(quality_report)

    print("🧠 AI Agent Decision Report")
    print(json.dumps(decision, indent=4))