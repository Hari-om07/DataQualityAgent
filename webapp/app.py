import os
from flask import Flask, render_template, request
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits

from src.data_quality_check import run_data_quality_audit
from src.bias_fairness_check import run_bias_fairness_check
from src.agent_reasoning import run_agent_reasoning
from src.aif360_adult import run_aif360_adult_comparison

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 1️⃣ Built-in datasets mapping
BUILTIN_DATASETS = {
    "iris": load_iris,
    "wine": load_wine,
    "breast_cancer": load_breast_cancer,
    "digits": load_digits
}

def sklearn_to_df(loader_func):
    data = loader_func()
    import pandas as pd
    df = pd.DataFrame(data.data, columns=data.feature_names if hasattr(data, "feature_names") else [f"feature_{i}" for i in range(data.data.shape[1])])
    # Add target if exists
    if hasattr(data, "target"):
        df["target"] = data.target
    return df


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        dataset_type = request.form.get("dataset_type", "upload")

        # Uploaded CSV
        if dataset_type == "upload":
            file = request.files.get("dataset")
            if not file or file.filename == "":
                return "No file selected"
            
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            # Try UTF-8 first, fallback to ISO-8859-1 (Latin1)
            try:
                df = pd.read_csv(filepath)
            except UnicodeDecodeError:
                df = pd.read_csv(filepath, encoding="ISO-8859-1")
            
            dataset_name = file.filename.rsplit(".", 1)[0]  # remove extension

        # Built-in dataset
        elif dataset_type == "builtin":
            ds_name = request.form.get("builtin_dataset")
            loader_func = BUILTIN_DATASETS.get(ds_name)
            if not loader_func:
                return f"Dataset '{ds_name}' not found"
            df = sklearn_to_df(loader_func)
            dataset_name = ds_name.capitalize()

        else:
            return "Invalid dataset type"
        

        # -----------------------------
        # Run analysis
        # -----------------------------
        quality_report = run_data_quality_audit(df)
        bias_report = run_bias_fairness_check(df)

        # Extract fairness metadata for predictive fairness
        target_col = bias_report.get("target_column")

        # Protected attributes are the keys of fairness_metrics
        fairness_metrics = bias_report.get("fairness_metrics", {})
        sensitive_cols = list(fairness_metrics.keys())

        print("Target column:", target_col)
        print("Sensitive columns:", sensitive_cols)

        # Run governance reasoning (NOW WITH PREDICTIVE FAIRNESS)
        decision_report = run_agent_reasoning(
            quality_report,
            bias_report,
            df=df,
            target_col=target_col,
            sensitive_cols=sensitive_cols
        )
        # ---------------------------------------
        # Extract Agent Metrics (SEX only)
        # ---------------------------------------
        agent_eod = None

        predictive_results = decision_report.get("predictive_fairness", [])

        for result in predictive_results:
            if result.get("protected_attribute") == "sex":
                agent_eod = result.get("equal_opportunity_difference")

        agent_spd = None
        agent_dir = None

        fairness_metrics = bias_report.get("fairness_metrics", {})

        if "sex" in fairness_metrics:
            sex_metrics = fairness_metrics["sex"]
            agent_spd = sex_metrics.get("statistical_parity_difference")
            agent_dir = sex_metrics.get("disparate_impact_ratio")
        
        # ---------------------------------------
        # AIF360 Benchmark (Adult dataset only)
        # ---------------------------------------

        if dataset_name.lower().startswith("adult"):

            aif360_results = run_aif360_adult_comparison(df)

            def safe_round(val):
                return round(val, 4) if isinstance(val, (int, float)) else "N/A"

            def safe_diff(a, b):
                if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    return abs(a - b)
                return None

            def alignment_status(diff):
                if diff is None:
                    return "N/A"
                elif diff < 0.005:
                    return "Aligned"
                elif diff < 0.02:
                    return "Minor Deviation"
                else:
                    return "Significant Deviation"

            spd_diff = safe_diff(agent_spd, aif360_results.get("SPD"))
            dir_diff = safe_diff(agent_dir, aif360_results.get("DIR"))
            eod_diff = safe_diff(agent_eod, aif360_results.get("EOD"))

            comparison_table = [
                {
                    "metric": "Statistical Parity Difference (SPD)",
                    "agent": safe_round(agent_spd),
                    "aif360": safe_round(aif360_results.get("SPD")),
                    "diff": safe_round(spd_diff),
                    "alignment": alignment_status(spd_diff)
                },
                {
                    "metric": "Disparate Impact Ratio (DIR)",
                    "agent": safe_round(agent_dir),
                    "aif360": safe_round(aif360_results.get("DIR")),
                    "diff": safe_round(dir_diff),
                    "alignment": alignment_status(dir_diff)
                },
                {
                    "metric": "Equal Opportunity Difference (EOD)",
                    "agent": safe_round(agent_eod),
                    "aif360": safe_round(aif360_results.get("EOD")),
                    "diff": safe_round(eod_diff),
                    "alignment": alignment_status(eod_diff)
                }
            ]

        else:
            comparison_table = None

        # Merge all for template
        merged_report = {
            "severity": quality_report.get("severity", "UNKNOWN"),
            "missing_values": quality_report.get("missing_values", {}),
            "schema_warnings": quality_report.get("schema_warnings", []),
            "fairness_analysis": bias_report,
            "predictive_fairness": decision_report.get("predictive_fairness", []),
            "agent_decision": decision_report
        }

        return render_template(
            "result.html",
            report=merged_report,
            dataset_name=dataset_name,
            comparison_table=comparison_table
        )

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)