import pandas as pd
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def run_aif360_adult_comparison(df):

    df = df.copy()

    # Clean target
    df["income"] = df["income"].str.strip()
    df["income"] = df["income"].apply(lambda x: 1 if x == ">50K" else 0)

    # Encode protected attribute
    df["sex"] = df["sex"].apply(lambda x: 1 if x.strip() == "Male" else 0)

    # One-hot encode
    df_encoded = pd.get_dummies(df, drop_first=True)

    # -------------------------
    # Dataset Fairness
    # -------------------------
    dataset = BinaryLabelDataset(
        df=df_encoded,
        label_names=["income"],
        protected_attribute_names=["sex"]
    )

    metric = BinaryLabelDatasetMetric(
        dataset,
        privileged_groups=[{"sex": 1}],
        unprivileged_groups=[{"sex": 0}]
    )

    spd = float(metric.statistical_parity_difference())
    dir_ratio = float(metric.disparate_impact())

    # -------------------------
    # Model Fairness
    # -------------------------
    X = df_encoded.drop(columns=["income"])
    y = df_encoded["income"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = LogisticRegression(max_iter=3000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    df_test = X_test.copy()
    df_test["income"] = y_test
    df_test["pred"] = y_pred

    dataset_true = BinaryLabelDataset(
        df=df_test,
        label_names=["income"],
        protected_attribute_names=["sex"]
    )

    dataset_pred = dataset_true.copy()
    dataset_pred.labels = df_test["pred"].values.reshape(-1, 1)

    metric_pred = ClassificationMetric(
        dataset_true,
        dataset_pred,
        privileged_groups=[{"sex": 1}],
        unprivileged_groups=[{"sex": 0}]
    )

    eod = float(metric_pred.equal_opportunity_difference())

    return {
        "SPD": round(spd, 4),
        "DIR": round(dir_ratio, 4),
        "EOD": round(eod, 4)
    }