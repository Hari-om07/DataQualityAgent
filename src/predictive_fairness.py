import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix


def compute_equal_opportunity(df, target_col, protected_cols):

    results = []

    # -----------------------------
    # Proper ML Pipeline
    # -----------------------------

    df_model = df.copy()

    # Clean target column
    df_model[target_col] = df_model[target_col].astype(str).str.strip()

    unique_vals = df_model[target_col].unique()
    if len(unique_vals) != 2:
        raise ValueError("Target column must be binary for Equal Opportunity.")

    positive_label = unique_vals[1]
    df_model["_target_binary"] = (
        df_model[target_col] == positive_label
    ).astype(int)

    # Separate features and target BEFORE encoding
    X = df_model.drop(columns=[target_col, "_target_binary"])
    y = df_model["_target_binary"]

    # One-hot encode features only
    X = pd.get_dummies(X, drop_first=True)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Train logistic regression
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Attach predictions back to original (non-encoded) rows
    test_df = df_model.loc[X_test.index].copy()
    test_df["_actual"] = y_test
    test_df["_predicted"] = y_pred

    for protected_col in protected_cols:

        if protected_col not in test_df.columns:
            continue

        groups = test_df[protected_col].unique()
        if len(groups) < 2:
            continue

        group_counts = test_df[protected_col].value_counts()
        privileged_group = group_counts.idxmax()

        tpr_values = {}

        for group in groups:
            group_df = test_df[test_df[protected_col] == group]

            actual_positive = group_df[group_df["_actual"] == 1]
            true_positive = actual_positive[
                actual_positive["_predicted"] == 1
            ]

            if len(actual_positive) == 0:
                tpr = 0
            else:
                tpr = len(true_positive) / len(actual_positive)

            tpr_values[group] = round(float(tpr), 4)

        unprivileged_group = [g for g in groups if g != privileged_group][0]

        privileged_tpr = tpr_values[privileged_group]
        unprivileged_tpr = tpr_values[unprivileged_group]

        eod = round(unprivileged_tpr - privileged_tpr, 4)

        risk = "LOW"
        if abs(eod) > 0.2:
            risk = "HIGH"
        elif abs(eod) > 0.1:
            risk = "MEDIUM"

        results.append({
            "protected_attribute": protected_col,
            "privileged_group": str(privileged_group),
            "unprivileged_group": str(unprivileged_group),
            "tpr_values": tpr_values,
            "equal_opportunity_difference": eod,
            "risk_level": risk
        })

    return results