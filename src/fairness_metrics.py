import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def statistical_parity_difference(df, outcome_col, protected_col, privileged_value):
    groups = df[protected_col].unique()
    
    rates = {}
    for group in groups:
        group_df = df[df[protected_col] == group]
        positive_rate = (group_df[outcome_col] == 1).mean()
        rates[group] = positive_rate

    unprivileged = [g for g in groups if g != privileged_value][0]
    
    spd = rates[unprivileged] - rates[privileged_value]
    dir_ratio = rates[unprivileged] / rates[privileged_value] if rates[privileged_value] != 0 else np.nan
    
    return {
        "rates": rates,
        "statistical_parity_difference": spd,
        "disparate_impact_ratio": dir_ratio
    }


def equal_opportunity_difference(y_true, y_pred, protected, privileged_value):
    groups = np.unique(protected)

    tpr = {}

    for group in groups:
        idx = protected == group
        tn, fp, fn, tp = confusion_matrix(
            y_true[idx], y_pred[idx], labels=[0,1]
        ).ravel()
        
        tpr[group] = tp / (tp + fn) if (tp + fn) > 0 else 0

    unprivileged = [g for g in groups if g != privileged_value][0]
    
    eod = tpr[unprivileged] - tpr[privileged_value]

    return {
        "true_positive_rates": tpr,
        "equal_opportunity_difference": eod
    }