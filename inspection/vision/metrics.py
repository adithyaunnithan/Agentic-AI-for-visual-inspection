from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)


def classification_metrics(y_true, y_pred, p_pos=None, classes=None) -> dict:
    """Metrics with class index 1.. treated as 'defect' for binary tasks; macro averages for multiclass."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    binary = len(classes or set(y_true)) <= 2
    avg = "binary" if binary else "macro"
    out = {"n": int(len(y_true)), "accuracy": float(accuracy_score(y_true, y_pred)),
           "precision": float(precision_score(y_true, y_pred, average=avg, zero_division=0)),
           "recall": float(recall_score(y_true, y_pred, average=avg, zero_division=0)),
           "f1": float(f1_score(y_true, y_pred, average=avg, zero_division=0)),
           "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(), "classes": classes}
    if p_pos is not None and binary and len(set(y_true.tolist())) == 2:
        p = np.asarray(p_pos)
        out["roc_auc"] = float(roc_auc_score(y_true, p))
        fpr, tpr, _ = roc_curve(y_true, p)
        out["roc_curve"] = {"fpr": fpr.round(4).tolist(), "tpr": tpr.round(4).tolist()}
    return out
