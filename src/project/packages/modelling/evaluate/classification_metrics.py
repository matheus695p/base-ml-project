import typing as tp

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ...python_utils.typing import Matrix, Vector


def compute_binary_classification_metrics(
    y_true: Vector, y_pred: Vector, y_score: Matrix
) -> tp.Dict[str, float]:
    """Computes classification metrics.

    Calculates various classification metrics for a classification problem.

    Args:
        y_true (np.array): Array of true labels, with shape (n_samples,).
        y_pred (np.array): Array of predicted labels, with shape (n_samples,).

    Returns:
        Dictionary with the following keys:
            - 'accuracy': Accuracy score
            - 'balanced_accuracy': Balanced accuracy score
            - 'f1': F1-score
            - 'f1_micro': Micro-averaged F1-score
            - 'f1_macro': Macro-averaged F1-score
            - 'f1_weighted': Weighted F1-score
            - 'precision': Precision score
            - 'precision_micro': Micro-averaged precision
            - 'precision_macro': Macro-averaged precision
            - 'precision_weighted': Weighted precision
            - 'recall': Recall score
            - 'recall_micro': Micro-averaged recall
            - 'recall_macro': Macro-averaged recall
            - 'recall_weighted': Weighted recall
            - 'roc_auc': ROC AUC score
            - 'roc_auc_ovr': ROC AUC One-vs-Rest (OvR)
            - 'roc_auc_ovo': ROC AUC One-vs-One (OvO)
            - 'roc_auc_ovr_weighted': Weighted ROC AUC OvR
            - 'roc_auc_ovo_weighted': Weighted ROC AUC OvO
            - 'matthews_corrcoef': Matthews correlation coefficient

    """
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "f1_micro": f1_score(y_true, y_pred, average="micro"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "precision": precision_score(y_true, y_pred),
        "precision_micro": precision_score(y_true, y_pred, average="micro"),
        "precision_macro": precision_score(y_true, y_pred, average="macro"),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred),
        "recall_micro": recall_score(y_true, y_pred, average="micro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted"),
        "matthews_corrcoef": matthews_corrcoef(y_true, y_pred),
    }

    # if probabilities are available in the model
    if y_score is not None:
        prob_scores = {
            "roc_auc": roc_auc_score(y_true, y_score),
            "roc_auc_ovr": roc_auc_score(y_true, y_score, multi_class="ovr"),
            "roc_auc_ovo": roc_auc_score(y_true, y_score, multi_class="ovo"),
            "roc_auc_ovr_weighted": roc_auc_score(
                y_true, y_score, multi_class="ovr", average="weighted"
            ),
            "roc_auc_ovo_weighted": roc_auc_score(
                y_true, y_score, multi_class="ovo", average="weighted"
            ),
        }
        metrics.update(prob_scores)

    return metrics
