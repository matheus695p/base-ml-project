import pytest
import numpy as np
from project.packages.modelling.evaluate.classification_metrics import (
    compute_binary_classification_metrics,
)


class TestBinaryClassificationMetrics:
    def test_compute_metrics(self):
        # Create test data
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        y_score = np.array([0.2, 0.8, 0.6, 0.3, 0.9])

        # Define the expected metrics
        expected_metrics = {
            'accuracy': 0.8,
            'balanced_accuracy': 0.8333333333333333,
            'f1': 0.8,
            'f1_micro': 0.8000000000000002,
            'f1_macro': 0.8,
            'f1_weighted': 0.8,
            'precision': 1.0,
            'precision_micro': 0.8,
            'precision_macro': 0.8333333333333333,
            'precision_weighted': 0.8666666666666666,
            'recall': 0.6666666666666666,
            'recall_micro': 0.8,
            'recall_macro': 0.8333333333333333,
            'recall_weighted': 0.8,
            'matthews_corrcoef': 0.6666666666666666,
            'roc_auc': 1.0,
            'roc_auc_ovr': 1.0,
            'roc_auc_ovo': 1.0,
            'roc_auc_ovr_weighted': 1.0,
            'roc_auc_ovo_weighted': 1.0,
        }

        # Compute the metrics
        metrics = compute_binary_classification_metrics(y_true, y_pred, y_score)

        # Check if computed metrics match expected values
        for metric_name, expected_value in expected_metrics.items():
            assert metric_name in metrics
            assert pytest.approx(metrics[metric_name], rel=1e-4) == expected_value
