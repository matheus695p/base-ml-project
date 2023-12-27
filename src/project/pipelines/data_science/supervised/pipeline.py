"""Model Segmentation pipeline."""

from kedro.pipeline import Pipeline, node, pipeline


# from project.reporting.html_report import create_html_report

from .nodes import (
    model_hypertune,
)


def create_pipeline(**kwargs) -> Pipeline:
    """It creates a pipeline for each model in the list of namespaces.

    Returns:
      Sum of Pipelines for all namespaces.
    """
    return sum([_create_pipeline(namespace) for namespace in kwargs.get("namespaces", [])])


def _create_pipeline(namespace: str) -> Pipeline:
    model_pipe = Pipeline(
        [
            node(
                func=model_hypertune,
                inputs={
                    "df": "feat_titanic_train",
                    "params": "params:model_artifact",
                },
                outputs={
                    "study": "study",
                    "best_trial_params": "best_trial_params",
                    # Test metrics comes from the hypertune cross validation exercise
                    "cross_validation_metrics": "cross_validation_metrics",
                    "model_artifact": "model_artifact",
                    "train_dataset": "train_dataset",
                },
                name="hypertune_model",
            ),
        ]
    )

    inputs = {
        "feat_titanic_train": "feat_titanic_train",
    }
    return pipeline(
        pipe=model_pipe,
        namespace=namespace,
        inputs=inputs,
        outputs={
            "study": f"{namespace}.study",
            "best_trial_params": f"{namespace}.best_trial_params",
            "cross_validation_metrics": f"{namespace}.cross_validation_metrics",
            "model_artifact": f"{namespace}.model_artifact",
            "train_dataset": f"{namespace}.train_dataset",
        },
        parameters={},
    )
