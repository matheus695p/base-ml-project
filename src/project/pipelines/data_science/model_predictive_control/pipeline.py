"""Model Segmentation pipeline."""

from kedro.pipeline import Pipeline, node, pipeline

from project.packages.reporting.html_report import create_html_report

from .nodes import model_predictive_control_exploration


def create_pipeline(**kwargs) -> Pipeline:
    """It creates a pipeline for each model in the list of namespaces.

    Returns:
      Sum of Pipelines for all namespaces.
    """
    return sum([_create_pipeline(namespace) for namespace in kwargs.get("namespaces", [])])


def _create_pipeline(namespace: str) -> Pipeline:
    preprocessors = [f"{layer}_preprocessor" for layer in ["raw", "int", "prm", "feat", "cluster"]]
    model_pipe = Pipeline(
        [
            node(
                func=model_predictive_control_exploration,
                inputs=[
                    "origin_titanic_train",
                    "model_artifact",
                    "params:mpc",
                    "params:raw_transform.schemas",
                    *preprocessors,
                ],
                outputs="model_predictive_control_explorer",
                name="model_predictive_control_explorer",
            ),
            node(
                func=create_html_report,
                inputs=[
                    "params:model_predictive_control_report",
                    "model_predictive_control_explorer",
                ],
                outputs=[
                    "model_predictive_control_report",
                    "model_predictive_control_error_report",
                ],
                name="model_predictive_control_report",
                tags=["model_predictive_control_report"],
            ),
        ]
    )

    inputs = {
        "origin_titanic_train": "origin_titanic_train",
        "raw_preprocessor": "raw_preprocessor",
        "int_preprocessor": "int_preprocessor",
        "prm_preprocessor": "prm_preprocessor",
        "feat_preprocessor": "feat_preprocessor",
    }
    parameters = {
        "params:raw_transform.schemas": "params:raw_transform.schemas",
    }
    return pipeline(
        pipe=model_pipe,
        namespace=namespace,
        inputs=inputs,
        parameters=parameters,
        outputs={
            "model_predictive_control_explorer": f"{namespace}.model_predictive_control_explorer",
            "model_predictive_control_report": f"{namespace}.model_predictive_control_report",
            "model_predictive_control_error_report": f"{namespace}.model_predictive_control_error_report",
        },
    )
