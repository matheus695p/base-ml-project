"""Primary pipeline - Data Processing."""

from kedro.pipeline import Pipeline, node, pipeline
from project.packages.reporting.html_report import create_html_report
from project.namespaces import NAMESPACES as model_namespaces

model_artifacts = [f"{namespace}.model_artifact" for namespace in model_namespaces]


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=create_html_report,
                inputs=[
                    "params:global_modelling_report",
                ]
                + model_artifacts,
                outputs=[
                    "global_optimization_report",
                    "global_optimization_notebook_error_report",
                ],
                name="global_optimization_report",
                tags=["global_optimization_report"],
            ),
        ],
        tags=[
            "reporting",
        ],
    )
