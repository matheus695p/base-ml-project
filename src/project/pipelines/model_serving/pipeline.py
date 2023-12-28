"""Primary pipeline - Data Processing."""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import registry_best_model_to_mlflow
from project.namespaces import NAMESPACES as model_namespaces

model_artifacts = [f"{namespace}.model_artifact" for namespace in model_namespaces]


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=registry_best_model_to_mlflow,
                inputs=["params:model_serving", *model_artifacts] + model_artifacts,
                outputs="production_model",
                name="model_serving",
                tags=["model_serving"],
            ),
        ],
        tags=[
            "model_serving",
        ],
    )
