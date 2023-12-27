"""Primary pipeline - Data Processing."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import primary_data_process


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=primary_data_process,
                inputs={
                    "df_train": "int_titanic_train",
                    "df_test": "int_titanic_test",
                    "params": "params:primary_transform",
                },
                outputs={
                    "train": "prm_titanic_train",
                    "test": "prm_titanic_test",
                },
                name="primary_data_process",
            ),
        ],
        tags=["primary_layer", "ingestion"],
    )
