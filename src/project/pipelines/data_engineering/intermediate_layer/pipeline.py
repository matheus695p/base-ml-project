"""Intermediate pipeline - Data Processing."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import intermediate_data_process


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=intermediate_data_process,
                inputs={
                    "df_train": "raw_titanic_train",
                    "df_test": "raw_titanic_test",
                    "params": "params:intermediate_transform",
                },
                outputs={
                    "train": "int_titanic_train",
                    "test": "int_titanic_test",
                    "preprocessor": "int_preprocessor",
                },
                name="intermediate_data_process",
            ),
        ],
        tags=["intermediate_layer", "ingestion"],
    )
