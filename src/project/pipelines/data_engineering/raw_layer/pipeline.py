"""Raw pipeline - Data Ingestion."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import raw_data_process


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=raw_data_process,
                inputs={
                    "df_train": "origin_titanic_train",
                    "df_test": "origin_titanic_test",
                    "params": "params:raw_transform",
                },
                outputs={
                    "train": "raw_titanic_train",
                    "test": "raw_titanic_test",
                    "preprocessor": "raw_preprocessor",
                },
                name="raw_data_process",
            ),
        ],
        tags=["raw_layer", "ingestion"],
    )
