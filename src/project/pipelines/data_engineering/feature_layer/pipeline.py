"""Primary pipeline - Data Processing."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import feature_data_process


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=feature_data_process,
                inputs={
                    "df_train": "prm_titanic_train",
                    "df_test": "prm_titanic_test",
                    "params": "params:feature_transform",
                },
                outputs={
                    "train": "feat_titanic_train",
                    "test": "feat_titanic_test",
                },
                name="feature_data_process",
            ),
        ],
        tags=[
            "feature_layer",
            "feature_engineering",
        ],
    )