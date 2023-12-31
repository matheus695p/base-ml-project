"""Primary pipeline - Data Processing."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import clustering_feature_process, feature_data_process


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
                    "train": "pre_feat_titanic_train",
                    "test": "pre_feat_titanic_test",
                    "preprocessor": "feat_preprocessor",
                },
                name="pre_feature_data_process",
            ),
            node(
                func=clustering_feature_process,
                inputs={
                    "df_train": "pre_feat_titanic_train",
                    "df_test": "pre_feat_titanic_test",
                    "params": "params:feature_transform.clustering_features",
                },
                outputs={
                    "train": "feat_titanic_train",
                    "test": "feat_titanic_test",
                    "preprocessor": "cluster_preprocessor",
                },
                name="feature_data_process",
            ),
        ],
        tags=[
            "feature_layer",
            "feature_engineering",
        ],
    )
