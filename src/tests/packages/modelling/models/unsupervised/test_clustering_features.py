import pytest
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from project.packages.modelling.models.unsupervised.clustering_features import (
    KMeansClusteringFeatures,
)


config_dict = {
    "clustering_features": {
        "imputer": {
            "class": "project.packages.modelling.models.unsupervised.imputer.ColumnsPreserverImputer",
            "kwargs": {
                "imputer_params": {
                    "class": "sklearn.impute.KNNImputer",
                    "kwargs": {"n_neighbors": 10, "weights": "distance"},
                }
            },
        },
        "scaler": {
            "class": "project.packages.modelling.transformers.scaler.ColumnsPreserverScaler",
            "kwargs": {
                "scaler_params": {"class": "sklearn.preprocessing.MinMaxScaler", "kwargs": {}}
            },
        },
        "model": {
            "class": "project.packages.modelling.models.unsupervised.segmentation.KMeansElbowSelector",
            "kwargs": {"min_clusters": 1, "max_clusters": 15},
        },
        "features": {
            "cluster_feature": [
                "feature1",
                "feature2",
                "feature3",
                "feature4",
            ],
        },
    },
}


imputer_params = config_dict["clustering_features"]["imputer"]
scaler_params = config_dict["clustering_features"]["scaler"]
model_params = config_dict["clustering_features"]["model"]
feature_params = config_dict["clustering_features"]["features"]
X, y = make_classification(n_samples=100, n_features=5, random_state=42)
X = pd.DataFrame(X, columns=["feature1", "feature2", "feature3", "feature4", "feature5"])
y = pd.DataFrame(y, columns=["target_column"])


@pytest.fixture(scope="module")
def clustering_model():
    return KMeansClusteringFeatures(
        model_params=model_params,
        scaler_params=scaler_params,
        imputer_params=imputer_params,
        feature_params=feature_params,
    )


class TestKMeansClusteringFeatures:
    def test_fit(self, clustering_model):
        clustering_model.fit(X)
        assert clustering_model.is_fitted
        cluster_models = list(clustering_model.clustering_models.values())
        for cluster in cluster_models:
            assert isinstance(cluster, Pipeline)

    def test_transform(self, clustering_model):
        clustering_model.fit(X)
        X_transformed = clustering_model.transform(X)
        assert "cluster_feature" in X_transformed.columns
