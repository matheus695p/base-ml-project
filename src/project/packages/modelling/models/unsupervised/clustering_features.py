import logging

import typing as tp
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.pipeline import Pipeline

from project.packages.modelling.transformers.columns_selector import ColumnsSelector
from ....python_utils.load.object_injection import load_estimator, load_object
from ....python_utils.typing import Matrix, Vector

logger = logging.getLogger(__name__)


class KMeansClusteringFeatures(BaseEstimator, TransformerMixin, ClusterMixin):
    """A class for feature transformation using K-Means clustering.

    This class applies K-Means clustering to subsets of features in monotonic increasing
    cluster features.

    Args:
        model_params (Dict[str, str]): A dictionary containing model parameters.
        scaler_params (Dict[str, str]): A dictionary containing scaler parameters.
        imputer_params (Dict[str, str]): A dictionary containing imputer parameters.
        feature_params (Dict[str, str]): A dictionary specifying features and their corresponding
            clustering configurations.

    Attributes:
        model_params (Dict[str, str]): The model parameters.
        scaler_params (Dict[str, str]): The scaler parameters.
        feature_params (Dict[str, str]): The feature-specific clustering configurations.
        imputer_params (Dict[str, str]): The imputer parameters.
        is_fitted (bool): Indicates whether the model has been fitted.
        clustering_models (Dict[str, Pipeline]): A dictionary of feature-specific clustering models.

    """

    def __init__(
        self,
        model_params: tp.Dict[str, str],
        scaler_params: tp.Dict[str, str],
        imputer_params: tp.Dict[str, str],
        feature_params: tp.Dict[str, str],
    ) -> "KMeansClusteringFeatures":
        self.model_params = model_params
        self.scaler_params = scaler_params
        self.feature_params = feature_params
        self.imputer_params = imputer_params
        self.is_fitted = False
        self.clustering_models = {
            feature_name: Pipeline(
                [
                    ("selector", ColumnsSelector(features)),
                    ("imputer", load_object(imputer_params)),
                    ("scaler", load_object(scaler_params)),
                    ("model", load_estimator(model_params)),
                ]
            )
            for feature_name, features in feature_params.items()
        }

    def fit(self, X: Matrix, y: Vector = None) -> "KMeansClusteringFeatures":
        """Fit the K-Means clustering models to the input data.

        This method fits the K-Means clustering models to the input data and creates cluster mapping
        for each feature.

        Args:
            X (Matrix): The input feature matrix.
            y (Vector, optional): The target variable. Defaults to None.

        Returns:
            KMeansClusteringFeatures: The fitted instance of the class.

        """
        mappers = {}
        for feature_name, model in self.clustering_models.items():
            model = model.fit(X)
            transformers = model[:-1]
            data_normalized = transformers.transform(X)
            data_normalized[feature_name] = model.predict(X)

            # create a mapper ids, so clustering feature are always monotonic
            df_mapper = data_normalized.groupby(feature_name).mean()
            df_mapper["sum"] = df_mapper.sum(axis=1)
            df_mapper = (
                df_mapper.sort_values(by="sum", ascending=True)
                .reset_index()
                .reset_index()
                .rename(columns={"index": "cluster_id"})[[feature_name, "cluster_id"]]
            )
            df_mapper = df_mapper.set_index(feature_name)
            mapper = df_mapper.to_dict()["cluster_id"]
            mappers[feature_name] = mapper

        self.mappers = mappers
        return self

    def transform(self, X: Matrix, y: Vector = None) -> Vector:
        """Transform input features using K-Means clustering.

        This method transforms input features using the trained K-Means clustering models and cluster mapping.

        Args:
            X (Matrix): The input feature matrix.
            y (Vector, optional): The target variable. Defaults to None.

        Returns:
            Vector: The transformed feature matrix.

        """
        for feature_name, model in self.clustering_models.items():
            mapper = self.mappers[feature_name]
            X[feature_name] = model.predict(X)
            X[feature_name] = X[feature_name].map(mapper)
        # Assign cluster labels to the input data
        return X
