import logging

from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.pipeline import Pipeline

from project.packages.modelling.transformers.columns_selector import ColumnsSelector
from project.packages.python_utils.load.object_injection import load_estimator, load_object

logger = logging.getLogger(__name__)


class KMeansClusteringFeatures(BaseEstimator, TransformerMixin, ClusterMixin):
    def __init__(self, model_params, scaler_params, imputer_params, feature_params):
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

    def fit(self, X, y=None):
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

    def transform(self, X, y=None):
        for feature_name, model in self.clustering_models.items():
            mapper = self.mappers[feature_name]
            X[feature_name] = model.predict(X)
            X[feature_name] = X[feature_name].map(mapper)
        # Assign cluster labels to the input data
        return X
