"""Feature layer nodes."""

import typing as tp

import pandas as pd

from project.packages.preprocessing.transformers.feature import FeatureDataProcessor
from project.packages.modelling.models.unsupervised.clustering_features import (
    KMeansClusteringFeatures,
)


def feature_data_process(
    df_train: pd.DataFrame, df_test: pd.DataFrame, params: tp.Dict
) -> pd.DataFrame:
    preprocessor = FeatureDataProcessor(params)
    df_train = preprocessor.fit_transform(df_train)
    df_test = preprocessor.transform(df_test)

    return {
        "train": df_train,
        "test": df_test,
        "preprocessor": preprocessor,
    }


def clustering_feature_process(
    df_train: pd.DataFrame, df_test: pd.DataFrame, params: tp.Dict
) -> pd.DataFrame:

    preprocessor = KMeansClusteringFeatures(
        model_params=params["model"],
        scaler_params=params["scaler"],
        feature_params=params["features"],
        imputer_params=params["imputer"],
    )
    df_train = preprocessor.fit_transform(df_train)
    df_test = preprocessor.transform(df_test)

    return {
        "train": df_train,
        "test": df_test,
        "preprocessor": preprocessor,
    }
