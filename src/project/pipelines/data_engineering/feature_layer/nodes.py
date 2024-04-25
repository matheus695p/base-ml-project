"""Feature layer nodes."""

import typing as tp

import pandas as pd

from project.packages.modelling.models.unsupervised.clustering_features import (
    KMeansClusteringFeatures,
)
from project.packages.preprocessing.transformers.feature import FeatureDataProcessor


def feature_data_process(
    df_train: pd.DataFrame, df_test: pd.DataFrame, params: tp.Dict
) -> tp.Dict[str, tp.Union[pd.DataFrame, "FeatureDataProcessor"]]:
    """Process feature data for machine learning tasks.

    Args:
        df_train (pd.DataFrame): Training data DataFrame.
        df_test (pd.DataFrame): Testing data DataFrame.
        params (dict): Parameters for data processing.

    Returns:
        dict: A dictionary containing the processed training and testing data,
        and the feature data preprocessor.

    """
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
) -> tp.Dict[str, tp.Union[pd.DataFrame, "KMeansClusteringFeatures"]]:
    """Process clustering features for machine learning tasks.

    Args:
        df_train (pd.DataFrame): Training data DataFrame.
        df_test (pd.DataFrame): Testing data DataFrame.
        params (dict): Parameters for clustering feature processing.

    Returns:
        dict: A dictionary containing the processed training and testing data,
        and the clustering feature preprocessor.

    """
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
