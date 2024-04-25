"""Intermediate layer nodes."""

import typing as tp

import pandas as pd

from project.packages.preprocessing.transformers.primary import PrimaryDataProcessor


def primary_data_process(
    df_train: pd.DataFrame, df_test: pd.DataFrame, params: tp.Dict
) -> tp.Dict[str, tp.Union[pd.DataFrame, "PrimaryDataProcessor"]]:
    """Process primary data for machine learning tasks.

    Args:
        df_train (pd.DataFrame): Training data DataFrame.
        df_test (pd.DataFrame): Testing data DataFrame.
        params (dict): Parameters for primary data processing.

    Returns:
        dict: A dictionary containing the processed training and testing data,
        and the primary data preprocessor.

    """
    preprocessor = PrimaryDataProcessor(params)
    df_train = preprocessor.fit_transform(df_train)
    df_test = preprocessor.transform(df_test)

    return {
        "train": df_train,
        "test": df_test,
        "preprocessor": preprocessor,
    }
