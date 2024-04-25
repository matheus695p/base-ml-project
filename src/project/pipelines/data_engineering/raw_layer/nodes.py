"""Raw layer nodes."""

import typing as tp

import pandas as pd

from project.packages.preprocessing.transformers.raw import RawDataProcessor


def raw_data_process(
    df_train: pd.DataFrame, df_test: pd.DataFrame, params: tp.Dict
) -> tp.Dict[str, tp.Union[pd.DataFrame, "RawDataProcessor"]]:
    """Process raw data for machine learning tasks.

    Args:
        df_train (pd.DataFrame): Training data DataFrame.
        df_test (pd.DataFrame): Testing data DataFrame.
        params (dict): Parameters for raw data processing.

    Returns:
        dict: A dictionary containing the processed training and testing data,
        and the raw data preprocessor.

    """
    preprocessor = RawDataProcessor(params)
    df_train = preprocessor.fit_transform(df_train)
    df_test = preprocessor.transform(df_test)

    return {
        "train": df_train,
        "test": df_test,
        "preprocessor": preprocessor,
    }
