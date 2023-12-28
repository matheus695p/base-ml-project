"""Feature layer nodes."""

import typing as tp

import pandas as pd

from project.packages.preprocessing.transformers.feature import FeatureDataProcessor


def feature_data_process(
    df_train: pd.DataFrame, df_test: pd.DataFrame, params: tp.Dict
) -> pd.DataFrame:
    preprocessor = FeatureDataProcessor(params)
    df_train = preprocessor.fit_transform(df_train)
    df_test = preprocessor.transform(df_test)

    return {
        "train": df_train,
        "test": df_test,
    }
