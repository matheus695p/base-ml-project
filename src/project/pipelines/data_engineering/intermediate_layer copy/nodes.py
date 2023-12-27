"""Intermediate layer nodes."""

import typing as tp
import pandas as pd
from project.packages.preprocessing.transformers.intermediate import (
    IntermediateDataProcessor,
)


def intermediate_data_process(
    df_train: pd.DataFrame, df_test: pd.DataFrame, params: tp.Dict
) -> pd.DataFrame:
    raw_preprocessor = IntermediateDataProcessor(params)
    df_train = raw_preprocessor.fit_transform(df_train)
    df_test = raw_preprocessor.transform(df_test)

    return {
        "train": df_train,
        "test": df_test,
    }
