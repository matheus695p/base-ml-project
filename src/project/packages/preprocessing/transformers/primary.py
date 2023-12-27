"""Primary Preprocessor class."""

import typing as tp
import pandas as pd
import logging
from sklearn.base import BaseEstimator, TransformerMixin
from ..clean.clean_strings import _unidecode_strings

logger = logging.getLogger(__name__)


class PrimaryDataProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, params: tp.Dict) -> pd.DataFrame:
        self.params = params
        self.target = params["target"]
        self.categorical_fillna_columns_params = params["categorical_columns_fillna"]

    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None) -> "PrimaryDataProcessor":
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = self._fillna_categorical_values(X)
        return X

    def fit_transform(self, X: pd.DataFrame, y: pd.DataFrame = None) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X)

    def _fillna_categorical_values(self, X: pd.DataFrame):
        for col in self.categorical_fillna_columns_params:
            X[col] = X[col].fillna(self.categorical_fillna_columns_params[col])
            X[col] = X[col].apply(lambda x: _unidecode_strings(x))
        return X
