"""Primary Preprocessor class."""

import logging
import typing as tp

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ..clean.clean_strings import _unidecode_strings

logger = logging.getLogger(__name__)


class PrimaryDataProcessor(BaseEstimator, TransformerMixin):
    """Process primary data for machine learning tasks.

    Args:
        params (dict): Parameters for data processing.

    Attributes:
        params (dict): Parameters for data processing.
        target (str): Target variable name.
        categorical_fillna_columns_params (dict): Parameters for filling NA values in categorical columns.

    """

    def __init__(self, params: tp.Dict) -> pd.DataFrame:
        """Initialize PrimaryDataProcessor."""
        self.params = params
        self.target = params["target"]
        self.categorical_fillna_columns_params = params["categorical_columns_fillna"]

    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None) -> "PrimaryDataProcessor":
        """Fit the primary data processor.

        Args:
            X (pd.DataFrame): Input data DataFrame.
            y (pd.DataFrame, optional): Target DataFrame.

        Returns:
            PrimaryDataProcessor: Fitted data processor.

        """
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the input data.

        Args:
            X (pd.DataFrame): Input data DataFrame.

        Returns:
            pd.DataFrame: Transformed DataFrame.

        """
        X = self._fillna_categorical_values(X)
        return X

    def fit_transform(self, X: pd.DataFrame, y: pd.DataFrame = None) -> pd.DataFrame:
        """Fit and transform the input data.

        Args:
            X (pd.DataFrame): Input data DataFrame.
            y (pd.DataFrame, optional): Target DataFrame.

        Returns:
            pd.DataFrame: Transformed DataFrame.

        """
        self.fit(X, y)
        return self.transform(X)

    def _fillna_categorical_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values in categorical columns.

        Args:
            X (pd.DataFrame): Input data DataFrame.

        Returns:
            pd.DataFrame: DataFrame with missing values filled.

        """
        for col in self.categorical_fillna_columns_params:
            X[col] = X[col].fillna(self.categorical_fillna_columns_params[col])
            X[col] = X[col].apply(lambda x: _unidecode_strings(x))
        return X
