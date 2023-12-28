"""Intermediate Preprocessor class."""

import logging
import typing as tp

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


class IntermediateDataProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, params: tp.Dict) -> pd.DataFrame:
        self.params = params
        self.target = self.params["target"]
        self.outlier_params = self.params["outlier_params"]
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None) -> "IntermediateDataProcessor":
        self.features_in_ = list(X.columns)
        X = X[self.features_in_]
        X = self._drop_unnecessary_columns(X)
        X = self._replace_unknown_values(X)

        self.numerical_cols = self._get_numerical_cols(X)
        self._get_numerical_columns_current_distribution(X)
        self.is_fitted = True

        return self

    def _get_numerical_columns_current_distribution(self, X: pd.DataFrame) -> pd.DataFrame:
        q1 = (
            X[self.numerical_cols]
            .quantile(self.outlier_params["q1_quantile"])
            .to_frame()
            .T.reset_index(drop=True)
        )
        q3 = (
            X[self.numerical_cols]
            .quantile(self.outlier_params["q3_quantile"])
            .to_frame()
            .T.reset_index(drop=True)
        )
        self.iqr = self.outlier_params["iqr_alpha"] * (q3 - q1)
        self.inf_lim = q1 - self.iqr
        self.sup_lim = q3 + self.iqr
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = self._select_columns(X)
        X = self._drop_unnecessary_columns(X)
        X = self._replace_unknown_values(X)

        return X

    def fit_transform(self, X: pd.DataFrame, y: pd.DataFrame = None) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X)

    def _infer_data_distribution(self, X: pd.DataFrame, params: tp.Dict) -> pd.DataFrame:
        return X

    def _validate_data_quality(self, X: pd.DataFrame, params: tp.Dict) -> pd.DataFrame:
        return X

    def _get_categorical_cols(self, X: pd.DataFrame) -> list:
        """Get a list of categorical column names from the DataFrame."""
        categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
        return categorical_cols

    def _get_numerical_cols(self, X: pd.DataFrame) -> list:
        """Get a list of numerical column names from the DataFrame."""
        numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()
        return numerical_cols

    def _drop_unnecessary_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """Drop unnecessary columns."""
        X = X.drop(columns=self.params["drop_columns"], errors="ignore")
        return X

    def _replace_unknown_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Replace unknown values with np.nan."""
        X = X.replace([None, "NaN", "nan", ""], value=np.nan)
        return X

    def _out_of_range_numerical_distributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Clip numerical distributions."""
        logger.info(f"Before applying clip: {X[self.numerical_cols].isna().sum()}")

        for col in self.numerical_cols:
            X.loc[X[col] < self.inf_lim[col].values[0], col] = np.nan
            X.loc[X[col] > self.sup_lim[col].values[0], col] = np.nan
        logger.info(f"After applying clip: {X[self.numerical_cols].isna().sum()}")

        return X

    def _select_columns(self, X: pd.DataFrame):
        """Select columns from the DataFrame."""
        try:
            X = X[self.features_in_]
        except Exception:
            X = X[[col for col in self.features_in_ if col != self.target]]
        return X
