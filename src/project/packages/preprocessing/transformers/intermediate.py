"""Intermediate Preprocessor class."""

import logging
import typing as tp

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


class IntermediateDataProcessor(BaseEstimator, TransformerMixin):
    """Process intermediate data for machine learning tasks.

    Args:
        params (dict): Parameters for data processing.

    Attributes:
        params (dict): Parameters for data processing.
        target (str): Target variable name.
        outlier_params (dict): Parameters for outlier detection.
        is_fitted (bool): Indicates if the processor has been fitted.

    """

    def __init__(self, params: tp.Dict) -> pd.DataFrame:
        self.params = params
        self.target = self.params["target"]
        self.outlier_params = self.params["outlier_params"]
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None) -> "IntermediateDataProcessor":
        """Fit the intermediate data processor.

        Args:
            X (pd.DataFrame): Input data DataFrame.
            y (pd.DataFrame, optional): Target DataFrame.

        Returns:
            IntermediateDataProcessor: Fitted data processor.

        """
        self.features_in_ = list(X.columns)
        X = X[self.features_in_]
        X = self._drop_unnecessary_columns(X)
        X = self._replace_unknown_values(X)

        self.numerical_cols = self._get_numerical_cols(X)
        self._get_numerical_columns_current_distribution(X)
        self.is_fitted = True

        return self

    def _get_numerical_columns_current_distribution(self, X: pd.DataFrame) -> pd.DataFrame:
        """Calculate current distribution statistics of numerical columns.

        Args:
            X (pd.DataFrame): Input data DataFrame.

        Returns:
            pd.DataFrame: DataFrame with distribution statistics.

        """
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
        """Transform the input data.

        Args:
            X (pd.DataFrame): Input data DataFrame.

        Returns:
            pd.DataFrame: Transformed DataFrame.

        """
        X = self._select_columns(X)
        X = self._drop_unnecessary_columns(X)
        X = self._replace_unknown_values(X)

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

    def _infer_data_distribution(self, X: pd.DataFrame, params: tp.Dict) -> pd.DataFrame:
        """Infer data distribution based on specified parameters.

        Args:
            X (pd.DataFrame): Input data DataFrame.
            params (dict): Parameters for data inference.

        Returns:
            pd.DataFrame: DataFrame with inferred data distribution.

        """
        return X

    def _validate_data_quality(self, X: pd.DataFrame, params: tp.Dict) -> pd.DataFrame:
        """Validate data quality based on specified parameters.

        Args:
            X (pd.DataFrame): Input data DataFrame.
            params (dict): Parameters for data validation.

        Returns:
            pd.DataFrame: DataFrame with validated data quality.

        """
        return X

    def _get_categorical_cols(self, X: pd.DataFrame) -> list:
        """Get a list of categorical column names from the DataFrame.

        Args:
            X (pd.DataFrame): Input data DataFrame.

        Returns:
            list: List of categorical column names.

        """
        categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
        return categorical_cols

    def _get_numerical_cols(self, X: pd.DataFrame) -> list:
        """Get a list of numerical column names from the DataFrame.

        Args:
            X (pd.DataFrame): Input data DataFrame.

        Returns:
            list: List of numerical column names.

        """
        numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()
        return numerical_cols

    def _drop_unnecessary_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """Drop unnecessary columns.

        Args:
            X (pd.DataFrame): Input data DataFrame.

        Returns:
            pd.DataFrame: DataFrame with unnecessary columns dropped.

        """
        X = X.drop(columns=self.params["drop_columns"], errors="ignore")
        return X

    def _replace_unknown_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Replace unknown values with np.nan.

        Args:
            X (pd.DataFrame): Input data DataFrame.

        Returns:
            pd.DataFrame: DataFrame with unknown values replaced.

        """
        X = X.replace([None, "NaN", "nan", ""], value=np.nan)
        return X

    def _out_of_range_numerical_distributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Clip numerical distributions.

        Args:
            X (pd.DataFrame): Input data DataFrame.

        Returns:
            pd.DataFrame: DataFrame with clipped numerical distributions.

        """
        logger.info(f"Before applying clip: {X[self.numerical_cols].isna().sum()}")

        for col in self.numerical_cols:
            X.loc[X[col] < self.inf_lim[col].values[0], col] = np.nan
            X.loc[X[col] > self.sup_lim[col].values[0], col] = np.nan
        logger.info(f"After applying clip: {X[self.numerical_cols].isna().sum()}")

        return X

    def _select_columns(self, X: pd.DataFrame):
        """Select columns from the DataFrame.

        Args:
            X (pd.DataFrame): Input data DataFrame.

        Returns:
            pd.DataFrame: DataFrame with selected columns.

        """
        try:
            X = X[self.features_in_]
        except Exception:
            X = X[[col for col in self.features_in_ if col != self.target]]
        return X
