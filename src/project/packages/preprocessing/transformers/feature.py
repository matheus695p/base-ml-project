"""Feature Preprocessor class."""

import logging
import typing as tp

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

from ..features.titanic import (
    _extract_cabin_number,
    _parse_passenger_cabin,
    _parse_passenger_ticket,
)

logger = logging.getLogger(__name__)


class FeatureDataProcessor(BaseEstimator, TransformerMixin):
    """Process feature data for machine learning tasks.

    Args:
        params (dict): Parameters for feature processing.

    Attributes:
        params (dict): Parameters for feature processing.
        target (str): Target variable name.
        encoding_transform_params (dict): Parameters for encoding transformation.
        one_hot_encoder_columns (list): Columns to be one-hot encoded.
        one_hot_encoders (dict): One-hot encoder objects for each column.

    """

    def __init__(self, params: tp.Dict) -> pd.DataFrame:
        self.params = params
        self.target = params["target"]
        self.encoding_transform_params = params["encoding_transform"]
        self.one_hot_encoder_columns = self.encoding_transform_params.get("one_hot_encoder", [])
        self.one_hot_encoders = {col: OneHotEncoder() for col in self.one_hot_encoder_columns}

    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None) -> "FeatureDataProcessor":
        """Fit the feature data processor.

        Args:
            X (pd.DataFrame): Input feature DataFrame.
            y (pd.DataFrame, optional): Target DataFrame.

        Returns:
            FeatureDataProcessor: Fitted feature data processor.

        """
        self.is_fitted = True
        X = self._build_ticket_features(X)
        X = self._build_cabin_level_feature(X)
        X = self._passenger_info_features(X)
        self._fit_one_hot_encoder(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the input feature data.

        Args:
            X (pd.DataFrame): Input feature DataFrame.

        Returns:
            pd.DataFrame: Transformed feature DataFrame.

        """
        X = self._build_ticket_features(X)
        X = self._build_cabin_level_feature(X)
        X = self._passenger_info_features(X)
        X = self._transform_one_hot_encoder(X)
        self.numerical_cols = self._get_numerical_cols(X)
        return X

    def fit_transform(self, X: pd.DataFrame, y: pd.DataFrame = None) -> pd.DataFrame:
        """Fit and transform the feature data.

        Args:
            X (pd.DataFrame): Input feature DataFrame.
            y (pd.DataFrame, optional): Target DataFrame.

        Returns:
            pd.DataFrame: Transformed feature DataFrame.

        """
        self.fit(X, y)
        return self.transform(X)

    def _build_ticket_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Build features related to passenger ticket information.

        Args:
            X (pd.DataFrame): Input feature DataFrame.

        Returns:
            pd.DataFrame: DataFrame with ticket features added.

        """
        X["passenger_ticket_base"] = X["passenger_ticket"].apply(
            lambda x: _parse_passenger_ticket(x)[0]
        )
        X["passenger_ticket_number"] = X["passenger_ticket"].apply(
            lambda x: _parse_passenger_ticket(x)[1]
        )
        X["passenger_ticket_unknown_base"] = X["passenger_ticket_base"].apply(
            lambda x: 1 if x == "unknown" else 0
        )
        return X

    def _build_cabin_level_feature(self, X: pd.DataFrame) -> pd.DataFrame:
        """Build features related to passenger cabin information.

        Args:
            X (pd.DataFrame): Input feature DataFrame.

        Returns:
            pd.DataFrame: DataFrame with cabin features added.

        """
        X["passenger_cabin_level"] = X["passenger_cabin"].apply(lambda x: _parse_passenger_cabin(x))
        X["passenger_cabin_number"] = X["passenger_cabin"].apply(lambda x: _extract_cabin_number(x))
        return X

    def _passenger_info_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Build features related to passenger information.

        Args:
            X (pd.DataFrame): Input feature DataFrame.

        Returns:
            pd.DataFrame: DataFrame with passenger info features added.

        """
        X["passenger_number_of_family_onboard"] = X["passenger_siblings"] + X["passenger_parch"]
        X["passenger_is_single"] = X["passenger_number_of_family_onboard"].apply(
            lambda x: 1 if x == 0 else 0
        )
        X["passenger_has_significant_other"] = X["passenger_siblings"].apply(
            lambda x: 1 if x == 0 else 0
        )
        X["passenger_has_childs"] = X["passenger_parch"].apply(lambda x: 1 if x == 0 else 0)
        return X

    def _fit_one_hot_encoder(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit one-hot encoders for specified columns.

        Args:
            X (pd.DataFrame): Input feature DataFrame.

        Returns:
            pd.DataFrame: Original DataFrame.

        """
        for column, encoder in self.one_hot_encoders.items():
            encoder.set_params(categories="auto")
            encoded_data = X[column].values.reshape(-1, 1)
            encoder.fit(encoded_data)
            self.one_hot_encoders[column] = encoder
        return self

    def _transform_one_hot_encoder(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted one-hot encoders.

        Args:
            X (pd.DataFrame): Input feature DataFrame.

        Returns:
            pd.DataFrame: Transformed DataFrame.

        """
        dfs = []
        for column, encoder in self.one_hot_encoders.items():
            encoder.set_params(categories="auto")
            encoded_data = X[column].values.reshape(-1, 1)
            encoded_data = pd.DataFrame(
                encoder.transform(encoded_data).toarray(),
                columns=[col.replace("x0", column) for col in encoder.get_feature_names_out()],
            )
            dfs.append(encoded_data)

        encoded_dfs = pd.concat(dfs, axis=1)
        encoded_dfs.index = X.index
        encoded_dfs.index.name = X.index.name

        X = pd.concat([X, encoded_dfs], axis=1)

        return X

    def _get_numerical_cols(self, X: pd.DataFrame) -> list:
        """Get a list of numerical column names from the DataFrame.

        Args:
            X (pd.DataFrame): Input feature DataFrame.

        Returns:
            list: List of numerical column names.

        """
        numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()
        return numerical_cols
