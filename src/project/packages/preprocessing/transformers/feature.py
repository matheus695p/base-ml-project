"""Feature Preprocessor class."""

import typing as tp
import pandas as pd

import logging
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

from ..features.titanic import (
    _parse_passenger_ticket,
    _parse_passenger_cabin,
    _extract_cabin_number,
)

logger = logging.getLogger(__name__)


class FeatureDataProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, params: tp.Dict) -> pd.DataFrame:
        self.params = params
        self.target = params["target"]
        self.encoding_transform_params = params["encoding_transform"]
        self.one_hot_encoder_columns = self.encoding_transform_params.get("one_hot_encoder", [])
        self.one_hot_encoders = {col: OneHotEncoder() for col in self.one_hot_encoder_columns}

    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None) -> "FeatureDataProcessor":
        self.is_fitted = True
        X = self._build_ticket_features(X)
        X = self._build_cabin_level_feature(X)
        X = self._passenger_info_features(X)
        self._fit_one_hot_encoder(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = self._build_ticket_features(X)
        X = self._build_cabin_level_feature(X)
        X = self._passenger_info_features(X)
        X = self._transform_one_hot_encoder(X)
        self.numerical_cols = self._get_numerical_cols(X)
        return X

    def fit_transform(self, X: pd.DataFrame, y: pd.DataFrame = None) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X)

    def _build_ticket_features(self, X: pd.DataFrame) -> pd.DataFrame:
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
        X["passenger_cabin_level"] = X["passenger_cabin"].apply(lambda x: _parse_passenger_cabin(x))
        X["passenger_cabin_number"] = X["passenger_cabin"].apply(lambda x: _extract_cabin_number(x))
        return X

    def _passenger_info_features(self, X: pd.DataFrame) -> pd.DataFrame:
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
        for column, encoder in self.one_hot_encoders.items():
            encoder.set_params(categories="auto")
            encoded_data = X[column].values.reshape(-1, 1)
            encoder.fit(encoded_data)
            self.one_hot_encoders[column] = encoder
        return self

    def _transform_one_hot_encoder(self, X: pd.DataFrame) -> pd.DataFrame:
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
        """Get a list of numerical column names from the DataFrame."""
        numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()
        return numerical_cols
