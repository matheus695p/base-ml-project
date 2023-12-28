"""Raw Preprocessor class."""

import logging
import typing as tp

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


class RawDataProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, params: tp.Dict) -> pd.DataFrame:
        self.params = params

    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None) -> "RawDataProcessor":
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = self._validate_data_schema(X)
        X = self._create_new_data_schema(X)
        X = self._validate_data_types(X)
        X = self._index_dataframe(X)
        return X

    def fit_transform(self, X: pd.DataFrame, y: pd.DataFrame = None) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X)

    def _validate_data_schema(self, X: pd.DataFrame) -> pd.DataFrame:
        target_col = self.params.get("target", None)
        missing_cols = list(set(self.params["schemas"].keys()).difference(set(X.columns)))
        missing_cols = [col for col in missing_cols if col not in [target_col]]
        if len(missing_cols) > 0:
            raise KeyError(f"Missing columns: {missing_cols}")
        return X

    def _create_new_data_schema(self, X: pd.DataFrame) -> pd.DataFrame:
        renaming_dict = {
            key: self.params["schemas"][key]["name"] for key in self.params["schemas"].keys()
        }
        X = X.rename(columns=renaming_dict)
        return X

    def _validate_data_types(self, X: pd.DataFrame) -> pd.DataFrame:
        # data validation dict
        data_validation_dict = {
            self.params["schemas"][key]["name"]: self.params["schemas"][key]["dtype"]
            for key in self.params["schemas"].keys()
            if key != self.params.get("target", None)
        }
        data_validation_df = pd.DataFrame.from_dict(
            data_validation_dict, orient="index", columns=["dtype"]
        )

        # actual data validation
        actual_types_df = pd.DataFrame(
            X.dtypes,
            columns=[
                "actual_dtype",
            ],
        )
        actual_types_df["actual_dtype"] = actual_types_df["actual_dtype"].apply(str)
        data_validation = actual_types_df.merge(
            data_validation_df, left_index=True, right_index=True, how="right"
        )

        # validator
        validator = data_validation["actual_dtype"] == data_validation["dtype"]

        logger.debug(f"Data validation: {validator}")

        if validator.all():
            logger.debug("Dataset data types successfully validated")
            self.dataset_data_types_validated = True
        else:
            validator = pd.DataFrame(validator, columns=["is_valid"])
            validator = validator[validator["is_valid"] is False]
            columns_to_check_schema = " / ".join(validator.index.tolist())
            msg = (
                f"Data types are not the same as expected. The column/s '{columns_to_check_schema}'"
                " must be checked to follow the appropriate data types"
            )
            raise TypeError(msg)
        return X

    def _index_dataframe(self, X: pd.DataFrame) -> pd.DataFrame:
        index_column = self.params.get("index", None)
        if index_column is not None:
            X = X.set_index(index_column)
        return X
