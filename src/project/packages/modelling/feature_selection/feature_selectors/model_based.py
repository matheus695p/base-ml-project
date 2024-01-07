import logging
import pandas as pd
import typing as tp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectFromModel
from project.packages.python_utils.typing.tensors import Tensor, Matrix, Vector
from project.packages.python_utils.load.object_injection import load_estimator


logger = logging.getLogger(__name__)


class ModelBasedFeatureSelector(BaseEstimator, TransformerMixin):
    """A custom transformer for model-based feature selection.

    This transformer selects the most important features based on a specified model.

    Args:
        model_based_params (dict): Parameters to be passed to the SelectFromModel constructor.
    """

    def __init__(self, model_based_params: tp.Dict[str, str]) -> "ModelBasedFeatureSelector":
        """
        Initialize the ModelBasedFeatureSelector.

        Args:
            model_based_params (dict): Parameters to be passed to the SelectFromModel constructor.
        """
        self.bypass_features = model_based_params.pop("bypass_features", [])
        self.model_based_params = {
            key: load_estimator(value) if isinstance(value, dict) else value
            for key, value in model_based_params.items()
        }
        self.selector = SelectFromModel(**self.model_based_params)

    def fit(
        self, X: tp.Union[Tensor, Matrix], y: tp.Union[Vector, Matrix] = None
    ) -> "ModelBasedFeatureSelector":
        """
        Fit the feature selector on the input data.

        Args:
            X (pd.DataFrame): The input data to fit the feature selector on.
            y (array-like, optional): The target variable. Defaults to None.

        Returns:
            ModelBasedFeatureSelector: This transformer object.
        """
        self.selector.fit(X, y)
        self.columns = sorted(
            set(list(self.selector.get_feature_names_out()) + self.bypass_features)
        )
        self.initial_columns = list(X.columns)

        columns_dropped = list(set(self.initial_columns).difference(set(self.columns)))

        logger.info(
            f"Model based feature selection drops the following features: {columns_dropped}"
        )
        return self

    def transform(self, X: tp.Union[Tensor, Matrix]) -> pd.DataFrame:
        """
        Transform the input data by selecting important features.

        Args:
            X (pd.DataFrame): The input data to transform.

        Returns:
            pd.DataFrame: The transformed data with selected features.
        """
        return X[self.columns]
