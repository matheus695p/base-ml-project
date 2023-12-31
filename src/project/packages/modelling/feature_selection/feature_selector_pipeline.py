import logging
import pandas as pd
import typing as tp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from project.packages.python_utils.typing.tensors import Tensor, Matrix, Vector
from .feature_selectors import ModelBasedFeatureSelector

logger = logging.getLogger(__name__)


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    A custom transformer for feature selection using different feature selection methods.

    This transformer allows you to apply various feature selection methods to select
    important features from the input dataset.

    Args:
        fs_params (dict): A dictionary containing the parameters for feature selection methods.
            It should include the keys 'selectors' and 'fs_params'.
            - 'selectors': A list of feature selection methods to apply (e.g., ['model_based']).
            - 'fs_params': A dictionary where keys are selector names and values are their
              corresponding parameters.

    Attributes:
        feature_selector_classes (dict): A dictionary mapping selector names to their respective classes.
        selectors (list): A list of feature selection methods to apply.
        fs_params (dict): A dictionary containing parameters for feature selection methods.
        steps (list): A list of tuples representing the feature selection steps in the pipeline.
        fs_pipeline (Pipeline): The scikit-learn pipeline for feature selection.
    """

    def __init__(
        self,
        fs_params: tp.Dict[str, str],
    ) -> "FeatureSelector":
        self.feature_selector_classes = {
            "model_based": ModelBasedFeatureSelector,
        }

        self.selectors = fs_params.get("selectors", [])
        self.fs_params = fs_params
        self.fs_pipeline = Pipeline(
            [
                (
                    selector_name,
                    self.feature_selector_classes[selector_name](self.fs_params[selector_name]),
                )
                for selector_name in self.selectors
            ]
        )

    def fit(
        self, X: tp.Union[Tensor, Matrix], y: tp.Union[Vector, Matrix] = None
    ) -> "FeatureSelector":
        """
        Fit the feature selector on the input data.

        Args:
            X (array-like or pd.DataFrame): The input data to fit the feature selector on.
            y (array-like, optional): The target variable. Defaults to None.

        Returns:
            FeatureSelector: This transformer object.
        """
        self.fs_pipeline.fit(X, y)
        self.columns = list(self.fs_pipeline.transform(X).columns)
        return self

    def transform(self, X: tp.Union[Tensor, Matrix]) -> pd.DataFrame:
        """
        Transform the input data by applying feature selection methods.

        Args:
            X (array-like or pd.DataFrame): The input data to transform.

        Returns:
            pd.DataFrame: The transformed data with selected features.
        """
        return self.fs_pipeline.transform(X)
