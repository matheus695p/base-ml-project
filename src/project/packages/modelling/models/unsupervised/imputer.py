import warnings
import typing as tp
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ....python_utils.load.object_injection import load_object
from ....python_utils.typing import Matrix, Vector

warnings.filterwarnings("ignore")


class ColumnsPreserverImputer(BaseEstimator, TransformerMixin):
    """A class for imputing missing values while preserving column and index information.

    This class applies imputation to missing values in a pandas DataFrame while preserving
    the column and index information.

    Args:
        imputer_params (Dict[str, str], optional): A dictionary containing imputer parameters.
            Defaults to DEFAULT_PARAMS.

    Attributes:
        imputer_params (Dict[str, str]): The imputer parameters.
        imputer: The imputer object used for imputation.
        columns (List[str]): The list of column names from the input DataFrame.
        is_fitted (bool): Indicates whether the imputer has been fitted.

    """

    DEFAULT_PARAMS = {
        "class": "sklearn.impute.KNNImputer",
        "kwargs": {"n_neighbors": 5, "weights": "distance"},
    }

    def __init__(self, imputer_params: tp.Dict[str, str] = DEFAULT_PARAMS):
        self.imputer_params = imputer_params
        self.imputer = load_object(self.imputer_params)

    def fit(self, X: Matrix, y: Vector = None) -> "ColumnsPreserverImputer":
        """Fit the imputer to the input data.

        This method fits the imputer to the input data and stores the column names.

        Args:
            X (Matrix): The input feature matrix.
            y (Vector, optional): The target variable. Defaults to None.

        Returns:
            ColumnsPreserverImputer: The fitted instance of the class.

        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError(
                "X must be a pandas DataFrame in order to preserve column and index information"
            )

        self.columns = list(X.columns)
        self.imputer = self.imputer.fit(X)
        self.is_fitted = True
        return self

    def transform(self, X: Matrix) -> Matrix:
        """Transform input data by imputing missing values.

        This method transforms input data by imputing missing values while preserving
        column and index information.

        Args:
            X (Matrix): The input feature matrix.

        Returns:
            Matrix: The transformed feature matrix.

        Raises:
            KeyError: If the input DataFrame does not have the expected columns.

        """
        if not isinstance(X, pd.DataFrame):
            msg = (
                "X must be a pandas DataFrame in order to preserve "
                f"column and index information and must contains the following columns {self.columns}"
            )
            raise KeyError(msg)

        X = pd.DataFrame(
            self.imputer.transform(X[self.columns]), index=X.index, columns=self.columns
        )

        return X
