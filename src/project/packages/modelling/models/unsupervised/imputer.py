import warnings
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from project.packages.python_utils.load.object_injection import load_object

warnings.filterwarnings("ignore")


class ColumnsPreserverImputer(BaseEstimator, TransformerMixin):

    DEFAULT_PARAMS = {
        "class": "sklearn.impute.KNNImputer",
        "kwargs": {"n_neighbors": 5, "weights": "distance"},
    }

    def __init__(self, imputer_params=DEFAULT_PARAMS):
        self.imputer_params = imputer_params
        self.imputer = load_object(self.imputer_params)

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError(
                "X must be a pandas DataFrame in order to preserve column and index information"
            )

        self.columns = list(X.columns)
        self.imputer = self.imputer.fit(X)
        self.is_fitted = True
        return self

    def transform(self, X):
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
