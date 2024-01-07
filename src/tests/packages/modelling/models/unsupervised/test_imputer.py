import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from project.packages.modelling.models.unsupervised.imputer import ColumnsPreserverImputer


class TestColumnsPreserverImputer:
    def setup_method(self):
        # Load a sample dataset with missing values
        iris = load_iris()
        self.X = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        self.X.iloc[1:5, 0] = np.nan  # Insert missing values in the first column

        # Create an instance of the ColumnsPreserverImputer
        self.imputer = ColumnsPreserverImputer()

    def test_fit(self):
        # Fit the imputer on the dataset
        fitted_imputer = self.imputer.fit(self.X)

        # Check if the imputer has been fitted
        assert fitted_imputer.is_fitted

        # Check if the columns are preserved
        assert fitted_imputer.columns == list(self.X.columns)

    def test_transform(self):
        # Fit the imputer on the dataset
        self.imputer.fit(self.X)

        # Transform the dataset
        X_transformed = self.imputer.transform(self.X)

        # Check if the transformed data has the same columns and index
        assert np.array_equal(X_transformed.columns, self.X.columns)
        assert np.array_equal(X_transformed.index, self.X.index)

        # Check if missing values have been imputed
        assert X_transformed.isna().sum().sum() == 0
