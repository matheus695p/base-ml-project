import pandas as pd
from project.packages.preprocessing.transformers.primary import PrimaryDataProcessor


class TestPrimaryDataProcessor:
    @classmethod
    def setup_class(cls):
        cls.params = {
            "target": "target_column",
            "categorical_columns_fillna": {
                "categorical_column1": "Unknown",
                "categorical_column2": "NA",
            },
        }
        cls.transformer = PrimaryDataProcessor(cls.params)

    def test_fit(self):
        # Arrange
        data = {
            "target_column": [1, 2, 3],
            "categorical_column1": ["A", "B", None],
            "categorical_column2": ["X", "Y", "Z"],
        }
        X = pd.DataFrame(data)

        # Act
        fitted_transformer = self.transformer.fit(X)

        # Assert
        assert fitted_transformer.is_fitted

    def test_transform(self):
        # Arrange
        data = {
            "target_column": [1, 2, 3],
            "categorical_column1": ["A", "B", None],
            "categorical_column2": ["X", "Y", "Z"],
        }
        X = pd.DataFrame(data)

        # Act
        transformed_X = self.transformer.transform(X)

        # Assert
        assert "categorical_column1" in transformed_X.columns
        assert "categorical_column2" in transformed_X.columns
        assert transformed_X["categorical_column1"].isna().sum() == 0
        assert transformed_X["categorical_column2"].isna().sum() == 0
