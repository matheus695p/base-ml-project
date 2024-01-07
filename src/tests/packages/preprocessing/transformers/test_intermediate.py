import pandas as pd
from project.packages.preprocessing.transformers.intermediate import IntermediateDataProcessor


class TestIntermediateDataProcessor:
    @classmethod
    def setup_class(cls):
        cls.params = {
            "target": "target_column",
            "outlier_params": {"q1_quantile": 0.25, "q3_quantile": 0.75, "iqr_alpha": 1.5},
            "drop_columns": ["column_to_drop"],
        }
        cls.transformer = IntermediateDataProcessor(cls.params)

    def test_fit(self):
        # Arrange
        data = {
            "target_column": [1, 2, 3],
            "column_to_drop": [4, 5, 6],
            "numerical_column1": [7, 8, 9],
            "numerical_column2": [10, 11, 12],
        }
        X = pd.DataFrame(data)

        # Act
        fitted_transformer = self.transformer.fit(X)

        # Assert
        assert fitted_transformer.is_fitted
        assert "numerical_column1" in fitted_transformer.numerical_cols
        assert "numerical_column2" in fitted_transformer.numerical_cols

    def test_transform(self):
        # Arrange
        data = {
            "target_column": [1, 2, 3],
            "column_to_drop": [4, 5, 6],
            "numerical_column1": [7, 8, 9],
            "numerical_column2": [10, 11, 12],
        }
        X = pd.DataFrame(data)

        # Act
        transformed_X = self.transformer.transform(X)

        # Assert
        assert "numerical_column1" in transformed_X.columns
        assert "numerical_column2" in transformed_X.columns
        assert "column_to_drop" not in transformed_X.columns
