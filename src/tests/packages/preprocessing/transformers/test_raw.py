import pandas as pd
from project.packages.preprocessing.transformers.raw import RawDataProcessor


class TestRawDataProcessor:
    @classmethod
    def setup_class(cls):
        cls.params = {
            "schemas": {
                "column1": {"name": "new_column1", "dtype": "int64"},
                "column2": {"name": "new_column2", "dtype": "float64"},
                "column3": {"name": "new_column3", "dtype": "object"},
            }
        }
        cls.transformer = RawDataProcessor(cls.params)

    def test_fit(self):
        # Arrange
        data = {"column1": [1, 2, 3], "column2": [1.1, 2.2, 3.3], "column3": ["A", "B", "C"]}
        X = pd.DataFrame(data)

        # Act
        fitted_transformer = self.transformer.fit(X)

        # Assert
        assert fitted_transformer.is_fitted

    def test_transform(self):
        # Arrange
        data = {"column1": [1, 2, 3], "column2": [1.1, 2.2, 3.3], "column3": ["A", "B", "C"]}
        X = pd.DataFrame(data)

        # Act
        transformed_X = self.transformer.transform(X)

        # Assert
        assert "new_column1" in transformed_X.columns
        assert "new_column2" in transformed_X.columns
        assert "new_column3" in transformed_X.columns
