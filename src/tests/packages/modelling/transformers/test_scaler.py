import pandas as pd
from project.packages.modelling.transformers.scaler import (
    ColumnsPreserverScaler,
    NotScalerTransformer,
)


class TestColumnsPreserverScaler:
    def test_fit_transform(self):
        # Arrange
        data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
        df = pd.DataFrame(data)
        scaler = ColumnsPreserverScaler(
            scaler_params={"class": "sklearn.preprocessing.StandardScaler", "kwargs": {}}
        )

        # Act
        transformed_df = scaler.fit_transform(df)

        # Assert
        assert isinstance(transformed_df, pd.DataFrame)
        assert transformed_df.shape == df.shape
        assert transformed_df.columns.tolist() == df.columns.tolist()
        assert transformed_df.index.tolist() == df.index.tolist()


class TestNotScalerTransformer:
    def test_fit_transform(self):
        # Arrange
        data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
        df = pd.DataFrame(data)
        transformer = NotScalerTransformer()

        # Act
        transformed_df = transformer.fit_transform(df)

        # Assert
        assert isinstance(transformed_df, pd.DataFrame)
        assert transformed_df.shape == df.shape
        assert transformed_df.columns.tolist() == df.columns.tolist()
        assert transformed_df.index.tolist() == df.index.tolist()

    def test_transform_without_fit(self):
        # Arrange
        data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
        df = pd.DataFrame(data)
        transformer = NotScalerTransformer()

        # Act
        transformed_df = transformer.transform(df)

        # Assert
        assert isinstance(transformed_df, pd.DataFrame)
        assert transformed_df.shape == df.shape
        assert transformed_df.columns.tolist() == df.columns.tolist()
        assert transformed_df.index.tolist() == df.index.tolist()
