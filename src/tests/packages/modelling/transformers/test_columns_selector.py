import pandas as pd
from project.packages.modelling.transformers.columns_selector import ColumnsSelector


class TestColumnsSelector:
    def test_fit(self):
        # Arrange
        columns_selector = ColumnsSelector(columns=['feature1', 'feature2'])
        data = {'feature1': [1, 2, 3], 'feature2': [4, 5, 6], 'feature3': [7, 8, 9]}
        df = pd.DataFrame(data)

        # Act
        columns_selector.fit(df)

        # Assert
        assert columns_selector.is_fitted is True

    def test_transform(self):
        # Arrange
        columns_selector = ColumnsSelector(columns=['feature1', 'feature2'])
        data = {'feature1': [1, 2, 3], 'feature2': [4, 5, 6], 'feature3': [7, 8, 9]}
        df = pd.DataFrame(data)

        # Act
        transformed_df = columns_selector.transform(df)

        # Assert
        expected_df = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        pd.testing.assert_frame_equal(transformed_df, expected_df)
