import pandas as pd
from project.packages.preprocessing.transformers.feature import FeatureDataProcessor


class TestFeatureDataProcessor:
    @classmethod
    def setup_class(cls):
        cls.params = {
            "target": "target_column",
            "encoding_transform": {
                "one_hot_encoder": ["passenger_ticket_base", "passenger_cabin_level"]
            },
        }
        cls.transformer = FeatureDataProcessor(cls.params)

    def test_fit_transform(self):
        # Arrange
        data = {
            "passenger_ticket": ["PC 12345", "unknown", "PC 67890"],
            "passenger_cabin": ["C123", "unknown", "D456"],
            "passenger_siblings": [1, 0, 2],
            "passenger_parch": [0, 0, 1],
        }
        X = pd.DataFrame(data)
        expected_columns = [
            'passenger_ticket',
            'passenger_cabin',
            'passenger_siblings',
            'passenger_parch',
            'passenger_ticket_base',
            'passenger_ticket_number',
            'passenger_ticket_unknown_base',
            'passenger_cabin_level',
            'passenger_cabin_number',
            'passenger_number_of_family_onboard',
            'passenger_is_single',
            'passenger_has_significant_other',
            'passenger_has_childs',
            'passenger_ticket_base_PC',
            'passenger_ticket_base_unknown',
            'passenger_cabin_level_C',
            'passenger_cabin_level_D',
            'passenger_cabin_level_unknown',
        ]

        # Act
        transformed_X = self.transformer.fit_transform(X)

        # Assert
        assert transformed_X.shape[1] == 18
        assert all(col in transformed_X.columns for col in expected_columns)

    def test_get_numerical_cols(self):
        # Arrange
        data = {
            "passenger_ticket": ["PC 12345", "unknown", "PC 67890"],
            "passenger_cabin": ["C123", "unknown", "D456"],
            "passenger_siblings": [1, 0, 2],
            "passenger_parch": [0, 0, 1],
        }
        X = pd.DataFrame(data)
        self.transformer.fit(X)

        # Act
        numerical_cols = self.transformer._get_numerical_cols(X)

        # Assert
        expected_numerical_cols = [
            'passenger_siblings',
            'passenger_parch',
            'passenger_ticket_number',
            'passenger_ticket_unknown_base',
            'passenger_cabin_number',
            'passenger_number_of_family_onboard',
            'passenger_is_single',
            'passenger_has_significant_other',
            'passenger_has_childs',
        ]
        assert numerical_cols == expected_numerical_cols
