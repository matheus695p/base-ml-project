import pytest
from project.packages.modelling.models.mlflow.metrics import MlflowTransformations


@pytest.fixture(scope="module")
def input_dict():
    return {
        "a": 42,
        "b": [1, 2, 3],
    }


class TestMlflowTransformations:
    def test_format_metrics_dict_single_value(self, input_dict):
        # Create an instance of the MlflowTransformations class
        transformer = MlflowTransformations()

        # Transform the input dictionary
        transformed_dict = transformer.format_metrics_dict(input_dict)

        # Check if the transformation is correct for single values
        assert transformed_dict["a"] == {"value": 42, "step": 1}

    def test_format_metrics_dict_list_values(self, input_dict):
        # Create an instance of the MlflowTransformations class
        transformer = MlflowTransformations()

        # Transform the input dictionary
        transformed_dict = transformer.format_metrics_dict(input_dict)

        # Check if the transformation is correct for lists
        assert transformed_dict["b"] == [
            {"value": 1, "step": 1},
            {"value": 2, "step": 2},
            {"value": 3, "step": 3},
        ]
