import logging
import typing as tp

logger = logging.getLogger(__name__)


class MlflowTransformations:
    """MLFlow transformations class."""

    def format_metrics_dict(
        self, output_dict: tp.Dict[str, tp.Any]
    ) -> tp.Dict[str, tp.Union[float, tp.List[float]]]:
        """Transform an input dictionary of values into a dictionary of transformed values.

        MLFlow metrics logger transformation.

        This function takes an input dictionary where the values can be either
        a single numeric value or a list of numeric values.
        It transforms each value into a dictionary format with keys 'value' and 'step',
        where 'value' is the original value, and 'step' is the position of the value
        within the list (if applicable).

        Args:
            output_dict (Dict[str, Any]): A dictionary containing values to be transformed.

        Returns:
            Dict[str, Union[float, List[float]]]: A dictionary with the same keys as the input dictionary,
            where each value is transformed into a dictionary with keys 'value' and 'step'.

        Example:
            >>> input_dict = {'a': 42, 'b': [1, 2, 3]}
            >>> format_metrics_dict(input_dict)
            {'a': {'value': 42, 'step': 1}, 'b': [{'value': 1, 'step': 1}, {'value': 2, 'step': 2}, {'value': 3, 'step': 3}]}
        """
        transformed_dict = {}

        for key, value in output_dict.items():
            if isinstance(value, list):
                transformed_value = [{"value": v, "step": i + 1} for i, v in enumerate(value)]
            else:
                transformed_value = {"value": value, "step": 1}

            transformed_dict[key] = transformed_value

        return transformed_dict
