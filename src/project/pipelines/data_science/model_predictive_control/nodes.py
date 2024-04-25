import pandas as pd
import typing as tp
from project.packages.modelling.models.model_predictive_control.explorer import (
    ModelPredictiveControlExplorer,
)


def model_predictive_control_exploration(
    df: pd.DataFrame, model, params: tp.Dict, data_schemas: tp.Dict, *preprocessors
) -> "ModelPredictiveControlExplorer":
    """Explore model predictive control.

    Args:
        df (pd.DataFrame): Data DataFrame.
        model: Model object.
        params (dict): Parameters for the exploration.
        data_schemas (dict): Data schemas.
        *preprocessors: Preprocessor objects.

    Returns:
        ModelPredictiveControlExplorer: Model exploration object.

    """
    preprocessors = list(preprocessors)
    model_explorer = ModelPredictiveControlExplorer(
        model=model,
        preprocessors=preprocessors,
        params=params,
        data_schemas=data_schemas,
    )
    model_explorer.fit(df)
    return model_explorer
