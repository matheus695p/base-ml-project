"""model optimization nodes"""
import logging
import typing as tp

import pandas as pd
from sklearn.utils.validation import check_is_fitted
from project.packages.modelling.models.supervised.sklearn import (
    ClassifierSklearnCompatibleModel,
)


logger = logging.getLogger(__name__)


def model_hypertune(df: pd.DataFrame, params: tp.Dict[str, tp.Any]) -> tp.Dict[str, tp.Any]:
    """
    Hyperparameter tuning for a supervised learning model.

    Args:
        df (pd.DataFrame): The input DataFrame containing the dataset.
        params (Dict[str, Any]): A dictionary of hyperparameters and configuration settings.
            - 'features' (List[str]): A list of feature column names.
            - 'target' (str): The name of the target column.
            - Other hyperparameters and configuration settings specific to the model.

    Returns:
        Dict[str, Any]: A dictionary containing the results of hyperparameter tuning and related information.
            - 'model_artifact': The trained model artifact.
            - 'train_dataset': The training dataset used for hyperparameter tuning.
            - Other hyperparameter tuning results and metrics.

    This function performs hyperparameter tuning for a supervised learning model using the provided dataset
    and hyperparameter configurations. It returns a dictionary containing the results of the tuning process
    and the trained model.

    Example usage:
    ```python
    hyperparameters = {
        'features': ['feature1', 'feature2'],
        'target': 'target_column',
        # Add other hyperparameters and configurations here.
    }
    results = model_hypertune(dataset_df, hyperparameters)
    ```
    """
    # supervised learning definition
    features = params["features"]
    target = params["target"]

    X = df[features].astype(float)
    y = df[[target]].astype(float)
    model = ClassifierSklearnCompatibleModel(params)
    model = model.fit(X, y)
    check_is_fitted(model.model)

    results = model.hypertune_results
    results["model_artifact"] = model
    results["train_dataset"] = df
    return results
