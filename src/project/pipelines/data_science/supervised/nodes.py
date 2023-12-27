"""footprint model nodes"""
import logging
import typing as tp

import pandas as pd
from sklearn.utils.validation import check_is_fitted
from project.packages.modelling.models.supervised.sklearn import (
    ClassifierSklearnCompatibleModel,
)


logger = logging.getLogger(__name__)


def model_hypertune(df: pd.DataFrame, params: tp.Dict[str, tp.Any]) -> tp.Dict[str, tp.Any]:
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
