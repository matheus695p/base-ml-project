"""Project pipelines."""
import warnings
import typing as tp
from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

from project.pipelines.data_engineering import (
    raw_layer,
    intermediate_layer,
    primary_layer,
    feature_layer,
)
from project.pipelines.data_science import supervised


warnings.filterwarnings("ignore")


def register_pipelines() -> tp.Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    # namespaces definition
    model_namespaces = [
        "xgboost",
        "random_forest",
        "decision_tree",
        "bayesian_gaussian_mixture",
        "logistic_regression",
        "gaussian_naive_bayes",
        "gradient_boosting_machines",
        "knn",
        "neural_network",
        "perceptron",
        "quadratic_discriminant_analysis",
        "svm",
    ]

    # data engineering pipelines
    raw_pipe = raw_layer.create_pipeline()
    intermediate_pipe = intermediate_layer.create_pipeline()
    primary_pipe = primary_layer.create_pipeline()

    # # feature creation pipelines
    feature_pipe = feature_layer.create_pipeline()

    # data eng pipelines
    data_ingestion_pipes = raw_pipe + intermediate_pipe + primary_pipe
    # data engineering general
    data_engineering_pipes = data_ingestion_pipes + feature_pipe

    # data science
    models_pipe = supervised.create_pipeline(namespaces=model_namespaces)

    # find all pipelines
    pipelines = find_pipelines()

    # data engineering
    pipelines["__default__"] = sum(pipelines.values())
    pipelines["raw_layer"] = raw_pipe
    pipelines["intermediate_layer"] = intermediate_pipe
    pipelines["primary_layer"] = primary_pipe
    pipelines["feature_layer"] = feature_pipe
    pipelines["data_ingestion"] = data_ingestion_pipes
    pipelines["data_engineering"] = data_engineering_pipes
    # data science
    pipelines["data_science"] = models_pipe

    return pipelines
