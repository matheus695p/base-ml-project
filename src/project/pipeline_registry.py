"""Project pipelines."""
import typing as tp
import warnings

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

from project.pipelines import global_reporting, model_serving
from project.pipelines.data_engineering import (
    feature_layer,
    intermediate_layer,
    primary_layer,
    raw_layer,
)
from project.pipelines.data_science import supervised

from .namespaces import NAMESPACES

warnings.filterwarnings("ignore")


def register_pipelines() -> tp.Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    # namespaces definition
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

    # global reporting
    global_reporting_pipe = global_reporting.create_pipeline()
    # model serving pipeline
    model_serving_pipe = model_serving.create_pipeline()
    # data science
    models_pipe = (
        supervised.create_pipeline(namespaces=NAMESPACES)
        + global_reporting_pipe
        + model_serving_pipe
    )

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
    pipelines["global_reporting"] = global_reporting_pipe

    return pipelines
