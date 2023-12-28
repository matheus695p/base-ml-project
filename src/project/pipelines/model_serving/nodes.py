import logging
import pickle
import typing as tp

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import pandas as pd
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def registry_best_model_to_mlflow(params: tp.Dict, *models: tp.List[Pipeline]) -> Pipeline:
    models = list(models)
    model = get_best_model(params, models)
    best_model = registry_model(params, model)
    return best_model


def get_compile_metric_dataset(
    name: str, cross_validation_metrics: tp.Dict[str, tp.Dict[str, str]]
):
    metrics = []
    for metric, value in cross_validation_metrics.items():
        metrics.append([metric, value['value']])
    metrics = (
        pd.DataFrame(metrics, columns=["metric", "value"])
        .set_index("metric")
        .T.reset_index(drop=True)
    )
    metric_columns = list(metrics.columns)
    metrics["model"] = name
    metrics = metrics[["model"] + metric_columns]
    return metrics


def get_best_model(params: tp.Dict, models: tp.List[Pipeline]) -> Pipeline:

    # Specify the registered model name and version
    metric_cols = params["metric_cols"]
    models_dict = {}
    dfs_metrics = []
    for model in models:
        name = model.model[-1].__class__.__name__.lower()
        models_dict[name] = model
        cross_validation_metrics = model.hypertune_results["cross_validation_metrics"]
        metric = get_compile_metric_dataset(name, cross_validation_metrics)
        dfs_metrics.append(metric)

    metrics = pd.concat(dfs_metrics, axis=0).reset_index(drop=True).set_index("model")
    metric_col_names = [col.split(".")[1] for col in metric_cols]
    metrics["recent_scores"] = metrics[metric_col_names].mean(axis=1)
    metrics = metrics.sort_values("recent_scores", ascending=False)
    best_model_name = metrics.index[0]
    best_model = models_dict[best_model_name]

    logger.info(f"Best model for these iteration {best_model_name}")

    return best_model


def registry_model(params: tp.Dict, model: Pipeline) -> Pipeline:

    model_name = params["model_name"]
    metric_cols = params["metric_cols"]

    # Example: Train a model and log it to MLflow
    with mlflow.start_run(run_name="model_registry", nested=True) as run:
        metrics = model.hypertune_results["cross_validation_metrics"]
        metrics = {metric: value["value"] for metric, value in metrics.items()}
        mlflow.sklearn.log_model(model, model_name)
        mlflow.log_metrics(metrics)
        # Register the model
        model_uri = f"runs:/{run.info.run_id}/model"
        mlflow.register_model(model_uri, model_name)

    registered_models = mlflow.search_runs(filter_string="")
    registered_models = registered_models[
        registered_models["tags.mlflow.runName"] == "model_registry"
    ]
    registered_models = registered_models[
        ["run_id", "experiment_id", "artifact_uri", "end_time"] + metric_cols
    ]
    registered_models["score"] = registered_models[metric_cols].mean(axis=1)
    registered_models = registered_models.sort_values("score", ascending=False)
    artifact_uri = registered_models.iloc[0]["artifact_uri"]
    model_local_path = artifact_uri + "/" + model_name + "/model.pkl"
    model_path = "mlruns/" + model_local_path.split("mlruns")[-1]

    with open(model_path, "rb") as model_file:
        loaded_model = pickle.load(model_file)
    mlflow.end_run()

    name = loaded_model.model[-1].__class__.__name__
    logger.info(f"Best model in mlflow production: {name}")

    return loaded_model
