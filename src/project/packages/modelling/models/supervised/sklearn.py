import logging
import typing as tp
from copy import deepcopy

import optuna
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline

from project.packages.python_utils.load.object_injection import load_object
from project.packages.python_utils.typing.tensors import Matrix, Tensor

from ...reproducibility.set_seed import seed_file
from ...transformers.columns_selector import ColumnsSelector
from ...evaluate.classification_metrics import compute_binary_classification_metrics
from ..mlflow.metrics import MlflowTransformations

seed_file()


logger = logging.getLogger(__name__)


class BaseSklearnCompatibleModel(BaseEstimator, MlflowTransformations):
    def __init__(self, params: tp.Dict[str, str]) -> "BaseSklearnCompatibleModel":

        self.params = params
        self.is_fitted = False
        self.target = params.get("target", None)
        self.features = params.get("features", None)
        self.scoring_metrics = params.get("scoring_metrics", [])

    def get_params(self, deep: bool = True) -> tp.Dict[str, str]:
        return self.params

    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None) -> "BaseSklearnCompatibleModel":
        seed_file()
        self.hypertune_results = self.hypertune_cross_validated_model(
            X=X,
            y=y,
        )
        self.best_params = self.hypertune_results["best_trial_params"]
        self.model = self.build_model_pipeline(self.best_params)
        self.model = self.model.fit(X, y)
        self.is_fitted = True
        self.X_train = X
        self.y_train = y

        return self

    def build_model_pipeline(self, params):
        pipeline = Pipeline(
            [
                ("columns_selector", ColumnsSelector(columns=params["features"])),
                ("imputer", load_object(params["pipeline"]["imputer"])),
                ("scaler", load_object(params["pipeline"]["scaler"])),
                ("model", load_object(params["pipeline"]["model"])),
            ],
        )
        return pipeline

    def predict(self, X: pd.DataFrame):
        return self.model.predict(X)

    def hypertune_cross_validated_model(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
    ) -> dict[optuna.Study, dict]:
        """Performs hyperparameter tuning using cross-validation and Optuna library in Python.

        Args:
        X (pd.DataFrame): A pandas DataFrame containing the input features for the model.
        y (pd.DataFrame): Dataframe indexed with the target variable or dependent variable
            in a machine learning model.
        params (tp.Dict): The `params` parameter is a dictionary containing various
            hyperparameters and settings for the function `hypertune_cross_validated_model()`.

        Returns:
        a dictionary containing the optuna study object and the best trial parameters for
        the estimator.
        """
        kwargs_study = self.params.get("optuna")["kwargs_study"]
        kwargs_optimize = self.params.get("optuna")["kwargs_optimize"]
        sampler = load_object(self.params.get("optuna")["sampler"])
        pruner = load_object(self.params.get("optuna")["pruner"])

        # kwargs to inject in the objective function
        kwargs = {
            "X": deepcopy(X),
            "y": deepcopy(y),
            "params": deepcopy(self.params),
        }

        # optuna objective function
        objective = lambda trial: self.hypertune_cross_validation_objective_function(  # noqa: E731
            trial, **deepcopy(kwargs)
        )

        # optimize study
        study = optuna.create_study(
            sampler=sampler,
            pruner=pruner,
            **kwargs_study,
        )

        study.optimize(objective, **kwargs_optimize)

        # retrieve the information from the best iteration
        best_trial = study.best_trial
        best_estimator_number = best_trial.number
        best_estimator_params = study.trials[best_estimator_number].user_attrs["estimator_params"]

        # save best iteration using callback
        best_value = self.hypertune_cross_validation_objective_function(
            best_trial, callbacks=[self.save_best_model], **deepcopy(kwargs)
        )
        logger.debug(f"best_value: {best_value}")

        # compute test scoring
        final_estimator = self.build_model_pipeline(best_estimator_params)

        final_score = self.cross_validate_estimator(
            final_estimator,
            X,
            y,
            deepcopy(best_estimator_params),
        )
        scoring_metric = best_estimator_params.get("cv_score", {}).get("scoring", "")
        logger.info(f"final estimator: {final_estimator}")
        logger.info(f"final estimator {scoring_metric}: {final_score}")

        # mlflow logging metrics transform
        cross_validation_metrics = self.format_metrics_dict(self.scores)

        return {
            "study": study,
            "best_trial_params": best_estimator_params,
            "cross_validation_metrics": cross_validation_metrics,
        }

    def cross_validate_estimator(
        self,
        estimator: Pipeline,
        X: tp.Union[Matrix, Tensor],
        y: tp.Union[Matrix, Tensor],
        params: dict,
    ) -> float:
        """Perform cross-validation a estimator using the input data and parameters.

        Args:
        estimator (Pipeline): A pipeline containing the stacked models and the
            final meta-model.
        X (Union[Matrix, Tensor]): Input data matrix or tensor.
        y (Union[Matrix, Tensor]): Target data matrix or tensor.
        params (Dict): A dictionary containing the following keys:
            - "cv_strategy" (str): The type of cross-validation
                strategy to be used.
            - "cv_score" (Dict): A dictionary containing
            the following keys:
            - "type" (str): The type of cross-validation score to be used.
            - "kwargs" (Dict): A dictionary containing the arguments to be
                passed to the score function.

        Returns:
            float: The mean score of the cross-validation.
        """
        # cross validate the model
        cv_strategy = load_object(params["cv_strategy"])
        cv_score_params = params["cv_score"]
        cv_score_params["kwargs"]["X"] = X
        cv_score_params["kwargs"]["y"] = y[y.columns[0]].ravel()
        cv_score_params["kwargs"]["estimator"] = estimator
        cv_score_params["kwargs"]["cv"] = cv_strategy

        scoring_metric = cv_score_params.get("scoring")
        y_pred = load_object(cv_score_params)
        y_score = None

        if hasattr(estimator, "predict_proba"):
            cv_score_params["kwargs"]["method"] = "predict_proba"
            y_score = load_object(cv_score_params)
        scores = self.evaluate(y_true=y, y_pred=y_pred, y_score=y_score)

        # assign attributes
        self.y_pred = pd.DataFrame(y_pred, index=y.index, columns=["prediction"])
        self.y_score = pd.DataFrame(y_score, index=y.index)
        self.scores = scores
        return scores[scoring_metric]

    def save_best_model(self, study: optuna.Study, trial: optuna.trial):
        """Callback to save best model params."""
        if study.best_trial.number == trial.number:
            study.set_user_attr(key="estimator_params", value=trial.user_attrs["estimator_params"])

    def hypertune_cross_validation_objective_function(
        self,
        trial: optuna.Trial,
        **kwargs,
    ) -> float:
        """Hypertune thorough cross validation a Model pipeline.

        It takes a trial object, and a kwargs of parameters, and returns the
        cross validation result for a given trial model pipeline.

        Args:
        trial (optuna.Trial): optuna.Trial
        **kwargs (tp.Any): kwargs of parameters for cross validation.

        Returns:
        Cross validation error.
        """
        params = kwargs.get("params", {})
        X = kwargs.get("X")
        y = kwargs.get("y")

        # inject trial parameters
        new_data_params = self.inject_trial_parameter(deepcopy(params), trial)
        trial.set_user_attr(key="estimator_params", value=new_data_params)

        # build meta estimator
        estimator = self.build_model_pipeline(new_data_params)

        # cross validation score
        score = self.cross_validate_estimator(
            estimator,
            X,
            y,
            deepcopy(new_data_params),
        )

        return score

    def inject_trial_parameter(self, d: tp.Dict[str, tp.Any], trial: optuna.Trial):
        """Inject optuna trial parameter for a objective function.

        It takes a dictionary and a trial object, and replaces any string that starts
        with "trial." with the corresponding value from the trial object.

        Args:
        d (dict): dict
        trial (optuna.Trial): optuna.Trial

        Returns:
        A dictionary with the values of the dictionary d replaced with the values of the
        optuna trial.
        """
        for k, v in d.items():
            if isinstance(v, dict):
                self.inject_trial_parameter(v, trial)
            elif isinstance(v, str) and "trial." in v:
                d[k] = eval(v, {"trial": trial})
        return d


class BinaryClassifierSklearnPipeline(BaseSklearnCompatibleModel, ClassifierMixin):
    def __init__(self, params: tp.Dict[str, str]) -> "BinaryClassifierSklearnPipeline":
        super().__init__(params)

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        # Use the predict_proba method of the fitted model
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        elif hasattr(self.model, "predict"):
            return self.model.predict(X)
        else:
            raise AttributeError("The fitted model does not have a 'predict_proba' method.")

    def evaluate(self, y_true, y_pred, y_score=None):
        """Evaluate the model performance.

        Args:
            y_true (pd.DataFrame): The true labels.
            y_pred (pd.DataFrame): The predicted labels.
            y_score (pd.DataFrame, optional): The predicted scores of probabilities. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the evaluation metrics.
        """
        if y_score is not None:
            y_score = y_score[:, 1]
            import numpy as np

            if np.isnan(y_score).any():
                y_score = np.nan_to_num(y_score, nan=0.0)
                logger.warning("Nan values encountered filling with 0")

        return compute_binary_classification_metrics(y_true=y_true, y_pred=y_pred, y_score=y_score)
