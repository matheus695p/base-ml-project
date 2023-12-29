import logging
import typing as tp
from copy import deepcopy

import optuna
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from project.packages.python_utils.load.object_injection import load_object
from project.packages.python_utils.typing.tensors import Matrix, Tensor

from ...reproducibility.set_seed import seed_file
from ...transformers.columns_selector import ColumnsSelector

seed_file()


logger = logging.getLogger(__name__)


class BaseSklearnCompatibleModel(BaseEstimator):
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
        best_value = self.hypertune_cross_validation_objective_function(
            best_trial, callbacks=[self.save_best_model], **deepcopy(kwargs)
        )
        logger.info(f"best_value: {best_value}")

        # compute test scoring
        cross_validation_metrics = self._compute_test_cross_validation_metrics(
            X, y, best_estimator_params, best_value, self.scoring_metrics
        )

        # mlflow logging metric transform
        cross_validation_metrics = self.transform_output_dict(cross_validation_metrics)

        return {
            "study": study,
            "best_trial_params": best_estimator_params,
            "cross_validation_metrics": cross_validation_metrics,
        }

    def _compute_test_cross_validation_metrics(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        best_estimator_params: tp.Dict[str, tp.Any],
        best_value: float,
        scoring_metrics: tp.List[str],
    ) -> tp.Dict[str, float]:
        """
        The function computes cross-validation metrics
        for a given dataset using the best estimator parameters.

        Args:
            X (pd.DataFrame): X is a pandas DataFrame containing the input features for the model.
            y (pd.DataFrame): The parameter `y` is a pandas DataFrame that represents the target variable or
                the dependent variable in your dataset. It contains the values that you are trying to predict or
                model.
            best_estimator_params (tp.Dict[str, tp.Any]): The `best_estimator_params` parameter is a
                dictionary that contains the parameters of the best estimator model. It typically includes
                information such as the model artifact (serialized model object), cross-validation score, and other
                relevant parameters.
            best_value: The parameter `best_value` is the best value obtained for the specified metric during
                the cross-validation process. It is used to store the best value for the metric in the returned
                dictionary.

        Returns:
            (tp.Dict[str, str]) a dictionary containing the best value for the specified metric, as well as
            the scores for various other metrics obtained through cross-validation.
        """
        metric = best_estimator_params["cv_score"]["kwargs"]["scoring"]
        best_value_metric = {"best_value_" + metric: best_value}
        test_params = deepcopy(best_estimator_params)
        for metric in scoring_metrics:
            logger.debug(f"Computing metric {metric}")
            estimator = self.build_model_pipeline(best_estimator_params)
            test_params["cv_score"]["kwargs"]["scoring"] = metric
            score = self.cross_validate_estimator(
                estimator,
                X,
                y,
                deepcopy(test_params),
            )
            best_value_metric[metric] = score
        return best_value_metric

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
        scores = load_object(cv_score_params)
        mean_score = scores.mean()
        return mean_score

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

    def transform_output_dict(
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
            >>> transform_output_dict(input_dict)
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


class ClassifierSklearnCompatibleModel(BaseSklearnCompatibleModel):
    def __init__(self, params: tp.Dict[str, str]) -> "ClassifierSklearnCompatibleModel":
        super().__init__(params)

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        # Use the predict_proba method of the fitted model
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        elif hasattr(self.model, "predict"):
            return self.model.predict(X)
        else:
            raise AttributeError("The fitted model does not have a 'predict_proba' method.")
