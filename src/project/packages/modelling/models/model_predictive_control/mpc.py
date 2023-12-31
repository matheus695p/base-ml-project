import logging
import math
import typing as tp

import numpy as np
import optuna
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from project.packages.python_utils.load.object_injection import load_object

from ....python_utils.typing import Matrix, Tensor
from .constraints import MultiplePruners, ExpressionConstraintsPruner


logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


class ModelPredictiveControlOptimizer(BaseEstimator):

    DEFAULT_SAMPLER_PARAMS = {
        "class": "optuna.samplers.TPESampler",
        "kwargs": {
            "n_startup_trials": 0,
            "constant_liar": True,
            "seed": 42,
        },
    }

    DEFAULT_BASE_PRUNER_PARAMS = {
        "class": "optuna.pruners.SuccessiveHalvingPruner",
        "kwargs": {},
    }

    DEFAULT_THRESHOLD_PRUNER_PARAMS = {
        "class": "optuna.pruners.ThresholdPruner",
        "kwargs": {"upper": np.inf, "lower": -np.inf},
    }

    def __init__(
        self, model: Pipeline, optimizer_params: tp.Dict[str, str]
    ) -> "ModelPredictiveControlOptimizer":
        # model optimizer_params
        self.model = model
        self.optimizer_params = optimizer_params
        self.predict_method = optimizer_params.get("predict_method", "predict")

        # feature to optimize
        self.features = model.model[0].columns
        self.control_features = optimizer_params.get("control_features", [])
        self.context_features = [
            feat for feat in self.features if feat not in self.control_features
        ]

        # optimization optimizer_params
        self.boundaries = optimizer_params.get("boundaries", [])
        self.data_types = optimizer_params.get("data_types", {})
        self.direction = optimizer_params.get("direction", "minimize")
        self.n_trials = optimizer_params.get("n_trials", 1000)
        self.constraints = optimizer_params.get("constraints", [])

        # objective threshold optimizer_params
        self.DEFAULT_THRESHOLD_PRUNER_PARAMS["kwargs"]["lower"] = optimizer_params.get(
            "objective_value_boundaries", [-np.inf, np.inf]
        )[0]
        self.DEFAULT_THRESHOLD_PRUNER_PARAMS["kwargs"]["upper"] = optimizer_params.get(
            "objective_value_boundaries", [-np.inf, np.inf]
        )[1]

        # preprocessing transformers
        # preprocessing transformers include columns selector and
        # nan values imputer, transformer is used to get the starting
        # point of optimization
        self.preprocessing_transformers = self.model.model[:2]

        # trail numbers
        self.verbose = optimizer_params.get("verbose", False)
        self.trial_counter = 0

    def optimize(self, X: tp.Union[Tensor, Matrix], y: Matrix = None) -> tp.Union[Tensor, Matrix]:
        self.object_columns = [
            col for col in list(X.select_dtypes(include=["object"]).columns) if col in self.features
        ]
        X["prediction"] = self.inference(method_name=self.predict_method, data=X)

        dfs = []
        for index in X.index:
            self.current_data = X[X.index == index]
            # set starting point during optimization
            self.starting_point = (
                self.preprocessing_transformers.transform(self.current_data)[self.control_features]
                .reset_index(drop=True)
                .T.to_dict()[0]
            )
            self._check_nan_values_on_starting_point()

            # optimize
            self.current_study = self._optimize(direction=self.direction, n_trials=self.n_trials)
            self.current_data = self._assign_optimized_features(self.current_study)
            self.current_optimized_value = self.current_study.best_value

            # save results
            self.current_data["optimized_value"] = self.current_optimized_value
            self.current_data["uplift"] = (
                self.current_optimized_value - self.current_data["prediction"]
            )

            # analyse study results
            self.study_status = self.current_study.trials_dataframe()
            self.percentage_of_complete_status = round(
                self.study_status[self.study_status["state"] == "COMPLETE" ""].shape[0]
                / self.study_status.shape[0]
                * 100,
                1,
            )
            # reset trial counter for next optimization step
            self.trial_counter = 0
            dfs.append(self.current_data)

            if self.verbose:
                X_optimized = pd.concat(dfs, axis=0)
                self.uplift = round(self.current_data["uplift"].mean() * 100, 2)
                self.global_uplift = round(X_optimized["uplift"].mean() * 100, 2)
                msg = (
                    f"Optimized features: {self.current_study.best_params}" + "\n"
                    f"Uplift: {self.uplift} [%] / Global uplift: {self.global_uplift} [%]" + "\n"
                    f"Study status: {self.percentage_of_complete_status}% of trials completed"
                )
                logger.info(msg)

        X_optimized = pd.concat(dfs, axis=0)
        self.X_optimized = X_optimized

        return X_optimized

    def inference(self, method_name: str, data: pd.DataFrame) -> float:
        if hasattr(self.model, method_name) and callable(getattr(self.model, method_name)):
            method = getattr(self.model.model, method_name)
            result = method(data)

            if method_name == "predict":
                prediction = result
            elif method_name == "predict_proba":
                prediction = result[:, 1]
            return prediction

        else:
            raise ValueError(f"Method {method_name} not found in model {self.model}")

    def _assign_optimized_features(self, current_study: optuna.Study) -> pd.DataFrame:
        best_controlers = current_study.best_params
        best_controlers = {"optimized_" + key: value for key, value in best_controlers.items()}
        for opt_feat, value in best_controlers.items():
            self.current_data[opt_feat] = value
        self.current_data = self.current_data[
            self.context_features
            + self.control_features
            + list(best_controlers.keys())
            + [
                "prediction",
            ]
        ]
        return self.current_data

    def _check_nan_values_on_starting_point(
        self,
    ) -> None:
        for feat in self.starting_point.keys():
            if math.isnan(self.starting_point[feat]):
                logger.warning(
                    f"starting point for feature {feat} is NaN, replacing with mean value of boundaries"
                )
                self.starting_point[feat] = np.mean(self.boundaries[feat])

    def _get_feature_boundary(self, feat: str) -> tp.Tuple[float, float]:
        if self.trial_counter == 0:
            boundary = (self.starting_point[feat], self.starting_point[feat])
        else:
            boundary = self.boundaries[feat]
        return boundary

    def _get_trial_object(self, trial: optuna.Trial) -> tp.Dict[str, optuna.trial.Trial]:
        trial_object = {}
        for feat in self.control_features:
            param_data_type = self.data_types[feat]
            boundary = self._get_feature_boundary(feat)

            if param_data_type == "binary":
                trial_param = trial.suggest_categorical(feat, [0, 1])

            elif param_data_type == "int":

                trial_param = trial.suggest_int(feat, boundary[0], boundary[1])

            elif param_data_type == "float":
                trial_param = trial.suggest_float(feat, boundary[0], boundary[1])

            else:
                msg = (
                    f"feature {feat} with data type as **{param_data_type}** is not available as optuna trial param, "
                    "please try with **object, binary, int or float** options"
                )
                raise KeyError(msg)
            trial_object[feat] = trial_param
            self.current_trial_object = trial_object

        return trial_object

    def _optimize(self, direction: str = "minimize", n_trials: int = 1000) -> optuna.Study:
        study = optuna.create_study(
            direction=direction,
            sampler=load_object(self.DEFAULT_SAMPLER_PARAMS),
            pruner=MultiplePruners(
                (
                    ExpressionConstraintsPruner(self.constraints),
                    load_object(self.DEFAULT_BASE_PRUNER_PARAMS),
                    load_object(self.DEFAULT_THRESHOLD_PRUNER_PARAMS),
                ),
            ),
        )

        study.optimize(self.objective_function, n_trials=n_trials)

        return study

    def objective_function(self, trial: optuna.Study) -> float:

        trial_object = self._get_trial_object(trial)

        if trial.should_prune() and self.trial_counter > 0:
            raise optuna.TrialPruned()

        X_control = pd.DataFrame.from_dict(trial_object, orient="index").T
        X_context = self.current_data.drop(columns=self.control_features).reset_index(drop=True)

        X_optimize = pd.concat([X_context, X_control], axis=1)
        X_optimize = X_optimize[self.features]

        predict = self.inference(method_name=self.predict_method, data=X_optimize)

        self.trial_counter += 1
        return predict
