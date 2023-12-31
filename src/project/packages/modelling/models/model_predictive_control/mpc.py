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
from .constraints import ExpressionConstraintsPruner

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

    def __init__(
        self, model: Pipeline, params: tp.Dict[str, str]
    ) -> "ModelPredictiveControlOptimizer":
        self.model = model
        self.params = params
        self.direction = params["direction"]
        self.n_trials = params.get("n_trials", 1000)
        self.constraints = params.get("constraints", [])
        self.unnecessary_features = params.get("unnecessary_features", [])
        self.features = model.model[0].columns
        self.control_features = params.get("control_features", [])
        self.context_features = [
            feat for feat in self.features if feat not in self.control_features
        ]
        self.boundaries = params.get("boundaries", [])
        self.data_types = params.get("data_types", {})
        self.trial_counter = 0

    def optimize(self, X: tp.Union[Tensor, Matrix], y: Matrix = None) -> tp.Union[Tensor, Matrix]:
        self.object_columns = [
            col for col in list(X.select_dtypes(include=["object"]).columns) if col in self.features
        ]
        X["prediction"] = self.model.predict(X)
        X["proba_prediction"] = self.model.predict_proba(X)[:, 1]

        dfs = []
        for index in X.index:
            self.current_data = X[X.index == index]
            # set starting point during optimization
            self.starting_point = (
                self.current_data[self.control_features].reset_index(drop=True).T.to_dict()[0]
            )
            self._check_nan_values_on_starting_point()

            # optimize
            self.current_study = self._optimize(direction=self.direction, n_trials=self.n_trials)
            self.current_data = self._assign_optimized_features(self.current_study)
            self.current_optimized_value = self.current_study.best_value

            # save results
            self.current_data["optimized_value"] = self.current_optimized_value
            self.current_data["uplift"] = (
                self.current_optimized_value - self.current_data["proba_prediction"]
            )

            # analyse study results
            self.study_status = self.current_study.trials_dataframe()
            self.percentage_of_complete_status = round(
                self.study_status[self.study_status["state"] == "COMPLETE" ""].shape[0]
                / self.study_status.shape[0]
                * 100,
                1,
            )

            logger.info(f"Study status: {self.percentage_of_complete_status}% of trials completed")

            self.trial_counter = 0
            dfs.append(self.current_data)

        X_optimized = pd.concat(dfs, axis=0)
        self.X_optimized = X_optimized

        return X_optimized

    def _assign_optimized_features(self, current_study: optuna.Study) -> pd.DataFrame:
        best_controlers = current_study.best_params
        best_controlers = {"optimized_" + key: value for key, value in best_controlers.items()}
        for opt_feat, value in best_controlers.items():
            self.current_data[opt_feat] = value
        self.current_data = self.current_data[
            self.context_features
            + self.control_features
            + list(best_controlers.keys())
            + ["prediction", "proba_prediction"]
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

    def _get_trial_object(self, trial: optuna.Trial) -> tp.Dict[str, optuna.trial.Trial]:
        trial_object = {}
        for feat in self.control_features:
            classification = self.data_types[feat]
            if self.trial_counter == 0:
                boundary = (self.starting_point[feat], self.starting_point[feat])
            else:
                boundary = self.boundaries[feat]

            if classification == "binary":
                trial_param = trial.suggest_categorical(feat, [0, 1])

            elif classification == "int":

                trial_param = trial.suggest_int(feat, boundary[0], boundary[1])

            elif classification == "float":
                trial_param = trial.suggest_float(feat, boundary[0], boundary[1])

            else:
                msg = (
                    f"feature {feat} with data type as **{classification}** is not available as optuna trial param, "
                    "please try with **object, binary, int or float** options"
                )
                raise KeyError(msg)
            trial_object[feat] = trial_param
            self.current_trial_object = trial_object

        return trial_object

    def _optimize(self, direction: str = "maximize", n_trials: int = 1000) -> optuna.Study:
        study = optuna.create_study(
            direction=direction,
            sampler=load_object(self.DEFAULT_SAMPLER_PARAMS),
            pruner=ExpressionConstraintsPruner(self.constraints),
        )
        study.optimize(self.objective_function, n_trials=n_trials)
        optimized_features = study.best_params
        best_value = study.best_value

        logger.info(f"Optimized features: {optimized_features}")
        logger.info(f"Optimized value: {best_value}")
        return study

    def objective_function(self, trial: optuna.Study) -> float:

        trial_object = self._get_trial_object(trial)

        if trial.should_prune() and self.trial_counter > 0:
            raise optuna.TrialPruned()

        X_control = pd.DataFrame.from_dict(trial_object, orient="index").T
        X_context = self.current_data.drop(columns=self.control_features).reset_index(drop=True)

        X_optimize = pd.concat([X_context, X_control], axis=1)
        X_optimize = X_optimize[self.features]

        # TODO: add this to the predict method
        # predict = self.model.predict(X_optimize)[0]

        predict = self.model.predict_proba(X_optimize)[0][1]

        self.trial_counter += 1
        return predict
