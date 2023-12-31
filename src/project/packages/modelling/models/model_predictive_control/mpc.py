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
    """Model Predictive Control Optimizer.

    This class provides functionality to optimize features in a given dataset using Optuna
    as the optimization framework. It uses a specified machine learning model to perform
    predictions and optimize control features while keeping context features fixed.

    Args:
        model (Pipeline): The machine learning model to be used for predictions and optimization.
        optimizer_params (dict): A dictionary containing various parameters for optimization
            and feature control, such as 'predict_method', 'control_features', 'boundaries',
            'data_types', 'direction', 'n_trials', and 'constraints'.

    Attributes:
        model (Pipeline): The machine learning model to be used for predictions and optimization.
        optimizer_params (dict): A dictionary containing various parameters for optimization
            and feature control.
        predict_method (str): The name of the method to be used for making predictions (e.g., 'predict' or 'predict_proba').
        features (list): List of all features in the dataset.
        control_features (list): List of features to be optimized.
        context_features (list): List of features to be kept fixed during optimization.
        boundaries (list): List of feature boundaries used during optimization.
        data_types (dict): Dictionary containing data types for each feature.
        direction (str): The optimization direction, either 'minimize' or 'maximize'.
        n_trials (int): The number of trials to run during optimization.
        constraints (list): List of optimization constraints.
        X_optimized (DataFrame): The optimized dataset with updated features.
        verbose (bool): Whether to log optimization progress.
        trial_counter (int): Counter to track the number of optimization trials.

    Example:
        # Create an instance of the class
        model = Pipeline(...)

        optimizer_params = {
            "predict_method": "predict",
            "control_features": ["feature1", "feature2"],
            "boundaries": {"feature1": (0, 1), "feature2": (0, 1)},
            "data_types": {"feature1": "float", "feature2": "int"},
            "direction": "minimize",
            "n_trials": 1000,
            "constraints": ["feature1 > 0", "feature2 > 0"],
            "verbose": True
        }
        optimizer = ModelPredictiveControlOptimizer(model, optimizer_params)

        # Optimize the dataset
        optimized_data = optimizer.optimize(X)

    """

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
        """
        Optimize features in the input dataset using a specified optimization method.

        This method optimizes the input dataset by iteratively optimizing features using a
        specific optimization method. It stores the optimized dataset in 'self.X_optimized'
        after each optimization step.

        Steps:
            1. Identify object columns in the dataset that need to be optimized.
            2. Calculate initial predictions for the dataset using the 'predict_method'.
            3. Iterate through the dataset's rows, optimizing features for each row individually.
            4. Store the optimized dataset and relevant statistics for each optimization step.
            5. If 'verbose' is True, log optimization progress including feature
                values and uplift statistics.

        Note:
        - This method may use various internal attributes and functions to perform optimization.
        - The final optimized dataset can be accessed via 'self.X_optimized'.

        Args:
            X (Union[Tensor, Matrix]): The input dataset to be optimized.
            y (Matrix, optional): The target variable, if applicable. Default is None.

        Returns:
            Union[Tensor, Matrix]: The optimized dataset with updated features.
        """
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
        """Perform inference using a specified method of the machine learning model.

        This method allows you to perform inference using a specific method of the machine
        learning model encapsulated by this class. You can provide the method name and a
        DataFrame containing the input data for inference.

        Args:
            method_name (str): The name of the method to call on the model (e.g., 'predict' or 'predict_proba').
            data (pd.DataFrame): The input data for inference as a Pandas DataFrame.

        Returns:
            float: The inference result, which can be a prediction or a probability score
            depending on the specified method.

        Raises:
            ValueError: If the specified method is not found in the model.
        """
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
        """Assign the optimized hyperparameters as features to the current dataset.

        This method takes an Optuna study object and assigns the best hyperparameters found
        during the optimization process as new features to the current dataset. The names of
        the optimized hyperparameters are prefixed with 'optimized_' and added to the
        dataset. The method then selects a subset of columns, including context features,
        control features, the optimized hyperparameters, and a 'prediction' column if
        present, and returns the modified dataset.

        Args:
            current_study (optuna.Study): The Optuna study containing information about the
            best hyperparameters.

        Returns:
            pd.DataFrame: The modified DataFrame with optimized hyperparameters added as
            features.
        """
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
        """Check NaN values on starting point.

        Check for NaN values in the starting point and replace them with the mean value
        of the corresponding feature boundaries.

        This method iterates through the starting point dictionary and checks if any of
        its values are NaN. If a NaN value is found for a feature, it logs a warning and
        replaces the NaN value with the mean value of the feature's boundaries.

        Note:
            This method assumes that the 'starting_point' dictionary contains feature names
            as keys and initial values as values.
        """
        for feat in self.starting_point.keys():
            if math.isnan(self.starting_point[feat]):
                logger.warning(
                    f"starting point for feature {feat} is NaN, replacing with mean value of boundaries"
                )
                self.starting_point[feat] = np.mean(self.boundaries[feat])

    def _get_feature_boundary(self, feat: str) -> tp.Tuple[float, float]:
        """Get the boundary values for a specific feature.

        This method retrieves the boundary values for a given feature. If it is the
        first trial (trial_counter == 0), the boundary is set to a single value equal to
        the starting point of the feature. For subsequent trials, the boundary is taken
        from the pre-defined boundaries dictionary.

        Args:
            feat (str): The name of the feature for which boundary values are needed.

        Returns:
            Tuple[float, float]: A tuple containing the lower and upper boundaries for
            the specified feature.
        """
        if self.trial_counter == 0:
            boundary = (self.starting_point[feat], self.starting_point[feat])
        else:
            boundary = self.boundaries[feat]
        return boundary

    def _get_trial_object(self, trial: optuna.Trial) -> tp.Dict[str, optuna.trial.Trial]:
        """Generate a dictionary of trial parameters for optimization using Optuna.

        This method generates a dictionary of trial parameters for optimization using
        Optuna based on the control features and their respective data types. It uses the
        Optuna trial object to suggest values for each feature according to its data type
        (binary, int, float) and the defined boundaries.

        Args:
            trial (optuna.Trial): The Optuna trial object used for parameter suggestion.

        Returns:
            Dict[str, optuna.trial.Trial]: A dictionary containing trial parameters for each
            control feature.

        Raises:
            KeyError: If a control feature has an unsupported data type.
        """
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
        """Optimize the objective function using Optuna.

        This method performs optimization of the objective function using Optuna. It creates
        an Optuna study with specified optimization settings, including direction, sampler,
        and pruners. The optimization process runs for the specified number of trials.

        Args:
            direction (str, optional): The optimization direction, either 'minimize' or
            'maximize'. Defaults to 'minimize'.
            n_trials (int, optional): The number of trials to run during optimization.
            Defaults to 1000.

        Returns:
            optuna.Study: The Optuna study containing optimization results.
        """
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
        """Define the objective function for optimization using Optuna.

        This method defines the objective function to be optimized using Optuna. It generates
        trial parameters, combines control and context data, and performs inference using the
        specified method. The goal is to find the parameters that optimize the prediction
        according to the defined criteria.

        Args:
            trial (optuna.Study): The Optuna trial object used for optimization.

        Returns:
            float: The prediction result to be optimized.

        Raises:
            optuna.TrialPruned: If the trial should be pruned based on early stopping
            criteria.
        """
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
