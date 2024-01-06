import logging

import optuna
import pandas as pd
import typing as tp
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from ....python_utils.typing import Matrix, Vector

logger = logging.getLogger(__name__)


class ModelPredictiveControlExplorer(BaseEstimator):
    """A class for performing model predictive control exploration.

    This class enables model predictive control exploration with a given model and
    associated parameters, preprocessing steps, and data schemas.

    Args:
        model (Pipeline): The machine learning model to be used for exploration.
        preprocessors (Pipeline): A list of data preprocessing steps to be applied to the input data.
        params (Dict[str, str]): A dictionary of parameters for the exploration.
        data_schemas (Dict[str, str]): A dictionary defining data schemas for casting and handling data.

    Attributes:
        model (Pipeline): The machine learning model used for exploration.
        params (Dict[str, str]): The exploration parameters.
        direction (str): The direction of optimization (e.g., 'maximize' or 'minimize').
        n_trials (int): The number of exploration trials.
        preprocessors (Pipeline): The list of data preprocessing steps.
        data_schemas (Dict[str, str]): The data schemas for casting and handling data.
        constraints (Dict[str, Any]): Additional constraints for the exploration.
        unnecessary_features (List[str]): Features marked as unnecessary in constraints.

    """

    def __init__(
        self,
        model: Pipeline,
        preprocessors: Pipeline,
        params: tp.Dict[str, str],
        data_schemas: tp.Dict[str, str],
    ) -> "ModelPredictiveControlExplorer":
        self.model = model
        self.params = params
        self.direction = params["direction"]
        self.n_trials = params.get("n_trials", 1000)
        self.preprocessors = preprocessors
        self.data_schemas = data_schemas
        self.constraints = params.get("constraints", {})
        self.unnecessary_features = self.constraints.get("unnecessary_features", [])

    def fit(self, X: Matrix, y: Vector = None) -> "ModelPredictiveControlExplorer":
        """Fit the exploration model and perform optimization.

        This method fits the exploration model, performs optimization, and sets
        the optimized study and features.

        Args:
            X (Matrix): The input data for optimization.
            y (Vector, optional): The target labels (not used in this implementation).

        Returns:
            ModelPredictiveControlExplorer: The fitted instance of the class.

        """
        self.features = list(X.columns)
        self.features = [col for col in self.features if col != self.model.target.title()]
        self.object_columns = list(X.select_dtypes(include=["object"]).columns)

        self.get_meta_parameters(X)
        self.optimized_study = self.optimize(direction=self.direction, n_trials=self.n_trials)
        return self

    def predict(self, X: Matrix) -> Vector:
        return NotImplementedError("Method not implemented yet")

    def cast_values_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cast data values in a DataFrame according to data schemas.

        This method casts data values in the DataFrame to specified data types
        based on the data schemas.

        Args:
            df (pd.DataFrame): The DataFrame with data to be cast.

        Returns:
            pd.DataFrame: The DataFrame with casted data values.

        """
        for col in self.data_schemas.keys():
            cast_to = self.data_schemas[col]["dtype"]
            try:
                df[col] = df[col].astype(cast_to)
            except Exception:
                pass
        return df

    def get_meta_parameters(self, X: pd.DataFrame) -> tp.Dict[str, str]:
        """Extract meta-parameters and feature classifications from input data.

        This method extracts meta-parameters and feature classifications from the input data.

        Args:
            X (pd.DataFrame): The input data for extracting meta-parameters.

        Returns:
            tp.Dict[str, str]: A dictionary containing extracted meta-parameters.

        """
        feature_classification = {}
        boundaries = {}
        for feat in self.features:
            data = X[[feat]].dropna()
            if feat in self.object_columns:
                feature_classification[feat] = "object"
                boundaries[feat] = list(data[feat].unique())
            else:
                data_min = data[feat].min()
                data_max = data[feat].max()
                try:
                    data[feat] = pd.to_numeric(data[feat], downcast="integer", errors="raise")
                    is_integer_column = data[feat].apply(lambda x: isinstance(x, int)).all()
                except (ValueError, TypeError):
                    is_integer_column = False
                unique_values = data[feat].nunique()
                if unique_values == 2 and is_integer_column:
                    feature_classification[feat] = "binary"
                elif unique_values != 2 and is_integer_column:
                    feature_classification[feat] = "integer"
                else:
                    feature_classification[feat] = "continuous"

                boundaries[feat] = (data_min, data_max)

        self.boundaries = boundaries
        self.feature_classification = feature_classification
        return self

    def get_trial_object_exploration(self, trial):
        """Generate a trial object for exploration based on trial parameters.

        This method generates a trial object for exploration based on trial parameters.

        Args:
            trial (optuna.Trial): The Optuna trial object.

        Returns:
            tp.Dict[str, tp.Any]: A dictionary representing trial parameters for exploration.

        """
        trial_object = {}
        for feat, classification in self.feature_classification.items():
            boundary = self.boundaries[feat]

            if classification == "object":
                if feat in self.unnecessary_features:
                    trial_param = trial.suggest_categorical(feat, ["Unknown"])
                else:
                    trial_param = trial.suggest_categorical(feat, boundary)

            elif classification == "binary":
                trial_param = trial.suggest_categorical(feat, [0, 1])

            elif classification == "integer":
                if feat in self.unnecessary_features:
                    trial_param = trial.suggest_int(feat, 0, 1, step=1)
                else:
                    trial_param = trial.suggest_int(feat, boundary[0], boundary[1])

            else:
                # continous
                trial_param = trial.suggest_float(feat, boundary[0], boundary[1])

            trial_object[feat] = trial_param
        return trial_object

    def objective_function(self, trial: optuna.Trial) -> float:
        """Objective function for optimization.

        This method defines the objective function to be optimized during exploration.

        Args:
            trial (optuna.Trial): The Optuna trial object.

        Returns:
            float: The value to be optimized during exploration.

        """
        trial_object = self.get_trial_object_exploration(trial)
        data = pd.DataFrame.from_dict(trial_object, orient="index").T
        data = self.cast_values_df(data)
        for preprocessor in self.preprocessors:
            data = preprocessor.transform(data)

        try:
            proba_prediction = self.model.predict_proba(data)[0][1]
        except Exception:
            proba_prediction = self.model.predict(data)
        return proba_prediction

    def optimize(self, direction: str = "maximize", n_trials: int = 1000) -> optuna.Study:
        """Perform optimization of the exploration.

        This method performs optimization of the exploration using Optuna.

        Args:
            direction (str, optional): The direction of optimization (e.g., 'maximize' or 'minimize').
                Defaults to 'maximize'.
            n_trials (int, optional): The number of optimization trials. Defaults to 1000.

        Returns:
            optuna.Study: The optimized study object.

        """
        study = optuna.create_study(direction=direction)
        study.optimize(self.objective_function, n_trials=n_trials)
        optimized_features = study.best_params
        best_value = study.best_value

        logger.info(f"Optimized features: {optimized_features}")
        logger.info(f"Optimized value: {best_value}")
        return study
