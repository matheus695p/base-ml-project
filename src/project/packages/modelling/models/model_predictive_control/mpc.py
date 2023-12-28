import logging

import optuna
import pandas as pd
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)


class ModelPredictiveControlExplorer(BaseEstimator):
    def __init__(self, model, preprocessors, params, data_schemas):
        self.model = model
        self.params = params
        self.direction = params["direction"]
        self.n_trials = params.get("n_trials", 1000)
        self.preprocessors = preprocessors
        self.data_schemas = data_schemas
        self.constraints = params.get("constraints", {})
        self.unnecessary_features = self.constraints.get("unnecessary_features", [])

    def fit(self, X, y=None):
        self.features = list(X.columns)
        self.features = [col for col in self.features if col != self.model.target.title()]
        self.object_columns = list(X.select_dtypes(include=["object"]).columns)

        self.get_meta_parameters(X)
        self.optimized_study = self.optimize(direction=self.direction, n_trials=self.n_trials)
        return self

    def predict(self, X):
        return NotImplementedError("Method not implemented yet")

    def cast_values_df(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.data_schemas.keys():
            cast_to = self.data_schemas[col]["dtype"]
            try:
                df[col] = df[col].astype(cast_to)
            except Exception:
                pass
        return df

    def get_meta_parameters(self, X):
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

    def objective_function(self, trial):
        trial_object = self.get_trial_object_exploration(trial)
        data = pd.DataFrame.from_dict(trial_object, orient="index").T
        data = self.cast_values_df(data)
        for preprocessor in self.preprocessors:
            data = preprocessor.transform(data)

        proba_prediction = self.model.predict_proba(data)[0][1]
        return proba_prediction

    def optimize(self, direction="maximize", n_trials=1000):
        study = optuna.create_study(direction=direction)
        study.optimize(self.objective_function, n_trials=n_trials)
        optimized_features = study.best_params
        best_value = study.best_value

        logger.info(f"Optimized features: {optimized_features}")
        logger.info(f"Optimized value: {best_value}")
        return study
