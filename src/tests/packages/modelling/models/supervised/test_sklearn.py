import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from project.packages.modelling.models.supervised.sklearn import (
    BaseSklearnCompatibleModel,
    BinaryClassifierSklearnPipeline,
)


model_params = {
    'scoring_metrics': [
        'accuracy',
        'balanced_accuracy',
        'f1',
        'f1_micro',
        'f1_macro',
        'f1_weighted',
        'precision',
        'precision_micro',
        'precision_macro',
        'precision_weighted',
        'recall',
        'recall_micro',
        'recall_macro',
        'recall_weighted',
        'roc_auc',
        'roc_auc_ovr',
        'roc_auc_ovo',
        'roc_auc_ovr_weighted',
        'roc_auc_ovo_weighted',
    ],
    'optuna': {
        'kwargs_study': {'direction': 'maximize', 'study_name': 'xgboost', 'load_if_exists': False},
        'kwargs_optimize': {'n_trials': 10},
        'sampler': {
            'class': 'optuna.samplers.TPESampler',
            'kwargs': {'n_startup_trials': 0, 'constant_liar': True, 'seed': 42},
        },
        'pruner': {'class': 'optuna.pruners.SuccessiveHalvingPruner', 'kwargs': {}},
    },
    'cv_strategy': {
        'class': 'sklearn.model_selection.StratifiedKFold',
        'kwargs': {'n_splits': 5, 'random_state': 42, 'shuffle': True},
    },
    'cv_score': {
        'scoring': 'f1_weighted',
        'class': 'sklearn.model_selection.cross_val_predict',
        'kwargs': {
            'estimator': None,
            'X': None,
            'y': None,
            'cv': None,
            'n_jobs': -1,
            'method': 'predict',
        },
    },
    "target": "target_column",
    "features": ["feature1", "feature2", "feature3", "feature4", "feature5"],
    'pipeline': {
        'imputer': {
            'class': 'project.packages.modelling.models.unsupervised.imputer.ColumnsPreserverImputer',
            'kwargs': {
                'imputer_params': {
                    'class': 'sklearn.impute.KNNImputer',
                    'kwargs': {
                        'n_neighbors': 'trial.suggest_int("knn_imputer__n_neighbors", 2, 20, step=1)',
                        'weights': 'trial.suggest_categorical("knn_imputer__weights", ["distance", "uniform"])',
                    },
                }
            },
        },
        'scaler': {
            'class': 'project.packages.modelling.transformers.scaler.ColumnsPreserverScaler',
            'kwargs': {
                'scaler_params': {
                    'class': 'trial.suggest_categorical("scaler__transformer", ["project.packages.modelling.transformers.scaler.NotScalerTransformer", "sklearn.preprocessing.PowerTransformer", "sklearn.preprocessing.QuantileTransformer"])',
                    'kwargs': {},
                }
            },
        },
        'feature_selector': {
            'class': 'project.packages.modelling.feature_selection.feature_selector_pipeline.FeatureSelector',
            'kwargs': {
                'fs_params': {
                    'selectors': ['model_based'],
                    'model_based': {
                        'bypass_features': ['feature1'],
                        'estimator': {
                            'class': 'xgboost.XGBClassifier',
                            'kwargs': {
                                'n_estimators': 'trial.suggest_int("fs_mb_xgboost__n_estimators", 10, 500, step=10)',
                                'max_depth': 'trial.suggest_int("fs_mb_xgboost__max_depth", 2, 10)',
                                'random_state': 42,
                            },
                        },
                        'threshold': 'trial.suggest_float("fs_mb__threshold", 0.001, 0.1)',
                        'prefit': False,
                    },
                }
            },
        },
        'model': {
            'class': 'xgboost.XGBClassifier',
            'kwargs': {
                'n_estimators': 'trial.suggest_int("xgboost__n_estimators", 10, 500, step=5)',
                'learning_rate': 'trial.suggest_float("xgboost__learning_rate", 0.0001, 1)',
                'min_child_weight': 'trial.suggest_int("xgboost__min_child_weight", 0, 500, step=1)',
                'max_depth': 'trial.suggest_int("xgboost__max_depth", 1, 8)',
                'subsample': 'trial.suggest_float("xgboost__subsample", 0.5, 1)',
                'reg_lambda': 'trial.suggest_float("xgboost__reg_lambda", 0, 5)',
                'reg_alpha': 'trial.suggest_float("xgboost__reg_alpha", 0, 1)',
                'random_state': 42,
            },
        },
    },
}


# Generate sample data for testing.
X, y = make_classification(n_samples=100, n_features=5, random_state=42)
X = pd.DataFrame(X, columns=["feature1", "feature2", "feature3", "feature4", "feature5"])
y = pd.DataFrame(y, columns=["target_column"])

model = BinaryClassifierSklearnPipeline(model_params)


class TestBaseSklearnCompatibleModel:
    def test_get_params(
        self,
    ):
        base_model = BaseSklearnCompatibleModel(model_params)
        # Get model parameters.
        params = base_model.get_params()

        # Ensure that the returned parameters match the initial configuration.
        assert params == model_params


class TestBinaryClassifierSklearnPipeline:
    def test_fit_predict_evaluate(self):
        model = BinaryClassifierSklearnPipeline(model_params)
        model.fit(X, y)

        # Ensure that the model is fitted.
        assert model.is_fitted
        assert isinstance(model.model, Pipeline)

        # Make predictions on the same data.
        y_pred = model.predict(X)

        # Evaluate the model.
        metrics = model.evaluate(y_true=y, y_pred=y_pred)

        # Ensure that evaluation metrics are computed correctly.
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics

    def test_predict_proba(self):
        model = BinaryClassifierSklearnPipeline(model_params)
        model.fit(X, y)

        # Make probability predictions on the same data.
        y_proba = model.predict_proba(X)

        # Ensure that probability predictions are obtained correctly.
        assert isinstance(y_proba, np.ndarray)
        assert y_proba.shape[0] == X.shape[0]
        assert y_proba.shape[1] == 2  # Binary classification should have two classes.

    def test_get_params(self):
        model = BinaryClassifierSklearnPipeline(model_params)

        # Get model parameters.
        params = model.get_params()

        # Ensure that the returned parameters match the initial configuration.
        assert params == model_params

    def test_hypertune_cross_validated_model(
        self,
    ):
        model = BinaryClassifierSklearnPipeline(model_params)
        results = model.hypertune_cross_validated_model(X, y)

        # Ensure that results contain expected keys.
        assert "study" in results.keys()
        assert "best_trial_params" in results.keys()
        assert "cross_validation_metrics" in results.keys()

        # Ensure that cross-validation metrics are computed correctly.
        assert isinstance(results["cross_validation_metrics"], dict)

    def test_predict(
        self,
    ):
        model = BinaryClassifierSklearnPipeline(model_params)
        model.fit(X, y)

        # Make probability predictions on the same data.
        y_pred = model.predict(X)

        # Ensure that probability predictions are obtained correctly.
        assert isinstance(y_pred, np.ndarray)
        assert y_pred.shape[0] == X.shape[0]
