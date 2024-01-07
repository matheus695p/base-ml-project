import pytest
import seaborn as sns
from project.packages.modelling.feature_selection.feature_selectors import ModelBasedFeatureSelector


@pytest.fixture(scope="module")
def dataset():
    titanic = sns.load_dataset("titanic")
    target_column = "survived"
    features = [
        "sibsp",
        "pclass",
        "age",
        "parch",
        "fare",
    ]
    y = titanic[[target_column]]
    X = titanic[features]

    model_params = {
        "bypass_features": [],
        "estimator": {
            "class": "xgboost.XGBClassifier",
            "kwargs": {
                "n_estimators": 10,
                "max_depth": 2,
                "random_state": 42,
            },
        },
        "threshold": 0.001,
        "prefit": False,
    }

    return X, y, model_params


class TestModelBasedFeatureSelector:
    def test_fit_transform(self, dataset):
        X, y, model_params = dataset
        feature_selector = ModelBasedFeatureSelector(model_based_params=model_params)
        X_selected = feature_selector.fit_transform(X, y)

        assert X_selected.shape[1] > 0
        assert feature_selector.columns is not None

    def test_transform(self, dataset):
        X, y, model_params = dataset
        feature_selector = ModelBasedFeatureSelector(model_based_params=model_params)
        feature_selector.fit(X, y)
        X_selected = feature_selector.transform(X)

        assert X_selected.shape[1] > 0
        assert feature_selector.columns is not None
