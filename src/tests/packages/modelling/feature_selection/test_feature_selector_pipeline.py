import pytest
import seaborn as sns
from sklearn.pipeline import Pipeline
from project.packages.modelling.feature_selection.feature_selector_pipeline import FeatureSelector


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

    fs_params = {
        "selectors": ["model_based"],
        "model_based": {
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
        },
    }

    return X, y, fs_params


class TestFeatureSelector:
    def test_fit_transform(self, dataset):
        X, y, fs_params = dataset
        feature_selector = FeatureSelector(fs_params=fs_params)
        X_selected = feature_selector.fit_transform(X, y)
        assert X_selected.shape[1] > 0
        assert isinstance(feature_selector.fs_pipeline, Pipeline)

    def test_transform(self, dataset):
        X, y, fs_params = dataset
        feature_selector = FeatureSelector(fs_params=fs_params)
        feature_selector.fit(X, y)
        X_selected = feature_selector.transform(X)
        assert X_selected.shape[1] > 0
        assert isinstance(feature_selector.fs_pipeline, Pipeline)
