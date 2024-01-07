import sklearn
from project.packages.python_utils.load.object_injection import (
    _load_obj,
    load_object,
    load_estimator,
)


def test__load_obj():
    obj_path = "sklearn.preprocessing.MinMaxScaler"
    result = _load_obj(obj_path)()
    assert isinstance(result, sklearn.preprocessing.MinMaxScaler)


def test_load_object():
    result = load_object({"class": "sklearn.preprocessing.MinMaxScaler", "kwargs": {}})
    assert isinstance(result, sklearn.preprocessing.MinMaxScaler)


def test_estimator():
    result = load_estimator({"class": "sklearn.linear_model.LogisticRegression", "kwargs": {}})
    assert isinstance(result, sklearn.linear_model.LogisticRegression)
