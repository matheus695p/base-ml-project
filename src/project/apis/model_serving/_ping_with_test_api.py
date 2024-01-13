import logging
import pytest
import requests

logger = logging.getLogger(__name__)

API_URL = "http://127.0.0.1:5000/inference"

request_data = {
    "Body": {
        "features": {
            "PassengerId": 892,
            "Pclass": 3,
            "Name": "Matheus Pinto Arratia",
            "Sex": "male",
            "Age": 28.9,
            "SibSp": 0,
            "Parch": 0,
            "Ticket": "40101",
            "Fare": 7.8292,
            "Cabin": None,
            "Embarked": "Q",
        }
    },
}

response = requests.post(API_URL, json=request_data)
response_data = response.json()
text = response_data["text"]
logger.info(f"API response: {text}")


@pytest.fixture
def api_data():
    request_data = {
        "Body": {
            "features": {
                "PassengerId": 892,
                "Pclass": 3,
                "Name": "Matheus Pinto Arratia",
                "Sex": "male",
                "Age": 28.9,
                "SibSp": 0,
                "Parch": 0,
                "Ticket": "40101",
                "Fare": 7.8292,
                "Cabin": None,
                "Embarked": "Q",
            }
        },
    }

    return request_data


def test_api_response(api_data):
    response = requests.post(API_URL, json=api_data)
    assert (
        response.status_code == 200
    ), f"API did not return a successful response (Status code: {response.status_code})"

    response_data = response.json()
    assert "text" in response_data.keys(), "Response JSON does not contain 'text' key"
    assert "prediction" in response_data.keys(), "Response JSON does not contain 'prediction' key"

    text = response_data["text"]
    prediction = response_data["prediction"]

    assert text is not None, "Text in the response is None"
    assert prediction is not None, "Prediction in the response is None"
