import logging

import requests

logger = logging.getLogger(__name__)

# Define the JSON data you want to send in the request body
request_data = {
    "Body": {
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
}

body = request_data["Body"]

logger.info(f"Requesting survival probability for {body}")

# Define the API endpoint URL
api_url = "http://127.0.0.1:5000/inference"  # Replace with the actual URL of your API

# Send a POST request to the API
response = requests.post(api_url, json=request_data)

# Check the response status code
if response.status_code == 200:
    # Parse and print the response JSON data
    response_data = response.json()
    text = response_data["text"]
    logger.warning(f"Model response: {text}")
else:
    logger.error(f"Error: {response.status_code} | {response.text}")
