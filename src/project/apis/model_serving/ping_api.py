import requests

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
print(text)
