# Model Inference API

This API provides endpoints for performing inference and querying using machine learning models. It is built using Flask and integrates with a Kedro-based machine learning project for model loading and execution.

## Table of Contents

- [Model Inference API](#model-inference-api)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [API Endpoints](#api-endpoints)
    - [Endpoint: /inference](#endpoint-inference)


## Installation

1. Clone the repository to your local machine:

```bash
git clone <repository_url>
```

```bash
pip install -r src/requirements.txt
```


2. Run the Flask application:


For dev:

```bash
deploy-model-service-api-dev
```

For prd:

```bash
deploy-model-service-api-prd
```


## API Endpoints
### Endpoint: /inference

Method: POST

Description: Perform inference using a machine learning model.

Request Format:

```json
{
    "Body": {
        "features": {
            "feature1": 0.5,
            "feature2": 0.8,
            ...
        }
    }
}
```

```json
{
    "prediction": 0.762,
    "text": "Inference for next shift is: 76.2 ",
}
```


```json
{
    "error": "Error message describing the issue."
}
```


Example Usage
Here's an example of how to use the API endpoints:

Perform Inference:

```bash
curl -X POST -H "Content-Type: application/json" -d '{
    "Body": {
        "features": {
            "feature1": 0.5,
            "feature2": 0.8
        }
    }
}' http://localhost:5000/inference

```
