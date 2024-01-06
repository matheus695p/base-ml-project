import sys
import logging

sys.path.append("src/")

import pandas as pd
from flask import Flask, jsonify, request
from flask.typing import ResponseReturnValue
from flask_caching import Cache
from pathlib import Path
from kedro.framework.project import configure_project
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from project.apis.model_serving.api_package.constants import ENV as environment
from project.apis.model_serving.api_package.constants import APP_CONFIG
from .api_package.utils import cast_values_df

logger = logging.getLogger(__name__)

app = Flask(__name__)

logger = logging.getLogger(__name__)


def load_kedro_resources():
    """
    Load Kedro project resources and return a dictionary of loaded models.

    This function loads resources from a Kedro project, specifically model artifacts,
    using the Kedro framework. It expects command-line arguments to specify the
    environment (e.g., 'local' or 'production'). The loaded models are then
    returned as a dictionary.

    Returns:
        dict: A dictionary containing loaded model artifacts.

    Raises:
        Any exceptions raised by Kedro functions during project bootstrap and loading.

    Example:
        >>> resources = load_kedro_resources()
        >>> models = resources["models"]
        >>> # Access and use the loaded models as needed.

    Note:
        This function assumes the presence of Kedro project configuration and
        follows Kedro conventions for resource loading. Make sure you have a valid
        Kedro project structure before using this function.
    """
    metadata = bootstrap_project(Path.cwd())
    configure_project(metadata.package_name)
    with KedroSession.create(metadata.package_name, env=environment) as session:
        context = session.load_context()
        catalog = context.catalog

    return {
        "preprocessors": [
            catalog.load(f"{layer}_preprocessor")
            for layer in ["raw", "int", "prm", "feat", "cluster"]
        ],
        "model": catalog.load("production_model"),
        "data_schemas": catalog.load("params:raw_transform")["schemas"],
    }


kedro_resources = load_kedro_resources()
preprocessors = kedro_resources["preprocessors"]
model = kedro_resources["model"]
data_schemas = kedro_resources["data_schemas"]


def create_app() -> Flask:
    """Model inference API.

    The `create_app` function creates a Flask application with routes for
    requesting model predictions.

    Args:
        None

    Returns:
      The function `create_app` returns a Flask application object.
    """
    # create flask app
    app = Flask(__name__)
    app.config.from_mapping(APP_CONFIG)

    # If necessary use cache
    cache = Cache(app)  # noqa

    # TODO: decorator performing auth is necessary before expose the API endpoints
    @app.route("/inference", methods=["POST"])
    def inference() -> ResponseReturnValue:
        """
        Handle a POST request to perform inference using a machine learning model.

        This endpoint expects a JSON request containing a 'Body' field with the
        'namespace' and 'features' data necessary for inference. It retrieves the
        corresponding model based on the provided 'namespace', performs inference on
        the input features, and returns the prediction as a JSON response.

        Returns:
            Response: A JSON response containing the inference result or an error message.

        Raises:
            400 Bad Request: If the JSON request format is invalid or missing 'Body'.
            500 Internal Server Error: If an unexpected error occurs during inference.

        Example JSON Request:
            {
                "Body": {
                    "namespace": "model_namespace",
                    "features": {
                        "feature1": 0.5,
                        "feature2": 0.8,
                        ...
                    }
                }
            }

        Example JSON Response (Successful Inference):
            {
                "prediction": 0.762,
                "text": "Matheus Pinto has a survival probability of 76.2 [%]",
            }

        Example JSON Response (Error):
            {
                "error": "Error message describing the issue."
            }
        """
        try:
            json_request = request.get_json()
            if "Body" not in json_request:
                return jsonify({"error": "Invalid request format"}), 400

            data = pd.DataFrame.from_dict(json_request["Body"]["features"], orient="index").T

            data = cast_values_df(data, data_schemas)
            name = data["Name"].iloc[0]

            for preprocessor in preprocessors:
                data = preprocessor.transform(data)

            prediction = model.predict_proba(data)[0][1]
            round_prediction = round(prediction * 100, 3)

            output = {
                "prediction": round_prediction,
                "text": f"{name} has a survival probability of {round_prediction} [%]",
            }
            return jsonify(output), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/")
    def hello() -> ResponseReturnValue:
        return "Hello, World!"

    return app
