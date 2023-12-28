import sys

sys.path.append("src/")
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request
from kedro.framework.project import configure_project
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

from project.api.model_serving.api_package.args import get_command_line_arguments
from project.api.model_serving.api_package.utils import cast_values_df

app = Flask(__name__)


def load_kedro_resources():
    args = get_command_line_arguments()
    environment = args.env
    metadata = bootstrap_project(Path.cwd())
    configure_project(metadata.package_name)
    with KedroSession.create(metadata.package_name, env=environment) as session:
        context = session.load_context()
        catalog = context.catalog

    return {
        "preprocessors": [
            catalog.load(f"{layer}_preprocessor") for layer in ["raw", "int", "prm", "feat"]
        ],
        "model": catalog.load("production_model"),
        "data_schemas": catalog.load("params:raw_transform")["schemas"],
    }


kedro_resources = load_kedro_resources()
preprocessors = kedro_resources["preprocessors"]
model = kedro_resources["model"]
data_schemas = kedro_resources["data_schemas"]


@app.route("/inference", methods=["POST"])
def inference():
    try:
        json_request = request.get_json()
        if "Body" not in json_request:
            return jsonify({"error": "Invalid request format"}), 400

        data = pd.DataFrame.from_dict(json_request["Body"], orient="index").T
        data = cast_values_df(data, data_schemas)
        name = data["Name"].iloc[0]

        for preprocessor in preprocessors:
            data = preprocessor.transform(data)

        prediction = model.predict_proba(data)[0][1]
        round_prediction = round(prediction * 100, 2)

        output = {
            "survive_probability": prediction,
            "text": f"{name} has a survival probability of {round_prediction} [%]",
        }
        return jsonify(output), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
