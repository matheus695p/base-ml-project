import os
from project.packages.reporting.html_report_utils import set_env_var


def test_set_env_var():
    # Store the current value of the environment variable (if it exists)
    current_value = os.environ.get("MY_ENV_VAR")

    # Use the context manager to set the environment variable
    with set_env_var("MY_ENV_VAR", "new_value"):
        assert os.environ["MY_ENV_VAR"] == "new_value"

    # The environment variable should be restored to its original value
    if current_value is None:
        assert "MY_ENV_VAR" not in os.environ
    else:
        assert os.environ["MY_ENV_VAR"] == current_value
