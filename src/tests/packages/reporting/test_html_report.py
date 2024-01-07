import os
import tempfile
import pytest
from project.packages.reporting.html_report import (
    _run_template,
    validate_notebook_error,
)

# Define test data
TEMPLATE_CONTENT = """
{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": "print('Hello, World!')"
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
"""


@pytest.fixture
def temp_template_file():
    # Create a temporary Jupyter notebook template file for testing
    with tempfile.NamedTemporaryFile(delete=False, suffix=".ipynb") as tmp_file:
        tmp_file.write(TEMPLATE_CONTENT.encode("utf-8"))
        tmp_file_path = tmp_file.name
    yield tmp_file_path
    os.remove(tmp_file_path)


def test_run_template(temp_template_file):
    # Test _run_template function
    notebook_path = temp_template_file
    namespace = "test_namespace"
    kernel = "python3"
    timeout = 10
    env = "test_env"

    notebook, error = _run_template(
        template_path=notebook_path,
        namespace=namespace,
        kernel=kernel,
        timeout=timeout,
        env=env,
    )

    assert isinstance(notebook, dict)
    assert error is False


def test_run_template_invalid_path():
    # Test _run_template with an invalid template path
    with pytest.raises(ValueError) as excinfo:
        _run_template(template_path="invalid_path.ipynb")
    assert "Template" in str(excinfo.value)
    assert "is not a file." in str(excinfo.value)


def test_validate_notebook_error_no_error():
    # Test validate_notebook_error function without an error
    error = None
    validate_notebook_error(error)
