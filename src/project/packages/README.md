# Project Package Structure

This document provides a high-level overview of the project's structure, outlining the main directories and their contents.

## __modelling__

The `modelling` directory contains code related to machine learning models and their evaluation. The package is compatible with any ML model following the sklearn schema. The modelling package create a wrapper class that performs model hypertuning, testing and training using a cross validation strategy. The package also provides
model predictive capabilities to the model wrapper class. This is useful for interpreting and extracting more information about the models created. Also, solve any MPC problem associated.


- `evaluate`: Evaluate classification models with an mlflow metrics logger compatible structure.

- `feature_selection`: Sklearn transformers to perform feature selection, the projects only aplies model based feature selection.

  - `feature_selectors`: Contains implementations of various feature selectors used in the project, in this case only model based.

- `models`: This directory is dedicated to machine learning model implementations. It includes subdirectories such as `mlflow`, `model_predictive_control`, `supervised`, and `unsupervised`.

    - `mlflow`: Integration with MLFlow for model tracking. It includes metrics.py for tracking model metrics.

    - `model_predictive_control`: Scripts related to model predictive control, including constraints.py, explorer.py, and mpc.py.

    - `supervised`: Holds supervised machine learning models.

    - `unsupervised`: Contains unsupervised machine learning models.

- `reproducibility`: Scripts in this directory are designed to ensure reproducibility in the project.

- `transformers`: This subdirectory hosts data transformers used in the project.

## __preprocessing__

The `preprocessing` directory contains scripts for data preprocessing.

- `clean`: This directory houses scripts responsible for data cleaning, including clean_strings.py.

- `features`: Contains scripts for feature engineering, such as titanic.py.

- `transformers`: This subdirectory holds data transformers used in data preprocessing.

## __python_utils__

The `python_utils` directory houses custom Python utility modules used throughout the project.

- `definitions`: Contains definitions and constants used within the project.

- `load`: Includes data loading and manipulation utilities. It offers `object_injection.py` for loading and manipulating data objects.

- `typing`: This subdirectory provides custom typing definitions, such as tensors.py.

## __reporting__

The `reporting` directory is dedicated to reporting and visualization scripts.

- It includes scripts such as `html_report.py` and `html_report_utils.py` for generating reports and visualizations.
