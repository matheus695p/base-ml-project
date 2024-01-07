![Build Status](https://www.repostatus.org/badges/latest/concept.svg)

# Base Machine Learning Project based on Titanic Dataset

## Overview

ML project using __Titanic Dataset__ as example.

## Challenge Requirements

### **Object-Oriented Programming (OOP)**


<div class="alert alert-info">
<b></b>

> The project's codebase adheres to Object-Oriented Programming (OOP) principles. It follows the scikit-learn Transformers and Estimators schema and object injection, facilitating modularity and extensibility. This design enables compatibility with any machine learning model that adheres to the scikit-learn schema. The project's codebase is thoughtfully organized to support seamless object injection.

</div>


### **Command-Line Interface (CLI)**

<div class="alert alert-info">
<b></b>

> The project includes a command-line interface (CLI) that allow users to interact with the code in a streamlined manner. This CLI is based and integrated with the Kedro pipeline framework, which segregates data engineering and data science responsibilities. Leveraging Kedro, the project can be scaled in a modular way, simplifying the process of testing and evaluating numerous machine learning models. In this example, 9 different machine learning models are hypertuned, trained and evaluated using the package and scaling in a modular way using kedro modular pipelines.

</div>

### **Testing and Code Coverage**

<div class="alert alert-info">
<b></b>

> The project undergoes testing, encompassing both unit and integration tests, resulting in a test coverage of over 90% in the package. Continuous Integration (CI) pipelines, orchestrated by GitHub Actions, are implemented to validate the package comprehensively. These pipelines incorporate integration tests to ensure the seamless functioning of the pipelines and perform code formatting checks to maintain code quality.

</div>


### **About MLOps**

#### __MLFlow Integration__

- The project integrates a local implementation of MLFlow. MLFlow facilitates experiment tracking and model registry management. This integration allows for the logging of experiments, tracking of model performance, saving artifacts, metrics and models. This facilitate the final model productionalization.

#### __API Package__

- The project incorporates an API package, which has an inference endpoint (that uses Flask and Kedro). This endpoint enables the deployment of the top-performing model as a production-ready API. Users can access the model's predictions through this API, making it readily available for integration into other applications.

#### __Docker Containerization__

- To ensure the utmost reproducibility and portability, the project supplies a Dockerfile. This Dockerfile permits the encapsulation of the entire project, ensuring that the environment and dependencies remain consistent across different environments. This feature proves particularly valuable in production environments, where deploying the best model is pivotal for informed decision-making.
