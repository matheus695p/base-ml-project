![Build Status](https://www.repostatus.org/badges/latest/concept.svg)

# Machine Learning Project based on Titanic Dataset

Please clic on each of the links provided, they all contains information related to the project.

## Overview


ML project using __Titanic Dataset__ as example.

[Train Dataset](https://github.com/matheus695p/titanic-dataset/blob/main/data/01_raw/titanic_train.csv)

[Test Dataset](https://github.com/matheus695p/titanic-dataset/blob/main/data/01_raw/titanic_test.csv)

## Get started

[Install dependencies and get started](https://github.com/matheus695p/titanic-dataset/blob/main/docs/documentation/01_get_started.md)

[Rules and guidelines](https://github.com/matheus695p/titanic-dataset/blob/main/docs/documentation/02_rules_and_guidelines.md)

[Commit convention](https://gist.github.com/qoomon/5dfcdf8eec66a051ecd85625518cfd13)

## Challenge Requirements

### **Object-Oriented Programming (OOP)**


<div class="alert alert-info">
<b></b>

> The project's codebase adheres to Object-Oriented Programming (OOP) principles. It follows the scikit-learn Transformers and Estimators schema and object injection, facilitating modularity and extensibility. This design enables compatibility with any machine learning model that adheres to the scikit-learn schema. The project's codebase is thoughtfully organized to support seamless object injection.

</div>

[Code Packages](https://github.com/matheus695p/titanic-dataset/tree/main/src/project/packages/README.md)


### **Command-Line Interface (CLI)**

<div class="alert alert-info">
<b></b>

> The project includes a command-line interface (CLI) that allow users to interact with the code in a streamlined manner. This CLI is based and integrated with the Kedro pipeline framework, which segregates data engineering and data science responsibilities. Leveraging Kedro, the project can be scaled in a modular way, simplifying the process of testing and evaluating numerous machine learning models. In this example, 9 different machine learning models are hypertuned, trained and evaluated using the package and scaling in a modular way using kedro modular pipelines.

</div>

[Pipelines Structure and Code](https://github.com/matheus695p/titanic-dataset/blob/main/src/project/pipelines/README.md)


### **Testing and Code Coverage**

<div class="alert alert-info">
<b></b>

> The project undergoes testing, encompassing both unit and integration tests, resulting in a test coverage of over 90% in the package. Continuous Integration (CI) pipelines, orchestrated by GitHub Actions, are implemented to validate the package comprehensively. These pipelines incorporate integration tests to ensure the seamless functioning of the pipelines and perform code formatting checks to maintain code quality.

</div>

[Test code](https://github.com/matheus695p/titanic-dataset/blob/main/src/tests/README.md)



### **About MLOps**

#### __MLFlow Integration__

- The project integrates a local implementation of MLFlow. MLFlow facilitates experiment tracking and model registry management. This integration allows for the logging of experiments, tracking of model performance, saving artifacts, metrics and models. This facilitate the final model productionalization.

#### __API Package__

- The project incorporates an API package, which has an inference endpoint (that uses Flask and Kedro). This endpoint enables the deployment of the top-performing model as a production-ready API. Users can access the model's predictions through this API, making it readily available for integration into other applications.

[API package](https://github.com/matheus695p/titanic-dataset/blob/main/src/project/apis/README.md)


#### __Docker Containerization__

- To ensure the utmost reproducibility and portability, the project supplies a Dockerfile. This Dockerfile permits the encapsulation of the entire project, ensuring that the environment and dependencies remain consistent across different environments. This feature proves particularly valuable in production environments, where deploying the best model is pivotal for informed decision-making.

[Docker Image](https://github.com/matheus695p/titanic-dataset/blob/main/docker/README.md)


## Continuous deployment

A continuous deployment pipeline is provided in order to create and push container to any container registry cloud service.

[CD Pipeline](https://github.com/matheus695p/titanic-dataset/blob/main/.github/workflows/continuous-deployment.yml)



## Project usage

### Get started notebook.

This notebook serves as a comprehensive guide to understanding and effectively utilizing the classes within the project package. It aims to provide you with an overview of how these classes works, their core functionalities, and practical insights on how to incorporate them into your project. Below, we'll expand on the key points covered in this notebook:

[Get started notebook](https://github.com/matheus695p/titanic-dataset/blob/main/notebooks/documentation/00_GetStarted.ipynb)


## Project CLI.

Even if notebooks are good, they are not made for production, so I strongly recommend to use the project CLI instead. Here are the main commands to run this


```bash

# Run the hole project
kedro run --pipeline data_engineering && kedro run --pipeline data_science

# Raw layer data preprocessing
kedro run --pipeline raw_layer

# Intermediate layer data preprocessing
kedro run --pipeline intermediate_layer

# Primary layer data preprocessing
kedro run --pipeline primary_layer

# Feature layer data preprocessing
kedro run --pipeline feature_layer

# Data ingestion process, it runs raw + intermediate + primary  layers
kedro run --pipeline data_ingestion

# Runs the raw + intermediate + primary + feature layers. This ended up beeing the feature store for training
kedro run --pipeline data_engineering

# Runs all modelling and reporting pipelines for data science, it hypertune, train, evaluate and create htmls reports for theses processes
kedro run --pipeline data_science

# Compile all results and output what are the best models for this specif problem,
kedro run --pipeline global_reporting

# It runs a model predictive control class to understand model behavior
kedro run --pipeline model_predictive_control_explorer

# Register the best model from the previous pipeliens into a MLflow model in a automatized way
kedro run --pipeline model_serving

```


## Project Visualization.

In order to visualize each of the package pipelines, we can run the following command to generate an end to end visualization of the models.


```bash
kedro viz run
```

![DE pipeline visualization](https://github.com/matheus695p/titanic-dataset/blob/main/docs/images/de.png)

![DS pipeline visualization](https://github.com/matheus695p/titanic-dataset/blob/main/docs/images/ds.png)

![Code Exploration](https://github.com/matheus695p/titanic-dataset/blob/main/docs/images/exploration.png)




## Project Results (PLEASE READ !)

In these readme you can see my conclusion for running the project and get results.

Conclusions are based on metric selection, model analysis, hypertuning and model explainability.


[Project results](https://github.com/matheus695p/titanic-dataset/blob/main/docs/documentation/03_results.md)



## MLFlow server.

In order to visualize all artifacts, metrics and parameters saved during the CLI execution, after running the pipelines, you can run kedro mlflow ui in order to see all which is logged into MLFlow.


```bash
kedro mlfow ui
```

A local mlflow server will run on the port 3000.
You can access it here [Local mlfow server](http://127.0.0.1:3000)


## Project testing.


I focused on testing the project in 3 aspects:

1. `Code quality checks`
2. `Unit testing`
3. `Integration testing`


[Code quality workflow](https://github.com/matheus695p/titanic-dataset/blob/main/.github/workflows/code-formatting.yml)

[Unit testing](https://github.com/matheus695p/titanic-dataset/blob/main/.github/workflows/unit-tests.yml)

[Integration testing](https://github.com/matheus695p/titanic-dataset/blob/main/.github/workflows/integration-tests.yml)

Each of these workflows ensure that on each PR or push to the principal main/develop branches, **code is tested end-to-end**

## Testing coverage report

You can take a look at the full project coverage report at one of the workflows iteration, here it is an example of these

[Coverage report](https://github.com/matheus695p/titanic-dataset/actions/runs/7442957317/job/20247138409)

The report says 52 % coverage, but these contains pipelines and APIs code that are not in the project scope, I compute the coverage for only the base packages and is over 90 % of the coverage. I will continue adding more testing for increasing coverage, but main functionalities are fully tested.

Here is the coverage report only for the base packages:

![coverage_report](https://github.com/matheus695p/titanic-dataset/blob/main/docs/images/coverage_report.png)


## Project deployment.

As mentioned before the best model can be productionalized through an inference endpoint, in order to test these code and ping the API, you can run the following commands

Explose the API in the port 5000 of the terminal, before running these command you need to run before project pipelines.

```bash
flask --app src/project/apis/model_serving run --port=5000 --debug
```

Test the API.

```bash
python src/project/apis/model_serving/ping_api.py
```


As this is a single endpoint, when considering deploying these applications into production, there are several architectural options to choose from, each with its own set of advantages and disadvantages.


### Serverless deployment

Using azure functions, aws lambda functions or google cloud function --> API Gateway


**Pros:**
- Cost-effective as you only pay for actual usage.
- Auto-scaling and automatic resource management.
- Low operational overhead as the cloud provider handles infrastructure.
- Well-suited for event-driven applications and microservices.

**Cons:**
- Limited control over infrastructure, which may be a limitation for specific requirements.
- Cold start latency can impact response times for some functions.
- Debugging and monitoring can be more challenging in a serverless environment.


### Load Balancer + Container services

**Pros:**
- Provides control over the underlying infrastructure.
- Suitable for applications with specific hardware requirements.
- Can be cost-effective for steady-state workloads.
- Greater flexibility in configuring load balancing algorithms.

**Cons:**
- Increased operational complexity compared to serverless or containerized approaches.
- Limited scalability during traffic spikes without proper automation.


We can discuss in person which is the most suitable approach regardless of your current cloud services.

__In the way this project is designed is transparent to watever cloud service you are using or interested.__
