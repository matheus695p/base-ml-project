# Project Pipelines Documentation

Descriptions for each of the pipelines and the CLI command to execute it.

## Data Engineering Pipelines

### Raw Layer (`raw_pipe`)
- **Description:** This pipeline is responsible for ingesting raw data from the titanic dataset. It includes data extraction, initial data cleaning, and basic preprocessing steps.
- **Purpose:** Validates data schema, creating a new data schema, validating data types, and indexing the DataFrame.
- **Pipeline Execution:**
```bash
kedro run --pipeline raw_layer
```

### Intermediate Layer (`intermediate_pipe`)
- **Description:** This pipeline builds upon the raw data by performing intermediate-level data processing. It includes additional data cleaning, transformation, and intermediate feature engineering.
- **Purpose:** Data preprocessing component that handles intermediate-level data. It prepares data by selecting relevant features, addressing data quality, handling outliers, and ensuring data consistency.
- **Pipeline Execution:**
```bash
kedro run --pipeline intermediate_layer
```

### Primary Layer (`primary_pipe`)
- **Description:** This pipeline focuses on primary data engineering and feature engineering. It includes advanced data transformation, feature creation, and feature selection.
- **Purpose:** Data preprocessing component that handles primary-level data. It focuses on filling missing values in specified categorical columns and applying text normalization. This class plays a crucial role in ensuring data quality and consistency in primary data processing pipelines.
- **Pipeline Execution:**
```bash
kedro run --pipeline primary_layer
```

### Feature Layer (`feature_pipe`)
- **Description:** This pipeline specifically handles feature engineering tasks. It includes feature generation, feature selection, and any custom feature-related operations.
- **Purpose:** Feature engineering component that enriches data by creating new features related to passenger tickets, cabin levels, and passenger information. It also performs one-hot encoding on specified columns to prepare data for modeling or analysis. This class is crucial for enhancing feature sets in feature engineering pipelines.
- **Pipeline Execution:**
```bash
kedro run --pipeline feature_layer
```

### Data Ingestion (`data_ingestion_pipes`)
- **Description:** This pipeline combines the raw, intermediate, and primary layers to handle overall data ingestion and processing. It encompasses the steps from raw data extraction to primary feature engineering.
- **Purpose:** It streamlines the data preparation process from various sources to a feature-rich dataset.
- **Pipeline Execution:**
```bash
kedro run --pipeline data_ingestion
```


### Data Engineering (`data_engineering_pipes`)
- **Description:** This pipeline combines all data engineering steps, including data ingestion and feature engineering. It integrates the data from various layers and prepares it for modeling.
- **Purpose:** This pipeline serves as an end-to-end data engineering process to transform raw data into a suitable format for modeling.
- **Pipeline Execution:**
```bash
kedro run --pipeline data_engineering
```


## Data Science Pipelines

### Model Predictive Control (`mpc_pipe`)
- **Description:** This pipeline is focused on model predictive control (MPC) tasks. It may include constraint handling, exploration, and MPC modeling.
- **Purpose:** It's designed for applications requiring predictive control models.
- **Pipeline Execution:**
```bash
kedro run --pipeline model_predictive_control_explorer
```


### Supervised Learning (`supervised`)
- **Description:** This pipeline handles supervised machine learning tasks. It includes data preprocessing, model hypertuning, training, evaluation, and selection.
- **Purpose:** This pipeline is for building predictive models based on labeled data. It hypertune and fit these models:

<div class="alert alert-info">
  <h3>Models Trained:</h3>
  <ul>
    <li>xgboost</li>
    <li>gradient_boosting_machines</li>
    <li>random_forest</li>
    <li>decision_tree</li>
    <li>logistic_regression</li>
    <li>neural_network</li>
    <li>knn</li>
    <li>quadratic_discriminant_analysis</li>
  </ul>
</div>

- **Pipeline Execution:**
```bash
kedro run --pipeline data_science
```

## Global Reporting Pipeline (`global_reporting_pipe`)
- **Description:** This pipeline is dedicated to global reporting and visualization. It generates reports and visualizations to provide insights into models trained.
- **Purpose:** It aids in understanding the project's overall performance and results.
- **Pipeline Execution:**
```bash
kedro run --pipeline global_reporting
```

## Model Serving Pipeline (`model_serving_pipe`)
- **Description:** This pipeline is responsible for deploying and serving machine learning models. It includes the model registry of the best model available in a automatized manner.
- **Pipeline Execution:**
```bash
kedro run --pipeline model_serving
```
