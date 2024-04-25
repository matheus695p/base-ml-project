###
# This file is an automation tool that contains useful shell commands to be executed following a set of rules.
# To read more see https://makefiletutorial.com
###
.DEFAULT_GOAL     := help_minimal

ENV_NAME = titanic-dataset-test
PYTHON_VERSION = 3.10.13


#SHELL = /bin/zsh
CONDA_ACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

install:
	conda create --name $(ENV_NAME) -y python=$(PYTHON_VERSION)
	$(CONDA_ACTIVATE) $(ENV_NAME) && \
	pip install uv \
	uv pip install -r src/requirements.txt && \
	uv pip install pre-commit && \
	pre-commit install
	@echo "Environment '$(ENV_NAME)' successfully created"

_black: ## check code is black formatted
	@echo "Checking black"
	@black --check  .

_flake8: ## check flake8 passes
	@echo "Checking Flake8"
	@flake8 --config .flake8

_safety: ## check repo safety
	@echo "Checking Safety"
	@safety check


###################################
# API deployment
###################################
deploy-model-service-api-dev:
	flask --app src/project/apis/model_serving run --port=5000 --debug

deploy-model-service-api-prd:
	gunicorn --bind 0.0.0.0:80 'src.project.apis.model_serving:create_app()'

test-api:
	python src/project/apis/model_serving/ping_api.py


restart-folders:
	rm -rf data/02_intermediate
	rm -rf data/03_primary
	rm -rf data/04_feature
	rm -rf data/05_model_input
	rm -rf data/06_models
	rm -rf data/07_model_output
	rm -rf data/08_reporting
	mkdir data/02_intermediate
	mkdir data/03_primary
	mkdir data/04_feature
	mkdir data/05_model_input
	mkdir data/06_models
	mkdir data/07_model_output
	mkdir data/08_reporting
	touch data/02_intermediate/.gitkeep
	touch data/03_primary/.gitkept
	touch data/04_feature/.gitkept
	touch data/05_model_input/.gitkept
	touch data/06_models/.gitkept
	touch data/07_model_output/.gitkept
	touch data/08_reporting/.gitkept


###################################
## Docker build
###################################
docker-build:
	docker buildx install
	docker buildx build -t titanic:latest --platform linux/amd64 -f docker/Dockerfile.base .


commands:
	@kedro run --pipeline raw_layer
	@kedro run --pipeline intermediate_layer
	@kedro run --pipeline primary_layer
	@kedro run --pipeline feature_layer
	@kedro run --pipeline data_ingestion
	@kedro run --pipeline data_engineering
	@kedro run --pipeline data_science
	@kedro run --pipeline model_serving
