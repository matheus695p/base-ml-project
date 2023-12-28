###
# This file is an automation tool that contains useful shell commands to be executed following a set of rules.
# To read more see https://makefiletutorial.com
###
.DEFAULT_GOAL     := help_minimal

ENV_NAME = ml_project
PYTHON_VERSION = 3.10.13


#SHELL = /bin/zsh
CONDA_ACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

install:
	conda create --name $(ENV_NAME) -y python=$(PYTHON_VERSION)
	$(CONDA_ACTIVATE) $(ENV_NAME) && \
	pip install -r requirements.txt && \
	pip install pre-commit && \
	pre-commit install
	@echo "Environment '$(ENV_NAME)' successfully created"
	make create-env-file

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
# run-api-dev:
# 	flask src/project/api/model_serving/app.py run --port=5000 --debug


run-api-dev:
	python src/project/api/model_serving/app.py

test-api-dev:
	python src/project/api/model_serving/ping_api.py
