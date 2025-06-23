SHELL = /bin/bash
.DEFAULT_GOAL := help

# Makefile for building and serving MkDocs documentation locally

.PHONY: help venv install serve build deploy clean

help: ## help on rule's targets
	@awk --posix 'BEGIN {FS = ":.?## "} /^[[:alpha:][:space:]_-]+:.?## / {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

VENV_DIR=.venv
PYTHON=$(VENV_DIR)/bin/python
PIP=$(VENV_DIR)/bin/pip
MKDOCS=$(VENV_DIR)/bin/mkdocs

venv: ## Create a Python virtual environment in .venv
	python3 -m venv $(VENV_DIR)

install: venv ## Install requirements into the virtual environment
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

serve: ## Serve the documentation locally using mkdocs
	$(MKDOCS) serve -f mkdocs.yml

build: ## Build the static documentation site
	$(MKDOCS) build -f mkdocs.yml

deploy: install ## Deploy the documentation to the gh-pages branch on GitHub
	$(MKDOCS) gh-deploy --force -f mkdocs.yml

clean: ## Remove the virtual environment and build output
	rm -rf $(VENV_DIR) ../site site
