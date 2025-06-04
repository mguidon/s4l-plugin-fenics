# Some defaults
SHELL 								= /bin/bash
.DEFAULT_GOAL 						:= help

makefile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
makefile_dir := $(patsubst %/,%,$(dir $(makefile_path)))

VENV_DIR = $(makefile_dir)/.venv
VENV_PYTHON = $(VENV_DIR)/bin/python

# DEVELOPMENT TOOLS -----------------------------

.PHONY: devenv
devenv: $(VENV_DIR) ## builds development environment

$(VENV_DIR):
	# creating virtual environment
	@python3 -m venv $@
	# updating package managers
	@$@/bin/pip --no-cache install --upgrade \
		pip \
		setuptools \
		wheel \
		pip-tools

# BUILD TOOLS -----------------------------

requirements: devenv  ## create the requirements for the fenics docker image
	cd ${makefile_dir}/docker/requirements &&\
	 ${VENV_PYTHON} -m piptools compile base.in

docker-image docker-image-nc: ## create the fenics docker image
	cd ${makefile_dir} &&\
	 docker build . $(if $(findstring -nc,$@),--no-cache,) --tag fenicsx_runner:latest -f docker/Dockerfile

docker-image-dev docker-image-dev-nc: ## create the fenics docker image without editable installs
	cd ${makefile_dir} &&\
	 docker build . $(if $(findstring -nc,$@),--no-cache,) --build-arg PIP_FLAGS= --tag fenicsx_runner:latest -f docker/Dockerfile

docker-image-prod: docker-image ## create the fenics docker image for production
	cd ${makefile_dir} &&\
	 docker build . --tag s4l-plugin-fenics-runner:prod -f docker/Dockerfile.prod

publish: docker-image-prod ## publish the fenics docker image to the registry
	docker tag s4l-plugin-fenics-runner:prod mguidon/s4l-plugin-fenics-runner:prod
	docker push mguidon/s4l-plugin-fenics-runner:prod


.PHONY: help

help: ## display all callable targets
	@sort $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
