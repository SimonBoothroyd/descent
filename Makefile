PACKAGE_NAME := descent
PACKAGE_DIR  := $(PACKAGE_NAME)

CONDA_ENV_RUN := conda run --no-capture-output --name $(PACKAGE_NAME)

.PHONY: env lint format test test-examples docs-build docs-deploy docs-insiders

env:
	mamba create     --name $(PACKAGE_NAME)
	mamba env update --name $(PACKAGE_NAME) --file devtools/envs/base.yaml
	$(CONDA_ENV_RUN) pip install --no-deps -e .
	$(CONDA_ENV_RUN) pre-commit install || true

lint:
	$(CONDA_ENV_RUN) ruff check $(PACKAGE_DIR)

format:
	$(CONDA_ENV_RUN) ruff format                 $(PACKAGE_DIR)
	$(CONDA_ENV_RUN) ruff check --fix --select I $(PACKAGE_DIR)

test:
	$(CONDA_ENV_RUN) pytest -v --cov=$(PACKAGE_NAME) --cov-report=xml --color=yes $(PACKAGE_DIR)/tests/

docs-build:
	$(CONDA_ENV_RUN) mkdocs build

docs-deploy:
ifndef VERSION
	$(error VERSION is not set)
endif
	$(CONDA_ENV_RUN) mike deploy --push --update-aliases $(VERSION)

docs-insiders:
	$(CONDA_ENV_RUN) pip install git+https://$(INSIDER_DOCS_TOKEN)@github.com/SimonBoothroyd/mkdocstrings-python.git \
                    			 git+https://$(INSIDER_DOCS_TOKEN)@github.com/SimonBoothroyd/griffe-pydantic.git@fix-inheritence-static
