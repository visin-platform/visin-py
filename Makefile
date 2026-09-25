# Every check CI runs, runnable here. `make check` is the one to run before a PR.
#
# Uses .venv when it exists (make install creates it), the PATH otherwise, which
# is how CI runs the same targets.

VENV ?= .venv
BIN := $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/,)
PYTHON ?= python3

.DEFAULT_GOAL := help
.PHONY: help install lint format typecheck test coverage contract build docs docs-serve check clean

help: ## List the targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  make %-11s %s\n", $$1, $$2}'

install: ## Create .venv with every dev tool, and install the git hooks
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -e '.[dev]'
	$(VENV)/bin/pre-commit install

lint: ## Lint and check formatting (ruff)
	$(BIN)ruff check .
	$(BIN)ruff format --check .

format: ## Fix formatting and the lint findings ruff can fix
	$(BIN)ruff format .
	$(BIN)ruff check --fix .

typecheck: ## Type-check the package (mypy, strict)
	$(BIN)mypy

test: ## Run the tests
	$(BIN)pytest

coverage: ## Run the tests with coverage; fails under the floor in pyproject.toml
	$(BIN)pytest --cov --cov-report=term-missing --cov-report=xml

contract: ## Check every request against Visin's API spec (needs VISIN_OPENAPI or ../visin)
	VISIN_REQUIRE_CONTRACT=1 $(BIN)pytest tests/contract

build: ## Build the sdist and wheel, and check their metadata
	rm -rf dist
	$(BIN)python -m build
	$(BIN)twine check --strict dist/*

docs: ## Build the documentation site, failing on any warning
	$(BIN)mkdocs build --strict

docs-serve: ## Serve the documentation with live reload
	$(BIN)mkdocs serve

check: lint typecheck coverage build docs ## Everything CI checks
	@echo "all checks passed"

clean: ## Remove build, test and docs output
	rm -rf dist site .coverage coverage.xml .pytest_cache .mypy_cache .ruff_cache
