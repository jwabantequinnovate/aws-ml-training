.PHONY: help install test lint format clean build run-notebooks

help:
	@echo "Available commands:"
	@echo "  make install        - Install dependencies with Poetry"
	@echo "  make test          - Run all tests"
	@echo "  make test-unit     - Run unit tests only"
	@echo "  make test-int      - Run integration tests (requires AWS credentials)"
	@echo "  make lint          - Run linting checks"
	@echo "  make format        - Format code with Black"
	@echo "  make clean         - Remove cache and build artifacts"
	@echo "  make build         - Build package"
	@echo "  make coverage      - Generate coverage report"

install:
	@echo "Installing dependencies..."
	poetry install

test:
	@echo "Running all tests..."
	poetry run pytest tests/ -v --cov=src/ml_toolkit --cov-report=html --cov-report=term

test-unit:
	@echo "Running unit tests..."
	poetry run pytest tests/unit -v -m "not integration"

test-int:
	@echo "Running integration tests (requires AWS credentials)..."
	poetry run pytest tests/integration -v -m integration

lint:
	@echo "Running linting checks..."
	poetry run black --check src/ tests/
	poetry run ruff check src/ tests/
	poetry run mypy src/ || true

format:
	@echo "Formatting code..."
	poetry run black src/ tests/
	poetry run ruff check --fix src/ tests/

clean:
	@echo "Cleaning up..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/ htmlcov/ .coverage

build:
	@echo "Building package..."
	poetry build

coverage:
	@echo "Generating coverage report..."
	poetry run pytest tests/unit -v --cov=src/ml_toolkit --cov-report=html
	@echo "Coverage report generated in htmlcov/index.html"

run-notebooks:
	@echo "Starting Jupyter Lab..."
	poetry run jupyter lab

mlflow-ui:
	@echo "Starting MLflow UI..."
	@echo "Access at http://localhost:5000"
	poetry run mlflow ui

dvc-init:
	@echo "Initializing DVC..."
	poetry run dvc init
	@echo "✅ DVC initialized. Configure remote with: dvc remote add -d <name> <url>"
