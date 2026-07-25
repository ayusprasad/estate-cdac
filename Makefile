.PHONY: help install dev-install lint format type-check test test-cov migrate migrate-create run setup-dirs clean

PYTHON := python
UVICORN_CMD := uvicorn app.main:app

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies
	pip install -r requirements.txt

dev-install: ## Install all dependencies including dev
	pip install -r requirements-dev.txt
	lint: ## Run ruff linter
	ruff check app/ config/ tests/

format: ## Auto-format code with ruff
	ruff format app/ config/ tests/

type-check: ## Run mypy type checker
	mypy app/ config/

test: ## Run tests
	pytest tests/

test-cov: ## Run tests with coverage
	pytest tests/ --cov=app --cov-report=html

migrate: ## Apply all pending migrations
	alembic upgrade head

migrate-create: ## Create a new migration (usage: make migrate-create MSG="description")
	alembic revision --autogenerate -m "$(MSG)"

run: ## Start the FastAPI dev server
	$(UVICORN_CMD) --reload --host 0.0.0.0 --port 8000

setup-dirs: ## Create required runtime directories
	$(PYTHON) scripts/setup_dirs.py

clean: ## Remove Python cache files
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -exec rm -rf {} +
	rm -rf .pytest_cache htmlcov .mypy_cache
