# Everything below runs offline unless the target says otherwise.
PYTHON ?= python
CONFIG ?= config/offline.yaml
QUESTION ?= Чему равен retention.hours по умолчанию?

.DEFAULT_GOAL := help
.PHONY: help install up down logs models index query eval bench test lint typecheck check dataset clean

help: ## Show this help
	@grep -hE '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "%-12s %s\n", $$1, $$2}'

install: ## Install the package with development dependencies
	$(PYTHON) -m pip install -e ".[dev]"

up: ## Start postgres, ollama and the API
	docker compose up -d --build

down: ## Stop everything and drop the volumes
	docker compose down -v

logs: ## Follow the API logs
	docker compose logs -f api

models: ## Pull the models the pgvector profile expects
	docker compose exec ollama ollama pull bge-m3
	docker compose exec ollama ollama pull qwen2.5:3b
	docker compose exec ollama ollama pull qwen2.5:7b-instruct

index: ## Index the evaluation corpus with CONFIG
	$(PYTHON) -m rag.cli --config $(CONFIG) index --corpus

query: ## Ask one question: make query QUESTION="..."
	$(PYTHON) -m rag.cli --config $(CONFIG) query "$(QUESTION)"

eval: ## Run the harness once with CONFIG
	$(PYTHON) -m eval.harness --config $(CONFIG)

bench: ## Run every sweep and write eval/results.json
	$(PYTHON) -m eval.harness --config $(CONFIG) --sweep --json eval/results.json

dataset: ## Regenerate the evaluation corpus from its seed
	$(PYTHON) -m eval.datasets.generate

test: ## Run the test suite
	$(PYTHON) -m pytest

lint: ## Run ruff
	$(PYTHON) -m ruff check .

typecheck: ## Run mypy in strict mode
	$(PYTHON) -m mypy --strict rag eval api tests

check: lint typecheck test ## Everything CI runs

clean: ## Remove caches and harness output
	rm -rf .mypy_cache .ruff_cache .pytest_cache htmlcov .coverage eval/results.json
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
