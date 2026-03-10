.PHONY: lint test test-e2e test-all format setup qdrant-up qdrant-down download-models

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/
	mypy --strict src/

format:
	ruff format src/ tests/

test:
	pytest tests/ -k "not e2e" -x -q

test-e2e:
	pytest tests/e2e/ -v

test-all: lint test test-e2e

qdrant-up:
	docker compose up -d

qdrant-down:
	docker compose down

download-models:
	bash scripts/download-models.sh

setup: qdrant-up download-models
	@echo ""
	@echo "Setup complete. Run 'rag init' to configure folders and LLM CLI,"
	@echo "then 'rag index' to index your documents."
