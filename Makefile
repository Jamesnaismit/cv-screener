.PHONY: test test-unit test-integration test-slow test-embedder test-app test-api help setup pipeline run

help:
	@echo "Available commands:"
	@echo ""
	@echo "Pipeline Commands:"
	@echo "  make setup             - Start database and wait for it to be ready"
	@echo "  make pipeline          - Run full pipeline: embed feed -> start API"
	@echo "  make run               - Quick start: setup + pipeline"
	@echo ""
	@echo "Test Commands:"
	@echo "  make test              - Run all tests"
	@echo "  make test-unit         - Run only unit tests"
	@echo "  make test-integration  - Run only integration tests"
	@echo "  make test-slow         - Run slow tests"
	@echo "  make test-embedder     - Run embedder tests"
	@echo "  make test-app          - Run API tests"

setup:
	@echo "=========================================="
	@echo "      STARTING DATABASE"
	@echo "=========================================="
	@docker compose up -d postgres
	@echo "Waiting for database to be ready..."
	@sleep 10
	@echo "Database is ready!"
	@echo ""

pipeline:
	@echo "=========================================="
	@echo "      RUNNING FULL PIPELINE"
	@echo "=========================================="
	@echo ""
	@echo "Step 1/2: Embedding CV feed..."
	@docker compose run --rm embedder python embedder.py
	@echo ""
	@echo "Step 2/2: Starting API..."
	@echo "------------------------------------------"
	@echo "API will be available at http://localhost:8000"
	@docker compose up api

run: setup pipeline

test:
	@echo "=========================================="
	@echo "       RUNNING ALL TESTS"
	@echo "=========================================="
	@echo ""
	@echo "Testing EMBEDDER..."
	@echo "------------------------------------------"
	@docker compose run --rm embedder pytest /app/tests -v
	@echo ""
	@echo "Testing API..."
	@echo "------------------------------------------"
	@docker compose run --rm api pytest /app/tests -v
	@echo ""
	@echo "=========================================="
	@echo "       ALL TESTS COMPLETED"
	@echo "=========================================="

test-unit:
	@echo "=========================================="
	@echo "       RUNNING UNIT TESTS"
	@echo "=========================================="
	@echo ""
	@echo "Testing EMBEDDER (unit)..."
	@echo "------------------------------------------"
	@docker compose run --rm embedder pytest /app/tests -m unit -v
	@echo ""
	@echo "Testing API (unit)..."
	@echo "------------------------------------------"
	@docker compose run --rm api pytest /app/tests -m unit -v
	@echo ""
	@echo "=========================================="
	@echo "       UNIT TESTS COMPLETED"
	@echo "=========================================="

test-integration:
	@echo "=========================================="
	@echo "    RUNNING INTEGRATION TESTS"
	@echo "=========================================="
	@echo ""
	@echo "Testing EMBEDDER (integration)..."
	@echo "------------------------------------------"
	@docker compose run --rm embedder pytest /app/tests -m integration -v
	@echo ""
	@echo "Testing API (integration)..."
	@echo "------------------------------------------"
	@docker compose run --rm api pytest /app/tests -m integration -v
	@echo ""
	@echo "=========================================="
	@echo "    INTEGRATION TESTS COMPLETED"
	@echo "=========================================="

test-slow:
	@echo "=========================================="
	@echo "       RUNNING SLOW TESTS"
	@echo "=========================================="
	@echo ""
	@echo "Testing EMBEDDER (slow)..."
	@echo "------------------------------------------"
	@docker compose run --rm embedder pytest /app/tests -m slow -v
	@echo ""
	@echo "Testing API (slow)..."
	@echo "------------------------------------------"
	@docker compose run --rm api pytest /app/tests -m slow -v
	@echo ""
	@echo "=========================================="
	@echo "       SLOW TESTS COMPLETED"
	@echo "=========================================="

test-embedder:
	@echo "=========================================="
	@echo "      TESTING EMBEDDER COMPONENT"
	@echo "=========================================="
	@docker compose run --rm embedder pytest /app/tests -v
	@echo "=========================================="
	@echo "      EMBEDDER TESTS COMPLETED"
	@echo "=========================================="

test-app:
	@echo "=========================================="
	@echo "        TESTING API COMPONENT"
	@echo "=========================================="
	@docker compose run --rm api pytest /app/tests -v
	@echo "=========================================="
	@echo "        API TESTS COMPLETED"
	@echo "=========================================="

test-api: test-app
