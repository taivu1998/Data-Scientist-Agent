.PHONY: install install-dev clean evaluate inference test test-unit test-integration lint format setup help

# Default target
help:
	@echo "Auto-Analyst Makefile Commands"
	@echo "=============================="
	@echo ""
	@echo "Setup:"
	@echo "  make install      - Install production dependencies"
	@echo "  make install-dev  - Install with dev dependencies (pytest, black, etc.)"
	@echo "  make setup        - Full setup (install + create directories)"
	@echo ""
	@echo "Running:"
	@echo "  make evaluate     - Run benchmark evaluation against Golden Set"
	@echo "  make inference    - Run single query inference (set QUERY and CSV_PATH)"
	@echo ""
	@echo "Testing:"
	@echo "  make test         - Run all tests"
	@echo "  make test-unit    - Run unit tests only (no API keys needed)"
	@echo "  make test-integration - Run integration tests (requires API keys)"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint         - Run linter (ruff)"
	@echo "  make format       - Format code (black)"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean        - Remove cache files and logs"
	@echo ""
	@echo "Examples:"
	@echo "  make inference QUERY=\"Show data shape\" CSV_PATH=\"data/csvs/salaries.csv\""

# Installation
install:
	pip install -e .
	pip install -r requirements.txt

install-dev:
	pip install -e ".[dev]"
	pip install -r requirements.txt

# Full setup
setup: install
	mkdir -p data/csvs logs
	@echo "Setup complete! Don't forget to:"
	@echo "  1. Copy .env.example to .env"
	@echo "  2. Add your ANTHROPIC_API_KEY and E2B_API_KEY"

# Clean up
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf logs/*.log logs/*.json 2>/dev/null || true
	rm -rf build/ dist/ 2>/dev/null || true
	@echo "Cleaned up cache and log files"

# Testing
test:
	pytest tests/ -v

test-unit:
	pytest tests/ -v --ignore=tests/test_integration.py

test-integration:
	pytest tests/test_integration.py -v -m integration

# Code quality
lint:
	ruff check src/ scripts/ tests/

format:
	black src/ scripts/ tests/

# Run benchmark evaluation
evaluate:
	python scripts/evaluate.py --config configs/default.yaml

# Run single inference
# Usage: make inference QUERY="your query" CSV_PATH="path/to/csv"
QUERY ?= "Show me basic statistics about the data"
CSV_PATH ?= "data/csvs/salaries.csv"

inference:
	python scripts/inference.py --config configs/default.yaml --query "$(QUERY)" --csv_path "$(CSV_PATH)"

# Ablation study: run without visual critic
evaluate-no-critic:
	python scripts/evaluate.py --config configs/default.yaml --enable_visual_critic false
