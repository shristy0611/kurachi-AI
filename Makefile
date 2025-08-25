.PHONY: help smoke test test-unit test-full compile-deps upgrade-deps install-dev spacy-models ci-validate clean lint format pre-commit-install

# Default target
help:
	@echo "🔧 Validation System Hardening - Development Commands"
	@echo ""
	@echo "Testing:"
	@echo "  make smoke          Run fast smoke tests (< 2 min)"
	@echo "  make test           Run all tests excluding slow/NLP"
	@echo "  make test-unit      Run unit tests only"
	@echo "  make test-full      Run complete test suite with NLP"
	@echo ""
	@echo "Dependencies:"
	@echo "  make compile-deps   Compile requirements.in → requirements.txt"
	@echo "  make upgrade-deps   Upgrade and compile dependencies"
	@echo "  make install-dev    Install development environment"
	@echo "  make spacy-models   Download required spaCy models"
	@echo ""
	@echo "Quality:"
	@echo "  make ci-validate    Run full CI validation locally"
	@echo "  make lint           Run linting checks"
	@echo "  make format         Format code with black/isort"
	@echo "  make pre-commit-install  Install pre-commit hooks"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean          Clean cache and temp files"

# Fast smoke tests - critical path validation
smoke:
	@echo "🚀 Running smoke tests (fast gate)..."
	@PYTHONPATH=. time pytest -m smoke -q --maxfail=1 --tb=short
	@echo "✅ Smoke tests completed"

# Standard test suite (no NLP, no slow tests)
test:
	@echo "🧪 Running standard test suite..."
	@PYTHONPATH=. pytest -m "not nlp and not slow" -v --tb=short
	@echo "✅ Standard tests completed"

# Unit tests only
test-unit:
	@echo "🔬 Running unit tests..."
	@PYTHONPATH=. pytest tests/unit/ -v --tb=short
	@echo "✅ Unit tests completed"

# Full test suite including NLP
test-full:
	@echo "🌟 Running full test suite (including NLP)..."
	@PYTHONPATH=. pytest -v --tb=short --cov=. --cov-report=term-missing
	@echo "✅ Full test suite completed"

# Compile requirements with hashes
compile-deps:
	@echo "📦 Compiling dependencies..."
	@pip-compile --generate-hashes --resolver=backtracking -o requirements.txt requirements.in
	@pip-compile --generate-hashes --resolver=backtracking -o requirements-dev.txt requirements-dev.in
	@echo "✅ Dependencies compiled with hashes"

# Upgrade and compile dependencies
upgrade-deps:
	@echo "⬆️  Upgrading dependencies..."
	@pip-compile --upgrade --generate-hashes --resolver=backtracking -o requirements.txt requirements.in
	@pip-compile --upgrade --generate-hashes --resolver=backtracking -o requirements-dev.txt requirements-dev.in
	@echo "✅ Dependencies upgraded and compiled"

# Install development environment
install-dev:
	@echo "🛠️  Setting up development environment..."
	@python -m pip install --upgrade pip
	@pip install pip-tools
	@pip-sync requirements.txt requirements-dev.txt
	@echo "✅ Development environment ready"

# Download spaCy models
spacy-models:
	@echo "🧠 Downloading spaCy models..."
	@python -m spacy download en_core_web_sm
	@python -m spacy download ja_core_news_sm || echo "⚠️  Japanese model optional"
	@echo "✅ spaCy models downloaded"

# Full CI validation locally
ci-validate:
	@echo "🔍 Running full CI validation locally..."
	@echo "1️⃣ Smoke tests..."
	@make smoke
	@echo "2️⃣ Unit tests..."
	@make test
	@echo "3️⃣ Dependency check..."
	@pip-compile --dry-run --check-hashes requirements.in
	@pip-compile --dry-run --check-hashes requirements-dev.in
	@echo "4️⃣ Linting..."
	@make lint
	@echo "✅ CI validation completed successfully"

# Linting
lint:
	@echo "🔍 Running linting checks..."
	@flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	@flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
	@echo "✅ Linting completed"

# Code formatting
format:
	@echo "🎨 Formatting code..."
	@black . --line-length=127
	@isort . --profile=black --line-length=127
	@echo "✅ Code formatted"

# Install pre-commit hooks
pre-commit-install:
	@echo "🪝 Installing pre-commit hooks..."
	@pip install pre-commit
	@pre-commit install
	@echo "✅ Pre-commit hooks installed"

# Clean cache and temporary files
clean:
	@echo "🧹 Cleaning cache and temporary files..."
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -delete
	@find . -type d -name "*.egg-info" -exec rm -rf {} +
	@rm -rf .pytest_cache/
	@rm -rf .coverage
	@rm -rf htmlcov/
	@rm -rf dist/
	@rm -rf build/
	@echo "✅ Cleanup completed"

# Performance timing for smoke tests
smoke-timing:
	@echo "⏱️  Measuring smoke test performance..."
	@PYTHONPATH=. time -p pytest -m smoke -q --maxfail=1 2>&1 | tee smoke_timing.log
	@echo "📊 Timing results saved to smoke_timing.log"