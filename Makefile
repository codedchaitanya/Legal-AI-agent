# All commands use uv — never use pip directly in this project.
# Install uv: brew install uv

.PHONY: venv install test lint

venv:
	uv venv --python 3.12.7

install:
	uv pip install -r requirements.txt

install-dev:
	uv pip install -r requirements.txt pytest pytest-asyncio pytest-mock

test:
	source .venv/bin/activate && pytest tests/unit/ -v

test-all:
	source .venv/bin/activate && pytest tests/ -v
