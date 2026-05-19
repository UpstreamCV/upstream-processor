.PHONY: install run test lint format typecheck check

install:
	uv sync

run:
	uv run python -m app.main

test:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check . --fix

typecheck:
	uv run mypy app

check: lint typecheck test
