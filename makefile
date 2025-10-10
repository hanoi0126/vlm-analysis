.PHONY: lint
lint:
	uv run ruff format src
	uv run ruff check --fix src
	uv run mypy src --explicit-package-bases