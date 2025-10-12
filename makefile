.PHONY: lint
lint:
	uv run ruff format src
	uv run ruff check --fix src
	uv run mypy src --explicit-package-bases


.PHONY: clear-cache
clear-cache:
	rm -rf .triton_cache
	rm -rf .matplotlib
	rm -rf .ruff_cache
	rm -rf .mypy_cache