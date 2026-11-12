.PHONY: check lint test fmt
check: lint test
lint:
	ruff check . && ruff format --check . && mypy
test:
	pytest -m "not live" --cov=finagent
fmt:
	ruff format . && ruff check . --fix
