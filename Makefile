PY_VERSION = python3
FILE = src



run:
	uv run $(PY_VERSION) -m src


install:
	uv sync


debug:
	$(PY_VERSION) -m pdp $(FILE)


clean:
	find . -type d \( -name "__pycache__" -o -name ".mypy_cache" -o -name ".pytest_cache" -o -name ".ruff_cache" -o -name ".ipynb_checkpoints" \) -prune -exec rm -rf {} +
	find . -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete
	rm -rf build dist *.egg-info .coverage htmlcov


lint:
	flake8 $(FILE)
	mypy $(FILE) --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs


lint-strict:
	flake8 $(FILE)
	mypy $(FILE) --strict