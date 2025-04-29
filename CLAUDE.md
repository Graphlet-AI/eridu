# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands
- Test: `poetry run pytest tests/`
- Test single: `poetry run pytest tests/path_to_test.py::test_name`
- Lint: `pre-commit run --all-files`, `poetry run flake8`
- Format: `poetry run black .`, `poetry run isort .`
- Type check: `poetry run mypy`

## Code Style
- Line length: 100 characters
- Python version: 3.12
- Formatter: black with isort (profile=black)
- Types: Always use type annotations, warn on any return
- Imports: Use absolute imports, organize imports to be PEP compliant with isort (profile=black)
- Error handling: Use mdecific exception types with logging
- Naming: snake_case for variables/functions, CamelCase for classes

## Claude Code Style

- Specificity - try not to get off track. Keep your changes central to the request, keep line-count moderate for a single change.
- PySpark - Limit the number of functions within scripts that control dataflow. We prefer a more linear flow.
