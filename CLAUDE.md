# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

- Install Dependencies: `poetry install`
- Run CLI: `poetry run abzu`
- Build/Generate abzu/baml_client code: `baml-cli generate`
- Test baml_src code: `baml-cli test`, `poetry run pytest tests/`
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
- BAML: Use for LLM-related code, regenerate client with `baml-cli generate`
- Whitespaces: leave no trailing whitespaces, use 4 spaces for indentation, leave no whitespace on blank lines
- Strings: Use double quotes for strings, use f-strings for string interpolation
- Docstrings: Use Numpy style for docstrings, include type hints in docstrings
- Comments: Use comments to explain complex code, avoid obvious comments
- Tests: Use pytest for testing, include type hints in test functions, use fixtures for setup/teardown
- Type hints: Use type hints for all function parameters and return types
- Type checking: Use mypy for type checking, run mypy before committing code
- Logging: Use logging for error handling, avoid print statements
- Documentation: Use Sphinx for documentation, include docstrings in all public functions/classes
- Code style: Follow PEP 8 for Python code style, use flake8 for linting

## Claude Code Style

- Command Line Interfaces - at the end of your coding tasks, please alter the 'abzu' CLI to accommodate the changes.
- PySpark - Limit the number of functions within scripts that control dataflow in Spark scripts. We prefer a more linear flow. This only applies to Spark code.
- Flake8 - fix flake8 errors without being asked and without my verification.
- Black - fix black errors without being asked and without my verification.
- Isort - fix isort errors without being asked and without my verification.
- Mypy - fix mypy errors without being asked and without my verification.
- Pre-commit - fix pre-commit errors without being asked and without my verification.
