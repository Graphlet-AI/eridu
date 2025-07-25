# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

- Install Dependencies: `poetry install`
- Run CLI: `poetry run eridu`
- Test single: `poetry run pytest tests/path_to_test.py::test_name`
- Lint: `pre-commit run --all-files`, `poetry run flake8`
- Format: `poetry run black .`, `poetry run isort .`
- Type check: `poetry run mypy`

## Code Style

- KISS: KEEP IT SIMPLE STUPID. Do not over-engineer solutions. ESPECIALLY for Spark / PySpark.
- Line length: 100 characters
- Python version: 3.12
- Formatter: black with isort (profile=black)
- Types: Always use type annotations, warn on any return
- Imports: Use absolute imports, organize imports to be PEP compliant with isort (profile=black)
- Error handling: Use mdecific exception types with logging
- Naming: snake_case for variables/functions, CamelCase for classes
- BAML: Use for LLM-related code, regenerate client with `baml-cli generate`
- Whitespaces: leave no trailing whitespaces, use 4 spaces for indentation, leave no whitespace on blank lines
- Blank lines: Do not indent any blank lines in Python files. Indent should be 0 for these lines. Indent to 0 spaces when replacing a line with a blank line.
- Strings: Use double quotes for strings, use f-strings for string interpolation
- Docstrings: Use Numpy style for docstrings, include type hints in docstrings
- Comments: Use comments to explain complex code, avoid obvious comments
- Tests: Don't make a class to contain unit tests. Just write the tests in pytest style.
- Tests: Use pytest for testing, include type hints in test functions, use fixtures for setup/teardown. Don't create a class for each test file, use functions instead. Use fixtures to feed the tests data. Use `pytest.mark.parametrize` for parameterized tests.
- Type hints: Use type hints for all function parameters and return types
- Type checking: Use mypy for type checking, run mypy before committing code
- Logging: Use logging for error handling, avoid print statements
- Documentation: Use Sphinx for documentation, include docstrings in all public functions/classes
- Code style: Follow PEP 8 for Python code style, use flake8 for linting
- Always put spaces around operators and variables, e.g. `x = 1 + 2`, not `x=1+2`

## Claude Logic

- Find the root cause of an issue before figuring out a solution. Fix problems.
- Do not create workarounds for issues without asking. Always find the root cause of an issue and fix it.
- Command Line Interfaces - at the end of your coding tasks, please alter the 'eridu' CLI to accommodate the changes.
- Separate logic from the CLI - separate the logic under `eridu` and sub-modules from the command line interface (CLI) code in `eridu.cli`. The CLI should only handle input/output from/to the user and should not contain any business logic. ETL code should go in `eridu.etl`, training code should go in `eridu.train` and neither should have logic in the CLI. The CLI should only call the ETL, training and other code and handle input/output.
- Help strings - never put the default option values in the help strings. The help strings should only describe what the option does, not what the default value is. The default values are already documented in the `config.yml` file and will be printed via the `@click.command(context_settings={"show_default": True})` decorator of each Click command.
- PySpark - Limit the number of functions within scripts that control dataflow in Spark scripts. We prefer a more linear flow. This only applies to Spark code.
- Flake8 - fix flake8 errors without being asked and without my verification.
- Black - fix black errors without being asked and without my verification.
- Isort - fix isort errors without being asked and without my verification.
- Mypy - fix mypy errors without being asked and without my verification.
- Pre-commit - fix pre-commit errors without being asked and without my verification.
- New Modules - create a folder for a new module without being asked and without my verification.
- __init__.py - add these files to new module directories without being asked and without my verification.
- Edit Multiple Files at Once - if you need to edit multiple files for a single TODO operation, do so in a single step. Do not create multiple steps for the same task.
- Git - Keep commit messsages straightforward and to the point - do not put extraneous details, simply summarize the work performed. Do not put anything in commit messages other than a description of the code changes. Do not put "Generated with [Claude Code](https://claude.ai/code)" or anything else relating to Claude or Anthropic.
- Always use `context_settings={"show_default": True}` in Click commands to show default values in the help text.
- Don't ever put defaults in Click descriptions, only describe what the option does. The default options are set via the `@click.command(context_settings={"show_default": True})` decorator of each Click command or group.
- Always import modules at the top of the file, do not import them inside functions or classes.
- Ask questions before mitigating a simple problem with a complex fix.

## Alerts

- BEEP only ONCE when you are done with something and prompt me, the user in your UI. I need to hear that you're done because I do more than one thing at once. Use the command `echo -ne '\007'` to beep. Do not keep beeping multiple times or continuously. Just beep once when you are done with a task.
- Use the applescript-mcp server to send me a message when you are done with something. Say "Done with task X" where X is the task you are done with. Alternatively, use the command `osascript -e 'tell application "System Events" to display dialog "Done with task X"'` to send me a message. Send only ONE alert, not multiple alerts.
