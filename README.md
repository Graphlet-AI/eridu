# Eridu

> Fuzzy matching people and company names for entity resolution using representation learning

This project fine-tunes common sentence transformers to understand common differences between people and company names. This has the potential to outperform parsing based methods.

## TLDR

First go through <a href="#project-setup">Project Setup</a>, then run the CLI: <a href="#eridu-cli">`eridu --help`</a>

## `eridu` CLI

This project has a `eridu` CLI to run everything. It self describes.

```bash
eridu --help
```

NOTE! This README may get out of date, so please run `eridu --help`

```bash
Usage: eridu [OPTIONS] COMMAND [ARGS]...

  Eridu: Fuzzy matching people and company names for entity resolution using
  representation learning

Options:
  --version  Show the version and exit.
  --help     Show this message and exit.

Commands:
  download  Download and convert the labeled entity pairsa CSV file to...
  etl       ETL commands for data processing.
```

## Project Setup

This project uses Python 3.12 with `poetry` for package management.

### Create Python Environment

```bash
# Conda environment
conda create -n abzu python=3.12 -y
conda activate abzu

# Virtualenv
pthon -m venv venv
source venv/bin/activate
```

### Install `poetry` with `pipx`

```bash
# Install pipx on OS X
brew install pipx

# Install pipx on Ubuntu
sudo apt update
sudo apt install -y pipx

# Install poetry
pipx install poetry
```

### Install `poetry` with 'Official Installer'

```bash
# Try pipx if your firewall prevents this...
curl -sSL https://install.python-poetry.org | python3 -
```

### Install Python Dependencies

```bash
# Install dependencies
poetry install
```

### Install Pre-Commit Checks

```bash
# black, isort, flake8, mypy
pre-commit install
```
