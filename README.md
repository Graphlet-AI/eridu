# Eridu

> Deep fuzzy matching people and company names for multilingual entity resolution using representation learning... that incorporates a deep understanding of people and company names and works _much better_ than string distance methods.

<center><img src="images/Ancient-Eridu-Tell-Abu-Shahrain.jpg" alt="Google Maps overhead view of Tell Abu Shahrein - Ancient Eridu" /></center>

## About Ancient Eridu

Ancient [Eridu](https://en.wikipedia.org/wiki/Eridu) (modern [Tell Abu Shahrain in Southern Iraq](https://maps.app.goo.gl/xXACdHh1Ppmx7NAf6)) was the world's first city, by Sumerian tradition, with a history spanning 7,000 years. It was the first place where "kingship descended from heaven" to lead farmers to build and operate the first complex irrigation network that enabled intensive agriculture sufficient to support the first true urban population.

## Project Overview

This project is a deep fuzzy matching system for entity resolution using representation learning. It is designed to match people and company names across languages and character sets, using a pre-trained text embedding model from HuggingFace that we fine-tune using contrastive learning on 2 million labeled pairs of person and company names from the [Open Sanctions Matcher training data](https://www.opensanctions.org/docs/pairs/). The project includes a command-line interface (CLI) utility for training the model and comparing pairs of names using cosine similarity.

Matching people and company names is an intractable problem using traditional parsing based methods: there is too much variation across cultures and jurisdictions to solve the problem by humans programming. Machine learning is used in problems like this one of cultural relevance, where programming a solution approaches infinite complexity, to automatically write a program. Since 2008 there has been an explosion of deep learning methods that automate feature engineering via representation learning methods including such as text embeddings. This project loads the pre-trained [paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) paraphrase model from HuggingFace and fine-tunes it for the name matching task using contrastive learning on more than 2 million labeled pairs of matching and non-matching (just as important) person and company names from the [Open Sanctions Matcher training data](https://www.opensanctions.org/docs/pairs/) to create a deep fuzzy matching system for entity resolution.

## Getting Started

First go through <a href="#project-setup">Project Setup</a>, then run the CLI: <a href="#eridu-cli">`eridu --help`</a>

## `eridu` CLI

The interface to this work is a command-line (CLI) utility `eridu` that trains a model and a utility that compares a pair of names using our fine-tuned embedding and a metric called cosine similarity that incorporates a deep understanding of people and company names and works _much better_ than string distance methods. This works across languages and charactersets  The distance returned is a number between 0 and 1, where 0 means the names are identical and 1 means they are completely different. The CLI utility is called `eridu` and it has three subcommands: `download`, `train` and `compare`. More will be added in the near future, so check the documentation for updates: `eridu --help`.

This project has a `eridu` CLI to run everything. It self describes.

```bash
eridu --help
```

NOTE! This README may get out of date, so please run `eridu --help` for the latest API.

```bash
Usage: eridu [OPTIONS] COMMAND [ARGS]...

  Eridu: Fuzzy matching people and company names for entity resolution using
  representation learning

Options:
  --version  Show the version and exit.
  --help     Show this message and exit.

Commands:
  download  Download and convert the labeled entity pairs CSV file to...
  etl       ETL commands for data processing.
  train     Fine-tune a sentence transformer (SBERT) model for entity...
```

To train the model, run the commands in the order they appear in the documentation. Default arguments will probably work.

## Project Setup

This project uses Python 3.12 with `poetry` for package management.

### Create Python Environment

You can use any Python environment manager you like. Here are some examples:

```bash
# Conda environment
conda create -n abzu python=3.12 -y
conda activate abzu

# Virtualenv
pthon -m venv venv
source venv/bin/activate
```

### Install `poetry` with `pipx`

You can install `poetry` using `pipx`, which is a tool to install and run Python applications in isolated environments. This is the recommended way to install `poetry`.

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

Alternatively, you can install `poetry` using the official installer. Some firewalls block this installation script as a security risk.

```bash
# Try pipx if your firewall prevents this...
curl -sSL https://install.python-poetry.org | python3 -
```

### Install Python Dependencies

```bash
# Install dependencies
poetry install
```

## Contributing

We welcome contributions to this project! Please follow the guidelines below:

### Install Pre-Commit Checks

```bash
# black, isort, flake8, mypy
pre-commit install
```

### Claude Code

This project was written by Russell Jurney with the help of [Claude Code](https://claude.ai/code), a large language model (LLM) from Anthropic. This is made possible by the permissions in [.claude/settings.json](.claude/settings.json) and configuration in [CLAUDE.md](CLAUDE.md). You will want to 'fine-tune' them both to your requirements. Please be sure to double check that you are comfortable with the permissions in `.claude/settings.json` before using this project, as there are security considations. I gave it the ability to perform read-only tasks without my intervention, but some minor write operations are enabled (like `touch`, `git add`, etc.) but not `git commit`.

## License

This project is licensed under the [Apache 2.0 License](LICENSE). See the [LICENSE](LICENSE) file for details.

## Acknowledgements

This work is made possible by the [Open Sanctions Matcher training data](https://www.opensanctions.org/docs/pairs/), the [Sentence Transformers Project](https://sbert.net/) and the [HuggingFace](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) community.
