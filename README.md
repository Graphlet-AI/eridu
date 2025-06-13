# Eridu

NOTE: this model is a work in progress. It is not yet ready for production use.

Deep fuzzy matching people and company names for multilingual entity resolution using representation learning... that incorporates a deep understanding of people and company names and works _much better_ than string distance methods.

# TLDR: 5 Lines of Code

```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("Graphlet-AI/eridu")

names = [
    "Frank Lloyd Right",
    "Jim Jeffries",
    "Russell H. Jurney",
    "Русс Джерни"
]

embeddings = model.encode(names)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]

print(similarities.numpy())
# [[1.         0.7203882  0.7773637  0.80202234]
#  [0.7203882  1.0000001  0.7606021  0.829355  ]
#  [0.7773637  0.7606021  0.99999994 0.81948197]
#  [0.80202234 0.829355   0.81948197 1.        ]]
```

## Project Overview

This project is a deep fuzzy matching system for person and company names for entity resolution using representation learning. It is designed to match people and company names across languages and character sets, using a pre-trained text embedding model from HuggingFace that we fine-tune using contrastive learning on 2 million labeled pairs of person and company names from the [Open Sanctions Matcher training data](https://www.opensanctions.org/docs/pairs/). The project includes a command-line interface (CLI) utility for training the model and comparing pairs of names using cosine similarity.

Matching people and company names is an intractable problem using traditional parsing based methods: there is too much variation across cultures and jurisdictions to solve the problem by humans programming. This results in complex, cost prohibitive enterprise solutions for name matching like [IBM InfoSphere Global Name Management](https://www.ibm.com/products/ibm-infosphere-global-name-management). Machine learning is used on problems like this one of cultural relevance, where the time to manually programming a solution appproaches infinity, to automatically write a program. Since 2008 there has been an explosion of deep learning methods that automate feature engineering via representation learning methods including such as text embeddings.

This project loads the pre-trained [paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) paraphrase model from HuggingFace and fine-tunes it for the name matching task using contrastive learning on more than 2 million labeled pairs of matching and non-matching (just as important) person and company names from the [Open Sanctions Matcher training data](https://www.opensanctions.org/docs/pairs/) to create a deep fuzzy matching system for entity resolution.

This model is available on HuggingFace Hub as [Graphlet-AI/eridu](https://huggingface.co/Graphlet-AI/eridu) and can be used in any Python project using the [Sentence Transformers](https://sbert.net/) library in five lines of code. The model is designed to be used for entity resolution tasks, such as matching people and company names across different languages and character sets when matching records.

## Getting Started

First go through <a href="#project-setup">Project Setup</a>, then run the CLI: <a href="#eridu-cli">`eridu --help`</a>

## `eridu` CLI

The interface to this work is a command-line (CLI) utility `eridu` that trains a model and a utility that compares a pair of names using our fine-tuned embedding and a metric called cosine similarity that incorporates a deep understanding of people and company names and works _much better_ than string distance methods. This works across languages and charactersets  The distance returned is a number between 0 and 1, where 0 means the names are identical and 1 means they are completely different. The CLI utility is called `eridu` and it has three subcommands: `download`, `train` and `compare`. More will be added in the near future, so check the documentation for updates: `eridu --help`.

Note: this project can be cost-effectively scaled with GPU acceleration comparing many name pairs at once - the `eridu compare` command is slow because it loads the models. This is not indicative of the model's performance or scalability properties.

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
  compare   Compare two names using the fine-tuned SentenceTransformer model.
```

To train the model, run the commands in the order they appear in the documentation. Default arguments will probably work.

### Compare Command Examples

After training a model, you can compare names using the `compare` command:

```bash
# Basic usage - returns a similarity score from 0.0 to 1.0
eridu compare "John Smith" "Jon Smith"

# Compare names with non-Latin characters
eridu compare "Yevgeny Prigozhin" "Евгений Пригожин"

# Specify a different model path
eridu compare "John Smith" "Jon Smith" --model-path /path/to/custom/model

# Disable GPU acceleration
eridu compare "John Smith" "Jon Smith" --no-gpu
```

The output is a number between 0.0 and 1.0, where higher values indicate greater similarity.

### Training with Multiple Epochs and Dataset Resampling

The training process supports multiple epochs with an early stopping mechanism to prevent overfitting:

- Use `--epochs` to specify the number of training epochs (default: 10)
- Use `--patience` to control early stopping patience (default: 2)
- Use `--resampling/--no-resampling` to enable/disable dataset resampling for each epoch
- Use `--weight-decay` to control L2 regularization strength (default: 0.01)
- Use `--random-seed` to set the random seed for reproducibility (default: 31337)
- Use `--warmup-ratio` to control the learning rate warmup (default: 0.1)
- Use `--learning-rate` to set the optimizer learning rate (default: 3e-5)
- Use `--save-strategy` to control when checkpoints are saved (`steps`, `epoch`, or `no`) (default: `steps`)
- Use `--eval-strategy` to control when evaluation happens (`steps`, `epoch`, or `no`) (default: `steps`)

When working with a sample of the dataset (`--sample-fraction` < 1.0), the resampling feature creates a fresh sample for each epoch, allowing the model to see different examples in each training cycle. This is particularly useful when working with very large datasets where using the full dataset is impractical.

Example:

```bash
# Train for 20 epochs with dataset resampling on 10% of the data
eridu train --epochs 20 --patience 5 --resampling --sample-fraction 0.1

# Train for 15 epochs without resampling on 20% of the data
eridu train --epochs 15 --no-resampling --sample-fraction 0.2

# Train with lower weight decay to reduce regularization (potentially better for similar pairs)
eridu train --epochs 10 --weight-decay 0.001 --sample-fraction 0.1

# Train with custom random seed and warmup ratio
eridu train --random-seed 42 --warmup-ratio 0.2 --sample-fraction 0.1

# Train with a different learning rate
eridu train --learning-rate 5e-5 --sample-fraction 0.1

# Train with evaluation at the end of each epoch instead of at steps
eridu train --save-strategy epoch --eval-strategy epoch --sample-fraction 0.1
```

### GPU Acceleration

This project supports GPU acceleration for both training and inference. If available, it will automatically use:

- NVIDIA GPUs via CUDA
- Apple Silicon GPUs via Metal Performance Shaders (MPS)

You can control GPU usage with command-line flags:

- For training: `eridu train --use-gpu` or `eridu train --no-gpu`
- For comparison: `eridu compare "Name One" "Name Two" --use-gpu` or `eridu compare "Name One" "Name Two" --no-gpu`

GPU acceleration significantly improves performance, especially for large datasets and batch inference operations.

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

### Optional: Weights and Biases for Experiment Tracking

This project uses [Weights and Biases](https://wandb.ai/) for experiment tracking. If you want to use this feature, you need to log in to your Weights and Biases account. You can do this by running the following command:

```bash
wandb login
```

Then you need to set the `--wandb-project` and --wandb-entity`options in the`eridu train` CLI.

```bash
eridu train --wandb-project "<my_project>" --wandb-entity "<my_entity>"
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

## Pre-Trained Model vs Fine-Tuned Model

The pre-trained model is the [paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) model from HuggingFace.

```csv
           sentence1         sentence2  similarity
0         John Smith        John Smith    1.000000
1         John Smith     John H. Smith    0.953342
2  Yevgeny Prigozhin  Евгений Пригожин    0.744036
3         Ben Lorica               罗瑞卡    0.764319
```

The fine-tuned model is the same model, but trained on 2 million labeled pairs of person and company names from the [Open Sanctions Matcher training data](https://www.opensanctions.org/docs/pairs/). The fine-tuned model is much better at matching names than the pre-trained model.

```csv
           sentence1         sentence2  similarity
0         John Smith        John Smith    1.000000
1         John Smith     John H. Smith    0.960119
2  Yevgeny Prigozhin  Евгений Пригожин    0.997346
3         Ben Lorica               罗瑞卡    0.968592
```

Note that a full performance analysis is underway...

## Production Run Configuration

The production run was done on a 10% sample of the data using a Lambda Labs A100 `gpu_1x_a100` with 40GB GPU RAM. The process is described in the script [lambda.sh](lambda.sh), which is not yet fully automated. I monitored the process using `nvidia-smi -l 1` to verify GPU utilization (bursty 100% GPU).

Metrics from the last production training run are up on [Weights and Biases, project Eridu](https://wandb.ai/rjurney/eridu/runs/nn06qw3r?nw=nwuserrjurney).

The commands used to train are:

```bash
# These are the default arguments...
eridu download --url "https://storage.googleapis.com/data.opensanctions.org/contrib/sample/pairs-all.csv.gz" --output-dir data

# These are the default arguments...
eridu etl report --parquet-path data/pairs-all.parquet

# Login to your Weights and Biases account
wandb login

# I needed to increase the batch size to utilize A100 GPUs' 40GB GPU RAM
# Using 4 epochs with resampling on each epoch (new example pairs in each epoch)
# Using a weight decay of 0.01 for regularization
# Using a random seed of 31337 for reproducibility
# Using warmup ratio of 0.1 for learning rate schedule
# Using epoch-based save and evaluation strategy
# Using learning rate of 3e-5 for optimization
# Using FP16 mixed precision for faster training
eridu train --use-gpu \
            --batch-size 64 \
            --epochs 5 \
            --patience 1 \
            --resampling \
            --weight-decay 0.01 \
            --random-seed 31337 \
            --warmup-ratio 0.1 \
            --learning-rate 3e-5 \
            --save-strategy epoch \
            --eval-strategy epoch \
            --sample-fraction 0.05
            # --fp16
```

## License

This project is licensed under the [Apache 2.0 License](LICENSE). See the [LICENSE](LICENSE) file for details.

## Acknowledgements

This work is made possible by the [Open Sanctions Matcher training data](https://www.opensanctions.org/docs/pairs/), the [Sentence Transformers Project](https://sbert.net/) and the [HuggingFace](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) community.

## About Ancient Eridu

<center><img src="https://rjurneyopen.s3.amazonaws.com/Ancient-Eridu-Tell-Abu-Shahrain.jpg" width="500px" alt="Google Maps overhead view of Tell Abu Shahrein - Ancient Eridu" /></center>

Ancient [Eridu](https://en.wikipedia.org/wiki/Eridu) (modern [Tell Abu Shahrain in Southern Iraq](https://maps.app.goo.gl/xXACdHh1Ppmx7NAf6)) was the world's first city, by Sumerian tradition, with a history spanning 7,000 years. It was the first place where "kingship descended from heaven" to lead farmers to build and operate the first complex irrigation network that enabled intensive agriculture sufficient to support the first true urban population.
