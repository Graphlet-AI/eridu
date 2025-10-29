"""Main CLI module for Eridu."""

import os
import warnings
from collections import OrderedDict
from importlib import import_module
from pathlib import Path
from typing import Optional

import click
import pandas as pd
import requests
from tqdm import tqdm

# from eridu.cluster import cluster_names, cluster_names_bert
from eridu.etl import evaluate as evaluate_module
from eridu.etl.filter import filter_pairs, filter_statements_to_addresses
from eridu.utils import get_model_path_for_entity_type

# Suppress SyntaxWarnings from HDBSCAN library
warnings.filterwarnings("ignore", category=SyntaxWarning)


def _download_with_progress(url: str, path: Path, desc: str) -> bool:
    """Download a file with progress bar.

    Returns:
        True if successful, False otherwise
    """
    try:
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            click.echo(f"Warning: Failed to download {desc} (HTTP {response.status_code})")
            return False

        total_size = int(response.headers.get("content-length", 0))
        with open(path, "wb") as f:
            with tqdm(total=total_size, unit="B", unit_scale=True, desc=desc) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        click.echo(f"Successfully downloaded {desc}: {path}")
        return True
    except Exception as e:
        click.echo(f"Warning: Error downloading {desc}: {e}")
        return False


def _download_text_file(url: str, path: Path, desc: str) -> bool:
    """Download a text file.

    Returns:
        True if successful, False otherwise
    """
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(path, "w", encoding="utf-8") as f:
                f.write(response.text)
            click.echo(f"Successfully downloaded {desc}: {path}")
            return True
        else:
            click.echo(f"Warning: Failed to download {desc} (HTTP {response.status_code})")
            return False
    except Exception as e:
        click.echo(f"Warning: Error downloading {desc}: {e}")
        return False


class OrderedGroup(click.Group):
    """Custom Click Group that maintains order of commands in help."""

    def __init__(self, name=None, commands=None, **attrs):
        super(OrderedGroup, self).__init__(name, commands, **attrs)
        self.commands = OrderedDict()

    def list_commands(self, ctx):
        """Return commands in the order they were added."""
        return self.commands.keys()


@click.group(cls=OrderedGroup, context_settings={"show_default": True})
@click.version_option()
def cli() -> None:
    """Eridu: Fuzzy matching people and company names for entity resolution using representation learning"""
    pass


@cli.command(context_settings={"show_default": True})
@click.option(
    "--url",
    default="https://storage.googleapis.com/data.opensanctions.org/contrib/sample/pairs-all.csv.gz",
    show_default=True,
    help="URL to download the pairs gzipped CSV file from",
)
@click.option(
    "--output-dir",
    default="./data",
    show_default=True,
    help="Directory to save the downloaded and extracted files",
)
def download(url: str, output_dir: str) -> None:
    """Download labeled entity pairs, checks.yml, and statements.csv files.

    Downloads:
    1. Entity pairs CSV file (gzipped) and converts to Parquet
    2. checks.yml for entity matching evaluation
    3. statements.csv from OpenSanctions for additional entity data
    """
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Get filename from URL
    filename = url.split("/")[-1]
    gz_path = output_dir_path / filename

    # Step 1: Download the pairs file
    click.echo(f"Downloading {url} to {gz_path}")
    _download_with_progress(url, gz_path, filename)

    # Step 2: Download checks.yml file
    checks_url = "https://raw.githubusercontent.com/opensanctions/nomenklatura/refs/heads/main/contrib/name_benchmark/checks.yml"
    checks_path = output_dir_path / "checks.yml"
    click.echo(f"Downloading checks.yml to {checks_path}")
    _download_text_file(checks_url, checks_path, "checks.yml")

    # Step 3: Download statements.csv file
    statements_url = "https://data.opensanctions.org/datasets/latest/default/statements.csv"
    statements_path = output_dir_path / "statements.csv"
    click.echo(f"\nDownloading statements.csv from {statements_url}")
    _download_with_progress(statements_url, statements_path, "statements.csv")

    # Step 4: Read the gzipped CSV directly and convert to Parquet
    click.echo(f"\nReading gzipped CSV file: {gz_path}")
    try:
        # Pandas automatically detects and handles gzipped files
        df = pd.read_csv(gz_path, compression="gzip")
        click.echo(f"Successfully parsed CSV. Shape: {df.shape}")
        click.echo(f"Columns: {', '.join(df.columns)}")
        # Create Parquet file
        parquet_path = output_dir_path / filename.replace(".csv.gz", ".parquet")
        click.echo(f"Converting to Parquet: {parquet_path}")
        df.to_parquet(parquet_path, index=False)
        click.echo(f"Successfully created Parquet file: {parquet_path}")
        # Display basic info about the data
        click.echo("Data sample (first 5 rows):")
        click.echo(df.head(5))
    except Exception as e:
        click.echo(f"Error processing CSV: {e}")
        raise

    click.echo("\nDownload and conversion to Parquet completed successfully.")
    click.echo("\nTo generate a report on this data, run:")
    click.echo(f"  eridu etl report --parquet-path {parquet_path}")


@cli.group(context_settings={"show_default": True})
def etl() -> None:
    """ETL commands for data processing."""
    pass


@etl.command(name="report", context_settings={"show_default": True})
@click.option(
    "--parquet-path",
    default="./data/pairs-all.parquet",
    show_default=True,
    help="Path to the Parquet file to analyze (default is the output from 'eridu download')",
)
@click.option(
    "--truncate",
    default=20,
    show_default=True,
    help="Truncation limit for string display",
)
def etl_report(parquet_path: str, truncate: int) -> None:
    """Generate a report on entity pairs data."""
    from eridu.etl.report import generate_pairs_report

    generate_pairs_report(parquet_path, truncate)


@etl.group(cls=OrderedGroup, context_settings={"show_default": True})
def filter() -> None:
    """Filter commands for data processing."""
    pass


@filter.command(name="pairs", context_settings={"show_default": True})
@click.option(
    "--input",
    "--input-path",
    default="./data/pairs-all.parquet",
    type=click.Path(exists=True, dir_okay=True, readable=True),
    show_default=True,
    help="Path to the input Parquet file to filter",
)
@click.option(
    "--output",
    "--output-path",
    default="./data/filtered",
    type=click.Path(dir_okay=True, writable=True),
    show_default=True,
    help="Directory to save the filtered Parquet files",
)
def filter_pairs_cmd(input: str, output: str) -> None:
    """Filter entity pairs data to exclude sources starting with 'Q'."""

    filter_pairs(input, output)


@filter.command(name="addresses", context_settings={"show_default": True})
@click.option(
    "--statements-path",
    default="./data/statements.csv",
    type=click.Path(exists=True, readable=True),
    show_default=True,
    help="Path to the statements.csv file",
)
@click.option(
    "--output",
    "--output-path",
    default="./data/addresses.yml",
    type=click.Path(writable=True),
    show_default=True,
    help="Path to output YAML file for address pairs",
)
@click.option(
    "--max-pairs",
    default=100,
    show_default=True,
    help="Maximum number of address pairs to generate",
)
def filter_addresses_cmd(statements_path: str, output: str, max_pairs: int) -> None:
    """Filter statements.csv to create address training pairs in YAML format."""

    filter_statements_to_addresses(statements_path, output, max_pairs)


# @cli.group(cls=OrderedGroup, context_settings={"show_default": True})
# def cluster() -> None:
#     """Clustering commands for name entity resolution."""
#     pass


# @cluster.group(cls=OrderedGroup, context_settings={"show_default": True})
# def compute() -> None:
#     """Compute clustering using different approaches."""
#     pass


# @compute.command(name="embed", context_settings={"show_default": True})
# @click.option(
#     "--input",
#     "--input-path",
#     default=None,
#     type=click.Path(exists=True, dir_okay=True, readable=True),
#     help="Path to the input Parquet file containing names to cluster (auto-set based on --data-type if not specified)",
# )
# @click.option(
#     "--data-type",
#     type=click.Choice(["people", "companies", "addresses"]),
#     default="companies",
#     show_default=True,
#     help="Type of data to cluster (determines default input file if --input not specified)",
# )
# @click.option(
#     "--image-dir",
#     default="./images",
#     type=click.Path(file_okay=False, dir_okay=True, writable=True),
#     help="Directory to save the visualization PNG file",
# )
# @click.option(
#     "--output-dir",
#     default="./data",
#     type=click.Path(file_okay=False, dir_okay=True, writable=True),
#     help="Directory to save CSV files and embeddings",
# )
# @click.option(
#     "--model",
#     default=None,
#     help="Sentence transformer model to use for embeddings (auto-selected based on data-type if not specified)",
# )
# @click.option(
#     "--sample-size",
#     default=None,
#     type=int,
#     help="Number of names to sample for clustering (use all names if not specified)",
# )
# @click.option(
#     "--min-cluster-size",
#     default=5,
#     help="Minimum cluster size for HDBSCAN clustering",
# )
# @click.option(
#     "--min-samples",
#     default=3,
#     help="Minimum samples parameter for HDBSCAN clustering",
# )
# @click.option(
#     "--cluster-selection-epsilon",
#     default=0.1,
#     type=float,
#     help="HDBSCAN epsilon for cluster selection (higher values = more noise points)",
# )
# @click.option(
#     "--use-gpu/--no-gpu",
#     default=True,
#     help="Whether to use GPU acceleration for embeddings",
# )
# @click.option(
#     "--random-seed",
#     default=31337,
#     help="Random seed for reproducibility",
# )
# def compute_embed(
#     input: Optional[str],
#     data_type: str,
#     image_dir: str,
#     output_dir: str,
#     model: Optional[str],
#     sample_size: Optional[int],
#     min_cluster_size: int,
#     min_samples: int,
#     cluster_selection_epsilon: float,
#     use_gpu: bool,
#     random_seed: int,
# ) -> None:
#     """Compute clusters using HDBSCAN on sentence transformer embeddings, with T-SNE for visualization."""
#     # Set default input path based on data_type if not specified
#     if input is None:
#         if data_type == "people":
#             input = "./data/filtered/people.parquet"
#         elif data_type == "companies":
#             input = "./data/filtered/companies.parquet"
#         elif data_type == "addresses":
#             input = "./data/filtered/addresses.parquet"
#         click.echo(f"Using input path: {input}")

#     # Set default model path based on data_type if not specified
#     if model is None:
#         model = get_model_path_for_entity_type(data_type)
#         click.echo(f"Using model: {model}")

#     # Input should be set at this point
#     if input is None:
#         raise ValueError(
#             "Input path is None after default assignment. "
#             f"Data type: {data_type}, Expected path should have been set."
#         )

#     cluster_names(
#         input_path=input,
#         entity_type=data_type,
#         image_dir=image_dir,
#         output_dir=output_dir,
#         model_name=model,
#         sample_size=sample_size,
#         min_cluster_size=min_cluster_size,
#         min_samples=min_samples,
#         cluster_selection_epsilon=cluster_selection_epsilon,
#         use_gpu=use_gpu,
#         random_seed=random_seed,
#     )


# @compute.command(name="token", context_settings={"show_default": True})
# @click.option(
#     "--input",
#     "--input-path",
#     default=None,
#     type=click.Path(exists=True, dir_okay=True, readable=True),
#     help="Path to the input Parquet file containing names to cluster (auto-set based on --data-type if not specified)",
# )
# @click.option(
#     "--data-type",
#     type=click.Choice(["people", "companies", "addresses"]),
#     default="companies",
#     show_default=True,
#     help="Type of data to cluster (determines default input file if --input not specified)",
# )
# @click.option(
#     "--image-dir",
#     default="./images",
#     type=click.Path(file_okay=False, dir_okay=True, writable=True),
#     help="Directory to save the visualization PNG file",
# )
# @click.option(
#     "--output-dir",
#     default="./data",
#     type=click.Path(file_okay=False, dir_okay=True, writable=True),
#     help="Directory to save CSV files and features",
# )
# @click.option(
#     "--model",
#     default="bert-base-uncased",
#     help="BERT model to use for tokenization",
# )
# @click.option(
#     "--sample-size",
#     default=None,
#     type=int,
#     help="Number of names to sample for clustering (use all names if not specified)",
# )
# @click.option(
#     "--min-cluster-size",
#     default=5,
#     help="Minimum cluster size for HDBSCAN clustering",
# )
# @click.option(
#     "--min-samples",
#     default=3,
#     help="Minimum samples parameter for HDBSCAN clustering",
# )
# @click.option(
#     "--cluster-selection-epsilon",
#     default=0.1,
#     type=float,
#     help="HDBSCAN epsilon for cluster selection (higher values = more noise points)",
# )
# @click.option(
#     "--max-features",
#     default=10000,
#     help="Maximum number of features for TF-IDF vectorizer",
# )
# @click.option(
#     "--min-df",
#     default=2,
#     help="Minimum document frequency for TF-IDF features",
# )
# @click.option(
#     "--max-df",
#     default=0.95,
#     type=float,
#     help="Maximum document frequency for TF-IDF features",
# )
# @click.option(
#     "--ngram-range",
#     default="1,3",
#     help="N-gram range for TF-IDF features (format: min,max)",
# )
# @click.option(
#     "--random-seed",
#     default=31337,
#     help="Random seed for reproducibility",
# )
# def compute_token(
#     input: Optional[str],
#     data_type: str,
#     image_dir: str,
#     output_dir: str,
#     model: str,
#     sample_size: Optional[int],
#     min_cluster_size: int,
#     min_samples: int,
#     cluster_selection_epsilon: float,
#     max_features: int,
#     min_df: int,
#     max_df: float,
#     ngram_range: str,
#     random_seed: int,
# ) -> None:
#     """Cluster names using traditional NLP approach with BERT tokenization and TF-IDF."""
#     # Set default input path based on data_type if not specified
#     if input is None:
#         if data_type == "people":
#             input = "./data/filtered/people.parquet"
#         elif data_type == "companies":
#             input = "./data/filtered/companies.parquet"
#         elif data_type == "addresses":
#             input = "./data/filtered/addresses.parquet"
#         click.echo(f"Using input path: {input}")

#     # Parse ngram_range from string
#     ngram_min, ngram_max = map(int, ngram_range.split(","))

#     # Input should be set at this point
#     if input is None:
#         raise ValueError(
#             "Input path is None after default assignment. "
#             f"Data type: {data_type}, Expected path should have been set."
#         )

#     cluster_names_bert(
#         input_path=input,
#         image_dir=image_dir,
#         output_dir=output_dir,
#         model_name=model,
#         sample_size=sample_size,
#         min_cluster_size=min_cluster_size,
#         min_samples=min_samples,
#         cluster_selection_epsilon=cluster_selection_epsilon,
#         max_features=max_features,
#         min_df=min_df,
#         max_df=max_df,
#         ngram_range=(ngram_min, ngram_max),
#         random_seed=random_seed,
#     )


# @cluster.command(name="analyze", context_settings={"show_default": True})
# @click.option(
#     "--csv-path",
#     default="./data/cluster_results.csv",
#     type=click.Path(exists=True, dir_okay=False, readable=True),
#     help="Path to the cluster results CSV file from compute command",
# )
# def cluster_analyze(csv_path: str) -> None:
#     """Analyze clustering results and show examples of names in each cluster."""
#     # Analyze the cluster results
#     analyze_cluster_results(csv_path)


# @cluster.command(name="quality", context_settings={"show_default": True})
# @click.option(
#     "--csv-path",
#     default="./data/cluster_results.csv",
#     type=click.Path(exists=True, dir_okay=False, readable=True),
#     help="Path to the cluster results CSV file from compute command",
# )
# def cluster_quality(csv_path: str) -> None:
#     """Analyze cluster quality using distance metrics."""
#     # Load the cluster results and analyze quality
#     df = pd.read_csv(csv_path)
#     analyze_cluster_quality(df)


# @cluster.command(name="split", context_settings={"show_default": True})
# @click.option(
#     "--pairs-file",
#     default="./data/pairs-all.parquet",
#     show_default=True,
#     help="Path to the pairs file (CSV or Parquet)",
# )
# @click.option(
#     "--cluster-file",
#     default="./data/cluster_results.csv",
#     show_default=True,
#     help="Path to the cluster results CSV file",
# )
# @click.option(
#     "--output-dir",
#     default="./data/cluster_splits",
#     show_default=True,
#     help="Directory to save split files",
# )
# @click.option(
#     "--intra-threshold",
#     default=0.2,
#     type=float,
#     help="Intra-cluster distance threshold for tight clusters",
# )
# @click.option(
#     "--min-size",
#     default=5,
#     type=int,
#     help="Minimum cluster size for tight clusters",
# )
# @click.option(
#     "--train-ratio",
#     default=0.7,
#     type=float,
#     help="Training set ratio",
# )
# @click.option(
#     "--test-ratio",
#     default=0.2,
#     type=float,
#     help="Test set ratio",
# )
# @click.option(
#     "--eval-ratio",
#     default=0.1,
#     type=float,
#     help="Evaluation set ratio",
# )
# @click.option(
#     "--random-state",
#     default=42,
#     type=int,
#     help="Random seed for reproducibility",
# )
# def cluster_split(
#     pairs_file: str,
#     cluster_file: str,
#     output_dir: str,
#     intra_threshold: float,
#     min_size: int,
#     train_ratio: float,
#     test_ratio: float,
#     eval_ratio: float,
#     random_state: int,
# ) -> None:
#     """Create cluster-aware train/test/eval splits to prevent overfitting.

#     This command ensures that tight clusters (groups of very similar names) are
#     placed entirely in one partition (train, test, or eval) rather than being
#     split across multiple partitions. This prevents overfitting by ensuring
#     the model cannot memorize cluster patterns during training.

#     Example: eridu cluster split --pairs-file data/pairs-all.parquet
#     """
#     from eridu.etl.cluster_split import create_cluster_aware_splits

#     create_cluster_aware_splits(
#         pairs_file=pairs_file,
#         cluster_file=cluster_file,
#         output_dir=output_dir,
#         intra_distance_threshold=intra_threshold,
#         min_cluster_size=min_size,
#         train_ratio=train_ratio,
#         test_ratio=test_ratio,
#         eval_ratio=eval_ratio,
#         random_state=random_state,
#     )


@cli.command(name="train", context_settings={"show_default": True})
@click.option(
    "--model",
    required=True,
    help="Base SBERT model to fine-tune (e.g., sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)",
)
@click.option(
    "--input",
    "--input-path",
    default=None,
    help="Path to the input Parquet file for training (auto-set based on --data-type if not specified)",
)
@click.option(
    "--data-type",
    type=click.Choice(["people", "companies", "addresses", "both"]),
    default="both",
    show_default=True,
    help="Type of data to train on (determines default input file and logs to WandB)",
)
@click.option(
    "--output",
    "--output-path",
    type=click.Path(file_okay=False, dir_okay=True, writable=True),
    default="./data/output",
    show_default=True,
    help="Directory where training output and model will be saved",
)
@click.option(
    "--sample-fraction",
    default=0.01,
    show_default=True,
    help="Fraction of data to sample for training (1.0 = use all data)",
)
@click.option("--batch-size", default=1024, show_default=True, help="Batch size for training")
@click.option("--epochs", default=10, show_default=True, help="Number of training epochs")
@click.option(
    "--patience",
    default=2,
    show_default=True,
    help="Early stopping patience (number of evaluation steps without improvement)",
)
@click.option(
    "--resampling/--no-resampling",
    default=True,
    show_default=True,
    help="Resample training data for each epoch when sample_fraction < 1.0",
)
@click.option(
    "--fp16/--no-fp16", default=False, show_default=True, help="Use mixed precision training (fp16)"
)
@click.option(
    "--quantization/--no-quantization",
    default=False,
    show_default=True,
    help="Apply quantization to Linear layers (reduces precision to save memory)",
)
@click.option(
    "--use-gpu/--no-gpu",
    default=True,
    show_default=True,
    help="Whether to use GPU for training and inference (if available)",
)
@click.option(
    "--wandb-project",
    default="eridu",
    show_default=True,
    help="Weights & Biases project name for tracking",
)
@click.option(
    "--wandb-entity",
    default="rjurney",
    show_default=True,
    help="Weights & Biases entity (username or team name)",
)
@click.option(
    "--gradient-checkpointing/--no-gradient-checkpointing",
    default=False,
    show_default=True,
    help="Enable gradient checkpointing to save memory at the cost of computation time",
)
@click.option(
    "--weight-decay",
    default=0.01,
    show_default=True,
    help="Weight decay (L2 regularization) to prevent overfitting",
)
@click.option(
    "--random-seed",
    default=31337,
    show_default=True,
    help="Random seed for reproducibility across all frameworks",
)
@click.option(
    "--warmup-ratio",
    default=0.1,
    show_default=True,
    help="Ratio of training steps for learning rate warmup",
)
@click.option(
    "--save-strategy",
    type=click.Choice(["steps", "epoch", "no"]),
    default="steps",
    show_default=True,
    help="Strategy to save model checkpoints during training",
)
@click.option(
    "--eval-strategy",
    type=click.Choice(["steps", "epoch", "no"]),
    default="steps",
    show_default=True,
    help="Strategy to evaluate model during training",
)
@click.option(
    "--learning-rate",
    default=3e-5,
    show_default=True,
    help="Learning rate for optimizer",
)
@click.option(
    "--post-sample-pct",
    default=0.01,
    show_default=True,
    help="Sample percentage for post-training evaluation",
)
@click.option(
    "--max-grad-norm",
    default=1.0,
    show_default=True,
    help="Maximum gradient norm for gradient clipping to prevent exploding gradients",
)
@click.option(
    "--gate-stats-steps",
    default=100,
    show_default=True,
    help="Number of steps between logging loss metrics to WandB",
)
@click.option(
    "--margin",
    default=0.5,
    show_default=True,
    help="Margin for contrastive loss function",
)
def train(
    model: str,
    input: str,
    data_type: str,
    output: str,
    sample_fraction: float,
    batch_size: int,
    epochs: int,
    patience: int,
    resampling: bool,
    fp16: bool,
    quantization: bool,
    use_gpu: bool,
    wandb_project: str,
    wandb_entity: str,
    gradient_checkpointing: bool,
    weight_decay: float,
    random_seed: int,
    warmup_ratio: float,
    save_strategy: str,
    eval_strategy: str,
    learning_rate: float,
    post_sample_pct: float,
    max_grad_norm: float,
    gate_stats_steps: int,
    margin: float,
) -> None:
    """Fine-tune a sentence transformer (SBERT) model for entity matching."""
    click.echo("\n\nNEW TRAINING RUN IS STARTING, PEOPLE...\n\n")

    # Validate that FP16 and quantization are not both enabled
    if fp16 and quantization:
        raise click.UsageError(
            "Error: Cannot use both FP16 and quantization together. Please choose only one option."
        )

    # Set default input path based on data_type if not specified
    data_type_paths = {
        "people": "./data/filtered/people.parquet",
        "companies": "./data/filtered/companies.parquet",
        "addresses": "./data/filtered/addresses.parquet",
        "both": "./data/pairs-all.parquet",
    }

    if input is None:
        input = data_type_paths[data_type]
        click.echo(f"Using default input path for {data_type} data: {input}")

    # Validate that the input path exists
    if not os.path.exists(input):
        error_msg = f"Input file not found: {input}"
        if input in data_type_paths.values() and data_type != "both":
            error_msg += "\nPlease run 'eridu etl filter' first to create filtered data files."
        raise click.UsageError(error_msg)

    click.echo(f"Fine-tuning SBERT model: {model}")
    click.echo(f"Input path: {input}")
    click.echo(f"Data type: {data_type}")
    click.echo(f"Output path: {output}")
    click.echo(f"Sample fraction: {sample_fraction}")
    click.echo(f"Batch size: {batch_size}")
    click.echo(f"Epochs: {epochs}")
    click.echo(f"Early stopping patience: {patience}")
    if sample_fraction < 1.0:
        click.echo(f"Resampling per epoch: {resampling}")
    click.echo(f"FP16: {fp16}")
    click.echo(f"Quantization: {quantization}")
    click.echo(f"Use GPU: {use_gpu}")
    click.echo(f"Gradient checkpointing: {gradient_checkpointing}")
    click.echo(f"Weight decay: {weight_decay}")
    click.echo(f"Random seed: {random_seed}")
    click.echo(f"Warmup ratio: {warmup_ratio}")
    click.echo(f"Save strategy: {save_strategy}")
    click.echo(f"Eval strategy: {eval_strategy}")
    click.echo(f"Learning rate: {learning_rate}")
    click.echo(f"Post-sample percentage: {post_sample_pct}")
    click.echo(f"Max gradient norm: {max_grad_norm}")
    click.echo(f"Gate stats steps: {gate_stats_steps}")
    click.echo(f"Margin: {margin}")
    click.echo(f"W&B Project: {wandb_project}")
    click.echo(f"W&B Entity: {wandb_entity}")

    # Set environment variables based on CLI options
    os.environ["SBERT_MODEL"] = model
    os.environ["INPUT_PATH"] = input
    os.environ["DATA_TYPE"] = data_type
    os.environ["SAMPLE_FRACTION"] = str(sample_fraction)
    os.environ["BATCH_SIZE"] = str(batch_size)
    os.environ["EPOCHS"] = str(epochs)
    os.environ["PATIENCE"] = str(patience)
    os.environ["USE_RESAMPLING"] = "true" if resampling else "false"
    os.environ["WEIGHT_DECAY"] = str(weight_decay)
    os.environ["RANDOM_SEED"] = str(random_seed)
    os.environ["WARMUP_RATIO"] = str(warmup_ratio)
    os.environ["SAVE_STRATEGY"] = save_strategy
    os.environ["EVAL_STRATEGY"] = eval_strategy
    os.environ["LEARNING_RATE"] = str(learning_rate)
    os.environ["POST_SAMPLE_PCT"] = str(post_sample_pct)
    os.environ["MAX_GRAD_NORM"] = str(max_grad_norm)
    os.environ["GATE_STATS_STEPS"] = str(gate_stats_steps)
    os.environ["MARGIN"] = str(margin)
    os.environ["WANDB_PROJECT"] = wandb_project
    os.environ["WANDB_ENTITY"] = wandb_entity
    os.environ["USE_GPU"] = "true" if use_gpu else "false"

    # Set fp16 environment variable (default is False)
    os.environ["USE_FP16"] = "true" if fp16 else "false"

    # Set quantization environment variable (default is False)
    os.environ["USE_QUANTIZATION"] = "true" if quantization else "false"

    # Set gradient checkpointing environment variable
    os.environ["USE_GRADIENT_CHECKPOINTING"] = "true" if gradient_checkpointing else "false"

    # Import the fine_tune_sbert module here to avoid circular imports and run the module
    fine_tune_module = import_module("eridu.train.fine_tune_sbert")
    fine_tune_module.main()


@cli.command("compare", context_settings={"show_default": True})
@click.argument("name1", type=str)
@click.argument("name2", type=str)
@click.option(
    "--model-path",
    default=None,
    help="Path to the fine-tuned SentenceTransformer model directory",
)
@click.option(
    "--data-type",
    type=click.Choice(["people", "companies", "addresses"]),
    default="companies",
    show_default=True,
    help="Type of entities being compared (used to select appropriate model if --model-path not specified)",
)
@click.option(
    "--use-gpu/--no-gpu",
    default=True,
    show_default=True,
    help="Whether to use GPU acceleration for inference",
)
def compare(
    name1: str, name2: str, model_path: Optional[str], data_type: str, use_gpu: bool
) -> None:
    """Compare two names using the fine-tuned SentenceTransformer model.

    Computes the similarity score between NAME1 and NAME2 using
    the specified SBERT model with optional GPU acceleration.

    Example: eridu compare "John Smith" "Jon Smith"
    """
    from eridu.etl.compare import compare_names

    # Use data-type specific model if model path not specified
    if model_path is None:
        model_path = get_model_path_for_entity_type(data_type)

    similarity, success = compare_names(name1, name2, model_path, use_gpu)

    if not success:
        click.echo(f"Error: Could not load model from '{model_path}'.")
        click.echo("Please specify a valid model path with --model-path or train a model first.")
        return

    # Print the similarity score rounded to 3 decimal places
    click.echo(f"{similarity:.3f}")


@cli.group(cls=OrderedGroup, context_settings={"show_default": True})
def evaluate() -> None:
    """Evaluate trained models using different methods."""
    pass


@evaluate.command(name="test", context_settings={"show_default": True})
@click.option(
    "--model-path",
    default=None,
    help="Path to the fine-tuned model directory (default: auto-detect based on environment)",
)
@click.option(
    "--use-gpu/--no-gpu",
    default=True,
    show_default=True,
    help="Whether to use GPU acceleration for evaluation",
)
@click.option(
    "--threshold",
    default=None,
    type=float,
    help="Classification threshold (default: auto-determine optimal threshold)",
)
@click.option(
    "--sample-size",
    default=None,
    type=int,
    help="Number of test samples to evaluate (default: use all available test data)",
)
def evaluate_test(
    model_path: Optional[str], use_gpu: bool, threshold: Optional[float], sample_size: Optional[int]
) -> None:
    """Evaluate a trained SBERT model on the test dataset.

    Loads the model and test data, runs inference, and produces an evaluation report
    with accuracy, precision, recall, F1 score, and other metrics.

    Example: eridu evaluate test
    """
    try:
        evaluate_module.evaluate_and_print_report(model_path, use_gpu, threshold, sample_size)
    except FileNotFoundError as e:
        click.echo(f"Error: {e}")
        click.echo("Make sure you've run 'eridu train' first to generate the model and test data.")
    except Exception as e:
        click.echo(f"Error during evaluation: {e}")


@evaluate.command(name="checks", context_settings={"show_default": True})
@click.option(
    "--checks-path",
    default=None,
    help="Path to the checks file (auto-selected based on entity-type if not specified)",
)
@click.option(
    "--model-path",
    default=None,
    help="Path to the fine-tuned model directory (default: auto-detect based on environment)",
)
@click.option(
    "--use-gpu/--no-gpu",
    default=True,
    show_default=True,
    help="Whether to use GPU acceleration for evaluation",
)
@click.option(
    "--threshold",
    default=0.5,
    type=float,
    show_default=True,
    help="Classification threshold for binary predictions",
)
@click.option(
    "--entity-type",
    type=click.Choice(["person", "company", "address", "both"]),
    default="company",
    show_default=True,
    help="Entity type to evaluate (person, company, address, or both)",
)
def evaluate_checks(
    checks_path: Optional[str],
    model_path: Optional[str],
    use_gpu: bool,
    threshold: float,
    entity_type: str,
) -> None:
    """Evaluate a trained SBERT model using checks.yml test cases.

    Loads the model and checks.yml file, runs evaluation on specified entity type
    and produces detailed reports with metrics and error examples.

    Example: eridu evaluate checks --entity-type company
    Example: eridu evaluate checks --entity-type address
    """
    from eridu.etl.checks_evaluation import generate_checks_report

    # Auto-select checks file based on entity type if not specified
    if checks_path is None:
        if entity_type == "address":
            checks_path = "./data/addresses.yml"
            click.echo(f"Using addresses file: {checks_path}")
        else:
            checks_path = "./data/checks.yml"
            click.echo(f"Using checks file: {checks_path}")

    generate_checks_report(checks_path, entity_type, model_path, use_gpu, threshold, save_csv=True)


if __name__ == "__main__":
    cli()
