"""Main CLI module for Eridu."""

from pathlib import Path

import click
import pandas as pd
import requests
from tqdm import tqdm


@click.group()
@click.version_option()
def cli() -> None:
    """Eridu: Fuzzy matching for entity resolution through representation learning."""
    pass


@cli.group()
def etl() -> None:
    """ETL commands for data processing."""
    pass


@cli.command()
@click.option(
    "--url",
    default="https://storage.googleapis.com/data.opensanctions.org/contrib/sample/pairs-all.csv.gz",
    show_default=True,
    help="URL to download the gzipped CSV file from",
)
@click.option(
    "--output-dir",
    default="./data",
    show_default=True,
    help="Directory to save the downloaded and extracted files",
)
def download(url: str, output_dir: str) -> None:
    """Download and convert a gzipped CSV file to Parquet format."""
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Get filename from URL
    filename = url.split("/")[-1]
    gz_path = output_dir_path / filename

    # Step 1: Download the file
    click.echo(f"Downloading {url} to {gz_path}")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))

    with open(gz_path, "wb") as f:
        with tqdm(total=total_size, unit="B", unit_scale=True, desc=filename) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    # Step 2: Read the gzipped CSV directly and convert to Parquet
    click.echo(f"Reading gzipped CSV file: {gz_path}")
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

    click.echo("Download and conversion to Parquet completed successfully.")


if __name__ == "__main__":
    cli()
