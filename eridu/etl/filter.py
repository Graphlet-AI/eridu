"""Filter entity pairs data using PySpark."""

import os
from pathlib import Path

import click
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


def filter_pairs(input_path: str, output_path: str) -> None:
    """Filter pairs data to exclude entries where source starts with 'Q'.

    Args:
        input_path: Path to input Parquet file
        output_path: Path to output directory for filtered Parquet files
    """
    # Validate output_path is a directory
    output_dir = Path(output_path)
    if output_dir.exists() and output_dir.is_file():
        raise click.BadParameter(f"Output path {output_path} is a file, not a directory")

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Spark session
    spark = SparkSession.builder.appName("eridu-filter").getOrCreate()

    try:
        # Read the parquet file
        click.echo(f"Reading data from {input_path}")
        pairs_df: DataFrame = spark.read.parquet(input_path)

        # Show initial count
        initial_count = pairs_df.count()
        click.echo(f"Initial record count: {initial_count:,}")

        # Apply filter and select columns
        filtered_df = pairs_df.filter(~F.col("source").startswith("Q"))

        # Show filtered count
        filtered_count = filtered_df.count()
        click.echo(f"Filtered record count: {filtered_count:,}")
        click.echo(
            f"Removed {initial_count - filtered_count:,} records ({(initial_count - filtered_count) / initial_count * 100:.1f}%)"
        )

        people_df = (
            filtered_df.filter(
                (F.col("left_category") == "PER") & (F.col("right_category") == "PER")
            )
            .unionAll(
                filtered_df.filter(
                    (F.col("left_category") == "PER")
                    & (F.col("right_category") == "ORG")
                    & (F.col("match") == 0)
                )
            )
            .unionAll(
                filtered_df.filter(
                    (F.col("left_category") == "ORG")
                    & (F.col("right_category") == "PER")
                    & (F.col("match") == 0)
                )
            )
        )
        click.echo(f"Filtered people records count: {people_df.count():,}")

        companies_df = (
            filtered_df.filter(
                (F.col("left_category") == "ORG") & (F.col("right_category") == "ORG")
            )
            .unionAll(
                filtered_df.filter(
                    (F.col("left_category") == "ORG")
                    & (F.col("right_category") == "PER")
                    & (F.col("match") == 0)
                )
            )
            .unionAll(
                filtered_df.filter(
                    (F.col("left_category") == "PER")
                    & (F.col("right_category") == "ORG")
                    & (F.col("match") == 0)
                )
            )
        )
        click.echo(f"Filtered companies records count: {companies_df.count():,}")

        # Write the filtered data to separate files in the output directory
        click.echo(f"Writing filtered data to {output_path}")

        filtered_path = os.path.join(output_path, "filtered.parquet")
        people_path = os.path.join(output_path, "people.parquet")
        companies_path = os.path.join(output_path, "companies.parquet")

        filtered_df.coalesce(1).write.mode("overwrite").parquet(filtered_path)
        people_df.coalesce(1).write.mode("overwrite").parquet(people_path)
        companies_df.coalesce(1).write.mode("overwrite").parquet(companies_path)

        click.echo("Filtering completed successfully!")
        click.echo(f"  - All filtered data: {filtered_path}")
        click.echo(f"  - People data: {people_path}")
        click.echo(f"  - Companies data: {companies_path}")

    finally:
        # Stop Spark session
        spark.stop()
