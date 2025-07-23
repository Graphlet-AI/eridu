"""Filter entity pairs data using PySpark."""

import os
import random
from itertools import combinations
from pathlib import Path

import click
import yaml
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

    # Create Spark session with proper memory configuration
    spark = (
        SparkSession.builder.appName("Eridu ETL Report")
        .config("spark.driver.memory", "16g")
        .config("spark.driver.maxResultSize", "8g")
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "10000")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.adaptive.skewJoin.enabled", "true")
        .getOrCreate()
    )

    try:
        # Read the parquet file
        click.echo(f"Reading data from {input_path}")
        pairs_df: DataFrame = spark.read.parquet(input_path)

        # Show initial count
        initial_count = pairs_df.count()
        click.echo(f"Initial record count: {initial_count:,}")

        # Apply filter and select columns
        # filtered_df = pairs_df.filter(~F.col("source").startswith("Q"))
        filtered_df = pairs_df

        # Remove duplicates based on key fields
        duplicate_cols = ["left_name", "right_name"]
        before_dedup_count = filtered_df.count()
        filtered_df = filtered_df.dropDuplicates(duplicate_cols)
        after_dedup_count = filtered_df.count()

        click.echo(f"Removed {before_dedup_count - after_dedup_count:,} duplicate records")
        click.echo(
            f"Records after left_name / right_name df.dropDuplicates() deduplication: {after_dedup_count:,}"
        )

        # Show final filtered count (after both source filter and deduplication)
        filtered_count = filtered_df.count()
        click.echo(f"Final filtered record count: {filtered_count:,}")
        click.echo(
            f"Total removed {initial_count - filtered_count:,} records ({(initial_count - filtered_count) / initial_count * 100:.1f}%)"
        )

        people_df = (
            filtered_df.filter(
                (F.col("left_category") == "PER") & (F.col("right_category") == "PER")
            )
            # .unionAll(
            #     filtered_df.filter(
            #         (F.col("left_category") == "PER")
            #         & (F.col("right_category") == "ORG")
            #         & (F.col("match") == 0)
            #     )
            # )
            # .unionAll(
            #     filtered_df.filter(
            #         (F.col("left_category") == "ORG")
            #         & (F.col("right_category") == "PER")
            #         & (F.col("match") == 0)
            #     )
            # )
        )
        click.echo(f"Filtered people records count: {people_df.count():,}")

        companies_df = (
            filtered_df.filter(
                (F.col("left_category") == "ORG") | (F.col("right_category") == "ORG")
            )
            # .unionAll(
            #     filtered_df.filter(
            #         (F.col("left_category") == "ORG")
            #         & (F.col("right_category") == "PER")
            #         & (F.col("match") == 0)
            #     )
            # )
            # .unionAll(
            #     filtered_df.filter(
            #         (F.col("left_category") == "PER")
            #         & (F.col("right_category") == "ORG")
            #         & (F.col("match") == 0)
            #     )
            # )
        )
        click.echo(f"Filtered companies records count: {companies_df.count():,}")

        # Filter addresses (LOC category for locations/addresses)
        addresses_df = filtered_df.filter(
            (F.col("left_category") == "LOC") & (F.col("right_category") == "LOC")
        )
        click.echo(f"Filtered addresses records count: {addresses_df.count():,}")

        # Write the filtered data to separate files in the output directory
        click.echo(f"Writing filtered data to {output_path}")

        filtered_path = os.path.join(output_path, "filtered.parquet")
        people_path = os.path.join(output_path, "people.parquet")
        companies_path = os.path.join(output_path, "companies.parquet")
        addresses_path = os.path.join(output_path, "addresses.parquet")

        filtered_df.coalesce(1).write.mode("overwrite").parquet(filtered_path)
        people_df.coalesce(1).write.mode("overwrite").parquet(people_path)
        companies_df.coalesce(1).write.mode("overwrite").parquet(companies_path)
        addresses_df.coalesce(1).write.mode("overwrite").parquet(addresses_path)

        click.echo("Filtering completed successfully!")
        click.echo(f"  - All filtered data: {filtered_path}")
        click.echo(f"  - People data: {people_path}")
        click.echo(f"  - Companies data: {companies_path}")
        click.echo(f"  - Addresses data: {addresses_path}")

    finally:
        # Stop Spark session
        spark.stop()


def filter_statements_to_addresses(  # noqa: C901
    statements_path: str = "./data/statements.csv",
    output_path: str = "./data/addresses.yml",
    max_pairs: int = 100,
) -> None:
    """Filter statements.csv to extract Address records and create training pairs in YAML format.

    Uses canonical_id to determine matches: same canonical_id = match, different = non-match.

    Args:
        statements_path: Path to statements.csv file
        output_path: Path to output YAML file for address pairs
        max_pairs: Maximum number of address pairs to generate
    """

    # Create Spark session with proper memory configuration
    spark = (
        SparkSession.builder.appName("Eridu Statements Filter")
        .config("spark.driver.memory", "16g")
        .config("spark.driver.maxResultSize", "8g")
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "10000")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.adaptive.skewJoin.enabled", "true")
        .getOrCreate()
    )

    try:
        # Read the statements CSV file
        click.echo(f"Reading statements data from {statements_path}")
        statements_df = spark.read.option("header", "true").csv(statements_path)

        # Filter to Address schema records with 'full' property (complete addresses)
        address_df = statements_df.filter(
            (F.col("schema") == "Address")
            & (F.col("prop") == "full")
            & (F.col("value").isNotNull())
            & (F.length(F.col("value")) > 10)  # Filter out very short addresses
        )

        address_count = address_df.count()
        click.echo(f"Found {address_count} full address records")

        # Sample the data to make it manageable (take a reasonable subset)
        sample_fraction = min(1.0, 10000.0 / address_count) if address_count > 10000 else 1.0
        if sample_fraction < 1.0:
            address_df = address_df.sample(sample_fraction, seed=42)
            click.echo(f"Sampling {sample_fraction:.3f} of data for processing")

        # Collect address data to Python
        address_data = address_df.select(
            F.col("entity_id"), F.col("value").alias("address_text"), F.col("canonical_id")
        ).collect()

        if len(address_data) < 2:
            click.echo(f"Not enough address data found: {len(address_data)}. Need at least 2.")
            return

        # Group addresses by canonical_id to find matches and non-matches
        canonical_groups: dict[str, list[dict]] = {}
        for row in address_data:
            canonical_id = row.canonical_id
            if canonical_id not in canonical_groups:
                canonical_groups[canonical_id] = []
            canonical_groups[canonical_id].append(
                {
                    "entity_id": row.entity_id,
                    "address_text": row.address_text,
                    "canonical_id": canonical_id,
                }
            )

        click.echo(f"Found {len(canonical_groups)} unique canonical address groups")

        # Create pairs
        pairs = []
        random.seed(42)

        # Create matching pairs (same canonical_id, different entity_id)
        matching_pairs = []
        for canonical_id, addresses in canonical_groups.items():
            if len(addresses) > 1:
                # Create all combinations within this canonical group
                for addr1, addr2 in combinations(addresses, 2):
                    matching_pairs.append((addr1, addr2, True))

        # Sample matching pairs
        random.shuffle(matching_pairs)
        selected_matching = matching_pairs[: max_pairs // 2]

        # Create non-matching pairs (different canonical_ids)
        non_matching_pairs = []
        canonical_list = list(canonical_groups.keys())
        for i in range(len(canonical_list)):
            for j in range(i + 1, len(canonical_list)):
                canonical_id1, canonical_id2 = canonical_list[i], canonical_list[j]
                # Pick one address from each canonical group
                addr1 = random.choice(canonical_groups[canonical_id1])
                addr2 = random.choice(canonical_groups[canonical_id2])
                non_matching_pairs.append((addr1, addr2, False))

        # Sample non-matching pairs
        random.shuffle(non_matching_pairs)
        selected_non_matching = non_matching_pairs[: max_pairs // 2]

        # Combine all pairs
        all_pairs = selected_matching + selected_non_matching
        random.shuffle(all_pairs)

        click.echo(
            f"Creating {len(all_pairs)} address pairs ({len(selected_matching)} matches, {len(selected_non_matching)} non-matches)"
        )

        # Create YAML pairs
        for addr1, addr2, is_match in all_pairs:
            if is_match:
                description = "Same address with formatting variations"
            else:
                description = "Different addresses"

            pair = {
                "match": is_match,
                "schema": "Address",
                "description": description,
                "query": {"name": addr1["address_text"]},
                "candidate": {"name": addr2["address_text"]},
            }
            pairs.append(pair)

        # Create YAML structure
        yaml_data = {"checks": pairs}

        # Write to YAML file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        match_count = sum(1 for p in pairs if p["match"])
        non_match_count = len(pairs) - match_count

        click.echo(f"Address pairs written to {output_path}")
        click.echo(f"  - Total pairs: {len(pairs)}")
        click.echo(f"  - Matching pairs: {match_count}")
        click.echo(f"  - Non-matching pairs: {non_match_count}")

    finally:
        # Stop Spark session
        spark.stop()
