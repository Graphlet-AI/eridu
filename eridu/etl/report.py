"""Generate reports on entity data pairs."""

import pyspark.sql.functions as F
from pyspark.sql import DataFrame, SparkSession


def format_counts(df: DataFrame) -> DataFrame:
    """Format counts with commas."""
    return df.withColumn("count", F.format_string("%,d", F.col("count")))


def generate_pairs_report(parquet_path: str, truncate: int = 20) -> None:
    """
    Generate a report on entity pairs data.

    Args:
        parquet_path: Path to the parquet file
        truncate: Truncation value for string display
    """
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

    # Load the data
    pairs_df = spark.read.parquet(parquet_path)

    # Repartition the data or we get Java heap space errors
    pairs_df = pairs_df.repartition(100)

    # Show basic info
    print(f"Total records: {pairs_df.count():,}")
    print(f"Columns: {', '.join(pairs_df.columns)}")

    # Do we have positive and negative pairs?
    print("\n=== Match Distribution ===")
    match_counts_df = pairs_df.groupBy("match").count().orderBy("count", ascending=False)
    format_counts(match_counts_df).show()

    # What are the categories?
    print("\n=== Category Pairs ===")
    category_counts_df = (
        pairs_df.groupBy("left_category", "right_category")
        .count()
        .orderBy("left_category", "right_category")
    )
    format_counts(category_counts_df).show(truncate=False)

    # What are the categories for positive / negative pairs?
    print("\n=== Category Pairs - Positive / Negative ===")
    category_match_counts_df = (
        pairs_df.groupBy("left_category", "right_category", "match")
        .count()
        .orderBy("left_category", "right_category", "match")
    )
    format_counts(category_match_counts_df).show(truncate=False)

    # What about language pairs?
    print("\n=== Language Pairs ===")
    lang_counts_df = (
        pairs_df.groupBy("left_lang", "right_lang").count().orderBy("count", ascending=False)
    )
    format_counts(lang_counts_df).show(truncate=False)

    # Sample of matching names
    print("\n=== Sample Names ===")
    pairs_df.select("left_name", "right_name", "match").limit(10).show(truncate=46)

    # Single word vs multi-word names
    single_word_names = pairs_df.filter(
        (F.size(F.split(pairs_df.left_name, " ")) == 1)
        & (pairs_df.left_lang == pairs_df.right_lang)
        & (pairs_df.match == "true")
    ).select("left_name", "right_name", "match")
    print(f"\n=== Single Word Names (Same Language): {single_word_names.count():,} ===")
    single_word_names.show(10, truncate=truncate)

    # Check for duplicates based on key fields
    print("\n=== Duplicate Analysis ===")
    duplicate_cols = ["left_name", "right_name"]

    # Repartition again...
    pairs_df = pairs_df.repartition(1000)

    # Count total records
    total_records: int = pairs_df.count()
    print(f"Total records: {total_records:,}")
    if total_records == 0:
        raise ValueError("No records found.")

    # Count unique records based on the key fields
    unique_records = pairs_df.select(*duplicate_cols).distinct().count()
    print(f"Unique records (by {', '.join(duplicate_cols)}): {unique_records:,}")

    # Calculate duplicates
    duplicate_count = total_records - unique_records
    duplicate_pct = (duplicate_count / total_records * 100) if total_records > 0 else 0.0
    print(f"Duplicate records: {duplicate_count:,} ({duplicate_pct:.1f}%)")

    # Show examples of duplicated records
    if duplicate_count > 0:
        print("\n=== Duplicate Examples ===")
        # Group by the key fields and show records that appear more than once
        # Limit and cache to avoid memory issues
        duplicated_df = (
            pairs_df.groupBy(*duplicate_cols)
            .count()
            .filter(F.col("count") > 1)
            .orderBy(F.col("count").desc())
            .limit(1000)  # Limit to top 1000 duplicate patterns
            .cache()
        )

        # Force computation and get count
        duplicated_count = duplicated_df.count()
        print(f"Top duplicate patterns found: {duplicated_count:,}")
        duplicated_df.show(10, truncate=False)

        # Show actual duplicate records for the top pattern
        if duplicated_count > 0:
            top_duplicate = duplicated_df.first()
            if top_duplicate is not None:
                print(
                    f"\n=== Sample Duplicate Records (Pattern appears {top_duplicate['count']} times) ==="
                )
                sample_duplicates = pairs_df.filter(
                    (F.col("left_name") == top_duplicate["left_name"])
                    & (F.col("right_name") == top_duplicate["right_name"])
                ).limit(
                    5
                )  # Limit to 5 sample records
                sample_duplicates.show(truncate=False)

    spark.stop()


def main(parquet_path: str, truncate: int = 20) -> None:
    """Main entry point for the report script."""
    generate_pairs_report(parquet_path, truncate)
