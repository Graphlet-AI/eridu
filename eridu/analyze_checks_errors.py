"""Analyze entity matching errors from checks.yml evaluation using PySpark."""

import re
import unicodedata

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType


def detect_script(text: str) -> str:  # noqa: C901
    """Detect the primary script/character set of a text."""
    if not text:
        return "Unknown"

    scripts = {
        "Latin": 0,
        "Cyrillic": 0,
        "Arabic": 0,
        "Chinese": 0,
        "Japanese": 0,
        "Hebrew": 0,
        "Greek": 0,
        "Devanagari": 0,
        "Georgian": 0,
        "Armenian": 0,
        "Thai": 0,
        "Korean": 0,
    }

    for char in text:
        try:
            name = unicodedata.name(char)
            if "LATIN" in name:
                scripts["Latin"] += 1
            elif "CYRILLIC" in name:
                scripts["Cyrillic"] += 1
            elif "ARABIC" in name:
                scripts["Arabic"] += 1
            elif "CJK" in name or "CHINESE" in name:
                scripts["Chinese"] += 1
            elif "HIRAGANA" in name or "KATAKANA" in name or "JAPANESE" in name:
                scripts["Japanese"] += 1
            elif "HEBREW" in name:
                scripts["Hebrew"] += 1
            elif "GREEK" in name:
                scripts["Greek"] += 1
            elif "DEVANAGARI" in name:
                scripts["Devanagari"] += 1
            elif "GEORGIAN" in name:
                scripts["Georgian"] += 1
            elif "ARMENIAN" in name:
                scripts["Armenian"] += 1
            elif "THAI" in name:
                scripts["Thai"] += 1
            elif "HANGUL" in name or "KOREAN" in name:
                scripts["Korean"] += 1
        except ValueError:
            continue

    # Return the script with the most characters
    max_script = max(scripts.items(), key=lambda x: x[1])
    if max_script[1] == 0:
        return "Unknown"
    return max_script[0]


def detect_language_heuristic(text: str) -> str:
    """Simple heuristic language detection based on common patterns."""
    text_lower = text.lower()

    # German patterns
    if any(pattern in text_lower for pattern in ["schaft", "gmbh", "ag ", "von ", "der ", "und "]):
        return "German"

    # Russian patterns (based on common endings and words)
    if any(char in text for char in "абвгдежзийклмнопрстуфхцчшщъыьэюя"):
        return "Russian"

    # Arabic patterns
    if any(char in text for char in "ابتثجحخدذرزسشصضطظعغفقكلمنهوي"):
        return "Arabic"

    # Chinese patterns
    if re.search(r"[\u4e00-\u9fff]", text):
        return "Chinese"

    # Japanese patterns
    if re.search(r"[\u3040-\u309f\u30a0-\u30ff]", text):
        return "Japanese"

    # Korean patterns
    if re.search(r"[\uac00-\ud7af]", text):
        return "Korean"

    # Default to English for Latin script
    if detect_script(text) == "Latin":
        return "English"

    return "Unknown"


def categorize_error_type(query: str, candidate: str, label: str) -> str:  # noqa: C901
    """Categorize the type of error based on the query, candidate, and label."""
    if label:
        return label

    query_lower = query.lower()
    candidate_lower = candidate.lower()

    # Character-based errors
    if len(query) == len(candidate) and sum(a != b for a, b in zip(query, candidate)) == 1:
        return "Single Character Difference"

    # Name variations
    if query_lower.replace(" ", "") == candidate_lower.replace(" ", ""):
        return "Spacing Difference"

    # Company type variations
    company_types = ["ag", "gmbh", "ltd", "limited", "inc", "corporation", "corp", "llc", "sa"]
    for ctype in company_types:
        if ctype in query_lower and ctype not in candidate_lower:
            return "Company Type Mismatch"
        if ctype not in query_lower and ctype in candidate_lower:
            return "Company Type Mismatch"

    # Title/prefix variations
    titles = ["mr", "dr", "mrs", "ms", "prof", "sir", "lord", "lady"]
    for title in titles:
        if title + "." in query_lower or title + " " in query_lower:
            return "Title/Prefix Variation"

    # Initial vs full name
    if "." in query and "." not in candidate:
        return "Initial vs Full Name"

    # Word order difference
    query_words = set(query_lower.split())
    candidate_words = set(candidate_lower.split())
    if query_words == candidate_words and query_lower != candidate_lower:
        return "Word Order Difference"

    # Subset/superset
    if query_lower in candidate_lower or candidate_lower in query_lower:
        return "Subset/Superset"

    # Default
    return "Other"


def analyze_errors(spark: SparkSession, csv_path: str, error_type: str) -> DataFrame:
    """Analyze errors from a CSV file."""
    # Read CSV
    df = spark.read.csv(csv_path, header=True, inferSchema=True)

    # Register UDFs
    detect_script_udf = F.udf(detect_script, StringType())
    detect_language_udf = F.udf(detect_language_heuristic, StringType())
    categorize_error_udf = F.udf(categorize_error_type, StringType())

    # Add analysis columns
    df_analyzed = (
        df.withColumn("query_script", detect_script_udf(F.col("query_name")))
        .withColumn("candidate_script", detect_script_udf(F.col("candidate_name")))
        .withColumn("query_language", detect_language_udf(F.col("query_name")))
        .withColumn("candidate_language", detect_language_udf(F.col("candidate_name")))
        .withColumn(
            "error_category",
            categorize_error_udf(F.col("query_name"), F.col("candidate_name"), F.col("label")),
        )
    )

    return df_analyzed


def generate_error_report(spark: SparkSession, output_dir: str = "data/evaluation_results") -> None:
    """Generate comprehensive error analysis report."""
    print("\n" + "=" * 80)
    print("ENTITY MATCHING ERROR ANALYSIS REPORT")
    print("=" * 80)

    # Analyze false positives
    print("\n### FALSE POSITIVES ANALYSIS ###")
    fp_df = analyze_errors(spark, f"{output_dir}/false_positives.csv", "false_positive")

    if fp_df.count() > 0:
        print(f"\nTotal False Positives: {fp_df.count()}")

        # By entity type
        print("\n1. False Positives by Entity Type:")
        fp_df.groupBy("schema").count().orderBy(F.desc("count")).show()

        # By error category
        print("\n2. False Positives by Error Category:")
        fp_df.groupBy("error_category").count().orderBy(F.desc("count")).show()

        # By script
        print("\n3. False Positives by Script (Query):")
        fp_df.groupBy("query_script").count().orderBy(F.desc("count")).show()

        # By language
        print("\n4. False Positives by Language (Query):")
        fp_df.groupBy("query_language").count().orderBy(F.desc("count")).show()

        # Cross-script errors
        print("\n5. Cross-Script False Positives:")
        cross_script_fp = fp_df.filter(F.col("query_script") != F.col("candidate_script"))
        if cross_script_fp.count() > 0:
            cross_script_fp.groupBy("query_script", "candidate_script").count().orderBy(
                F.desc("count")
            ).show()
        else:
            print("No cross-script false positives found.")

        # Examples by category
        print("\n6. Example False Positives by Category:")
        categories = fp_df.select("error_category").distinct().collect()
        for row in categories[:5]:  # Show top 5 categories
            category = row["error_category"]
            print(f"\n{category}:")
            fp_df.filter(F.col("error_category") == category).select(
                "query_name", "candidate_name", "similarity_score"
            ).show(3, truncate=False)

    # Analyze false negatives
    print("\n### FALSE NEGATIVES ANALYSIS ###")
    try:
        fn_df = analyze_errors(spark, f"{output_dir}/false_negatives.csv", "false_negative")

        if fn_df.count() > 0:
            print(f"\nTotal False Negatives: {fn_df.count()}")

            # Similar analysis for false negatives
            print("\n1. False Negatives by Entity Type:")
            fn_df.groupBy("schema").count().orderBy(F.desc("count")).show()

            print("\n2. False Negatives by Error Category:")
            fn_df.groupBy("error_category").count().orderBy(F.desc("count")).show()
        else:
            print("No false negatives found.")
    except Exception:
        print("No false negatives file found or no false negatives.")

    # Summary statistics
    print("\n### SUMMARY STATISTICS ###")
    if fp_df.count() > 0:
        print("\nFalse Positive Similarity Score Statistics:")
        fp_df.select(
            F.min("similarity_score").alias("min_score"),
            F.max("similarity_score").alias("max_score"),
            F.avg("similarity_score").alias("avg_score"),
            F.expr("percentile_approx(similarity_score, 0.5)").alias("median_score"),
        ).show()

    # Save detailed analysis
    if fp_df.count() > 0:
        fp_df.coalesce(1).write.mode("overwrite").csv(
            f"{output_dir}/false_positives_analyzed", header=True
        )
        print(f"\nDetailed analysis saved to {output_dir}/false_positives_analyzed/")


def main() -> None:
    """Main function to run the error analysis."""
    # Create Spark session
    spark = (
        SparkSession.builder.appName("EntityMatchingErrorAnalysis")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .getOrCreate()
    )

    try:
        generate_error_report(spark)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
