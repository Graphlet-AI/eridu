"""Analyze error categories and their prominence from evaluation results."""

import os
from typing import Any, Dict

import pandas as pd


def analyze_error_categories(csv_path: str) -> Dict[str, Any]:
    """Analyze error categories from error_analysis.csv."""
    df = pd.read_csv(csv_path)

    # Overall statistics
    total_errors = len(df)
    false_positives = len(df[df["error_type"] == "False Positive"])
    false_negatives = len(df[df["error_type"] == "False Negative"])

    # Error category distribution
    category_counts = df["error_category"].value_counts().to_dict()

    # Schema distribution
    schema_counts = df["schema"].value_counts().to_dict()

    # Script analysis
    script_counts = df["query_script"].value_counts().to_dict()
    cross_script_errors = df["cross_script"].sum()

    # Language analysis
    language_counts = df["query_language"].value_counts().to_dict()

    # Score statistics by category
    score_stats_by_category = {}
    for category in df["error_category"].unique():
        cat_df = df[df["error_category"] == category]
        score_stats_by_category[category] = {
            "count": len(cat_df),
            "mean_score": cat_df["similarity_score"].mean(),
            "min_score": cat_df["similarity_score"].min(),
            "max_score": cat_df["similarity_score"].max(),
            "std_score": cat_df["similarity_score"].std(),
        }

    return {
        "total_errors": total_errors,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "category_counts": category_counts,
        "schema_counts": schema_counts,
        "script_counts": script_counts,
        "cross_script_errors": cross_script_errors,
        "language_counts": language_counts,
        "score_stats_by_category": score_stats_by_category,
    }


def generate_category_report(output_dir: str = "data/evaluation_results") -> str:  # noqa: C901
    """Generate a detailed report about error categories."""

    # Load error analysis
    error_analysis_path = os.path.join(output_dir, "error_analysis.csv")
    if not os.path.exists(error_analysis_path):
        return "Error analysis file not found. Please run evaluation first."

    analysis = analyze_error_categories(error_analysis_path)

    # Build report
    report_lines = []
    report_lines.append("\n" + "=" * 80)
    report_lines.append("ERROR CATEGORY ANALYSIS REPORT")
    report_lines.append("=" * 80)

    # Overall statistics
    report_lines.append(f"\nTotal Errors Analyzed: {analysis['total_errors']}")
    report_lines.append(f"  - False Positives: {analysis['false_positives']}")
    report_lines.append(f"  - False Negatives: {analysis['false_negatives']}")

    # Error categories
    report_lines.append("\n" + "-" * 60)
    report_lines.append("ERROR CATEGORIES (sorted by frequency)")
    report_lines.append("-" * 60)

    total_errors = analysis["total_errors"]
    for category, count in sorted(
        analysis["category_counts"].items(), key=lambda x: x[1], reverse=True
    ):
        percentage = (count / total_errors) * 100
        stats = analysis["score_stats_by_category"][category]
        report_lines.append(f"\n{category}: {count} errors ({percentage:.1f}%)")
        report_lines.append(
            f"  - Similarity scores: mean={stats['mean_score']:.3f}, "
            f"min={stats['min_score']:.3f}, max={stats['max_score']:.3f}"
        )

        # Add interpretation
        if category == "Other":
            report_lines.append("  - These are general mismatches without specific patterns")
        elif category == "Company Type Mismatch":
            report_lines.append("  - Variations in company suffixes (AG, GmbH, Ltd, Inc, etc.)")
        elif category == "Initial vs Full Name":
            report_lines.append("  - Names with initials vs spelled out names")
        elif category == "Subset/Superset":
            report_lines.append("  - One name contains the other as a substring")
        elif category == "Title/Prefix Variation":
            report_lines.append("  - Names with titles (Mr., Dr., etc.) or prefixes")
        elif category == "Spacing Difference":
            report_lines.append("  - Same characters but different spacing")

    # Entity type distribution
    report_lines.append("\n" + "-" * 60)
    report_lines.append("ERROR DISTRIBUTION BY ENTITY TYPE")
    report_lines.append("-" * 60)

    for schema, count in sorted(
        analysis["schema_counts"].items(), key=lambda x: x[1], reverse=True
    ):
        percentage = (count / total_errors) * 100
        report_lines.append(f"{schema}: {count} errors ({percentage:.1f}%)")

    # Script analysis
    report_lines.append("\n" + "-" * 60)
    report_lines.append("ERROR DISTRIBUTION BY SCRIPT/CHARACTER SET")
    report_lines.append("-" * 60)

    for script, count in sorted(
        analysis["script_counts"].items(), key=lambda x: x[1], reverse=True
    ):
        percentage = (count / total_errors) * 100
        report_lines.append(f"{script}: {count} errors ({percentage:.1f}%)")

    report_lines.append(
        f"\nCross-script errors: {analysis['cross_script_errors']} "
        f"({(analysis['cross_script_errors'] / total_errors) * 100:.1f}%)"
    )
    report_lines.append(
        "  - These are particularly challenging as they involve different writing systems"
    )

    # Language analysis
    report_lines.append("\n" + "-" * 60)
    report_lines.append("ERROR DISTRIBUTION BY LANGUAGE (heuristic)")
    report_lines.append("-" * 60)

    for language, count in sorted(
        analysis["language_counts"].items(), key=lambda x: x[1], reverse=True
    ):
        percentage = (count / total_errors) * 100
        report_lines.append(f"{language}: {count} errors ({percentage:.1f}%)")

    # Key findings and recommendations
    report_lines.append("\n" + "=" * 80)
    report_lines.append("KEY FINDINGS AND RECOMMENDATIONS")
    report_lines.append("=" * 80)

    # Find most problematic category
    top_category = list(analysis["category_counts"].items())[0]
    if top_category[0] == "Other" and len(analysis["category_counts"]) > 1:
        top_category = list(analysis["category_counts"].items())[1]

    report_lines.append(
        f"\n1. Most common specific error pattern: {top_category[0]} "
        f"({top_category[1]} cases, {(top_category[1] / total_errors) * 100:.1f}%)"
    )

    # Check if cross-script is significant
    if analysis["cross_script_errors"] > total_errors * 0.05:
        report_lines.append(
            f"\n2. Cross-script matching is a significant challenge "
            f"({analysis['cross_script_errors']} cases)"
        )
        report_lines.append("   Recommendation: Consider script-specific preprocessing or features")

    # Check entity type distribution
    if "Person" in analysis["schema_counts"] and "Company" in analysis["schema_counts"]:
        person_pct = analysis["schema_counts"]["Person"] / total_errors
        company_pct = analysis["schema_counts"]["Company"] / total_errors
        if abs(person_pct - company_pct) > 0.3:
            dominant = "Person" if person_pct > company_pct else "Company"
            report_lines.append(
                f"\n3. {dominant} entities show more errors ({analysis['schema_counts'][dominant]} cases)"
            )
            report_lines.append(
                f"   Recommendation: Focus optimization on {dominant} entity matching"
            )

    # Score threshold recommendations
    report_lines.append("\n4. Similarity Score Analysis:")
    high_score_errors = 0
    for category, stats in analysis["score_stats_by_category"].items():
        if stats["mean_score"] > 0.9:
            high_score_errors += stats["count"]

    if high_score_errors > total_errors * 0.3:
        report_lines.append(f"   - {high_score_errors} errors have high similarity scores (>0.9)")
        report_lines.append(
            "   Recommendation: Current threshold may be too low; consider increasing it"
        )

    # Category-specific recommendations
    if (
        "Company Type Mismatch" in analysis["category_counts"]
        and analysis["category_counts"]["Company Type Mismatch"] > 5
    ):
        report_lines.append("\n5. Company suffix variations are causing many false positives")
        report_lines.append(
            "   Recommendation: Normalize company suffixes (AG→Aktiengesellschaft, Ltd→Limited)"
        )

    if (
        "Initial vs Full Name" in analysis["category_counts"]
        and analysis["category_counts"]["Initial vs Full Name"] > 3
    ):
        report_lines.append("\n6. Initial matching is problematic")
        report_lines.append("   Recommendation: Implement special handling for initials")

    report_text = "\n".join(report_lines)
    return report_text


def append_category_report_to_evaluation(output_dir: str = "data/evaluation_results") -> None:
    """Append category analysis to the standard evaluation output."""
    report = generate_category_report(output_dir)
    print(report)

    # Also save to file
    with open(os.path.join(output_dir, "category_analysis_report.txt"), "w") as f:
        f.write(report)
    print(f"\nCategory analysis saved to: {output_dir}/category_analysis_report.txt")


if __name__ == "__main__":
    append_category_report_to_evaluation()
