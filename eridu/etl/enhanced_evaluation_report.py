"""Enhanced evaluation report generation for entity matching checks."""

import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

import pandas as pd

from eridu.analyze_checks_errors import (
    categorize_error_type,
    detect_language_heuristic,
    detect_script,
)
from eridu.etl.checks_evaluation import (
    _categorize_results,
    evaluate_checks,
    filter_checks_by_schema,
    load_checks_yaml,
)
from eridu.etl.evaluate import load_model


def generate_enhanced_report(
    checks_path: str,
    model_path: Optional[str] = None,
    use_gpu: bool = True,
    threshold: float = 0.5,
    entity_type: str = "both",
    output_dir: str = "data/evaluation_results",
) -> None:
    """Generate an enhanced evaluation report with detailed error analysis.

    Args:
        checks_path: Path to the checks.yml file
        model_path: Path to the SBERT model (None for default)
        use_gpu: Whether to use GPU acceleration
        threshold: Classification threshold for binary predictions
        entity_type: Entity type to evaluate ("person", "company", or "both")
        output_dir: Directory to save report and analysis files
    """
    print(f"Loading checks from: {checks_path}")

    # Load checks
    try:
        checks = load_checks_yaml(checks_path)
        print(f"Loaded {len(checks)} total checks")
    except Exception as e:
        print(f"Error loading checks.yml: {e}")
        return

    # Load model
    try:
        model = load_model(model_path, use_gpu)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Filter checks by schema based on entity_type
    if entity_type == "person":
        target_checks = filter_checks_by_schema(checks, ["Person"])
        print(f"Found {len(target_checks)} Person checks")
    elif entity_type == "company":
        target_checks = filter_checks_by_schema(checks, ["Company"])
        print(f"Found {len(target_checks)} Company checks")
    elif entity_type == "address":
        target_checks = filter_checks_by_schema(checks, ["Address", "Location"])
        print(f"Found {len(target_checks)} Address checks")
    else:  # both
        person_checks = filter_checks_by_schema(checks, ["Person"])
        company_checks = filter_checks_by_schema(checks, ["Company"])
        org_checks = filter_checks_by_schema(checks, ["Organization"])
        address_checks = filter_checks_by_schema(checks, ["Address", "Location"])
        target_checks = person_checks + company_checks + org_checks + address_checks
        print(f"Found {len(person_checks)} Person checks")
        print(f"Found {len(company_checks)} Company checks")
        print(f"Found {len(org_checks)} Organization checks")
        print(f"Found {len(address_checks)} Address checks")

    # Evaluate
    results = evaluate_checks(target_checks, model, use_gpu, threshold)
    results_list = results["results"]

    if not isinstance(results_list, list):
        print("No results to analyze")
        return

    # Categorize results
    true_positives, false_positives, true_negatives, false_negatives = _categorize_results(
        results_list
    )

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate comprehensive HTML report
    generate_html_report(
        results,
        true_positives,
        false_positives,
        true_negatives,
        false_negatives,
        output_dir,
        entity_type,
        threshold,
        model_path,
    )

    # Generate detailed error analysis
    analyze_errors_detailed(false_positives, false_negatives, output_dir)

    print(f"\nEnhanced report saved to: {output_dir}/evaluation_report.html")
    print(f"Detailed error analysis saved to: {output_dir}/error_analysis.csv")


def analyze_errors_detailed(
    false_positives: List[Dict[str, Any]],
    false_negatives: List[Dict[str, Any]],
    output_dir: str,
) -> None:
    """Perform detailed error analysis and save to CSV."""
    error_analysis = []

    # Analyze false positives
    for fp in false_positives:
        query = fp["query_name"]
        candidate = fp["candidate_name"]

        analysis = {
            "error_type": "False Positive",
            "query_name": query,
            "candidate_name": candidate,
            "schema": fp["schema"],
            "similarity_score": fp["similarity_score"],
            "label": fp.get("label", ""),
            "query_script": detect_script(query),
            "candidate_script": detect_script(candidate),
            "query_language": detect_language_heuristic(query),
            "candidate_language": detect_language_heuristic(candidate),
            "error_category": categorize_error_type(query, candidate, fp.get("label", "")),
            "cross_script": detect_script(query) != detect_script(candidate),
            "name_length_diff": abs(len(query) - len(candidate)),
            "word_count_diff": abs(len(query.split()) - len(candidate.split())),
        }
        error_analysis.append(analysis)

    # Analyze false negatives
    for fn in false_negatives:
        query = fn["query_name"]
        candidate = fn["candidate_name"]

        analysis = {
            "error_type": "False Negative",
            "query_name": query,
            "candidate_name": candidate,
            "schema": fn["schema"],
            "similarity_score": fn["similarity_score"],
            "label": fn.get("label", ""),
            "query_script": detect_script(query),
            "candidate_script": detect_script(candidate),
            "query_language": detect_language_heuristic(query),
            "candidate_language": detect_language_heuristic(candidate),
            "error_category": categorize_error_type(query, candidate, fn.get("label", "")),
            "cross_script": detect_script(query) != detect_script(candidate),
            "name_length_diff": abs(len(query) - len(candidate)),
            "word_count_diff": abs(len(query.split()) - len(candidate.split())),
        }
        error_analysis.append(analysis)

    # Save to CSV
    if error_analysis:
        df = pd.DataFrame(error_analysis)
        df.to_csv(os.path.join(output_dir, "error_analysis.csv"), index=False)


def generate_html_report(
    results: Dict[str, Any],
    true_positives: List[Dict[str, Any]],
    false_positives: List[Dict[str, Any]],
    true_negatives: List[Dict[str, Any]],
    false_negatives: List[Dict[str, Any]],
    output_dir: str,
    entity_type: str,
    threshold: float,
    model_path: Optional[str],
) -> None:
    """Generate an HTML report with evaluation results and error analysis."""
    # Analyze error patterns
    fp_by_category = defaultdict(list)
    for fp in false_positives:
        category = categorize_error_type(
            fp["query_name"], fp["candidate_name"], fp.get("label", "")
        )
        fp_by_category[category].append(fp)

    fn_by_category = defaultdict(list)
    for fn in false_negatives:
        category = categorize_error_type(
            fn["query_name"], fn["candidate_name"], fn.get("label", "")
        )
        fn_by_category[category].append(fn)

    # Script analysis
    fp_scripts: dict[str, int] = defaultdict(int)
    cross_script_fps = []
    for fp in false_positives:
        query_script = detect_script(fp["query_name"])
        candidate_script = detect_script(fp["candidate_name"])
        fp_scripts[query_script] += 1
        if query_script != candidate_script:
            cross_script_fps.append(fp)

    # Generate HTML
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Entity Matching Evaluation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2, h3 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .metric {{ font-size: 1.2em; margin: 10px 0; }}
        .good {{ color: green; }}
        .bad {{ color: red; }}
        .section {{ margin: 30px 0; }}
        .example {{ background-color: #f9f9f9; padding: 10px; margin: 10px 0; border-radius: 5px; }}
        .category-section {{ margin: 20px 0; padding: 15px; background-color: #f5f5f5; border-radius: 8px; }}
    </style>
</head>
<body>
    <h1>Entity Matching Evaluation Report</h1>
    <p>Entity Type: <strong>{entity_type.upper()}</strong></p>
    <p>Model: <strong>{model_path or "Default SBERT model"}</strong></p>
    <p>Threshold: <strong>{threshold}</strong></p>

    <div class="section">
        <h2>Overall Performance Metrics</h2>
        <div class="metric">Total Checks: <strong>{results['total_checks']}</strong></div>
        <div class="metric">Accuracy: <strong>{results['accuracy']:.4f}</strong></div>
        <div class="metric">Precision: <strong>{results['precision']:.4f}</strong></div>
        <div class="metric">Recall: <strong>{results['recall']:.4f}</strong></div>
        <div class="metric">F1 Score: <strong>{results['f1']:.4f}</strong></div>
    </div>

    <div class="section">
        <h2>Confusion Matrix</h2>
        <table>
            <tr>
                <th></th>
                <th>Predicted Match</th>
                <th>Predicted Non-Match</th>
            </tr>
            <tr>
                <th>Actual Match</th>
                <td class="good">{results['true_positives']}</td>
                <td class="bad">{results['false_negatives']}</td>
            </tr>
            <tr>
                <th>Actual Non-Match</th>
                <td class="bad">{results['false_positives']}</td>
                <td class="good">{results['true_negatives']}</td>
            </tr>
        </table>
    </div>

    <div class="section">
        <h2>Error Analysis by Category</h2>
        <h3>False Positives ({len(false_positives)} total)</h3>
        {"".join([f'''
        <div class="category-section">
            <h4>{category} ({len(examples)} cases)</h4>
            <table>
                <tr>
                    <th>Query</th>
                    <th>Candidate</th>
                    <th>Score</th>
                    <th>Schema</th>
                </tr>
                {"".join([f'''
                <tr>
                    <td>{ex["query_name"]}</td>
                    <td>{ex["candidate_name"]}</td>
                    <td>{ex["similarity_score"]:.4f}</td>
                    <td>{ex["schema"]}</td>
                </tr>
                ''' for ex in examples[:3]])}
            </table>
        </div>
        ''' for category, examples in sorted(fp_by_category.items(), key=lambda x: len(x[1]), reverse=True)[:5]])}
    </div>

    <div class="section">
        <h2>Script and Language Analysis</h2>
        <h3>False Positives by Script</h3>
        <table>
            <tr>
                <th>Script</th>
                <th>Count</th>
                <th>Percentage</th>
            </tr>
            {"".join([f'''
            <tr>
                <td>{script}</td>
                <td>{count}</td>
                <td>{count / len(false_positives) * 100:.1f}%</td>
            </tr>
            ''' for script, count in sorted(fp_scripts.items(), key=lambda x: x[1], reverse=True)])}
        </table>

        <h3>Cross-Script False Positives ({len(cross_script_fps)} total)</h3>
        {f'''
        <table>
            <tr>
                <th>Query</th>
                <th>Query Script</th>
                <th>Candidate</th>
                <th>Candidate Script</th>
                <th>Score</th>
            </tr>
            {"".join([f'''
            <tr>
                <td>{ex["query_name"]}</td>
                <td>{detect_script(ex["query_name"])}</td>
                <td>{ex["candidate_name"]}</td>
                <td>{detect_script(ex["candidate_name"])}</td>
                <td>{ex["similarity_score"]:.4f}</td>
            </tr>
            ''' for ex in cross_script_fps[:10]])}
        </table>
        ''' if cross_script_fps else "<p>No cross-script false positives found.</p>"}
    </div>

    <div class="section">
        <h2>Recommendations</h2>
        <ul>
            <li>Consider adjusting the threshold based on the error patterns observed</li>
            <li>Cross-script matching needs special attention</li>
            <li>Company type variations (AG, GmbH, Ltd, etc.) are a common source of errors</li>
            <li>Name variations (initials, titles, word order) require specific handling</li>
        </ul>
    </div>
</body>
</html>
"""

    # Save HTML report
    with open(os.path.join(output_dir, "evaluation_report.html"), "w", encoding="utf-8") as f:
        f.write(html_content)


if __name__ == "__main__":
    # Example usage
    generate_enhanced_report(
        checks_path="./data/checks.yml",
        entity_type="both",
        output_dir="data/evaluation_results",
    )
