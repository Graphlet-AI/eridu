"""Module for evaluating entity matching performance using checks.yml test cases."""

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (  # type: ignore
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from eridu.etl.evaluate import load_model
from eridu.train.utils import sbert_compare_multiple


def load_checks_yaml(checks_path: str) -> List[Dict[str, Any]]:
    """Load and parse the checks.yml file.

    Args:
        checks_path: Path to the checks.yml file

    Returns:
        List of check dictionaries containing match, schema, query, and candidate
    """
    with open(checks_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data.get("checks", [])  # type: ignore


def filter_checks_by_schema(
    checks: List[Dict[str, Any]], schemas: List[str]
) -> List[Dict[str, Any]]:
    """Filter checks by schema type.

    Args:
        checks: List of check dictionaries
        schemas: List of schema types to filter by (e.g., ['Person', 'Company'])

    Returns:
        Filtered list of checks
    """
    return [check for check in checks if check.get("schema") in schemas]


def extract_names_from_check(check: Dict[str, Any]) -> Tuple[str, str]:
    """Extract name strings from query and candidate objects.

    Args:
        check: Check dictionary with query and candidate

    Returns:
        Tuple of (query_name, candidate_name)
    """
    query = check.get("query", {})
    candidate = check.get("candidate", {})

    # Extract name field
    query_name = query.get("name", "")
    candidate_name = candidate.get("name", "")

    return query_name, candidate_name


def evaluate_checks(
    checks: List[Dict[str, Any]],
    model: SentenceTransformer,
    use_gpu: bool = True,
    threshold: float = 0.5,
) -> Dict[str, Union[float, int, List[Dict[str, Any]]]]:
    """Evaluate model performance on check test cases.

    Args:
        checks: List of check dictionaries
        model: SBERT model for evaluation
        use_gpu: Whether to use GPU acceleration
        threshold: Classification threshold for binary predictions

    Returns:
        Dictionary with evaluation metrics and detailed results
    """
    if not checks:
        return {
            "total_checks": 0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "true_positives": 0,
            "false_positives": 0,
            "true_negatives": 0,
            "false_negatives": 0,
            "results": [],
        }

    # Extract names and ground truth labels
    query_names = []
    candidate_names = []
    ground_truth = []

    for check in checks:
        query_name, candidate_name = extract_names_from_check(check)
        if query_name and candidate_name:  # Only include if both names exist
            query_names.append(query_name)
            candidate_names.append(candidate_name)
            ground_truth.append(check.get("match", False))

    if not query_names:
        return {
            "total_checks": len(checks),
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "true_positives": 0,
            "false_positives": 0,
            "true_negatives": 0,
            "false_negatives": 0,
            "results": [],
        }

    # Get similarity scores from model
    similarity_scores = sbert_compare_multiple(model, query_names, candidate_names, use_gpu=use_gpu)

    # Make binary predictions
    predictions = similarity_scores >= threshold

    # Convert to numpy arrays for sklearn
    y_true = np.array(ground_truth, dtype=int)
    y_pred = np.array(predictions, dtype=int)

    # Calculate metrics using sklearn
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Get confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Create detailed results
    results: List[Dict[str, Any]] = []
    for i, check in enumerate(checks):
        if i < len(similarity_scores):
            results.append(
                {
                    "match": check.get("match", False),
                    "schema": check.get("schema", ""),
                    "query_name": query_names[i] if i < len(query_names) else "",
                    "candidate_name": candidate_names[i] if i < len(candidate_names) else "",
                    "similarity_score": float(similarity_scores[i]),
                    "predicted_match": bool(predictions[i]),
                    "correct": (
                        ground_truth[i] == predictions[i] if i < len(ground_truth) else False
                    ),
                    "label": check.get("label", ""),
                }
            )

    return {
        "total_checks": len(checks),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "results": results,
    }


def _categorize_results(
    results: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Categorize results into TP, FP, TN, FN."""
    true_positives = []
    false_positives = []
    true_negatives = []
    false_negatives = []

    for result in results:
        ground_truth = result["match"]
        predicted = result["predicted_match"]

        if ground_truth and predicted:
            true_positives.append(result)
        elif not ground_truth and predicted:
            false_positives.append(result)
        elif not ground_truth and not predicted:
            true_negatives.append(result)
        elif ground_truth and not predicted:
            false_negatives.append(result)

    return true_positives, false_positives, true_negatives, false_negatives


def _create_example_df(examples: List[Dict[str, Any]], limit: int) -> pd.DataFrame:
    """Create DataFrame from examples."""
    if not examples:
        return pd.DataFrame(columns=["Query Name", "Candidate Name", "Score", "Label"])

    limited_examples = examples[:limit]
    df_data = []
    for ex in limited_examples:
        query_name = ex["query_name"]
        candidate_name = ex["candidate_name"]
        label = ex.get("label", "")

        df_data.append(
            {
                "Query Name": query_name[:50] + "..." if len(query_name) > 50 else query_name,
                "Candidate Name": (
                    candidate_name[:50] + "..." if len(candidate_name) > 50 else candidate_name
                ),
                "Score": f"{ex['similarity_score']:.4f}",
                "Label": label[:30] + "..." if len(label) > 30 else label,
            }
        )
    return pd.DataFrame(df_data)


def display_examples_tables(results: List[Dict[str, Any]], n_examples: int = 10) -> None:
    """Display examples of true/false positives and true/false negatives in pandas-style tables.

    Args:
        results: List of evaluation result dictionaries
        n_examples: Number of examples to show for each category
    """
    true_positives, false_positives, true_negatives, false_negatives = _categorize_results(results)

    # Display tables for each category
    print(f"\n{'=' * 80}")
    print("CLASSIFICATION EXAMPLES")
    print(f"{'=' * 80}")

    print(f"\n🟢 TRUE POSITIVES ({len(true_positives)} total, showing up to {n_examples}):")
    print("   Correctly identified as matches")
    tp_df = _create_example_df(true_positives, n_examples)
    if not tp_df.empty:
        print(tp_df.to_string(index=False, max_colwidth=50))
    else:
        print("   No true positives found.")

    print(f"\n🔴 FALSE POSITIVES ({len(false_positives)} total, showing up to {n_examples}):")
    print("   Incorrectly identified as matches")
    fp_df = _create_example_df(false_positives, n_examples)
    if not fp_df.empty:
        print(fp_df.to_string(index=False, max_colwidth=50))
    else:
        print("   No false positives found.")

    print(f"\n🟢 TRUE NEGATIVES ({len(true_negatives)} total, showing up to {n_examples}):")
    print("   Correctly identified as non-matches")
    tn_df = _create_example_df(true_negatives, n_examples)
    if not tn_df.empty:
        print(tn_df.to_string(index=False, max_colwidth=50))
    else:
        print("   No true negatives found.")

    print(f"\n🔴 FALSE NEGATIVES ({len(false_negatives)} total, showing up to {n_examples}):")
    print("   Incorrectly identified as non-matches")
    fn_df = _create_example_df(false_negatives, n_examples)
    if not fn_df.empty:
        print(fn_df.to_string(index=False, max_colwidth=50))
    else:
        print("   No false negatives found.")


def save_results_to_csv(
    results: List[Dict[str, Any]],
    output_dir: str = "data/evaluation_results",
) -> None:
    """Save categorized results to CSV files.

    Args:
        results: List of evaluation result dictionaries
        output_dir: Directory to save CSV files to
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Categorize results
    true_positives, false_positives, true_negatives, false_negatives = _categorize_results(results)

    # Prepare data for CSV export
    def prepare_for_csv(results_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        csv_data = []
        for result in results_list:
            csv_data.append(
                {
                    "query_name": result["query_name"],
                    "candidate_name": result["candidate_name"],
                    "schema": result["schema"],
                    "similarity_score": result["similarity_score"],
                    "ground_truth_match": result["match"],
                    "predicted_match": result["predicted_match"],
                    "label": result.get("label", ""),
                }
            )
        return csv_data

    # Save each category to CSV
    if true_positives:
        tp_df = pd.DataFrame(prepare_for_csv(true_positives))
        tp_df.to_csv(os.path.join(output_dir, "true_positives.csv"), index=False)
        print(f"Saved {len(true_positives)} true positives to {output_dir}/true_positives.csv")

    if false_positives:
        fp_df = pd.DataFrame(prepare_for_csv(false_positives))
        fp_df.to_csv(os.path.join(output_dir, "false_positives.csv"), index=False)
        print(f"Saved {len(false_positives)} false positives to {output_dir}/false_positives.csv")

    if true_negatives:
        tn_df = pd.DataFrame(prepare_for_csv(true_negatives))
        tn_df.to_csv(os.path.join(output_dir, "true_negatives.csv"), index=False)
        print(f"Saved {len(true_negatives)} true negatives to {output_dir}/true_negatives.csv")

    if false_negatives:
        fn_df = pd.DataFrame(prepare_for_csv(false_negatives))
        fn_df.to_csv(os.path.join(output_dir, "false_negatives.csv"), index=False)
        print(f"Saved {len(false_negatives)} false negatives to {output_dir}/false_negatives.csv")

    # Save all results to a single CSV for convenience
    all_results_df = pd.DataFrame(prepare_for_csv(results))
    all_results_df.to_csv(os.path.join(output_dir, "all_results.csv"), index=False)
    print(f"Saved all {len(results)} results to {output_dir}/all_results.csv")


def generate_checks_report(  # noqa: C901
    checks_path: str,
    entity_type: str,
    model_path: Optional[str] = None,
    use_gpu: bool = True,
    threshold: float = 0.5,
    save_csv: bool = False,
    output_dir: str = "data/evaluation_results",
) -> None:
    """Generate a comprehensive evaluation report using checks.yml test cases.

    Args:
        checks_path: Path to the checks.yml file
        model_path: Path to the SBERT model (None for default)
        use_gpu: Whether to use GPU acceleration
        threshold: Classification threshold for binary predictions
        entity_type: Entity type to evaluate ("person", "company", or "both")
        save_csv: Whether to save results to CSV files
        output_dir: Directory to save CSV files to
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
        eval_title = "PERSON ENTITY MATCHING EVALUATION"
    elif entity_type == "company":
        target_checks = filter_checks_by_schema(checks, ["Company"])
        print(f"Found {len(target_checks)} Company checks")
        eval_title = "COMPANY ENTITY MATCHING EVALUATION"
    elif entity_type == "address":
        target_checks = filter_checks_by_schema(checks, ["Address", "Location"])
        print(f"Found {len(target_checks)} Address checks")
        eval_title = "ADDRESS ENTITY MATCHING EVALUATION"
    else:  # both
        person_checks = filter_checks_by_schema(checks, ["Person"])
        company_checks = filter_checks_by_schema(checks, ["Company"])
        address_checks = filter_checks_by_schema(checks, ["Address", "Location"])
        target_checks = person_checks + company_checks + address_checks
        print(f"Found {len(person_checks)} Person checks")
        print(f"Found {len(company_checks)} Company checks")
        print(f"Found {len(address_checks)} Address checks")
        eval_title = "COMBINED ENTITY MATCHING EVALUATION"

    # Evaluate the target checks
    print("\n" + "=" * 60)
    print(eval_title)
    print("=" * 60)

    results = evaluate_checks(target_checks, model, use_gpu, threshold)

    print(f"Total checks: {results['total_checks']}")
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1 Score:  {results['f1']:.4f}")
    print(f"True Positives:  {results['true_positives']}")
    print(f"False Positives: {results['false_positives']}")
    print(f"True Negatives:  {results['true_negatives']}")
    print(f"False Negatives: {results['false_negatives']}")

    # Display examples in categorized tables
    results_list = results["results"]
    if isinstance(results_list, list):
        display_examples_tables(results_list, n_examples=10)

    # Save results to CSV if requested
    if save_csv and isinstance(results_list, list):
        print(f"\n{'=' * 80}")
        print("SAVING RESULTS TO CSV FILES")
        print(f"{'=' * 80}")
        save_results_to_csv(results_list, output_dir)

        # Also run enhanced analysis
        from eridu.etl.enhanced_evaluation_report import analyze_errors_detailed

        true_positives, false_positives, true_negatives, false_negatives = _categorize_results(
            results_list
        )
        analyze_errors_detailed(false_positives, false_negatives, output_dir)

    print(f"\n{'=' * 80}")
    print("EVALUATION SUMMARY")
    print(f"{'=' * 80}")
    print(f"Classification threshold used: {threshold:.4f}")
    print(f"Model evaluated: {model_path or 'Default SBERT model'}")
    print(f"Entity type: {entity_type}")

    # Add category analysis if CSV files were saved
    if save_csv and os.path.exists(os.path.join(output_dir, "error_analysis.csv")):
        from eridu.etl.error_category_analysis import generate_category_report

        category_report = generate_category_report(output_dir)
        print(category_report)
