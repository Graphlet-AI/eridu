"""Module for evaluating entity matching performance using checks.yml test cases."""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
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


def generate_checks_report(  # noqa: C901
    checks_path: str,
    model_path: Optional[str] = None,
    use_gpu: bool = True,
    threshold: float = 0.5,
    entity_type: str = "company",
) -> None:
    """Generate a comprehensive evaluation report using checks.yml test cases.

    Args:
        checks_path: Path to the checks.yml file
        model_path: Path to the SBERT model (None for default)
        use_gpu: Whether to use GPU acceleration
        threshold: Classification threshold for binary predictions
        entity_type: Entity type to evaluate ("person", "company", or "both")
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
    else:  # both
        person_checks = filter_checks_by_schema(checks, ["Person"])
        company_checks = filter_checks_by_schema(checks, ["Company"])
        target_checks = person_checks + company_checks
        print(f"Found {len(person_checks)} Person checks")
        print(f"Found {len(company_checks)} Company checks")
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

    # Show some error examples
    print(f"\n{entity_type.title()} Error Examples:")
    results_list = results["results"]
    if isinstance(results_list, list):
        errors = [r for r in results_list if not r["correct"]]
        for i, error in enumerate(errors[:5]):
            print(f"  {i + 1}. '{error['query_name']}' vs '{error['candidate_name']}'")
            print(
                f"     Ground truth: {error['match']}, Predicted: {error['predicted_match']}, Score: {error['similarity_score']:.4f}"
            )
            if error["label"]:
                print(f"     Label: {error['label']}")

    print(f"\nClassification threshold used: {threshold:.4f}")
    print(f"Model evaluated: {model_path or 'Default SBERT model'}")
    print(f"Entity type: {entity_type}")
