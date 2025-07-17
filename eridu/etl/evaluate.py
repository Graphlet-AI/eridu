"""Evaluation module for SBERT models."""

import os
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (  # type: ignore
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from eridu.train.utils import sbert_compare_multiple

# Constants from fine_tune_sbert.py
SBERT_MODEL: str = os.environ.get(
    "SBERT_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
VARIANT: str = os.environ.get("VARIANT", "original")
OPTIMIZER: str = os.environ.get("OPTIMIZER", "adafactor")
DATA_TYPE: str = os.environ.get("DATA_TYPE", "companies")
MODEL_SAVE_NAME: str = (SBERT_MODEL + "-" + VARIANT + "-" + OPTIMIZER).replace("/", "-")
SBERT_OUTPUT_FOLDER: str = f"data/fine-tuned-sbert-{MODEL_SAVE_NAME}-{DATA_TYPE}"

# Fallback model from HuggingFace Hub
FALLBACK_MODEL: str = "Graphlet-AI/eridu"


def get_device(use_gpu: bool = True) -> str:
    """Determine the appropriate device for inference.

    Args:
        use_gpu: Whether to attempt to use GPU acceleration

    Returns:
        Device type string: 'cuda', 'mps', or 'cpu'
    """
    if not use_gpu:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def load_model(model_path: Optional[str] = None, use_gpu: bool = True) -> SentenceTransformer:
    """Load a fine-tuned SentenceTransformer model.

    Args:
        model_path: Path to the model directory, or None to use the default path
        use_gpu: Whether to use GPU acceleration if available

    Returns:
        The loaded model (either from disk or fallback to HuggingFace Hub)
    """
    # Use default path if none provided
    if model_path is None:
        model_path = SBERT_OUTPUT_FOLDER

    # Determine device
    device = get_device(use_gpu)
    print(f"Using device: {device}")

    # First, try to load from disk
    if os.path.exists(model_path):
        try:
            print(f"Loading model from: {model_path}")
            model = SentenceTransformer(model_path, device=device)
            print(f"Successfully loaded model from {model_path}")
            return model
        except Exception as e:
            print(f"Error loading model from disk: {e}")
            # Continue to fallback
    else:
        print(f"Model not found at: {model_path}")

        # Try without entity type suffix (for backward compatibility)
        base_path = (
            model_path.rsplit("-", 1)[0]
            if model_path.endswith(("-companies", "-people", "-addresses"))
            else None
        )
        if base_path and os.path.exists(base_path):
            try:
                print(f"Trying model path without entity type suffix: {base_path}")
                model = SentenceTransformer(base_path, device=device)
                print(f"Successfully loaded model from {base_path}")
                return model
            except Exception as e:
                print(f"Error loading model from {base_path}: {e}")

    # Fallback to HuggingFace Hub
    print(
        f"Could not load model from {model_path}. "
        f"Falling back to HuggingFace Hub model: {FALLBACK_MODEL}"
    )

    try:
        print(f"Loading model from HuggingFace Hub: {FALLBACK_MODEL}")
        model = SentenceTransformer(FALLBACK_MODEL, device=device)
        print("Successfully loaded model from HuggingFace Hub")
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load both local model and HuggingFace fallback: {e}")


def load_test_data(model_path: Optional[str] = None) -> pd.DataFrame:
    """Load the test split for evaluation.

    Args:
        model_path: Path to model directory containing test data, or None to use default

    Returns:
        DataFrame containing the test data
    """
    # Determine model path
    if model_path is None:
        model_path = SBERT_OUTPUT_FOLDER

    # Look for test_split.parquet
    test_path = os.path.join(model_path, "test_split.parquet")
    if os.path.exists(test_path):
        print(f"Loading test data from: {test_path}")
        return pd.read_parquet(test_path)

    # If not found, try test_results.parquet (which contains only a sample)
    results_path = os.path.join(model_path, "test_results.parquet")
    if os.path.exists(results_path):
        print(f"Test split not found, using test results from: {results_path}")
        return pd.read_parquet(results_path)

    # Try without entity type suffix (for backward compatibility)
    base_path = (
        model_path.rsplit("-", 1)[0]
        if model_path.endswith(("-companies", "-people", "-addresses"))
        else None
    )
    if base_path:
        test_path_base = os.path.join(base_path, "test_split.parquet")
        if os.path.exists(test_path_base):
            print(f"Loading test data from base path: {test_path_base}")
            return pd.read_parquet(test_path_base)

        results_path_base = os.path.join(base_path, "test_results.parquet")
        if os.path.exists(results_path_base):
            print(f"Test split not found, using test results from base path: {results_path_base}")
            return pd.read_parquet(results_path_base)

    # If still not found, raise error
    raise FileNotFoundError(
        f"Could not find test data at {test_path} or {results_path}. "
        f"Make sure to run 'eridu train' first."
    )


def evaluate_model(
    model: SentenceTransformer,
    test_df: pd.DataFrame,
    use_gpu: bool = True,
    threshold: Optional[float] = None,
) -> Dict[str, Union[float, int]]:
    """Evaluate model performance on test data.

    Args:
        model: The SBERT model to evaluate
        test_df: DataFrame containing test data
        use_gpu: Whether to use GPU acceleration
        threshold: Optional threshold for binary classification
                  (if None, finds optimal threshold using F1 score)

    Returns:
        Dictionary of evaluation metrics
    """
    print(f"Running inference on {len(test_df):,} test pairs")

    # Get similarity scores using the model
    similarity_scores = sbert_compare_multiple(
        model, test_df["left_name"], test_df["right_name"], use_gpu=use_gpu
    )

    # Prepare ground truth
    if "match" in test_df.columns:
        y_true = test_df["match"].astype(float).values
    elif "true_label" in test_df.columns:
        y_true = test_df["true_label"].values
    else:
        raise ValueError("Test data must contain either 'match' or 'true_label' column")

    # Find optimal threshold if not provided
    if threshold is None:
        precision, recall, thresholds = precision_recall_curve(y_true, similarity_scores)
        f1_scores = [f1_score(y_true, similarity_scores >= t) for t in thresholds]
        best_idx = np.argmax(f1_scores)
        threshold = thresholds[best_idx]
        print(f"Found optimal threshold: {threshold:.4f}")

    # Binary predictions using threshold
    y_pred = similarity_scores >= threshold

    # Calculate metrics
    metrics = {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "auc_roc": roc_auc_score(y_true, similarity_scores),
        "count": len(test_df),
        "true_positives": int(np.sum((y_true == 1) & y_pred)),
        "false_positives": int(np.sum((y_true == 0) & y_pred)),
        "true_negatives": int(np.sum((y_true == 0) & ~y_pred)),
        "false_negatives": int(np.sum((y_true == 1) & ~y_pred)),
    }

    return metrics


def _categorize_test_results(
    test_df: pd.DataFrame, similarity_scores: np.ndarray, threshold: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Categorize test results into TP, FP, TN, FN."""
    # Create results dataframe
    results_df = test_df.copy()
    results_df["similarity"] = similarity_scores

    # Get ground truth
    if "match" in test_df.columns:
        results_df["true_label"] = test_df["match"].astype(float)
    elif "true_label" in test_df.columns:
        results_df["true_label"] = test_df["true_label"]
    else:
        raise ValueError("Test data must contain either 'match' or 'true_label' column")

    # Get predictions
    results_df["predicted_match"] = results_df["similarity"] >= threshold

    # Categorize results
    true_positives = results_df[
        (results_df["true_label"] == 1) & results_df["predicted_match"]
    ].sort_values("similarity", ascending=False)

    false_positives = results_df[
        (results_df["true_label"] == 0) & results_df["predicted_match"]
    ].sort_values("similarity", ascending=False)

    true_negatives = results_df[
        (results_df["true_label"] == 0) & ~results_df["predicted_match"]
    ].sort_values("similarity", ascending=True)

    false_negatives = results_df[
        (results_df["true_label"] == 1) & ~results_df["predicted_match"]
    ].sort_values("similarity", ascending=False)

    return true_positives, false_positives, true_negatives, false_negatives


def _create_test_example_df(examples: pd.DataFrame, limit: int) -> pd.DataFrame:
    """Create DataFrame from test examples."""
    if examples.empty:
        return pd.DataFrame(columns=["Left Name", "Right Name", "Score"])

    limited_examples = examples.head(limit)
    df_data = []
    for _, row in limited_examples.iterrows():
        left_name = str(row["left_name"])
        right_name = str(row["right_name"])

        df_data.append(
            {
                "Left Name": left_name[:50] + "..." if len(left_name) > 50 else left_name,
                "Right Name": right_name[:50] + "..." if len(right_name) > 50 else right_name,
                "Score": f"{row['similarity']:.4f}",
            }
        )
    return pd.DataFrame(df_data)


def display_test_examples_tables(
    test_df: pd.DataFrame, similarity_scores: np.ndarray, threshold: float, n_examples: int = 10
) -> None:
    """Display examples of true/false positives and true/false negatives in pandas-style tables.

    Args:
        test_df: Test data DataFrame
        similarity_scores: Array of similarity scores
        threshold: Classification threshold
        n_examples: Number of examples to show for each category
    """
    true_positives, false_positives, true_negatives, false_negatives = _categorize_test_results(
        test_df, similarity_scores, threshold
    )

    # Display tables for each category
    print(f"\n{'=' * 80}")
    print("CLASSIFICATION EXAMPLES")
    print(f"{'=' * 80}")

    print(f"\n🟢 TRUE POSITIVES ({len(true_positives)} total, showing up to {n_examples}):")
    print("   Correctly identified as matches")
    tp_df = _create_test_example_df(true_positives, n_examples)
    if not tp_df.empty:
        print(tp_df.to_string(index=False, max_colwidth=50))
    else:
        print("   No true positives found.")

    print(f"\n🔴 FALSE POSITIVES ({len(false_positives)} total, showing up to {n_examples}):")
    print("   Incorrectly identified as matches")
    fp_df = _create_test_example_df(false_positives, n_examples)
    if not fp_df.empty:
        print(fp_df.to_string(index=False, max_colwidth=50))
    else:
        print("   No false positives found.")

    print(f"\n🟢 TRUE NEGATIVES ({len(true_negatives)} total, showing up to {n_examples}):")
    print("   Correctly identified as non-matches")
    tn_df = _create_test_example_df(true_negatives, n_examples)
    if not tn_df.empty:
        print(tn_df.to_string(index=False, max_colwidth=50))
    else:
        print("   No true negatives found.")

    print(f"\n🔴 FALSE NEGATIVES ({len(false_negatives)} total, showing up to {n_examples}):")
    print("   Incorrectly identified as non-matches")
    fn_df = _create_test_example_df(false_negatives, n_examples)
    if not fn_df.empty:
        print(fn_df.to_string(index=False, max_colwidth=50))
    else:
        print("   No false negatives found.")


def analyze_errors(
    test_df: pd.DataFrame, similarity_scores: np.ndarray, threshold: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Analyze model errors to identify patterns.

    Args:
        test_df: Test data
        similarity_scores: Similarity scores from model
        threshold: Classification threshold

    Returns:
        Tuple of (false_positives, false_negatives) DataFrames
    """
    # Get categorized results
    _, false_positives, _, false_negatives = _categorize_test_results(
        test_df, similarity_scores, threshold
    )

    # Return the error DataFrames (maintaining backward compatibility)
    return false_positives, false_negatives


def evaluate_and_print_report(
    model_path: Optional[str] = None,
    use_gpu: bool = True,
    threshold: Optional[float] = None,
    sample_size: Optional[int] = None,
) -> Dict[str, Union[float, int]]:
    """Load model and test data, evaluate model, and print a report.

    Args:
        model_path: Path to model directory, or None to use default
        use_gpu: Whether to use GPU acceleration
        threshold: Optional classification threshold
        sample_size: Optional number of test samples to evaluate (None = use all)

    Returns:
        Dictionary of evaluation metrics
    """
    # Load model
    model = load_model(model_path, use_gpu)

    # Load test data
    test_df = load_test_data(model_path)

    # Sample test data if requested
    if sample_size is not None and len(test_df) > sample_size:
        print(f"Sampling {sample_size:,} test pairs from {len(test_df):,} available")
        test_df = test_df.sample(n=sample_size, random_state=42)

    # Run inference to get similarity scores
    print(f"Running inference on {len(test_df):,} test pairs")
    similarity_scores = sbert_compare_multiple(
        model, test_df["left_name"], test_df["right_name"], use_gpu=use_gpu
    )

    # Prepare ground truth
    if "match" in test_df.columns:
        y_true = test_df["match"].astype(float).values
    elif "true_label" in test_df.columns:
        y_true = test_df["true_label"].values
    else:
        raise ValueError("Test data must contain either 'match' or 'true_label' column")

    # Determine threshold if not provided
    if threshold is None:
        precision, recall, thresholds = precision_recall_curve(y_true, similarity_scores)
        f1_scores = [f1_score(y_true, similarity_scores >= t) for t in thresholds]
        best_idx = np.argmax(f1_scores)
        threshold = thresholds[best_idx]
        print(f"Found optimal threshold: {threshold:.4f}")

    # Evaluate model
    metrics = evaluate_model(model, test_df, use_gpu, threshold)

    # Print report
    print("\nModel Evaluation Report")
    print("=" * 40)
    print(f"Model: {model_path or SBERT_OUTPUT_FOLDER}")
    print(f"Test data: {len(test_df):,} pairs")
    print(f"Classification threshold: {metrics['threshold']:.4f}")
    print("\nPerformance Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
    print("\nConfusion Matrix:")
    print(f"  True Positives:  {metrics['true_positives']:,}")
    print(f"  False Positives: {metrics['false_positives']:,}")
    print(f"  True Negatives:  {metrics['true_negatives']:,}")
    print(f"  False Negatives: {metrics['false_negatives']:,}")

    # Display examples in categorized tables
    display_test_examples_tables(test_df, similarity_scores, metrics["threshold"], n_examples=10)

    print(f"\n{'=' * 80}")
    print("EVALUATION SUMMARY")
    print(f"{'=' * 80}")
    print(f"Model: {model_path or SBERT_OUTPUT_FOLDER}")
    print(f"Test data: {len(test_df):,} pairs")
    print(f"Classification threshold: {metrics['threshold']:.4f}")

    # Return metrics
    return metrics


if __name__ == "__main__":
    # If run directly, evaluate the default model
    evaluate_and_print_report()
