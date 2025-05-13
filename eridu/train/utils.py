import logging
import os
import sys
from numbers import Number
from typing import Any, Dict, List, Literal, Tuple, Type, TypeVar, Union

import numpy as np
import pandas as pd
import torch
from scipy.spatial import distance
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (  # type: ignore
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from transformers import AutoTokenizer

COLUMN_SPECIAL_CHAR = "[COL]"
VALUE_SPECIAL_CHAR = "[VAL]"

# Setup basic logging
logging.basicConfig(stream=sys.stderr, level=logging.ERROR)
logger = logging.getLogger(__name__)


def compute_sbert_metrics(eval_pred: Tuple[List, List]) -> Dict[str, Number]:
    """compute_metrics - Compute accuracy, precision, recall, f1 and roc_auc

    This function is called during model evaluation and logs metrics to W&B automatically
    through the WandbCallback.
    """
    predictions, labels = eval_pred

    # Apply threshold to predictions (0.5 is default)
    if isinstance(predictions[0], float):
        # If predictions are similarity scores (between 0 and 1)
        binary_preds = [1 if pred >= 0.5 else 0 for pred in predictions]
    else:
        # If predictions are already binary
        binary_preds = predictions

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(labels, binary_preds),
        "precision": precision_score(labels, binary_preds, zero_division=0),
        "recall": recall_score(labels, binary_preds, zero_division=0),
        "f1": f1_score(labels, binary_preds, zero_division=0),
    }

    # Calculate AUC only if predictions are continuous (not binary)
    if isinstance(predictions[0], float):
        metrics["auc"] = roc_auc_score(labels, predictions)

    return metrics


def preprocess_logits_for_metrics(logits, labels):
    return logits.argmax(dim=-1)


def compute_classifier_metrics(eval_pred):
    logits, labels = eval_pred
    logits = torch.tensor(logits)
    labels = torch.tensor(labels)
    predictions = (logits > 0.5).long().squeeze()

    if len(predictions) != len(labels):
        raise ValueError(
            f"Mismatch in lengths: predictions ({len(predictions)}) and labels ({len(labels)})"
        )

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary"
    )
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def format_dataset(dataset):
    dataset.set_format(
        type="torch",
        columns=["input_ids_a", "attention_mask_a", "input_ids_b", "attention_mask_b", "labels"],
    )
    return dataset


def save_transformer(model: torch.nn.Module, save_path: str) -> None:
    """Save a trained Transformers model and its tokenizer.

    Args:
    model (torch.nn.Module): The trained transformers model to save.
    save_path (str): The directory path where the model will be saved.
    """

    os.makedirs(save_path, exist_ok=True)

    # Save the model state
    torch.save(model.state_dict(), os.path.join(save_path, "model_state.pt"))

    # Save the tokenizer
    model.tokenizer.save_pretrained(save_path)  # type: ignore

    # Save the model configuration (optional, but recommended)
    config = {"model_name": model.model_name, "dim": model.ffnn[0].in_features}  # type: ignore
    torch.save(config, os.path.join(save_path, "config.pt"))

    logging.info(f"Model saved to {save_path}")


T = TypeVar("T", bound=torch.nn.Module)


def load_transformer(
    model_cls: Type[T], load_path: str, device: Union[str, torch.device] = "cpu"
) -> T:
    """load_transformer Load a saved Transformers model and its tokenizer.

    Parameters
    ----------
    model_cls : torch.nn.Module class
        Model class name
    load_path : _type_
        Saved model directory path

    Returns
    -------
    Any
        Model with weights loaded from the saved directory
    """
    # Load the configuration
    config = torch.load(os.path.join(load_path, "config.pt"))

    # Initialize the model
    model: T = model_cls(model_name=config["model_name"], dim=config["dim"])

    # Load the model state
    model.load_state_dict(torch.load(os.path.join(load_path, "model_state.pt")))

    # Load the tokenizer
    model.tokenizer = AutoTokenizer.from_pretrained(load_path)

    # Send it to the right device
    model.to(device)

    logging.info(f"Model loaded from {load_path}")
    return model


def sbert_compare_multiple(
    sbert_model: SentenceTransformer, names1: List[str] | pd.Series, names2: List[str] | pd.Series
) -> np.ndarray:
    """sbert_compare_multiple - Efficiently compute cosine similarity between two lists of names using numpy arrays.

    Args:
        sbert_model (SentenceTransformer): The SentenceTransformer model to use for encoding
        names1 (List[str]): First list of names to compare
        names2 (List[str]): Second list of names to compare

    Returns:
        np.ndarray: Array of cosine similarities between corresponding pairs of names
    """
    # Handle pandas Series and convert to lists
    if isinstance(names1, pd.Series):
        names1 = names1.astype(str).tolist()
    if isinstance(names2, pd.Series):
        names2 = names2.astype(str).tolist()

    # Encode both lists of names into embeddings
    embeddings1 = sbert_model.encode(names1, convert_to_numpy=True)
    embeddings2 = sbert_model.encode(names2, convert_to_numpy=True)

    # Normalize the embeddings for efficient cosine similarity computation
    embeddings1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
    embeddings2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)

    # Compute cosine similarity using dot product of normalized vectors
    similarities: np.ndarray[Any, Any] = np.sum(embeddings1 * embeddings2, axis=1)

    return similarities


def sbert_compare_multiple_df(
    sbert_model: SentenceTransformer,
    names1: List[str] | pd.Series,
    names2: List[str] | pd.Series,
    matches: List[bool] | pd.Series,
) -> pd.DataFrame:
    """sbert_compare_multiple_df - Efficiently compute cosine similarity between two lists of names using numpy arrays."""
    similarities = sbert_compare_multiple(sbert_model, names1, names2)
    return pd.DataFrame(
        {"name1": names1, "name2": names2, "similarity": similarities, "match": matches}
    )


def sbert_match_multiple(
    df: pd.DataFrame,
    sbert_model: SentenceTransformer,
    name1_col: str = "name1",
    name2_col: str = "name2",
) -> pd.Series:
    """sbert_match_multiple - Efficiently compute cosine similarities for all rows in a DataFrame

    Args:
        df (pd.DataFrame): DataFrame containing name pairs to compare
        sbert_model (SentenceTransformer): The SentenceTransformer model to use
        name1_col (str): Column name for first name
        name2_col (str): Column name for second name

    Returns:
        pd.Series: Series of cosine similarities between name pairs
    """
    similarities = sbert_compare_multiple(
        sbert_model, df[name1_col].tolist(), df[name2_col].tolist()
    )
    return pd.Series(similarities, index=df.index)


def sbert_compare(sbert_model: SentenceTransformer, name1: str, name2: str) -> float:
    """sbert_compare - sentence encode each name into a fixed-length text embedding.
    Fixed-length means they can be compared with cosine similarity."""
    embedding1 = sbert_model.encode(name1)
    embedding2 = sbert_model.encode(name2)

    # Compute cosine similarity
    diff: float = 1 - distance.cosine(embedding1, embedding2)
    return diff


def sbert_match(sbert_model: SentenceTransformer, row: pd.Series) -> pd.Series:
    """sbert_match - SentenceTransformer name matching, float iytoyt"""
    bin_match: Literal[0, 1] = sbert_compare_binary(sbert_model, row["name1"], row["name2"])
    return pd.Series(bin_match, index=row.index)


def sbert_compare_binary(
    sbert_model: SentenceTransformer, name1: str, name2: str, threshold: float = 0.5
) -> Literal[0, 1]:
    """sbert_match - compare and return a binary match"""
    similarity = sbert_compare(sbert_model, name1, name2)
    return 1 if similarity >= threshold else 0


def sbert_match_binary(
    sbert_model: SentenceTransformer, row: pd.Series, threshold: float = 0.5
) -> pd.Series:
    """sbert_match_binary - SentenceTransformer name matching, binary output"""
    bin_match = sbert_compare_binary(sbert_model, row["name1"], row["name2"], threshold=threshold)
    return pd.Series(bin_match, index=row.index)
