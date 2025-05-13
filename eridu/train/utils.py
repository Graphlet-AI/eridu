import logging
import os
import sys
from numbers import Number
from typing import Dict, List, Literal, Tuple, Type, TypeVar, Union

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
    sbert_model: SentenceTransformer,
    names1: List[str] | pd.Series,
    names2: List[str] | pd.Series,
    use_gpu: bool = True,
) -> np.ndarray:
    """sbert_compare_multiple - Efficiently compute cosine similarity between two lists of names using GPU when available.

    Args:
        sbert_model (SentenceTransformer): The SentenceTransformer model to use for encoding
        names1 (List[str]): First list of names to compare
        names2 (List[str]): Second list of names to compare
        use_gpu (bool, optional): Whether to use GPU acceleration. Defaults to True.

    Returns:
        np.ndarray: Array of cosine similarities between corresponding pairs of names
    """
    # Handle pandas Series and convert to lists
    if isinstance(names1, pd.Series):
        names1 = names1.astype(str).tolist()
    if isinstance(names2, pd.Series):
        names2 = names2.astype(str).tolist()

    # Get the device from the model
    device = next(sbert_model.parameters()).device if use_gpu else torch.device("cpu")
    device_str = str(device)

    # Determine whether to use GPU based on availability and parameter
    convert_to_tensor = use_gpu and (device.type == "cuda" or device.type == "mps")

    # Encode both lists of names into embeddings
    if convert_to_tensor:
        # GPU path - encode with tensors
        embeddings1_tensor: torch.Tensor = sbert_model.encode(
            names1,
            convert_to_tensor=True,
            convert_to_numpy=False,
            device=device_str,
        )
        embeddings2_tensor: torch.Tensor = sbert_model.encode(
            names2,
            convert_to_tensor=True,
            convert_to_numpy=False,
            device=device_str,
        )

        # Normalize the embeddings for efficient cosine similarity computation (on GPU)
        # Special case for a single embedding (very small sample sizes)
        if len(embeddings1_tensor.shape) == 1:
            embeddings1_tensor = embeddings1_tensor.unsqueeze(0)
            embeddings2_tensor = embeddings2_tensor.unsqueeze(0)

        # Always normalize along dimension 1 (embedding dimension)
        embeddings1_tensor = embeddings1_tensor / torch.norm(
            embeddings1_tensor, dim=1, keepdim=True
        )
        embeddings2_tensor = embeddings2_tensor / torch.norm(
            embeddings2_tensor, dim=1, keepdim=True
        )

        # Compute cosine similarity using dot product of normalized vectors (on GPU)
        tensor_similarities = torch.sum(embeddings1_tensor * embeddings2_tensor, dim=1)

        # Convert back to numpy for consistent return type
        similarities = tensor_similarities.cpu().numpy()
    else:
        # CPU path
        embeddings1_np: np.ndarray = sbert_model.encode(names1, convert_to_numpy=True)
        embeddings2_np: np.ndarray = sbert_model.encode(names2, convert_to_numpy=True)

        # Special case for a single embedding
        if len(embeddings1_np.shape) == 1:
            embeddings1_np = np.expand_dims(embeddings1_np, axis=0)
            embeddings2_np = np.expand_dims(embeddings2_np, axis=0)

        # Always normalize along dimension 1 (embedding dimension)
        embeddings1_np = embeddings1_np / np.linalg.norm(embeddings1_np, axis=1, keepdims=True)
        embeddings2_np = embeddings2_np / np.linalg.norm(embeddings2_np, axis=1, keepdims=True)

        # Compute cosine similarity using dot product of normalized vectors
        similarities = np.sum(embeddings1_np * embeddings2_np, axis=1)

    return similarities


def sbert_compare_multiple_df(
    sbert_model: SentenceTransformer,
    names1: List[str] | pd.Series,
    names2: List[str] | pd.Series,
    matches: List[bool] | pd.Series,
    use_gpu: bool = True,
) -> pd.DataFrame:
    """sbert_compare_multiple_df - Efficiently compute cosine similarity between two lists of names using GPU when available."""
    similarities = sbert_compare_multiple(sbert_model, names1, names2, use_gpu=use_gpu)
    return pd.DataFrame(
        {"name1": names1, "name2": names2, "similarity": similarities, "match": matches}
    )


def sbert_match_multiple(
    df: pd.DataFrame,
    sbert_model: SentenceTransformer,
    name1_col: str = "name1",
    name2_col: str = "name2",
    use_gpu: bool = True,
) -> pd.Series:
    """sbert_match_multiple - Efficiently compute cosine similarities for all rows in a DataFrame

    Args:
        df (pd.DataFrame): DataFrame containing name pairs to compare
        sbert_model (SentenceTransformer): The SentenceTransformer model to use
        name1_col (str): Column name for first name
        name2_col (str): Column name for second name
        use_gpu (bool, optional): Whether to use GPU acceleration. Defaults to True.

    Returns:
        pd.Series: Series of cosine similarities between name pairs
    """
    similarities = sbert_compare_multiple(
        sbert_model, df[name1_col].tolist(), df[name2_col].tolist(), use_gpu=use_gpu
    )
    return pd.Series(similarities, index=df.index)


def sbert_compare(
    sbert_model: SentenceTransformer, name1: str, name2: str, use_gpu: bool = True
) -> float:
    """sbert_compare - sentence encode each name into a fixed-length text embedding.
    Fixed-length means they can be compared with cosine similarity.

    Args:
        sbert_model (SentenceTransformer): The SentenceTransformer model to use for encoding
        name1 (str): First name to compare
        name2 (str): Second name to compare
        use_gpu (bool, optional): Whether to use GPU acceleration. Defaults to True.

    Returns:
        float: Cosine similarity between the two name embeddings
    """
    # Get the device from the model
    device = next(sbert_model.parameters()).device if use_gpu else torch.device("cpu")
    device_str = str(device)

    # Determine whether to use GPU based on availability and parameter
    convert_to_tensor = use_gpu and (device.type == "cuda" or device.type == "mps")

    # Encode both names into embeddings
    if convert_to_tensor:
        # GPU path
        embedding1_tensor: torch.Tensor = sbert_model.encode(
            name1,
            convert_to_tensor=True,
            convert_to_numpy=False,
            device=device_str,
        )
        embedding2_tensor: torch.Tensor = sbert_model.encode(
            name2,
            convert_to_tensor=True,
            convert_to_numpy=False,
            device=device_str,
        )

        # Compute cosine similarity directly using torch
        # Make sure we use the correct dimension for norm if the tensor has more than 1 dimension
        if len(embedding1_tensor.shape) > 1 and embedding1_tensor.shape[0] == 1:
            # If we have a batch of 1, squeeze to get a single vector
            embedding1_tensor = embedding1_tensor.squeeze(0)
            embedding2_tensor = embedding2_tensor.squeeze(0)

        # Normalize the embeddings
        embedding1_tensor = embedding1_tensor / torch.norm(embedding1_tensor)
        embedding2_tensor = embedding2_tensor / torch.norm(embedding2_tensor)

        # For a single vector, use a simple dot product
        similarity = torch.sum(embedding1_tensor * embedding2_tensor).item()
        return float(similarity)
    else:
        # Original CPU implementation
        embedding1_np: np.ndarray = sbert_model.encode(name1, convert_to_numpy=True)
        embedding2_np: np.ndarray = sbert_model.encode(name2, convert_to_numpy=True)
        diff: float = 1 - distance.cosine(embedding1_np, embedding2_np)
        return diff


def sbert_match(
    sbert_model: SentenceTransformer, row: pd.Series, use_gpu: bool = True
) -> pd.Series:
    """sbert_match - SentenceTransformer name matching, float iytoyt"""
    bin_match: Literal[0, 1] = sbert_compare_binary(
        sbert_model, row["name1"], row["name2"], use_gpu=use_gpu
    )
    return pd.Series(bin_match, index=row.index)


def sbert_compare_binary(
    sbert_model: SentenceTransformer,
    name1: str,
    name2: str,
    threshold: float = 0.5,
    use_gpu: bool = True,
) -> Literal[0, 1]:
    """sbert_match - compare and return a binary match"""
    similarity = sbert_compare(sbert_model, name1, name2, use_gpu=use_gpu)
    return 1 if similarity >= threshold else 0


def sbert_match_binary(
    sbert_model: SentenceTransformer, row: pd.Series, threshold: float = 0.5, use_gpu: bool = True
) -> pd.Series:
    """sbert_match_binary - SentenceTransformer name matching, binary output"""
    bin_match = sbert_compare_binary(
        sbert_model, row["name1"], row["name2"], threshold=threshold, use_gpu=use_gpu
    )
    return pd.Series(bin_match, index=row.index)
