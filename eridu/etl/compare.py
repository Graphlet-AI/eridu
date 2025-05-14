"""Name comparison functionality for entity resolution."""

import os
from typing import Optional, Tuple

import torch
from sentence_transformers import SentenceTransformer

from eridu.train.utils import sbert_compare


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


def load_model(model_path: str, device: str) -> Optional[SentenceTransformer]:
    """Load a SentenceTransformer model from disk.

    Args:
        model_path: Path to the model directory
        device: Device to load the model on ('cuda', 'mps', or 'cpu')

    Returns:
        The loaded model or None if an error occurred
    """
    if not os.path.exists(model_path):
        return None

    try:
        return SentenceTransformer(model_path, device=device)
    except Exception:
        return None


def compare_names(
    name1: str, name2: str, model_path: str, use_gpu: bool = True
) -> Tuple[float, bool]:
    """Compare two names using a fine-tuned SentenceTransformer model.

    Args:
        name1: First name to compare
        name2: Second name to compare
        model_path: Path to the model directory
        use_gpu: Whether to use GPU acceleration

    Returns:
        Tuple of (similarity_score, success_flag)
        where similarity_score is a float between 0 and 1,
        and success_flag indicates whether the operation was successful
    """
    # Determine device
    device = get_device(use_gpu)

    # Load model
    model = load_model(model_path, device)
    if model is None:
        return 0.0, False

    # Compute similarity
    try:
        similarity = sbert_compare(model, name1, name2, use_gpu=(device != "cpu"))
        return similarity, True
    except Exception:
        return 0.0, False
