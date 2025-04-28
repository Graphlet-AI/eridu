import logging
import os
import sys
from numbers import Number
from typing import Any, Callable, Dict, List, Tuple, Type, TypeVar, Union

import pandas as pd
import torch
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
    """compute_metrics - Compute accuracy, precision, recall, f1 and roc_auc"""
    predictions, labels = eval_pred
    metrics = {}
    for metric in accuracy_score, precision_score, recall_score, f1_score, roc_auc_score:
        metrics[metric.__name__] = metric(labels, predictions)

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


def tokenize_function(examples, tokenizer):
    encoded_a = tokenizer(examples["sentence1"], padding="max_length", truncation=True)
    encoded_b = tokenizer(examples["sentence2"], padding="max_length", truncation=True)
    return {
        "input_ids_a": encoded_a["input_ids"],
        "attention_mask_a": encoded_a["attention_mask"],
        "input_ids_b": encoded_b["input_ids"],
        "attention_mask_b": encoded_b["attention_mask"],
        "labels": examples["label"],
    }


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


def gold_label_report(
    s: pd.Series, eval_methods: List[Callable], threshold: float = 0.5
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """gold_label_evaluate Evaluate a model on our original gold labels and return a report.
    Returns a Tuple of two pd.DataFrames, one with raw results and one with aggregated results."""
    s = s.copy(deep=True)

    raw_df: pd.DataFrame = pd.DataFrame()
    raw_df["Description"] = s["Description"]
    raw_df["Address1"] = s["Address1"]
    raw_df["Address2"] = s["Address2"]
    raw_df["Label"] = s["Label"]

    agg_funcs = {}
    kwargs = {"threshold": threshold}

    for eval_method in eval_methods:
        # Apply the matching model to the address pair
        func_col_name: str = eval_method.__name__

        if "sbert" in func_col_name:

            def apply_eval_method(row: pd.Series, threshold=threshold) -> Any:
                return eval_method(row, threshold=threshold)

            raw_df[func_col_name] = s.apply(apply_eval_method, axis=1, **kwargs)  # type: ignore
        else:
            raw_df[func_col_name] = s.apply(eval_method, axis=1)

        raw_df[f"{func_col_name}_correct"] = raw_df[func_col_name] == raw_df["Label"]

        agg_funcs[f"{eval_method.__name__}_acc"] = (
            f"{eval_method.__name__}",
            lambda x: x.mean(),
        )

    grouped_df: pd.DataFrame = raw_df.groupby("Description").agg(**agg_funcs)  # type: ignore

    return raw_df, grouped_df
