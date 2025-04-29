import json
import logging
import os
import random
import re
import sys
import time
import warnings
from numbers import Number
from typing import Callable, Dict, List, Literal, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from scipy.spatial import distance  # type: ignore
from scipy.stats import iqr  # type: ignore
from sentence_transformers import (
    InputExample,
    SentencesDataset,
    SentenceTransformer,
    SentenceTransformerTrainer,
    losses,
)
from sentence_transformers.evaluation import (
    BinaryClassificationEvaluator,
    EmbeddingSimilarityEvaluator,
    SimilarityFunction,
)
from sentence_transformers.model_card import SentenceTransformerModelCardData
from sentence_transformers.training_args import (
    BatchSamplers,
    SentenceTransformerTrainingArguments,
)
from sklearn.metrics import (  # type: ignore
    accuracy_score,
    auc,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split  # type: ignore
from torch.optim import RAdam
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from transformers.integrations import WandbCallback

from eridu.train.utils import (
    compute_classifier_metrics,
    compute_sbert_metrics,
    format_dataset,
    load_transformer,
    preprocess_logits_for_metrics,
    save_transformer,
    sbert_compare,
    sbert_compare_binary,
    sbert_compare_multiple,
    sbert_compare_multiple_df,
    sbert_match,
    sbert_match_binary,
    tokenize_function,
)

# For reproducibility
RANDOM_SEED = 31337
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.mps.manual_seed(RANDOM_SEED)

# Setup logging and suppress warnings
logging.basicConfig(stream=sys.stderr, level=logging.ERROR)
logger = logging.getLogger(__name__)
warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", UserWarning)

# HuggingFace settings
os.environ["HF_ENDPOINT"] = "https://huggingface.co/"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure Pandas to show more rows
pd.set_option("display.max_rows", 40)
pd.set_option("display.max_columns", None)

# Check for CUDA or MPS availability and set the device
device: torch.device | str
if torch.backends.mps.is_available():
    device = torch.device("mps")
    logger.debug("Using Apple GPU acceleration")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    logger.debug("Using NVIDIA CUDA GPU acceleration")
else:
    device = "cpu"
    logger.debug("Using CPU for ML")

print(f"Device for fine-tuning SBERT: {device}")

# Load the dataset
dataset = pd.read_parquet("data/pairs-all.parquet")

# Display the first few rows of the dataset
print("\nRaw training data sample:\n")
print(dataset.head())

# Split the dataset into training, evaluation, and test sets
train_df, tmp_df = train_test_split(dataset, test_size=0.2, random_state=RANDOM_SEED, shuffle=True)
eval_df, test_df = train_test_split(tmp_df, test_size=0.5, random_state=RANDOM_SEED, shuffle=True)

print(f"\nTraining data:   {len(train_df):,}")
print(f"Evaluation data: {len(eval_df):,}")
print(f"Test data:       {len(eval_df):,}")

# Convert the training, evaluation, and test sets to HuggingFace Datasets
train_dataset = Dataset.from_dict(
    {
        "sentence1": train_df["left_name"].tolist(),
        "sentence2": train_df["right_name"].tolist(),
        "label": train_df["match"].astype(bool).tolist(),
    }
)
eval_dataset = Dataset.from_dict(
    {
        "sentence1": eval_df["left_name"].tolist(),
        "sentence2": eval_df["right_name"].tolist(),
        "label": eval_df["match"].astype(bool).tolist(),
    }
)
test_dataset = Dataset.from_dict(
    {
        "sentence1": test_df["left_name"].tolist(),
        "sentence2": test_df["right_name"].tolist(),
        "label": test_df["match"].astype(bool).tolist(),
    }
)

# Configure training parameters
# SBERT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SBERT_MODEL = "BAAI/bge-m3"
VARIANT = "original"
MODEL_SAVE_NAME = (SBERT_MODEL + "-" + VARIANT).replace("/", "-")
EPOCHS = 20
BATCH_SIZE = 32
PATIENCE = 2
LEARNING_RATE = 5e-5
SBERT_OUTPUT_FOLDER = f"data/fine-tuned-sbert-{MODEL_SAVE_NAME}"
SAVE_EVAL_STEPS = 100

# Initialize the SBERT model
sbert_model = SentenceTransformer(
    SBERT_MODEL,
    device=str(device),
    model_card_data=SentenceTransformerModelCardData(
        language="en",
        license="apache-2.0",
        model_name=f"{SBERT_MODEL}-address-matcher-{VARIANT}",
    ),
)

# Try it out - doesn't work very well without fine-tuning, although cross-lingual works somewhat
print("John Smith", "John Smith", sbert_compare(sbert_model, "John Smith", "John Smith"))
print("John Smith", "John H. Smith", sbert_compare(sbert_model, "John Smith", "John H. Smith"))
# Decent starting russian performance
print(
    "Yevgeny Prigozhin",
    "Евгений Пригожин",
    sbert_compare(sbert_model, "Yevgeny Prigozhin", "Евгений Пригожин"),
)
# Poor starting chinese performance - can we improve?
print("Ben Lorica", "罗瑞卡", sbert_compare(sbert_model, "Ben Lorica", "罗瑞卡"))

# Evaluate a sample of the evaluation data compared using raw SBERT before fine-tuning
sample_df = eval_df.sample(n=2000)
result_df = sbert_compare_multiple_df(
    sbert_model, sample_df["left_name"], sample_df["right_name"], sample_df["match"]
)
error_df = np.abs(result_df.match.astype(float) - result_df.similarity)

# Compute the mean, standard deviation, and interquartile range of the error
mean_error, std_error, iqt_error = error_df.mean(), error_df.std(), iqr(error_df)
stats_df = pd.DataFrame(  # retain and append fine-tuned SBERT stats for comparison
    [
        {"mean": mean_error, "std": std_error, "iqr": iqt_error},
    ],
    index=["Raw SBERT"],
)
print(stats_df)
