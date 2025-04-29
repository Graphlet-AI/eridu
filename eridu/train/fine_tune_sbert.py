"""Fine-tunes a sentence transformer for people and company name matching using contrastive loss."""

import logging
import os
import random
import sys
import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from datasets import Dataset  # type: ignore
from scipy.stats import iqr
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    losses,
)
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from sentence_transformers.model_card import SentenceTransformerModelCardData
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sklearn.metrics import (  # type: ignore
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split  # type: ignore
from transformers import EarlyStoppingCallback

import wandb
from eridu.train.utils import compute_classifier_metrics  # noqa: F401
from eridu.train.utils import (
    compute_sbert_metrics,
    sbert_compare,
    sbert_compare_multiple,
    sbert_compare_multiple_df,
)

# For reproducibility
RANDOM_SEED: int = 31337
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.mps.manual_seed(RANDOM_SEED)

# Setup logging and suppress warnings
logging.basicConfig(stream=sys.stderr, level=logging.ERROR)
logger: logging.Logger = logging.getLogger(__name__)
warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", UserWarning)

# HuggingFace settings
os.environ["HF_ENDPOINT"] = "https://huggingface.co/"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Configure Pandas to show more rows
pd.set_option("display.max_rows", 40)
pd.set_option("display.max_columns", None)

# Configure sample size and model training parameters
SAMPLE_FRACTION: float = 1.0
SBERT_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# SBERT_MODEL: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
VARIANT: str = "original"
OPTIMIZER: str = "adafactor"
MODEL_SAVE_NAME: str = (SBERT_MODEL + "-" + VARIANT + "-" + OPTIMIZER).replace("/", "-")
EPOCHS: int = 6
BATCH_SIZE: int = 512
GRADIENT_ACCUMULATION_STEPS: int = 4
PATIENCE: int = 2
LEARNING_RATE: float = 5e-5
SBERT_OUTPUT_FOLDER: str = f"data/fine-tuned-sbert-{MODEL_SAVE_NAME}"
SAVE_EVAL_STEPS: int = 1000

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
dataset: pd.DataFrame = pd.read_parquet("data/pairs-all.parquet")

# Display the first few rows of the dataset
print("\nRaw training data sample:\n")
print(dataset.sample(n=20).head())

# Optionally sample the dataset
if SAMPLE_FRACTION < 1.0:
    dataset = dataset.sample(frac=SAMPLE_FRACTION)

# Split the dataset into training, evaluation, and test sets
train_df: pd.DataFrame
tmp_df: pd.DataFrame
eval_df: pd.DataFrame
test_df: pd.DataFrame
train_df, tmp_df = train_test_split(dataset, test_size=0.2, random_state=RANDOM_SEED, shuffle=True)
eval_df, test_df = train_test_split(tmp_df, test_size=0.5, random_state=RANDOM_SEED, shuffle=True)

print(f"\nTraining data:   {len(train_df):,}")
print(f"Evaluation data: {len(eval_df):,}")
print(f"Test data:       {len(eval_df):,}\n")

# Convert the training, evaluation, and test sets to HuggingFace Datasets
# Use float instead of bool for labels to avoid the subtraction error with boolean tensors
train_dataset: Dataset = Dataset.from_dict(
    {
        "sentence1": train_df["left_name"].tolist(),
        "sentence2": train_df["right_name"].tolist(),
        "label": train_df["match"].astype(float).tolist(),
    }
)
eval_dataset: Dataset = Dataset.from_dict(
    {
        "sentence1": eval_df["left_name"].tolist(),
        "sentence2": eval_df["right_name"].tolist(),
        "label": eval_df["match"].astype(float).tolist(),
    }
)
test_dataset: Dataset = Dataset.from_dict(
    {
        "sentence1": test_df["left_name"].tolist(),
        "sentence2": test_df["right_name"].tolist(),
        "label": test_df["match"].astype(float).tolist(),
    }
)

# Initialize the SBERT model
sbert_model: SentenceTransformer = SentenceTransformer(
    SBERT_MODEL,
    device=str(device),
    model_card_data=SentenceTransformerModelCardData(
        language="en",
        license="apache-2.0",
        model_name=f"{SBERT_MODEL}-address-matcher-{VARIANT}",
    ),
)
# Enable gradient checkpointing to save memory
sbert_model.gradient_checkpointing_enable()

# Try it out - doesn't work very well without fine-tuning, although cross-lingual works somewhat
print("\nTesting un-fine-tuned SBERT model:\n")
examples: list[str | float | object] = []
examples.append(
    [
        "John Smith",
        "John Smith",
        sbert_compare(sbert_model, "John Smith", "John Smith"),
    ]
)
examples.append(
    ["John Smith", "John H. Smith", sbert_compare(sbert_model, "John Smith", "John H. Smith")]
)
# Decent starting russian performance
examples.append(
    [
        "Yevgeny Prigozhin",
        "Евгений Пригожин",
        sbert_compare(sbert_model, "Yevgeny Prigozhin", "Евгений Пригожин"),
    ]
)
# Poor starting chinese performance - can we improve?
examples.append(["Ben Lorica", "罗瑞卡", sbert_compare(sbert_model, "Ben Lorica", "罗瑞卡")])
examples_df: pd.DataFrame = pd.DataFrame(examples, columns=["sentence1", "sentence2", "similarity"])
print(str(examples_df) + "\n")

# Evaluate a sample of the evaluation data compared using raw SBERT before fine-tuning
sample_df: pd.DataFrame = eval_df.sample(n=10000, random_state=RANDOM_SEED)
result_df: pd.DataFrame = sbert_compare_multiple_df(
    sbert_model, sample_df["left_name"], sample_df["right_name"], sample_df["match"]
)
error_s: pd.Series = np.abs(result_df.match.astype(float) - result_df.similarity)
score_diff_s: pd.Series = np.abs(error_s - sample_df.score)

# Compute the mean, standard deviation, and interquartile range of the error
stats_df: pd.DataFrame = pd.DataFrame(  # retain and append fine-tuned SBERT stats for comparison
    [
        {"mean": error_s.mean(), "std": error_s.std(), "iqr": iqr(error_s)},
        {"mean": score_diff_s.mean(), "std": score_diff_s.std(), "iqr": iqr(score_diff_s.dropna())},
    ],
    index=["Raw SBERT", "Raw SBERT - Levenshtein Score"],
)
print("\nRaw SBERT model stats:")
print(str(stats_df) + "\n")

# Make a Dataset from the sample data
sample_dataset: Dataset = Dataset.from_dict(
    {
        "sentence1": sample_df["left_name"].tolist(),
        "sentence2": sample_df["right_name"].tolist(),
        "label": sample_df["match"].astype(float).tolist(),  # Use float instead of bool
    }
)

# Initialize the evaluator
binary_acc_evaluator: BinaryClassificationEvaluator = BinaryClassificationEvaluator(
    sentences1=sample_dataset["sentence1"],
    sentences2=sample_dataset["sentence2"],
    labels=sample_dataset["label"],  # Already converted to float above
    name=SBERT_MODEL,
)
binary_acc_df: pd.DataFrame = pd.DataFrame([binary_acc_evaluator(sbert_model)])
print(str(binary_acc_df) + "\n")

#
# Fine-tune the SBERT model using contrastive loss
#

# This will effectively train the embedding model. MultipleNegativesRankingLoss did not work.
loss: losses.ContrastiveLoss = losses.ContrastiveLoss(model=sbert_model)

sbert_args: SentenceTransformerTrainingArguments = SentenceTransformerTrainingArguments(
    output_dir=SBERT_OUTPUT_FOLDER,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    fp16=True,
    fp16_opt_level="O1",  # “auto” AMP level
    warmup_ratio=0.1,
    run_name=SBERT_MODEL,
    load_best_model_at_end=True,
    save_total_limit=5,
    save_steps=SAVE_EVAL_STEPS,
    eval_steps=SAVE_EVAL_STEPS,
    save_strategy="steps",
    eval_strategy="steps",
    greater_is_better=False,
    metric_for_best_model="eval_loss",
    learning_rate=LEARNING_RATE,
    logging_dir="./logs",
    weight_decay=0.02,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    gradient_checkpointing=True,
    optim=OPTIMIZER,
)

trainer: SentenceTransformerTrainer = SentenceTransformerTrainer(
    model=sbert_model,
    args=sbert_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
    evaluator=binary_acc_evaluator,
    compute_metrics=compute_sbert_metrics,  # type: ignore
    callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)],
)

# This will take a while - if you're using CPU you need to sample the training dataset down a lot
trainer.train()  # type: ignore

print(f"Best model checkpoint path: {trainer.state.best_model_checkpoint}")  # type: ignore

print(pd.DataFrame([trainer.evaluate()]))

trainer.save_model(SBERT_OUTPUT_FOLDER)  # type: ignore
print(f"Saved model to {SBERT_OUTPUT_FOLDER}")

wandb.finish()

#
# Test out the fine-tuned model on the same examples as before. Note any improvements?
#
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


#
# Evaluate ROC curve and determine optimal threshold
#
y_true: list[float] = test_df["match"].astype(float).tolist()
y_scores: np.ndarray[Any, Any] = sbert_compare_multiple(
    sbert_model, test_df["left_name"], test_df["right_name"]
)

# Compute precision-recall curve
precision: list[float]
recall: list[float]
thresholds: list[float]
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

# Compute F1 score for each threshold
f1_scores: list[float] = [f1_score(y_true, y_scores >= t) for t in thresholds]

# Find the threshold that maximizes the F1 score
best_threshold_index = np.argmax(f1_scores)
best_threshold: float = thresholds[best_threshold_index]
best_f1_score: float = f1_scores[best_threshold_index]

print(f"Best Threshold: {best_threshold}")
print(f"Best F1 Score: {best_f1_score}")

roc_auc: float = roc_auc_score(y_true, y_scores)
print(f"AUC-ROC: {roc_auc}")

# Create a DataFrame for Seaborn
pr_data: pd.DataFrame = pd.DataFrame(
    {"Precision": precision[:-1], "Recall": recall[:-1], "F1 Score": f1_scores}
)

# Plot Precision-Recall curve using Seaborn and save to disk
sns.lineplot(data=pr_data, x="Recall", y="Precision", marker="o")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Augmented Test Set Precision-Recall Curve")
plt.savefig("images/precision_recall_curve.png")
