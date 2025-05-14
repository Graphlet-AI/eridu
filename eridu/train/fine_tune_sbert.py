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
import torch.quantization as tq
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
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split  # type: ignore
from transformers import EarlyStoppingCallback
from transformers.integrations import WandbCallback

import wandb
from eridu.train.utils import compute_classifier_metrics  # noqa: F401
from eridu.train.utils import (
    compute_sbert_metrics,
    sbert_compare,
    sbert_compare_multiple,
    sbert_compare_multiple_df,
)

#
# Training run and pandas configuration, environment variables, and runtime parameters
#

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

# Configure sample size and model training parameters from environment or defaults
SAMPLE_FRACTION: float = float(os.environ.get("SAMPLE_FRACTION", "0.01"))
SBERT_MODEL: str = os.environ.get(
    "SBERT_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
# SBERT_MODEL: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
VARIANT: str = os.environ.get("VARIANT", "original")
OPTIMIZER: str = os.environ.get("OPTIMIZER", "adafactor")
MODEL_SAVE_NAME: str = (SBERT_MODEL + "-" + VARIANT + "-" + OPTIMIZER).replace("/", "-")
EPOCHS: int = int(os.environ.get("EPOCHS", "6"))
BATCH_SIZE: int = int(os.environ.get("BATCH_SIZE", "1024"))
GRADIENT_ACCUMULATION_STEPS: int = int(os.environ.get("GRADIENT_ACCUMULATION_STEPS", "4"))
PATIENCE: int = int(os.environ.get("PATIENCE", "2"))
LEARNING_RATE: float = float(os.environ.get("LEARNING_RATE", "5e-5"))
SBERT_OUTPUT_FOLDER: str = f"data/fine-tuned-sbert-{MODEL_SAVE_NAME}"
SAVE_EVAL_STEPS: int = int(os.environ.get("SAVE_EVAL_STEPS", "100"))
USE_FP16: bool = os.environ.get("USE_FP16", "True").lower() == "true"

# Get Weights & Biases configuration from environment variables
WANDB_PROJECT: str = os.environ.get("WANDB_PROJECT", "eridu")
WANDB_ENTITY: str = os.environ.get("WANDB_ENTITY", "rjurney")

# Initialize Weights & Biases
wandb.init(
    entity=WANDB_ENTITY,
    # set the wandb project where this run will be logged
    project=WANDB_PROJECT,
    # track hyperparameters and run metadata
    config={
        "variant": VARIANT,
        "optimizer": OPTIMIZER,
        "epochs": EPOCHS,
        "sample_fraction": SAMPLE_FRACTION,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "use_fp16": USE_FP16,
        "batch_size": BATCH_SIZE,
        "patience": PATIENCE,
        "learning_rate": LEARNING_RATE,
        "sbert_model": SBERT_MODEL,
        "model_save_name": MODEL_SAVE_NAME,
        "sbert_output_folder": SBERT_OUTPUT_FOLDER,
        "save_eval_steps": SAVE_EVAL_STEPS,
    },
    save_code=True,
)

#
# Check for CUDA or MPS availability and set the device based on USE_GPU env var
#

# Check if GPU should be used
USE_GPU: bool = os.environ.get("USE_GPU", "True").lower() == "true"

device: torch.device | str
if USE_GPU:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.debug("Using Apple GPU acceleration")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.debug("Using NVIDIA CUDA GPU acceleration")
    else:
        device = "cpu"
        logger.debug("No GPU available, falling back to CPU")
else:
    device = "cpu"
    logger.debug("GPU disabled by user, using CPU for ML")

print(f"Device for fine-tuning SBERT: {device}")

#
# Load and prepare the dataset
#

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

# Initialize and configure the SBERT model
sbert_model: SentenceTransformer = SentenceTransformer(
    SBERT_MODEL,
    device=str(device),
    model_card_data=SentenceTransformerModelCardData(
        language="en",
        license="apache-2.0",
        model_name=f"{SBERT_MODEL}-name-matcher-{VARIANT}",
    ),
)
# Enable gradient checkpointing to save memory
sbert_model.gradient_checkpointing_enable()

# Put network in training mode
sbert_model.train()

# Only apply quantization when not using fp16 to avoid conflicts
if not USE_FP16:
    # 2. Tell PyTorch to quantize the Linear layers in the encoder
    for module in sbert_model.modules():
        if isinstance(module, torch.nn.Linear):
            module.qconfig = tq.get_default_qat_qconfig("fbgemm")

    # 4. Prepare QAT: inserts FakeQuant and Observer modules
    tq.prepare_qat(sbert_model, inplace=True)

#
# Try the SBERT model out without fine-tuning. Multi-lingual comaprisons work somewhat.
#

print("\nTesting raw [un-fine-tuned] SBERT model:\n")
examples: list[str | float | object] = []
examples.append(
    [
        "John Smith",
        "John Smith",
        sbert_compare(sbert_model, "John Smith", "John Smith", use_gpu=True),
    ]
)
examples.append(
    [
        "John Smith",
        "John H. Smith",
        sbert_compare(sbert_model, "John Smith", "John H. Smith", use_gpu=True),
    ]
)
# Decent starting russian performance
examples.append(
    [
        "Yevgeny Prigozhin",
        "Евгений Пригожин",
        sbert_compare(sbert_model, "Yevgeny Prigozhin", "Евгений Пригожин", use_gpu=True),
    ]
)
# Poor starting chinese performance - can we improve?
examples.append(
    ["Ben Lorica", "罗瑞卡", sbert_compare(sbert_model, "Ben Lorica", "罗瑞卡", use_gpu=True)]
)
examples_df: pd.DataFrame = pd.DataFrame(examples, columns=["sentence1", "sentence2", "similarity"])
print(str(examples_df) + "\n")

#
# Evaluate a sample of the evaluation data compared using raw SBERT before fine-tuning
#

# Use a simple approach for evaluation sampling
sample_df: pd.DataFrame = eval_df.sample(frac=0.1, random_state=RANDOM_SEED)
# Make sure we have at least a few samples
if len(sample_df) < 5 and len(eval_df) >= 5:
    sample_df = eval_df.sample(n=5, random_state=RANDOM_SEED)
print(f"Running initial evaluation on {device} for {len(sample_df):,} sample records")

result_df: pd.DataFrame = sbert_compare_multiple_df(
    sbert_model, sample_df["left_name"], sample_df["right_name"], sample_df["match"], use_gpu=True
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

# Log initial model metrics to W&B
wandb.log(
    {
        "raw_model/error_mean": error_s.mean(),
        "raw_model/error_std": error_s.std(),
        "raw_model/error_iqr": iqr(error_s),
    }
)

# Make a Dataset from the sample data
sample_dataset: Dataset = Dataset.from_dict(
    {
        "sentence1": sample_df["left_name"].tolist(),
        "sentence2": sample_df["right_name"].tolist(),
        "label": sample_df["match"].astype(float).tolist(),  # Use float instead of bool
    }
)

# Ensure evaluation directory exists
eval_dir = (
    f"{SBERT_OUTPUT_FOLDER}/eval/binary_classification_evaluation_{SBERT_MODEL.replace('/', '-')}"
)
os.makedirs(eval_dir, exist_ok=True)

# Set the name property of the evaluator to the sanitized model name
evaluation_name = SBERT_MODEL.replace("/", "-")

# Use an evaluator to get trustworthy metrics for the match classification
binary_acc_evaluator: BinaryClassificationEvaluator = BinaryClassificationEvaluator(
    sentences1=sample_dataset["sentence1"],
    sentences2=sample_dataset["sentence2"],
    labels=sample_dataset["label"],  # Already converted to float above
    name=evaluation_name,
)
binary_acc_results = binary_acc_evaluator(sbert_model)
binary_acc_df: pd.DataFrame = pd.DataFrame([binary_acc_results])
print(str(binary_acc_df) + "\n")

# Log binary classification evaluation metrics to W&B
wandb.log(
    {
        "raw_model/binary_accuracy": binary_acc_results.get("accuracy", 0.0),
        "raw_model/binary_f1": binary_acc_results.get("f1", 0.0),
        "raw_model/binary_precision": binary_acc_results.get("precision", 0.0),
        "raw_model/binary_recall": binary_acc_results.get("recall", 0.0),
        "raw_model/binary_ap": binary_acc_results.get("ap", 0.0),
    }
)

#
# Fine-tune the SBERT model using contrastive loss
#

# This will effectively train the embedding model. MultipleNegativesRankingLoss did not work.
loss: losses.ContrastiveLoss = losses.ContrastiveLoss(model=sbert_model)

# Set lots of options to reduce memory usage and improve training speed
sbert_args: SentenceTransformerTrainingArguments = SentenceTransformerTrainingArguments(
    output_dir=SBERT_OUTPUT_FOLDER,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    fp16=USE_FP16,
    fp16_opt_level="O1" if USE_FP16 else "O0",
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
    callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE), WandbCallback()],
)

# This will take a while - if you're using CPU you need to sample the training dataset down a lot
trainer.train()  # type: ignore

print(f"Best model checkpoint path: {trainer.state.best_model_checkpoint}")  # type: ignore
print(pd.DataFrame([trainer.evaluate()]))

trainer.save_model(SBERT_OUTPUT_FOLDER)  # type: ignore
print(f"Saved model to {SBERT_OUTPUT_FOLDER}")

# Don't finish wandb yet - keep it running for the final evaluation metrics

#
# Test out the fine-tuned model on the same examples as before. Note any improvements?
#

print("\nTesting fine-tuned SBERT model:\n")
tuned_examples: list[str | float | object] = []
tuned_examples.append(
    [
        "John Smith",
        "John Smith",
        sbert_compare(sbert_model, "John Smith", "John Smith", use_gpu=True),
    ]
)
tuned_examples.append(
    [
        "John Smith",
        "John H. Smith",
        sbert_compare(sbert_model, "John Smith", "John H. Smith", use_gpu=True),
    ]
)
# Decent starting russian performance
tuned_examples.append(
    [
        "Yevgeny Prigozhin",
        "Евгений Пригожин",
        sbert_compare(sbert_model, "Yevgeny Prigozhin", "Евгений Пригожин", use_gpu=True),
    ]
)
# Poor starting chinese performance - can we improve?
tuned_examples.append(
    ["Ben Lorica", "罗瑞卡", sbert_compare(sbert_model, "Ben Lorica", "罗瑞卡", use_gpu=True)]
)
tuned_examples_df: pd.DataFrame = pd.DataFrame(
    tuned_examples, columns=["sentence1", "sentence2", "similarity"]
)
print(str(tuned_examples_df) + "\n")

#
# Evaluate ROC curve of the fine-tuned model and determine optimal threshold
#

# Use a simpler sampling approach for test data
# Use 10% of test data or at least 10 samples, whichever is larger
test_sample_size = max(int(len(test_df) * 0.1), min(len(test_df), 10))
test_df = test_df.sample(n=test_sample_size, random_state=RANDOM_SEED)

y_true: list[float] = test_df["match"].astype(float).tolist()
# Use GPU acceleration for inference
print(f"Running inference on {device} for {len(test_df):,} test records")

y_scores: np.ndarray[Any, Any] = sbert_compare_multiple(
    sbert_model, test_df["left_name"], test_df["right_name"], use_gpu=True
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

# Log final metrics to W&B
wandb.log(
    {
        "final/best_threshold": best_threshold,
        "final/best_f1_score": best_f1_score,
        "final/accuracy": accuracy_score(y_true, y_scores >= best_threshold),
        "final/precision": precision_score(y_true, y_scores >= best_threshold),
        "final/recall": recall_score(y_true, y_scores >= best_threshold),
        "final/auc": roc_auc,
    }
)

# Log the precision-recall curve to W&B - avoid using labels parameter to prevent indexing error
try:
    # Convert to the format W&B expects (binary classification probabilities)
    # For binary classification, W&B expects shape (n_samples, 2) for probabilities
    y_probs_formatted = np.vstack([1 - y_scores, y_scores]).T
    wandb.log({"final/pr_curve": wandb.plot.pr_curve(y_true, y_probs_formatted)})
except Exception as e:
    print(f"Warning: Could not log PR curve to W&B: {e}")
    # Log individual metrics instead
    wandb.log({"final/y_true": y_true, "final/y_scores": y_scores.tolist()})

# Now it's safe to finish wandb
wandb.finish()


def main() -> None:
    """Main function for running the SBERT fine-tuning process from CLI.

    This function is intended to be called by the Eridu CLI.
    It has already accessed environment variables for configuration.
    All other processing code is in the module's global scope.
    """
    # Settings have already been processed when module was imported
    print("Fine-tuning SBERT model with the following parameters:")
    print(f"  Model: {SBERT_MODEL}")
    print(f"  Sample fraction: {SAMPLE_FRACTION}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  FP16: {USE_FP16}")
    print(f"  GPU enabled: {USE_GPU}")
    print(f"  Device: {device}")
    print(f"  Output folder: {SBERT_OUTPUT_FOLDER}")

    # The model training actually happens at import time
    # This is a simple stub function to match the CLI's expected interface


if __name__ == "__main__":
    # If directly executed (not imported), run main
    main()
