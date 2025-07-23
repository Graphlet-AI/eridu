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
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from sentence_transformers.losses import ContrastiveLoss
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
from transformers import EarlyStoppingCallback, TrainerCallback

import wandb
from eridu.train.callbacks import ResamplingCallback
from eridu.train.dataset import ResamplingDataset

# from eridu.train.loss import MetricsContextAdaptiveContrastiveLoss
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
RANDOM_SEED: int = int(os.environ.get("RANDOM_SEED", "31337"))
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
SAMPLE_FRACTION: float = float(os.environ.get("SAMPLE_FRACTION", "0.1"))
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
LEARNING_RATE: float = float(os.environ.get("LEARNING_RATE", "3e-5"))
WEIGHT_DECAY: float = float(os.environ.get("WEIGHT_DECAY", "0.01"))
WARMUP_RATIO: float = float(os.environ.get("WARMUP_RATIO", "0.1"))
SAVE_STRATEGY: str = os.environ.get("SAVE_STRATEGY", "epoch")
EVAL_STRATEGY: str = os.environ.get("EVAL_STRATEGY", "epoch")

# Get input path and data type from environment variables
INPUT_PATH: str = os.environ.get("INPUT_PATH", "data/filtered/companies.parquet")
OUTPUT_PATH: str = os.environ.get("OUTPUT_PATH", "data/output")
DATA_TYPE: str = os.environ.get("DATA_TYPE", "companies")

# Include entity type in the output folder name
SBERT_OUTPUT_FOLDER: str = f"data/fine-tuned-sbert-{MODEL_SAVE_NAME}-{DATA_TYPE}"
SAVE_EVAL_STEPS: int = int(os.environ.get("SAVE_EVAL_STEPS", "100"))
USE_FP16: bool = os.environ.get("USE_FP16", "False").lower() == "true"
USE_QUANTIZATION: bool = os.environ.get("USE_QUANTIZATION", "False").lower() == "true"
# Enable gradient checkpointing to save memory if requested
USE_GRADIENT_CHECKPOINTING: bool = (
    os.environ.get("USE_GRADIENT_CHECKPOINTING", "false").lower() == "true"
)

# Enable resampling of training data for each epoch when sample fraction < 1.0
USE_RESAMPLING: bool = os.environ.get("USE_RESAMPLING", "True").lower() == "true"
POST_SAMPLE_PCT: float = float(os.environ.get("POST_SAMPLE_PCT", "0.10"))
MAX_GRAD_NORM: float = float(os.environ.get("MAX_GRAD_NORM", "1.0"))
GATE_STATS_STEPS: int = int(os.environ.get("GATE_STATS_STEPS", "100"))
MARGIN: float = float(os.environ.get("MARGIN", "0.5"))

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
        "input_path": INPUT_PATH,
        "output_path": OUTPUT_PATH,
        "data_type": DATA_TYPE,
        "variant": VARIANT,
        "optimizer": OPTIMIZER,
        "epochs": EPOCHS,
        "sample_fraction": SAMPLE_FRACTION,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "use_fp16": USE_FP16,
        "batch_size": BATCH_SIZE,
        "patience": PATIENCE,
        "early_stopping_patience": PATIENCE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "random_seed": RANDOM_SEED,
        "warmup_ratio": WARMUP_RATIO,
        "max_grad_norm": MAX_GRAD_NORM,
        "margin": MARGIN,
        "save_strategy": os.environ.get("SAVE_STRATEGY", "epoch"),
        "eval_strategy": os.environ.get("EVAL_STRATEGY", "epoch"),
        "sbert_model": SBERT_MODEL,
        "model_save_name": MODEL_SAVE_NAME,
        "sbert_output_folder": SBERT_OUTPUT_FOLDER,
        # "save_eval_steps": SAVE_EVAL_STEPS,
        "use_resampling": USE_RESAMPLING,
        "save_total_limit": 0,  # Save all checkpoints, no limit
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

dataset: pd.DataFrame = pd.read_parquet(INPUT_PATH)

# Display the first few rows of the dataset
print("\nRaw training data sample:\n")
print(dataset.sample(n=20).head())

# Split the dataset into training, evaluation, and test sets
train_df: pd.DataFrame
tmp_df: pd.DataFrame
eval_df: pd.DataFrame
test_df: pd.DataFrame
train_df, tmp_df = train_test_split(dataset, test_size=0.2, random_state=RANDOM_SEED, shuffle=True)
eval_df, test_df = train_test_split(tmp_df, test_size=0.5, random_state=RANDOM_SEED, shuffle=True)

print(f"\nTraining data (full):   {len(train_df):,}")
print(f"Evaluation data:        {len(eval_df):,}")
print(f"Test data:              {len(test_df):,}\n")

# Save the full test split for later evaluation
os.makedirs(SBERT_OUTPUT_FOLDER, exist_ok=True)
test_split_path = os.path.join(SBERT_OUTPUT_FOLDER, "test_split.parquet")
test_df.to_parquet(test_split_path)
print(f"Saved full test split ({len(test_df):,} records) to {test_split_path}")

# Check if we're using resampling
if USE_RESAMPLING and SAMPLE_FRACTION < 1.0:
    print(f"Using resampling with {SAMPLE_FRACTION:.1%} of training data per epoch")
    # Create a resampling dataset that will give us different samples for each epoch
    resampling_dataset = ResamplingDataset(
        train_df, sample_fraction=SAMPLE_FRACTION, random_seed=RANDOM_SEED
    )
    # Get the initial sample for the first epoch
    train_dataset = resampling_dataset.get_current_dataset()
    sample_size = len(train_dataset)
    print(
        f"Initial training sample: {sample_size:,} examples ({SAMPLE_FRACTION:.1%} of full training data)"
    )
else:
    # Without resampling, just sample once if sample_fraction < 1.0
    if SAMPLE_FRACTION < 1.0:
        sampled_train_df = train_df.sample(frac=SAMPLE_FRACTION, random_state=RANDOM_SEED)
        print(f"Training sample: {len(sampled_train_df):,} examples (no resampling)")
    else:
        sampled_train_df = train_df
        print("Using full training dataset (no sampling)")

    # Convert to HuggingFace Dataset
    train_dataset = Dataset.from_dict(
        {
            "sentence1": sampled_train_df["left_name"].tolist(),
            "sentence2": sampled_train_df["right_name"].tolist(),
            "label": sampled_train_df["match"].astype(float).tolist(),
        }
    )

# Convert the evaluation and test sets to HuggingFace Datasets
# Use float instead of bool for labels to avoid the subtraction error with boolean tensors
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
if USE_GRADIENT_CHECKPOINTING:
    sbert_model.gradient_checkpointing_enable()
    logger.debug("Gradient checkpointing enabled to save memory")

# Put network in training mode
sbert_model.train()

# Validate that FP16 and quantization are not both enabled
if USE_FP16 and USE_QUANTIZATION:
    raise ValueError(
        "Error: Cannot use both FP16 and quantization together. Please choose only one option."
    )

# Apply quantization if explicitly requested
if USE_QUANTIZATION:
    logger.debug("Applying quantization to Linear layers in the encoder")
    # Tell PyTorch to quantize the Linear layers in the encoder
    for module in sbert_model.modules():
        if isinstance(module, torch.nn.Linear):
            module.qconfig = tq.get_default_qat_qconfig("fbgemm")

    # Prepare QAT: inserts FakeQuant and Observer modules
    tq.prepare_qat(sbert_model, inplace=True)

#
# Try the SBERT model out without fine-tuning. Multi-lingual comaprisons work somewhat.
#

print("\nTesting raw [un-fine-tuned] SBERT model:\n")
examples: list[list[str | float | bool]] = []

name_pairs: list[dict[str, str | bool]] = [
    {"sentence1": "Acme Corp.", "sentence2": "Acme Corporation", "similarity": True},
    {
        "sentence1": "Beta Technologies LLC",
        "sentence2": "Beta Technologies, LLC",
        "similarity": True,
    },
    {
        "sentence1": "Global Solutions, Inc",
        "sentence2": "Global Solutions Inc.",
        "similarity": True,
    },
    {"sentence1": "Johnson & Johnson", "sentence2": "Johnson and Johnson", "similarity": True},
    {"sentence1": "The Coca-Cola Company", "sentence2": "Coca Cola Co.", "similarity": True},
    {"sentence1": "Acme Corp", "sentence2": "Acme Tools Inc.", "similarity": False},
    {
        "sentence1": "Alpha Capital LLC",
        "sentence2": "Alpha Capital Partners LLC",
        "similarity": False,
    },
    {
        "sentence1": "Tech Innovations Inc.",
        "sentence2": "Tech Innovation Inc.",
        "similarity": False,
    },
    {"sentence1": "Sunrise Energy AG", "sentence2": "Sunrise Energy GmbH", "similarity": False},
    {
        "sentence1": "Future Holdings PLC",
        "sentence2": "Future Holdings Limited",
        "similarity": False,
    },
]

for name_pair in name_pairs:
    examples.append(
        [
            name_pair["sentence1"],
            name_pair["sentence2"],
            sbert_compare(
                sbert_model,
                str(name_pair["sentence1"]),
                str(name_pair["sentence2"]),
                use_gpu=True,
            ),
            name_pair["similarity"],
        ]
    )

examples_df: pd.DataFrame = pd.DataFrame(
    examples, columns=["sentence1", "sentence2", "similarity", "expected"]
)
print(str(examples_df) + "\n")

#
# Evaluate a sample of the evaluation data compared using raw SBERT before fine-tuning
#

# Use a simple approach for evaluation sampling - use the same sample fraction as for training
sample_df: pd.DataFrame = eval_df.sample(frac=SAMPLE_FRACTION, random_state=RANDOM_SEED)

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

# # This will effectively train the embedding model. MultipleNegativesRankingLoss did not work.
loss: ContrastiveLoss = ContrastiveLoss(model=sbert_model, margin=MARGIN)

# # Use MetricsOnlineContrastiveLoss for detailed loss tracking
# loss: MetricsOnlineContrastiveLoss = MetricsOnlineContrastiveLoss(
#     model=sbert_model,
#     margin=0.5,  # Margin for contrastive loss
#     distance_metric=SiameseDistanceMetric.COSINE_DISTANCE,
#     gate_stats_steps=GATE_STATS_STEPS,
# )

# loss: MetricsContextAdaptiveContrastiveLoss = MetricsContextAdaptiveContrastiveLoss(
#     model=sbert_model,
#     margin=MARGIN,  # Margin for contrastive loss
#     gate_scale=5.0,  # Scale for the gate function
# )

# Set lots of options to reduce memory usage and improve training speed
sbert_args: SentenceTransformerTrainingArguments = SentenceTransformerTrainingArguments(
    output_dir=SBERT_OUTPUT_FOLDER,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    fp16=USE_FP16,
    fp16_opt_level="O1" if USE_FP16 else "O0",
    warmup_ratio=WARMUP_RATIO,
    run_name=SBERT_MODEL,
    load_best_model_at_end=True,
    save_total_limit=0,  # Save all checkpoints, no limit
    save_steps=SAVE_EVAL_STEPS,
    eval_steps=SAVE_EVAL_STEPS,
    save_strategy=SAVE_STRATEGY,
    eval_strategy=EVAL_STRATEGY,
    greater_is_better=False,
    metric_for_best_model="eval_loss",
    learning_rate=LEARNING_RATE,
    logging_dir="./logs",
    weight_decay=WEIGHT_DECAY,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
    max_grad_norm=MAX_GRAD_NORM,
    optim=OPTIMIZER,
)

# Prepare the callbacks (WandbCallback is automatically added by SentenceTransformerTrainer when wandb is initialized)
callbacks: list[TrainerCallback] = [
    EarlyStoppingCallback(early_stopping_patience=PATIENCE),
]

# Add resampling callback if resampling is enabled
if USE_RESAMPLING and SAMPLE_FRACTION < 1.0:
    resampling_callback = ResamplingCallback(resampling_dataset, epochs=EPOCHS)
    callbacks.append(resampling_callback)

trainer: SentenceTransformerTrainer = SentenceTransformerTrainer(
    model=sbert_model,
    args=sbert_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
    evaluator=binary_acc_evaluator,
    compute_metrics=compute_sbert_metrics,  # type: ignore
    callbacks=callbacks,
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
tuned_examples_df: pd.DataFrame = pd.DataFrame(
    examples, columns=["sentence1", "sentence2", "similarity", "expected"]
)
print(str(tuned_examples_df) + "\n")

#
# Evaluate ROC curve of the fine-tuned model and determine optimal threshold
#

# Use the SAMPLE_FRACTION but ensure we have at least 1,000 test records (or all available if fewer)
test_sample_size = max(int(len(test_df) * POST_SAMPLE_PCT), min(len(test_df), 1000))
test_df = test_df.sample(n=test_sample_size, random_state=RANDOM_SEED)

# Create a copy of test_df with predictions for later analysis
results_test_df = test_df.copy()

# Use GPU acceleration for inference
print(f"Running inference on {device} for {len(test_df):,} test records")

y_scores: np.ndarray[Any, Any] = sbert_compare_multiple(
    sbert_model, test_df["left_name"], test_df["right_name"], use_gpu=True
)

# Store scores in results dataframe
results_test_df["similarity"] = y_scores
results_test_df["true_label"] = test_df["match"].astype(float)

y_true: list[float] = test_df["match"].astype(float).tolist()

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

print(f"Best Epoch (threshold index): {best_threshold_index + 1}")
print(f"Best Threshold: {best_threshold}")
print(f"Best F1 Score: {best_f1_score}")

roc_auc: float = roc_auc_score(y_true, y_scores)
print(f"AUC-ROC: {roc_auc}")

# Add best threshold and predicted labels to the results dataframe
results_test_df["best_threshold"] = best_threshold
results_test_df["predicted_match"] = results_test_df["similarity"] >= best_threshold
results_test_df["correct_prediction"] = (
    results_test_df["predicted_match"] == results_test_df["true_label"]
)

# Save test results to parquet in the same folder as the model
os.makedirs(OUTPUT_PATH, exist_ok=True)
results_path = os.path.join(OUTPUT_PATH, "test_results.parquet")
results_test_df.to_parquet(results_path)
print(f"Saved test results to {results_path}")

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
    print(f"  Early stopping patience: {PATIENCE}")
    if SAMPLE_FRACTION < 1.0:
        print(f"  Resampling enabled: {USE_RESAMPLING}")
        if USE_RESAMPLING:
            print(f"  Training data will be re-sampled ({SAMPLE_FRACTION:.1%}) for each epoch")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Weight decay: {WEIGHT_DECAY}")
    print(f"  Random seed: {RANDOM_SEED}")
    print(f"  Warmup ratio: {WARMUP_RATIO}")
    print(f"  Save strategy: {SAVE_STRATEGY}")
    print(f"  Eval strategy: {EVAL_STRATEGY}")
    print(f"  FP16: {USE_FP16}")
    print(f"  Quantization: {USE_QUANTIZATION}")
    print(f"  GPU enabled: {USE_GPU}")
    print(f"  Device: {device}")
    print(f"  Output folder: {SBERT_OUTPUT_FOLDER}")
    print(f"  Gradient checkpointing: {USE_GRADIENT_CHECKPOINTING}")

    # The model training actually happens at import time
    # This is a simple stub function to match the CLI's expected interface


if __name__ == "__main__":
    # If directly executed (not imported), run main
    main()
