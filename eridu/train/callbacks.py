"""Custom callbacks for training models in Eridu."""

import logging
from typing import Optional, cast

import pandas as pd
from datasets import Dataset
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from eridu.train.dataset import ResamplingDataset

logger = logging.getLogger(__name__)


class ResamplingCallback(TrainerCallback):
    """Callback to resample the training dataset at the beginning of each epoch.

    This callback works with the ResamplingDataset class to create a new sample
    from the full dataset at the beginning of each epoch, allowing the model to
    see different data when training with a subset of the full dataset.
    """

    def __init__(
        self,
        resampling_dataset: ResamplingDataset,
        epochs: int = 10,
        cleanco_df: Optional[pd.DataFrame] = None,
    ):
        """Initialize the resampling callback.

        Args:
            resampling_dataset: The dataset with resampling capability
            epochs: Total number of epochs for logging purposes
            cleanco_df: Optional cleanco dataframe to always include in training
        """
        self.dataset = resampling_dataset
        self.total_epochs = epochs
        self.current_epoch = 0
        self.sample_fraction = resampling_dataset.sample_fraction
        self.cleanco_df = cleanco_df

    def on_epoch_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        """Called at the beginning of each epoch.

        This method resamples the training dataset for the new epoch.

        Args:
            args: Training arguments
            state: Trainer state
            control: Training control
            **kwargs: Additional keyword arguments
        """
        # Skip if we're using the full dataset (no need to resample)
        if self.sample_fraction >= 1.0:
            return

        # We need an integer epoch for resampling (state.epoch can be a float)
        int_epoch = int(cast(float, state.epoch))

        # Only resample if the epoch has changed
        if int_epoch != self.current_epoch:
            self.current_epoch = int_epoch

            # Resample the dataset for this epoch
            new_dataset = self.dataset.resample_for_epoch(int_epoch)

            # Add cleanco data to the resampled dataset
            if self.cleanco_df is not None:
                # Convert current dataset back to DataFrame to merge
                current_df = pd.DataFrame(
                    {
                        "left_name": new_dataset["sentence1"],
                        "right_name": new_dataset["sentence2"],
                        "match": new_dataset["label"],
                    }
                )
                # Add cleanco data
                combined_df = pd.concat(
                    [current_df, self.cleanco_df[["left_name", "right_name", "match"]]],
                    ignore_index=True,
                )
                # Convert back to Dataset
                new_dataset = Dataset.from_dict(
                    {
                        "sentence1": combined_df["left_name"].tolist(),
                        "sentence2": combined_df["right_name"].tolist(),
                        "label": combined_df["match"].astype(float).tolist(),
                    }
                )

            # Update the trainer's train_dataset with the new sampled dataset
            if "trainer" in kwargs and hasattr(kwargs["trainer"], "train_dataset"):
                kwargs["trainer"].train_dataset = new_dataset

                # Log the resampling
                if self.cleanco_df is not None:
                    logger.info(
                        f"Epoch {int_epoch + 1}/{self.total_epochs}: "
                        f"Resampled training dataset with {len(new_dataset)} examples "
                        f"({self.sample_fraction:.1%} of full dataset + {len(self.cleanco_df)} cleanco)"
                    )
                else:
                    logger.info(
                        f"Epoch {int_epoch + 1}/{self.total_epochs}: "
                        f"Resampled training dataset with {len(new_dataset)} examples "
                        f"({self.sample_fraction:.1%} of full dataset)"
                    )
            else:
                logger.warning(
                    "Unable to update trainer's dataset: "
                    "trainer or train_dataset not available in kwargs"
                )

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        """Called at the beginning of training.

        Ensures the dataset is properly initialized with the first sample.

        Args:
            args: Training arguments
            state: Trainer state
            control: Training control
            **kwargs: Additional keyword arguments
        """
        # Reset the current epoch counter
        self.current_epoch = 0

        # Log the initial sampling
        if self.sample_fraction < 1.0:
            logger.info(
                f"Training begins: Using resampling with {self.sample_fraction:.1%} "
                f"of full dataset per epoch"
            )
        else:
            logger.info("Training begins: Using full dataset (no resampling)")
