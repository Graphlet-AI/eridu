"""Dataset classes for Eridu training."""

import pandas as pd
from datasets import Dataset  # type: ignore


class ResamplingDataset:
    """A dataset that can resample from a larger dataset on each epoch.

    This class wraps around the Hugging Face Dataset class and provides
    functionality to resample the dataset with a different random seed
    for each epoch, allowing the model to see different samples in each epoch
    when training with a subset of the full dataset.
    """

    def __init__(
        self,
        full_dataset: pd.DataFrame,
        sample_fraction: float = 1.0,
        random_seed: int = 31337,  # Use the same default as RANDOM_SEED
        left_name_col: str = "left_name",
        right_name_col: str = "right_name",
        match_col: str = "match",
    ):
        """Initialize the resampling dataset.

        Args:
            full_dataset: The complete dataset to sample from
            sample_fraction: Fraction of data to sample (1.0 = use all data)
            random_seed: Base random seed for reproducibility
            left_name_col: Column name for the left entity name
            right_name_col: Column name for the right entity name
            match_col: Column name for the match indicator
        """
        self.full_dataset = full_dataset
        self.sample_fraction = sample_fraction
        self.base_random_seed = random_seed
        self.current_epoch = 0
        self.left_name_col = left_name_col
        self.right_name_col = right_name_col
        self.match_col = match_col

        # Initialize with the first sample
        self.current_dataset = self._create_initial_sample()

    def _create_initial_sample(self) -> Dataset:
        """Create the initial dataset sample.

        Returns:
            A HuggingFace Dataset object with the sampled data
        """
        # Sample the dataset if sample_fraction < 1.0
        if self.sample_fraction < 1.0:
            sampled_df = self.full_dataset.sample(
                frac=self.sample_fraction, random_state=self.base_random_seed + self.current_epoch
            )
        else:
            sampled_df = self.full_dataset

        # Convert to HuggingFace Dataset
        return Dataset.from_dict(
            {
                "sentence1": sampled_df[self.left_name_col].tolist(),
                "sentence2": sampled_df[self.right_name_col].tolist(),
                # Use float instead of bool for labels to avoid subtraction error with boolean tensors
                "label": sampled_df[self.match_col].astype(float).tolist(),
            }
        )

    def resample_for_epoch(self, epoch: int) -> Dataset:
        """Resample the dataset for a specific epoch.

        Args:
            epoch: The epoch number (0-indexed)

        Returns:
            A HuggingFace Dataset object with the newly sampled data
        """
        self.current_epoch = epoch

        # If we're using the full dataset, no need to resample
        if self.sample_fraction >= 1.0:
            return self.current_dataset

        # Sample with a seed based on the epoch
        current_seed = self.base_random_seed + epoch
        sampled_df = self.full_dataset.sample(frac=self.sample_fraction, random_state=current_seed)

        # Convert to HuggingFace Dataset
        self.current_dataset = Dataset.from_dict(
            {
                "sentence1": sampled_df[self.left_name_col].tolist(),
                "sentence2": sampled_df[self.right_name_col].tolist(),
                # Use float instead of bool for labels
                "label": sampled_df[self.match_col].astype(float).tolist(),
            }
        )

        return self.current_dataset

    def get_current_dataset(self) -> Dataset:
        """Get the current dataset sample.

        Returns:
            The current HuggingFace Dataset
        """
        return self.current_dataset

    def __len__(self) -> int:
        """Get the length of the current dataset.

        Returns:
            The number of samples in the current dataset
        """
        return len(self.current_dataset)
