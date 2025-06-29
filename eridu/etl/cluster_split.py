"""Cluster-aware data splitting to prevent overfitting by ensuring tight clusters are entirely in one partition."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split  # type: ignore

logger = logging.getLogger(__name__)


def load_cluster_results(cluster_file: Path) -> pd.DataFrame:
    """Load clustering results from CSV file.

    Parameters
    ----------
    cluster_file : Path
        Path to cluster results CSV file

    Returns
    -------
    pd.DataFrame
        DataFrame with cluster assignments for each name
    """
    if not cluster_file.exists():
        raise FileNotFoundError(f"Cluster file not found: {cluster_file}")

    df = pd.read_csv(cluster_file)
    logger.info(f"Loaded {len(df)} clustered names from {cluster_file}")

    return df


def identify_tight_clusters(
    cluster_df: pd.DataFrame, intra_distance_threshold: float = 0.2, min_cluster_size: int = 5
) -> List[int]:
    """Identify tight clusters based on intra-cluster distance.

    Parameters
    ----------
    cluster_df : pd.DataFrame
        DataFrame with cluster information
    intra_distance_threshold : float
        Maximum intra-cluster distance to consider a cluster "tight"
    min_cluster_size : int
        Minimum cluster size to consider

    Returns
    -------
    List[int]
        List of cluster IDs that are considered tight
    """
    # Calculate cluster metrics
    cluster_metrics = (
        cluster_df.groupby("cluster")
        .agg({"name": "count", "intra_cluster_mean_dist": "mean"})
        .rename(columns={"name": "size"})
    )

    # Filter clusters that are both tight and have minimum size
    tight_clusters = cluster_metrics[
        (cluster_metrics["intra_cluster_mean_dist"] <= intra_distance_threshold)
        & (cluster_metrics["size"] >= min_cluster_size)
        & (cluster_metrics.index != -1)  # Exclude noise cluster
    ]

    tight_cluster_ids = tight_clusters.index.tolist()
    logger.info(f"Identified {len(tight_cluster_ids)} tight clusters: {tight_cluster_ids}")

    # Log details about each tight cluster
    for cluster_id in tight_cluster_ids:
        size = cluster_metrics.loc[cluster_id, "size"]
        intra_dist = cluster_metrics.loc[cluster_id, "intra_cluster_mean_dist"]
        logger.info(f"  Cluster {cluster_id}: {size} names, intra-distance={intra_dist:.4f}")

    return tight_cluster_ids


def assign_clusters_to_partitions(
    tight_cluster_ids: List[int],
    train_ratio: float = 0.7,
    test_ratio: float = 0.2,
    eval_ratio: float = 0.1,
    random_state: int = 42,
) -> Dict[int, str]:
    """Assign each tight cluster to exactly one partition.

    Parameters
    ----------
    tight_cluster_ids : List[int]
        List of tight cluster IDs
    train_ratio : float
        Target proportion for training set
    test_ratio : float
        Target proportion for test set
    eval_ratio : float
        Target proportion for evaluation set
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    Dict[int, str]
        Mapping from cluster ID to partition name ('train', 'test', 'eval')
    """
    np.random.seed(random_state)

    # Shuffle clusters for random assignment
    shuffled_clusters = tight_cluster_ids.copy()
    np.random.shuffle(shuffled_clusters)

    # Calculate target counts for each partition
    total_clusters = len(tight_cluster_ids)
    train_count = max(1, int(total_clusters * train_ratio))
    test_count = max(1, int(total_clusters * test_ratio))
    eval_count = total_clusters - train_count - test_count

    # Ensure eval gets at least one if there are clusters left
    if eval_count <= 0 and total_clusters > train_count + test_count:
        eval_count = 1
        test_count = max(1, test_count - 1)

    logger.info(f"Assigning {total_clusters} tight clusters to partitions:")
    logger.info(f"  Train: {train_count} clusters")
    logger.info(f"  Test: {test_count} clusters")
    logger.info(f"  Eval: {eval_count} clusters")

    # Assign clusters to partitions
    cluster_assignments = {}
    idx = 0

    # Assign to train
    for i in range(train_count):
        if idx < len(shuffled_clusters):
            cluster_assignments[shuffled_clusters[idx]] = "train"
            idx += 1

    # Assign to test
    for i in range(test_count):
        if idx < len(shuffled_clusters):
            cluster_assignments[shuffled_clusters[idx]] = "test"
            idx += 1

    # Assign remaining to eval
    while idx < len(shuffled_clusters):
        cluster_assignments[shuffled_clusters[idx]] = "eval"
        idx += 1

    # Log assignments
    for partition in ["train", "test", "eval"]:
        assigned_clusters = [cid for cid, part in cluster_assignments.items() if part == partition]
        logger.info(f"  {partition.capitalize()} clusters: {assigned_clusters}")

    return cluster_assignments


def _detect_column_names(pairs_df: pd.DataFrame) -> Tuple[str, str, str]:
    """Detect column naming convention in pairs DataFrame."""
    if "name1" in pairs_df.columns and "name2" in pairs_df.columns:
        return "name1", "name2", "label"
    elif "left_name" in pairs_df.columns and "right_name" in pairs_df.columns:
        return "left_name", "right_name", "match"
    else:
        raise ValueError(
            "Could not detect name columns. Expected either (name1, name2) or (left_name, right_name)"
        )


def _assign_pair_to_partition(row: pd.Series, cluster_assignments: Dict[int, str]) -> Optional[str]:
    """Assign a pair to a partition based on cluster membership."""
    name1_cluster = row["name1_cluster"]
    name2_cluster = row["name2_cluster"]

    # Check if either name belongs to a tight cluster
    name1_partition = None
    name2_partition = None

    if pd.notna(name1_cluster) and name1_cluster in cluster_assignments:
        name1_partition = cluster_assignments[name1_cluster]

    if pd.notna(name2_cluster) and name2_cluster in cluster_assignments:
        name2_partition = cluster_assignments[name2_cluster]

    # Assign pair based on cluster membership
    if name1_partition is not None and name2_partition is not None:
        # Both names in tight clusters - they must be in the same partition
        if name1_partition == name2_partition:
            return name1_partition
        else:
            # Conflict: names from different tight clusters assigned to different partitions
            # Assign to the partition of the first name (arbitrary but consistent)
            logger.warning(
                f"Pair with names from different tight cluster partitions: "
                f"cluster {name1_cluster} -> {name1_partition}, "
                f"cluster {name2_cluster} -> {name2_partition}. "
                f"Assigning to {name1_partition}"
            )
            return name1_partition
    elif name1_partition is not None:
        # Only name1 in tight cluster
        return name1_partition
    elif name2_partition is not None:
        # Only name2 in tight cluster
        return name2_partition
    else:
        # Neither name in tight cluster - assign to unassigned for later random split
        return None


def cluster_aware_split(
    pairs_df: pd.DataFrame,
    cluster_df: pd.DataFrame,
    tight_cluster_ids: List[int],
    cluster_assignments: Dict[int, str],
    train_ratio: float = 0.7,
    test_ratio: float = 0.2,
    eval_ratio: float = 0.1,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data ensuring tight clusters are entirely in one partition each.

    Parameters
    ----------
    pairs_df : pd.DataFrame
        DataFrame containing name pairs
    cluster_df : pd.DataFrame
        DataFrame with cluster assignments
    tight_cluster_ids : List[int]
        List of tight cluster IDs
    cluster_assignments : Dict[int, str]
        Assignment of each tight cluster to a partition
    train_ratio : float
        Proportion for training set (for non-tight cluster pairs)
    test_ratio : float
        Proportion for test set (for non-tight cluster pairs)
    eval_ratio : float
        Proportion for evaluation set (for non-tight cluster pairs)
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Train, test, and eval DataFrames
    """
    # Create cluster lookup for names
    cluster_lookup = dict(zip(cluster_df["name"], cluster_df["cluster"]))

    # Detect column naming convention
    name1_col, name2_col, label_col = _detect_column_names(pairs_df)
    logger.info(f"Using columns: {name1_col}, {name2_col}, label: {label_col}")

    # Add cluster information to pairs
    pairs_df = pairs_df.copy()
    pairs_df["name1_cluster"] = pairs_df[name1_col].map(cluster_lookup)
    pairs_df["name2_cluster"] = pairs_df[name2_col].map(cluster_lookup)

    # Separate pairs into partitions based on cluster assignments
    train_pairs = []
    test_pairs = []
    eval_pairs = []
    unassigned_pairs = []

    for idx, row in pairs_df.iterrows():
        partition = _assign_pair_to_partition(row, cluster_assignments)
        if partition is None:
            unassigned_pairs.append(row)
        elif partition == "train":
            train_pairs.append(row)
        elif partition == "test":
            test_pairs.append(row)
        elif partition == "eval":
            eval_pairs.append(row)

    # Convert lists to DataFrames
    train_df = pd.DataFrame(train_pairs) if train_pairs else pd.DataFrame(columns=pairs_df.columns)
    test_df = pd.DataFrame(test_pairs) if test_pairs else pd.DataFrame(columns=pairs_df.columns)
    eval_df = pd.DataFrame(eval_pairs) if eval_pairs else pd.DataFrame(columns=pairs_df.columns)
    unassigned_df = (
        pd.DataFrame(unassigned_pairs)
        if unassigned_pairs
        else pd.DataFrame(columns=pairs_df.columns)
    )

    logger.info("Assigned pairs based on tight clusters:")
    logger.info(f"  Train: {len(train_df)} pairs")
    logger.info(f"  Test: {len(test_df)} pairs")
    logger.info(f"  Eval: {len(eval_df)} pairs")
    logger.info(f"  Unassigned: {len(unassigned_df)} pairs")

    # Randomly split unassigned pairs
    if len(unassigned_df) > 0:
        unassigned_train, unassigned_temp = train_test_split(
            unassigned_df,
            test_size=(test_ratio + eval_ratio),
            random_state=random_state,
            stratify=unassigned_df[label_col] if label_col in unassigned_df.columns else None,
        )

        if len(unassigned_temp) > 0:
            unassigned_test, unassigned_eval = train_test_split(
                unassigned_temp,
                test_size=eval_ratio / (test_ratio + eval_ratio),
                random_state=random_state,
                stratify=(
                    unassigned_temp[label_col] if label_col in unassigned_temp.columns else None
                ),
            )
        else:
            unassigned_test = pd.DataFrame(columns=pairs_df.columns)
            unassigned_eval = pd.DataFrame(columns=pairs_df.columns)

        # Combine with cluster-assigned pairs
        train_df = pd.concat([train_df, unassigned_train], ignore_index=True)
        test_df = pd.concat([test_df, unassigned_test], ignore_index=True)
        eval_df = pd.concat([eval_df, unassigned_eval], ignore_index=True)

    # Remove cluster columns from final output
    columns_to_drop = ["name1_cluster", "name2_cluster"]
    train_df = train_df.drop(columns=columns_to_drop, errors="ignore")
    test_df = test_df.drop(columns=columns_to_drop, errors="ignore")
    eval_df = eval_df.drop(columns=columns_to_drop, errors="ignore")

    logger.info("Final cluster-aware splits:")
    logger.info(f"  Train: {len(train_df)} pairs ({len(train_df) / len(pairs_df) * 100:.1f}%)")
    logger.info(f"  Test: {len(test_df)} pairs ({len(test_df) / len(pairs_df) * 100:.1f}%)")
    logger.info(f"  Eval: {len(eval_df)} pairs ({len(eval_df) / len(pairs_df) * 100:.1f}%)")

    return train_df, test_df, eval_df


def analyze_cluster_distribution(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    cluster_df: pd.DataFrame,
    tight_cluster_ids: List[int],
) -> Dict[str, Dict[int, int]]:
    """Analyze how tight clusters are distributed across splits.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data
    test_df : pd.DataFrame
        Test data
    eval_df : pd.DataFrame
        Evaluation data
    cluster_df : pd.DataFrame
        Cluster assignments
    tight_cluster_ids : List[int]
        List of tight cluster IDs

    Returns
    -------
    Dict[str, Dict[int, int]]
        Distribution of tight clusters across splits
    """
    distribution: Dict[str, Dict[int, int]] = {"train": {}, "test": {}, "eval": {}}

    # Detect column naming convention
    sample_df = train_df if len(train_df) > 0 else (test_df if len(test_df) > 0 else eval_df)
    try:
        name1_col, name2_col, _ = _detect_column_names(sample_df)
    except ValueError:
        logger.warning("Could not detect name columns for analysis")
        return distribution

    for split_name, split_df in [("train", train_df), ("test", test_df), ("eval", eval_df)]:
        if len(split_df) == 0:
            continue
        # Get all names in this split
        all_names = pd.concat([split_df[name1_col], split_df[name2_col]]).unique()

        # Count names from each tight cluster
        for cluster_id in tight_cluster_ids:
            cluster_names = cluster_df[cluster_df["cluster"] == cluster_id]["name"].values
            names_in_split = len(set(all_names) & set(cluster_names))
            distribution[split_name][cluster_id] = names_in_split

    # Log distribution and verify no cluster spans multiple partitions
    logger.info("Tight cluster distribution across splits:")
    for cluster_id in tight_cluster_ids:
        train_count = distribution["train"][cluster_id]
        test_count = distribution["test"][cluster_id]
        eval_count = distribution["eval"][cluster_id]

        # Count non-zero partitions
        partitions_with_cluster = sum([train_count > 0, test_count > 0, eval_count > 0])

        if partitions_with_cluster > 1:
            logger.warning(
                f"Cluster {cluster_id} spans multiple partitions! "
                f"Train={train_count}, Test={test_count}, Eval={eval_count}"
            )
        else:
            partition = "train" if train_count > 0 else ("test" if test_count > 0 else "eval")
            count = max(train_count, test_count, eval_count)
            logger.info(f"  Cluster {cluster_id}: {count} names in {partition}")

    return distribution


def create_cluster_aware_splits(
    pairs_file: Union[str, Path],
    cluster_file: Union[str, Path],
    output_dir: Union[str, Path],
    intra_distance_threshold: float = 0.2,
    min_cluster_size: int = 5,
    train_ratio: float = 0.7,
    test_ratio: float = 0.2,
    eval_ratio: float = 0.1,
    random_state: int = 42,
) -> None:
    """Create cluster-aware train/test/eval splits with each tight cluster in one partition.

    Parameters
    ----------
    pairs_file : Union[str, Path]
        Path to pairs file (CSV or Parquet)
    cluster_file : Union[str, Path]
        Path to cluster results CSV
    output_dir : Union[str, Path]
        Directory to save split files
    intra_distance_threshold : float
        Maximum intra-cluster distance for tight clusters
    min_cluster_size : int
        Minimum size for tight clusters
    train_ratio : float
        Training set proportion
    test_ratio : float
        Test set proportion
    eval_ratio : float
        Evaluation set proportion
    random_state : int
        Random seed
    """
    pairs_file = Path(pairs_file)
    cluster_file = Path(cluster_file)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info(f"Loading pairs from {pairs_file}")
    if pairs_file.suffix == ".parquet":
        pairs_df = pd.read_parquet(pairs_file)
    else:
        pairs_df = pd.read_csv(pairs_file)

    cluster_df = load_cluster_results(cluster_file)

    # Identify tight clusters
    tight_cluster_ids = identify_tight_clusters(
        cluster_df, intra_distance_threshold, min_cluster_size
    )

    if not tight_cluster_ids:
        logger.warning("No tight clusters found, falling back to standard split")
        # Standard split
        train_df, temp_df = train_test_split(
            pairs_df, test_size=(test_ratio + eval_ratio), random_state=random_state
        )
        test_df, eval_df = train_test_split(
            temp_df, test_size=eval_ratio / (test_ratio + eval_ratio), random_state=random_state
        )
    else:
        # Assign tight clusters to partitions
        cluster_assignments = assign_clusters_to_partitions(
            tight_cluster_ids, train_ratio, test_ratio, eval_ratio, random_state
        )

        # Cluster-aware split
        train_df, test_df, eval_df = cluster_aware_split(
            pairs_df,
            cluster_df,
            tight_cluster_ids,
            cluster_assignments,
            train_ratio,
            test_ratio,
            eval_ratio,
            random_state,
        )

        # Analyze distribution to verify no cluster spans multiple partitions
        analyze_cluster_distribution(train_df, test_df, eval_df, cluster_df, tight_cluster_ids)

    # Save splits
    train_file = output_dir / "train.parquet"
    test_file = output_dir / "test.parquet"
    eval_file = output_dir / "eval.parquet"

    train_df.to_parquet(train_file, index=False)
    test_df.to_parquet(test_file, index=False)
    eval_df.to_parquet(eval_file, index=False)

    logger.info(f"Saved cluster-aware splits to {output_dir}")
    logger.info(f"  Train: {train_file} ({len(train_df)} pairs)")
    logger.info(f"  Test: {test_file} ({len(test_df)} pairs)")
    logger.info(f"  Eval: {eval_file} ({len(eval_df)} pairs)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create cluster-aware data splits")
    parser.add_argument("--pairs-file", required=True, help="Path to pairs file")
    parser.add_argument("--cluster-file", required=True, help="Path to cluster results CSV")
    parser.add_argument("--output-dir", required=True, help="Output directory for splits")
    parser.add_argument(
        "--intra-threshold",
        type=float,
        default=0.2,
        help="Intra-cluster distance threshold for tight clusters",
    )
    parser.add_argument(
        "--min-size", type=int, default=5, help="Minimum cluster size for tight clusters"
    )
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Training set ratio")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Test set ratio")
    parser.add_argument("--eval-ratio", type=float, default=0.1, help="Eval set ratio")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    create_cluster_aware_splits(
        pairs_file=args.pairs_file,
        cluster_file=args.cluster_file,
        output_dir=args.output_dir,
        intra_distance_threshold=args.intra_threshold,
        min_cluster_size=args.min_size,
        train_ratio=args.train_ratio,
        test_ratio=args.test_ratio,
        eval_ratio=args.eval_ratio,
        random_state=args.random_state,
    )
