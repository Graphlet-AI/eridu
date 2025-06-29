"""Name clustering using embeddings, T-SNE, and HDBSCAN."""

import warnings
from pathlib import Path
from typing import Optional

import cuml  # type: ignore
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE  # type: ignore
from sklearn.metrics.pairwise import cosine_distances  # type: ignore
from sklearn.preprocessing import normalize  # type: ignore

# Suppress SyntaxWarnings from HDBSCAN library - these are from the library itself, not our code
warnings.filterwarnings("ignore", category=SyntaxWarning)
# Suppress sklearn deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")


def _compute_distance_metrics(
    unique_names: np.ndarray,
    cluster_labels: np.ndarray,
    embeddings_normalized: np.ndarray,
) -> pd.DataFrame:
    """Compute high-dimensional distance metrics for cluster analysis."""
    print("Computing high-dimensional distance metrics...")

    # Compute cosine distances in high-dimensional space (more meaningful for embeddings)
    cosine_dist_matrix = cosine_distances(embeddings_normalized)

    # For each point, compute distances to cluster centroids and other cluster members
    cluster_distance_metrics = []

    for i, (name, cluster_id) in enumerate(zip(unique_names, cluster_labels)):
        metrics = {"name": name, "index": i}

        if cluster_id != -1:  # Not noise
            # Find all points in the same cluster
            same_cluster_mask = cluster_labels == cluster_id
            same_cluster_indices = np.where(same_cluster_mask)[0]

            # Compute intra-cluster distances (to other members of same cluster)
            if len(same_cluster_indices) > 1:
                intra_distances = cosine_dist_matrix[i, same_cluster_indices]
                intra_distances = intra_distances[intra_distances > 0]  # Remove self-distance (0)

                metrics.update(
                    {
                        "intra_cluster_mean_dist": (
                            intra_distances.mean() if len(intra_distances) > 0 else 0.0
                        ),
                        "intra_cluster_max_dist": (
                            intra_distances.max() if len(intra_distances) > 0 else 0.0
                        ),
                        "intra_cluster_min_dist": (
                            intra_distances.min() if len(intra_distances) > 0 else 0.0
                        ),
                    }
                )
            else:
                metrics.update(
                    {
                        "intra_cluster_mean_dist": 0.0,
                        "intra_cluster_max_dist": 0.0,
                        "intra_cluster_min_dist": 0.0,
                    }
                )

            # Compute inter-cluster distances (to points in other clusters)
            other_cluster_mask = (cluster_labels != cluster_id) & (cluster_labels != -1)
            if other_cluster_mask.any():
                other_cluster_indices = np.where(other_cluster_mask)[0]
                inter_distances = cosine_dist_matrix[i, other_cluster_indices]

                metrics.update(
                    {
                        "inter_cluster_mean_dist": inter_distances.mean(),
                        "inter_cluster_min_dist": inter_distances.min(),
                        "nearest_other_cluster_dist": inter_distances.min(),
                    }
                )
            else:
                metrics.update(
                    {
                        "inter_cluster_mean_dist": np.nan,
                        "inter_cluster_min_dist": np.nan,
                        "nearest_other_cluster_dist": np.nan,
                    }
                )

            # Compute cluster centroid distance
            cluster_centroid = embeddings_normalized[same_cluster_mask].mean(axis=0)
            centroid_distance = cosine_distances([embeddings_normalized[i]], [cluster_centroid])[
                0, 0
            ]
            metrics["distance_to_centroid"] = centroid_distance

        else:  # Noise point
            # For noise points, compute distance to nearest cluster
            non_noise_mask = cluster_labels != -1
            if non_noise_mask.any():
                non_noise_indices = np.where(non_noise_mask)[0]
                distances_to_clusters = cosine_dist_matrix[i, non_noise_indices]
                nearest_cluster_dist = distances_to_clusters.min()
                metrics["nearest_cluster_dist"] = nearest_cluster_dist
            else:
                metrics["nearest_cluster_dist"] = np.nan

            # Set other metrics to NaN for noise points
            metrics.update(
                {
                    "intra_cluster_mean_dist": np.nan,
                    "intra_cluster_max_dist": np.nan,
                    "intra_cluster_min_dist": np.nan,
                    "inter_cluster_mean_dist": np.nan,
                    "inter_cluster_min_dist": np.nan,
                    "nearest_other_cluster_dist": np.nan,
                    "distance_to_centroid": np.nan,
                }
            )

        cluster_distance_metrics.append(metrics)

    # Convert to DataFrame and return
    distance_df = pd.DataFrame(cluster_distance_metrics)
    print("High-dimensional distance metrics computed")

    return distance_df


def cluster_names(  # noqa: C901
    input_path: str,
    image_dir: str = "./images",
    output_dir: str = "./data",
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    sample_size: Optional[int] = None,
    min_cluster_size: int = 5,
    min_samples: int = 3,
    cluster_selection_epsilon: float = 0.1,
    use_gpu: bool = True,
    random_seed: int = 31337,
) -> None:
    """Cluster names using HDBSCAN on original embeddings, with T-SNE for visualization.

    Parameters
    ----------
    input_path : str
        Path to the input Parquet file containing names to cluster
    image_dir : str, optional
        Directory to save the visualization PNG file, by default "./images"
    output_dir : str, optional
        Directory to save the CSV files and embeddings, by default "./data"
    model_name : str, optional
        Name of the sentence transformer model to use for embeddings
    sample_size : Optional[int], optional
        Number of names to sample for clustering (None = use all), by default None
    min_cluster_size : int, optional
        Minimum cluster size for HDBSCAN, by default 5
    min_samples : int, optional
        Minimum samples parameter for HDBSCAN, by default 3
    cluster_selection_epsilon : float, optional
        HDBSCAN epsilon for cluster selection (higher values = more noise points), by default 0.1
    use_gpu : bool, optional
        Whether to use GPU acceleration for embeddings, by default True
    random_seed : int, optional
        Random seed for reproducibility, by default 31337
    """
    print(f"Loading data from {input_path}")
    df = pd.read_parquet(input_path)

    # Extract unique names from left_name and right_name columns
    print("Extracting unique names...")
    left_names = df["left_name"].dropna().unique()
    right_names = df["right_name"].dropna().unique()
    unique_names = np.unique(np.concatenate([left_names, right_names]))

    print(f"Found {len(unique_names):,} unique names")

    # Sample if requested
    if sample_size is not None and len(unique_names) > sample_size:
        np.random.seed(random_seed)
        unique_names = np.random.choice(unique_names, size=sample_size, replace=False)
        print(f"Sampled {len(unique_names):,} names for clustering")

    # Check for GPU availability and set device
    device = "cpu"
    if use_gpu:
        if torch.backends.mps.is_available():
            device = "mps"
            print("Using Apple GPU (MPS) for embeddings")
        elif torch.cuda.is_available():
            device = "cuda"
            print("Using NVIDIA GPU (CUDA) for embeddings")
        else:
            print("GPU requested but not available, using CPU")
    else:
        print("Using CPU for embeddings")

    # Load the sentence transformer model
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name, device=device)

    # Generate embeddings with GPU acceleration
    print("Generating embeddings...")
    embeddings = model.encode(
        unique_names.tolist(),
        show_progress_bar=True,
        convert_to_numpy=True,
        batch_size=32,  # Adjust based on GPU memory
    )

    print(f"Generated embeddings shape: {embeddings.shape}")

    # Normalize embeddings for cosine-like distance using euclidean metric
    # This allows HDBSCAN to work with normalized embeddings using euclidean distance
    # which approximates cosine similarity
    embeddings_normalized = normalize(embeddings, norm="l2")
    print("Normalized embeddings for cosine-like clustering")

    # Perform HDBSCAN clustering on normalized embeddings
    print("Performing HDBSCAN clustering on normalized embeddings...")
    clusterer = cuml.cluster.hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",  # Euclidean on normalized vectors approximates cosine distance
        cluster_selection_epsilon=cluster_selection_epsilon,
        prediction_data=True,
    )

    cluster_labels = clusterer.fit_predict(embeddings_normalized)

    # Perform T-SNE dimension reduction to 2D for visualization only
    print("Performing T-SNE dimension reduction for visualization...")
    tsne = TSNE(
        n_components=2,
        random_state=random_seed,
        perplexity=min(30, len(unique_names) - 1),  # Adjust perplexity for small datasets
        max_iter=1000,
        verbose=1,
    )

    embeddings_2d = tsne.fit_transform(embeddings)
    print(f"T-SNE completed. 2D embeddings shape: {embeddings_2d.shape}")

    # Count clusters and noise points
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)

    print("HDBSCAN completed:")
    print(f"  Number of clusters: {n_clusters}")
    print(f"  Number of noise points: {n_noise}")
    print(f"  Clustered points: {len(cluster_labels) - n_noise}")

    # Create comprehensive results DataFrame for embedding exploration
    results_data = {
        "name": unique_names,
        "cluster": cluster_labels,
        "x": embeddings_2d[:, 0],
        "y": embeddings_2d[:, 1],
    }

    # Add HDBSCAN-specific attributes if available
    if hasattr(clusterer, "probabilities_") and clusterer.probabilities_ is not None:
        results_data["cluster_probability"] = clusterer.probabilities_
    else:
        results_data["cluster_probability"] = np.ones(len(unique_names))  # Default to 1.0

    if hasattr(clusterer, "outlier_scores_") and clusterer.outlier_scores_ is not None:
        results_data["outlier_score"] = clusterer.outlier_scores_
    else:
        results_data["outlier_score"] = np.zeros(len(unique_names))  # Default to 0.0

    results_df = pd.DataFrame(results_data)

    # Add cluster size information
    cluster_sizes = results_df["cluster"].value_counts().to_dict()
    results_df["cluster_size"] = results_df["cluster"].map(cluster_sizes)

    # Add embedding statistics for exploration
    for i in range(min(10, embeddings.shape[1])):  # First 10 embedding dimensions
        results_df[f"embed_dim_{i}"] = embeddings[:, i]

    # Compute high-dimensional distance metrics
    distance_df = _compute_distance_metrics(unique_names, cluster_labels, embeddings_normalized)

    # Add distance metrics to results DataFrame
    for col in distance_df.columns:
        if col not in ["name", "index"]:
            results_df[col] = distance_df[col]

    # Create output directories
    image_path = Path(image_dir)
    image_path.mkdir(parents=True, exist_ok=True)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create visualization using seaborn
    print("Creating visualization...")
    plt.figure(figsize=(12, 8))

    # Create a color palette for clusters
    unique_clusters = sorted(set(cluster_labels))
    palette = sns.color_palette("husl", n_colors=len(unique_clusters))
    cluster_colors = dict(zip(unique_clusters, palette))

    # Plot points
    for cluster_id in unique_clusters:
        mask = results_df["cluster"] == cluster_id
        if cluster_id == -1:
            # Noise points in gray
            plt.scatter(
                results_df[mask]["x"],
                results_df[mask]["y"],
                c="lightgray",
                alpha=0.6,
                s=30,
                label="Noise",
            )
        else:
            plt.scatter(
                results_df[mask]["x"],
                results_df[mask]["y"],
                c=[cluster_colors[cluster_id]],
                alpha=0.7,
                s=40,
                label=f"Cluster {cluster_id}",
            )

    plt.title(
        f"Name Clustering Visualization\\n{len(unique_names):,} names, {n_clusters} clusters, {n_noise} noise points"
    )
    plt.xlabel("T-SNE Dimension 1")
    plt.ylabel("T-SNE Dimension 2")

    # Add legend, but limit to reasonable number of clusters
    if n_clusters <= 20:
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    else:
        plt.text(
            0.02,
            0.98,
            f"{n_clusters} clusters\\n{n_noise} noise points",
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    plt.tight_layout()

    # Save the plot
    image_file = image_path / "name_clusters.png"
    plt.savefig(image_file, dpi=300, bbox_inches="tight")
    print(f"Visualization saved to: {image_file}")

    # Save comprehensive results for embedding exploration
    csv_file = output_path / "cluster_results.csv"
    results_df.to_csv(csv_file, index=False)
    print(f"Cluster results saved to: {csv_file}")

    # Save full embeddings as numpy array for embedding exploration tools
    embeddings_file = output_path / "embeddings.npy"
    np.save(embeddings_file, embeddings)
    print(f"Full embeddings saved to: {embeddings_file}")

    # Save cluster analysis summary
    cluster_summary = []
    for cluster_id in sorted(set(cluster_labels)):
        if cluster_id == -1:
            cluster_summary.append(
                {
                    "cluster_id": cluster_id,
                    "label": "noise",
                    "size": (cluster_labels == cluster_id).sum(),
                    "avg_probability": 0.0,
                    "avg_outlier_score": results_df[results_df["cluster"] == cluster_id][
                        "outlier_score"
                    ].mean(),
                }
            )
        else:
            cluster_mask = results_df["cluster"] == cluster_id
            cluster_summary.append(
                {
                    "cluster_id": cluster_id,
                    "label": f"cluster_{cluster_id}",
                    "size": cluster_mask.sum(),
                    "avg_probability": results_df[cluster_mask]["cluster_probability"].mean(),
                    "avg_outlier_score": results_df[cluster_mask]["outlier_score"].mean(),
                }
            )

    cluster_summary_df = pd.DataFrame(cluster_summary)
    summary_file = output_path / "cluster_summary.csv"
    cluster_summary_df.to_csv(summary_file, index=False)
    print(f"Cluster summary saved to: {summary_file}")

    # Print some example clusters
    print("\\nExample clusters:")
    for cluster_id in sorted(set(cluster_labels)):
        if cluster_id == -1:
            continue  # Skip noise
        cluster_names = results_df[results_df["cluster"] == cluster_id]["name"].tolist()
        if len(cluster_names) <= 10:
            print(f"  Cluster {cluster_id}: {', '.join(cluster_names)}")
        else:
            print(
                f"  Cluster {cluster_id} ({len(cluster_names)} names): {', '.join(cluster_names[:5])}..."
            )

        if cluster_id >= 4:  # Limit output
            remaining_clusters = len([c for c in set(cluster_labels) if c > cluster_id and c != -1])
            if remaining_clusters > 0:
                print(f"  ... and {remaining_clusters} more clusters")
            break

    plt.show()
