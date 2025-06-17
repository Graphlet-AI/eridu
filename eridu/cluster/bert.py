"""Traditional NLP clustering using NeoBERT tokenization and TF-IDF features."""

import warnings
from pathlib import Path
from typing import Optional

import hdbscan  # type: ignore
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.manifold import TSNE  # type: ignore
from sklearn.metrics.pairwise import cosine_distances  # type: ignore
from sklearn.preprocessing import normalize  # type: ignore
from transformers import AutoTokenizer

# Suppress SyntaxWarnings from HDBSCAN library - these are from the library itself, not our code
warnings.filterwarnings("ignore", category=SyntaxWarning)
# Suppress sklearn deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")


def _compute_distance_metrics(
    unique_names: np.ndarray,
    cluster_labels: np.ndarray,
    features_normalized: np.ndarray,
) -> pd.DataFrame:
    """Compute high-dimensional distance metrics for cluster analysis."""
    print("Computing high-dimensional distance metrics...")

    # Compute cosine distances in high-dimensional space (more meaningful for TF-IDF features)
    cosine_dist_matrix = cosine_distances(features_normalized)

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
            cluster_centroid = features_normalized[same_cluster_mask].mean(axis=0)
            centroid_distance = cosine_distances([features_normalized[i]], [cluster_centroid])[0, 0]
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


def _tokenize_with_bert(names: list[str], model_name: str = "bert-base-uncased") -> list[str]:
    """Tokenize names using BERT tokenizer and return tokenized text."""
    print(f"Loading BERT tokenizer: {model_name}")

    # Load the BERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenized_names = []
    print("Tokenizing names with BERT...")

    for name in names:
        # Tokenize the name and convert tokens back to text
        tokens = tokenizer.tokenize(name)
        # Join tokens with spaces for TF-IDF processing
        tokenized_text = " ".join(tokens)
        tokenized_names.append(tokenized_text)

    print(f"Tokenized {len(names)} names")
    return tokenized_names


def cluster_names_bert(  # noqa: C901
    input_path: str,
    image_dir: str = "./images",
    output_dir: str = "./data",
    model_name: str = "bert-base-uncased",
    sample_size: Optional[int] = None,
    min_cluster_size: int = 5,
    min_samples: int = 3,
    cluster_selection_epsilon: float = 0.1,
    max_features: int = 10000,
    min_df: int = 2,
    max_df: float = 0.95,
    ngram_range: tuple[int, int] = (1, 3),
    random_seed: int = 31337,
) -> None:
    """Cluster names using traditional NLP approach with BERT tokenization and TF-IDF.

    Parameters
    ----------
    input_path : str
        Path to the input Parquet file containing names to cluster
    image_dir : str, optional
        Directory to save the visualization PNG file, by default "./images"
    output_dir : str, optional
        Directory to save the CSV files and features, by default "./data"
    model_name : str, optional
        Name of the BERT model to use for tokenization
    sample_size : Optional[int], optional
        Number of names to sample for clustering (None = use all), by default None
    min_cluster_size : int, optional
        Minimum cluster size for HDBSCAN, by default 5
    min_samples : int, optional
        Minimum samples parameter for HDBSCAN, by default 3
    cluster_selection_epsilon : float, optional
        HDBSCAN epsilon for cluster selection (higher values = more noise points), by default 0.1
    max_features : int, optional
        Maximum number of features for TF-IDF vectorizer, by default 10000
    min_df : int, optional
        Minimum document frequency for TF-IDF features, by default 2
    max_df : float, optional
        Maximum document frequency for TF-IDF features, by default 0.95
    ngram_range : tuple[int, int], optional
        N-gram range for TF-IDF features, by default (1, 3)
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

    # Tokenize names using BERT
    tokenized_names = _tokenize_with_bert(unique_names.tolist(), model_name)

    # Create TF-IDF features from tokenized text
    print("Creating TF-IDF features from tokenized names...")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range,
        stop_words=None,  # Don't use stop words for names
        lowercase=True,
        analyzer="word",
        token_pattern=r"\b\w+\b",  # Match word tokens
    )

    tfidf_features = vectorizer.fit_transform(tokenized_names)
    print(f"Generated TF-IDF features shape: {tfidf_features.shape}")

    # Convert to dense array for further processing
    features_dense = tfidf_features.toarray()

    # Normalize features for cosine-like distance using euclidean metric
    features_normalized = normalize(features_dense, norm="l2")
    print("Normalized TF-IDF features for cosine-like clustering")

    # Perform HDBSCAN clustering on normalized features
    print("Performing HDBSCAN clustering on TF-IDF features...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",  # Euclidean on normalized vectors approximates cosine distance
        cluster_selection_epsilon=cluster_selection_epsilon,
    )

    cluster_labels = clusterer.fit_predict(features_normalized)

    # Perform T-SNE dimension reduction to 2D for visualization only
    print("Performing T-SNE dimension reduction for visualization...")
    tsne = TSNE(
        n_components=2,
        random_state=random_seed,
        perplexity=min(30, len(unique_names) - 1),  # Adjust perplexity for small datasets
        n_iter=1000,
        verbose=1,
    )

    features_2d = tsne.fit_transform(features_dense)
    print(f"T-SNE completed. 2D features shape: {features_2d.shape}")

    # Count clusters and noise points
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)

    print("HDBSCAN completed:")
    print(f"  Number of clusters: {n_clusters}")
    print(f"  Number of noise points: {n_noise}")
    print(f"  Clustered points: {len(cluster_labels) - n_noise}")

    # Create comprehensive results DataFrame for feature exploration
    results_df = pd.DataFrame(
        {
            "name": unique_names,
            "tokenized_name": tokenized_names,
            "cluster": cluster_labels,
            "x": features_2d[:, 0],
            "y": features_2d[:, 1],
            "cluster_probability": clusterer.probabilities_,  # HDBSCAN membership probability
            "outlier_score": clusterer.outlier_scores_,  # HDBSCAN outlier scores
        }
    )

    # Add cluster size information
    cluster_sizes = results_df["cluster"].value_counts().to_dict()
    results_df["cluster_size"] = results_df["cluster"].map(cluster_sizes)

    # Add TF-IDF feature statistics for exploration
    for i in range(min(10, features_dense.shape[1])):  # First 10 TF-IDF features
        results_df[f"tfidf_feature_{i}"] = features_dense[:, i]

    # Compute high-dimensional distance metrics
    distance_df = _compute_distance_metrics(unique_names, cluster_labels, features_normalized)

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
        f"BERT + TF-IDF Name Clustering\\n{len(unique_names):,} names, {n_clusters} clusters, {n_noise} noise points"
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
    image_file = image_path / "bert_clusters.png"
    plt.savefig(image_file, dpi=300, bbox_inches="tight")
    print(f"Visualization saved to: {image_file}")

    # Save comprehensive results for feature exploration
    csv_file = output_path / "bert_cluster_results.csv"
    results_df.to_csv(csv_file, index=False)
    print(f"Cluster results saved to: {csv_file}")

    # Save TF-IDF features as numpy array for feature exploration tools
    features_file = output_path / "bert_tfidf_features.npy"
    np.save(features_file, features_dense)
    print(f"TF-IDF features saved to: {features_file}")

    # Save the TF-IDF vectorizer for potential reuse
    import pickle

    vectorizer_file = output_path / "bert_tfidf_vectorizer.pkl"
    with open(vectorizer_file, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"TF-IDF vectorizer saved to: {vectorizer_file}")

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
    summary_file = output_path / "bert_cluster_summary.csv"
    cluster_summary_df.to_csv(summary_file, index=False)
    print(f"Cluster summary saved to: {summary_file}")

    # Print some example clusters
    print("\\nExample clusters:")
    for cluster_id in sorted(set(cluster_labels)):
        if cluster_id == -1:
            continue  # Skip noise
        cluster_names_list = results_df[results_df["cluster"] == cluster_id]["name"].tolist()
        if len(cluster_names_list) <= 10:
            print(f"  Cluster {cluster_id}: {', '.join(cluster_names_list)}")
        else:
            print(
                f"  Cluster {cluster_id} ({len(cluster_names_list)} names): {', '.join(cluster_names_list[:5])}..."
            )

        if cluster_id >= 4:  # Limit output
            remaining_clusters = len([c for c in set(cluster_labels) if c > cluster_id and c != -1])
            if remaining_clusters > 0:
                print(f"  ... and {remaining_clusters} more clusters")
            break

    plt.show()
