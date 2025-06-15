"""Analyze clustering results and show examples of names in each cluster."""

import pandas as pd


def _print_cluster_overview(df_sorted: pd.DataFrame) -> None:
    """Print basic overview of clustering results."""
    print("Dataset Overview:")
    print(f"Total records: {len(df_sorted):,}")
    print(f'Clusters found: {df_sorted["cluster"].nunique()}')
    print(f'Noise points (cluster -1): {(df_sorted["cluster"] == -1).sum():,}')
    print(f'Clustered points: {(df_sorted["cluster"] != -1).sum():,}')
    print()


def _print_cluster_size_distribution(df_sorted: pd.DataFrame) -> None:
    """Print cluster size distribution."""
    print("Cluster Size Distribution (largest to smallest):")
    cluster_sizes = df_sorted["cluster"].value_counts().sort_values(ascending=False)
    for cluster_id, size in cluster_sizes.items():
        if cluster_id == -1:
            print(f"  Noise (cluster -1): {size:,} points")
        else:
            print(f"  Cluster {cluster_id}: {size} points")
    print()


def _print_cluster_examples(df_sorted: pd.DataFrame) -> None:
    """Print examples of names in each cluster."""
    print("Examples of Names in Each Cluster:")
    print("=" * 60)

    for cluster_id in sorted(df_sorted["cluster"].unique()):
        if cluster_id == -1:
            continue  # Skip noise for now

        cluster_data = df_sorted[df_sorted["cluster"] == cluster_id]
        cluster_size = len(cluster_data)

        print(f"\nCluster {cluster_id} ({cluster_size} names):")
        print("-" * 40)

        # Show all names if cluster is small, otherwise show first 10
        if cluster_size <= 15:
            for name in cluster_data["name"]:
                print(f"  {name}")
        else:
            for name in cluster_data["name"].head(10):
                print(f"  {name}")
            print(f"  ... and {cluster_size - 10} more names")

        # Show some cluster statistics
        avg_prob = cluster_data["cluster_probability"].mean()
        avg_outlier = cluster_data["outlier_score"].mean()

        print(f"  Avg cluster probability: {avg_prob:.3f}")
        print(f"  Avg outlier score: {avg_outlier:.3f}")

        # Show distance metrics if available
        if "intra_cluster_mean_dist" in cluster_data.columns:
            intra_mean = cluster_data["intra_cluster_mean_dist"].mean()
            inter_mean = cluster_data["inter_cluster_mean_dist"].mean()
            centroid_mean = cluster_data["distance_to_centroid"].mean()

            print(f"  Avg intra-cluster distance: {intra_mean:.3f}")
            print(f"  Avg inter-cluster distance: {inter_mean:.3f}")
            print(f"  Avg distance to centroid: {centroid_mean:.3f}")


def _print_noise_sample(df_sorted: pd.DataFrame) -> None:
    """Print sample of noise points."""
    print("\n" + "=" * 60)
    print("Noise Points Sample (first 20):")
    print("-" * 40)

    noise_data = df_sorted[df_sorted["cluster"] == -1]
    for name in noise_data["name"].head(20):
        print(f"  {name}")

    if len(noise_data) > 20:
        print(f"  ... and {len(noise_data) - 20:,} more noise points")

    print()


def _print_embedding_stats(df: pd.DataFrame) -> None:
    """Print embedding dimension statistics."""
    print("Embedding Dimension Statistics (first 5 dimensions):")
    print("-" * 60)

    embed_cols = [col for col in df.columns if col.startswith("embed_dim_")][:5]
    for col in embed_cols:
        dim_num = col.split("_")[-1]
        print(f"Dimension {dim_num}:")
        print(f"  Mean: {df[col].mean():.4f}")
        print(f"  Std:  {df[col].std():.4f}")
        print(f"  Min:  {df[col].min():.4f}")
        print(f"  Max:  {df[col].max():.4f}")
        print()


def analyze_cluster_results(csv_path: str = "images/cluster_results.csv") -> pd.DataFrame:
    """Analyze cluster results and show examples of names in each cluster."""
    # Load the cluster results
    df = pd.read_csv(csv_path)

    # Sort by cluster_id, then by name
    df_sorted = df.sort_values(["cluster", "name"])

    # Use helper functions to print analysis
    _print_cluster_overview(df_sorted)
    _print_cluster_size_distribution(df_sorted)
    _print_cluster_examples(df_sorted)
    _print_noise_sample(df_sorted)
    _print_embedding_stats(df)

    return df_sorted


def analyze_cluster_quality(df: pd.DataFrame) -> None:
    """Analyze the quality of clusters using distance metrics."""
    print("Cluster Quality Analysis:")
    print("=" * 60)

    # Filter out noise points for quality analysis
    clustered_data = df[df["cluster"] != -1]

    if "intra_cluster_mean_dist" not in df.columns:
        print("Distance metrics not available in the data.")
        return

    # Overall statistics
    print("Overall Cluster Quality Metrics:")
    print("-" * 40)
    print(f'Average intra-cluster distance: {clustered_data["intra_cluster_mean_dist"].mean():.4f}')
    print(f'Average inter-cluster distance: {clustered_data["inter_cluster_mean_dist"].mean():.4f}')
    print(f'Average distance to centroid: {clustered_data["distance_to_centroid"].mean():.4f}')
    print()

    # Cluster-by-cluster quality
    print("Per-Cluster Quality Metrics:")
    print("-" * 40)

    cluster_quality = (
        clustered_data.groupby("cluster")
        .agg(
            {
                "intra_cluster_mean_dist": "mean",
                "inter_cluster_mean_dist": "mean",
                "distance_to_centroid": "mean",
                "cluster_probability": "mean",
                "outlier_score": "mean",
                "name": "count",  # cluster size
            }
        )
        .round(4)
    )

    cluster_quality.columns = [
        "Intra_Dist",
        "Inter_Dist",
        "Centroid_Dist",
        "Avg_Prob",
        "Avg_Outlier",
        "Size",
    ]

    # Sort by cluster quality (lower intra-cluster distance is better)
    cluster_quality = cluster_quality.sort_values("Intra_Dist")

    print(cluster_quality)
    print()

    # Find the most cohesive clusters (low intra-cluster distance)
    print("Most Cohesive Clusters (lowest intra-cluster distance):")
    print("-" * 60)

    top_cohesive = cluster_quality.head(5)
    for cluster_id in top_cohesive.index:
        cluster_names = df[df["cluster"] == cluster_id]["name"].tolist()
        intra_dist = top_cohesive.loc[cluster_id, "Intra_Dist"]
        size = int(top_cohesive.loc[cluster_id, "Size"])  # type: ignore

        print(f"Cluster {cluster_id} (intra-distance: {intra_dist:.4f}, size: {size}):")
        if size <= 10:
            print(f'  Names: {", ".join(cluster_names)}')
        else:
            print(f'  Names: {", ".join(cluster_names[:5])}... and {size - 5} more')
        print()


if __name__ == "__main__":
    # Analyze the cluster results
    df_sorted = analyze_cluster_results()
    print()

    # Analyze cluster quality
    analyze_cluster_quality(df_sorted)
