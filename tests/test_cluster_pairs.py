#!/usr/bin/env python3

import pandas as pd

# Load cluster results to get tight cluster names
cluster_df = pd.read_csv("data/cluster_results.csv")
tight_cluster_ids = [4, 6, 9, 11, 12, 13, 15, 16]

# Get names from tight clusters
tight_cluster_names: set[str] = set()
for cluster_id in tight_cluster_ids:
    names = cluster_df[cluster_df["cluster"] == cluster_id]["name"].values
    tight_cluster_names.update(names)

print(f"Found {len(tight_cluster_names)} names in tight clusters")

# Load full pairs dataset
pairs_df = pd.read_parquet("data/pairs-all.parquet")

# Find pairs involving tight cluster names
mask = pairs_df["left_name"].isin(tight_cluster_names) | pairs_df["right_name"].isin(
    tight_cluster_names
)
pairs_with_tight_clusters = pairs_df[mask]

print(f"Found {len(pairs_with_tight_clusters)} pairs involving tight cluster names")

if len(pairs_with_tight_clusters) > 0:
    # Sample some of these pairs plus some random pairs
    tight_sample = pairs_with_tight_clusters.sample(
        n=min(100, len(pairs_with_tight_clusters)), random_state=42
    )
    regular_sample = pairs_df.sample(n=900, random_state=42)

    # Combine samples
    test_sample = pd.concat([tight_sample, regular_sample], ignore_index=True)
    test_sample = test_sample.drop_duplicates().reset_index(drop=True)

    test_sample.to_parquet("data/pairs-test-sample.parquet", index=False)
    print(f"Created test sample with {len(test_sample)} pairs")

    # Show some examples
    print("\nTight cluster examples:")
    for cluster_id in tight_cluster_ids[:3]:
        names = cluster_df[cluster_df["cluster"] == cluster_id]["name"].values
        print(f"  Cluster {cluster_id}: {list(names)}")
else:
    print("No pairs found with tight cluster names")
