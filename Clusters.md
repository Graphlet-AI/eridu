# Entity-Aware Clustering for Name Pair Data

## Overview

This document describes clustering-based approaches to address data leakage and overfitting issues in entity matching datasets. The core problem is that traditional random train/test splits can place semantically similar name pairs in both training and test sets, leading to inflated performance metrics.

## Problem Statement

Current train/test splitting is **record-based**, meaning similar name pairs like:
- "John Smith" vs "J. Smith" 
- "Microsoft Corp" vs "Microsoft Corporation"
- "Robert Johnson" vs "Bob Johnson"

Could appear in both training and test sets, causing:
- **Data leakage**: Model sees variations during training and testing
- **Overfitting**: Model memorizes entity-specific patterns
- **Inflated metrics**: Test performance doesn't reflect real-world capability

## Solution: Entity-Aware Clustering

### Implementation Code

```python
import numpy as np
from sklearn.cluster import HDBSCAN, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

def cluster_name_pairs_hdbscan(df, model, min_cluster_size=2, min_samples=1):
    """Cluster name pairs using HDBSCAN on embeddings"""
    
    # Create combined name strings for embedding
    df['combined_names'] = df['left_name'] + ' ||| ' + df['right_name']
    
    # Get embeddings
    embeddings = model.encode(df['combined_names'].tolist(), 
                            convert_to_tensor=False, 
                            show_progress_bar=True)
    
    # HDBSCAN clustering
    # min_cluster_size: minimum size of clusters
    # min_samples: min samples in neighborhood for core point
    clustering = HDBSCAN(min_cluster_size=min_cluster_size, 
                        min_samples=min_samples, 
                        metric='cosine')
    cluster_labels = clustering.fit_predict(embeddings)
    
    # Assign cluster IDs
    df['cluster_id'] = cluster_labels
    
    # Handle noise points (-1 labels) as individual clusters
    noise_mask = cluster_labels == -1
    max_cluster = cluster_labels.max() if cluster_labels.max() >= 0 else -1
    df.loc[noise_mask, 'cluster_id'] = range(max_cluster + 1, 
                                           max_cluster + 1 + noise_mask.sum())
    
    return df

def cluster_name_pairs_hierarchical(df, model, threshold=0.85):
    """Alternative: Hierarchical clustering with distance threshold"""
    
    df['combined_names'] = df['left_name'] + ' ||| ' + df['right_name']
    embeddings = model.encode(df['combined_names'].tolist())
    
    # Hierarchical clustering
    clustering = AgglomerativeClustering(
        n_clusters=None, 
        distance_threshold=1-threshold,  # Convert similarity to distance
        linkage='average',
        affinity='cosine'
    )
    
    cluster_labels = clustering.fit_predict(embeddings)
    df['cluster_id'] = cluster_labels
    
    return df

def create_entity_aware_splits(df, test_size=0.2, random_state=42):
    """Split by clusters, not individual records"""
    
    unique_clusters = df['cluster_id'].unique()
    
    train_clusters, test_clusters = train_test_split(
        unique_clusters, 
        test_size=test_size, 
        random_state=random_state
    )
    
    train_df = df[df['cluster_id'].isin(train_clusters)]
    test_df = df[df['cluster_id'].isin(test_clusters)]
    
    print(f"Train clusters: {len(train_clusters)}, Test clusters: {len(test_clusters)}")
    print(f"Train records: {len(train_df)}, Test records: {len(test_df)}")
    
    return train_df, test_df
```

## Current State Analysis

From our duplicate analysis:
- **Total records**: 26,632,762
- **Exact duplicates found**: 466,565 (1.8%)
- **After deduplication**: 26,166,197 unique records

The clustering approach would identify **semantic duplicates** beyond exact matches, likely finding significantly more similar name pairs that should be grouped together.

## Clustering Methods

### 1. HDBSCAN Clustering
- **Advantages**: 
  - Better at finding clusters of varying densities
  - More robust to noise than traditional DBSCAN
  - Hierarchical approach provides cluster stability
- **Parameters**:
  - `min_cluster_size`: Minimum size of clusters (default 2)
  - `min_samples`: Minimum samples in neighborhood for core point (default 1)

### 2. Hierarchical Clustering
- **Advantages**:
  - More deterministic than HDBSCAN
  - Better control over cluster sizes via threshold
  - No noise points - all records get assigned to clusters
- **Parameters**:
  - `threshold`: Similarity threshold for cluster formation (default 0.85)

### 3. Entity-Aware Splitting
- **Purpose**: Splits data by clusters, not individual records
- **Method**:
  - Identifies unique cluster IDs
  - Splits clusters into train/test sets
  - Assigns all records in a cluster to the same split
- **Benefit**: Ensures similar name pairs are entirely in train OR test, never both

## Implementation Benefits

1. **Eliminates Data Leakage**: No entity variations across train/test boundary
2. **Realistic Performance Metrics**: Test accuracy reflects real-world performance  
3. **Reduces Overfitting**: Model learns semantic patterns, not entity-specific features
4. **Better Generalization**: Forces model to understand name matching concepts

## Technical Requirements

- **Dependencies**: numpy, sklearn, hdbscan, sentence-transformers (or similar embedding model)
- **Input**: DataFrame with `left_name`, `right_name` columns
- **Output**: DataFrame with additional `cluster_id` column for splitting

## Next Steps

1. Implement clustering functions in the codebase
2. Integrate with existing training pipeline
3. Compare performance metrics before/after entity-aware splitting
4. Analyze cluster distributions and quality
5. Tune clustering parameters for optimal grouping

This approach should provide much more robust and realistic model evaluation, addressing the overfitting concerns identified in the current training results.