"""Generate training pairs using cleanco to create corporate ending variations."""

import random
from typing import Any, Optional

import pandas as pd
from cleanco import basename  # type: ignore
from cleanco.termdata import terms_by_type  # type: ignore


def generate_cleanco_training_pairs(
    input_parquet: str = "data/pairs-all.parquet",
    output_parquet: str = "data/pairs-cleanco.parquet",
    num_examples: int = 10000,
    random_seed: Optional[int] = 42,
) -> None:
    """Generate training pairs by swapping corporate endings.

    Creates both matching pairs (same type, different ending) and
    non-matching pairs (different type endings).

    Args:
        input_parquet: Path to input pairs parquet file
        output_parquet: Path to output parquet file
        num_examples: Number of example pairs to generate
        random_seed: Random seed for reproducibility
    """
    random.seed(random_seed)

    print(f"Reading data from {input_parquet}")
    df = pd.read_parquet(input_parquet)

    # Filter to matching ORG pairs
    company_df = df[
        (df["left_category"] == "ORG")
        & (df["right_category"] == "ORG")
        & (df["match"] == True)  # noqa: E712
    ].copy()

    print(f"Found {len(company_df):,} matching company pairs")

    # Get all unique company names
    all_company_names = pd.concat([company_df["left_name"], company_df["right_name"]]).unique()

    print(f"Found {len(all_company_names):,} unique company names")

    # Get term types and their terms
    type_names = list(terms_by_type.keys())
    print(f"\nAvailable corporate types: {len(type_names)}")
    for type_name in type_names:
        print(f"  - {type_name}: {len(terms_by_type[type_name])} terms")

    # Generate pairs
    pairs: list[dict[str, Any]] = []
    target_matches = num_examples // 2
    target_non_matches = num_examples - target_matches

    print(
        f"\nGenerating {target_matches:,} matching pairs and {target_non_matches:,} non-matching pairs..."
    )

    # Sample companies to use
    sampled_companies = random.sample(
        list(all_company_names),
        min(num_examples * 2, len(all_company_names)),  # Sample extra for safety
    )

    for i, company_name in enumerate(sampled_companies):
        if len(pairs) >= num_examples:
            break

        if (i + 1) % 1000 == 0:
            matching = sum(1 for p in pairs if p["match"] is True)
            non_matching = len(pairs) - matching
            print(
                f"  Processed {i + 1:,} companies - {matching:,} matches, {non_matching:,} non-matches"
            )

        # Strip corporate ending
        base_name = basename(company_name)

        # Skip if basename is empty, too short, or same as original
        if not base_name or len(base_name) < 3 or base_name == company_name:
            continue

        # Decide whether to create matching or non-matching pair
        matching_count = sum(1 for p in pairs if p["match"] is True)
        non_matching_count = len(pairs) - matching_count

        create_match = matching_count < target_matches and (
            non_matching_count >= target_non_matches or random.random() < 0.5
        )

        if create_match:
            # Create matching pair: same type, different endings
            # Pick a random type
            type_name = random.choice(type_names)
            type_terms = terms_by_type[type_name]

            # Need at least 2 terms to create a pair
            if len(type_terms) < 2:
                continue

            # Pick 2 different terms from same type
            term1, term2 = random.sample(type_terms, 2)

            # Create the pair
            left_name = f"{base_name} {term1}".strip()
            right_name = f"{base_name} {term2}".strip()
            match = True
            score = 0.9  # High score for same company

        else:
            # Create non-matching pair: different types, different endings
            # Pick 2 different types
            if len(type_names) < 2:
                continue

            type1, type2 = random.sample(type_names, 2)
            terms1 = terms_by_type[type1]
            terms2 = terms_by_type[type2]

            if not terms1 or not terms2:
                continue

            # Pick one term from each type
            term1 = random.choice(terms1)
            term2 = random.choice(terms2)

            # Create the pair
            left_name = f"{base_name} {term1}".strip()
            right_name = f"{base_name} {term2}".strip()
            match = False
            score = 0.3  # Lower score for different company types

        # Use the original record's norm and fp as template, but keep names as-is
        # We'll just copy the structure from a random existing record
        template = company_df.sample(1).iloc[0]

        # Create pair record
        pair = {
            "left_name": left_name,
            "left_norm": template["left_norm"],  # Keep original format
            "left_fp": template["left_fp"],
            "left_lang": template["left_lang"],
            "left_category": "ORG",
            "right_name": right_name,
            "right_norm": template["right_norm"],
            "right_fp": template["right_fp"],
            "right_lang": template["right_lang"],
            "right_category": "ORG",
            "match": match,
            "dist_norm": template["dist_norm"],
            "dist_fp": template["dist_fp"],
            "score": score,
            "source": "cleanco-generated",
        }

        pairs.append(pair)

    print(f"\nGenerated {len(pairs):,} training pairs")

    # Create DataFrame
    pairs_df = pd.DataFrame(pairs)

    # Ensure column order matches input
    pairs_df = pairs_df[df.columns]

    # Save to parquet
    print(f"Saving to {output_parquet}")
    pairs_df.to_parquet(output_parquet, index=False)

    # Statistics
    matching = pairs_df["match"].sum()
    non_matching = len(pairs_df) - matching

    print(f"\nSuccess! Generated {len(pairs_df):,} pairs")
    print(f"  - Matching pairs: {matching:,}")
    print(f"  - Non-matching pairs: {non_matching:,}")
    print(f"Output saved to: {output_parquet}")

    # Show samples
    print("\n=== Sample MATCHING pairs (same base company, same type) ===")
    print(
        pairs_df[pairs_df["match"]][["left_name", "right_name", "match", "score"]]
        .head(5)
        .to_string(index=False)
    )

    print("\n=== Sample NON-MATCHING pairs (same base company, different types) ===")
    print(
        pairs_df[~pairs_df["match"]][["left_name", "right_name", "match", "score"]]
        .head(5)
        .to_string(index=False)
    )


if __name__ == "__main__":
    generate_cleanco_training_pairs(
        input_parquet="data/pairs-all.parquet",
        output_parquet="data/pairs-cleanco.parquet",
        num_examples=10000,
        random_seed=42,
    )
