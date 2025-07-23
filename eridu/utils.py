"""Shared utility functions for eridu."""

import os


def get_model_path_for_entity_type(entity_type: str) -> str:
    """Get the appropriate model path based on entity type.

    Args:
        entity_type: The entity type (people, companies, addresses) - required

    Returns:
        The model path including entity type

    Raises:
        ValueError: If entity_type is not provided
        FileNotFoundError: If model doesn't exist
    """
    if not entity_type:
        raise ValueError("entity_type must be provided")

    base_model = "data/fine-tuned-sbert-sentence-transformers-paraphrase-multilingual-MiniLM-L12-v2-original-adafactor"
    entity_model_path = f"{base_model}-{entity_type}"

    # Check if the model exists
    if not os.path.exists(entity_model_path):
        # For backward compatibility, check without suffix
        if os.path.exists(base_model):
            return base_model
        raise FileNotFoundError(f"Model not found at {entity_model_path}")

    return entity_model_path
