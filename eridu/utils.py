"""Shared utility functions for eridu."""

import os


def get_model_config() -> dict[str, str]:
    """Get model configuration from environment variables."""
    sbert_model = os.environ.get("SBERT_MODEL")
    if not sbert_model:
        raise ValueError(
            "SBERT_MODEL environment variable must be set. Use --model parameter in CLI."
        )

    variant = os.environ.get("VARIANT", "original")
    optimizer = os.environ.get("OPTIMIZER", "adafactor")
    data_type = os.environ.get("DATA_TYPE", "companies")
    model_save_name = (sbert_model + "-" + variant + "-" + optimizer).replace("/", "-")
    sbert_output_folder = f"data/fine-tuned-sbert-{model_save_name}-{data_type}"

    return {
        "SBERT_MODEL": sbert_model,
        "VARIANT": variant,
        "OPTIMIZER": optimizer,
        "DATA_TYPE": data_type,
        "MODEL_SAVE_NAME": model_save_name,
        "SBERT_OUTPUT_FOLDER": sbert_output_folder,
    }


def get_model_path_for_entity_type(entity_type: str) -> str:
    """Get the appropriate model path based on entity type.

    Args:
        entity_type: The entity type (people, companies) - required

    Returns:
        The model path including entity type

    Raises:
        ValueError: If entity_type is not provided
        FileNotFoundError: If model doesn't exist
    """
    if not entity_type:
        raise ValueError("entity_type must be provided")

    base_model = "data/fine-tuned-sbert-intfloat-multilingual-e5-base-original-adafactor-companies"
    entity_model_path = f"{base_model}-{entity_type}"

    # Check if the model exists
    if not os.path.exists(entity_model_path):
        # For backward compatibility, check without suffix
        if os.path.exists(base_model):
            return base_model
        raise FileNotFoundError(f"Model not found at {entity_model_path}")

    return entity_model_path
