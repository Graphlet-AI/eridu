"""Unit tests for ContextAdaptiveContrastiveLoss."""

import pytest
import torch
from sentence_transformers import SentenceTransformer
from torch import Tensor

from eridu.train.loss import ContextAdaptiveContrastiveLoss


@pytest.fixture
def model() -> SentenceTransformer:
    """Create a small SentenceTransformer model for testing."""
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


@pytest.fixture
def loss_function(model: SentenceTransformer) -> ContextAdaptiveContrastiveLoss:
    """Create ContextAdaptiveContrastiveLoss instance for testing."""
    return ContextAdaptiveContrastiveLoss(model=model, margin=0.5, gate_scale=5.0)


@pytest.fixture
def sample_data(model: SentenceTransformer) -> tuple[list[dict[str, Tensor]], Tensor]:
    """Create sample sentence features with different lengths and labels."""
    batch_size = 4
    device = model.device

    sentence_features = [
        {
            "input_ids": torch.randint(0, 30522, (batch_size, 15), device=device),
            "attention_mask": torch.ones(batch_size, 15, dtype=torch.long, device=device),
        },
        {
            "input_ids": torch.randint(0, 30522, (batch_size, 20), device=device),
            "attention_mask": torch.ones(batch_size, 20, dtype=torch.long, device=device),
        },
    ]
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0], device=device)
    return sentence_features, labels


def test_forward_pass_returns_valid_loss(loss_function, sample_data):
    """Test that forward pass returns a non-negative scalar loss."""
    sentence_features, labels = sample_data
    loss = loss_function(sentence_features, labels)

    assert isinstance(loss, Tensor)
    assert loss.dim() == 0  # Scalar
    assert loss.item() >= 0  # Non-negative


def test_handles_different_sequence_lengths(loss_function, sample_data):
    """Test that loss function handles different sequence lengths without errors."""
    sentence_features, labels = sample_data

    # Verify sequences have different lengths
    seq1_len = sentence_features[0]["input_ids"].size(1)
    seq2_len = sentence_features[1]["input_ids"].size(1)
    assert seq1_len != seq2_len

    # Should not raise RuntimeError about tensor dimensions
    loss = loss_function(sentence_features, labels)
    assert isinstance(loss, Tensor)


def test_gradient_computation_works(loss_function, sample_data):
    """Test that gradients can be computed for model parameters."""
    sentence_features, labels = sample_data
    loss_function.model.train()
    loss_function.model.zero_grad()

    loss = loss_function(sentence_features, labels)
    loss.backward()

    # Check that at least some parameters have gradients
    has_gradients = any(
        param.grad is not None and param.grad.abs().sum() > 0
        for param in loss_function.model.parameters()
    )
    assert has_gradients, "Model should have gradients after backward pass"
