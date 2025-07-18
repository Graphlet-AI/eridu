from __future__ import annotations

from collections.abc import Iterable
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from sentence_transformers.SentenceTransformer import SentenceTransformer
from torch import Tensor, nn

import wandb


class ContextAdaptiveContrastiveLoss(nn.Module):
    def __init__(
        self,
        model: SentenceTransformer,
        margin: float = 0.5,
        gate_scale: float = 1.0,
        gate_stats_steps: int = 100,
    ) -> None:
        super().__init__()
        self.model = model
        self.margin = margin
        self.gate_scale = gate_scale
        self.gate_stats_steps = gate_stats_steps
        self.call_count = 0

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        self.call_count += 1

        output1_dict = self.model(sentence_features[0], output_value="token_embeddings")  # type: ignore
        output2_dict = self.model(sentence_features[1], output_value="token_embeddings")  # type: ignore
        sent1_dict = self.model(sentence_features[0], output_value="sentence_embedding")  # type: ignore
        sent2_dict = self.model(sentence_features[1], output_value="sentence_embedding")  # type: ignore

        # Extract tensors from the returned dictionaries
        output1 = output1_dict["token_embeddings"]  # (b, t, d)
        output2 = output2_dict["token_embeddings"]  # (b, t, d)
        sent1 = sent1_dict["sentence_embedding"]  # (b, d)
        sent2 = sent2_dict["sentence_embedding"]  # (b, d)

        # Handle different sequence lengths by truncating to the shorter length
        min_length = min(output1.size(1), output2.size(1))
        output1_truncated = output1[:, :min_length, :]  # (b, min_len, d)
        output2_truncated = output2[:, :min_length, :]  # (b, min_len, d)

        local_diff = 1 - F.cosine_similarity(output1_truncated, output2_truncated, dim=-1).mean(
            dim=1
        )
        global_diff = 1 - F.cosine_similarity(sent1, sent2)

        gate = torch.sigmoid(self.gate_scale * (global_diff - local_diff))
        final_diff = gate * global_diff + (1 - gate) * local_diff

        # Log every 100 forward passes
        if self.call_count % self.gate_stats_steps == 0:
            wandb.log(
                {
                    "train/gate_mean": gate.mean().item(),
                    "train/gate_std": gate.std().item(),
                    "train/global_diff_mean": global_diff.mean().item(),
                    "train/global_diff_std": global_diff.std().item(),
                    "train/local_diff_mean": local_diff.mean().item(),
                    "train/local_diff_std": local_diff.std().item(),
                    "train/step": self.call_count,
                }
            )

        pos_mask = labels == 1
        neg_mask = labels == 0

        pos_loss = final_diff[pos_mask].pow(2).sum()
        neg_loss = F.relu(self.margin - final_diff[neg_mask]).pow(2).sum()

        loss: Tensor = pos_loss + neg_loss

        # Store metrics as attributes for external logging
        self.last_metrics = {
            "gate_mean": gate.mean().item(),
            "gate_std": gate.std().item(),
            "global_diff_mean": global_diff.mean().item(),
            "local_diff_mean": local_diff.mean().item(),
            "pos_loss": pos_loss.item(),
            "neg_loss": neg_loss.item(),
            "gate_saturation": ((gate < 0.01) | (gate > 0.99)).float().mean().item(),
        }

        return loss


class MetricsContextAdaptiveContrastiveLoss(nn.Module):
    """
    Context-Adaptive Contrastive Loss with online hard mining and detailed metrics tracking.

    This loss combines:
    1. Context-adaptive loss computation (token-level and sentence-level)
    2. Online hard positive/negative mining from OnlineContrastiveLoss
    3. Detailed metrics logging for monitoring training progress

    Args:
        model: SentenceTransformer model
        margin: Margin for contrastive loss
        gate_scale: Scale for the gate function in context-adaptive loss
        distance_metric: Distance metric to use (cosine, euclidean, etc.)
        gate_stats_steps: Steps between detailed gate statistics logging
    """

    def __init__(
        self,
        model: SentenceTransformer,
        margin: float = 0.5,
        gate_scale: float = 5.0,
        distance_metric: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
        gate_stats_steps: int = 100,
    ) -> None:
        super().__init__()
        self.model = model
        self.margin = margin
        self.gate_scale = gate_scale
        self.gate_stats_steps = gate_stats_steps
        self.call_count = 0

        # Initialize gate parameter for balancing local vs global differences
        self.gate = nn.Parameter(torch.tensor(0.0))

        # Default to cosine distance if not specified
        if distance_metric is None:
            self.distance_metric = lambda x, y: 1 - F.cosine_similarity(x, y)
        else:
            self.distance_metric = distance_metric

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        self.call_count += 1

        # Convert iterable to list for indexing
        features_list = list(sentence_features)

        # Get token and sentence embeddings
        output1_dict = self.model(features_list[0], output_value="token_embeddings")
        output2_dict = self.model(features_list[1], output_value="token_embeddings")
        sent1_dict = self.model(features_list[0], output_value="sentence_embedding")
        sent2_dict = self.model(features_list[1], output_value="sentence_embedding")

        # Extract tensors
        output1 = output1_dict["token_embeddings"]  # (b, t, d)
        output2 = output2_dict["token_embeddings"]  # (b, t, d)
        sent1 = sent1_dict["sentence_embedding"]  # (b, d)
        sent2 = sent2_dict["sentence_embedding"]  # (b, d)

        # Handle different sequence lengths
        min_length = min(output1.size(1), output2.size(1))
        output1_truncated = output1[:, :min_length, :]
        output2_truncated = output2[:, :min_length, :]

        # Compute local (token-level) and global (sentence-level) differences
        local_diff = 1 - F.cosine_similarity(output1_truncated, output2_truncated, dim=-1).mean(
            dim=1
        )
        global_diff = 1 - F.cosine_similarity(sent1, sent2)

        # Apply gating mechanism
        gate_value = torch.sigmoid(self.gate_scale * self.gate)
        combined_diff = gate_value * local_diff + (1 - gate_value) * global_diff

        # Online hard mining: select hard positives and hard negatives
        positive_mask = labels == 1
        negative_mask = labels == 0

        pos_diffs = combined_diff[positive_mask]
        neg_diffs = combined_diff[negative_mask]

        # Select hard examples
        if len(pos_diffs) > 0 and len(neg_diffs) > 0:
            # Hard positives: positive pairs that are far apart
            hard_positive_threshold = neg_diffs.min() if len(neg_diffs) > 0 else pos_diffs.mean()
            hard_positives = pos_diffs[pos_diffs > hard_positive_threshold]

            # Hard negatives: negative pairs that are close
            hard_negative_threshold = pos_diffs.max() if len(pos_diffs) > 0 else neg_diffs.mean()
            hard_negatives = neg_diffs[neg_diffs < hard_negative_threshold]

            # If no hard examples found, use all examples
            if len(hard_positives) == 0:
                hard_positives = pos_diffs
            if len(hard_negatives) == 0:
                hard_negatives = neg_diffs
        else:
            # Fallback to using all examples if only one class present
            hard_positives = pos_diffs
            hard_negatives = neg_diffs

        # Compute losses
        positive_loss = (
            hard_positives.pow(2).sum()
            if len(hard_positives) > 0
            else torch.tensor(0.0, device=combined_diff.device)
        )
        negative_loss = (
            F.relu(self.margin - hard_negatives).pow(2).sum()
            if len(hard_negatives) > 0
            else torch.tensor(0.0, device=combined_diff.device)
        )

        total_loss = positive_loss + negative_loss

        # Log detailed metrics
        if self.call_count % self.gate_stats_steps == 0:
            metrics = {
                "loss/total": total_loss.item(),
                "loss/positive": positive_loss.item(),
                "loss/negative": negative_loss.item(),
                "gate/value": gate_value.item(),
                "gate/raw": self.gate.item(),
                "distances/avg_positive": pos_diffs.mean().item() if len(pos_diffs) > 0 else 0,
                "distances/avg_negative": neg_diffs.mean().item() if len(neg_diffs) > 0 else 0,
                "distances/min_positive": pos_diffs.min().item() if len(pos_diffs) > 0 else 0,
                "distances/max_negative": neg_diffs.max().item() if len(neg_diffs) > 0 else 0,
                "hard_mining/num_hard_positives": len(hard_positives) if len(pos_diffs) > 0 else 0,
                "hard_mining/num_hard_negatives": len(hard_negatives) if len(neg_diffs) > 0 else 0,
                "hard_mining/total_positives": len(pos_diffs),
                "hard_mining/total_negatives": len(neg_diffs),
                "differences/avg_local": local_diff.mean().item(),
                "differences/avg_global": global_diff.mean().item(),
            }

            # Log to wandb if available
            if wandb.run is not None:
                wandb.log(metrics)

            # Also print metrics for debugging
            print(f"\nStep {self.call_count} Metrics:")
            for key, value in metrics.items():
                print(f"  {key}: {value:.4f}")

        return total_loss

    def get_config_dict(self) -> dict:
        """Return configuration for saving/loading."""
        return {
            "margin": self.margin,
            "gate_scale": self.gate_scale,
            "gate_stats_steps": self.gate_stats_steps,
        }
