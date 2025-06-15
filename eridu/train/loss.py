from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from sentence_transformers.losses import OnlineContrastiveLoss
from sentence_transformers.losses.ContrastiveLoss import SiameseDistanceMetric
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


class MetricsOnlineContrastiveLoss(OnlineContrastiveLoss):
    """OnlineContrastiveLoss with detailed metrics logging for WandB."""

    def __init__(
        self,
        model: SentenceTransformer,
        margin: float = 0.5,
        distance_metric: SiameseDistanceMetric = SiameseDistanceMetric.COSINE_DISTANCE,
        gate_stats_steps: int = 100,
    ) -> None:
        super().__init__(model=model, margin=margin, distance_metric=distance_metric)
        self.call_count = 0
        self.last_metrics: dict[str, float] = {}
        self.gate_stats_steps = gate_stats_steps

    def forward(
        self,
        sentence_features: Iterable[dict[str, Tensor]],
        labels: Tensor,
        size_average: bool = False,
    ) -> Tensor:
        """Forward pass with metrics collection.

        Note: size_average parameter is ignored by parent OnlineContrastiveLoss
        implementation - loss is always summed, never averaged.
        """
        self.call_count += 1

        # Call parent forward to get the loss (size_average is ignored by parent)
        total_loss = super().forward(sentence_features, labels, size_average)

        # Get embeddings and distances for metrics (recompute for metrics only)
        embeddings = [
            self.model(sentence_feature)["sentence_embedding"]
            for sentence_feature in sentence_features
        ]

        # Calculate distances using parent class distance metric
        if self.distance_metric == SiameseDistanceMetric.EUCLIDEAN:
            distances = F.pairwise_distance(embeddings[0], embeddings[1], p=2)
        elif self.distance_metric == SiameseDistanceMetric.MANHATTAN:
            distances = F.pairwise_distance(embeddings[0], embeddings[1], p=1)
        elif self.distance_metric == SiameseDistanceMetric.COSINE_DISTANCE:
            distances = 1 - F.cosine_similarity(embeddings[0], embeddings[1])
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

        # Calculate component losses for metrics
        pos_mask = labels == 1
        neg_mask = labels == 0

        pos_distances = (
            distances[pos_mask] if pos_mask.any() else torch.tensor(0.0, device=distances.device)
        )
        neg_distances = (
            distances[neg_mask] if neg_mask.any() else torch.tensor(0.0, device=distances.device)
        )

        pos_loss = (
            pos_distances.pow(2).sum()
            if pos_mask.any()
            else torch.tensor(0.0, device=distances.device)
        )
        neg_loss = (
            F.relu(self.margin - neg_distances).pow(2).sum()
            if neg_mask.any()
            else torch.tensor(0.0, device=distances.device)
        )

        # Store detailed metrics
        self.last_metrics = {
            "total_loss": total_loss.item(),
            "pos_loss": pos_loss.item(),
            "neg_loss": neg_loss.item(),
            "pos_distance_mean": pos_distances.mean().item() if pos_mask.any() else 0.0,
            "neg_distance_mean": neg_distances.mean().item() if neg_mask.any() else 0.0,
            "pos_distance_std": pos_distances.std().item() if pos_mask.any() else 0.0,
            "neg_distance_std": neg_distances.std().item() if neg_mask.any() else 0.0,
            "distance_mean": distances.mean().item(),
            "distance_std": distances.std().item(),
            "pos_pairs_count": pos_mask.sum().item(),
            "neg_pairs_count": neg_mask.sum().item(),
            "margin_violations": (
                (neg_distances < self.margin).sum().item() if neg_mask.any() else 0
            ),
        }

        # Log metrics periodically
        if self.call_count % self.gate_stats_steps == 0:
            wandb.log(
                {
                    "train/contrastive_total_loss": total_loss.item(),
                    "train/contrastive_pos_loss": pos_loss.item(),
                    "train/contrastive_neg_loss": neg_loss.item(),
                    "train/pos_distance_mean": self.last_metrics["pos_distance_mean"],
                    "train/neg_distance_mean": self.last_metrics["neg_distance_mean"],
                    "train/pos_distance_std": self.last_metrics["pos_distance_std"],
                    "train/neg_distance_std": self.last_metrics["neg_distance_std"],
                    "train/distance_mean": self.last_metrics["distance_mean"],
                    "train/distance_std": self.last_metrics["distance_std"],
                    "train/pos_pairs_count": self.last_metrics["pos_pairs_count"],
                    "train/neg_pairs_count": self.last_metrics["neg_pairs_count"],
                    "train/margin_violations": self.last_metrics["margin_violations"],
                    "train/step": self.call_count,
                }
            )

        return total_loss
