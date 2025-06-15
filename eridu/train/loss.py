from __future__ import annotations

import typing
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from sentence_transformers.SentenceTransformer import SentenceTransformer
from torch import Tensor, nn


class ContextAdaptiveContrastiveLoss(nn.Module):
    def __init__(
        self, model: SentenceTransformer, margin: float = 0.5, gate_scale: float = 1.0
    ) -> None:
        super().__init__()
        self.model = model
        self.margin = margin
        self.gate_scale = gate_scale

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
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

        gate = torch.sigmoid(self.gate_scale * (global_diff - local_diff))

        # Log every 100 forward passes
        if typing.cast(int, self.call_count) % 100 == 0:
            print(
                f"Step {self.call_count} - Gate stats: mean={gate.mean():.3f}, std={gate.std():.3f}"
            )
            print(f"Global diff: mean={global_diff.mean():.3f}, std={global_diff.std():.3f}")
            print(f"Local diff: mean={local_diff.mean():.3f}, std={local_diff.std():.3f}")

        pos_mask = labels == 1
        neg_mask = labels == 0

        pos_loss = final_diff[pos_mask].pow(2).sum()
        neg_loss = F.relu(self.margin - final_diff[neg_mask]).pow(2).sum()

        loss: Tensor = pos_loss + neg_loss
        return loss
