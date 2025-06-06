from __future__ import annotations
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sentence_transformers.SentenceTransformer import SentenceTransformer


class ContextAdaptiveContrastiveLoss(nn.Module):
    def __init__(self, model: SentenceTransformer, margin: float = 0.5, gate_scale: float = 5.0) -> None:
        super().__init__()
        self.model = model
        self.margin = margin
        self.gate_scale = gate_scale

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        output1 = self.model(sentence_features[0], output_value="token_embeddings")  # (b, t, d)
        output2 = self.model(sentence_features[1], output_value="token_embeddings")
        sent1 = self.model(sentence_features[0], output_value="sentence_embedding")  # (b, d)
        sent2 = self.model(sentence_features[1], output_value="sentence_embedding")

        local_diff = 1 - F.cosine_similarity(output1, output2, dim=-1).mean(dim=1)
        global_diff = 1 - F.cosine_similarity(sent1, sent2)

        gate = torch.sigmoid(self.gate_scale * (global_diff - local_diff))
        final_diff = gate * global_diff + (1 - gate) * local_diff

        pos_mask = labels == 1
        neg_mask = labels == 0

        pos_loss = final_diff[pos_mask].pow(2).sum()
        neg_loss = F.relu(self.margin - final_diff[neg_mask]).pow(2).sum()

        return pos_loss + neg_loss
