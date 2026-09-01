from typing import Sequence

import torch
from torch import nn


class MultimodalEmbeddingInjector(nn.Module):
    """Replace placeholder-token embeddings with multimodal features."""

    def forward(
        self,
        embeddings: torch.Tensor,
        multimodal_features: Sequence[torch.Tensor],
        multimodal_locs: torch.Tensor,
    ) -> torch.Tensor:
        if not multimodal_features:
            return embeddings
        if embeddings.dim() != 2:
            raise ValueError("embeddings must have shape [tokens, hidden_size]")
        if multimodal_locs.numel() != len(multimodal_features):
            raise ValueError(
                f"multimodal feature/location mismatch: "
                f"features={len(multimodal_features)} locations={multimodal_locs.numel()}"
            )

        locs = multimodal_locs.to(device="cpu", dtype=torch.long).view(-1).tolist()
        for index, (feature, loc) in enumerate(zip(multimodal_features, locs)):
            if feature.dim() != 2 or feature.size(1) != embeddings.size(1):
                raise ValueError(
                    f"feature[{index}] must have shape [tokens, {embeddings.size(1)}], "
                    f"got {tuple(feature.shape)}"
                )
            if loc < 0 or loc + feature.size(0) > embeddings.size(0):
                raise IndexError(
                    f"feature[{index}] at {loc} with length {feature.size(0)} "
                    f"does not fit {embeddings.size(0)} token embeddings"
                )
            embeddings.narrow(0, loc, feature.size(0)).copy_(
                feature.to(
                    device=embeddings.device, dtype=embeddings.dtype
                ).contiguous()
            )
        return embeddings
