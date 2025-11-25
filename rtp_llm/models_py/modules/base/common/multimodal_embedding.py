from typing import Sequence

import torch
from torch import nn


class MultimodalEmbeddingInjector(nn.Module):
    """Insert multimodal features into the base embeddings at predefined offsets."""

    def forward(
        self,
        embeddings: torch.Tensor,
        multimodal_features: Sequence[torch.Tensor],
        multimodal_locs: torch.Tensor,
    ) -> torch.Tensor:
        if not multimodal_features:
            return embeddings

        if multimodal_locs.numel() != len(multimodal_features):
            raise ValueError(
                f"multimodal_locs has {multimodal_locs.numel()} entries "
                f"but {len(multimodal_features)} features were provided"
            )

        if embeddings.dim() != 2:
            raise ValueError(
                "embeddings must be a 2D tensor of shape [tokens, hidden_size]"
            )

        locs = multimodal_locs.to(device="cpu", dtype=torch.long).view(-1).tolist()

        hidden_size = embeddings.size(-1)
        for idx, (feature, loc) in enumerate(zip(multimodal_features, locs)):
            if feature is None or feature.numel() == 0:
                continue

            if feature.dim() != 2 or feature.size(-1) != hidden_size:
                raise ValueError(
                    f"feature[{idx}] must have shape [N, {hidden_size}], "
                    f"but got {feature.shape}"
                )

            if feature.dtype != embeddings.dtype:
                raise TypeError(
                    f"dtype mismatch: embeddings are {embeddings.dtype}, "
                    f"feature[{idx}] is {feature.dtype}"
                )

            if feature.device != embeddings.device:
                feature = feature.to(embeddings.device)

            length = feature.size(0)
            if loc < 0 or (loc + length) > embeddings.size(0):
                raise IndexError(
                    f"feature[{idx}] with length {length} cannot be placed at loc {loc} "
                    f"within embeddings of length {embeddings.size(0)}"
                )

            embeddings.narrow(0, loc, length).copy_(feature.contiguous())

        return embeddings


class MultimodalDeepstackInjector(nn.Module):
    """Add per-layer multimodal deepstack embeddings into the hidden states."""

    def forward(
        self,
        hidden: torch.Tensor,
        mm_deepstack_embeds: Sequence[torch.Tensor],
        multimodal_locs: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        if not mm_deepstack_embeds or layer_id < 0:
            return hidden

        if multimodal_locs.numel() != len(mm_deepstack_embeds):
            raise ValueError(
                f"multimodal_locs has {multimodal_locs.numel()} entries "
                f"but {len(mm_deepstack_embeds)} deepstack tensors were provided"
            )

        locs = multimodal_locs.to(device="cpu", dtype=torch.long).view(-1).tolist()
        hidden_size = hidden.size(-1)

        for idx, (stack, loc) in enumerate(zip(mm_deepstack_embeds, locs)):
            if stack.dim() != 3:
                raise ValueError(
                    f"deepstack tensor[{idx}] must have shape [layers, tokens, {hidden_size}], "
                    f"but got {stack.shape}"
                )

            if layer_id >= stack.size(0):
                continue

            layer_embed = stack[layer_id]
            if layer_embed.size(-1) != hidden_size:
                raise ValueError(
                    f"deepstack tensor[{idx}] hidden size mismatch: expected {hidden_size}, "
                    f"got {layer_embed.size(-1)}"
                )

            if layer_embed.dtype != hidden.dtype:
                raise TypeError(
                    f"dtype mismatch: hidden is {hidden.dtype}, "
                    f"deepstack tensor[{idx}] is {layer_embed.dtype}"
                )

            if layer_embed.device != hidden.device:
                layer_embed = layer_embed.to(hidden.device)

            length = layer_embed.size(0)
            if loc < 0 or (loc + length) > hidden.size(0):
                raise IndexError(
                    f"deepstack tensor[{idx}] with length {length} cannot be placed at "
                    f"loc {loc} within hidden of length {hidden.size(0)}"
                )

            hidden_slice = hidden.narrow(0, loc, length)
            hidden_slice.add_(layer_embed.contiguous())

        return hidden
