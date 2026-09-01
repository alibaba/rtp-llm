from typing import List, Sequence

import torch
from torch import nn


def reshape_extra_input_to_deepstack(
    extra_input: Sequence[torch.Tensor],
    multimodal_features: Sequence[torch.Tensor],
) -> List[torch.Tensor]:
    """Reshape flat 1-D extra-input tensors back into deepstack [layers, tokens, hidden].

    Each extra-input tensor is the flattened deepstack embedding for one image. Tokens and
    hidden are taken from the matching multimodal feature ([tokens, hidden]); the number of
    layers is derived from the element count. This is the model-specific inverse of the
    flatten done in the qwen3-vl producer.
    """
    deepstack: List[torch.Tensor] = []
    for flat, feature in zip(extra_input, multimodal_features):
        tokens = feature.size(0)
        hidden = feature.size(-1)
        layers = flat.numel() // (tokens * hidden)
        deepstack.append(flat.reshape(layers, tokens, hidden))
    return deepstack


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

            # A feature may overlap only part of the current input: loc < 0 means
            # its head is in the reused prefix, while the right edge may extend
            # into a later prefill chunk. Inject only the intersection and keep
            # the source rows aligned with the destination rows.
            src_start = -loc if loc < 0 else 0
            dst_start = max(loc, 0)
            if src_start >= feature.size(0) or dst_start >= embeddings.size(0):
                continue

            length = min(
                feature.size(0) - src_start,
                embeddings.size(0) - dst_start,
            )
            if length <= 0:
                continue

            embeddings.narrow(0, dst_start, length).copy_(
                feature.narrow(0, src_start, length).contiguous()
            )

        return embeddings


class MultimodalDeepstackInjector(nn.Module):
    """Add per-layer multimodal deepstack embeddings into the hidden states."""

    def forward(
        self,
        hidden: torch.Tensor,
        mm_deepstack_embeds: Sequence[torch.Tensor],
        multimodal_locs: "torch.Tensor | Sequence[int]",
        layer_id: int,
    ) -> torch.Tensor:
        if not mm_deepstack_embeds or layer_id < 0:
            return hidden

        if isinstance(multimodal_locs, torch.Tensor):
            if multimodal_locs.numel() != len(mm_deepstack_embeds):
                raise ValueError(
                    f"multimodal_locs has {multimodal_locs.numel()} entries "
                    f"but {len(mm_deepstack_embeds)} deepstack tensors were provided"
                )
            locs = multimodal_locs.to(device="cpu", dtype=torch.long).view(-1).tolist()
        else:
            if len(multimodal_locs) != len(mm_deepstack_embeds):
                raise ValueError(
                    f"multimodal_locs has {len(multimodal_locs)} entries "
                    f"but {len(mm_deepstack_embeds)} deepstack tensors were provided"
                )
            locs = multimodal_locs
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

            # Same partial-prefix handling as the embedding injector: drop the head rows of a
            # partially-cached leading image (loc < 0) and add only the remaining tail at position 0.
            if loc < 0:
                layer_embed = layer_embed[-loc:]
                loc = 0
                if layer_embed.size(0) == 0:
                    continue

            length = layer_embed.size(0)
            if loc + length > hidden.size(0):
                raise IndexError(
                    f"deepstack tensor[{idx}] with length {length} cannot be placed at "
                    f"loc {loc} within hidden of length {hidden.size(0)}"
                )

            hidden_slice = hidden.narrow(0, loc, length)
            hidden_slice.add_(layer_embed.contiguous())

        return hidden
