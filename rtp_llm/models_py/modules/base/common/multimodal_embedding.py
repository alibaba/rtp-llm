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
    if len(extra_input) != len(multimodal_features):
        raise ValueError(
            f"extra_input count ({len(extra_input)}) must match multimodal_features count "
            f"({len(multimodal_features)})"
        )

    deepstack: List[torch.Tensor] = []
    for idx, (flat, feature) in enumerate(zip(extra_input, multimodal_features)):
        if flat.dim() != 1:
            raise ValueError(
                f"extra_input[{idx}] must be a 1-D flat tensor, got shape={list(flat.shape)}"
            )
        if feature.dim() != 2:
            raise ValueError(
                f"multimodal_features[{idx}] must be 2-D ([tokens, hidden]), "
                f"got shape={list(feature.shape)}"
            )

        tokens = feature.size(0)
        hidden = feature.size(-1)
        if tokens <= 0 or hidden <= 0:
            raise ValueError(
                f"multimodal_features[{idx}] must have positive tokens and hidden size, "
                f"got shape={list(feature.shape)}"
            )

        expected_per_layer = tokens * hidden
        if flat.numel() % expected_per_layer != 0:
            raise ValueError(
                f"extra_input[{idx}] numel ({flat.numel()}) is not divisible by "
                f"tokens*hidden ({expected_per_layer}) inferred from "
                f"multimodal_features[{idx}] shape={list(feature.shape)}"
            )

        layers = flat.numel() // expected_per_layer
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

        if isinstance(multimodal_locs, torch.Tensor):
            if multimodal_locs.numel() != len(multimodal_features):
                raise ValueError(
                    f"multimodal_locs has {multimodal_locs.numel()} entries "
                    f"but {len(multimodal_features)} features were provided"
                )
            locs = multimodal_locs.to(device="cpu", dtype=torch.long).view(-1).tolist()
        else:
            if len(multimodal_locs) != len(multimodal_features):
                raise ValueError(
                    f"multimodal_locs has {len(multimodal_locs)} entries "
                    f"but {len(multimodal_features)} features were provided"
                )
            locs = list(multimodal_locs)

        if embeddings.dim() != 2:
            raise ValueError(
                "embeddings must be a 2D tensor of [tokens, hidden_size]"
            )

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

            # A partially-cached leading image arrives with loc < 0: its head rows already live in the
            # reused KV prefix, so drop them and inject only the remaining tail at the recompute start.
            if loc < 0:
                feature = feature[-loc:]
                loc = 0
                if feature.size(0) == 0:
                    continue

            length = feature.size(0)
            if loc + length > embeddings.size(0):
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
