from typing import List, Optional, Sequence, Tuple

import torch
from torch import nn


def _sequence_offsets(
    cu_seqlens: torch.Tensor,
    token_count: int,
    cu_seqlens_host: Optional[torch.Tensor] = None,
) -> List[Tuple[int, int]]:
    """Return validated packed-request ranges without assuming CUDA metadata."""
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must be a one-dimensional [batch + 1] tensor")
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise ValueError("cu_seqlens must use an integer dtype")
    source = (
        cu_seqlens_host
        if cu_seqlens_host is not None and cu_seqlens_host.numel()
        else cu_seqlens
    )
    offsets = [int(value) for value in source.detach().cpu().tolist()]
    if offsets[0] != 0 or offsets[-1] != token_count:
        raise ValueError(
            f"cu_seqlens must start at 0 and end at {token_count}, got {offsets}"
        )
    if any(left > right for left, right in zip(offsets, offsets[1:])):
        raise ValueError("cu_seqlens must be non-decreasing")
    return list(zip(offsets, offsets[1:]))


def prepare_mtp_multimodal_inputs(
    input_ids: torch.Tensor,
    multimodal_features: Sequence[torch.Tensor],
    multimodal_locs: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_host: Optional[torch.Tensor] = None,
    already_shifted: bool = False,
) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
    """Shift multimodal spans with MTP prefill and mask feature-hash token IDs.

    MTP prefill shifts each packed request left by one token and appends the
    target sample. Multimodal feature rows must follow that shift. Context
    parallel preprocessing performs the shift before rank-local slicing and
    sets ``already_shifted`` so it is not applied twice. Feature token IDs are
    cache hashes rather than vocab IDs, so injected rows are replaced with a
    safe ID before embedding lookup.
    """
    if multimodal_locs.numel() != len(multimodal_features):
        raise ValueError(
            f"multimodal_locs has {multimodal_locs.numel()} entries "
            f"but {len(multimodal_features)} features were provided"
        )
    ranges = _sequence_offsets(
        cu_seqlens,
        input_ids.numel(),
        cu_seqlens_host=cu_seqlens_host,
    )
    locs = multimodal_locs.to(device="cpu", dtype=torch.long).view(-1).tolist()
    shifted_features: List[torch.Tensor] = []
    shifted_locs: List[int] = []
    if already_shifted:
        shifted_features = list(multimodal_features)
        shifted_locs = locs
    else:
        for feature, loc in zip(multimodal_features, locs):
            feature_end = loc + feature.size(0) - 1
            request_starts = [start for start, _ in ranges if start <= feature_end]
            if not request_starts:
                raise ValueError(
                    f"multimodal feature ending at {feature_end} is outside packed requests"
                )
            request_start = max(request_starts)
            dropped_rows = max(0, request_start - loc + 1)
            shifted_features.append(feature[dropped_rows:])
            shifted_locs.append(max(loc - 1, request_start))

    shifted_locs_tensor = torch.tensor(shifted_locs, dtype=torch.int32)
    masked_ids = input_ids.clone()
    for feature, loc in zip(shifted_features, shifted_locs):
        if feature.numel() == 0:
            continue
        length = min(feature.size(0), masked_ids.size(0) - loc)
        if length > 0:
            masked_ids.narrow(0, loc, length).fill_(0)
    return masked_ids, shifted_features, shifted_locs_tensor


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
