from typing import List, Optional, Sequence

import torch
from torch import nn

# Keep this layout contract aligned with cpp/multimodal_processor/MultimodalInputUtils.h.
# Python consumes the flattened C++ representation after transport.


def _normalize_multimodal_locs(
    multimodal_locs: Optional["torch.Tensor | Sequence[int]"],
    expected_count: int,
    value_name: str,
) -> List[int]:
    if multimodal_locs is None:
        raise ValueError(f"multimodal_locs must be provided with {value_name}")

    if isinstance(multimodal_locs, torch.Tensor):
        actual_count = multimodal_locs.numel()
        if (
            multimodal_locs.device.type == "cpu"
            and multimodal_locs.ndim == 1
            and multimodal_locs.dtype in (torch.int32, torch.int64)
        ):
            locs = multimodal_locs.tolist()
        else:
            locs = multimodal_locs.to(device="cpu", dtype=torch.long).view(-1).tolist()
    else:
        actual_count = len(multimodal_locs)
        locs = list(multimodal_locs)

    if actual_count != expected_count:
        raise ValueError(
            f"multimodal_locs has {actual_count} entries "
            f"but {expected_count} {value_name} were provided"
        )
    return locs


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
    """Insert multimodal features into a caller-selected representation space.

    This module only performs positional replacement. Callers own the injection
    stage and must provide features already projected into the representation
    space expected at that stage (for example, before or after an embedding
    LayerNorm).
    """

    def forward(
        self,
        embeddings: torch.Tensor,
        multimodal_features: Sequence[torch.Tensor],
        multimodal_locs: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if not multimodal_features:
            return embeddings

        locs = _normalize_multimodal_locs(
            multimodal_locs, len(multimodal_features), "features"
        )

        if embeddings.dim() != 2:
            raise ValueError(
                "embeddings must be a 2D tensor of shape [tokens, hidden_size]"
            )

        if (
            embeddings.is_cuda
            and embeddings.is_contiguous()
            and isinstance(multimodal_locs, torch.Tensor)
            and all(
                isinstance(feature, torch.Tensor) for feature in multimodal_features
            )
        ):
            from rtp_llm.ops.compute_ops import rtp_llm_ops

            if rtp_llm_ops.try_fused_multimodal_copy_(
                embeddings, multimodal_features, locs
            ):
                return embeddings

        hidden_size = embeddings.size(-1)
        copy_features: List[torch.Tensor] = []
        copy_locs: List[int] = []
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

            if loc < 0:
                raise ValueError(f"feature[{idx}] loc must be non-negative, got {loc}")

            length = feature.size(0)
            if loc + length > embeddings.size(0):
                raise IndexError(
                    f"feature[{idx}] with length {length} cannot be placed at loc {loc} "
                    f"within embeddings of length {embeddings.size(0)}"
                )

            copy_features.append(feature.contiguous())
            copy_locs.append(loc)

        intervals = sorted(
            (loc, loc + feature.size(0))
            for feature, loc in zip(copy_features, copy_locs)
        )
        has_overlap = any(
            current_start < previous_end
            for (_, previous_end), (current_start, _) in zip(intervals, intervals[1:])
        )

        if (
            embeddings.is_cuda
            and embeddings.is_contiguous()
            and copy_features
            and not has_overlap
        ):
            # Batch the small feature copies into one launch. Vision-BERT commonly
            # has one feature per request item, so the scalar fallback otherwise
            # creates O(batch_size) D2D launches on the model stream.
            from rtp_llm.ops.compute_ops import rtp_llm_ops

            rtp_llm_ops.fused_multimodal_copy_(embeddings, copy_features, copy_locs)
        else:
            # Sequential copies preserve strided destination views and define
            # last-writer-wins semantics for overlapping ranges.
            storage = embeddings.untyped_storage()
            dst_begin = storage.data_ptr()
            dst_end = dst_begin + storage.nbytes()
            for feature, loc in zip(copy_features, copy_locs):
                src_begin = feature.data_ptr()
                if src_begin < dst_end and dst_begin < src_begin + feature.nbytes:
                    feature = feature.clone()
                embeddings.narrow(0, loc, feature.size(0)).copy_(feature)

        return embeddings


class MultimodalDeepstackInjector(nn.Module):
    """Add per-layer multimodal deepstack embeddings into the hidden states."""

    def forward(
        self,
        hidden: torch.Tensor,
        mm_deepstack_embeds: Sequence[torch.Tensor],
        multimodal_locs: Optional["torch.Tensor | Sequence[int]"],
        layer_id: int,
    ) -> torch.Tensor:
        if not mm_deepstack_embeds or layer_id < 0:
            return hidden

        locs = _normalize_multimodal_locs(
            multimodal_locs,
            len(mm_deepstack_embeds),
            "deepstack tensors",
        )
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

            if loc < 0:
                raise ValueError(
                    f"deepstack tensor[{idx}] loc must be non-negative, got {loc}"
                )

            length = layer_embed.size(0)
            if loc + length > hidden.size(0):
                raise IndexError(
                    f"deepstack tensor[{idx}] with length {length} cannot be placed at "
                    f"loc {loc} within hidden of length {hidden.size(0)}"
                )

            hidden_slice = hidden.narrow(0, loc, length)
            hidden_slice.add_(layer_embed.contiguous())

        return hidden
