"""Input contracts for Kimi K3's independent MTP draft model.

The draft consumes target hidden states and shifted token IDs. It embeds media
placeholder tokens in place of the feature hashes used by the target's cache.
These helpers do not depend on the K3 text model or its attention kernels.
"""

from typing import Optional, Sequence

import torch


def mtp_positions(inputs) -> torch.Tensor:
    """Recover absolute draft positions when a NoPE batch omits position IDs."""
    positions = inputs.combo_position_ids
    token_count = inputs.input_ids.numel()
    if positions is not None and positions.numel():
        if positions.ndim != 1 or positions.numel() != token_count:
            raise ValueError("K3 MTP requires one absolute position per input token")
        return positions

    attention = inputs.attention_inputs
    device = inputs.input_ids.device
    lengths = attention.input_lengths.to(device=device, dtype=torch.long)
    decode = (
        lengths.new_empty(0)
        if getattr(attention, "is_prefill", False)
        or attention.sequence_lengths is None
        else attention.sequence_lengths.to(device=device, dtype=torch.long)
    )
    prefixes = (
        lengths.new_zeros(lengths.numel() - decode.numel())
        if attention.prefix_lengths is None
        else attention.prefix_lengths.to(device=device, dtype=torch.long)
    )
    query_lengths = torch.cat((torch.ones_like(decode), lengths[decode.numel() :]))
    offsets = torch.cat((decode, prefixes))
    if query_lengths.numel() != offsets.numel():
        raise ValueError("K3 MTP positions require matching query/cache lengths")
    if token_count == 0:
        return torch.empty(0, device=device, dtype=torch.long)
    if query_lengths.numel() == 0:
        raise ValueError("K3 MTP tokens require nonempty request metadata")
    ends = query_lengths.cumsum(0)
    starts = torch.cat((ends.new_zeros(1), ends[:-1]))
    tokens = torch.arange(token_count, device=device)
    requests = torch.searchsorted(ends, tokens, right=True).clamp(max=ends.numel() - 1)
    return tokens - starts[requests] + offsets[requests]


def restore_shifted_media_tokens(
    input_ids: torch.Tensor,
    features: Sequence[torch.Tensor],
    feature_locations: Optional[torch.Tensor],
    cu_seqlens: torch.Tensor,
    media_token_id: Optional[int],
    *,
    cu_seqlens_host: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Replace shifted image hashes with draft embedding IDs, per request.

    Feature locations refer to the target's unshifted input. The draft sees
    each feature one row earlier, except that a request's first row has no
    predecessor in that request and must be dropped.
    """
    if not features:
        return input_ids
    if feature_locations is None or feature_locations.numel() != len(features):
        raise ValueError("K3 MTP multimodal features require matching host locations")
    if media_token_id is None:
        raise ValueError("K3 MTP media tokens require media_placeholder_token_id")
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
    if offsets[0] != 0 or offsets[-1] != input_ids.numel():
        raise ValueError("cu_seqlens must span every K3 MTP input token")
    if any(left > right for left, right in zip(offsets, offsets[1:])):
        raise ValueError("cu_seqlens must be non-decreasing")

    ids = input_ids.clone()
    ranges = list(zip(offsets, offsets[1:]))
    for feature, loc in zip(features, feature_locations.tolist()):
        if feature.ndim == 0 or feature.size(0) == 0:
            raise ValueError("K3 MTP media features must contain token rows")
        last = loc + feature.size(0) - 1
        matching = [(start, end) for start, end in ranges if start <= last < end]
        if len(matching) != 1:
            raise ValueError("K3 MTP media feature location is outside a request")
        start, end = matching[0]
        if loc >= 0 and loc < start:
            raise ValueError("K3 MTP media feature crosses a request boundary")
        dropped = max(0, start - loc + 1)
        offset = max(loc - 1, start)
        count = feature.size(0) - dropped
        if offset + count > end:
            raise ValueError("K3 MTP media feature crosses a request boundary")
        if count > 0:
            ids.narrow(0, offset, count).fill_(media_token_id)
    return ids
