"""Optional token replacement produced alongside one media item's embeddings."""

from dataclasses import dataclass
from typing import List, Tuple

import torch


@dataclass(frozen=True)
class MMTokenPlan:
    """Replace a complete media marker with tokens and ordered embedding spans.

    Token IDs are vocabulary IDs, including the original IDs inside embedding
    spans. The backend replaces only those spans with feature hashes for cache
    identity. Geometry and tokenization belong to the producing model.
    """

    token_ids: List[int]
    spans: List[Tuple[int, int]]

    def validate(self, embeddings) -> None:
        if len(self.spans) != len(embeddings):
            raise ValueError("token plan spans must match embedding segments")
        end = 0
        for (offset, length), embedding in zip(self.spans, embeddings):
            if offset < end or length <= 0 or offset + length > len(self.token_ids):
                raise ValueError("token plan spans must be ordered and in range")
            if embedding.ndim != 2 or embedding.shape[0] != length:
                raise ValueError("token plan span length differs from embedding")
            end = offset + length
        if any(type(token) is not int or token < 0 for token in self.token_ids):
            raise ValueError("token plan requires nonnegative vocabulary IDs")


def semantic_token_ids(inputs) -> torch.Tensor:
    """Expand each Qwen4 segment's scalar vocabulary marker over its live rows."""
    ids = inputs.input_ids
    mm = getattr(inputs, "multimodal_inputs", None)
    extra = getattr(mm, "mm_extra_input", ()) if mm is not None else ()
    if not extra:
        return ids
    locations = mm.mm_features_locs.reshape(-1)
    features = mm.multimodal_features
    if len(extra) != locations.numel() or len(features) != len(extra):
        raise ValueError("semantic markers must match multimodal segments")
    result = ids.clone()
    for loc, marker, feature in zip(locations.unbind(), extra, features):
        # TP reconstructs opaque sidecars as flat buffers on follower ranks.
        if marker.ndim > 1 or marker.numel() != 1 or marker.dtype != torch.int32:
            raise ValueError("semantic markers require one int32 value")
        # Cropping a segment changes its live rows, never its vocabulary marker.
        # Device indices avoid a host scalar read of the segment location.
        count = feature.shape[0]
        indices = torch.arange(count, device=ids.device) + loc
        values = marker.to(device=ids.device, dtype=ids.dtype, non_blocking=True)
        result.index_copy_(0, indices.long(), values.expand(count))
    return result
