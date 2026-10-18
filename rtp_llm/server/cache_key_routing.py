from typing import List, Optional, Sequence, Tuple


def route_cache_keys_for_page_rr(
    block_cache_keys: List[int], page_rr_enabled: bool, cp_size: int
) -> List[int]:
    # Input keys must be computed at physical-block granularity. Page-RR only
    # changes which existing rolling-hash keys are routed to flexlb; it does not
    # change the request hash block size to virtualBlockSize.
    if not page_rr_enabled or cp_size <= 1:
        return block_cache_keys
    # Device/cache-connector Page-RR uses the last rank's logical block key as
    # the canonical key for one virtual block: K(cp_size-1), K(2*cp_size-1), ...
    return block_cache_keys[cp_size - 1 :: cp_size]


def get_v41_cache_key_seed(model_config=None) -> int:
    # The model owns the cache namespace; the frontend's local TP/CP is unrelated.
    if model_config is not None:
        return int(getattr(model_config, "cache_key_hash_seed", 0))
    return 0


def _multimodal_cache_identity(
    cache_spans: Sequence[Tuple[int, Sequence[int]]],
    block_begin: int,
    block_count: int,
) -> List[int]:
    """Mirror CompleteTokenIds::multimodalCacheIdentity for one block."""
    identity: List[int] = []
    block_end = block_begin + block_count
    for span_begin, row_hashes in cache_spans:
        span_end = span_begin + len(row_hashes)
        overlap_begin = max(block_begin, span_begin)
        overlap_end = min(block_end, span_end)
        if overlap_begin >= overlap_end:
            continue
        relative_span_begin = overlap_begin - span_begin
        overlap_len = overlap_end - overlap_begin
        identity.extend(
            (
                -42,
                overlap_begin - block_begin,
                len(row_hashes),
                relative_span_begin,
                overlap_len,
            )
        )
        identity.extend(
            row_hashes[relative_span_begin : relative_span_begin + overlap_len]
        )
    return identity


def get_block_cache_keys(
    token_ids,
    block_size,
    cache_key_seed=0,
    multimodal_cache_spans: Optional[Sequence[Tuple[int, Sequence[int]]]] = None,
):
    from rtp_llm.ops import cpp_get_block_cache_keys

    chunks = [
        list(token_ids[i : i + block_size])
        for i in range(0, len(token_ids) - block_size + 1, block_size)
    ]
    if not multimodal_cache_spans:
        return cpp_get_block_cache_keys(chunks, cache_key_seed)

    # The backend rolls each physical block's canonical token ids into the hash,
    # then rolls its cache-only multimodal identity into the same hash state.
    # Feed those as alternating segments to the existing C++ rolling hasher and
    # retain the post-identity key for each block.
    rolling_segments = []
    for block_index, chunk in enumerate(chunks):
        block_begin = block_index * block_size
        rolling_segments.extend(
            (
                chunk,
                _multimodal_cache_identity(
                    multimodal_cache_spans, block_begin, len(chunk)
                ),
            )
        )
    return cpp_get_block_cache_keys(rolling_segments, cache_key_seed)[1::2]
