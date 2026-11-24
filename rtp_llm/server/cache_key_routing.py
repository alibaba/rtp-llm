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
    # Hash each full page's tokens, then its overlapping image identities in
    # input order, matching CompleteTokenIds::multimodalCacheIdentity.
    full_length = len(chunks) * block_size
    for span_begin, row_hashes in multimodal_cache_spans or ():
        span_length = len(row_hashes)
        begin = max(0, span_begin)
        end = min(full_length, span_begin + span_length)
        if begin >= end:
            continue
        for block_index in range(begin // block_size, (end - 1) // block_size + 1):
            block_begin = block_index * block_size
            first = max(begin, block_begin)
            last = min(end, block_begin + block_size)
            chunk = chunks[block_index]
            chunk.extend(
                (
                    -42,
                    first - block_begin,
                    span_length,
                    first - span_begin,
                    last - first,
                )
            )
            chunk.extend(row_hashes[first - span_begin : last - span_begin])
    return cpp_get_block_cache_keys(chunks, cache_key_seed)
