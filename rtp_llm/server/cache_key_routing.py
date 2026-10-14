from typing import List


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


def get_block_cache_keys(token_ids, block_size, v41_inputs=None, cache_key_seed=0):
    from rtp_llm.ops import cpp_get_block_cache_keys

    if v41_inputs is not None:
        for image in v41_inputs.images:
            if not image.content_sha256 or not image.processor_identity:
                token_ids = token_ids[: image.start]
    chunks = [
        list(token_ids[i : i + block_size])
        for i in range(0, len(token_ids) - block_size + 1, block_size)
    ]
    if v41_inputs is not None:
        for image in v41_inputs.images:
            start, end = image.start, image.start + image.types.numel()
            words = [-41, start, end, image.n_vit_h, image.n_vit_w]
            for digest in (image.content_sha256, image.processor_identity):
                for offset in range(0, len(digest) - 7, 8):
                    word = int(digest[offset : offset + 8], 16)
                    words.append(word if word < 2**31 else word - 2**32)
            for block_index, chunk in enumerate(chunks):
                begin = block_index * block_size
                if end > begin and start < begin + block_size:
                    chunk.extend(words)
    return cpp_get_block_cache_keys(chunks, cache_key_seed)
