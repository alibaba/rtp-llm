"""Zero-copy block-planar FP8 cache views, including shared-arena gaps."""

import torch


def paged_indexer_cache_view(cache: torch.Tensor, block_size: int) -> torch.Tensor:
    """Preserve the physical block pitch in the paged MQA ABI.

    The nominal 132-byte entry axis contains block-planar K and scale bytes.
    Flattening blocks into entries would copy an entire shared cache arena.
    """
    if block_size <= 0 or cache.dtype != torch.uint8:
        raise ValueError(
            "paged indexer cache requires uint8 storage and positive block size"
        )
    if cache.ndim == 2:
        if cache.shape[1] != 132 or cache.shape[0] % block_size:
            raise ValueError(
                "flat paged indexer cache must have complete 132-byte pages"
            )
        cache = cache.view(-1, block_size, 1, 132)
    elif cache.ndim == 3:
        cache = cache.unsqueeze(2)
    if cache.ndim != 4 or cache.shape[1:] != (block_size, 1, 132):
        raise ValueError("paged indexer cache must have shape [pages,block_size,1,132]")
    if (
        cache.stride(3) != 1
        or cache.stride(1) != 132
        or cache.stride(0) < block_size * 132
        or cache.stride(0) % 4
    ):
        raise ValueError(
            "paged indexer cache requires packed, nonoverlapping block payloads"
        )
    return cache
