"""Zero-copy BF16 token views of paged caches with gaps between blocks."""

import torch


def flat_sparse_cache_view(cache: torch.Tensor) -> tuple[torch.Tensor, int]:
    if cache.ndim != 3 or cache.stride(2) != 1:
        raise ValueError("sparse cache requires packed token rows")
    blocks, entries, width = cache.shape
    row_stride = cache.stride(1)
    if row_stride != width or cache.stride(0) % row_stride:
        raise ValueError(
            "sparse cache block stride must be a whole number of token rows"
        )
    physical_entries = cache.stride(0) // row_stride
    rows = (blocks - 1) * physical_entries + entries
    # The flat view includes gaps owned by other groups/layers. Only remapped
    # sparse indices may read it; constructing this view copies no cache data.
    return cache.as_strided((rows, 1, width), (row_stride, width, 1)), physical_entries
