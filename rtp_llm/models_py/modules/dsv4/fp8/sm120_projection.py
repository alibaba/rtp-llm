"""SM120 compressor/indexer projections with fixed padding and rounding."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compressor_linear(x, weight, gemm):
    leading_shape = x.shape[:-1]
    x_2d = x.reshape(-1, x.shape[-1])
    rows = x_2d.shape[0]
    # Preserve the decode/verify reduction geometry before FP8 KV quantization.
    if x.ndim == 3 and 0 < rows < 8:
        x_2d = F.pad(x_2d, (0, 0, 0, 8 - rows))
    out_2d = gemm(x_2d, weight)
    return out_2d[:rows].reshape(*leading_shape, weight.shape[0])


def compressor_prefill(x, weight, project, chunk_lengths):
    if x.ndim == 2 and chunk_lengths is not None:
        if (
            any(length < 0 for length in chunk_lengths)
            or sum(chunk_lengths) != x.shape[0]
        ):
            raise ValueError("CP request chunks must cover compressor input rows")
    from rtp_llm.models_py.triton_kernels.sm120_batch_invariant_linear import (
        batch_invariant_linear,
        is_supported,
    )

    if is_supported(x, weight):
        return batch_invariant_linear(x, weight, out_dtype=torch.float32)
    return project(x, weight)


def indexer_prefill(x, weight):
    from rtp_llm.models_py.triton_kernels.sm120_batch_invariant_linear import (
        batch_invariant_linear,
        is_supported,
    )

    if is_supported(x, weight):
        return batch_invariant_linear(x, weight, out_dtype=x.dtype)
    return F.linear(x, weight)


def indexer_decode(x, weight):
    if x.dtype != torch.bfloat16:
        return F.linear(x, weight)
    flat = x.reshape(-1, x.shape[-1])
    if flat.shape[0] == 0:
        return F.linear(x, weight)
    # Retain the BF16 rounding boundary and fixed row count before TopK.
    tiles = []
    for start in range(0, flat.shape[0], 8):
        tile = flat[start : start + 8]
        rows = tile.shape[0]
        if rows < 8:
            tile = F.pad(tile, (0, 0, 0, 8 - rows))
        projected = torch.mm(tile, weight.t(), out_dtype=torch.float32)
        tiles.append(projected[:rows].to(x.dtype))
    result = tiles[0] if len(tiles) == 1 else torch.cat(tiles, dim=0)
    return result.reshape(*x.shape[:-1], weight.shape[0])
