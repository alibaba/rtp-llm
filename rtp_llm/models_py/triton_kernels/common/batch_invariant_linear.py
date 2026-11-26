"""BF16 matmul with a fixed K traversal, following vLLM's batch-invariant matmul."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["M"])
def _linear(
    x,
    weight,
    bias,
    output,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    STRIDE_WN: tl.constexpr,
    STRIDE_WK: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    rows = (tl.program_id(0) * BM + tl.arange(0, BM)).to(tl.int64)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)

    # Each output element is owned by one CTA. M only changes grid size and masks.
    for start in range(0, K, BK):
        inner = start + ks
        a = tl.load(
            x + rows[:, None] * K + inner[None, :],
            (rows[:, None] < M) & (inner[None, :] < K),
            other=0,
        )
        b = tl.load(
            weight + inner[:, None] * STRIDE_WK + cols[None, :] * STRIDE_WN,
            (inner[:, None] < K) & (cols[None, :] < N),
            other=0,
        )
        acc = tl.dot(a, b, acc)
    if HAS_BIAS:
        acc += tl.load(bias + cols, cols < N, other=0).to(tl.float32)[None, :]
    tl.store(
        output + rows[:, None] * N + cols[None, :],
        acc,
        (rows[:, None] < M) & (cols[None, :] < N),
    )


_BK = 64

# BM and BN only decide which CTA owns an output element; they never change the
# order in which that element's products are summed. Only BK and the ascending
# K traversal have to stay fixed for batch invariance, so the tile can be sized
# for the machine. Every entry below is measured bit-identical to the last one,
# the historical (16, 64) tile, on SM103 across M, both weight strides, FP32 and
# BF16 output, and K/N that do not divide the tile. Descending order; the first
# tile whose staged A and B buffers fit the device's opt-in shared memory wins,
# so a pre-Hopper GPU keeps a small tile instead of failing to compile.
_TILES = (
    (128, 256, 8, 4),
    (128, 256, 8, 3),
    (128, 128, 8, 4),
    (128, 128, 8, 3),
    (64, 128, 8, 3),
    (32, 128, 4, 3),
    (16, 64, 4, 3),
)

_tile_cache: dict[int, tuple[int, int, int, int]] = {}


def _select_tile(device: torch.device) -> tuple[int, int, int, int]:
    index = device.index if device.index is not None else torch.cuda.current_device()
    tile = _tile_cache.get(index)
    if tile is None:
        properties = torch.cuda.get_device_properties(index)
        budget = getattr(
            properties,
            "shared_memory_per_block_optin",
            properties.shared_memory_per_block,
        )
        # (BM x BK) + (BK x BN) BF16 elements per pipeline stage. Triton may
        # shorten the pipeline for small tiles, so this is an upper bound.
        tile = next(
            candidate
            for candidate in _TILES
            if (candidate[0] * _BK + _BK * candidate[1]) * 2 * candidate[3] <= budget
        )
        _tile_cache[index] = tile
    return tile


def batch_invariant_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    output_fp32: bool = False,
) -> torch.Tensor:
    """Multiply BF16 activations by a BF16 [out_features, in_features] weight."""
    n, k = weight.shape
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16 or x.size(-1) != k:
        raise ValueError(
            "batch-invariant linear requires compatible BF16 input and weight"
        )
    output = torch.empty(
        (*x.shape[:-1], n),
        dtype=torch.float32 if output_fp32 else torch.bfloat16,
        device=x.device,
    )
    if output.numel() == 0:
        return output

    rows = x.reshape(-1, k).contiguous()
    bias = bias.contiguous() if bias is not None else None
    m = rows.size(0)
    bm, bn, warps, stages = _select_tile(x.device)
    bn = min(bn, 16) if n <= 16 else bn
    _linear[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
        rows,
        weight,
        bias,
        output,
        m,
        n,
        k,
        *weight.stride(),
        bm,
        bn,
        _BK,
        bias is not None,
        num_warps=warps,
        num_stages=stages,
    )
    return output
