"""SM90 standalone prenorm, with runtime token strides and bounded split-K.

The caller validates the SM90 device and the <=256-token dispatch contract.
Keep this reduction separate from the SM100 constexpr specialization.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _reduce_prenorm_sm90_kernel(
    partial,
    square_sum,
    output,
    # Runtime, not constexpr: TOKENS is the per-chunk token count and changes on
    # every chunk. As a constexpr it produced 224 cached variants on SM90
    # prefill, each costing a full Triton compilation. SPLITS must stay
    # constexpr because it drives tl.static_range.
    TOKENS,
    SPLITS: tl.constexpr,
    EPS: tl.constexpr,
):
    token = tl.program_id(0)
    columns = tl.arange(0, 32)
    projection = tl.full((32,), 0, tl.float32)
    total_square = tl.full((), 0, tl.float32)
    # Match the existing TileLang normalization's ordered split accumulation.
    for split in tl.static_range(SPLITS):
        projection += tl.load(
            partial + (split * TOKENS + token) * 24 + columns,
            columns < 24,
            other=0,
        )
        total_square += tl.load(square_sum + split * TOKENS + token)
    norm = tl.rsqrt(total_square / 20480 + EPS)
    tl.store(output + token * 24 + columns, projection * norm, columns < 24)


def prenorm(
    residual: torch.Tensor, fn: torch.Tensor, eps: float
) -> torch.Tensor | None:
    """Return FP32 [..., 24] mixes, or None for the original-path fallback."""
    tokens = residual.numel() // 20480
    output = torch.empty(
        (*residual.shape[:-2], 24), dtype=torch.float32, device=residual.device
    )
    if tokens == 0:
        return output

    from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import tf32_hc_prenorm_gemm

    # Shorter K reductions keep large-batch cancellation within the same
    # FP64-oracle tolerance. Preserve the established <=64-token path.
    splits = 32 if tokens > 64 else 16
    partial = torch.empty(
        (splits, tokens, 24), dtype=torch.float32, device=residual.device
    )
    square_sum = torch.empty(
        (splits, tokens), dtype=torch.float32, device=residual.device
    )
    # The native tensor map converts the raw FP32 weight to TF32 with its
    # round-to-nearest-even rule. Pre-rounding with the old TileLang bias
    # changes exact halfway values before that conversion.
    tf32_hc_prenorm_gemm(residual.view(tokens, 20480), fn, partial, square_sum, splits)
    _reduce_prenorm_sm90_kernel[(tokens,)](
        partial,
        square_sum,
        output,
        tokens,
        splits,
        eps,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output
