"""Tensor API for the SM120 INT4/group8 GEMM."""

import torch


def transform(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    from rtp_llm.ops.compute_ops import rtp_llm_ops

    output_size, input_size = weight.shape
    packed = torch.empty(
        (input_size // 16, output_size // 2, 4),
        dtype=torch.int32,
        device=weight.device,
    )
    scales = torch.empty(
        (input_size // 32, output_size, 4), dtype=torch.uint8, device=weight.device
    )
    scratch = torch.empty(weight.shape, dtype=torch.uint8, device=weight.device)
    rtp_llm_ops.w4a16_sm120_transform(weight, packed, scales, scratch)
    return packed, scales


def gemm(
    inputs: torch.Tensor,
    packed: torch.Tensor,
    scales: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    split_k: int = 0,
) -> torch.Tensor:
    from rtp_llm.ops.compute_ops import rtp_llm_ops

    input_size, output_size = packed.shape[0] * 16, packed.shape[1] * 2
    if out is None:
        out = torch.empty(
            (inputs.shape[0], output_size), dtype=inputs.dtype, device=inputs.device
        )
    rtp_llm_ops.w4a16_sm120_gemm(
        inputs, packed, scales, out, output_size, input_size, split_k
    )
    return out
