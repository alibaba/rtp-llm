"""Shared input construction for direct SM120 FP8 binding contract tests."""

import torch

from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8


def per_output_channel_kblock_cast_to_fp8(weight: torch.Tensor, block: int = 128):
    """Reference quantizer for physical (N, K) CUTLASS weights."""
    n, k = weight.shape
    padded_k = (k + block - 1) // block * block
    padded = torch.zeros(n, padded_k, dtype=torch.float32, device=weight.device)
    padded[:, :k] = weight
    view = padded.reshape(n, padded_k // block, block)
    scale = view.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4) / 448.0
    quant = (view / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return (
        quant.reshape(n, padded_k)[:, :k].contiguous(),
        scale.squeeze(-1).contiguous(),
    )


def make_blockwise_op_inputs(M: int, K: int, N: int, device: str = "cuda"):
    input_tensor = torch.randn(M, K, dtype=torch.bfloat16, device=device).contiguous()
    weight_bf16 = (
        torch.randn((N, K), dtype=torch.bfloat16, device=device) * 0.1
    ).contiguous()
    A, A_sf = sgl_per_token_group_quant_fp8(
        input_tensor,
        group_size=128,
        eps=1e-4,
        column_major_scales=True,
        scale_tma_aligned=False,
        scale_ue8m0=False,
    )
    B, B_sf = per_output_channel_kblock_cast_to_fp8(weight_bf16)
    D = torch.empty(M, N, dtype=torch.bfloat16, device=device)
    return D, A, B, A_sf, B_sf
