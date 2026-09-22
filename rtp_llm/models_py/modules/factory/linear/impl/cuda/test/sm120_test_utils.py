"""Shared input construction for direct SM120 FP8 binding contract tests."""

import torch

from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8


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
    blocks = weight_bf16.float().reshape(N // 128, 128, K // 128, 128)
    scales = blocks.abs().amax(dim=(1, 3), keepdim=True).clamp_min(1e-4) / 448.0
    B = (blocks / scales).to(torch.float8_e4m3fn).reshape(N, K).contiguous()
    B_sf = scales.reshape(N // 128, K // 128).contiguous()
    D = torch.empty(M, N, dtype=torch.bfloat16, device=device)
    return D, A, B, A_sf, B_sf
