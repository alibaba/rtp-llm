"""Shared input construction for direct SM120 FP8 binding contract tests."""

import torch

from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
from rtp_llm.test.utils.numeric_util import per_block_cast_to_fp8


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
    B, B_sf = per_block_cast_to_fp8(weight_bf16, use_ue8m0=False)
    D = torch.empty(M, N, dtype=torch.bfloat16, device=device)
    return D, A, B, A_sf, B_sf
