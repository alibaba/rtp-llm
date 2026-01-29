import logging
import torch

from rtp_llm.models_py.utils.arch import is_cuda

if is_cuda():
    from rtp_llm.ops.compute_ops import (
        w4a8_group_gemm,
    )
else:
    logging.info("skip import w4a8 quant from rtp_llm_ops for non cuda platform")


def w4a8_group_gemm_ptpc(output: torch.Tensor,
                         a: torch.Tensor,
                         b: torch.Tensor,
                         b_scales: torch.Tensor,
                         a_out_scales: torch.Tensor,
                         expert_offsets: torch.Tensor,
                         problem_sizes: torch.Tensor,
                         a_strides: torch.Tensor,
                         b_strides: torch.Tensor,
                         b_scales_strides: torch.Tensor,
                         c_strides: torch.Tensor,
                         group_size: int):
    swap_ab = True
    per_act_token = True
    per_out_ch = False
    b_out_scales = torch.ones(
        (b.shape[0], b.shape[1] if per_out_ch else 1), dtype=torch.float32, device=b.device)
    profile = False
    m_tile = 0
    n_tile = 0
    k_tile = 0
    cluster_m = 0
    cluster_n = 0
    cluster_k = 0
    w4a8_group_gemm(output,
                    a,
                    b,
                    b_scales,
                    a_out_scales,
                    b_out_scales,
                    expert_offsets,
                    problem_sizes,
                    a_strides,
                    b_strides,
                    b_scales_strides,
                    c_strides,
                    group_size,
                    swap_ab,
                    per_act_token,
                    per_out_ch,
                    profile,
                    m_tile,
                    n_tile,
                    k_tile,
                    cluster_m,
                    cluster_n,
                    cluster_k)
