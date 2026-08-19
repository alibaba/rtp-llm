import logging
from typing import Tuple

import torch
from rtp_llm.models_py.kernels.cuda.fp8_kernel.get_best_config import (
    get_cutlass_groupgemm_best_config,
)
from rtp_llm.models_py.kernels.cuda.fp8_quant import (
    _transform_scale_ue8m0,
    align,
    ceil_to_ue8m0,
    create_per_token_group_quant_fp8_output_scale,
    fp8_dtype,
    fp8_max,
    fp8_min,
    per_block_cast_to_fp8,
    requant_weight_ue8m0,
    scaled_fp8_per_tensor_quant,
    scaled_fp8_per_token_quant,
    sgl_per_token_group_quant_fp8,
)
from rtp_llm.models_py.utils.arch import is_cuda

if is_cuda():
    from rtp_kernel.fp8_group_gemm import fp8_grouped_gemm_ptpc
else:
    logging.info("skip import fp8 quant from rtp_llm_ops for non cuda platform")

logger = logging.getLogger(__name__)

__all__ = (
    "create_per_token_group_quant_fp8_output_scale",
    "fp8_dtype",
    "fp8_max",
    "fp8_min",
    "per_block_cast_to_fp8",
    "requant_weight_ue8m0",
    "scaled_fp8_per_tensor_quant",
    "scaled_fp8_per_token_quant",
    "sgl_per_token_group_quant_fp8",
)


def cutlass_moe_mm_fp8_scaled(
    output,
    aq,
    w,
    aq_scale,
    w_scale,
    expert_offsets,
    problem_sizes,
    a_strides,
    b_strides,
    c_strides,
    per_act_token,
    per_out_ch,
    elements_m,
    swap_ab,
) -> None:

    assert per_act_token == True
    assert per_out_ch == False

    E, N, _ = w.shape
    M, K = aq.shape
    configs = get_cutlass_groupgemm_best_config(E, N, K)
    if configs:
        # Get the optimal config
        config = configs[min(configs.keys(), key=lambda x: abs(x - elements_m))]
        tile_m, tile_n, tile_k = config["tile_m"], config["tile_n"], config["tile_k"]
        cluster_m, cluster_n, cluster_k = (
            config["cluster_m"],
            config["cluster_n"],
            config["cluster_k"],
        )
        if swap_ab != config["swap_ab"]:
            logging.warning(
                "Using mismatched gemm config swap_ab, potentially causing cutlass groupgemm performance loss."
            )
        fp8_grouped_gemm_ptpc(
            output,
            aq,
            w,
            aq_scale,
            w_scale,
            expert_offsets,
            problem_sizes,
            a_strides,
            b_strides,
            c_strides,
            per_act_token,
            per_out_ch,
            tile_m,
            tile_n,
            tile_k,
            cluster_m,
            cluster_n,
            cluster_k,
            stage_count=0,
            mainloop_sched=0,
            epilogue_sched=0,
            swap_ab=swap_ab,
            profile=True,
        )
    else:
        fp8_grouped_gemm_ptpc(
            output,
            aq,
            w,
            aq_scale,
            w_scale,
            expert_offsets,
            problem_sizes,
            a_strides,
            b_strides,
            c_strides,
            per_act_token,
            per_out_ch,
            tile_m=128,
            tile_n=128,
            tile_k=128,
            cluster_m=1,
            cluster_n=1,
            cluster_k=1,
            stage_count=0,
            mainloop_sched=0,
            epilogue_sched=0,
            swap_ab=swap_ab,
            profile=True,
        )


def get_best_config_swap_ab(
    E: int,
    M: int,
    N: int,
    K: int,
):
    configs = get_cutlass_groupgemm_best_config(E, N, K)
    if configs:
        config = configs[min(configs.keys(), key=lambda x: abs(x - M))]
        return config["swap_ab"]
    else:
        return M <= 64 * E


def quant_weight_ue8m0_packed(
    weight_dequant: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize source weights once to FP8 with packed UE8M0 scales.

    This is the native SM100/SM120 DeepGEMM representation.  Keeping the
    original floating-point tensor until this operation avoids the lossy
    float-scale FP8 -> BF16 -> UE8M0 FP8 requantization path.
    """
    if weight_dequant.dim() != 2:
        raise ValueError(
            "Direct packed UE8M0 weight quantization requires a 2D tensor, "
            f"got shape {tuple(weight_dequant.shape)}"
        )

    # Limit floating-point temporaries during online loading.  Quantizing the
    # complete matrix in one call materializes both a padded source and a
    # scaled floating-point tensor in addition to the master weight.  Chunking
    # on a 128-row boundary preserves exactly the same quantization blocks.
    row_chunk_size = 1024
    n, k = weight_dequant.shape
    out_w = torch.empty((n, k), dtype=fp8_dtype, device=weight_dequant.device)
    scale_chunks = []
    for row_start in range(0, n, row_chunk_size):
        row_end = min(row_start + row_chunk_size, n)
        chunk_w, chunk_s = per_block_cast_to_fp8(
            weight_dequant[row_start:row_end], use_ue8m0=True
        )
        out_w[row_start:row_end].copy_(chunk_w)
        scale_chunks.append(chunk_s)

    out_s = torch.cat(scale_chunks, dim=0)
    return out_w, _transform_scale_ue8m0(out_s, mn=out_w.shape[-2])
def per_token_cast_to_fp8(
    x: torch.Tensor, use_ue8m0: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    padded_n = align(n, 128)
    x_padded = torch.empty((m, padded_n), dtype=x.dtype, device=x.device).fill_(0)
    x_padded[:, :n] = x
    x_view = x_padded.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    sf = x_amax / 448.0
    sf = ceil_to_ue8m0(sf) if use_ue8m0 else sf
    return (x_view * (1.0 / sf.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, padded_n)[
        :, :n
    ].contiguous(), sf
