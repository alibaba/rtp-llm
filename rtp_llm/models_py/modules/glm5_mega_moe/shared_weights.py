"""Validate and transform the shared FP8 weights used by MegaMoE."""

from typing import NamedTuple

import torch

from .mega_moe import GLM5MegaMoeCfg
from .quant_layouts import prepare_fp8_weight_scale_for_deepgemm


class SharedExpertWeights(NamedTuple):
    l1: tuple[torch.Tensor, torch.Tensor]
    l2: tuple[torch.Tensor, torch.Tensor]
    num_experts: int


def prepare_shared_expert_weights(
    cfg: GLM5MegaMoeCfg,
    w1: torch.Tensor,
    s1: torch.Tensor,
    w2: torch.Tensor,
    s2: torch.Tensor,
    *,
    fp8_routed: bool,
) -> SharedExpertWeights:
    import deep_gemm

    if fp8_routed and not hasattr(deep_gemm, "get_block_m_for_mega_moe_fp8"):
        raise RuntimeError("FP8 MegaMoE requires DeepGEMM shared-expert support")
    if w1.dtype != torch.float8_e4m3fn or w2.dtype != torch.float8_e4m3fn:
        raise TypeError(
            f"MegaMoE shared weights must use FP8 e4m3, got {w1.dtype} and {w2.dtype}"
        )
    intermediate = int(w2.shape[1])
    if intermediate <= 0 or intermediate % cfg.moe_inter_dim:
        raise ValueError(
            "shared intermediate size must be a positive multiple of routed "
            f"intermediate size, got {intermediate} and {cfg.moe_inter_dim}"
        )
    expected_w1, expected_w2 = (2 * intermediate, cfg.dim), (cfg.dim, intermediate)
    if tuple(w1.shape) != expected_w1 or tuple(w2.shape) != expected_w2:
        raise ValueError(
            f"shared weight shape mismatch: expected {expected_w1}/{expected_w2}, "
            f"got {tuple(w1.shape)}/{tuple(w2.shape)}"
        )
    if s1.dtype == torch.float32 or s2.dtype == torch.float32:
        if s1.dtype != torch.float32 or s2.dtype != torch.float32:
            raise TypeError(
                "both shared scales must be float32 or both native/packed UE8M0"
            )
        from rtp_llm.models_py.kernels.cuda.fp8_kernel import requant_weight_ue8m0

        w1, sf1 = requant_weight_ue8m0(w1.contiguous(), s1)
        w2, sf2 = requant_weight_ue8m0(w2.contiguous(), s2)
    else:
        sf1 = prepare_fp8_weight_scale_for_deepgemm(s1, 2 * intermediate, cfg.dim)
        sf2 = prepare_fp8_weight_scale_for_deepgemm(s2, cfg.dim, intermediate)
    transform = (
        deep_gemm.transform_weights_for_mega_moe_fp8
        if fp8_routed
        else deep_gemm.transform_weights_for_mega_moe
    )
    l1, l2 = transform((w1.contiguous(), sf1), (w2.contiguous(), sf2))
    return SharedExpertWeights(l1, l2, intermediate // cfg.moe_inter_dim)
