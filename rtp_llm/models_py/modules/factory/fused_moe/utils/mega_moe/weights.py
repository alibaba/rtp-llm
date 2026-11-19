"""Weight layout preparation shared by MegaMoE executor recipes."""

from __future__ import annotations

import logging

import torch

from rtp_llm.models_py.kernels.cuda.quant_layouts import (
    convert_fp8_moe_to_fp4_ue8m0,
    prepare_fp4_weight_scale_for_deepgemm,
    prepare_fp8_weight_scale_for_deepgemm,
)

logger = logging.getLogger(__name__)


def _interleave_stacked_up_gate(t: torch.Tensor, gran: int = 8) -> torch.Tensor:
    """Convert RTP's ``[up | gate]`` layout without full-size temporaries."""
    if t.dim() not in (2, 3):
        raise ValueError(f"Expected a 2D or 3D tensor, got shape={tuple(t.shape)}")

    squeeze_group_dim = t.dim() == 2
    if squeeze_group_dim:
        t = t.unsqueeze(0)

    groups, rows, *rest = t.shape
    half = rows // 2
    if rows % (2 * gran) != 0:
        raise ValueError(
            f"Expected the row dimension to be divisible by {2 * gran}, "
            f"got shape={tuple(t.shape)}"
        )

    up = t[:, :half].reshape(groups, half // gran, gran, *rest)
    gate = t[:, half:].reshape(groups, half // gran, gran, *rest)
    result = torch.empty_like(t)
    result_blocks = result.reshape(groups, half // gran, 2, gran, *rest)
    result_blocks[:, :, 0].copy_(gate)
    result_blocks[:, :, 1].copy_(up)
    return result.squeeze(0) if squeeze_group_dim else result


def transform_fp8_to_fp4_mega_weights(
    cfg,
    w1_fp8: torch.Tensor,  # [E_local, N_out, K_in] float8_e4m3fn (gate projection)
    w1_scale: torch.Tensor,  # [E_local, N_out, K_in//128] float32
    w2_fp8: torch.Tensor,  # [E_local, K_out, K_in] float8_e4m3fn (down projection)
    w2_scale: torch.Tensor,  # [E_local, K_out, K_in//128] float32
    w3_fp8: torch.Tensor,  # [E_local, N_out, K_in] float8_e4m3fn (up projection)
    w3_scale: torch.Tensor,  # [E_local, N_out, K_in//128] float32
    *,
    transform_name: str = "transform_weights_for_mega_moe",
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """Prepare weights from FP8 per-block format (GLM-5 checkpoint).

    Converts FP8 → BF16 → FP4 for the mega kernel.
    w1 = gate, w2 = down, w3 = up (following DeepSeek/GLM convention).
    """
    import deep_gemm

    E, D, inter = cfg.n_local_experts, cfg.dim, cfg.moe_inter_dim
    gate, gate_sf = convert_fp8_moe_to_fp4_ue8m0(w1_fp8, w1_scale)
    del w1_fp8, w1_scale
    up, up_sf = convert_fp8_moe_to_fp4_ue8m0(w3_fp8, w3_scale)
    del w3_fp8, w3_scale
    # Quantization groups run along K, so gate/up can be converted separately.
    w13 = torch.cat((gate, up), dim=1)
    s13 = torch.cat((gate_sf, up_sf), dim=1)
    del gate, gate_sf, up, up_sf
    s13 = prepare_fp4_weight_scale_for_deepgemm(s13, 2 * inter, D, E)
    torch.cuda.empty_cache()

    w2, s2 = convert_fp8_moe_to_fp4_ue8m0(w2_fp8, w2_scale)
    del w2_fp8, w2_scale
    s2 = prepare_fp4_weight_scale_for_deepgemm(s2, D, inter, E)
    torch.cuda.empty_cache()
    transformed = getattr(deep_gemm, transform_name)((w13, s13), (w2, s2))
    del w13, s13, w2, s2
    torch.cuda.empty_cache()
    return transformed


def transform_fp8_mega_weights(
    cfg,
    w1_fp8: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_fp8: torch.Tensor,
    w2_scale: torch.Tensor,
    w3_fp8: torch.Tensor,
    w3_scale: torch.Tensor,
):
    """Setup pre-quantized FP8 per-block expert weights for fp8_fp8_mega_moe.

    Inputs follow DeepGEMM convention: w1 is gate, w3 is up, w2 is down.
    Weight scales are either raw 128x128 per-block scales or DeepGEMM's
    packed int32 layout from the FP8 loader.
    """
    import deep_gemm

    E = cfg.n_local_experts
    D = cfg.dim
    inter = cfg.moe_inter_dim

    for name, weight in (
        ("w1_fp8", w1_fp8),
        ("w2_fp8", w2_fp8),
        ("w3_fp8", w3_fp8),
    ):
        if weight.dtype != torch.float8_e4m3fn:
            raise TypeError(
                f"mega_moe_fp8 requires FP8 e4m3 weights; {name} has {weight.dtype}"
            )

    logger.info(
        "[GLM5 MegaMoE FP8] preparing FP8 weights: layer=%d E=%d D=%d inter=%d",
        cfg.layer_id,
        E,
        D,
        inter,
    )

    w13_fp8 = torch.cat([w1_fp8, w3_fp8], dim=1).contiguous()
    if w1_scale.dtype == torch.float8_e8m0fnu:
        # The loader may hand us strided gate/up views. E8M0 has no
        # generic cat kernel; concatenate its bytes without conversion.
        s13 = torch.cat(
            [w1_scale.view(torch.uint8), w3_scale.view(torch.uint8)], dim=1
        ).view(w1_scale.dtype)
    else:
        s13 = torch.cat([w1_scale, w3_scale], dim=1).contiguous()
    del w1_fp8, w1_scale, w3_fp8, w3_scale

    if s13.dtype == torch.float32 or w2_scale.dtype == torch.float32:
        if s13.dtype != torch.float32 or w2_scale.dtype != torch.float32:
            raise TypeError(
                "mega_moe_fp8 requires both FP8 MoE scales to be raw "
                f"float32 or both prepacked int32/e8m0, got {s13.dtype} "
                f"and {w2_scale.dtype}"
            )
        from rtp_llm.models_py.kernels.cuda.fp8_kernel import requant_weight_ue8m0

        logger.info(
            "[GLM5 MegaMoE FP8] requantizing FP8 weights to UE8M0 scales: " "layer=%d",
            cfg.layer_id,
        )
        w13_fp8, s13_int = requant_weight_ue8m0(w13_fp8, s13)
        del s13
        torch.cuda.empty_cache()
        w2_fp8, s2_int = requant_weight_ue8m0(w2_fp8.contiguous(), w2_scale)
        del w2_scale
        torch.cuda.empty_cache()
    else:
        s13_int = prepare_fp8_weight_scale_for_deepgemm(s13, 2 * inter, D, E)
        s2_int = prepare_fp8_weight_scale_for_deepgemm(w2_scale, D, inter, E)
        del s13, w2_scale
    torch.cuda.empty_cache()

    (l1_w, l1_sf), (l2_w, l2_sf) = deep_gemm.transform_weights_for_mega_moe_fp8(
        (w13_fp8, s13_int),
        (w2_fp8.contiguous(), s2_int),
    )
    del w13_fp8, s13_int, w2_fp8, s2_int
    torch.cuda.empty_cache()

    return (l1_w, l1_sf), (l2_w, l2_sf)
