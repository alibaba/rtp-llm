"""Weight layout preparation shared by MegaMoE executor recipes."""

from __future__ import annotations

import logging

import torch

from rtp_llm.models_py.kernels.cuda.quant_layouts import (
    FP4_BLOCK,
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
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """Prepare weights from FP8 per-block format (GLM-5 checkpoint).

    Converts FP8 → BF16 → FP4 for the mega kernel.
    w1 = gate, w2 = down, w3 = up (following DeepSeek/GLM convention).
    """
    import deep_gemm
    from deep_gemm.utils import per_token_cast_to_fp4

    E = cfg.n_local_experts
    D = cfg.dim
    inter = cfg.moe_inter_dim
    device = w1_fp8.device
    fp8_block = 128

    logger.info(
        "[GLM5 MegaMoE] Converting FP8 weights to FP4: layer=%d E=%d D=%d inter=%d",
        cfg.layer_id,
        E,
        D,
        inter,
    )

    # --- L1 (gate + up): dequant FP8 → BF16 → FP4 ---
    # Gate: [E, inter, D] and Up: [E, inter, D] → combined [E, 2*inter, D]
    def _dequant_fp8(w_fp8, w_scale):
        """Dequantize FP8 per-block to BF16."""
        shape = w_fp8.shape  # [E, N, K]
        E_, N_, K_ = shape
        if K_ % fp8_block != 0:
            raise ValueError(
                f"FP8 MoE K dimension must be divisible by {fp8_block}, "
                f"got weight shape={tuple(shape)}"
            )
        if not torch.is_floating_point(w_scale):
            raise ValueError(
                "MegaMoE FP8 fallback requires raw floating-point FP8 "
                "checkpoint scales. Got packed/transformed scale "
                f"dtype={w_scale.dtype}, shape={tuple(w_scale.shape)}. "
                "For moe_strategy=mega_moe, MoE weights should be wrapped "
                "to FP4 at load time before executor initialization."
            )
        n_blocks_k = K_ // fp8_block
        n_blocks_n = (N_ + fp8_block - 1) // fp8_block
        if w_scale.shape == (E_, N_, n_blocks_k):
            scale_per_row = w_scale
        elif w_scale.shape == (E_, n_blocks_n, n_blocks_k):
            scale_per_row = w_scale.repeat_interleave(fp8_block, dim=1)[:, :N_, :]
        elif w_scale.numel() == E_ * N_ * n_blocks_k:
            scale_per_row = w_scale.reshape(E_, N_, n_blocks_k)
        elif w_scale.numel() == E_ * n_blocks_n * n_blocks_k:
            scale_per_block = w_scale.reshape(E_, n_blocks_n, n_blocks_k)
            scale_per_row = scale_per_block.repeat_interleave(fp8_block, dim=1)[
                :, :N_, :
            ]
        else:
            raise ValueError(
                "Cannot interpret FP8 MoE scale shape "
                f"{tuple(w_scale.shape)} for weight shape={tuple(shape)}"
            )
        w_f = w_fp8.float().view(E_, N_, n_blocks_k, fp8_block)
        s_exp = scale_per_row.to(dtype=w_f.dtype).unsqueeze(-1).expand_as(w_f)
        return (w_f * s_exp).reshape(E_, N_, K_).to(torch.bfloat16)

    w1_bf16 = _dequant_fp8(w1_fp8, w1_scale)  # [E, inter, D]
    w3_bf16 = _dequant_fp8(w3_fp8, w3_scale)  # [E, inter, D]
    del w1_fp8, w1_scale, w3_fp8, w3_scale

    # Combine gate + up into [E, 2*inter, D]
    w13_bf16 = torch.cat([w1_bf16, w3_bf16], dim=1)  # [E, 2*inter, D]
    del w1_bf16, w3_bf16
    torch.cuda.empty_cache()

    # Quantize L1 to FP4: [E, 2*inter, D] → packed [E, 2*inter, D//2] + sf
    w13_packed = torch.empty((E, 2 * inter, D // 2), dtype=torch.int8, device=device)
    s13_raw = torch.empty(
        (E, 2 * inter, D // FP4_BLOCK), dtype=torch.float, device=device
    )
    for i in range(E):
        w13_packed[i], s13_raw[i] = per_token_cast_to_fp4(
            w13_bf16[i], use_ue8m0=True, gran_k=FP4_BLOCK
        )
    del w13_bf16

    s13_int = prepare_fp4_weight_scale_for_deepgemm(s13_raw, 2 * inter, D, E)
    del s13_raw
    torch.cuda.empty_cache()

    # --- L2 (down): dequant FP8 → BF16 → FP4 ---
    w2_bf16 = _dequant_fp8(w2_fp8, w2_scale)  # [E, D, inter]
    del w2_fp8, w2_scale

    w2_packed = torch.empty((E, D, inter // 2), dtype=torch.int8, device=device)
    s2_raw = torch.empty((E, D, inter // FP4_BLOCK), dtype=torch.float, device=device)
    for i in range(E):
        w2_packed[i], s2_raw[i] = per_token_cast_to_fp4(
            w2_bf16[i], use_ue8m0=True, gran_k=FP4_BLOCK
        )
    del w2_bf16

    s2_int = prepare_fp4_weight_scale_for_deepgemm(s2_raw, D, inter, E)
    del s2_raw
    torch.cuda.empty_cache()

    # Apply mega MoE transform
    (l1_w, l1_sf), (l2_w, l2_sf) = deep_gemm.transform_weights_for_mega_moe(
        (w13_packed, s13_int),
        (w2_packed, s2_int),
    )
    del w13_packed, s13_int, w2_packed, s2_int
    torch.cuda.empty_cache()

    return (l1_w, l1_sf), (l2_w, l2_sf)


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
