"""FP8 (MXFP8) Mixture-of-Experts masked kernel using FlashInfer's CuteDSL backend.

The kernel mirrors `flashinfer_cutedsl_moe_masked` in the fp4_kernel package but
uses FP8 (E4M3FN) for both activations and weights, with UE8M0 (Float8E8M0FNU)
block scales. The flashinfer `grouped_gemm_nt_masked` kernel only accepts FP8
inputs when paired with sf_dtype=Float8E8M0FNU and sf_vec_size=32, so this is
strictly the "MXFP8" variant.
"""

from typing import Optional, Tuple

import torch
from flashinfer import mxfp8_quantize
from flashinfer.cute_dsl.blockscaled_gemm import grouped_gemm_nt_masked

# Layout constants required by grouped_gemm_nt_masked on Blackwell.
SF_VEC_SIZE = 32
SF_BLOCK_M = 128
SF_BLOCK_K = 4


def _compute_padded_sf_shape(m: int, k: int) -> Tuple[int, int, int]:
    """Compute the swizzled scale-factor shape for MXFP8.

    Returns (padded_m, padded_k_int32, sf_byte_count).
    """
    assert k % SF_VEC_SIZE == 0, f"k={k} must be a multiple of {SF_VEC_SIZE}"
    scale_k = k // SF_VEC_SIZE
    padded_k = (scale_k + (SF_BLOCK_K - 1)) // SF_BLOCK_K * SF_BLOCK_K
    padded_k_int32 = padded_k // 4
    padded_m = (m + (SF_BLOCK_M - 1)) // SF_BLOCK_M * SF_BLOCK_M
    return padded_m, padded_k_int32, padded_m * padded_k


def quant_mxfp8_per_expert(
    weight_bf16: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a per-expert BF16/FP16 weight tensor to MXFP8.

    Args:
        weight_bf16: Input weight tensor of shape [E, N, K] (BF16/FP16).

    Returns:
        - weight_fp8: Per-expert FP8 weights, contiguous shape [E, N, K] (FP8 E4M3FN).
        - weight_sf:  Per-expert scale tensor, shape [E, padded_n, padded_k_int32]
          stored as int32 (each int32 packs 4 UE8M0 bytes).
    """
    assert weight_bf16.ndim == 3, f"expected [E, N, K], got {weight_bf16.shape}"
    e, n, k = weight_bf16.shape
    padded_n, padded_k_int32, _ = _compute_padded_sf_shape(n, k)

    weight_fp8 = torch.empty(
        (e, n, k), dtype=torch.float8_e4m3fn, device=weight_bf16.device
    )
    weight_sf = torch.empty(
        (e, padded_n, padded_k_int32), dtype=torch.int32, device=weight_bf16.device
    )

    for i in range(e):
        w_q, w_sf = mxfp8_quantize(
            weight_bf16[i].contiguous(),
            is_sf_swizzled_layout=True,
            alignment=SF_VEC_SIZE,
        )
        weight_fp8[i].copy_(w_q.view(torch.float8_e4m3fn).view(n, k))
        # `w_sf` is a flat uint8 buffer of size padded_n * padded_k_int32 * 4 bytes.
        weight_sf[i].copy_(w_sf.view(torch.int32).view(padded_n, padded_k_int32))
    return weight_fp8, weight_sf


def quant_mxfp8_grouped_activation(
    activation: torch.Tensor,
    masked_m: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize per-expert activations to MXFP8 with masked layout.

    The flashinfer `mxfp8_quantize` op operates on a single [M, K] tensor, so we
    invoke it per-expert. `masked_m` is honoured by zero-padding the unused rows
    (the MoE kernel will skip them via its own mask).

    Args:
        activation: BF16/FP16 tensor of shape [E, M, K].
        masked_m: int32 tensor of shape [E,] giving valid token count per expert.

    Returns:
        - act_fp8: [M, K, E] in float8_e4m3fn (logical (m, k, l), physical (l, m, k)).
        - act_sf: scale tensor with logical shape (32, 4, rm, 4, rk, l) and
          physical layout (l, rm, rk, 32, 4, 4).
    """
    assert activation.ndim == 3, f"expected [E, M, K], got {activation.shape}"
    e, m, k = activation.shape
    padded_m, padded_k_int32, _ = _compute_padded_sf_shape(m, k)

    act_fp8 = torch.empty(
        (e, m, k), dtype=torch.float8_e4m3fn, device=activation.device
    )
    act_sf_int32 = torch.empty(
        (e, padded_m, padded_k_int32), dtype=torch.int32, device=activation.device
    )

    for i in range(e):
        x_q, x_sf = mxfp8_quantize(
            activation[i].contiguous(),
            is_sf_swizzled_layout=True,
            alignment=SF_VEC_SIZE,
        )
        act_fp8[i].copy_(x_q.view(torch.float8_e4m3fn).view(m, k))
        act_sf_int32[i].copy_(x_sf.view(torch.int32).view(padded_m, padded_k_int32))

    # Convert physical (l, m, k) → logical (m, k, l) order required by kernel.
    act_fp8 = act_fp8.permute(1, 2, 0)
    # Same swizzle/permute trick as the FP4 path: the byte buffer already has
    # the swizzled (l, padded_m // 128, padded_k // 4, 32, 4, 4) layout; we just
    # reshape and permute the logical view.
    act_sf = act_sf_int32.view(torch.float8_e4m3fn).view(
        e, padded_m // SF_BLOCK_M, padded_k_int32, 32, 4, 4
    )
    act_sf = act_sf.permute(3, 4, 1, 5, 2, 0)
    return act_fp8, act_sf


def reshape_mxfp8_weight(
    weight_fp8: torch.Tensor,
    weight_sf_int32: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Permute MXFP8 weight + scale tensors into the layout required by
    flashinfer's grouped_gemm_nt_masked.

    Args:
        weight_fp8: [E, N, K] in FP8 E4M3FN (contiguous).
        weight_sf_int32: [E, padded_n, padded_k_int32] in int32.

    Returns:
        - weight_view: [N, K, E] FP8 E4M3FN (logical (n, k, l)).
        - weight_sf_view: scale view with logical shape
          (32, 4, rn, 4, rk, l) and physical layout (l, rn, rk, 32, 4, 4).
    """
    e, n, k = weight_fp8.shape
    padded_n, padded_k_int32, _ = _compute_padded_sf_shape(n, k)
    assert weight_sf_int32.shape == (
        e,
        padded_n,
        padded_k_int32,
    ), f"unexpected scale shape {weight_sf_int32.shape}"

    weight_view = weight_fp8.permute(1, 2, 0)
    weight_sf_view = weight_sf_int32.view(torch.float8_e4m3fn).view(
        e, padded_n // SF_BLOCK_M, padded_k_int32, 32, 4, 4
    )
    weight_sf_view = weight_sf_view.permute(3, 4, 1, 5, 2, 0)
    return weight_view, weight_sf_view


def get_cute_dtype(input: torch.Tensor) -> str:
    if input.dtype == torch.bfloat16:
        return "bfloat16"
    elif input.dtype == torch.float16:
        return "float16"
    elif input.dtype == torch.float32:
        return "float32"
    else:
        raise ValueError(f"Unsupported cute dtype {input.dtype}")


def flashinfer_cutedsl_moe_masked_fp8(
    hidden_states: Tuple[torch.Tensor, Optional[torch.Tensor]],
    w1: torch.Tensor,
    w1_blockscale: torch.Tensor,
    w2: torch.Tensor,
    w2_blockscale: torch.Tensor,
    masked_m: torch.Tensor,
    w1_alpha: Optional[torch.Tensor] = None,
    w2_alpha: Optional[torch.Tensor] = None,
    down_sm_count: Optional[int] = None,
    down_signals: Optional[torch.Tensor] = None,
    down_start_event: Optional[torch.cuda.Event] = None,
):
    """MXFP8 masked MoE forward using FlashInfer's CuteDSL kernels.

    Args:
        hidden_states: Either of:
            * (BF16/FP16 tensor [E, M, K], None) — quantize on the fly to MXFP8.
            * (FP8 E4M3FN tensor [M, K, E], scale tensor swizzled) — already
              quantized inputs in flashinfer logical layout.
        w1: FP8 E4M3FN, contiguous shape [E, 2 * N, K] (logical (2n, k, l)).
        w1_blockscale: int32 scale tensor of shape [E, padded_2n, padded_k_int32]
            (UE8M0 bytes packed into int32).
        w2: FP8 E4M3FN, contiguous shape [E, K, N].
        w2_blockscale: int32 scale tensor of shape [E, padded_k, padded_n_int32].
        masked_m: int32 tensor [E,] of valid token counts per expert.
        w1_alpha, w2_alpha: optional float32 [E,] alpha factors applied to GEMM output.

    Returns:
        Output tensor of shape [E, M, K] (BF16).
    """
    assert (
        len(hidden_states) == 2
    ), f"hidden_states must be a tuple of length 2, got {len(hidden_states)}"
    assert w1.dtype == torch.float8_e4m3fn, f"w1 must be float8_e4m3fn, got {w1.dtype}"
    assert w2.dtype == torch.float8_e4m3fn, f"w2 must be float8_e4m3fn, got {w2.dtype}"

    n = w2.shape[-1]  # intermediate dim (after final permute it's the (n, l) axis)
    if w2.ndim == 3 and w2.shape[0] == w1.shape[0]:
        # logical [E, K, N] form: derive n from last dim.
        n = w2.shape[-1]

    if hidden_states[1] is None:
        a = hidden_states[0]
        assert a.dtype in (
            torch.bfloat16,
            torch.float16,
        ), f"BF16/FP16 input expected, got {a.dtype}"
        assert a.ndim == 3, f"expected [E, M, K], got {a.shape}"
        a_q, a_q_sf = quant_mxfp8_grouped_activation(a, masked_m)
        num_experts = a.shape[0]
        m = a.shape[1]
        k = a.shape[2]
    else:
        a_q = hidden_states[0]
        a_q_sf = hidden_states[1]
        # already in logical (m, k, l)
        m, k, num_experts = a_q.shape

    # Bring weights into the layout expected by grouped_gemm_nt_masked.
    if w1.ndim == 3 and w1.shape[0] == num_experts:
        w1_view, w1_sf_view = reshape_mxfp8_weight(w1, w1_blockscale)
    else:
        w1_view, w1_sf_view = w1, w1_blockscale
    if w2.ndim == 3 and w2.shape[0] == num_experts:
        w2_view, w2_sf_view = reshape_mxfp8_weight(w2, w2_blockscale)
    else:
        w2_view, w2_sf_view = w2, w2_blockscale

    intermediate_size = w1_view.shape[0] // 2  # 2 * intermediate_size on row dim
    assert intermediate_size * 2 == w1_view.shape[0]

    ab_dtype = "float8_e4m3fn"
    sf_dtype = "float8_e8m0fnu"
    c_dtype = "bfloat16"

    # Gemm1: (M, K, E) @ (2N, K, E) → (M, 2N, E)
    gateup_output = torch.empty(
        (num_experts, m, intermediate_size * 2),
        dtype=torch.bfloat16,
        device=a_q.device,
    )
    gateup_output = gateup_output.permute(1, 2, 0)

    gemm1_kwargs = dict(
        ab_dtype=ab_dtype,
        sf_dtype=sf_dtype,
        c_dtype=c_dtype,
        sf_vec_size=SF_VEC_SIZE,
    )
    if w1_alpha is not None:
        assert w1_alpha.dtype == torch.float32 and w1_alpha.shape == (num_experts,)
        gemm1_kwargs.update(
            dict(
                alpha=w1_alpha.view(1, 1, num_experts),
                alpha_dtype=get_cute_dtype(w1_alpha),
            )
        )

    grouped_gemm_nt_masked(
        (a_q, a_q_sf),
        (w1_view, w1_sf_view),
        gateup_output,
        masked_m,
        **gemm1_kwargs,
    )

    # SiLU + multiplicative gating, then re-quantize to MXFP8 for GEMM2.
    silu_out = torch.empty(
        (num_experts, m, intermediate_size),
        dtype=torch.bfloat16,
        device=a_q.device,
    )
    # Weight w1 is stacked by ``stack_moe_w1`` as ``concat([up, gate], dim=N)``,
    # so the gemm output along the 2N axis is ``[up_output, gate_output]``.
    # SwiGLU is ``silu(gate) * up``.
    gate_view = gateup_output.permute(2, 0, 1)  # back to (E, M, 2N)
    up, gate = gate_view[..., :intermediate_size], gate_view[..., intermediate_size:]
    silu_out.copy_(torch.nn.functional.silu(gate) * up)

    diq, diq_sf = quant_mxfp8_grouped_activation(silu_out, masked_m)

    if down_start_event is not None:
        down_start_event.record()

    # Gemm2: (M, N, E) @ (K, N, E) → (M, K, E)
    out = torch.empty((num_experts, m, k), dtype=torch.bfloat16, device=a_q.device)
    out = out.permute(1, 2, 0)

    gemm2_kwargs = dict(
        ab_dtype=ab_dtype,
        sf_dtype=sf_dtype,
        c_dtype=c_dtype,
        sf_vec_size=SF_VEC_SIZE,
    )
    if w2_alpha is not None:
        assert w2_alpha.dtype == torch.float32 and w2_alpha.shape == (num_experts,)
        gemm2_kwargs.update(
            dict(
                alpha=w2_alpha.view(1, 1, num_experts),
                alpha_dtype=get_cute_dtype(w2_alpha),
            )
        )
    if down_sm_count is not None:
        gemm2_kwargs["sm_count"] = down_sm_count
    if down_signals is not None:
        gemm2_kwargs["dst_signals"] = down_signals

    grouped_gemm_nt_masked(
        (diq, diq_sf),
        (w2_view, w2_sf_view),
        out,
        masked_m,
        **gemm2_kwargs,
    )
    return out.permute(2, 0, 1)
