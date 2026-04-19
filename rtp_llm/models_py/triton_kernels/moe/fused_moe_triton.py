# Adapt from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py
# Adapted for RTP-LLM. Only the high-perf Triton fused-MoE path (no DeepEP) is
# kept; DeepEP/TMA/swap_ab/Marlin/GPTQ-AWQ specific code paths are removed.
# Licensed under the Apache License, Version 2.0
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import triton.language as tl

from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
    scaled_fp8_per_tensor_quant,
    scaled_fp8_per_token_quant,
    sgl_per_token_group_quant_fp8,
)

from .fused_moe_triton_config import get_config_dtype_str, try_get_optimal_moe_config
from .fused_moe_triton_kernels import (
    act_and_mul_triton,
    invoke_fused_moe_kernel,
    moe_align_block_size,
    moe_sum_reduce_triton,
)


# Small-token reduce path observed in sglang's MTP profiling timeline.
# torch.compile fuses sum + mul into a single kernel ``triton_per_fused_copy__mul_sum_0``
# that beats the dedicated triton reduce when num_tokens <= 32 (which is
# essentially always true for MTP/decode). We deliberately avoid
# ``@torch.compile`` here because Dynamo retracing is not permitted while a
# CUDA graph stream is capturing. A plain eager torch implementation matches
# the same op pattern and is graph-capture safe.
def _moe_sum_reduce_torch_compile(x, out, routed_scaling_factor):
    torch.sum(x, dim=1, out=out)
    out.mul_(routed_scaling_factor)


def _quantize_input_fp8(
    A: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    block_shape: Optional[List[int]],
    per_channel_quant: bool,
):
    """FP8 W8A8 activation quantization (reuses RTP-LLM's existing wrappers)."""
    if block_shape is None:
        if per_channel_quant:
            return scaled_fp8_per_token_quant(A, A_scale)
        return scaled_fp8_per_tensor_quant(A, A_scale)
    block_n, block_k = block_shape[0], block_shape[1]
    return sgl_per_token_group_quant_fp8(A, block_k)


def fused_experts_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    inplace: bool = False,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    per_channel_quant: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
    routed_scaling_factor: Optional[float] = None,
    filter_expert: bool = True,
    no_combine: bool = False,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Triton fused MoE forward.

    Mirrors sglang's ``fused_experts_impl`` for the subset of features RTP-LLM
    needs today: BF16/FP16 no-quant and FP8 W8A8 (per-tensor / per-token /
    per-block). See module docstring for the dropped variants.
    """
    assert hidden_states.is_contiguous()
    assert w1.is_contiguous()
    assert w2.is_contiguous()
    assert hidden_states.shape[1] == w1.shape[2], "Hidden size mismatch"
    assert topk_weights.shape == topk_ids.shape

    num_tokens = hidden_states.shape[0]
    E, N, _ = w1.shape
    topk = topk_ids.shape[1]
    # ``hidden_states.dtype`` may be FP8 when the router pre-quantized the
    # input, so it cannot be used as the compute / intermediate dtype. The
    # caller passes ``out_dtype`` (typically ``payload.expert_x_origin_dtype``)
    # which holds the original BF16/FP16 model dtype.
    effective_dtype = out_dtype if out_dtype is not None else hidden_states.dtype
    if effective_dtype == torch.float8_e4m3fn:
        effective_dtype = torch.bfloat16
    compute_type = tl.bfloat16 if effective_dtype == torch.bfloat16 else tl.float16

    config_dtype = get_config_dtype_str(
        use_fp8_w8a8=use_fp8_w8a8, dtype=effective_dtype
    )
    config = try_get_optimal_moe_config(
        w1.shape,
        (w2.shape[0], w2.shape[1], w2.shape[2]),
        topk,
        config_dtype,
        num_tokens,
        block_shape=block_shape,
    )

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, config["BLOCK_SIZE_M"], E
    )

    if no_combine:
        assert not inplace
        out_hidden_states = torch.empty(
            (num_tokens, topk, w2.shape[1]),
            device=hidden_states.device,
            dtype=effective_dtype,
        )
    elif inplace:
        out_hidden_states = hidden_states
    else:
        out_hidden_states = torch.empty(
            (num_tokens, w2.shape[1]),
            device=hidden_states.device,
            dtype=effective_dtype,
        )

    # NOTE: must be ``zeros`` (not ``empty``). Rows belonging to filtered
    # (-1 expert) tokens are never written by ``fused_moe_kernel`` /
    # ``act_and_mul_kernel`` (they early-return). The downstream
    # ``_quantize_input_fp8`` of ``intermediate_cache2`` reduces ``max(|row|)``
    # over those rows; uninitialized memory containing NaN/Inf would
    # propagate into the per-token scales and corrupt the second GEMM.
    intermediate_cache1 = torch.zeros(
        (num_tokens * topk, N),
        device=hidden_states.device,
        dtype=effective_dtype,
    )
    intermediate_cache2 = torch.zeros(
        (num_tokens * topk, N // 2),
        device=hidden_states.device,
        dtype=effective_dtype,
    )
    intermediate_cache3 = torch.zeros(
        (num_tokens, topk, w2.shape[1]),
        device=hidden_states.device,
        dtype=effective_dtype,
    )

    if use_fp8_w8a8:
        assert w1_scale is not None
        assert w2_scale is not None
        if hidden_states.dtype == torch.float8_e4m3fn:
            # Router pre-quantized: reuse the provided (a1_q, a1_scale) and
            # skip the redundant per_token_group_quant_8bit dispatch which
            # only accepts BF16/FP16 inputs.
            assert (
                a1_scale is not None
            ), "fused_experts_impl: hidden_states already FP8 but a1_scale is None"
            a1_q, a1_s = hidden_states, a1_scale
        else:
            a1_q, a1_s = _quantize_input_fp8(
                hidden_states, a1_scale, block_shape, per_channel_quant
            )
    else:
        a1_q, a1_s = hidden_states, None

    invoke_fused_moe_kernel(
        a1_q,
        w1,
        None,
        intermediate_cache1,
        a1_s,
        w1_scale,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        apply_router_weight_on_input,
        topk,
        config,
        compute_type=compute_type,
        use_fp8_w8a8=use_fp8_w8a8,
        per_channel_quant=per_channel_quant,
        block_shape=block_shape,
        filter_expert=filter_expert,
    )

    # Activation. When no per-rank expert filtering is needed we can fall back
    # to the fast C++/flashinfer-style ``silu_and_mul`` (matches sglang MTP's
    # trace which shows ``flashinfer::act_and_mul_kernel`` here). Otherwise we
    # use the Triton variant that honors ``-1`` filter sentinels in topk_ids.
    if activation == "silu" and not filter_expert:
        from rtp_llm.models_py.triton_kernels.common.activation import (
            silu_and_mul as _silu_and_mul,
        )

        _silu_and_mul(intermediate_cache2, intermediate_cache1.view(-1, N))
    elif activation in ("silu", "gelu"):
        act_and_mul_triton(
            intermediate_cache1.view(-1, N),
            intermediate_cache2,
            topk_ids=topk_ids,
            activation=activation,
        )
    else:
        raise ValueError(f"Unsupported activation: {activation}")

    if use_fp8_w8a8:
        a2_q, a2_s = _quantize_input_fp8(
            intermediate_cache2, a2_scale, block_shape, per_channel_quant
        )
    else:
        a2_q, a2_s = intermediate_cache2, None

    invoke_fused_moe_kernel(
        a2_q,
        w2,
        None,
        intermediate_cache3,
        a2_s,
        w2_scale,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        not apply_router_weight_on_input,
        1,
        config,
        compute_type=compute_type,
        use_fp8_w8a8=use_fp8_w8a8,
        per_channel_quant=per_channel_quant,
        block_shape=block_shape,
        filter_expert=filter_expert,
    )

    if no_combine:
        return intermediate_cache3

    if routed_scaling_factor is None:
        routed_scaling_factor = 1.0

    if topk == 1 and routed_scaling_factor == 1.0:
        out_hidden_states.copy_(intermediate_cache3.squeeze(1))
    elif topk == 2 and routed_scaling_factor == 1.0:
        torch.add(
            intermediate_cache3[:, 0],
            intermediate_cache3[:, 1],
            out=out_hidden_states,
        )
    elif num_tokens <= 32:
        # Small-token path used by sglang MTP (see _moe_sum_reduce_torch_compile).
        _moe_sum_reduce_torch_compile(
            intermediate_cache3, out_hidden_states, routed_scaling_factor
        )
    else:
        moe_sum_reduce_triton(
            intermediate_cache3, out_hidden_states, routed_scaling_factor
        )

    return out_hidden_states
