"""PPU MoE for DeepSeek V4.1 Flash: split W4A8, with a W4A4 override.

Implements the routed-expert half of the V4.1 backend contract
(``rtp_llm/models_py/modules/dsv41/backend.py``, ``moe`` op) for the PPU
strategy ``bf16_dense_w4a16_moe_sparse_mla_host_enp_v1``:

  * ``sqrtsoftplus_gate`` — routing gate matching rtp-llm
    ``models_py/modules/dsv41/moe/gate.py`` semantics (bf16 GEMM -> fp32
    epilogue, ``scores = sqrt(softplus(logits))``, ``router_bias`` used ONLY
    for ranking, weights gathered from UN-biased scores,
    ``weights /= sum + 1e-12`` then ``* routed_scaling_factor`` (1.5),
    non-finite rows -> uniform ``route_scale / topk``), plus a fused Triton
    fast path adapted from sglang-for-sail.
  * ``pad_topk_for_dispatch`` — 8-slot ``-1``/``0.0`` padding replicating rtp
    ``moe/strategies/deepep.py::_pad_topk_for_deepep``.
  * ``PPUMoE`` — DeepEP normal dispatch/combine and two DeepGEMM-for-sail
    grouped GEMMs. The default W4A8 path uses per-row E4M3 activations and
    MXFP4 weights. ``RTP_V41_PPU_MOE_PRECISION=w4a4`` selects the original
    MXFP4 activation path. CPU/diagnostic execution uses a dequant reference.
  * ``shared_expert_forward`` / ``combine_routed_and_shared`` — BF16-resident
    shared expert with the clamp-10 SwiGLU convention.

Top-k weight application (IMPORTANT):
  rtp-llm's ``DeepEPStrategy`` passes ``weights_p`` through
  ``buffer.dispatch(...)`` and applies the received ``recv_topk_weights``
  inside the local per-expert compute; ``buffer.combine`` then only sums
  across ranks and does NOT re-scale (sglang-for-sail likewise applies
  weights in ``ep_gather`` after the grouped GEMM, not in combine). This
  W4A8 instead follows DeepInfer's weight placement: FP32 SwiGLU -> multiply
  the per-route weight -> FP8 quantization -> GEMM2. Its gather uses unit
  weights, and combine only sums. The W4A4 override retains the old order:
    * W4A4 grouped-GEMM path: weights applied by ``ep_gather``.
    * DeepEP + reference path / LOCAL (no dispatch) path: weights multiplied
      explicitly in the per-expert loop.
  If a future PPU ``deep_ep`` build makes ``combine`` scale by the weights
  handed to ``dispatch``, the DeepEP paths here would double-apply weights —
  validate on hardware (see also DeepEPWrapper's dispatch call signature).

Import discipline: top-level imports are stdlib + torch + the pure-python
``rtp_llm.models_py.modules.dsv41.ppu.linear`` module only. ``triton``,
``deep_gemm`` and ``deep_ep`` are imported lazily inside functions so this
module imports cleanly on a CPU host with no accelerator wheels.
"""

from __future__ import annotations

import importlib.util
import os
import threading
import weakref
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.dsv41.ppu.linear import (
    PPUResidentBF16Linear,
    _ue8m0_scales_to_float,
)
from rtp_llm.models_py.modules.dsv41.ppu.moe_precision import (
    moe_backend,
    moe_kernel_name,
    moe_precision,
)

__all__ = [
    "sqrtsoftplus_gate",
    "pad_topk_for_dispatch",
    "PPUMoE",
    "expert_forward_reference",
    "dequant_fp4_weight",
    "shared_expert_forward",
    "combine_routed_and_shared",
    "preprocess_mxfp4_f4f4_scales",
    "downcast_to_mxfp4",
    "COMM_SLOTS",
    "SWIGLU_LIMIT",
    "DEFAULT_TOP_K",
    "ROUTED_SCALING_FACTOR",
]

# V4.1 communication slots (top-6 padded to 8 with -1/0.0) and SwiGLU limit.
COMM_SLOTS = 8
SWIGLU_LIMIT = 10.0
DEFAULT_TOP_K = 6
ROUTED_SCALING_FACTOR = 1.5

_FP4_GROUP = 32  # UE8M0 scale group along K for packed FP4 e2m1 weights

# MXFP4 activation-quantization group (sglang-for-sail ppu_mxfp4_utils.py
# MXFP_BLOCK_SIZE) — same 32, kept separate for documentation clarity.
_MXFP_BLOCK_SIZE = 32

# FP4 e2m1 lookup: 4-bit raw -> fp32 value (same table as rtp
# models_py/modules/dsv41/qlinear.py::_FP4_LUT).
_FP4_E2M1_LUT = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)

# DeepEP normal-mode supported topk widths (rtp deepep.py:_DEEPEP_SUPPORTED_TOPK).
_DEEPEP_SUPPORTED_TOPK = (2, 4, 8, 16)

# Failed module imports are not cached by Python, and importing the rtp
# DeepEPWrapper pulls a heavy config chain that raises on hosts without the
# rtp C++ ops — remember the failure so the probe runs at most once.
_DEEPEP_WRAPPER_IMPORT_FAILED = False


def _deepep_wrapper_initialized() -> bool:
    global _DEEPEP_WRAPPER_IMPORT_FAILED
    if _DEEPEP_WRAPPER_IMPORT_FAILED:
        return False
    try:
        from rtp_llm.models_py.distributed.deepep_wrapper import DeepEPWrapper
    except Exception:
        _DEEPEP_WRAPPER_IMPORT_FAILED = True
        return False
    try:
        return DeepEPWrapper.is_initialized()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# FP4 e2m1 reference dequantization
# ---------------------------------------------------------------------------


def dequant_fp4_weight(
    packed_weight: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """Dequantize one expert's packed FP4 e2m1 weight to BF16.

    Args:
        packed_weight: ``[N, K/2]`` int8/uint8, two e2m1 nibbles per byte —
            the LOW nibble is the even ``k`` (k = 2*j), the HIGH nibble the
            odd ``k`` (k = 2*j+1), matching rtp ``qlinear.py``.
        scale: ``[N, K/32]`` UE8M0 scales (``float8_e8m0fnu``, raw ``uint8``
            exponent bytes with value ``2 ** (byte - 127)``, or float).

    Returns:
        ``[N, K]`` BF16 weight: ``LUT[nibble] * 2 ** (scale_byte - 127)``.
    """
    if packed_weight.dim() != 2:
        raise ValueError(
            f"packed FP4 weight must be 2D [N, K/2], got {tuple(packed_weight.shape)}"
        )
    n, packed_k = packed_weight.shape
    k = packed_k * 2
    scale_values = _ue8m0_scales_to_float(scale)
    if tuple(scale_values.shape) != (n, (k + _FP4_GROUP - 1) // _FP4_GROUP):
        raise ValueError(
            f"FP4 scale shape {tuple(scale_values.shape)} does not match weight "
            f"{(n, k)}: expected {(n, (k + _FP4_GROUP - 1) // _FP4_GROUP)}"
        )
    if packed_weight.dtype not in (torch.int8, torch.uint8):
        raise TypeError(
            f"packed FP4 weight must be int8/uint8, got {packed_weight.dtype}"
        )
    # int8 -> uint8 wraps negative bytes (packed nibbles are raw bytes).
    w_uint = packed_weight.to(torch.uint8)
    low = (w_uint & 0x0F).long()
    high = (w_uint >> 4).long()
    lut = _FP4_E2M1_LUT.to(packed_weight.device)
    vals = torch.empty(n, k, dtype=torch.float32, device=packed_weight.device)
    vals[:, 0::2] = lut[low]
    vals[:, 1::2] = lut[high]
    scale_full = scale_values.repeat_interleave(_FP4_GROUP, dim=-1)[:, :k]
    return (vals * scale_full).to(torch.bfloat16)


def expert_forward_reference(
    x: torch.Tensor,
    w13_weight: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_scale: torch.Tensor,
    swiglu_limit: float = SWIGLU_LIMIT,
) -> torch.Tensor:
    """Single-expert SwiGLU forward on FP4 weights dequantized to BF16.

    Diagnostic / test oracle (and the CPU fallback inside ``PPUMoE``): only
    the experts actually hit by tokens are ever dequantized — never all 384.

    ``w13`` packs [gate; up] rows (``[2I, K/2]`` packed, ``[2I, K/32]`` scale);
    ``w2`` is ``[D, I/2]`` packed with ``[D, I/32]`` scale. The SwiGLU clamp
    matches sglang-for-sail ``_apply_swiglu_limit``: ``up`` clamped to
    ``[-limit, limit]`` and ``gate`` to ``[-inf, limit]`` BEFORE the SiLU.
    """
    w13 = dequant_fp4_weight(w13_weight, w13_scale)
    w2 = dequant_fp4_weight(w2_weight, w2_scale)
    if x.dtype != torch.bfloat16:
        x = x.to(torch.bfloat16)
    gate_up = F.linear(x, w13)
    gate, up = gate_up.chunk(2, dim=-1)
    if swiglu_limit and swiglu_limit > 0:
        up = up.clamp(min=-float(swiglu_limit), max=float(swiglu_limit))
        gate = gate.clamp(max=float(swiglu_limit))
    hidden = F.silu(gate.float()) * up.float()
    return F.linear(hidden.to(torch.bfloat16), w2)


# ---------------------------------------------------------------------------
# Routing gate
# ---------------------------------------------------------------------------


def sqrtsoftplus_gate(
    hidden: torch.Tensor,
    router_weight: torch.Tensor,
    router_bias: Optional[torch.Tensor],
    *,
    top_k: int = DEFAULT_TOP_K,
    routed_scaling_factor: float = ROUTED_SCALING_FACTOR,
    image_rows: Optional[torch.Tensor] = None,
    graph_routes: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    allow_top3_graph_routes: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """DeepSeek-V4.1 sqrtsoftplus routing gate (torch reference).

    Exactly matches ``rtp_llm/models_py/modules/dsv41/moe/gate.py`` for the
    non-hash path:

      1. BF16 inputs with FP32 projection output; no intermediate BF16
         rounding before the ranking.
      2. ``original = sqrt(softplus(scores))``.
      3. ``router_bias`` is added for RANKING ONLY; weights are gathered from
         the un-biased ``original`` scores.
      4. ``weights = original[ids] / (sum + 1e-12) * routed_scaling_factor``.
      5. Rows whose (biased) scores contain non-finite values fall back to
         uniform ``routed_scaling_factor / top_k`` so NaN/Inf never reach the
         dispatch kernels.

    ``image_rows`` is an optional boolean ``[T]`` mask. rtp ``gate.py`` only
    diverges for image tokens when a separate ``bias_vl`` exists; with no VL
    bias supplied here, image rows use the same router bias as text rows (the
    argument is a forward-compatibility hook and currently has no effect).

    Returns:
        ``(ids int64 [T, top_k], weights float32 [T, top_k])``. Empty ``T``
        returns correctly-shaped empty tensors. Ordinary Decode Graph can pass
        ``graph_routes=(valid_mask, routed_row_count)`` to get dispatch-ready
        slots and write the number of rows with live routes.

    PPU fast path: when triton is importable and the logits live on an
    accelerator device, the fused sglang-for-sail router kernel runs instead
    (see :func:`_fused_sqrtsoftplus_gate_triton`). The torch path stays
    authoritative; known fused-path deviations are documented there.
    """
    t = hidden.size(0)
    device = hidden.device
    if t == 0:
        slots = top_k
        if graph_routes is not None:
            slots = max(slots, COMM_SLOTS)
            graph_routes[1].zero_()
        return (
            torch.zeros((0, slots), dtype=torch.int64, device=device),
            torch.zeros((0, slots), dtype=torch.float32, device=device),
        )
    x_bf16 = hidden if hidden.dtype == torch.bfloat16 else hidden.to(torch.bfloat16)
    w_bf16 = (
        router_weight
        if router_weight.dtype == torch.bfloat16
        else router_weight.to(torch.bfloat16)
    )
    # Casting a BF16 GEMM result afterward loses score differences that can
    # change top-k experts. PPU torch.mm supports an FP32 output directly.
    if x_bf16.is_cuda:
        logits = torch.mm(x_bf16, w_bf16.t(), out_dtype=torch.float32)
    else:
        logits = F.linear(x_bf16.float(), w_bf16.float())
    if (
        router_bias is not None
        and image_rows is None
        and _fused_gate_available(logits, router_bias)
    ):
        direct_routes = (
            graph_routes
            if graph_routes is not None
            and _can_write_graph_routes(
                logits, top_k, graph_routes, allow_top3=allow_top3_graph_routes
            )
            else None
        )
        weights, ids = _fused_sqrtsoftplus_gate_triton(
            logits.contiguous(),
            router_bias.contiguous(),
            topk=top_k,
            routed_scaling_factor=float(routed_scaling_factor),
            graph_routes=direct_routes,
        )
        if direct_routes is not None:
            return ids, weights
    else:
        original = F.softplus(logits).sqrt()
        ranking = (
            original + router_bias.to(torch.float32)
            if router_bias is not None else original
        )
        weights, ids = _select_routes_with_nonfinite_fallback(
            original, ranking, top_k, float(routed_scaling_factor)
        )
    ids = ids.to(torch.int64)
    if graph_routes is not None:
        valid_mask, routed_rows = graph_routes
        ids, weights = pad_topk_for_dispatch(ids, weights)
        ids = torch.where(valid_mask[:, None], ids, -1)
        weights = torch.where(valid_mask[:, None], weights, 0)
        routed_rows.copy_((ids >= 0).any(-1).sum())
    return ids, weights


def _select_routes_with_nonfinite_fallback(
    original_scores: torch.Tensor,
    ranking_scores: torch.Tensor,
    topk: int,
    route_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Route selection that never leaks non-finite weights downstream.

    Port of ``rtp_llm/models_py/modules/dsv41/moe/gate.py::
    _select_routes_with_nonfinite_fallback`` (weights normalized from
    un-biased scores, biased scores drive the top-k; bad rows get uniform
    ``route_scale / topk`` weights).
    """
    row_is_finite = torch.isfinite(ranking_scores).all(dim=-1)
    safe_ranking_scores = torch.nan_to_num(
        ranking_scores,
        nan=-float("inf"),
        posinf=-float("inf"),
        neginf=-float("inf"),
    )
    indices = safe_ranking_scores.topk(topk, dim=-1)[1]
    weights = original_scores.gather(1, indices)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-12)
    weights = weights * route_scale
    fallback_weight = float(route_scale) / float(topk)
    weights = torch.where(
        row_is_finite.unsqueeze(-1),
        weights,
        torch.full_like(weights, fallback_weight),
    )
    return weights, indices


def _fused_gate_available(logits: torch.Tensor, bias: torch.Tensor) -> bool:
    """Device / triton availability check for the fused router gate."""
    if os.environ.get("RTP_V41_PPU_GATE_TRITON", "1") == "0":
        return False
    if logits.size(0) == 0:
        return False
    if logits.device.type != "cuda":  # PPU torch builds surface as cuda
        return False
    if logits.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        return False
    if bias.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        return False
    if logits.size(1) != bias.size(0):
        return False
    return importlib.util.find_spec("triton") is not None


def _can_write_graph_routes(logits, top_k, graph_routes, *, allow_top3=False) -> bool:
    valid, count = graph_routes
    return (
        (top_k == DEFAULT_TOP_K or (allow_top3 and top_k == 3))
        and valid.shape == (logits.shape[0],)
        and valid.dtype == torch.bool
        and valid.is_contiguous()
        and count.ndim == 0
        and count.dtype == torch.int64
        and valid.device == count.device == logits.device
    )


def _fused_sqrtsoftplus_gate_triton(
    logits: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    routed_scaling_factor: float,
    *,
    graph_routes: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused Triton sqrtsoftplus router gate (PPU fast path).

    Kernel adapted from sglang-for-sail
    ``python/sglang/kernels/ops/moe/moe_fused_gate.py::_router_triton_kernel``
    (512b03d, Apache-2.0) — SCORING_FUNC=1 sqrtsoftplus with the
    ``sqrtsoftplus_log1p`` numerics, bias for ranking only, renormalize +
    routed scaling. CUDA-only branches (PDL, token-bias alt, padded rows,
    packed id output) and the V3 grouped-routing branch were stripped; V4.1
    is ungrouped.

    Numerics vs the torch reference: identical on finite inputs (verified in
    tests) except
      * renormalize epsilon is 1e-12 here (rtp semantics) instead of
        sglang's 1e-20;
      * NaN handling: the kernel maps NaN scores to -1e30 (or ranks them
        first under ``sqrtsoftplus_log1p``), while rtp replaces the whole row
        with uniform ``route_scale / topk`` — the torch path is
        authoritative for non-finite rows;
      * tie-breaking picks the lowest expert id, matching torch.topk on the
        platforms tested.
    """
    kernels = _get_triton_kernels()
    triton = kernels.triton
    m, n = logits.shape
    k = COMM_SLOTS if graph_routes is not None else topk
    weights = torch.empty((m, k), dtype=torch.float32, device=logits.device)
    indices = torch.empty(
        (m, k),
        dtype=torch.int64 if graph_routes is not None else torch.int32,
        device=logits.device,
    )
    if m == 0:
        return weights, indices
    block_n = triton.next_power_of_2(n)
    block_k = triton.next_power_of_2(topk)
    block_m = max(1, min(4, 256 // block_n))
    num_warps = 1 if block_n <= 512 else 4
    grid = (triton.cdiv(m, block_m),)
    valid_mask, routed_rows = graph_routes if graph_routes is not None else (None, None)
    kernels.router_gate[grid](
        logits,
        bias,
        weights,
        indices,
        valid_mask,
        routed_rows,
        m,
        float(routed_scaling_factor),
        logits.stride(0),
        logits.stride(1),
        bias.stride(0),
        weights.stride(0),
        weights.stride(1),
        indices.stride(0),
        indices.stride(1),
        N=n,
        K=k,
        K_ROUTED=topk,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        SCORING_FUNC=1,
        SQRTSOFTPLUS_LOG1P=True,
        RENORMALIZE=True,
        RENORMALIZE_EPSILON=1e-12,
        APPLY_SCALE=True,
        HAS_BIAS=True,
        GRAPH_ROWS=triton.next_power_of_2(m) if graph_routes is not None else 0,
        num_warps=num_warps,
    )
    return weights, indices


# ---------------------------------------------------------------------------
# Dispatch padding
# ---------------------------------------------------------------------------


def pad_topk_for_dispatch(
    ids: torch.Tensor,
    weights: torch.Tensor,
    slots: int = COMM_SLOTS,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad ``(ids, weights)`` to ``slots`` width with ``-1`` / ``0.0``.

    Replicates ``rtp_llm/models_py/modules/dsv41/moe/strategies/deepep.py::
    _pad_topk_for_deepep``: the communication kernel only accepts topk widths
    in ``{2, 4, 8, 16}``, and V4.1's top-6 is padded to 8. Padding slots are
    silently dropped downstream (``-1`` never matches an expert id, and the
    zero weight contributes nothing).
    """
    n_act = ids.size(-1)
    if n_act >= slots:
        return ids, weights
    n = ids.size(0)
    pad_n = slots - n_act
    pad_idx = torch.full((n, pad_n), -1, dtype=ids.dtype, device=ids.device)
    pad_w = torch.zeros((n, pad_n), dtype=weights.dtype, device=weights.device)
    return (
        torch.cat([ids, pad_idx], dim=-1),
        torch.cat([weights, pad_w], dim=-1),
    )


# ---------------------------------------------------------------------------
# Triton kernels (lazy: module imports fine without triton installed)
# ---------------------------------------------------------------------------

_TRITON_LOCK = threading.Lock()
_TRITON_KERNELS = None


def _get_triton_kernels():
    """JIT-define the copied sglang-for-sail Triton kernels on first use.

    The kernels below are adapted from sglang-for-sail (commit 512b03d,
    Apache-2.0):
      * router gate — ``kernels/ops/moe/moe_fused_gate.py::_router_triton_kernel``
      * scatter — ``kernels/ops/moe/ep_moe_kernels.py::
        (_fwd_kernel_ep_scatter_1, _fwd_kernel_ep_scatter_2_optimal,
        ep_scatter_sail)``
      * gather — ``kernels/ops/moe/ep_moe_kernels.py::
        (_fwd_kernel_ep_gather, ep_gather)``
      * MXFP4 activation downcast — ``srt/layers/quantization/
        ppu_mxfp4_utils.py::_downcast_to_mxfp4``
    CUDA-only / unused branches were stripped (PDL, token-bias alt, padded
    rows, packed id output, FP8/uint16 activation scales in the scatter,
    MUSA relaxed atomics).
    """
    global _TRITON_KERNELS
    with _TRITON_LOCK:
        if _TRITON_KERNELS is not None:
            return _TRITON_KERNELS
        import triton
        import triton.language as tl
        from triton.language.extra import libdevice

        @triton.jit
        def _accumulate_mega_expert_load_kernel(
            counts_ptr,
            current_ptr,
            previous_ptr,
            EXPERTS: tl.constexpr,
            BLOCK: tl.constexpr,
        ):
            expert = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            live = expert < EXPERTS
            previous = tl.load(previous_ptr + expert, live, 0)
            current = tl.load(current_ptr + expert, live, 0)
            counts = tl.load(counts_ptr + expert, live, 0)
            tl.store(counts_ptr + expert, counts + (current - previous), live)

        # Adapted from sglang-for-sail kernels/ops/moe/moe_fused_gate.py
        # (512b03d), Apache-2.0: _router_triton_kernel.
        @triton.jit
        def _router_gate_kernel(
            scores_ptr,  # [M, N] raw logits, fp32/fp16/bf16 (fp32 on load)
            bias_ptr,  # [N] fp32/fp16/bf16 (fp32 on load)
            out_weights_ptr,  # [M, K] fp32
            out_indices_ptr,  # [M, K] int32, or int64 for dispatch-ready output
            valid_mask_ptr,
            routed_rows_ptr,
            M,
            routed_scaling_factor,
            stride_sm,
            stride_sn,
            stride_bias,
            stride_wm,
            stride_wk,
            stride_im,
            stride_ik,
            N: tl.constexpr,
            K: tl.constexpr,
            K_ROUTED: tl.constexpr,
            BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr,  # >= N, power of 2
            BLOCK_K: tl.constexpr,  # >= K, power of 2
            SCORING_FUNC: tl.constexpr,  # 0 = sigmoid, 1 = sqrtsoftplus, 2 = softmax
            SQRTSOFTPLUS_LOG1P: tl.constexpr,
            RENORMALIZE: tl.constexpr,
            APPLY_SCALE: tl.constexpr,
            HAS_BIAS: tl.constexpr,
            RENORMALIZE_EPSILON: tl.constexpr,
            GRAPH_ROWS: tl.constexpr,
        ) -> None:
            pid = tl.program_id(0)
            offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
            offs_n = tl.arange(0, BLOCK_N)  # [BLOCK_N]
            mask_m = offs_m < M
            mask_n = offs_n < N

            if HAS_BIAS:
                bias = tl.load(
                    bias_ptr + offs_n * stride_bias, mask=mask_n, other=0.0
                ).to(tl.float32)
            else:
                bias = tl.zeros([BLOCK_N], dtype=tl.float32)
            row_bias = bias[None, :]

            row_ptr = (
                scores_ptr + offs_m[:, None] * stride_sm + offs_n[None, :] * stride_sn
            )
            mask2d = mask_m[:, None] & mask_n[None, :]
            scores = tl.load(row_ptr, mask=mask2d, other=0.0).to(tl.float32)

            if SCORING_FUNC == 0:
                # sigmoid: bias is for ranking only, weight is bias-free.
                activated = tl.sigmoid(scores)
                biased = activated + row_bias
            elif SCORING_FUNC == 1:
                if SQRTSOFTPLUS_LOG1P:
                    # log1p preserves small positive scores for negative logits.
                    sp = tl.where(
                        scores > 20.0, scores, libdevice.log1p(libdevice.exp(scores))
                    )
                    activated = libdevice.sqrt(sp)
                else:
                    # Open-coded log1p; reproduces the DeepSeek-V4 numerics.
                    z = tl.exp(-tl.abs(scores))
                    u = 1.0 + z
                    exact = u == 1.0
                    log1p_z = tl.where(
                        exact, z, z * tl.log(u) / tl.where(exact, 1.0, u - 1.0)
                    )
                    sp = tl.maximum(scores, 0.0) + log1p_z
                    activated = tl.sqrt(sp)
                biased = activated + row_bias
            else:
                # softmax over the row: weight is the softmax probability.
                logit = scores + row_bias
                biased = tl.where(mask_n[None, :], logit, -float("inf"))
                row_max = tl.max(biased, axis=1)[:, None]
                exp_row = tl.where(mask_n[None, :], tl.exp(biased - row_max), 0.0)
                row_sum = tl.sum(exp_row, axis=1)[:, None]
                activated = exp_row / row_sum

            biased = tl.where(mask_n[None, :], biased, -float("inf"))
            if SCORING_FUNC == 1 and SQRTSOFTPLUS_LOG1P:
                # Rank NaNs above finite scores, matching torch.topk.
                biased = tl.where(biased == biased, biased, float("inf"))
            else:
                biased = tl.where(biased == biased, biased, -1e30)

            offs_k = tl.arange(0, BLOCK_K)  # [BLOCK_K]
            mask_k_total = offs_k < K
            mask_k_routed = offs_k < K_ROUTED
            selected_vals = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)
            selected_idx = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.int32)

            cur = biased
            remaining = tl.broadcast_to(mask_n[None, :], (BLOCK_M, BLOCK_N))
            for k in tl.static_range(K_ROUTED):
                max_val = tl.max(cur, axis=1)[:, None]
                is_max = remaining & (cur == max_val)
                lane_id = tl.where(is_max, offs_n[None, :], N + 1)
                win_lane = tl.min(lane_id, axis=1)[:, None].to(tl.int32)
                win_activated = tl.sum(
                    tl.where(offs_n[None, :] == win_lane, activated, 0.0), axis=1
                )[:, None]
                slot = offs_k[None, :] == k
                selected_vals = tl.where(slot, win_activated, selected_vals)
                selected_idx = tl.where(slot, win_lane, selected_idx)
                remaining = remaining & (offs_n[None, :] != win_lane)
                cur = tl.where(remaining, cur, -float("inf"))

            routed_sum = tl.sum(
                tl.where(mask_k_routed[None, :], selected_vals, 0.0), axis=1
            )[:, None]

            if RENORMALIZE:
                if RENORMALIZE_EPSILON > 0.0:
                    norm = routed_sum + RENORMALIZE_EPSILON
                else:
                    norm = tl.where(routed_sum > 0.0, routed_sum, 1.0)
                selected_vals = selected_vals / norm
            if APPLY_SCALE:
                selected_vals = selected_vals * routed_scaling_factor

            out_w_ptr = (
                out_weights_ptr
                + offs_m[:, None] * stride_wm
                + offs_k[None, :] * stride_wk
            )
            out_i_ptr = (
                out_indices_ptr
                + offs_m[:, None] * stride_im
                + offs_k[None, :] * stride_ik
            )
            store_mask = mask_m[:, None] & mask_k_total[None, :]
            if GRAPH_ROWS:
                real = tl.load(valid_mask_ptr + offs_m, mask_m, other=False)
                live_routes = real[:, None] & mask_k_routed[None, :]
                selected_vals = tl.where(live_routes, selected_vals, 0.0)
                selected_idx = tl.where(live_routes, selected_idx, -1)
                if pid == 0:
                    # A live row always has a nonnegative first ID, including
                    # the existing nonfinite-score handling above.
                    rows = tl.arange(0, GRAPH_ROWS)
                    valid_rows = tl.load(valid_mask_ptr + rows, rows < M, other=False)
                    tl.store(routed_rows_ptr, tl.sum(valid_rows.to(tl.int32), 0))
            tl.store(out_w_ptr, selected_vals, mask=store_mask)
            tl.store(out_i_ptr, selected_idx, mask=store_mask)
            if K > BLOCK_K:
                # Draft top3 keeps the legacy four-lane FP32 reduction.
                tail_mask = mask_m[:, None] & (offs_k[None, :] + BLOCK_K < K)
                tl.store(out_w_ptr + BLOCK_K * stride_wk, 0.0, mask=tail_mask)
                tl.store(out_i_ptr + BLOCK_K * stride_ik, -1, mask=tail_mask)

        # Adapted from sglang-for-sail kernels/ops/moe/ep_moe_kernels.py
        # (512b03d), Apache-2.0: _fwd_kernel_ep_scatter_1.
        @triton.jit
        def _ep_scatter_m_indices_kernel(
            num_recv_tokens_per_expert,
            num_valid_tokens_per_expert,
            expert_start_loc,
            m_indices,
            num_experts: tl.constexpr,
            BLOCK_E: tl.constexpr,
            BLOCK_EXPERT_NUM: tl.constexpr,
        ):
            cur_expert = tl.program_id(0)
            offset_cumsum = tl.arange(0, BLOCK_EXPERT_NUM)
            tokens_per_expert = tl.load(
                num_recv_tokens_per_expert + offset_cumsum,
                mask=offset_cumsum < num_experts,
                other=0,
            )
            cumsum = tl.cumsum(tokens_per_expert).to(tl.int32) - tokens_per_expert
            tl.store(
                expert_start_loc + offset_cumsum,
                cumsum,
                mask=offset_cumsum < num_experts,
            )
            # PPU needs a forced sync here, otherwise cur_expert_start would
            # load staled data (sglang gates this on _is_sail_tl; kept
            # unconditional for the PPU build).
            tl.debug_barrier()
            cur_expert_start = tl.load(expert_start_loc + cur_expert)
            cur_expert_padded_token_num = tl.load(
                num_recv_tokens_per_expert + cur_expert
            )
            cur_expert_valid_token_num = tl.load(
                num_valid_tokens_per_expert + cur_expert
            )
            m_indices_start_ptr = m_indices + cur_expert_start
            off_expert = tl.arange(0, BLOCK_E)
            for start_m in tl.range(
                0, cur_expert_padded_token_num, BLOCK_E, num_stages=4
            ):
                offsets = start_m + off_expert
                tl.store(
                    m_indices_start_ptr + offsets,
                    tl.where(offsets < cur_expert_valid_token_num, cur_expert, -1),
                )

        # Adapted from sglang-for-sail kernels/ops/moe/ep_moe_kernels.py
        # (512b03d), Apache-2.0: _fwd_kernel_ep_scatter_2_optimal (bf16-only,
        # no activation scales).
        @triton.jit
        def _ep_scatter_tokens_kernel(
            expert_start_loc,
            recv_x,
            recv_x_stride0,
            recv_x_stride1,
            recv_topk,
            recv_topk_stride0,
            recv_topk_stride1,
            output_tensor,
            output_tensor_stride0,
            output_tensor_stride1,
            output_index,
            output_index_stride0,
            output_index_stride1,
            topk_num: tl.constexpr,
            HIDDEN_SIZE: tl.constexpr,
            COPY_SIZE: tl.constexpr,
            BLOCK_SIZE: tl.constexpr,
            num_stages: tl.constexpr,
        ):
            token_id_int32 = tl.program_id(0)
            token_id = token_id_int32.to(tl.int64)

            expert_offsets = tl.arange(0, BLOCK_SIZE)
            expert_mask = expert_offsets < topk_num
            expert_loc = tl.load(
                recv_topk + token_id * recv_topk_stride0 + expert_offsets,
                mask=expert_mask,
                other=-1,
            )

            tt_mask = expert_mask & (expert_loc >= 0)
            if tt_mask.sum() == 0:
                return
            dest_token_index_int32 = tl.atomic_add(
                expert_start_loc + expert_loc, 1, mask=tt_mask
            )
            tl.store(
                output_index + token_id * output_index_stride0 + expert_offsets,
                dest_token_index_int32,
                mask=tt_mask,
                eviction_policy="evict_last",
            )
            tl.debug_barrier()
            dest_token_index = tl.load(
                output_index + token_id * output_index_stride0 + expert_offsets,
                mask=tt_mask,
            ).to(tl.int64)

            value_offsets = tl.arange(0, COPY_SIZE)
            for _ in tl.range(
                0, triton.cdiv(HIDDEN_SIZE, COPY_SIZE), num_stages=num_stages
            ):
                copy_mask = value_offsets < HIDDEN_SIZE
                to_copy = tl.load(
                    recv_x + token_id * recv_x_stride0 + value_offsets, mask=copy_mask
                )
                output_offsets = (
                    dest_token_index[:, None] * output_tensor_stride0
                    + value_offsets[None, :]
                )
                to_copy = to_copy[None, :].broadcast_to(BLOCK_SIZE, COPY_SIZE)
                tl.store(
                    output_tensor + output_offsets,
                    to_copy,
                    mask=(copy_mask[None, :] & tt_mask[:, None]),
                )
                value_offsets += COPY_SIZE

        # Adapted from sglang-for-sail kernels/ops/moe/ep_moe_kernels.py
        # (512b03d), Apache-2.0: _fwd_kernel_ep_gather.
        @triton.jit
        def _ep_gather_kernel(
            total_token_num,
            input_tensor,
            input_tensor_stride0,
            input_tensor_stride1,
            recv_topk_ids,
            recv_topk_ids_stride0,
            recv_topk_ids_stride1,
            recv_topk_weight,
            recv_topk_weight_stride0,
            recv_topk_weight_stride1,
            input_index,
            input_index_stride0,
            input_index_stride1,
            output_tensor,
            output_tensor_stride0,
            output_tensor_stride1,
            routed_scaling_factor,
            topk_num: tl.constexpr,
            BLOCK_D: tl.constexpr,
        ):
            cur_block_int32 = tl.program_id(0)
            cur_block = cur_block_int32.to(tl.int64)
            start_cur_token_int32 = tl.program_id(1)
            grid_num = tl.num_programs(1)
            for cur_token_int32 in range(
                start_cur_token_int32, total_token_num, grid_num
            ):
                cur_token = cur_token_int32.to(tl.int64)
                off_d = tl.arange(0, BLOCK_D)
                accumulator = tl.zeros([BLOCK_D], dtype=tl.float32)
                for topk_index_int32 in range(0, topk_num):
                    topk_index = topk_index_int32.to(tl.int64)
                    expert_id = tl.load(
                        recv_topk_ids + cur_token * recv_topk_ids_stride0 + topk_index
                    )
                    if expert_id >= 0:
                        source_token_index_int32 = tl.load(
                            input_index + cur_token * input_index_stride0 + topk_index
                        )
                        source_token_index = source_token_index_int32.to(tl.int64)
                        acc_weight = tl.load(
                            recv_topk_weight
                            + cur_token * recv_topk_weight_stride0
                            + topk_index
                        )
                        tmp = tl.load(
                            input_tensor
                            + source_token_index * input_tensor_stride0
                            + cur_block * BLOCK_D
                            + off_d
                        )
                        accumulator += tmp.to(tl.float32) * acc_weight
                tl.store(
                    output_tensor
                    + cur_token * output_tensor_stride0
                    + cur_block * BLOCK_D
                    + off_d,
                    (accumulator * routed_scaling_factor).to(
                        output_tensor.dtype.element_ty
                    ),
                )

        class _Kernels:
            pass

        # Adapted from sglang-for-sail
        # python/sglang/srt/layers/quantization/ppu_mxfp4_utils.py
        # (512b03d, Apache-2.0): _downcast_to_mxfp4. The only deviation: the
        # module-level ``MXFP_BLOCK_SIZE`` tl.constexpr global becomes a
        # constexpr kernel parameter (closure-defined kernels cannot capture
        # module globals here); the body is otherwise verbatim.
        @triton.jit
        def _downcast_to_mxfp4_kernel(
            # Output: packed e2m1 tensor (uint8, two fp4 per byte)
            mx_tensor_ptr,
            stride_mxt_outer,
            stride_mxt_quant: tl.constexpr,
            # Output: preprocessed E8M0 scales (uint16, packed + transposed
            # layout [S_pairs, N])
            mx_scale_ptr,
            stride_mx_scale_pair,
            stride_mx_scale_outer,
            # Input: source tensor (bf16/fp16/fp32)
            src_ptr,
            stride_src_outer,
            stride_src_quant,
            outer_dim,
            quant_dim,
            orig_quant_dim,
            BLOCK_SIZE_OUT_DIM: tl.constexpr,
            BLOCK_SIZE_QUANT_DIM: tl.constexpr,
            DEQUANT_SCALE_ROUNDING_MODE: tl.constexpr,
            MXFP_BLOCK_SIZE: tl.constexpr,
        ):
            """
            Fused MXFP4 downcast kernel: E8M0 scale computation + e2m1
            quantization in a single kernel (PPU hardware path).
            """
            tl.static_assert(
                stride_mxt_quant == 1, f"Output stride, {stride_mxt_quant=} must be 1."
            )
            tl.static_assert(
                BLOCK_SIZE_QUANT_DIM % MXFP_BLOCK_SIZE == 0,
                f"{BLOCK_SIZE_QUANT_DIM=} must be a multiple of 32",
            )
            tl.static_assert(
                (BLOCK_SIZE_QUANT_DIM // MXFP_BLOCK_SIZE) % 2 == 0,
                "Number of scale groups per block must be even for uint16 packing",
            )

            src_dtype: tl.constexpr = src_ptr.dtype.element_ty
            tl.static_assert(
                mx_tensor_ptr.dtype.element_ty == tl.uint8,
                "Output must be uint8 (packed e2m1).",
            )
            tl.static_assert(
                mx_scale_ptr.dtype.element_ty == tl.uint16,
                "Scale must be uint16 (preprocessed E8M0).",
            )
            tl.static_assert(
                (src_dtype == tl.bfloat16)
                or (src_dtype == tl.float16)
                or (src_dtype == tl.float32),
                f"{src_dtype=} must be bfloat16, float16, or float32",
            )

            outer_block = tl.program_id(0).to(tl.int64)
            quant_block = tl.program_id(1).to(tl.int64)

            BLOCK_SIZE_QUANT_MX_SCALE: tl.constexpr = (
                BLOCK_SIZE_QUANT_DIM // MXFP_BLOCK_SIZE
            )
            BLOCK_SIZE_QUANT_MX_SCALE_PAIRS: tl.constexpr = (
                BLOCK_SIZE_QUANT_MX_SCALE // 2
            )
            BLOCK_SIZE_QUANT_MX_TENSOR: tl.constexpr = (
                BLOCK_SIZE_QUANT_DIM // 2
            )  # 2 e2m1 per uint8

            start_src_quant = quant_block * BLOCK_SIZE_QUANT_DIM
            start_mx_scale_pair = quant_block * BLOCK_SIZE_QUANT_MX_SCALE_PAIRS
            start_mx_quant = quant_block * BLOCK_SIZE_QUANT_MX_TENSOR
            start_out = outer_block * BLOCK_SIZE_OUT_DIM

            src_ptr += start_src_quant * stride_src_quant + start_out * stride_src_outer
            mx_scale_ptr += (
                start_mx_scale_pair * stride_mx_scale_pair
                + start_out * stride_mx_scale_outer
            )
            mx_tensor_ptr += (
                start_mx_quant * stride_mxt_quant + start_out * stride_mxt_outer
            )

            offs_src_quant = tl.arange(0, BLOCK_SIZE_QUANT_DIM)[None, :].to(tl.int64)
            offs_mxt_quant = tl.arange(0, BLOCK_SIZE_QUANT_MX_TENSOR)[None, :].to(
                tl.int64
            )
            offs_scale_pairs = tl.arange(0, BLOCK_SIZE_QUANT_MX_SCALE_PAIRS)[
                None, :
            ].to(tl.int64)
            offs_outer = tl.arange(0, BLOCK_SIZE_OUT_DIM)[:, None].to(tl.int64)

            mask_src_quant = start_src_quant + offs_src_quant < orig_quant_dim
            mask_n = start_out + offs_outer < outer_dim
            full_mask_src = mask_src_quant & mask_n

            mask_mxt_quant = start_mx_quant + offs_mxt_quant < orig_quant_dim // 2
            full_mask_mxt = mask_mxt_quant & mask_n

            S_GROUPS_PAIRS = quant_dim // MXFP_BLOCK_SIZE // 2
            scale_mask_k = start_mx_scale_pair + offs_scale_pairs < S_GROUPS_PAIRS
            full_scale_mask = scale_mask_k & mask_n

            src_tensor_offsets = (
                offs_src_quant * stride_src_quant + offs_outer * stride_src_outer
            )
            mx_scale_offsets = (
                offs_scale_pairs * stride_mx_scale_pair
                + offs_outer * stride_mx_scale_outer
            )
            mx_tensor_offsets = (
                offs_mxt_quant * stride_mxt_quant + offs_outer * stride_mxt_outer
            )
            src_tensor = tl.load(src_ptr + src_tensor_offsets, mask=full_mask_src)

            # ======== E8M0 scale computation ========
            f32_tensor = src_tensor.to(tl.float32)
            abs_tensor = tl.abs(f32_tensor)
            abs_tensor = tl.where(full_mask_src, abs_tensor, -1.0)
            abs_tensor = tl.reshape(
                abs_tensor,
                [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, MXFP_BLOCK_SIZE],
            )
            max_val = tl.max(abs_tensor, axis=2, keep_dims=True)
            # Clamp absmax to avoid division by zero in scale computation
            max_val = tl.maximum(max_val, 1e-10)

            if DEQUANT_SCALE_ROUNDING_MODE == 0:
                # ROUND_UP: 2 ** ceil(log2(max / 6.0))
                dequant_scale = max_val / 6.0
                dequant_scale_exponent = (
                    dequant_scale.to(tl.uint32, bitcast=True) + 0x007FFFFF
                ) & 0x7F800000
            else:
                # ROUND_DOWN: 2 ** floor(log2(max / 4.0))
                dequant_scale = max_val / 4.0
                dequant_scale_exponent = (
                    dequant_scale.to(tl.uint32, bitcast=True) & 0x7F800000
                )

            # dequant_scale_rounded is guaranteed non-zero (absmax >= 1e-10)
            dequant_scale_rounded = dequant_scale_exponent.to(tl.float32, bitcast=True)
            quant_scale = 1.0 / dequant_scale_rounded

            # Apply per-group scale
            f32_tensor = tl.reshape(
                f32_tensor,
                [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, MXFP_BLOCK_SIZE],
            )
            quant_tensor = f32_tensor * quant_scale
            quant_tensor = quant_tensor.reshape(
                [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM]
            )
            # Note: padding elements become 0 * quant_scale = 0 after load with
            # mask (other=0.0), and e2m1(0.0) = 0x0, so no explicit zeroing is
            # needed for correctness.

            # Pack scale exponents into uint16 pairs directly from uint32
            dequant_scale_exponent = dequant_scale_exponent.reshape(
                [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE]
            )
            # Zero out exponents for padding groups (beyond original quant_dim)
            ORIG_S_GROUPS = orig_quant_dim // MXFP_BLOCK_SIZE
            scale_group_idx = (
                quant_block * BLOCK_SIZE_QUANT_MX_SCALE
                + tl.arange(0, BLOCK_SIZE_QUANT_MX_SCALE)[None, :]
            )
            dequant_scale_exponent = tl.where(
                scale_group_idx < ORIG_S_GROUPS, dequant_scale_exponent, 0
            )
            # Reshape to pair groups and extract exponent bytes, pack into
            # uint16 in one go
            de_pairs = tl.reshape(
                dequant_scale_exponent,
                [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE_PAIRS, 2],
            )
            lo_u32, hi_u32 = tl.split(de_pairs)
            scale_tensor = ((lo_u32 >> 23) | ((hi_u32 >> 23) << 8)).to(tl.uint16)

            # ======== FP32 -> e2m1 conversion (PPU hardware instruction) ========
            pairs = tl.reshape(
                quant_tensor, [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM // 2, 2]
            )
            lo_f, hi_f = tl.split(pairs)
            lo_f32 = lo_f.to(tl.float32)
            hi_f32 = hi_f.to(tl.float32)

            # cvt.rn.satfinite.e2m1x2.f32: two f32 -> one packed e2m1x2 (uint8)
            out_tensor = tl.inline_asm_elementwise(
                """
                {
                    .reg .b8 r;
                    cvt.rn.satfinite.e2m1x2.f32 r, $1, $2;
                    mov.b32 $0, {r, r, r, r};
                }
                """,
                constraints="=r,f,f",
                args=[hi_f32, lo_f32],
                dtype=tl.uint8,
                is_pure=True,
                pack=1,
            )

            # Write scale (uint16 packed, transposed [S_pairs, N] layout) and
            # tensor
            tl.store(
                mx_scale_ptr + mx_scale_offsets, scale_tensor, mask=full_scale_mask
            )
            tl.store(mx_tensor_ptr + mx_tensor_offsets, out_tensor, mask=full_mask_mxt)

        bundle = _Kernels()
        bundle.triton = triton
        bundle.accumulate_mega_expert_load = _accumulate_mega_expert_load_kernel
        bundle.router_gate = _router_gate_kernel
        bundle.scatter_m_indices = _ep_scatter_m_indices_kernel
        bundle.scatter_tokens = _ep_scatter_tokens_kernel
        bundle.gather = _ep_gather_kernel
        bundle.downcast_mxfp4 = _downcast_to_mxfp4_kernel
        _TRITON_KERNELS = bundle
        return bundle


def ep_scatter_sail(
    recv_x: torch.Tensor,
    recv_topk: torch.Tensor,
    num_recv_tokens_per_expert: torch.Tensor,
    expert_start_loc: torch.Tensor,
    output_tensor: torch.Tensor,
    m_indices: torch.Tensor,
    output_index: torch.Tensor,
    block_e: int = 1,
) -> None:
    """Scatter DeepEP normal-mode recv tokens into the grouped-GEMM layout.

    Adapted from sglang-for-sail ``ep_scatter_sail`` (512b03d, Apache-2.0),
    bf16 dispatch only, ``BLOCK_E=1`` (PPU expert_alignment=1).
    ``recv_topk`` holds LOCAL expert ids (``-1`` for padding/non-local slots).
    """
    kernels = _get_triton_kernels()
    triton = kernels.triton
    num_experts = num_recv_tokens_per_expert.shape[0]
    hidden_size = recv_x.shape[1]
    num_warps = 8
    grid = (num_experts,)
    assert m_indices.shape[0] % block_e == 0
    kernels.scatter_m_indices[grid](
        num_recv_tokens_per_expert,
        num_recv_tokens_per_expert,
        expert_start_loc,
        m_indices,
        num_experts=num_experts,
        num_warps=num_warps,
        BLOCK_E=block_e,
        BLOCK_EXPERT_NUM=triton.next_power_of_2(num_experts),
    )
    grid = (recv_x.shape[0],)
    kernels.scatter_tokens[grid](
        expert_start_loc,
        recv_x,
        recv_x.stride(0),
        recv_x.stride(1),
        recv_topk,
        recv_topk.stride(0),
        recv_topk.stride(1),
        output_tensor,
        output_tensor.stride(0),
        output_tensor.stride(1),
        output_index,
        output_index.stride(0),
        output_index.stride(1),
        topk_num=recv_topk.shape[1],
        HIDDEN_SIZE=hidden_size,
        COPY_SIZE=512,
        BLOCK_SIZE=triton.next_power_of_2(recv_topk.shape[1]),
        num_stages=3,
        num_warps=num_warps,
    )


def ep_gather(
    input_tensor: torch.Tensor,
    recv_topk_ids: torch.Tensor,
    recv_topk_weight: torch.Tensor,
    input_index: torch.Tensor,
    output_tensor: torch.Tensor,
    routed_scaling_factor: float = 1.0,
) -> None:
    """Gather grouped-GEMM expert outputs back to recv-token rows (weighted).

    Adapted from sglang-for-sail ``ep_gather`` (512b03d, Apache-2.0). Slots
    with ``expert_id < 0`` contribute nothing. The router weights already
    carry the routed scaling factor, so ``routed_scaling_factor`` stays 1.0.
    """
    kernels = _get_triton_kernels()
    triton = kernels.triton
    num_tokens = output_tensor.shape[0]
    hidden_size = input_tensor.shape[1]
    num_warps = 2
    block_d = 128 if hidden_size % 1024 != 0 else 1024
    while hidden_size % block_d:
        block_d = block_d // 2
    assert hidden_size % block_d == 0
    grid = (triton.cdiv(hidden_size, block_d), min(num_tokens, 1024))
    kernels.gather[grid](
        num_tokens,
        input_tensor,
        input_tensor.stride(0),
        input_tensor.stride(1),
        recv_topk_ids,
        recv_topk_ids.stride(0),
        recv_topk_ids.stride(1),
        recv_topk_weight,
        recv_topk_weight.stride(0),
        recv_topk_weight.stride(1),
        input_index,
        input_index.stride(0),
        input_index.stride(1),
        output_tensor,
        output_tensor.stride(0),
        output_tensor.stride(1),
        routed_scaling_factor,
        topk_num=recv_topk_ids.shape[1],
        num_warps=num_warps,
        BLOCK_D=block_d,
    )


# ---------------------------------------------------------------------------
# PPU MXFP4 weight/activation preprocessing (copied from sglang-for-sail)
# ---------------------------------------------------------------------------


def preprocess_mxfp4_w4a16_mma_scales(scales: torch.Tensor) -> torch.Tensor:
    """Reorder group-32 E8M0 scales for the PPU W4A16 MMA kernels.

    Adapted from sglang-for-sail
    ``python/sglang/srt/layers/quantization/ppu_mxfp4_utils.py::
    preprocess_mxfp4_w4a16_mma_scales`` (512b03d, Apache-2.0) — pure-torch
    permutation, no triton. Expects ``[E, N, K/32]`` scales with N and K
    multiples of 64.

    NOTE: kept as the documented W4A16-MMA layout reference (and for its CPU
    tests); the DeepEP device path no longer uses it — the W4A16 nopad kernel
    derives ``group_size = k // (n // 64)``, which can never divide k for
    V4.1 Flash (w13 n=4608), so the device path runs FP4xFP4 (f4f4) instead
    (see :func:`preprocess_mxfp4_f4f4_scales`).
    """
    if scales.ndim != 3:
        raise ValueError(f"Expected [E, N, K/32] scales, got {scales.shape}.")
    if scales.dtype == torch.float8_e8m0fnu:
        scales = scales.view(torch.uint8)
    elif scales.dtype != torch.uint8:
        raise TypeError(f"Expected uint8/E8M0 scales, got {scales.dtype}.")

    num_experts, n, k_groups = scales.shape
    k = k_groups * 32
    if n % 64 != 0 or k % 64 != 0:
        raise ValueError(
            f"PPU W4A16 MMA requires N and K to be multiples of 64, got N={n}, K={k}."
        )

    scale_perm = torch.tensor(
        [
            2 * i + j * 16 + offset
            for i in range(8)
            for j in range(8)
            for offset in (0, 1)
        ],
        dtype=torch.long,
        device=scales.device,
    )
    scales = (
        scales.reshape(num_experts, n // 64, 64, k // 64, 2)
        .permute(0, 1, 3, 2, 4)
        .reshape(-1, 128)
    )
    return (
        scales.index_select(1, scale_perm)
        .reshape(num_experts, n // 64, k * 2)
        .contiguous()
    )


def preprocess_mxfp4_f4f4_scales(scales: torch.Tensor) -> torch.Tensor:
    """Pack group-32 UE8M0 weight scales for the PPU FP4xFP4 GEMM kernels.

    Mirrors the PPU ``deep_gemm.preprocess_mxfp4_scales`` wheel entry
    (``jit_kernels/gemm_fp4.py``, 3-D grouped case) — the packing sglang-for-
    sail applies to the stacked expert scales in ``mxfp4.py`` (512b03d,
    Apache-2.0) before ``m_grouped_gemm_fp4_fp4_bf16_nt_nopad``. ``[E, N,
    K/32]`` uint8 scales become ``[E, N, K/64]`` uint16 pairs (even group =
    low byte) in the N-major layout the kernel asserts (``stride(1) == 1``);
    an odd group count is zero-padded exactly like the wheel. Pure torch —
    CPU-testable (verified byte-identical to the wheel on the P machine).
    """
    if scales.ndim != 3:
        raise ValueError(f"Expected [E, N, K/32] scales, got {scales.shape}.")
    if scales.dtype == torch.float8_e8m0fnu:
        scales = scales.view(torch.uint8)
    elif scales.dtype != torch.uint8:
        raise TypeError(f"Expected uint8/E8M0 scales, got {scales.dtype}.")
    scales = scales.contiguous()
    if scales.shape[-1] % 2:
        scales = torch.nn.functional.pad(scales, (0, 1))
    return scales.view(torch.uint16).permute(0, 2, 1).contiguous().permute(0, 2, 1)


def downcast_to_mxfp4(
    src_tensor: torch.Tensor, axis: int = 1, round_up: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Downcast bf16/fp16/fp32 activations to packed MXFP4 (e2m1) + scales.

    Adapted from sglang-for-sail ``python/sglang/srt/layers/quantization/
    ppu_mxfp4_utils.py::downcast_to_mxfp4`` (512b03d, Apache-2.0) — the
    verified PPU activation path of the f4f4 DeepGEMM MoE. The fused-MoE
    uint16 scale variant (``enable_fused_moe``) and the ``is_sail()`` /
    SM>=89 assert are dropped (the caller gates on the PPU device). Returns
    ``(out_quant uint8 [..., L/2], out_scale uint16 [..., ceil(L/64)])`` where
    the scale is already in the packed-UE8M0 transposed layout DeepGEMM
    expects (2-D result: ``stride(0) == 1``, i.e. physical ``[S_pairs, N]``) —
    equivalent to ``deep_gemm.preprocess_mxfp4_scales`` on the raw bytes.
    """
    kernels = _get_triton_kernels()
    triton = kernels.triton
    ndim = src_tensor.ndim
    assert -ndim <= axis < ndim, f"Invalid {axis=}"
    axis = axis if axis >= 0 else axis + ndim

    src_tensor = src_tensor.transpose(axis, ndim - 1)
    L = src_tensor.shape[-1]
    assert L % 2 == 0, f"axis dim must be divisible by 2 for e2m1. Got {L}"

    # Pad quant_dim to multiple of 64 for even scale groups (uint16 packing).
    # No actual data padding (no F.pad) — the kernel masks OOB elements.
    padded_L = ((L + 63) // 64) * 64
    out_shape = src_tensor.shape[:-1] + (L // 2,)

    S_groups_pairs = padded_L // _MXFP_BLOCK_SIZE // 2

    out_quant_tensor = src_tensor.new_empty(out_shape, dtype=torch.uint8)

    if src_tensor.numel() > 0:
        kernel_src = src_tensor.reshape(-1, src_tensor.shape[-1])
        kernel_quant = out_quant_tensor.view(-1, out_quant_tensor.shape[-1])
        N = kernel_src.shape[0]  # flattened outer dim

        # Scale output: [S_groups_pairs, N] uint16 contiguous (transposed
        # layout); the trailing ``.t()`` below yields the [N, S_pairs] view
        # with stride(0) == 1 that the DeepGEMM SFA layout check requires.
        kernel_scale = torch.empty(
            (S_groups_pairs, N), dtype=torch.uint16, device=src_tensor.device
        )
        kernel_scale_stride = kernel_scale.stride()

        BLOCK_OUT = 32
        BLOCK_QUANT = 128
        NUM_WARPS = 4

        grid = (
            triton.cdiv(N, BLOCK_OUT),
            triton.cdiv(padded_L, BLOCK_QUANT),
        )

        kernels.downcast_mxfp4[grid](
            kernel_quant,
            *kernel_quant.stride(),
            kernel_scale,
            *kernel_scale_stride,
            kernel_src,
            *kernel_src.stride(),
            N,
            padded_L,
            L,
            BLOCK_SIZE_OUT_DIM=BLOCK_OUT,
            BLOCK_SIZE_QUANT_DIM=BLOCK_QUANT,
            DEQUANT_SCALE_ROUNDING_MODE=0 if round_up else 1,
            MXFP_BLOCK_SIZE=_MXFP_BLOCK_SIZE,
            num_warps=NUM_WARPS,
        )

        batch_shape = src_tensor.shape[:-1]
        out_scale = kernel_scale.t().reshape(*batch_shape, S_groups_pairs)
    else:
        batch_shape = src_tensor.shape[:-1]
        out_scale = torch.empty(
            *batch_shape, S_groups_pairs, dtype=torch.uint16, device=src_tensor.device
        )

    out_quant_tensor = out_quant_tensor.transpose(axis, ndim - 1)
    out_scale = out_scale.transpose(axis, ndim - 1)
    return out_quant_tensor, out_scale


def _apply_swiglu_limit(gateup_output: torch.Tensor, swiglu_limit: float) -> None:
    """Clamp the grouped runner's GEMM workspace in place.

    Adapted from sglang-for-sail
    ``python/sglang/srt/layers/moe/moe_runner/deep_gemm.py::
    _apply_swiglu_limit`` (512b03d, Apache-2.0): clamp gate/up BEFORE the
    SiLU (up to ``[-limit, limit]``, gate to ``[-inf, limit]``).
    """
    assert swiglu_limit == 10
    num_tokens, hidden_size_x2 = gateup_output.shape
    assert gateup_output.dtype == torch.bfloat16
    gate, up = torch.chunk(gateup_output, chunks=2, dim=-1)
    assert gate.shape == (num_tokens, hidden_size_x2 // 2)
    assert up.shape == (num_tokens, hidden_size_x2 // 2)
    up.clamp_(min=-swiglu_limit, max=swiglu_limit)
    gate.clamp_(max=swiglu_limit)


def quantize_per_row_fp8(values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """E4M3 codes and contiguous FP32 [..., 1] dequantization scales.

    DeepGEMM-for-sail W4A8 consumes one scale per row, not packed group-32
    activation scales. Keep intermediates in FP32, including routed weights.
    The positive floor makes empty-valued rows quantize to finite zeros.
    """
    values = values.float()
    amax = values.abs().amax(dim=-1, keepdim=True).clamp_min(1e-10)
    scale = amax / 448.0
    quant = (values * (448.0 / amax)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return quant.contiguous(), scale.contiguous()


def grouped_route_weights(
    local_ids: torch.Tensor,
    weights: torch.Tensor,
    output_index: torch.Tensor,
    grouped_rows: int,
) -> torch.Tensor:
    """Map valid (received token, top-k slot) weights to scatter output rows.

    Scatter assigns a distinct row to each valid route, including multiple
    experts on the same rank. Never read uninitialized padded-slot indices.
    """
    valid = local_ids >= 0
    result = torch.empty(grouped_rows, dtype=torch.float32, device=weights.device)
    result.index_copy_(0, output_index[valid].long(), weights[valid].to(torch.float32))
    return result


def routed_swiglu_fp8(
    gateup: torch.Tensor, weights: torch.Tensor, limit: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Clamp BF16 GEMM1 output, then weight FP32 SwiGLU before FP8 casting."""
    gate, up = gateup.chunk(2, dim=-1)
    gate, up = gate.float(), up.float()
    if limit > 0:
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    hidden = F.silu(gate) * up
    hidden.mul_(weights.float().unsqueeze(-1))
    return quantize_per_row_fp8(hidden)


# ---------------------------------------------------------------------------
# PPUMoE
# ---------------------------------------------------------------------------


class PPUMoE:
    """Expert-parallel MoE for the PPU V4.1 strategy.

    Weight binding: ``bind_expert_weights(layer_id, w13_pair, w2_pair)`` with
    each pair ``(packed_fp4_weight, ue8m0_scale)``:

      * w13: weight int8/uint8 ``[E, 2I, K/2]`` + scale ``[E, 2I, K/32]``
      * w2:  weight int8/uint8 ``[E, D, I/2]`` + scale ``[E, D, I/32]``

    with ``E = num_local_experts`` (48), ``I = 2304``, ``K = D = 5120``
    for Flash-1017. The model constructor supplies checkpoint dimensions.
    Only the local expert shard is resident; the FP4 weights are NEVER
    expanded to BF16 wholesale (the reference path dequantizes only experts
    hit by the current tokens).

    Execution paths:
      * Explicit PPU890 MegaMoE W4A4: native symmetric dispatch, two GEMMs
        and weighted combine in one complete call; all ranks participate,
        including empty owners. This path owns a model-local workspace.
      * DeepEP normal dispatch/combine (``deep_ep`` importable / rtp
        ``DeepEPWrapper`` initialized): 8-slot padded dispatch with
        ``expert_alignment=1`` and rank-local received expert IDs. W4A8
        uses E4M3 activations with FP32 row scales; route weights multiply
        FP32 SwiGLU before the second quantization. The W4A4 override uses
        MXFP4 activations and applies route weights in ``ep_gather``.
        Both paths use two nopad grouped GEMMs and ``buffer.combine``.
      * LOCAL path (no ``deep_ep``, ``ep_size == 1``): no dispatch; the
        reference per-expert loop multiplies ``expert_weights`` explicitly.
      * T == 0: returns ``torch.empty(0, hidden)`` — on the DeepEP path
        dispatch + combine still run so the rank participates in the
        collectives.
    """

    _BUFFER_CACHE: Dict[Tuple[int, int], object] = {}
    _BUFFER_LOCK = threading.Lock()
    _MEGA_ENGINES = weakref.WeakSet()

    def __init__(
        self,
        *,
        num_routed_experts: int = 384,
        num_physical_experts: Optional[int] = None,
        num_local_experts: int = 48,
        ep_rank: int = 0,
        ep_size: int = 1,
        hidden_size: int = 5120,
        inter_dim: int = 2048,
        swiglu_limit: float = SWIGLU_LIMIT,
        group: Optional[object] = None,
    ):
        self.num_routed_experts = int(num_routed_experts)
        self.num_physical_experts = int(
            num_routed_experts if num_physical_experts is None else num_physical_experts
        )
        self.num_local_experts = int(num_local_experts)
        self.ep_rank = int(ep_rank)
        self.ep_size = int(ep_size)
        self.hidden_size = int(hidden_size)
        self.inter_dim = int(inter_dim)
        self.swiglu_limit = float(swiglu_limit)
        # Freeze the selected precision for this instance and its weight cache.
        selection = dict(os.environ)
        self.backend = moe_backend(selection)
        self.precision = moe_precision(selection)
        self.grouped_forward_calls = 0
        self.grouped_gemm_calls = 0
        self.mega_moe_calls = 0
        self._mega_module = None
        self._mega_buffer = None
        self._mega_capacity = 0
        self._mega_device = None
        self._mega_recv_stats = None
        self._fake_balance_config = None
        self._fake_balance_op = None
        self.group = group
        if self.num_physical_experts < self.num_routed_experts:
            raise ValueError("physical expert slots must cover logical experts")
        if self.num_local_experts <= 0 or self.num_physical_experts % self.num_local_experts != 0:
            raise ValueError(
                "num_physical_experts must be divisible by num_local_experts"
            )
        if self.ep_size * self.num_local_experts != self.num_physical_experts:
            raise ValueError(
                f"ep_size * num_local_experts must equal num_physical_experts, "
                f"got {self.ep_size} * {self.num_local_experts} != "
                f"{self.num_physical_experts}"
            )
        self.local_expert_start = self.ep_rank * self.num_local_experts
        self.local_expert_end = self.local_expert_start + self.num_local_experts
        self._weights: Dict[int, Tuple[torch.Tensor, ...]] = {}
        self._processed: Dict[Tuple[int, torch.device], Tuple[torch.Tensor, ...]] = {}
        self._eplb_layers = {}
        self._eplb_inventory = {}
        self._eplb_invariants = {}
        self._eplb_routes = {}
        self._current_layer: Optional[int] = None
        self._grouped_path: Optional[bool] = None
        if self.backend == "megamoe":
            with self._BUFFER_LOCK:
                self._MEGA_ENGINES.add(self)

    def configure_fake_balance(self, enabled, *, top_k, dp_rank, dp_size):
        """Freeze the performance-only route override for this model instance."""
        if type(enabled) is not bool:
            raise ValueError("fake balance must be an explicit boolean")
        if not 0 <= dp_rank < dp_size or not 0 < top_k <= self.num_routed_experts:
            raise ValueError("fake balance requires valid DP rank/size and top-k")
        config = (enabled, int(top_k), int(dp_rank), int(dp_size))
        if self._fake_balance_config not in (None, config):
            raise ValueError("fake balance configuration is frozen per model")
        self._fake_balance_config = config

    def _apply_fake_balance(self, expert_ids, expert_weights):
        """Override only real routes; preserve idle rows and communication padding.

        Native FakeBalanceExpertOp expects tightly packed [T, real_top_k]
        tensors. A [:, :6] view of padded [T, 8] storage is not contiguous.
        Work on private compact tensors, then restore masks without host reads
        so changed live-row counts remain valid during Graph replay.
        """
        config = self._fake_balance_config
        if config is None or not config[0] or expert_ids.shape[0] == 0:
            return expert_ids, expert_weights
        _, top_k, dp_rank, dp_size = config
        if expert_ids.ndim != 2 or expert_ids.shape != expert_weights.shape:
            raise ValueError("fake balance requires matching [tokens, slots] routes")
        if expert_ids.shape[1] < top_k:
            raise ValueError("route width does not cover the real MoE top-k")
        if expert_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("fake balance expert IDs must be int32 or int64")
        if expert_weights.dtype != torch.float32:
            raise TypeError("fake balance expert weights must be float32")
        original = expert_ids[:, :top_k]
        valid = original >= 0
        ids = original.clone(memory_format=torch.contiguous_format)
        weights = expert_weights[:, :top_k].clone(memory_format=torch.contiguous_format)
        if ids.is_cuda:
            if self._fake_balance_op is None:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "fake balance requires eager warmup before Graph capture"
                    )
                from rtp_llm.models_py.modules.base.cuda.select_topk import (
                    FakeBalanceExpert,
                )

                self._fake_balance_op = FakeBalanceExpert(
                    self.num_routed_experts, top_k, dp_rank, dp_size, self.ep_size
                )
            self._fake_balance_op(ids, weights)
        else:
            # CPU reference of RTP's existing fakeBalanceExpertKernel mapping.
            index = torch.arange(ids.numel(), device=ids.device).reshape(ids.shape)
            rank_offset = self.ep_size * dp_rank // dp_size
            logical_local = self.num_routed_experts // self.ep_size
            local_offset = int(dp_rank * max(logical_local / dp_size, 1.0))
            rank = (rank_offset + index % self.ep_size) % self.ep_size
            local = (local_offset + index // self.ep_size) % logical_local
            ids.copy_(rank * logical_local + local)
            weights.fill_(1.0)
        ids = torch.where(valid, ids, original)
        weights = torch.where(valid, weights, torch.zeros_like(weights))
        return pad_topk_for_dispatch(ids, weights, slots=expert_ids.shape[1])

    # -- weight binding ---------------------------------------------------

    def bind_expert_weights(
        self,
        layer_id: int,
        w13_pair: Tuple[torch.Tensor, torch.Tensor],
        w2_pair: Tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Bind one layer's local FP4 routed-expert weight shard."""
        if int(layer_id) in self._eplb_layers:
            raise RuntimeError("live EPLB executor allocations cannot be rebound")
        w13_weight, w13_scale = w13_pair
        w2_weight, w2_scale = w2_pair
        two_i, half_k = 2 * self.inter_dim, self.hidden_size // 2
        expected = {
            "w13_weight": (self.num_local_experts, two_i, half_k),
            "w13_scale": (
                self.num_local_experts,
                two_i,
                self.hidden_size // _FP4_GROUP,
            ),
            "w2_weight": (
                self.num_local_experts,
                self.hidden_size,
                self.inter_dim // 2,
            ),
            "w2_scale": (
                self.num_local_experts,
                self.hidden_size,
                self.inter_dim // _FP4_GROUP,
            ),
        }
        actual = {
            "w13_weight": tuple(w13_weight.shape),
            "w13_scale": tuple(w13_scale.shape),
            "w2_weight": tuple(w2_weight.shape),
            "w2_scale": tuple(w2_scale.shape),
        }
        for name in expected:
            if actual[name] != expected[name]:
                raise ValueError(
                    f"layer {layer_id} {name} shape {actual[name]} != expected "
                    f"{expected[name]}"
                )
        if w13_weight.dtype not in (torch.int8, torch.uint8) or w2_weight.dtype not in (
            torch.int8,
            torch.uint8,
        ):
            raise TypeError("packed FP4 expert weights must be int8 or uint8")
        for scale in (w13_scale, w2_scale):
            if scale.dtype not in (
                torch.float8_e8m0fnu,
                torch.uint8,
                torch.int8,
                torch.float32,
            ):
                raise TypeError(f"unsupported expert scale dtype {scale.dtype}")
        self._weights[int(layer_id)] = (w13_weight, w13_scale, w2_weight, w2_scale)
        for key in list(self._processed):
            if key[0] == int(layer_id):
                del self._processed[key]
        self._current_layer = int(layer_id)

    def execution_stats(self) -> dict:
        """Actual device-path counters for functional bring-up evidence."""
        return {
            "backend": self.backend,
            "precision": self.precision,
            "kernel": moe_kernel_name(self.precision, self.backend),
            "grouped_forward_calls": self.grouped_forward_calls,
            "grouped_gemm_calls": self.grouped_gemm_calls,
            "mega_moe_calls": self.mega_moe_calls,
            "fake_balance_expert": bool(
                self._fake_balance_config and self._fake_balance_config[0]
            ),
            "mega_workspace_capacity": self._mega_capacity,
            "mega_received_assignments": (
                0
                if self._mega_recv_stats is None
                else int(self._mega_recv_stats.sum().item())
            ),
            "route_weight_stage": (
                "before_gemm2_activation_quantization"
                if self.precision == "w4a8"
                else "after_gemm2"
            ),
        }

    def _layer_weights(self, layer_id: Optional[int]) -> Tuple[torch.Tensor, ...]:
        return self._weights[self._resolve_layer_id(layer_id)]

    def _resolve_layer_id(self, layer_id: Optional[int]) -> int:
        if layer_id is None:
            layer_id = self._current_layer
        if layer_id is None or int(layer_id) not in self._weights:
            raise RuntimeError(
                "no expert weights bound; call bind_expert_weights first"
            )
        return int(layer_id)

    def _processed_weights(
        self, layer_id: int, device: torch.device
    ) -> Tuple[torch.Tensor, ...]:
        """Prepare and cache the selected backend's weight/scale ABI."""
        layer_id = self._resolve_layer_id(layer_id)
        key = (layer_id, device)
        cached = self._processed.get(key)
        if cached is not None:
            return cached
        if layer_id in self._eplb_layers:
            raise RuntimeError("live EPLB must reuse its bound preprocessed executor tensors")
        w13_w, w13_s, w2_w, w2_s = self._layer_weights(layer_id)
        w13_u = _packed_weight_uint8(w13_w)
        w2_u = _packed_weight_uint8(w2_w)
        if self.backend == "megamoe":
            mega = self._get_mega_module()
            (w13_u, w13_sp), (w2_u, w2_sp) = mega.transform_weights_for_mega_moe(
                (w13_u, _ue8m0_scale_bytes(w13_s).contiguous().view(torch.uint8)),
                (w2_u, _ue8m0_scale_bytes(w2_s).contiguous().view(torch.uint8)),
                activation="swiglu",
            )
            inv13 = inv2 = 1.0
        elif self.precision == "w4a8":
            import deep_gemm

            mode = deep_gemm.PreprocessMode.W4A8_MXFP4_MMA
            w13_u, w13_sp, inv13 = deep_gemm.preprocess_weights_and_scales(
                mode,
                False,
                w13_u,
                _ue8m0_scale_bytes(w13_s).contiguous().view(torch.uint8),
            )
            w2_u, w2_sp, inv2 = deep_gemm.preprocess_weights_and_scales(
                mode,
                False,
                w2_u,
                _ue8m0_scale_bytes(w2_s).contiguous().view(torch.uint8),
            )
        else:
            w13_sp = preprocess_mxfp4_f4f4_scales(_ue8m0_scale_bytes(w13_s))
            w2_sp = preprocess_mxfp4_f4f4_scales(_ue8m0_scale_bytes(w2_s))
            inv13 = inv2 = 1.0
        processed = (w13_u, w13_sp, w2_u, w2_sp, inv13, inv2)
        if os.environ.get("RTP_V41_PPU_WEIGHT_AUDIT_DIR"):
            from rtp_llm.models_py.modules.dsv41.ppu.quant_conversion import audit_loaded_tensor

            label = f"moe:{self.num_routed_experts}:L{layer_id}:{self.backend}:{self.precision}"
            for part, tensor in zip(("w13:weight", "w13:scale", "w2:weight", "w2:scale"), processed[:4]):
                audit_loaded_tensor(label + ":" + part, tensor, self.ep_rank, layout=True)
        self._processed[key] = processed
        return processed

    # -- forward ----------------------------------------------------------

    def _map_expert_ids(self, ids, weights, layer_id, *, single_replica=False):
        if not self._eplb_layers:
            return ids
        layer = self._eplb_layers.get(self._resolve_layer_id(layer_id))
        if layer is None:
            return ids
        if single_replica and layer.logical_to_physical.shape[1] == 1:
            # EplbLayer validates exactly one valid slot per logical expert
            # for a one-column map. There is no replica rotation to compute.
            lookup = ids.reshape(-1).clamp_min(0)
            physical = layer.logical_to_physical.reshape(-1).index_select(0, lookup).view_as(ids)
            physical.masked_fill_(ids < 0, -1)
        else:
            physical = layer.map_ids(ids)
        # Borrow the actual dispatched tensors for bounded native diagnostics.
        # Graph capture retains these same device buffers; no host reads here.
        self._eplb_routes[layer.layer_id] = (ids, physical, weights)
        return physical

    def forward(
        self,
        x: torch.Tensor,
        expert_ids: torch.Tensor,
        expert_weights: torch.Tensor,
        *,
        layer_id: Optional[int] = None,
        expert_load=None,
        prepared_buffer=None,
    ) -> torch.Tensor:
        """Routed-expert forward: BF16 ``[T, hidden]`` in, BF16 ``[T, hidden]`` out.

        ``expert_ids`` are GLOBAL expert ids in ``[T, 8]`` (six real choices
        plus ``-1`` padding); ``expert_weights`` is the matching fp32
        ``[T, 8]`` slot weights (zero on padding slots).
        """
        expert_ids, expert_weights = self._apply_fake_balance(
            expert_ids, expert_weights
        )
        expert_ids = self._map_expert_ids(
            expert_ids, expert_weights, layer_id,
            single_replica=(
                prepared_buffer is not None
                and os.environ.get("RTP_V41_PPU_CP8_YOCO_FAST_PREFILL", "0") == "1"
            ),
        )
        if self.backend == "megamoe":
            # Empty owners must still enter MegaMoE's cross-rank protocol.
            return self._forward_megamoe(
                x, expert_ids, expert_weights, layer_id, expert_load,
                prepared_buffer=prepared_buffer,
            )
        if x.size(0) == 0:
            if self._use_deepep_dispatch():
                # Collective participation: dispatch + combine still run.
                return self._forward_deepep(
                    x, expert_ids, expert_weights, layer_id, expert_load
                )
            return torch.empty((0, self.hidden_size), dtype=x.dtype, device=x.device)
        if self._use_deepep_dispatch():
            return self._forward_deepep(
                x, expert_ids, expert_weights, layer_id, expert_load
            )
        if self.ep_size == 1 or self._offline_single_process():
            return self._forward_local(
                x, expert_ids, expert_weights, layer_id, expert_load
            )
        raise RuntimeError(
            "ep_size > 1 requires deep_ep for dispatch/combine, but deep_ep is "
            "not importable and no DeepEPWrapper is initialized"
        )

    __call__ = forward

    def _expert_load_counts(self, expert_load, device):
        counts = expert_load.expert_token_counts
        if (
            counts.ndim != 1
            or counts.numel() != self.num_local_experts
            or counts.dtype != torch.int32
            or not counts.is_contiguous()
            or counts.device != device
        ):
            raise ValueError(
                "PPU expert counts must be contiguous int32 [local_experts] "
                "on the receiving device"
            )
        return counts

    def _record_expert_load(self, local_ids, expert_load):
        """Accumulate received assignments without the CUDA-only native op."""
        if expert_load is None:
            return
        counts = self._expert_load_counts(expert_load, local_ids.device)
        if local_ids.dtype not in (torch.int32, torch.int64) or local_ids.ndim not in (
            1,
            2,
        ):
            raise ValueError(
                "PPU expert IDs must be int32/int64 with one or two dimensions"
            )
        ids = local_ids.reshape(-1)
        experts = torch.arange(counts.numel(), dtype=ids.dtype, device=ids.device)
        # Equality excludes -1 and out-of-range sentinels. Fixed-size chunks
        # bound scratch memory and avoid data-dependent indexing or CPU reads.
        for start in range(0, ids.numel(), 4096):
            matches = ids[start : start + 4096, None] == experts[None, :]
            counts.add_(matches.sum(dim=0).to(counts.dtype))

    def _get_mega_module(self):
        if self._mega_module is None:
            try:
                module = importlib.import_module("deep_gemm")
            except Exception as error:
                raise RuntimeError(
                    "PPU890 MegaMoE requires a compatible DeepGEMM-for-sail build"
                ) from error
            required = (
                "get_symm_buffer_for_mega_moe",
                "transform_weights_for_mega_moe",
                "mega_moe_pre_dispatch",
                "fp4_fp4_mega_moe",
            )
            missing = [
                name for name in required if not callable(getattr(module, name, None))
            ]
            if missing:
                raise RuntimeError(
                    "PPU890 MegaMoE requires DeepGEMM APIs: " + ", ".join(missing)
                )
            self._mega_module = module
        return self._mega_module

    def _validate_megamoe_input(self, x):
        if not x.is_cuda or "M890" not in torch.cuda.get_device_name(x.device).upper():
            raise RuntimeError("MegaMoE W4A4 requires a PPU890 device")
        if x.dtype != torch.bfloat16 or x.ndim != 2 or x.shape[1] != self.hidden_size:
            raise ValueError("MegaMoE input must be BF16 [tokens, hidden_size]")
        if self.hidden_size % 512 or self.inter_dim % 128:
            raise ValueError(
                "MegaMoE requires hidden divisible by 512 and intermediate by 128"
            )
        if self._mega_device is not None and self._mega_device != x.device:
            raise RuntimeError("a MegaMoE workspace cannot move between devices")

    def _ensure_mega_buffer(self, x):
        return self._ensure_mega_capacity(x.device, x.shape[0])

    def prepare_full_cp8(self, device, rows):
        """Prepare once for the uniform host-padded CP8 role world."""
        import torch.distributed as dist

        if (
            self.backend != "megamoe"
            or self.ep_size != 8
            or device.type != "cuda"
            or not dist.is_initialized()
            or self.group is not dist.group.WORLD
        ):
            return None
        return self._ensure_mega_capacity(device, rows, uniform_rows=True)

    def _ensure_mega_capacity(self, device, row_count, *, uniform_rows=False):
        import torch.distributed as dist

        group = self.group
        if group is None:
            if not dist.is_initialized():
                raise RuntimeError("MegaMoE requires an initialized EP process group")
            group = dist.group.WORLD
        if group.size() != self.ep_size or group.rank() != self.ep_rank:
            raise RuntimeError("MegaMoE group must match this model's EP size and rank")
        capturing = torch.cuda.is_current_stream_capturing()
        if capturing:
            if self._mega_buffer is None or row_count > self._mega_capacity:
                raise RuntimeError(
                    "MegaMoE requires eager warmup with sufficient capacity before capture"
                )
            return self._mega_buffer
        if not uniform_rows:
            # A rank-local overflow check would deadlock CP/DP peers on growth.
            rows = torch.tensor([row_count], dtype=torch.int64, device=device)
            if self.ep_size > 1:
                dist.all_reduce(rows, op=dist.ReduceOp.MAX, group=group)
            row_count = int(rows.item())
        required = max(128, 1 << (max(1, row_count) - 1).bit_length())
        if self._mega_buffer is None or required > self._mega_capacity:
            # A growing symmetric buffer must remain live until all peers
            # have drained its previous users. The old reduction already
            # provides that rendezvous for nonuniform calls.
            self.close(group=group if uniform_rows else None)
            self._mega_device = device
            self._mega_buffer = self._get_mega_module().get_symm_buffer_for_mega_moe(
                group,
                self.num_physical_experts,
                required,
                COMM_SLOTS,
                self.hidden_size,
                self.inter_dim,
                num_shared_experts=0,
                mma_type="fp4xfp4",
                activation="swiglu",
            )
            self._mega_capacity = required
            if self._mega_recv_stats is None:
                self._mega_recv_stats = torch.zeros(
                    self.num_local_experts, dtype=torch.int32, device=device
                )
        return self._mega_buffer

    def prepare_shared_graph(self, x, layer_ids):
        """Graph-external allocation/weight preparation; never run in capture."""
        if self.backend != "megamoe" or self.precision != "w4a4":
            raise ValueError("shared Graph requires MegaMoE/W4A4")
        if self._fake_balance_config and self._fake_balance_config[0]:
            raise ValueError("shared Graph fake balance belongs to N9")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("shared Graph resources must be prepared outside capture")
        self._validate_megamoe_input(x)
        self._ensure_mega_buffer(x)
        for layer_id in layer_ids:
            self._processed_weights(layer_id, x.device)

    def forward_shared_graph(
        self,
        x,
        expert_ids,
        expert_weights,
        *,
        layer_id,
        expert_load=None,
        fused_expert_load=False,
    ):
        """Fixed workspace path for capture AND graph-safe eager reference."""
        if (self.backend, self.precision) != ("megamoe", "w4a4"):
            raise ValueError("shared Graph requires MegaMoE/W4A4")
        if (layer_id, x.device) not in self._processed or self._mega_buffer is None:
            raise RuntimeError("shared Graph weights/workspace were not prepared")
        if x.shape[0] > self._mega_capacity:
            raise ValueError("shared Graph exceeds its fixed MegaMoE capacity")
        expert_ids = self._map_expert_ids(expert_ids, expert_weights, layer_id)
        return self._forward_megamoe(
            x, expert_ids, expert_weights, layer_id, expert_load,
            prepared_buffer=self._mega_buffer,
            fused_expert_load=fused_expert_load,
        )

    def _forward_megamoe(
        self,
        x,
        expert_ids,
        expert_weights,
        layer_id,
        expert_load=None,
        *,
        prepared_buffer=None,
        fused_expert_load=False,
    ):
        self._validate_megamoe_input(x)
        layer_id = self._resolve_layer_id(layer_id)
        w13, s13, w2, s2, _, _ = self._processed_weights(layer_id, x.device)
        ids, weights = pad_topk_for_dispatch(
            expert_ids.to(torch.int64),
            expert_weights.to(torch.float32),
            slots=COMM_SLOTS,
        )
        if prepared_buffer is None:
            buffer = self._ensure_mega_buffer(x)
        else:
            if prepared_buffer is not self._mega_buffer or x.shape[0] > self._mega_capacity:
                raise ValueError("prepared MegaMoE capacity does not cover this forward")
            buffer = prepared_buffer
        mega = self._get_mega_module()
        counts = (
            self._expert_load_counts(expert_load, x.device)
            if expert_load is not None
            else None
        )
        previous_counts = self._mega_recv_stats.clone() if counts is not None else None
        mega.mega_moe_pre_dispatch(
            x.contiguous(), ids.contiguous(), weights.contiguous(), buffer
        )
        y = torch.empty_like(x, dtype=torch.bfloat16)
        # Complete native pipeline includes dispatch, both GEMMs and weighted
        # combine. Do not run DeepEP or apply split-W4A8's pre-quant weighting.
        mega.fp4_fp4_mega_moe(
            y,
            (w13, s13),
            (w2, s2),
            buffer,
            cumulative_local_expert_recv_stats=self._mega_recv_stats,
            recipe=(1, 1, 32),
            activation="swiglu",
            activation_clamp=self.swiglu_limit if self.swiglu_limit > 0 else None,
            fast_math=True,
        )
        if counts is not None:
            # The fused dispatch exposes destination counts rather than IDs.
            if fused_expert_load and importlib.util.find_spec("triton") is not None:
                kernels = _get_triton_kernels()
                block = 128
                grid = (kernels.triton.cdiv(counts.numel(), block),)
                kernels.accumulate_mega_expert_load[grid](
                    counts,
                    self._mega_recv_stats,
                    previous_counts,
                    EXPERTS=counts.numel(),
                    BLOCK=block,
                    num_warps=4,
                )
            else:
                counts.add_(self._mega_recv_stats - previous_counts)
        self.mega_moe_calls += 1
        return y

    def close(self, *, group=None):
        """Release this model's workspace after coordinated rank drain."""
        if self._mega_buffer is not None:
            torch.cuda.synchronize(device=self._mega_device)
            if group is not None:
                torch.distributed.barrier(group=group, device_ids=[self._mega_device.index])
            self._mega_buffer.destroy()
            self._mega_buffer = None
            self._mega_capacity = 0

    @classmethod
    def close_mega_resources(cls):
        """Native shutdown calls this before destroying process groups."""
        with cls._BUFFER_LOCK:
            engines = list(cls._MEGA_ENGINES)
        for engine in engines:
            engine.close()

    def _offline_single_process(self) -> bool:
        """True when no process group is available: the offline driver mode.

        Without an explicit group AND without an initialized
        ``torch.distributed`` there is nothing to dispatch over, so the
        forward runs this rank's LOCAL experts through the reference oracle
        (rank-local semantics — same as ``_forward_reference_tokens``).
        """
        if self.group is not None:
            return False
        return not (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        )

    # -- DeepEP path ------------------------------------------------------

    def _use_deepep_dispatch(self) -> bool:
        """Whether the DeepEP dispatch/combine collectives should run.

        ``ep_size > 1`` dispatches when deep_ep is importable AND a process
        group exists (the engine, or the multi-process offline drivers); an
        initialized rtp DeepEPWrapper also qualifies. Single-rank instances
        use DeepEP only when the wrapper is already initialized; otherwise
        the LOCAL path runs so CPU/diagnostic execution works without
        wheels. The offline single-process mode (no group at all) never
        dispatches — it cannot (wave2-fix: the buffer requires an
        initialized torch.distributed).

        An initialized DeepEPWrapper implies deep_ep is importable (its
        buffer construction requires it), so when ``deep_ep`` is absent the
        wrapper probe — whose import chain is heavy and raises on hosts
        without the rtp C++ ops — is skipped entirely.
        """
        if importlib.util.find_spec("deep_ep") is None:
            return False
        if self.ep_size > 1:
            if self._offline_single_process():
                return False
            return True
        return _deepep_wrapper_initialized()

    def _get_deepep_buffer(self):
        """Lazy deep_ep.Buffer singleton.

        Prefers the rtp ``DeepEPWrapper`` when the engine already initialized
        it in normal mode; otherwise creates a direct ``deep_ep.Buffer``
        whose sizing follows sglang-for-sail
        ``srt/layers/moe/token_dispatcher/deepep.py:257-273`` (hint-based
        nvl/rdma sizes over the dispatch + combine configs, normal mode).
        """
        with PPUMoE._BUFFER_LOCK:
            try:
                from rtp_llm.models_py.distributed.deepep_wrapper import (
                    DeepEPMode,
                    DeepEPWrapper,
                )

                if (
                    DeepEPWrapper.is_initialized()
                    and DeepEPWrapper._instance is not None
                ):
                    if DeepEPWrapper._instance.mode == DeepEPMode.NORMAL:
                        return DeepEPWrapper._instance.buffer
            except Exception:
                # The wrapper import chain can be heavy and raise on hosts
                # without the rtp C++ ops; cache the failure (failed imports
                # are not cached by Python) and fall through to the direct
                # buffer below.
                global _DEEPEP_WRAPPER_IMPORT_FAILED
                _DEEPEP_WRAPPER_IMPORT_FAILED = True
            key = (self.ep_size, self.hidden_size)
            cached = PPUMoE._BUFFER_CACHE.get(key)
            if cached is not None:
                return cached
            import deep_ep

            group = self.group
            if group is None:
                if (
                    not torch.distributed.is_available()
                    or not torch.distributed.is_initialized()
                ):
                    raise RuntimeError(
                        "torch.distributed must be initialized (or an explicit "
                        "group passed to PPUMoE) before creating a deep_ep buffer"
                    )
                group = torch.distributed.group.WORLD
            buffer_cls = deep_ep.Buffer
            # sglang-for-sail deepep.py:257-273 — normal-mode buffer sizing.
            hidden_bytes = self.hidden_size * 2  # bf16 param bytes
            num_nvl_bytes, num_rdma_bytes = 0, 0
            for config in (
                buffer_cls.get_dispatch_config(group.size()),
                buffer_cls.get_combine_config(group.size()),
            ):
                num_nvl_bytes = max(
                    config.get_nvl_buffer_size_hint(hidden_bytes, group.size()),
                    num_nvl_bytes,
                )
                num_rdma_bytes = max(
                    config.get_rdma_buffer_size_hint(hidden_bytes, group.size()),
                    num_rdma_bytes,
                )
            # sglang normal mode uses DeepEPConfig num_sms (default 24) QPs.
            buffer = buffer_cls(
                group,
                num_nvl_bytes,
                num_rdma_bytes,
                low_latency_mode=False,
                num_qps_per_rank=24,
            )
            PPUMoE._BUFFER_CACHE[key] = buffer
            return buffer

    def _forward_deepep(
        self,
        x: torch.Tensor,
        expert_ids: torch.Tensor,
        expert_weights: torch.Tensor,
        layer_id: Optional[int],
        expert_load=None,
    ) -> torch.Tensor:
        """DeepEP normal dispatch -> local compute -> combine (rtp pattern)."""
        buffer = self._get_deepep_buffer()
        ids = expert_ids.to(torch.int64)
        weights = expert_weights.to(torch.float32)
        # Pad topk 6 -> 8 slots (-1 / 0.0).
        ids_p, weights_p = pad_topk_for_dispatch(ids, weights, slots=COMM_SLOTS)

        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            _,
        ) = buffer.get_dispatch_layout(ids_p, self.num_physical_experts)
        (
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert,
            handle,
            _,
        ) = buffer.dispatch(
            x,
            None,
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            is_token_in_rank,
            num_tokens_per_expert,
            ids_p,
            weights_p,
            # PPU: expert_alignment=1 for compatibility (sglang deepep.py:813).
            expert_alignment=1,
        )
        # DeepEP has converted GLOBAL source IDs to destination LOCAL IDs.
        # Record once even on idle source ranks, which may receive real work.
        self._record_expert_load(recv_topk_idx, expert_load)
        m = recv_x.size(0)
        if m > 0:
            # recv ids are LOCAL ([-1] for tokens not destined here); keep -1.
            local_ids = recv_topk_idx.to(torch.int64).contiguous()
            y_local = self._local_compute(
                recv_x.contiguous(),
                local_ids,
                recv_topk_weights.contiguous(),
                num_recv_tokens_per_expert,
                layer_id,
            )
        else:
            y_local = torch.zeros(
                (m, self.hidden_size), dtype=x.dtype, device=recv_x.device
            )
        # Local compute already applies weights, before A2 quantization for
        # W4A8 or in gather for W4A4. Combine only sums across ranks.
        y_combined, _, _ = buffer.combine(y_local.to(x.dtype), handle)
        return y_combined

    # -- local compute ----------------------------------------------------

    def _local_compute(
        self,
        recv_x: torch.Tensor,
        local_ids: torch.Tensor,
        recv_weights: torch.Tensor,
        num_recv_tokens_per_expert,
        layer_id: Optional[int],
    ) -> torch.Tensor:
        """Per-expert compute on dispatched recv tokens (BF16 [M, hidden] -> [M, hidden])."""
        if recv_x.size(0) == 0:
            return torch.zeros(
                (0, self.hidden_size), dtype=recv_x.dtype, device=recv_x.device
            )
        if self._use_grouped_gemm(recv_x):
            return self._local_compute_grouped(
                recv_x, local_ids, recv_weights, num_recv_tokens_per_expert, layer_id
            )
        # CPU / no deep_gemm reference path: shift local -> global ids.
        global_ids = torch.where(
            local_ids == -1,
            local_ids,
            local_ids + self.local_expert_start,
        )
        return self._forward_reference_tokens(
            recv_x, global_ids, recv_weights.to(torch.float32), layer_id
        )

    def _use_grouped_gemm(self, x: torch.Tensor) -> bool:
        if not x.is_cuda:
            return False
        if self._grouped_path is None:
            # Capability check (cached); the device is re-checked per call.
            if importlib.util.find_spec("deep_gemm") is None:
                self._grouped_path = False
            elif importlib.util.find_spec("triton") is None:
                self._grouped_path = False
            else:
                # downcast_to_mxfp4 needs an even number of group-32 scales
                # per row for the uint16 packing: the quantized dims must be
                # multiples of 64. (The FP4xFP4 GEMM itself has no group_size
                # divisibility requirement — verified for V4.1 Flash shapes.)
                alignment = 128 if self.precision == "w4a8" else 64
                self._grouped_path = (
                    self.hidden_size % alignment == 0
                    and self.inter_dim % alignment == 0
                )
        if self.precision == "w4a8" and not self._grouped_path:
            raise RuntimeError(
                "PPU W4A8 requires DeepGEMM-for-sail, Triton and hidden/inter "
                "dimensions divisible by 128; use "
                "RTP_V41_PPU_MOE_PRECISION=w4a4 for the legacy device path"
            )
        return self._grouped_path

    def _local_compute_grouped(
        self,
        recv_x: torch.Tensor,
        local_ids: torch.Tensor,
        recv_weights: torch.Tensor,
        num_recv_tokens_per_expert,
        layer_id: Optional[int],
    ) -> torch.Tensor:
        """DeepEP's expert-grouped rows through two selected-precision GEMMs.

        Retains the sglang-for-sail scatter/gather implementation (512b03d,
        Apache-2.0). BF16 dispatch is common to both precisions. W4A8 uses
        native per-row E4M3 x MXFP4 GEMMs with an independent inverse weight
        scale per projection. The explicit W4A4 override retains its original
        MXFP4 activation casts and post-GEMM2 routing-weight application.
        """
        import deep_gemm

        layer_id = self._resolve_layer_id(layer_id)
        counts = self._local_expert_counts(num_recv_tokens_per_expert)
        all_tokens = int(sum(int(c) for c in counts))
        m = recv_x.size(0)
        hidden, inter = self.hidden_size, self.inter_dim
        if all_tokens <= 0:
            # sglang deep_gemm.py:708 zero-token short-circuit.
            return torch.zeros((m, hidden), dtype=recv_x.dtype, device=recv_x.device)
        w13_u, w13_sp, w2_u, w2_sp, inv13, inv2 = self._processed_weights(
            layer_id, recv_x.device
        )
        device = recv_x.device
        m_rows = torch.tensor(
            [int(c) for c in counts], dtype=torch.int32, device=device
        )
        input_tensor = torch.empty(
            (all_tokens, hidden), dtype=recv_x.dtype, device=device
        )
        m_indices = torch.empty((all_tokens,), dtype=torch.int32, device=device)
        local_ids_i32 = local_ids.to(torch.int32)
        output_index = torch.empty_like(local_ids_i32)
        expert_start_loc = torch.empty_like(m_rows)
        ep_scatter_sail(
            recv_x,
            local_ids_i32,
            m_rows,
            expert_start_loc,
            input_tensor,
            m_indices,
            output_index,
            block_e=1,
        )
        # Quantize already-grouped rows; DeepEP continues to dispatch BF16.
        if self.precision == "w4a8":
            a_q, a_s = quantize_per_row_fp8(input_tensor)
        else:
            a_q, a_s = downcast_to_mxfp4(input_tensor, axis=1)
        gateup_output = torch.empty(
            (all_tokens, 2 * inter), dtype=torch.bfloat16, device=device
        )
        if self.precision == "w4a8":
            gemm = deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_nopad
            gemm(
                (a_q, a_s),
                (w13_u, w13_sp),
                gateup_output,
                m_indices,
                m_rows,
                inv_scale=inv13,
            )
            route_weights = grouped_route_weights(
                local_ids_i32, recv_weights, output_index, all_tokens
            )
            a2_q, a2_s = routed_swiglu_fp8(
                gateup_output, route_weights, self.swiglu_limit
            )
            # GEMM2 consumes already-weighted activations. Gather must not
            # multiply the router weights a second time.
            gather_weights = torch.ones_like(recv_weights, dtype=torch.float32)
        else:
            _m_grouped_gemm_f4f4_nopad(
                deep_gemm, (a_q, a_s), (w13_u, w13_sp), gateup_output, m_indices, m_rows
            )
            _apply_swiglu_limit(gateup_output, self.swiglu_limit)
            gate, up = gateup_output.chunk(2, dim=-1)
            down_input = (F.silu(gate.float()) * up.float()).to(torch.bfloat16)
            a2_q, a2_s = downcast_to_mxfp4(down_input, axis=1)
            gather_weights = recv_weights.to(torch.float32)
        down_output = torch.empty(
            (all_tokens, hidden), dtype=torch.bfloat16, device=device
        )
        if self.precision == "w4a8":
            gemm(
                (a2_q, a2_s),
                (w2_u, w2_sp),
                down_output,
                m_indices,
                m_rows,
                inv_scale=inv2,
            )
        else:
            _m_grouped_gemm_f4f4_nopad(
                deep_gemm, (a2_q, a2_s), (w2_u, w2_sp), down_output, m_indices, m_rows
            )
        gather_out = torch.empty((m, hidden), dtype=torch.bfloat16, device=device)
        ep_gather(
            down_output,
            local_ids_i32,
            gather_weights,
            output_index,
            gather_out,
        )
        self.grouped_forward_calls += 1
        self.grouped_gemm_calls += 2
        return gather_out

    def _local_expert_counts(self, num_recv_tokens_per_expert) -> List[int]:
        counts = [int(c) for c in num_recv_tokens_per_expert]
        if len(counts) > self.num_local_experts:
            # Some builds return one count per global expert; keep the local
            # shard's slice.
            counts = counts[self.local_expert_start : self.local_expert_end]
        if len(counts) != self.num_local_experts:
            raise RuntimeError(
                f"expected {self.num_local_experts} per-expert recv counts, "
                f"got {len(counts)}"
            )
        return counts

    # -- reference / LOCAL paths -------------------------------------------

    def _forward_local(
        self,
        x: torch.Tensor,
        expert_ids: torch.Tensor,
        expert_weights: torch.Tensor,
        layer_id: Optional[int],
        expert_load=None,
    ) -> torch.Tensor:
        """LOCAL path: ep_size == 1, no dispatch. Weights applied explicitly."""
        ids = expert_ids.to(torch.int64)
        weights = expert_weights.to(torch.float32)
        if expert_load is not None:
            local_ids = torch.where(
                (ids >= self.local_expert_start) & (ids < self.local_expert_end),
                ids - self.local_expert_start,
                torch.full_like(ids, -1),
            )
            self._record_expert_load(local_ids, expert_load)
        return self._forward_reference_tokens(x, ids, weights, layer_id)

    def _forward_reference_tokens(
        self,
        x: torch.Tensor,
        global_ids: torch.Tensor,
        weights: torch.Tensor,
        layer_id: Optional[int],
    ) -> torch.Tensor:
        """Per-expert loop grouped by expert (rtp local_loop.py oracle).

        ``global_ids`` are GLOBAL expert ids with ``-1`` padding; only
        experts in ``[local_expert_start, local_expert_end)`` are computed and
        only those experts' FP4 weights are dequantized.
        """
        t = x.size(0)
        y = torch.zeros((t, self.hidden_size), dtype=torch.float32, device=x.device)
        if t == 0:
            return y.to(torch.bfloat16)
        w13_w, w13_s, w2_w, w2_s = self._layer_weights(layer_id)
        for e in range(self.local_expert_start, self.local_expert_end):
            idx, top = torch.where(global_ids == e)
            if idx.numel() == 0:
                continue
            out = expert_forward_reference(
                x[idx],
                w13_w[e - self.local_expert_start],
                w13_s[e - self.local_expert_start],
                w2_w[e - self.local_expert_start],
                w2_s[e - self.local_expert_start],
                self.swiglu_limit,
            )
            y.index_add_(
                0, idx, (out.float() * weights[idx, top, None]).to(torch.float32)
            )
        return y.to(torch.bfloat16)


def _float_scales_to_ue8m0_bytes(scales: torch.Tensor) -> torch.Tensor:
    """Convert floating-point power-of-two scales back to UE8M0 bytes."""
    values = scales.to(torch.float32)
    exponents = torch.round(torch.log2(values)).to(torch.int32) + 127
    if ((exponents < 0) | (exponents > 255)).any():
        raise ValueError("float scales are not representable as UE8M0 bytes")
    return exponents.to(torch.uint8)


def _ue8m0_scale_bytes(scale: torch.Tensor) -> torch.Tensor:
    """UE8M0 scale tensor as uint8 bytes (e8m0fnu is handled natively)."""
    if scale.dtype in (torch.float8_e8m0fnu, torch.uint8):
        return scale
    if scale.dtype == torch.int8:
        return scale.contiguous().view(torch.uint8)
    return _float_scales_to_ue8m0_bytes(scale)


def _packed_weight_uint8(weight: torch.Tensor) -> torch.Tensor:
    """Packed FP4 weight as uint8 (int8 bytes are reinterpreted, not cast)."""
    if weight.dtype == torch.uint8:
        return weight
    if weight.dtype == torch.int8:
        return weight.contiguous().view(torch.uint8)
    raise TypeError(f"packed FP4 expert weight must be int8/uint8, got {weight.dtype}")


def _m_grouped_gemm_f4f4_nopad(deep_gemm, lhs, rhs, out, m_indices, m_rows) -> None:
    """``deep_gemm.m_grouped_gemm_fp4_fp4_bf16_nt_nopad`` call (sglang-for-sail).

    Follows sglang-for-sail
    ``srt/layers/deep_gemm_wrapper/entrypoint.py::grouped_gemm_nt_f4f4bf16_
    nopad`` (512b03d, Apache-2.0), which forwards ``(lhs, rhs), bias, out,
    m_indices, m_rows`` to the wheel (bias ``None`` — V4.1 has no expert
    bias; the tuner ``configs`` argument defaults to ``None`` on this wheel,
    signature verified empirically on the P machine). ``lhs``/``rhs`` are
    ``(packed e2m1 uint8, uint16 packed-UE8M0 scale)`` pairs; the SFA/SFB
    layouts are produced by :func:`downcast_to_mxfp4` /
    :func:`preprocess_mxfp4_f4f4_scales`.
    """
    deep_gemm.m_grouped_gemm_fp4_fp4_bf16_nt_nopad(
        lhs, rhs, None, out, m_indices, m_rows
    )


# ---------------------------------------------------------------------------
# Shared expert
# ---------------------------------------------------------------------------


def shared_expert_forward(
    x: torch.Tensor,
    w13: PPUResidentBF16Linear,
    w2: PPUResidentBF16Linear,
    swiglu_limit: float = SWIGLU_LIMIT,
    *,
    fused_activation: bool = False,
    prefill_exact: bool = False,
) -> torch.Tensor:
    """Shared-expert SwiGLU forward on BF16-resident weights.

    Same clamp convention as the routed experts (sglang ``_apply_swiglu_limit``
    / rtp ``silu_mul_split``): clamp ``up`` to ``[-limit, limit]`` and ``gate``
    to ``[-inf, limit]`` BEFORE the SiLU, then ``silu(gate) * up`` and the w2
    projection. Returns BF16.
    """
    gate_up = w13(x)
    if (
        prefill_exact
        and gate_up.is_cuda
        and gate_up.dtype == torch.bfloat16
        and gate_up.ndim == 2
        and gate_up.is_contiguous()
        and importlib.util.find_spec("triton") is not None
    ):
        from rtp_llm.models_py.modules.dsv41.moe._silu_mul_bf16_triton import (
            silu_mul_split_bf16_torch_exact,
        )

        return w2(silu_mul_split_bf16_torch_exact(gate_up, clamp_limit=swiglu_limit))
    if (
        fused_activation
        and gate_up.is_cuda
        and gate_up.dtype == torch.bfloat16
        and gate_up.ndim == 2
        and gate_up.is_contiguous()
        and importlib.util.find_spec("triton") is not None
    ):
        from rtp_llm.models_py.modules.dsv41.moe._silu_mul_bf16_triton import (
            silu_mul_split_bf16,
        )

        return w2(silu_mul_split_bf16(gate_up, clamp_limit=swiglu_limit))
    gate, up = gate_up.chunk(2, dim=-1)
    if swiglu_limit and swiglu_limit > 0:
        up = up.clamp(min=-float(swiglu_limit), max=float(swiglu_limit))
        gate = gate.clamp(max=float(swiglu_limit))
    hidden = F.silu(gate.float()) * up.float()
    return w2(hidden.to(torch.bfloat16))


def combine_routed_and_shared(
    routed: torch.Tensor, shared: torch.Tensor
) -> torch.Tensor:
    """Combine routed MoE output with the shared-expert output.

    Both inputs already carry their router weights / scaling factors; the
    combination is a plain add (rtp ``combine_routed_and_shared`` fallback
    path). Returns the routed tensor's dtype.
    """
    return routed + shared.to(routed.dtype)
