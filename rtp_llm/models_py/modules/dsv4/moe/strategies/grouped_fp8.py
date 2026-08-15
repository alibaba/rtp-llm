"""GroupedFP8Strategy: single-card DeepGEMM FP8-act x FP8-weight MoE for Hopper.

Every other routed-expert strategy in this package reaches DeepGEMM through an
FP8xFP4 kernel, and those kernels exist only on SM100. That makes the routed
experts the one part of V4-Flash that cannot run on an H20, even though the
checkpoint's attention, indexer, compressor and shared-expert weights are
already plain FP8 block-quantised tensors this hardware runs natively.

This strategy closes that gap by consuming routed experts in the same FP8
block-quantised layout, via ``deepgemm_wrapper.m_grouped_fp8_gemm_nt_contiguous``
(the wrapper because DeepGEMM renamed this kernel between 1.x and 2.x, and the
SM90 FP8 impl only ships in 2.x). It
expects a checkpoint whose experts were rewritten from FP4 to FP8 by
``dsv4_fp8/convert_fp4_experts_to_fp8.py``; that rewrite is value-exact, so this
path computes on the same numbers the FP4 path would, just with a wider weight
container and a coarser (128x128 rather than group-32) scale granularity.

Structurally it mirrors :class:`GroupedFP4Strategy` — quant once pre-permute,
Triton ``ep_scatter``, grouped gate/up GEMM, SwiGLU, grouped down GEMM, Triton
``ep_gather`` with router-weight reduce. Two things differ, both because SM90
DeepGEMM scales are FP32 rather than packed UE8M0:

  * scatter/quant run with ``scale_ue8m0=False``;
  * the fused ``silu_mul_fp8_quant_packed`` kernel only emits packed UE8M0, so
    SwiGLU runs as the explicit bf16 sequence that kernel was written to match
    followed by an ordinary per-token-group quant.

CUDA graph capture goes through the masked layout (``_local_experts_masked``). The
contiguous path cannot be captured: its buffer is ``sum_e align(count_e, 128)``, which
is device-resident, and reading it back is the per-layer ``num_recv.cpu()`` sync. The
masked layout fixes the shapes at capture time and leaves the per-expert row count on
the device in ``masked_m``, so the work still follows the routing. Set
``DSV4_MOE_MASKED=1`` to use it in eager decode as well; capture always uses it.

EP > 1
------
On SM90 this is also the only strategy that can serve ``ep_size > 1``: Mega is
gated on SM100 (``mega_buf._mega_moe_unavailable_reason``) and DeepEP delegates
its local compute to ``LocalLoopStrategy``, which is FP4-only. That matters
because EP is not optional for V4-Flash on a 143 GiB card — routed experts are
declared with ``identity``/``stack_`` and are NOT TP-sharded, so at ``ep_size=1``
every rank holds all 256 experts (256 x 25.2 MiB x 43 layers = 277 GiB) and the
load OOMs partway through. At ``ep_size=4`` the loader hands each rank 64
experts (~69 GiB) and it fits.

The combine here is all-gather + reduce-scatter over ``Group.TP`` rather than
DeepEP's all-to-all. Each rank owns a disjoint expert range but only its own
token shard (``x`` is ``[N_local, D]``), so tokens have to move: gather the full
token set, let every rank apply just its local experts to all of them, then
reduce-scatter so each rank gets the total over ALL experts for its own shard.
FLOPs are not duplicated — a token's 6 experts are spread over the ranks, so
each rank does ~1.5 expert-applications per token and the sum across ranks is
exactly the 6 required. Only the tokens are duplicated on the wire. This uses
plain NCCL collectives, so unlike DeepEP it needs no NVSHMEM and no
``init_deepep_wrapper``.
"""

from __future__ import annotations

import functools
import os
from typing import Dict

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
    m_grouped_fp8_gemm_nt_contiguous,
    m_grouped_fp8_gemm_nt_masked,
)
from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
from rtp_llm.models_py.triton_kernels.moe.ep_kernels import (
    ep_gather,
    ep_scatter,
    recompute_topk_ids_sum_expert_count,
)
from rtp_llm.models_py.utils.math import align

from .base import MoeCfg, RoutedExpertsStrategy, register_strategy
from ...quant_layouts import FP8_BLOCK

# Set to 1 to verify every rank enters each MoE layer with the same token count.
# Off by default because the check needs a device->host sync per layer. Equal
# per-rank token counts are a CP invariant, not an assumption added here:
# ``cp.cp_all_gather_full`` already uses plain ``all_gather`` on the local
# hidden states, which requires identical shapes across ranks.
_EP_CHECK_SIZES_ENV = "DSV4_EP_CHECK_SIZES"

# ep_scatter needs m_indices.shape[0] % BLOCK_E == 0 (BLOCK_E=128), and
# DeepGEMM's contiguous layout needs each expert's M aligned to
# get_m_alignment_for_contiguous_layout() (128). One constant covers both.
_GROUPED_ALIGNMENT = 128

_HOPPER_MAJOR = 9

# Use the masked layout in eager decode too, not only under capture. Default off: the
# capture path has to take it (see _local_experts_masked) but eager is the configuration
# every banked number was measured under, so it stays byte-identical unless asked.
_MASKED_ENV = "DSV4_MOE_MASKED"

# The masked GEMM's M tile: it writes whole tiles of this many rows, clipped to the
# tensor's own max_m (measured on H20, DeepGEMM 2.2.0 -- with masked_m=1 it writes rows 0..63 for
# max_m in {64, 128}, and 0..max_m-1 below that). Everything after the first GEMM can
# therefore work on align(N, _MASKED_TILE) rows per group instead of max_m.
_MASKED_TILE = 64

# Above this many tokens per MoE call the masked layout is refused, because its buffer is
# E * align(N, 128) rows: decode's N (batch x ep, <=256 here) gives 8-16k rows, while a
# 16k-token prefill chunk would ask for E * 16384 = a million rows. The contiguous layout
# is the right one there -- at prefill N its per-expert 128-row padding is a few percent,
# not the 64x it is at decode N.
_MASKED_MAX_N = 1024


# The fused swiglu+quant kernel below replaces ~8 kernels per MoE layer with 1 and is
# 6.3x faster on the decode shape (238.5 -> 37.8 us, measured), but it is NOT bit-identical
# to the sequence it replaces: 7623 of 8388608 quantised activation elements (0.09%) differ
# by exactly one fp8 e4m3 ULP, deterministically. The per-group scales are not
# bit-identical either, though the difference is smaller than it sounds and runs
# the other way: this kernel produces exactly max(absmax_fp32, eps)/448, the fp32
# definition, while the reference quantiser lands one fp32 ULP off it on roughly
# half the groups (measured over three input distributions in
# grouped_fp8_swiglu_fusion_test). So the fused path is the more faithful of the
# two, and the residual byte difference is that one-ULP scale plus the reference's
# rounding at the halfway boundary, which could not be matched from Triton because
# the reference is a CUDA op (matching torch's x*sigmoid(x) form of silu removed the
# rest). One fp8 ULP on an activation that was just quantised to fp8 is
# far below the model's numerical noise -- and this engine is not run-to-run reproducible at
# token granularity anyway -- but it is a real difference, so it gets its own switch.
_FUSED_SWIGLU_ENV = "DSV4_MOE_FUSED_SWIGLU"

# Sentinel ROWS_PER_E for the E == 1 (contiguous) call: any constant >= M works,
# and holding it constant is what keeps the kernel out of a per-request JIT.
_ROWS_PER_E_FLAT = 1 << 30

# Replace the all-gather/reduce-scatter EP combine with DeepEP's low-latency
# dispatch/combine. Off by default: it needs an RDMA-capable DeepEP build on the
# library path, so a missing libmlx5 must not take down a run that never asked.
#
# This lives here rather than in DeepEPStrategy because the two are not the same
# integration. DeepEPStrategy wraps NORMAL-mode DeepEP and delegates its local
# compute to LocalLoopStrategy, which hardcodes FP4 storage -- unusable against an
# FP8 checkpoint. The low-latency kernels, by contrast, hand back exactly the
# layout this strategy's masked path already consumes: fp8 ``[E, M, D]`` with
# ``[E, M, D/128]`` fp32 scales column-major in the last two dims, and a
# ``recv_count`` that *is* ``masked_m``. So dispatch replaces ep_scatter + the
# input quant + three all-gathers, and combine replaces ep_gather + the
# reduce-scatter, with the two GEMMs and the fused swiglu untouched.
_DEEPEP_LL_ENV = "DSV4_MOE_DEEPEP_LL"

# Tokens per rank the LL buffer is sized for. The buffer is allocated once, before
# any forward, so this cannot be inferred from a batch; unset means "the smallest
# legal size", which already covers any decode batch up to _MASKED_TILE. A call
# with more tokens than the buffer holds falls back to the all-gather combine.
_DEEPEP_LL_MAX_TOKENS_ENV = "DSV4_MOE_LL_MAX_TOKENS"

# (Buffer, max_tokens_per_rank) for this process, or None. One buffer serves every
# layer: it is a communication scratchpad, not per-layer state.
_ll_buffer_state = None


def _masked_enabled() -> bool:
    return os.environ.get(_MASKED_ENV, "0").strip().lower() in ("1", "true", "on", "yes")


def _deepep_ll_enabled() -> bool:
    return os.environ.get(_DEEPEP_LL_ENV, "0").strip().lower() in ("1", "true", "on", "yes")


def _fused_swiglu_enabled() -> bool:
    return os.environ.get(_FUSED_SWIGLU_ENV, "1").strip().lower() not in ("0", "false", "off", "no")


@functools.cache
def _has_grouped_fp8_kernel() -> bool:
    """True iff the grouped FP8 routed-expert path should be used.

    ``DSV4_USE_GROUPED_FP8`` semantics mirror ``DSV4_USE_GROUPED_FP4``:
    unset/"auto" enables when the runtime supports it, "0" disables, "1"
    requests it but still requires the probe below to pass.
    """
    flag = os.environ.get("DSV4_USE_GROUPED_FP8", "auto").strip().lower()
    if flag in ("0", "false", "off", "no"):
        return False
    if not torch.cuda.is_available():
        return False
    # Probe the resolved impl, not a raw attribute: the wrapper is what maps
    # DeepGEMM's 1.x and 2.x spellings of this kernel onto one name, and it
    # leaves the impl None (rather than raising) when the installed DeepGEMM has
    # neither — which is exactly the case this predicate must detect.
    from rtp_llm.models_py.kernels.cuda import deepgemm_wrapper

    if deepgemm_wrapper._m_grouped_fp8_gemm_nt_contiguous_impl is None:
        return False
    # Gated to Hopper rather than "not SM100" on purpose: on SM100 the FP4
    # kernels are both available and faster, so this path would only ever be a
    # regression there.
    return torch.cuda.get_device_capability()[0] == _HOPPER_MAJOR


_ep_group_unavailable_reason: str = ""


def _ep_group():
    """The process group the EP combine runs over, or None if unavailable.

    "Unavailable" means one of two expected things: ``torch.distributed`` or the
    collective helper is not importable, or the group has not been initialised
    yet. Anything else -- a renamed group, a lookup that raises -- is a real
    configuration fault and propagates, because swallowing it produced a
    ``can_handle`` of False whose reason never reached the operator: the error
    ``select_strategy`` raised named only Mega's reason, and a ``Group.TP`` that
    could not be resolved looked like an unrelated failure.
    """
    global _ep_group_unavailable_reason
    try:
        import torch.distributed as dist

        from rtp_llm.models_py.distributed.collective_torch import Group, _get_group
    except (ImportError, ModuleNotFoundError) as exc:
        _ep_group_unavailable_reason = f"torch.distributed unavailable: {exc}"
        return None

    if not dist.is_initialized():
        _ep_group_unavailable_reason = "torch.distributed is not initialized"
        return None

    group = _get_group(Group.TP)
    if group is None:
        _ep_group_unavailable_reason = "collective_torch has no Group.TP"
    else:
        _ep_group_unavailable_reason = ""
    return group


def ep_group_unavailable_reason() -> str:
    """Why :func:`_ep_group` last returned None, for inclusion in error text."""
    return _ep_group_unavailable_reason


def _ep_group_size() -> int:
    """World size of the EP combine group, or 0 when there is no group."""
    group = _ep_group()
    if group is None:
        return 0
    import torch.distributed as dist

    return dist.get_world_size(group=group)


def _ll_max_tokens(ep: int) -> int:
    """Tokens per rank to size the LL buffer for.

    Dispatch's recv buffer is ``max_tokens * ep`` rows per local expert, and that
    total -- not ``max_tokens`` -- is what has to clear DeepGEMM's masked tile, so
    ``calc_low_latency_max_token_per_rank``'s round-up to 64 is one ep factor too
    conservative. Rounding to ``_MASKED_TILE / gcd(_MASKED_TILE, ep)`` instead keeps
    ``max_tokens * ep`` a multiple of the tile while making it as small as the tile
    allows: at ep=4 that is 16 rather than 64, so the recv buffer is 64 rows -- the
    fewest the GEMM can use -- instead of 256. Measured on 8x H20 at 8
    tokens/rank: 0.666 -> 0.609 ms per MoE layer, because everything between the two
    GEMMs (the swiglu, the quant, and 67 MB of buffer per layer) shrinks 4x while the
    GEMMs themselves compute the same tiles.
    """
    from math import gcd

    unit = _MASKED_TILE // gcd(_MASKED_TILE, ep)
    return align(max(int(os.environ.get(_DEEPEP_LL_MAX_TOKENS_ENV, "0") or 0), 1), unit)


def _ll_buffer(ep: int, dim: int, n_experts: int):
    """The process-wide DeepEP low-latency buffer, created on first use.

    Collective: every rank in the EP group must reach this together. Creation is
    driven from ``setup_weights`` so it lands during weight load rather than inside
    a CUDA graph capture, where the NVSHMEM handshake could not run.
    """
    global _ll_buffer_state
    if _ll_buffer_state is not None:
        return _ll_buffer_state
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            f"{_DEEPEP_LL_ENV}=1 but the low-latency buffer does not exist yet and "
            "this is a graph capture; its NVSHMEM setup cannot be captured. The "
            "buffer is normally built in setup_weights -- check that "
            f"{_DEEPEP_LL_ENV} was set before the model was loaded."
        )
    group = _ep_group()
    if group is None:
        raise RuntimeError(
            f"{_DEEPEP_LL_ENV}=1 needs an initialised Group.TP process group."
        )
    try:
        from deep_ep import Buffer
    except ImportError as exc:  # RDMA libs missing from the loader path
        raise RuntimeError(
            f"{_DEEPEP_LL_ENV}=1 but DeepEP is not importable ({exc}). Its extension "
            "links libmlx5/libibverbs; put the RDMA libs on LD_LIBRARY_PATH ahead of "
            "any older libibverbs (IBVERBS_PRIVATE_34 is required)."
        ) from exc

    max_tokens = _ll_max_tokens(ep)
    buf = Buffer(
        group,
        num_nvl_bytes=0,
        num_rdma_bytes=Buffer.get_low_latency_rdma_size_hint(
            max_tokens, dim, ep, n_experts
        ),
        low_latency_mode=True,
        # One QP per local expert is what the LL kernels index by.
        num_qps_per_rank=n_experts // ep,
        # The kernels' docstrings say IBGDA; on a single NVLink node there is no NIC
        # to go through and this makes them use NVLink peer stores instead. Verified
        # working on a single 8x H20 host (no IB fabric between the ranks).
        allow_nvlink_for_low_latency_mode=True,
        allow_mnnvl=False,
    )
    # The LL kernels need part of the buffer zeroed and never re-clean it, so this
    # has to happen once here, outside capture, before the first dispatch.
    buf.clean_low_latency_buffer(max_tokens, dim, n_experts)
    _ll_buffer_state = (buf, max_tokens)
    return _ll_buffer_state


def _all_gather_cat(tensor: torch.Tensor, ep: int, group) -> torch.Tensor:
    """Concatenate ``tensor`` from every rank along dim 0, rank-major.

    ``all_gather_into_tensor`` lays rank r's contribution at rows
    ``[r*n, (r+1)*n)``, which is exactly the chunking
    ``reduce_scatter_tensor`` reverses, so the two compose into an identity on
    the token axis without any index bookkeeping.
    """
    import torch.distributed as dist

    out = torch.empty(
        (tensor.shape[0] * ep, *tensor.shape[1:]),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    dist.all_gather_into_tensor(out, tensor, group=group)
    return out



# ---------------------------------------------------------------------------
# Fused clamp + SiLU + multiply + per-token-group FP8 quant, FP32 scale.
#
# The unfused sequence this replaces is, per MoE layer:
#   gate.clamp(max=L), up.clamp(-L, L), silu(gate.float()), * up.float(),
#   .to(bf16), .contiguous(), then sgl_per_token_group_quant_fp8
# -- around eight kernels and several fp32 temporaries of E*t_rows*inter
# elements each. The capture-mode profile charged that chain ~8 ms of the
# 53 ms step (2 clamps at 42.8 us/layer, silu 33 us, mul 25 us, plus copies),
# all of it moving 8-33 MB per buffer per layer to produce one bf16 result.
#
# ``moe/_silu_mul_fp8_quant_triton.py`` already fuses exactly this, with the
# same fp32-internal arithmetic and the same "round through bf16 before
# quantising" step -- but it emits a *packed UE8M0* scale, because that is what
# the SM100 FP4 path consumes. UE8M0 rounds the scale up to a power of two,
# which changes the quantised values, so it cannot be reused as-is on SM90
# where DeepGEMM takes FP32 scales. This is that kernel with the exponent
# rounding and the int32 packing removed, and eps matched to the
# ``sgl_per_token_group_quant_fp8(eps=1e-4)`` call it stands in for:
# ``scale = max(absmax, eps) / fp8_max``.
@triton.jit(do_not_specialize=["M"])
def _silu_mul_fp8_quant_fp32scale_kernel(
    input_ptr,          # [M, 2*inter] bf16, gate in [:inter], up in [inter:]
    output_q_ptr,       # [M, inter] fp8 e4m3fn
    output_scale_ptr,   # [M, inter/GROUP_SIZE] fp32, stride (1, M): M-major
    M,
    input_stride_e,     # gate_up.stride(0): the gap between groups
    input_stride_m,     # gate_up.stride(1)
    ROWS_PER_E: tl.constexpr,
    output_q_stride_m,
    output_scale_stride_k,
    clamp_limit,
    N: tl.constexpr,            # 2 * inter
    NUM_GROUPS: tl.constexpr,   # inter // GROUP_SIZE
    eps: tl.constexpr,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    HAS_CLAMP: tl.constexpr,
):
    N_2: tl.constexpr = N // 2

    pid_g = tl.program_id(0).to(tl.int64)   # which 128-wide quant group
    pid_m = tl.program_id(1).to(tl.int64)   # which BLOCK_M row tile
    m_offset = pid_m * BLOCK_M
    if m_offset >= M:
        return

    offs_m = tl.arange(0, BLOCK_M).to(tl.int64)
    offs_n = tl.arange(0, GROUP_SIZE)
    row_mask = (m_offset + offs_m) < M

    n_offset = pid_g * GROUP_SIZE
    # Flat row -> (group, row-within-group). The caller hands us a [E, ROWS_PER_E, 2*inter]
    # slice of a taller buffer, so the two dims cannot be merged into one stride and a
    # reshape would copy the whole thing -- which is the copy this kernel exists to avoid.
    rows = m_offset + offs_m
    row_e = rows // ROWS_PER_E
    row_j = rows % ROWS_PER_E
    row_base = row_e * input_stride_e + row_j * input_stride_m
    act_ptrs = input_ptr + row_base[:, None] + n_offset + offs_n[None, :]
    act_in = tl.load(act_ptrs, mask=row_mask[:, None], other=0.0)
    mul_in = tl.load(act_ptrs + N_2, mask=row_mask[:, None], other=0.0)

    act_f32 = act_in.to(tl.float32)
    mul_f32 = mul_in.to(tl.float32)
    # V4 SwiGLU clamp convention: gate upper-only, up symmetric. Clamping in fp32
    # after the widening is identical to the eager path's clamp in bf16 -- min/max
    # select an operand, and both operands are exactly representable in both types.
    if HAS_CLAMP:
        act_f32 = tl.minimum(act_f32, clamp_limit)
        mul_f32 = tl.clamp(mul_f32, -clamp_limit, clamp_limit)

    # x * sigmoid(x), not x / (1 + exp(-x)): torch's silu does the reciprocal first and
    # the two differ in the last bit, which occasionally survives the bf16 rounding below
    # and shows up as a 1-ULP difference in the quantised fp8 activation.
    y = (act_f32 * (1.0 / (1.0 + tl.exp(-act_f32)))) * mul_f32
    # Round through bf16 exactly where the eager path does: it materialises
    # hidden as bf16 and the quant reads that, not the fp32 product.
    y = y.to(tl.bfloat16).to(tl.float32)

    absmax = tl.max(tl.abs(y), axis=1)
    scale = tl.maximum(absmax, eps) / fp8_max
    y_q = tl.clamp(y / scale[:, None], fp8_min, fp8_max)

    out_q_ptrs = (
        output_q_ptr + (m_offset + offs_m[:, None]) * output_q_stride_m
        + n_offset + offs_n[None, :]
    )
    tl.store(out_q_ptrs, y_q.to(output_q_ptr.dtype.element_ty), mask=row_mask[:, None])

    scale_ptrs = output_scale_ptr + pid_g * output_scale_stride_k + m_offset + offs_m
    tl.store(scale_ptrs, scale, mask=row_mask)


def _silu_mul_quant_fp32scale(
    gate_up3: torch.Tensor, inter: int, clamp_limit: float
):
    """``[E, T, 2*inter]`` bf16 -> (``[E*T, inter]`` fp8, ``[E*T, inter/128]`` fp32 M-major).

    Takes the 3D view rather than a flattened one on purpose: the caller passes
    ``gate_up[:, :t_rows]``, whose first two dims are not mergeable, so flattening it
    would copy every element.

    The scale is allocated transposed so its M stride is 1, which is the
    column-major/TMA-aligned layout DeepGEMM wants for the LHS and the same one
    ``sgl_per_token_group_quant_fp8(column_major_scales=True)`` produces.
    """
    E, T = gate_up3.shape[0], gate_up3.shape[1]
    M = E * T
    assert gate_up3.shape[2] == 2 * inter, (tuple(gate_up3.shape), inter)

    # ROWS_PER_E is a tl.constexpr, so a value that changes between calls forces a
    # fresh Triton compile. The masked path passes a max_m fixed at load, but the
    # contiguous path's all_tokens is the sum of per-expert routed counts and so
    # differs on every request -- which turned a 0.2 ms kernel into a per-request
    # JIT and took 32k TTFT from 1.77 s to 5.4 s. When E == 1 the group index is
    # identically 0 and row_j is the flat row, so any constant >= M gives the same
    # arithmetic; pin a power of two so the div/mod stay shifts and the kernel is
    # compiled once for every shape this path will ever see.
    rows_per_e = T if E > 1 else _ROWS_PER_E_FLAT
    num_groups = inter // FP8_BLOCK
    q = torch.empty((M, inter), dtype=torch.float8_e4m3fn, device=gate_up3.device)
    scale = torch.empty(
        (num_groups, M), dtype=torch.float32, device=gate_up3.device
    ).transpose(0, 1)
    block_m = 64
    grid = (num_groups, triton.cdiv(M, block_m))
    _silu_mul_fp8_quant_fp32scale_kernel[grid](
        gate_up3,
        q,
        scale,
        M,
        gate_up3.stride(0),
        gate_up3.stride(1),
        rows_per_e,
        q.stride(0),
        scale.stride(1),
        float(clamp_limit),
        N=2 * inter,
        NUM_GROUPS=num_groups,
        eps=1e-4,
        fp8_min=-448.0,
        fp8_max=448.0,
        GROUP_SIZE=FP8_BLOCK,
        BLOCK_M=block_m,
        HAS_CLAMP=clamp_limit > 0,
    )
    return q, scale

@register_strategy
class GroupedFP8Strategy(RoutedExpertsStrategy):
    name = "grouped_fp8"

    # Both are set in setup_weights; the class-level defaults exist so forward is
    # safe for a caller that drives the strategy without one (the equivalence test
    # does call setup_weights, but nothing enforces that).
    _ll_ok: bool = False
    _captured_ns = None

    @classmethod
    def can_handle(cls, cfg: MoeCfg) -> bool:
        """Deliberately does NOT check whether the checkpoint's experts are FP8.

        ``MoeCfg`` carries no dtype and threading one through Block/MoE would
        buy nothing: on SM90 this is the only strategy that can run routed
        experts at all (``grouped_fp4`` and ``mega`` require SM100,
        ``local_loop`` — and therefore ``deepep``, which composes it — hardcodes
        FP4 storage). So claiming the slot and failing in ``setup_weights`` with
        "convert the checkpoint" is strictly better than declining and letting
        ``local_loop`` die inside an FP4 kernel.
        """
        if not (
            cfg.dim % FP8_BLOCK == 0
            and cfg.moe_inter_dim % FP8_BLOCK == 0
            and _has_grouped_fp8_kernel()
        ):
            return False
        if cfg.ep_size == 1:
            return True
        # EP needs a process group whose size matches ep_size, because the
        # combine reduce-scatters over Group.TP. TP is the right group: this
        # model's CP shards tokens over it (``cp.py`` all-gathers with
        # ``Group.TP``), and ep_size is derived as tp_size * dp_size.
        return _ep_group_size() == cfg.ep_size

    def setup_weights(self, layer_weights: Dict) -> None:
        """Stack routed experts into the ``[E, N, K]`` FP8 buffers DeepGEMM reads.

        Pops ``W.v4_routed_w{1,2,3}_{w,s}`` (each ``[E_local, ...]``). Gate and
        up are concatenated along N so one grouped GEMM covers both, which is
        valid because ``moe_inter_dim`` is a multiple of the 128-row scale block.

        Scales arrive as UE8M0 because that is how the checkpoint stores them;
        SM90 DeepGEMM wants FP32, so they are widened once here rather than per
        forward. The widened copy is small (one value per 128x128 block).
        """
        from rtp_llm.utils.model_weight import W

        cfg = self.cfg
        # n_local_experts, not n_routed_experts: the loader hands every strategy
        # already-EP-sliced stacks (see MegaMoEStrategy.setup_weights). At
        # ep_size==1 the two are equal, so this is also correct for pure TP.
        E, D, inter = cfg.n_local_experts, cfg.dim, cfg.moe_inter_dim
        stacked_w1_w = layer_weights.pop(W.v4_routed_w1_w)
        stacked_w1_s = layer_weights.pop(W.v4_routed_w1_s)
        stacked_w2_w = layer_weights.pop(W.v4_routed_w2_w)
        stacked_w2_s = layer_weights.pop(W.v4_routed_w2_s)
        stacked_w3_w = layer_weights.pop(W.v4_routed_w3_w)
        stacked_w3_s = layer_weights.pop(W.v4_routed_w3_s)
        device = stacked_w1_w.device

        if stacked_w1_w.dtype != torch.float8_e4m3fn:
            raise TypeError(
                f"{self.name} needs FP8 routed experts, got {stacked_w1_w.dtype}. "
                "Convert the checkpoint with dsv4_fp8/convert_fp4_experts_to_fp8.py, "
                "or force an FP4 strategy via DSV4_MOE_STRATEGY."
            )
        # Guards the EP contract explicitly. If the loader ever stopped slicing,
        # the mismatch would otherwise surface as a silently wrong expert range
        # (copy_ into a smaller buffer raises, but a LARGER one would not).
        if stacked_w1_w.shape[0] != E:
            raise ValueError(
                f"{self.name} expected {E} EP-local experts "
                f"(n_local_experts, ep_size={cfg.ep_size}), got a stack of "
                f"{stacked_w1_w.shape[0]}."
            )

        self._w13 = torch.empty((E, 2 * inter, D), dtype=torch.float8_e4m3fn, device=device)
        self._w13[:, :inter].copy_(stacked_w1_w)
        self._w13[:, inter:].copy_(stacked_w3_w)
        self._w2 = stacked_w2_w.contiguous()

        self._s13 = torch.empty(
            (E, 2 * inter // FP8_BLOCK, D // FP8_BLOCK), dtype=torch.float32, device=device
        )
        self._s13[:, : inter // FP8_BLOCK].copy_(stacked_w1_s.to(torch.float32))
        self._s13[:, inter // FP8_BLOCK :].copy_(stacked_w3_s.to(torch.float32))
        self._s2 = stacked_w2_s.to(torch.float32).contiguous()

        del stacked_w1_w, stacked_w1_s, stacked_w2_w, stacked_w2_s
        del stacked_w3_w, stacked_w3_s

        self._captured_ns = set()

        # Build the LL buffer here, not on first forward: it is collective and does
        # an NVSHMEM handshake, neither of which can happen inside a graph capture.
        # Doing it before empty_cache() also means KV-pool sizing sees it as taken.
        #
        # Gated on the role's own bound, not just on the env flag.
        # ``resolve_moe_max_tokens_per_rank`` returns ``max_generate_batch_size *
        # tokens_per_batch`` for a decode role (8 here) and a >= 4096 budget for
        # anything else, so ``max_tokens_per_rank <= ll_max_tokens`` is exactly the
        # question "can this role ever reach the low-latency branch". Without the
        # gate a prefill rank allocates a 136 MB NVSHMEM buffer it can never use,
        # and a short enough prefill request would fall into ``_ll_buffer`` and do
        # that allocation *mid-request*, since the forward branch tests the runtime
        # token count rather than the bound.
        self._ll_ok = self._ll_gate(cfg)
        if self._ll_ok:
            _ll_buffer(cfg.ep_size, D, cfg.n_routed_experts)

        # Hand the loader's freed blocks back to the driver so KV-pool sizing
        # sees real residual HBM instead of allocator-cached blocks.
        torch.cuda.empty_cache()

    @staticmethod
    def _ll_gate(cfg: MoeCfg) -> bool:
        """Whether this role can ever reach the low-latency exchange.

        Extracted so it is callable without standing up NVSHMEM: it decides
        whether a 136 MB RDMA buffer gets allocated, and the only way to test that
        decision through ``setup_weights`` is to allocate one.

        Every term is known at load time, which is what keeps the forward branch
        rank-uniform.
        """
        return (
            cfg.ep_size > 1
            and _deepep_ll_enabled()
            and cfg.max_tokens_per_rank <= _ll_max_tokens(cfg.ep_size)
        )

    def forward(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """Returns ``[N, D]`` float32: sum over (token, top-k) of
        ``weight * expert[idx](x)``.

        Args:
          x: ``[N, D]`` BF16 flattened tokens for THIS rank (post-MoE-gate
            activation). Under CP each rank holds a different token shard.
          weights: ``[N, topk]`` FP32 router weights.
          indices: ``[N, topk]`` int64 GLOBAL expert IDs.
        """
        cfg = self.cfg
        N, D = x.shape
        device = x.device

        if cfg.ep_size == 1:
            if N == 0:
                return torch.zeros(N, D, dtype=torch.float32, device=device)
            return self._local_experts(x, weights, indices, 0).float()

        # --- EP combine: all-gather tokens, apply local experts, reduce-scatter.
        #
        # N == 0 still has to participate: the collectives below are symmetric
        # across ranks, and a rank that returned early would hang the others.
        import torch.distributed as dist

        group = _ep_group()
        if group is None:
            raise RuntimeError(
                f"{self.name} with ep_size={cfg.ep_size} needs an initialised "
                "Group.TP process group for the combine."
            )
        if os.environ.get(_EP_CHECK_SIZES_ENV) == "1":
            self._assert_uniform_token_count(N, group, device)

        ep = cfg.ep_size

        # Every term below is host-side and equal on every rank -- an env var, this
        # rank's token count (a CP invariant, see _EP_CHECK_SIZES_ENV) and a buffer
        # size fixed at load. So all ranks take the same branch and no collective is
        # left half-entered. N == 0 keeps the all-gather path, whose zero-token case
        # is already handled below; LL dispatch has no such contract.
        self._assert_one_captured_size(N)

        if self._ll_ok and N > 0:
            buf, ll_max_tokens = _ll_buffer(ep, D, cfg.n_routed_experts)
            if N > ll_max_tokens:
                # _ll_gate already established max_tokens_per_rank <=
                # ll_max_tokens, so reaching here means the scheduler handed this
                # rank more tokens than its own declared bound. Falling back to
                # the all-gather path -- what this used to do -- is the one
                # outcome worse than stopping: a rank at or under the bound would
                # still take the low-latency branch, the two sets of EP
                # collectives would not match, and the symptom would be an NCCL
                # hang followed by the P/D timeout rather than an error here.
                raise RuntimeError(
                    f"grouped_fp8: {N} tokens on this rank exceeds the "
                    f"low-latency buffer's {ll_max_tokens} per-rank capacity "
                    f"(cfg.max_tokens_per_rank={cfg.max_tokens_per_rank}). Raise "
                    "DSV4_MOE_LL_MAX_TOKENS to cover the scheduler's bound, or "
                    "set DSV4_MOE_DEEPEP_LL=0 to use the all-gather exchange."
                )
            return self._local_experts_ll(
                x, weights, indices, buf, ll_max_tokens
            ).float()

        x_full = _all_gather_cat(x.contiguous(), ep, group)
        w_full = _all_gather_cat(weights.contiguous(), ep, group)
        i_full = _all_gather_cat(indices.contiguous(), ep, group)

        if x_full.shape[0] == 0:
            partial = torch.zeros((0, D), dtype=torch.bfloat16, device=device)
        else:
            partial = self._local_experts(
                x_full, w_full, i_full, cfg.local_expert_start
            )
        # reduce_scatter sums element-wise across ranks and hands rank r the
        # r-th chunk. Each rank contributed only its own experts' terms, so the
        # sum is the full top-k total, and chunk r is exactly this rank's shard.
        out = torch.empty((N, D), dtype=partial.dtype, device=device)
        dist.reduce_scatter_tensor(out, partial.contiguous(), group=group)
        return out.float()

    @staticmethod
    def _assert_uniform_token_count(n: int, group, device) -> None:
        """Fail loudly (not deadlock) if ranks disagree on the token count."""
        import torch.distributed as dist

        counts = torch.empty(
            dist.get_world_size(group=group), dtype=torch.int64, device=device
        )
        dist.all_gather_into_tensor(
            counts, torch.tensor([n], dtype=torch.int64, device=device), group=group
        )
        seen = counts.tolist()
        if len(set(seen)) != 1:
            raise RuntimeError(
                f"{GroupedFP8Strategy.name} EP combine requires an equal token "
                f"count on every rank, got {seen}. The all-gather/reduce-scatter "
                "combine would need per-rank padding to support ragged shards."
            )

    def _local_experts(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        expert_start: int,
    ) -> torch.Tensor:
        """Apply only this rank's experts to every token in ``x``.

        Returns ``[N, D]`` BF16 holding the partial sum over the local expert
        range; tokens routed entirely to remote experts come back as zeros
        (``_fwd_kernel_ep_gather`` skips ``expert_id < 0`` slots but always
        stores its accumulator).

        BF16 rather than FP32 to halve the reduce-scatter payload, matching what
        DeepEP's combine moves; the MoE output is cast to BF16 by
        ``combine_routed_and_shared`` anyway.
        """
        if self._should_mask(x.shape[0]):
            return self._local_experts_masked(x, weights, indices, expert_start)

        cfg = self.cfg
        N, D = x.shape
        E = cfg.n_local_experts
        inter = cfg.moe_inter_dim
        device = x.device

        a_fp8, a_scale = sgl_per_token_group_quant_fp8(
            x.contiguous(),
            group_size=FP8_BLOCK,
            eps=1e-4,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=False,
        )

        # Maps global expert IDs into this rank's [0, E) range and marks
        # non-local ones -1, which ep_scatter/ep_gather both skip.
        adjusted_topk_ids, num_recv = recompute_topk_ids_sum_expert_count(
            indices,
            current_expert_start_id=expert_start,
            num_local_experts=E,
        )

        # ep_scatter's kernel_1 builds expert_start_loc as the exclusive cumsum
        # of the counts it is handed, so it must receive the ALIGNED counts —
        # raw counts would let consecutive experts overlap each other's padding
        # rows and the GEMM would read garbage.
        aligned_counts_list = [align(c, _GROUPED_ALIGNMENT) for c in num_recv.cpu().tolist()]
        all_tokens = sum(aligned_counts_list)
        if all_tokens == 0:
            # No token on this rank routes to a local expert. Legal under EP.
            return torch.zeros((N, D), dtype=torch.bfloat16, device=device)
        aligned_counts = torch.tensor(
            aligned_counts_list, dtype=torch.int32, pin_memory=True, device="cpu"
        ).to(device, non_blocking=True)

        scatter_out = torch.empty(
            (all_tokens, D), dtype=torch.float8_e4m3fn, device=device
        )
        # Allocated transposed so the scale is column-major, which is the
        # TMA-aligned layout DeepGEMM wants for the LHS.
        scatter_out_scale = torch.zeros(
            [D // FP8_BLOCK, all_tokens], device=device, dtype=torch.float32
        ).transpose(0, 1)
        # ep_scatter fully overwrites m_indices, tagging padding rows with a
        # real expert. Those rows do wasted compute; ep_gather only reads the
        # valid rows recorded in output_index, so the waste is discarded.
        m_indices = torch.empty(all_tokens, dtype=torch.int32, device=device)
        output_index = torch.empty_like(adjusted_topk_ids)
        expert_start_loc = torch.empty_like(aligned_counts)
        ep_scatter(
            a_fp8,
            a_scale,
            adjusted_topk_ids,
            aligned_counts,
            expert_start_loc,
            scatter_out,
            scatter_out_scale,
            m_indices,
            output_index,
            scale_ue8m0=False,
        )
        m_indices.clamp_(min=0, max=E - 1)
        del a_fp8, a_scale

        gate_up = torch.empty(all_tokens, 2 * inter, device=device, dtype=torch.bfloat16)
        m_grouped_fp8_gemm_nt_contiguous(
            (scatter_out, scatter_out_scale),
            (self._w13, self._s13),
            gate_up,
            m_indices,
        )
        del scatter_out, scatter_out_scale

        # V4 SwiGLU: gate clamps from above only, up clamps symmetrically, and
        # the product rounds through bf16 before quantisation.
        limit = cfg.swiglu_limit
        if _fused_swiglu_enabled():
            # The same kernel the masked path uses. Here ``gate_up`` is
            # ``[all_tokens, 2*inter]`` rather than ``[E, T, 2*inter]``, and
            # ``unsqueeze(0)`` is a free view that makes E=1, T=all_tokens --
            # shapes the kernel's grid (num_groups, cdiv(M, 64)) already handles.
            # Its outputs are the same pair this path already feeds to
            # ``m_grouped_fp8_gemm_nt_contiguous``: ``[M, inter]`` fp8 plus an
            # ``[M, inter/128]`` fp32 scale whose M stride is 1, which is what
            # ``sgl_per_token_group_quant_fp8(column_major_scales=True)`` produced.
            #
            # This is where prefill spends its largest single non-GEMM block. The
            # explicit sequence below makes about eight passes over an
            # ``[all_tokens, 2*inter]`` tensor and several full-size fp32
            # temporaries; measured standalone at the CP4/32k row count
            # (all_tokens = 49152): 4.457 -> 0.206 ms per layer, 0.11 -> 2.46 TB/s,
            # i.e. ~183 ms of a ~1770 ms 32k TTFT. Scales come out bit-identical;
            # 0.092% of quantised bytes differ by one fp8 e4m3 ULP, the same
            # residue documented on the masked path, so this honours the same
            # ``DSV4_MOE_FUSED_SWIGLU=0`` escape hatch.
            h_fp8, h_scale = _silu_mul_quant_fp32scale(
                gate_up.unsqueeze(0), inter, limit
            )
            del gate_up
        else:
            # The exact sequence _silu_mul_fp8_quant_triton was written to
            # reproduce; spelled out because that kernel emits only packed UE8M0.
            gate, up = gate_up[:, :inter], gate_up[:, inter:]
            if limit > 0:
                gate = gate.clamp(max=limit)
                up = up.clamp(min=-limit, max=limit)
            hidden = (F.silu(gate.float()) * up.float()).to(torch.bfloat16)
            del gate_up, gate, up

            h_fp8, h_scale = sgl_per_token_group_quant_fp8(
                hidden.contiguous(),
                group_size=FP8_BLOCK,
                eps=1e-4,
                column_major_scales=True,
                scale_tma_aligned=True,
                scale_ue8m0=False,
            )
            del hidden

        down_out = torch.empty(all_tokens, D, device=device, dtype=torch.bfloat16)
        m_grouped_fp8_gemm_nt_contiguous(
            (h_fp8, h_scale),
            (self._w2, self._s2),
            down_out,
            m_indices,
        )
        del h_fp8, h_scale

        # Accumulates each output token's top-k source rows times the router
        # weight in an fp32 register, one bf16 store — no [N, topk, D] temporary.
        gather_out = torch.empty((N, D), dtype=torch.bfloat16, device=device)
        ep_gather(down_out, adjusted_topk_ids, weights, output_index, gather_out)
        return gather_out

    def _assert_one_captured_size(self, n_tokens: int) -> None:
        """Refuse a capture set that would deadlock at replay, at capture time.

        This strategy's EP exchange is inside the graph -- three all-gathers and a
        reduce-scatter per layer on the all-gather path, dispatch/combine on the
        low-latency one. ``CudaGraphRunner::tryGetRealGraphDecodeBatchSize`` picks a
        graph per rank by ``lower_bound`` over *that rank's own* batch size, and
        nothing equalises per-rank batch: ``mayAddFakeStream`` only guarantees each
        dp rank has at least one stream, not the same number. So if more than one
        batch size is captured, two ranks can replay different graphs and issue
        those collectives with different counts. NCCL neither errors nor recovers --
        it hangs, and the symptom is one token followed by the P/D timeout, tens of
        seconds away from the cause.

        A capture set of exactly one size, equal to the largest per-rank batch the
        scheduler can produce, makes all ranks agree by construction. That is a
        deployment constraint that used to live only in a comment; this turns
        violating it into a startup error.

        The proper fix is an ``all_reduce(MAX)`` over the dp group before
        ``lower_bound``, so ranks pick the smallest graph that fits the *global*
        batch. That is a change in ``cuda_graph_runner.cc`` and is not attempted
        here, which is exactly why this check exists.
        """
        if self.cfg.ep_size <= 1 or not torch.cuda.is_current_stream_capturing():
            return
        seen = self._captured_ns
        if seen is None:
            seen = self._captured_ns = set()
        seen.add(int(n_tokens))
        if len(seen) > 1:
            raise RuntimeError(
                f"{self.name} with ep_size={self.cfg.ep_size} saw CUDA-graph "
                f"capture at more than one batch size ({sorted(seen)}). Its EP "
                "collectives are inside the graph and CudaGraphRunner selects a "
                "graph per rank, so two ranks can replay different graphs and hang "
                "in NCCL. Capture exactly one batch size, equal to "
                "--concurrency_limit (e.g. DECODE_CAPTURE_CONFIG=8 with "
                "--concurrency_limit 8)."
            )
        if int(n_tokens) < int(self.cfg.max_tokens_per_rank):
            raise RuntimeError(
                f"{self.name} with ep_size={self.cfg.ep_size} is capturing "
                f"{n_tokens} rows per rank while the scheduler may hand it up to "
                f"{self.cfg.max_tokens_per_rank}. A rank whose batch exceeds the "
                "captured size falls back to eager while the others replay the "
                "graph, which hangs in the in-graph EP collectives. Raise "
                "DECODE_CAPTURE_CONFIG to --concurrency_limit, or lower "
                "--concurrency_limit to the captured size."
            )

    @staticmethod
    def _should_mask(n_tokens: int) -> bool:
        """Masked layout, yes or no — decided from host state only.

        Every rank must reach the same answer or the group deadlocks at the next
        collective, so this reads a shape and an environment variable and never the
        data. ``n_tokens`` is the row count this call will process (post-all-gather
        under EP), which is a shape on every rank and equal across them.
        """
        capturing = torch.cuda.is_current_stream_capturing()
        if capturing or _masked_enabled():
            if n_tokens > _MASKED_MAX_N:
                if capturing:
                    raise RuntimeError(
                        f"{GroupedFP8Strategy.name}: CUDA-graph capture needs the masked "
                        f"layout, but n_tokens={n_tokens} exceeds _MASKED_MAX_N="
                        f"{_MASKED_MAX_N}; its buffer is n_local_experts * "
                        "align(n_tokens, 128) rows. Capture decode only, or raise the cap."
                    )
                return False
            return True
        return False

    def _local_experts_masked(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        expert_start: int,
    ) -> torch.Tensor:
        """``_local_experts`` with static shapes and no device->host read.

        Why this exists: the decode step is ~86% device-idle (40 ms of kernels spread
        over a 292 ms step, measured), so the fix is CUDA graph, and the only thing
        refusing capture was this strategy. Two properties of the contiguous path are
        what make it uncapturable, and both are the same fact -- the buffer is
        ``sum_e align(count_e, 128)``, a quantity that lives on the device:

          * ``num_recv.cpu()`` drains the pipeline once per MoE layer (43x per step);
          * the allocation size is therefore not known at capture time.

        Making that size host-known by padding every local expert to 128 rows whether or
        not it was routed to would cost ``64 * 128 = 8192`` rows of GEMM per layer at
        ep4 against the ~200 the dynamic path computes at decode batch. The masked
        layout avoids the trade: shapes are fixed at capture, while the per-expert row
        count stays on the device in ``masked_m`` and the kernel reads it, so the work
        still scales with the routing.

        The layout comes for free from ``ep_scatter``. It places expert e's rows at
        ``expert_start_loc[e]``, built as the exclusive cumsum of the counts it is
        handed; hand it a UNIFORM ``max_m`` for every expert and the flat
        ``[E*max_m, D]`` buffer it fills *is* ``[E, max_m, D]``. No new kernel, no
        transpose, and ``masked_m`` is ``num_recv`` exactly as it already exists.

        ``max_m`` must be a multiple of 128, and that is load-bearing rather than
        cosmetic: ``ep_scatter`` tiles ``m_indices`` in ``BLOCK_E=128`` row blocks, so
        per-expert segments shorter than that overrun the end of the buffer and corrupt
        whatever the caching allocator handed out next. Measured on H20: at
        ``max_m=32`` the MoE output is ~45% wrong with the damage landing in unrelated
        tensors (and the first call in a fresh process looking fine, because the stomped
        page was still untouched); at 128 and 256 the result is bit-identical to the
        contiguous path over repeated calls.
        """
        cfg = self.cfg
        N, D = x.shape
        E = cfg.n_local_experts
        inter = cfg.moe_inter_dim
        device = x.device

        # A token reaches an expert at most once per top-k slot, so no expert can be sent
        # more rows than there are tokens: N is a valid bound, and aligning it up to the
        # scatter's block size keeps the segments legal.
        max_m = align(N, _GROUPED_ALIGNMENT)
        rows = E * max_m

        # ...but only ep_scatter needs those 128-row segments. The first masked GEMM writes
        # whole 64-row tiles and at most ceil(count_e/64)*64 <= align(N, 64) rows per group,
        # so every row above that is provably dead and no stage after the GEMM has to touch
        # it. At decode batch that is the difference between 8192 and 4096 rows through
        # swiglu, the quant, the second GEMM and the gather -- and those move 67 MB per
        # buffer per layer, which the capture-mode profile showed as ~20 ms/step of copies,
        # fills and elementwise work in service of 32 rows of real output. Verified
        # bit-identical to the contiguous path at N=4/24/96 and max_m=128/256.
        t_rows = min(max_m, align(N, _MASKED_TILE))

        a_fp8, a_scale = sgl_per_token_group_quant_fp8(
            x.contiguous(),
            group_size=FP8_BLOCK,
            eps=1e-4,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=False,
        )

        adjusted_topk_ids, num_recv = recompute_topk_ids_sum_expert_count(
            indices,
            current_expert_start_id=expert_start,
            num_local_experts=E,
        )

        uniform_counts = torch.full(
            (E,), max_m, dtype=torch.int32, device=device
        )
        # zeros, not empty, for the same reason the contiguous path zeroes its scale: the
        # masked GEMM computes whole 64-row tiles, so rows between ``num_recv[e]`` and the
        # end of the tile are read. They are discarded downstream -- ep_gather only visits
        # the rows in ``output_index`` -- but an uninitialised fp8 byte is a legal NaN and
        # NaN * 0 is NaN, so they must not be garbage.
        scatter_out = torch.zeros((rows, D), dtype=torch.float8_e4m3fn, device=device)
        scatter_out_scale = torch.zeros(
            [D // FP8_BLOCK, rows], device=device, dtype=torch.float32
        ).transpose(0, 1)
        m_indices = torch.zeros(rows, dtype=torch.int32, device=device)
        output_index = torch.empty_like(adjusted_topk_ids)
        expert_start_loc = torch.empty_like(uniform_counts)
        ep_scatter(
            a_fp8,
            a_scale,
            adjusted_topk_ids,
            uniform_counts,
            expert_start_loc,
            scatter_out,
            scatter_out_scale,
            m_indices,
            output_index,
            scale_ue8m0=False,
        )
        del a_fp8, a_scale

        # No m_indices.clamp_ here: in the masked layout the group of a row is its
        # position, not a tag, so the GEMM never reads m_indices at all. ep_scatter still
        # writes it, so it still has to be allocated.
        #
        # Only the live prefix is zeroed. The GEMM skips groups whose masked_m is 0 and
        # writes one tile of the rest, so rows in [0, t_rows) are either written by it or
        # read by swiglu below; rows above t_rows are read by nothing.
        gate_up = torch.empty(E, max_m, 2 * inter, device=device, dtype=torch.bfloat16)
        gate_up[:, :t_rows].zero_()
        m_grouped_fp8_gemm_nt_masked(
            (scatter_out.view(E, max_m, D), scatter_out_scale.view(E, max_m, D // FP8_BLOCK)),
            (self._w13, self._s13),
            gate_up,
            num_recv,
            expected_m=max_m,
        )
        del scatter_out, scatter_out_scale

        # clamp + silu + mul + quant in one kernel over the live prefix. The eager
        # equivalent is eight kernels and several fp32 temporaries; see
        # _silu_mul_quant_fp32scale. gate_up[:, :t_rows] is a strided view (contiguous
        # within each group's block, with a gap between groups), which the kernel reads
        # through input_stride_m, so no gathering copy is needed either.
        if _fused_swiglu_enabled():
            h_fp8, h_scale = _silu_mul_quant_fp32scale(
                gate_up[:, :t_rows], inter, cfg.swiglu_limit
            )
        else:
            gu = gate_up[:, :t_rows]
            gate, up = gu[..., :inter], gu[..., inter:]
            limit = cfg.swiglu_limit
            if limit > 0:
                gate = gate.clamp(max=limit)
                up = up.clamp(min=-limit, max=limit)
            hidden = (F.silu(gate.float()) * up.float()).to(torch.bfloat16).reshape(
                E * t_rows, inter
            )
            del gu, gate, up
            h_fp8, h_scale = sgl_per_token_group_quant_fp8(
                hidden.contiguous(),
                group_size=FP8_BLOCK,
                eps=1e-4,
                column_major_scales=True,
                scale_tma_aligned=True,
                scale_ue8m0=False,
            )
            del hidden
        del gate_up

        down_out = torch.zeros(E, t_rows, D, device=device, dtype=torch.bfloat16)
        m_grouped_fp8_gemm_nt_masked(
            (h_fp8.view(E, t_rows, inter), h_scale.view(E, t_rows, inter // FP8_BLOCK)),
            (self._w2, self._s2),
            down_out,
            num_recv,
            expected_m=t_rows,
        )
        del h_fp8, h_scale

        # ep_scatter wrote output_index in units of max_m; down_out is in units of t_rows.
        # Two integer ops on an [N, topk] tensor, on the device, so the branch stays free of
        # host reads. A no-op when t_rows == max_m.
        if t_rows != max_m:
            output_index = (output_index // max_m) * t_rows + (output_index % max_m)

        gather_out = torch.empty((N, D), dtype=torch.bfloat16, device=device)
        ep_gather(
            down_out.view(E * t_rows, D),
            adjusted_topk_ids,
            weights,
            output_index,
            gather_out,
        )
        return gather_out

    def _local_experts_ll(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        buf,
        ll_max_tokens: int,
    ) -> torch.Tensor:
        """The whole EP combine over DeepEP low-latency kernels. ``[N, D]`` BF16.

        Replaces ``_all_gather_cat`` x3 + ``_local_experts_masked`` +
        ``reduce_scatter_tensor``. What is left of the masked path is its middle: the
        two grouped FP8 GEMMs and the fused swiglu, byte-for-byte the same calls.

        The stages that disappear are the ones that only existed to build the masked
        layout locally after moving every token to every rank:

          * the three all-gathers moved ``ep * N`` tokens, weights and indices to all
            ranks so each could pick out its own experts' rows. Dispatch sends each
            token only to the ranks that own an expert it routed to.
          * ``sgl_per_token_group_quant_fp8`` on the input: dispatch quantises to fp8
            in flight, and returns scales already column-major in the last two dims.
          * ``ep_scatter``, whose only job was to place rows at ``e * max_m``, and its
            two zero-filled buffers. Dispatch packs per expert and reports the counts.
          * ``ep_gather`` + the reduce-scatter, which summed ``ep`` full ``[N, D]``
            partials. Combine reduces with the router weights inside the kernel and
            each rank receives only its own tokens.

        No device->host read anywhere, and every shape is fixed by ``ll_max_tokens``
        and ``ep``, so this captures into a CUDA graph like the masked path does.
        """
        cfg = self.cfg
        E, inter = cfg.n_local_experts, cfg.moe_inter_dim
        device = x.device

        # topk_idx carries GLOBAL expert ids and -1 for "no expert": no remapping to
        # a local range, and no padding of top-6 to a power of two (top-6 is accepted
        # as-is; verified on H20).
        (recv_x, recv_scale), recv_count, handle, _, _ = buf.low_latency_dispatch(
            x.contiguous(),
            indices.contiguous(),
            ll_max_tokens,
            cfg.n_routed_experts,
            use_fp8=True,
            # SM90 DeepGEMM wants fp32 scales, which is what round_scale=False and
            # use_ue8m0=False produce -- the same widening setup_weights does.
            round_scale=False,
            use_ue8m0=False,
            async_finish=False,
            return_recv_hook=False,
        )
        # ll_max_tokens * ep, and by construction (see _ll_max_tokens) the smallest
        # multiple of _MASKED_TILE that can hold the dispatch, so there is no
        # equivalent of the masked path's t_rows narrowing left to do.
        rows = recv_x.shape[1]

        # torch.empty, not zeros: the GEMM writes whole _MASKED_TILE tiles, so rows
        # past ``recv_count[e]`` hold junk, and groups with recv_count 0 are skipped
        # entirely and stay uninitialised. Neither reaches the output -- combine reads
        # only each expert's live rows via the dispatch handle -- and the quant below
        # is per row-group, so a NaN cannot leak sideways into a live row. Confirmed
        # against the all-gather arm over 43 iterations with 27 of 64 groups empty.
        gate_up = torch.empty(E, rows, 2 * inter, device=device, dtype=torch.bfloat16)
        m_grouped_fp8_gemm_nt_masked(
            (recv_x, recv_scale),
            (self._w13, self._s13),
            gate_up,
            recv_count,
            expected_m=rows,
        )

        if _fused_swiglu_enabled():
            h_fp8, h_scale = _silu_mul_quant_fp32scale(gate_up, inter, cfg.swiglu_limit)
        else:
            gate, up = gate_up[..., :inter], gate_up[..., inter:]
            limit = cfg.swiglu_limit
            if limit > 0:
                gate = gate.clamp(max=limit)
                up = up.clamp(min=-limit, max=limit)
            hidden = (F.silu(gate.float()) * up.float()).to(torch.bfloat16).reshape(
                E * rows, inter
            )
            del gate, up
            h_fp8, h_scale = sgl_per_token_group_quant_fp8(
                hidden.contiguous(),
                group_size=FP8_BLOCK,
                eps=1e-4,
                column_major_scales=True,
                scale_tma_aligned=True,
                scale_ue8m0=False,
            )
            del hidden
        del gate_up

        # combine's input must be the full ``[E, rows, D]``; a narrowed view is not an
        # option anyway, DeepGEMM asserts stride(0) == rows * D on its output.
        down = torch.empty(E, rows, cfg.dim, device=device, dtype=torch.bfloat16)
        m_grouped_fp8_gemm_nt_masked(
            (h_fp8.view(E, rows, inter), h_scale.view(E, rows, inter // FP8_BLOCK)),
            (self._w2, self._s2),
            down,
            recv_count,
            expected_m=rows,
        )
        del h_fp8, h_scale

        out, _, _ = buf.low_latency_combine(
            down, indices.contiguous(), weights.contiguous(), handle
        )
        # ``out`` aliases the RDMA buffer, which the next layer's combine overwrites.
        # forward()'s .float() copies it out before that happens; returning it as bf16
        # would hand the caller a tensor with 42 layers left to live.
        return out
