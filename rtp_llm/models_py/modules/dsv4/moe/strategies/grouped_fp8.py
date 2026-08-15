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

# Above this many tokens per MoE call the masked layout is refused, because its buffer is
# E * align(N, 128) rows: decode's N (batch x ep, <=256 here) gives 8-16k rows, while a
# 16k-token prefill chunk would ask for E * 16384 = a million rows. The contiguous layout
# is the right one there -- at prefill N its per-expert 128-row padding is a few percent,
# not the 64x it is at decode N.
_MASKED_MAX_N = 1024


def _masked_enabled() -> bool:
    return os.environ.get(_MASKED_ENV, "0").strip().lower() in ("1", "true", "on", "yes")


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


def _ep_group():
    """The process group the EP combine runs over, or None if unavailable."""
    try:
        import torch.distributed as dist

        if not dist.is_initialized():
            return None
        from rtp_llm.models_py.distributed.collective_torch import Group, _get_group

        return _get_group(Group.TP)
    except Exception:
        return None


def _ep_group_size() -> int:
    """World size of the EP combine group, or 0 when there is no group."""
    group = _ep_group()
    if group is None:
        return 0
    import torch.distributed as dist

    return dist.get_world_size(group=group)


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


@register_strategy
class GroupedFP8Strategy(RoutedExpertsStrategy):
    name = "grouped_fp8"

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
        # Hand the loader's freed blocks back to the driver so KV-pool sizing
        # sees real residual HBM instead of allocator-cached blocks.
        torch.cuda.empty_cache()

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
        # the product rounds through bf16 before quantisation. This is the exact
        # sequence _silu_mul_fp8_quant_triton was written to reproduce; it is
        # spelled out here because that kernel emits only packed UE8M0 scales.
        gate, up = gate_up[:, :inter], gate_up[:, inter:]
        limit = cfg.swiglu_limit
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
        whatever the caching allocator handed out next. Measured on 11.24.224.84: at
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
        gate_up = torch.zeros(E, max_m, 2 * inter, device=device, dtype=torch.bfloat16)
        m_grouped_fp8_gemm_nt_masked(
            (scatter_out.view(E, max_m, D), scatter_out_scale.view(E, max_m, D // FP8_BLOCK)),
            (self._w13, self._s13),
            gate_up,
            num_recv,
            expected_m=max_m,
        )
        del scatter_out, scatter_out_scale

        # Identical arithmetic to the contiguous path, on a [rows, 2*inter] view.
        gate_up_2d = gate_up.view(rows, 2 * inter)
        gate, up = gate_up_2d[:, :inter], gate_up_2d[:, inter:]
        limit = cfg.swiglu_limit
        if limit > 0:
            gate = gate.clamp(max=limit)
            up = up.clamp(min=-limit, max=limit)
        hidden = (F.silu(gate.float()) * up.float()).to(torch.bfloat16)
        del gate_up, gate_up_2d, gate, up

        h_fp8, h_scale = sgl_per_token_group_quant_fp8(
            hidden.contiguous(),
            group_size=FP8_BLOCK,
            eps=1e-4,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=False,
        )
        del hidden

        down_out = torch.zeros(E, max_m, D, device=device, dtype=torch.bfloat16)
        m_grouped_fp8_gemm_nt_masked(
            (h_fp8.view(E, max_m, inter), h_scale.view(E, max_m, inter // FP8_BLOCK)),
            (self._w2, self._s2),
            down_out,
            num_recv,
            expected_m=max_m,
        )
        del h_fp8, h_scale

        gather_out = torch.empty((N, D), dtype=torch.bfloat16, device=device)
        ep_gather(
            down_out.view(rows, D), adjusted_topk_ids, weights, output_index, gather_out
        )
        return gather_out
