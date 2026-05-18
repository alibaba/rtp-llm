"""MegaMoEStrategy: DeepGEMM ``fp8_fp4_mega_moe`` symm-mem fused kernel.

EP > 1 only. The Mega kernel fuses dispatch + L1 GEMM + SwiGLU + L2 GEMM +
combine into one kernel backed by a PyTorch symmetric-memory buffer for
NVLink communication. Requires SM100, PyTorch ≥ 2.9 (symmetric_memory),
DeepGEMM ≥ 2.5, and an initialised process group.

Wired into ``MoE`` via ``select_strategy`` when ep_size > 1 and Mega is
available. Direct port of the pre-refactor ``_setup_mega_moe`` +
``_routed_experts_mega_moe`` methods.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional

import torch

from ...quant_layouts import FP4_BLOCK, prepare_fp4_weight_scale_for_deepgemm
from ..input_packer import get_mega_moe_input_packer
from ..mega_buf import (
    _get_or_create_mega_buf,
    _get_or_create_mega_output,
    _mega_moe_enabled,
)
from ..mega_jit_warmup import (
    clamp_token_counts,
    format_token_counts,
    generate_mega_moe_jit_token_counts,
    mega_moe_jit_warmup_enabled,
    parse_mega_moe_jit_warmup_tokens_override,
)
from ..shared_expert import strict_fused_moe_enabled
from .base import MoeCfg, RoutedExpertsStrategy, register_strategy

_MEGA_MOE_JIT_WARMED_KEYS: set[tuple] = set()


def _mega_output_capacity(buf, requested_capacity: int) -> int:
    """Output rows must cover DeepGEMM's internally aligned token capacity."""
    capacity = max(int(requested_capacity), 1)
    aligned_capacity = getattr(buf, "num_max_tokens_per_rank", None)
    if aligned_capacity is not None:
        capacity = max(capacity, int(aligned_capacity))
    return capacity


@register_strategy
class MegaMoEStrategy(RoutedExpertsStrategy):
    name = "mega"

    @classmethod
    def can_handle(cls, cfg: MoeCfg) -> bool:
        # Mega requires EP > 1, SM100, dist-init — all checked by
        # ``_mega_moe_enabled()`` except ep_size > 1, which we check here.
        return cfg.ep_size > 1 and _mega_moe_enabled()

    def setup_weights(self, layer_weights: Dict) -> None:
        """Stack EP-local routed-expert SFs into the int32 UTCCP-transposed
        layout ``fp8_fp4_mega_moe`` expects, then register the symm-mem
        dispatch buffer.

        Routed weights arrive as already-EP-sliced stacks (loader handles
        the rank slicing): ``layer_weights[W.v4_routed_w{1,2,3}_{w,s}]``
        each shaped ``[E_local, ...]``. We pop them so the only references
        kept alive are the kernel-consumable l1/l2 buffers below.

        Mega MoE expects, per expert:
          L1 w [2*inter, dim//2] int8 (gate | up rows concatenated)
          L1 sf [2*inter, ...] int32  (post-``transform_sf_into_required_layout``
            + ``transform_weights_for_mega_moe``: gate/up interleaved gran=8
            along N, SF UTCCP-transposed)
          L2 w [dim, inter//2] int8
          L2 sf [dim, ...] int32

        Memory: serialise L1 → L2 with ``del`` + ``empty_cache()`` between
        stages. Pre-allocating both fp32 SF stacks at once (and feeding
        the live tuple into ``transform_weights_for_mega_moe`` whose internal
        interleave allocates another ~size(w13)+size(w2) transient) OOMs
        268 GB on V4-Pro cp4. Splitting keeps the live set ≤ one stack.
        """
        import deep_gemm
        import torch.distributed as dist

        from rtp_llm.utils.model_weight import W

        cfg = self.cfg
        E = cfg.n_local_experts
        D = cfg.dim
        inter = cfg.moe_inter_dim

        # Pop L1 (w1/w3) stacks from layer_weights so the framework's
        # ModelWeights drops its references.
        st_w1_w = layer_weights.pop(W.v4_routed_w1_w)
        st_w1_s = layer_weights.pop(W.v4_routed_w1_s)
        st_w3_w = layer_weights.pop(W.v4_routed_w3_w)
        st_w3_s = layer_weights.pop(W.v4_routed_w3_s)
        device = st_w1_w.device

        # --- L1 (gate + up): stack, transform SF, drop the raw stack.
        w13 = torch.empty((E, 2 * inter, D // 2), dtype=torch.int8, device=device)
        s13_raw = torch.empty(
            (E, 2 * inter, D // FP4_BLOCK),
            dtype=torch.float8_e8m0fnu,
            device=device,
        )
        w13[:, :inter].copy_(st_w1_w)
        s13_raw[:, :inter].copy_(st_w1_s)
        w13[:, inter:].copy_(st_w3_w)
        s13_raw[:, inter:].copy_(st_w3_s)
        del st_w1_w, st_w1_s, st_w3_w, st_w3_s
        s13_int = prepare_fp4_weight_scale_for_deepgemm(s13_raw, 2 * inter, D, E)
        del s13_raw
        torch.cuda.empty_cache()

        # --- L2 (down): only after L1's fp32 buffer has been freed.
        st_w2_w = layer_weights.pop(W.v4_routed_w2_w)
        st_w2_s = layer_weights.pop(W.v4_routed_w2_s)
        w2 = torch.empty((E, D, inter // 2), dtype=torch.int8, device=device)
        s2_raw = torch.empty(
            (E, D, inter // FP4_BLOCK),
            dtype=torch.float8_e8m0fnu,
            device=device,
        )
        w2.copy_(st_w2_w)
        s2_raw.copy_(st_w2_s)
        del st_w2_w, st_w2_s
        s2_int = prepare_fp4_weight_scale_for_deepgemm(s2_raw, D, inter, E)
        del s2_raw
        torch.cuda.empty_cache()

        # Mega MoE transform: L1 gate/up interleave (gran=8 along N) +
        # both SFs UTCCP-transposed. Drop inputs immediately after.
        (l1_w, l1_sf), (l2_w, l2_sf) = deep_gemm.transform_weights_for_mega_moe(
            (w13, s13_int),
            (w2, s2_int),
        )
        del w13, s13_int, w2, s2_int
        torch.cuda.empty_cache()

        # Stash as plain attributes (not Parameters — the kernel reads
        # raw int8/int32 buffers with no autograd).  Original stacked
        # fp32 SFs are dropped now that the int layout has been derived.
        self._mega_l1_w = l1_w
        self._mega_l1_sf = l1_sf
        self._mega_l2_w = l2_w
        self._mega_l2_sf = l2_sf

        # (4) Allocate the symmetric-memory buffer.  Uses
        # ``torch.distributed.group.WORLD`` because our DP+EP layout has
        # ``ep_size == world_size`` — every rank holds a distinct 64/256
        # slice.  ``num_max_tokens_per_rank`` caps per-rank token count
        # fed into the MoE; bounded from ``max_tokens_per_rank`` (plumbed
        # from ``V4Args.max_seq_len`` upstream).  The library aligns this
        # up to ``get_token_alignment_for_mega_moe()`` internally (384 on
        # SM100).
        assert dist.is_initialized(), (
            "Mega MoE requires torch.distributed initialised; "
            "_mega_moe_available() should have gated this earlier"
        )
        group = dist.group.WORLD
        # Symm buffer is single-layer staging — share one across all
        # MoE layers via the module-level cache (see _get_or_create_mega_buf).
        self._mega_buf = _get_or_create_mega_buf(
            group=group,
            num_experts=cfg.n_routed_experts,
            num_max_tokens_per_rank=max(cfg.max_tokens_per_rank, 1),
            num_topk=cfg.n_activated_experts,
            hidden=D,
            intermediate_hidden=inter,
            use_fp8_dispatch=True,
            activation="swiglu",
        )
        # Single-layer staging output. All MoE layers execute sequentially, so one
        # process-local buffer is enough and avoids O(layers) persistent memory.
        self._mega_y = _get_or_create_mega_output(
            _mega_output_capacity(self._mega_buf, cfg.max_tokens_per_rank),
            D,
            torch.bfloat16,
            device,
        )
        self._input_packer = get_mega_moe_input_packer()
        self._maybe_warmup_jit_once()

    def _resolve_jit_warmup_token_counts(self, num_sms: int) -> list[int]:
        cfg = self.cfg
        # Use the logical model/runtime token cap, not DeepGEMM's internally
        # aligned buffer capacity.  The JIT key is driven by request-visible T
        # buckets; the aligned capacity only needs to be large enough to hold
        # those representatives.
        max_tokens_per_rank = int(cfg.max_tokens_per_rank)
        override = parse_mega_moe_jit_warmup_tokens_override()
        if override is not None:
            return clamp_token_counts(override, max_tokens_per_rank)
        return generate_mega_moe_jit_token_counts(
            num_ranks=cfg.ep_size,
            num_experts=cfg.n_routed_experts,
            num_experts_per_rank=cfg.n_local_experts,
            num_topk=cfg.n_activated_experts,
            intermediate_hidden=cfg.moe_inter_dim,
            num_sms=num_sms,
            max_tokens_per_rank=max_tokens_per_rank,
        )

    def _maybe_warmup_jit_once(self) -> None:
        if not mega_moe_jit_warmup_enabled():
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "MegaMoE JIT warmup must not run inside CUDA graph capture"
            )

        import deep_gemm
        import torch.distributed as dist

        cfg = self.cfg
        num_sms = int(deep_gemm.get_num_sms())
        token_counts = self._resolve_jit_warmup_token_counts(num_sms)
        if not token_counts:
            return

        max_tokens_per_rank = int(cfg.max_tokens_per_rank)
        warmup_key = (
            cfg.ep_size,
            cfg.n_routed_experts,
            cfg.n_local_experts,
            cfg.n_activated_experts,
            cfg.dim,
            cfg.moe_inter_dim,
            max_tokens_per_rank,
            cfg.swiglu_limit,
            num_sms,
            tuple(token_counts),
        )
        if warmup_key in _MEGA_MOE_JIT_WARMED_KEYS:
            return

        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            logging.info(
                "[DSV4 MegaMoE] JIT warmup start: layer=%d tokens=[%s] "
                "max_tokens_per_rank=%d ep=%d experts=%d topk=%d hidden=%d "
                "intermediate=%d num_sms=%d",
                cfg.layer_id,
                format_token_counts(token_counts),
                max_tokens_per_rank,
                cfg.ep_size,
                cfg.n_routed_experts,
                cfg.n_activated_experts,
                cfg.dim,
                cfg.moe_inter_dim,
                num_sms,
            )
        self.warmup_jit(token_counts)
        _MEGA_MOE_JIT_WARMED_KEYS.add(warmup_key)
        if rank == 0:
            logging.info(
                "[DSV4 MegaMoE] JIT warmup done: layer=%d tokens=[%s]",
                cfg.layer_id,
                format_token_counts(token_counts),
            )

    @torch.inference_mode()
    def warmup_jit(self, token_counts: list[int]) -> None:
        """Compile MegaMoE JIT buckets with synthetic rank-local tokens."""
        import torch.distributed as dist

        cfg = self.cfg
        device = self._mega_l1_w.device
        max_tokens = max(token_counts)
        x = torch.zeros((max_tokens, cfg.dim), dtype=torch.bfloat16, device=device)
        weights = torch.zeros(
            (max_tokens, cfg.n_activated_experts),
            dtype=torch.float32,
            device=device,
        )
        local_expert_ids = cfg.local_expert_start + torch.arange(
            cfg.n_activated_experts, dtype=torch.long, device=device
        ) % max(cfg.n_local_experts, 1)
        indices = local_expert_ids.view(1, -1).expand(max_tokens, -1).contiguous()

        for token_count in token_counts:
            dist.barrier()
            self.forward(
                x[:token_count],
                weights[:token_count],
                indices[:token_count],
            )
            torch.cuda.synchronize(device)
        dist.barrier()

    def forward(
        self,
        x: torch.Tensor,  # [T, D] BF16 local-rank tokens
        weights: torch.Tensor,  # [T, topk] FP32 router weights
        indices: torch.Tensor,  # [T, topk] int64 GLOBAL expert IDs
    ) -> torch.Tensor:
        """Run the fused DeepGEMM Mega MoE kernel: dispatch + L1 GEMM +
        SwiGLU + L2 GEMM + combine — all fused, symm-mem backed.

        Returns the combined routed-expert output in BF16.  The MoE epilogue
        owns the final routed+shared cast.
        """
        import deep_gemm

        T = x.size(0)
        buf = self._mega_buf
        if T > buf.num_max_tokens_per_rank:
            raise RuntimeError(
                f"Mega MoE input tokens={T} exceeds num_max_tokens_per_rank="
                f"{buf.num_max_tokens_per_rank} (derived from max_seq_len / "
                f"max_tokens_per_rank). Raise the budget at startup."
            )
        if T > self._mega_y.size(0):
            raise RuntimeError(
                f"Mega MoE output buffer rows={self._mega_y.size(0)} is smaller "
                f"than input tokens={T}. This indicates inconsistent aligned "
                "MegaMoE buffer sizing."
            )

        # ``deep_gemm.fp8_fp4_mega_moe`` is a peer-symmetric NVLink collective:
        # every rank in ``buf.group`` MUST enter the kernel together. The kernel
        # is symmetric — each rank both *dispatches* its ``T`` local tokens to
        # peers' experts AND *hosts* its local-expert slice to compute peers'
        # tokens routed to it.  Skipping a rank with ``T == 0`` (e.g. an EP/CP
        # rank that holds no input tokens for a given batch shape) does two
        # things, both bad:
        #   1. Strands the routed-expert work that peers dispatched to its
        #      local experts -> peers see zero contribution from those experts
        #      (silent wrong output).
        #   2. Triggers NVLink barrier timeout in the surviving peers
        #      (``deep_gemm/include/deep_gemm/comm/barrier.cuh:72``,
        #      ``DG_DEVICE_ASSERT(false and "NVLink barrier timeout")``) ->
        #      kernel-side ``asm("trap;")`` -> SIGTRAP after 30 s.  The trap is
        #      what surfaces as the prod ``CUDA_ERROR_LAUNCH_FAILED`` (719)
        #      cascading from ``sm100_fp8_fp4_mega_moe_impl``.
        # Therefore: always pack and always call the kernel, even when local
        # ``T == 0``.  ``pack`` becomes a no-op (zero-row slices), ``y[:0]``
        # signals ``num_tokens=0`` so this rank's dispatch loop iterates zero
        # times, and the rank still participates as expert host.  No control
        # flow depending on a GPU-side scalar -> CUDA-graph-capture safe.
        self._input_packer.pack(x, weights, indices, buf, T)

        y = self._mega_y[:T]
        deep_gemm.fp8_fp4_mega_moe(
            y,
            (self._mega_l1_w, self._mega_l1_sf),
            (self._mega_l2_w, self._mega_l2_sf),
            buf,
            recipe=(1, 1, FP4_BLOCK),
            activation="swiglu",
            activation_clamp=(
                self.cfg.swiglu_limit if self.cfg.swiglu_limit > 0 else None
            ),
            fast_math=os.environ.get("DSV4_MEGA_MOE_FAST_MATH", "1") != "0",
        )
        return y
