"""MegaMoEFusedStrategy: DeepGEMM ``fp8_fp4_mega_moe_fused`` symm-mem kernel.

This is the fused sibling of :class:`MegaMoEStrategy`.  The fused kernel folds
the shared expert *and* the ``routed + shared`` add into the single Mega MoE
kernel: it computes ``dispatch + L1 GEMM + SwiGLU + L2 GEMM + combine`` for the
routed experts AND the shared-expert L1/SwiGLU/L2 pipeline in one launch, with
the shared-expert output accumulated directly into ``y`` (the routed combine
reduction seeds its accumulator from ``y``).  On return ``y`` already holds
``routed + shared``.

Opt-in only: selected when ``DSV4_USE_MEGA_MOE_FUSED=1`` (see
``mega_fused_buf._mega_moe_fused_enabled`` and ``select_strategy``).  The
default MoE path keeps using the non-fused :class:`MegaMoEStrategy`.

Because the shared expert is fused into this kernel, ``routed_includes_shared``
is True — the ``MoE`` layer skips its standalone shared-expert executor and the
``combine_routed_and_shared`` add when this strategy is active.

Differences from :class:`MegaMoEStrategy` (everything else — JIT warmup,
pre-kernel barrier, nvcc tmpdir, T==0 collective semantics — is inherited):

  * routed weights use ``transform_weights_for_mega_moe_fused``
  * the symm buffer comes from ``get_symm_buffer_for_mega_moe_fused``
  * shared-expert FP8 weights are prepared via
    ``transform_shared_expert_weights_for_mega_moe_fused`` and a per-rank
    ``mid_fp8`` / ``mid_sf`` scratch is allocated
  * ``forward`` calls ``deep_gemm.fp8_fp4_mega_moe_fused``
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from ...quant_layouts import FP4_BLOCK, prepare_fp4_weight_scale_for_deepgemm
from ..._profiler import record_function_range
from ..input_packer import get_mega_moe_input_packer
from ..mega_fused_buf import (
    _get_or_create_mega_fused_buf,
    _get_or_create_mega_fused_mid,
    _get_or_create_mega_fused_output,
    _mega_moe_fused_enabled,
)
from ..warmup_sync import sync_cuda_graph_warmup_ranks
from .base import MoeCfg, register_strategy
from .mega import MegaMoEStrategy, _get_gate_pack_kernels, _mega_output_capacity


@register_strategy
class MegaMoEFusedStrategy(MegaMoEStrategy):
    name = "mega_fused"
    # The fused kernel computes routed + shared expert and their add in a
    # single launch, so the MoE layer must NOT run its own shared expert.
    routed_includes_shared = True

    @classmethod
    def can_handle(cls, cfg: MoeCfg) -> bool:
        # Fused Mega requires EP > 1 plus the opt-in env and the fused DeepGEMM
        # entrypoints — all checked by ``_mega_moe_fused_enabled()`` except
        # ep_size > 1, which we check here.
        return cfg.ep_size > 1 and _mega_moe_fused_enabled()

    def setup_weights(self, layer_weights: Dict) -> None:
        """Prepare routed + shared-expert kernel weights and symm/scratch
        buffers for ``fp8_fp4_mega_moe_fused``.

        Routed weights mirror :meth:`MegaMoEStrategy.setup_weights` but use the
        fused weight transform.  Shared-expert weights — popped here so the
        framework drops its references (``MoE`` skips ``W13SharedExpert`` for
        the fused path) — are FP8 e4m3 with UE8M0 block scale; they are
        converted to the INT32 MN-major layout and then to the fused SE UTCCP
        4x32 layout the kernel consumes.
        """
        import deep_gemm
        import torch.distributed as dist

        from rtp_llm.utils.model_weight import W

        cfg = self.cfg
        E = cfg.n_local_experts
        D = cfg.dim
        inter = cfg.moe_inter_dim

        # --- Routed experts (same memory-careful staging as the non-fused
        # strategy; only the final transform differs). -----------------------
        st_w1_w = layer_weights.pop(W.v4_routed_w1_w)
        st_w1_s = layer_weights.pop(W.v4_routed_w1_s)
        st_w3_w = layer_weights.pop(W.v4_routed_w3_w)
        st_w3_s = layer_weights.pop(W.v4_routed_w3_s)
        device = st_w1_w.device

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

        (l1_w, l1_sf), (l2_w, l2_sf) = deep_gemm.transform_weights_for_mega_moe_fused(
            (w13, s13_int),
            (w2, s2_int),
        )
        del w13, s13_int, w2, s2_int
        torch.cuda.empty_cache()

        self._mega_l1_w = l1_w
        self._mega_l1_sf = l1_sf
        self._mega_l2_w = l2_w
        self._mega_l2_sf = l2_sf

        # --- Shared expert: prepare FP8 weights for the fused SE pipeline. ----
        # Popped (not just read) so the framework drops its refs; the fused MoE
        # path does not build ``W13SharedExpert``.
        self._setup_shared_expert_weights(layer_weights, deep_gemm, W, D, inter)

        # --- Symmetric-memory dispatch buffer (fused variant). ---------------
        assert dist.is_initialized(), (
            "Mega MoE fused requires torch.distributed initialised; "
            "_mega_moe_fused_available() should have gated this earlier"
        )
        group = dist.group.WORLD
        self._mega_group = group
        self._mega_buf = _get_or_create_mega_fused_buf(
            group=group,
            num_experts=cfg.n_routed_experts,
            num_max_tokens_per_rank=max(cfg.max_tokens_per_rank, 1),
            num_topk=cfg.n_activated_experts,
            hidden=D,
            intermediate_hidden=inter,
            use_fp8_dispatch=True,
            activation="swiglu",
        )
        capacity = _mega_output_capacity(self._mega_buf, cfg.max_tokens_per_rank)
        self._mega_y = _get_or_create_mega_fused_output(
            capacity,
            D,
            torch.bfloat16,
            device,
        )
        # Shared-expert intermediate-tile scratch (mid_fp8 / mid_sf).
        self._mid_fp8, self._mid_sf = _get_or_create_mega_fused_mid(
            capacity,
            inter,
            device,
        )
        self._input_packer = get_mega_moe_input_packer()
        self._maybe_warmup_jit_once()

    def _setup_shared_expert_weights(self, layer_weights, deep_gemm, W, D, inter):
        """Transform popped FP8 shared-expert weights into the fused SE layout.

        Inputs (V4 checkpoint):
          w13 weight  [2*inter, D]  float8_e4m3fn (gate rows | up rows)
          w13 scale   [2*inter/128, D/128]  float8_e8m0fnu (128x128 per-block)
          w2  weight  [D, inter]    float8_e4m3fn
          w2  scale   [D/128, inter/128]    float8_e8m0fnu

        Output kernel weights keep FP8 e4m3 + INT32 SF in the UTCCP 4x32 layout
        ``fp8_fp4_mega_moe_fused`` consumes (the FP8 weights keep gate/up row
        order; only the SF rows are reordered).
        """
        w13_fp8 = layer_weights.pop(W.v4_shared_w13_w)
        w13_s = layer_weights.pop(W.v4_shared_w13_s)
        w2_fp8 = layer_weights.pop(W.v4_shared_w2_w)
        w2_s = layer_weights.pop(W.v4_shared_w2_s)

        if w13_fp8.shape != (2 * inter, D):
            raise RuntimeError(
                "fused shared w13 weight shape mismatch: "
                f"got {tuple(w13_fp8.shape)}, expected {(2 * inter, D)}"
            )
        if w2_fp8.shape != (D, inter):
            raise RuntimeError(
                "fused shared w2 weight shape mismatch: "
                f"got {tuple(w2_fp8.shape)}, expected {(D, inter)}"
            )

        w13_sf_int = self._shared_expert_sf_to_int(deep_gemm, w13_s, 2 * inter, D)
        w2_sf_int = self._shared_expert_sf_to_int(deep_gemm, w2_s, D, inter)
        del w13_s, w2_s

        (se_l1_fp8, se_l1_sf), (se_l2_fp8, se_l2_sf) = (
            deep_gemm.transform_shared_expert_weights_for_mega_moe_fused(
                (w13_fp8.contiguous(), w13_sf_int),
                (w2_fp8.contiguous(), w2_sf_int),
            )
        )
        self._se_l1_fp8 = se_l1_fp8
        self._se_l1_sf = se_l1_sf
        self._se_l2_fp8 = se_l2_fp8
        self._se_l2_sf = se_l2_sf

    @staticmethod
    def _shared_expert_sf_to_int(deep_gemm, scale, mn, k):
        """Convert a V4 UE8M0 128x128 per-block shared-expert scale to the
        INT32 MN-major TMA-aligned layout (gran=(128,128)).
        """
        if scale.dtype == torch.int32:
            return scale
        if scale.dtype != torch.float8_e8m0fnu:
            raise TypeError(f"expected shared-expert UE8M0 scale, got {scale.dtype}")
        return deep_gemm.transform_sf_into_required_layout(
            scale.float(), mn, k, (128, 128), num_groups=None
        )

    def forward(
        self,
        x: torch.Tensor,  # [T, D] BF16 local-rank tokens
        weights: torch.Tensor,  # [T, topk] FP32 router weights
        indices: torch.Tensor,  # [T, topk] int64 GLOBAL expert IDs
    ) -> torch.Tensor:
        """Run ``fp8_fp4_mega_moe_fused`` (routed + shared, single kernel).

        Returns ``[T, D]`` BF16 already containing ``routed + shared`` — the
        ``MoE`` layer does no further shared-expert add for this strategy.
        """
        import deep_gemm

        T = x.size(0)
        buf = self._mega_buf
        if T > buf.num_max_tokens_per_rank:
            raise RuntimeError(
                f"Mega MoE fused input tokens={T} exceeds num_max_tokens_per_rank="
                f"{buf.num_max_tokens_per_rank} (derived from max_seq_len / "
                f"max_tokens_per_rank). Raise the budget at startup."
            )
        if T > self._mega_y.size(0):
            raise RuntimeError(
                f"Mega MoE fused output buffer rows={self._mega_y.size(0)} is "
                f"smaller than input tokens={T}. This indicates inconsistent "
                "aligned MegaMoE buffer sizing."
            )

        # Like the non-fused kernel, ``fp8_fp4_mega_moe_fused`` is a
        # peer-symmetric NVLink collective: every rank MUST enter together,
        # even when local ``T == 0`` (see MegaMoEStrategy.forward for the full
        # rationale — skipping a rank strands peer-dispatched work and trips
        # the NVLink barrier timeout).  So always pack and always launch.
        self._input_packer.pack(x, weights, indices, buf, T)
        self._maybe_pre_kernel_barrier(T)
        sync_cuda_graph_warmup_ranks(
            f"dsv4.mega_moe_fused.layer{self.cfg.layer_id}.before_deepgemm",
            x.device,
        )

        y = self._mega_y[:T]
        deep_gemm.fp8_fp4_mega_moe_fused(
            y,
            self._se_l1_fp8,
            self._se_l1_sf,
            self._se_l2_fp8,
            self._se_l2_sf,
            self._mid_fp8,
            self._mid_sf,
            (self._mega_l1_w, self._mega_l1_sf),
            (self._mega_l2_w, self._mega_l2_sf),
            buf,
            recipe=(1, 1, FP4_BLOCK),
            activation="swiglu",
            activation_clamp=(
                self.cfg.swiglu_limit if self.cfg.swiglu_limit > 0 else None
            ),
            fast_math=True,
        )
        return y

    def forward_with_gate_pack(
        self,
        x: torch.Tensor,
        gate,
        input_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run MegaMoE fused with router gate + input pack fused together."""
        kernels = _get_gate_pack_kernels()
        if kernels is None:
            raise RuntimeError(
                "MegaMoE gate-pack was selected but kernels are unavailable"
            )
        (
            fused_mega_moe_gate_pack_nonhash,
            fused_mega_moe_gate_pack_hash,
            _,
        ) = kernels

        T = x.size(0)
        buf = self._mega_buf
        if T > buf.num_max_tokens_per_rank:
            raise RuntimeError(
                f"Mega MoE fused input tokens={T} exceeds num_max_tokens_per_rank="
                f"{buf.num_max_tokens_per_rank} (derived from max_seq_len / "
                f"max_tokens_per_rank). Raise the budget at startup."
            )
        if T > self._mega_y.size(0):
            raise RuntimeError(
                f"Mega MoE fused output buffer rows={self._mega_y.size(0)} is "
                f"smaller than input tokens={T}. This indicates inconsistent "
                "aligned MegaMoE buffer sizing."
            )

        with record_function_range("dsv4.moe.gate_linear_bf16"):
            scores_bf16 = F.linear(x, gate._weight_bf16())

        with record_function_range("dsv4.moe.mega_gate_pack"):
            if gate.hash:
                assert input_ids is not None
                fused_mega_moe_gate_pack_hash(
                    x,
                    scores_bf16.contiguous(),
                    input_ids.reshape(-1).contiguous(),
                    gate.tid2eid.contiguous(),
                    buf.x[:T],
                    buf.x_sf[:T],
                    buf.topk_idx[:T],
                    buf.topk_weights[:T],
                    route_scale=float(gate.route_scale),
                    norm_eps=1.0e-12,
                )
            else:
                assert gate.bias is not None
                fused_mega_moe_gate_pack_nonhash(
                    x,
                    scores_bf16.contiguous(),
                    gate.bias.contiguous(),
                    buf.x[:T],
                    buf.x_sf[:T],
                    buf.topk_idx[:T],
                    buf.topk_weights[:T],
                    route_scale=float(gate.route_scale),
                    norm_eps=1.0e-12,
                )

        self._maybe_pre_kernel_barrier(T)
        sync_cuda_graph_warmup_ranks(
            f"dsv4.mega_moe_fused.layer{self.cfg.layer_id}.before_deepgemm",
            x.device,
        )

        y = self._mega_y[:T]
        import deep_gemm

        deep_gemm.fp8_fp4_mega_moe_fused(
            y,
            self._se_l1_fp8,
            self._se_l1_sf,
            self._se_l2_fp8,
            self._se_l2_sf,
            self._mid_fp8,
            self._mid_sf,
            (self._mega_l1_w, self._mega_l1_sf),
            (self._mega_l2_w, self._mega_l2_sf),
            buf,
            recipe=(1, 1, FP4_BLOCK),
            activation="swiglu",
            activation_clamp=(
                self.cfg.swiglu_limit if self.cfg.swiglu_limit > 0 else None
            ),
            fast_math=True,
        )
        return y
