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

from typing import Dict, Optional

import torch

from .base import MoeCfg, RoutedExpertsStrategy, register_strategy
from ..mega_buf import _get_or_create_mega_buf, _mega_moe_available
from ..quant_layouts import FP4_BLOCK, _per_token_cast_to_fp8_packed_ue8m0


@register_strategy
class MegaMoEStrategy(RoutedExpertsStrategy):
    name = "mega"

    @classmethod
    def can_handle(cls, cfg: MoeCfg) -> bool:
        # Mega requires EP > 1, factory mode, SM100, dist-init — all checked
        # by ``_mega_moe_available()`` except ep_size > 1, which we check here.
        return cfg.factory_mode and cfg.ep_size > 1 and _mega_moe_available()

    def setup_weights(
        self,
        weights: Optional[Dict[str, torch.Tensor]],
        prefix: str,
    ) -> None:
        """Stack EP-local routed expert weights, convert SFs to the
        int32 UTCCP-transposed layout required by ``fp8_fp4_mega_moe``,
        and register the symmetric-memory dispatch buffer.

        V4 ckpt stores, per expert i:
          w1.weight [inter, dim//2] int8  (FP4 gate)
          w3.weight [inter, dim//2] int8  (FP4 up)
          w2.weight [dim, inter//2] int8  (FP4 down)
          ... .scale [inter, dim//FP4_BLOCK] / [dim, inter//FP4_BLOCK] UE8M0

        Mega MoE expects, per expert:
          L1 w [2*inter, dim//2] int8 (gate | up rows concatenated)
          L1 sf [2*inter, ...] int32  (post-``transform_sf_into_required_layout``
            + ``transform_weights_for_mega_moe``: gate/up interleaved gran=8
            along N, SF UTCCP-transposed)
          L2 w [dim, inter//2] int8
          L2 sf [dim, ...] int32

        Pops the following keys from ``weights`` (factory_mode):
          ``{prefix}.experts.{i}.{w1,w2,w3}.{weight,scale}`` for
          i in [local_expert_start, local_expert_end).

        Eager init (weights=None) is not supported — Mega is a factory-mode
        path; tests that need a Mega instance must construct it with synthetic
        stacked weights via ``setup_weights`` themselves.
        """
        if weights is None:
            raise RuntimeError(
                "MegaMoEStrategy requires factory_mode (weights dict). "
                "Eager init is not supported."
            )

        import deep_gemm
        import torch.distributed as dist

        cfg = self.cfg
        E = cfg.n_local_experts
        D = cfg.dim
        inter = cfg.moe_inter_dim
        start = cfg.local_expert_start
        device = weights[f"{prefix}.experts.{start}.w1.weight"].device

        # (1) Stack EP-local experts into [E_local, ...] tensors.  SF is
        # kept as fp32 initially — ``transform_sf_into_required_layout``
        # is the only entry point that produces the int32 layout the
        # kernel consumes.
        w13 = torch.empty((E, 2 * inter, D // 2), dtype=torch.int8, device=device)
        s13 = torch.empty(
            (E, 2 * inter, D // FP4_BLOCK), dtype=torch.float32, device=device
        )
        w2 = torch.empty((E, D, inter // 2), dtype=torch.int8, device=device)
        s2 = torch.empty((E, D, inter // FP4_BLOCK), dtype=torch.float32, device=device)
        for local_i in range(E):
            p = f"{prefix}.experts.{start + local_i}"
            w13[local_i, :inter].copy_(weights.pop(f"{p}.w1.weight"))
            s13[local_i, :inter].copy_(weights.pop(f"{p}.w1.scale").float())
            w13[local_i, inter:].copy_(weights.pop(f"{p}.w3.weight"))
            s13[local_i, inter:].copy_(weights.pop(f"{p}.w3.scale").float())
            w2[local_i].copy_(weights.pop(f"{p}.w2.weight"))
            s2[local_i].copy_(weights.pop(f"{p}.w2.scale").float())

        # (2) Repack SFs to the kernel's required int32 layout (MN-major,
        # TMA-aligned, packed 4-UE8M0-per-int32).
        s13_int = deep_gemm.transform_sf_into_required_layout(
            s13, 2 * inter, D, (1, FP4_BLOCK), E
        )
        s2_int = deep_gemm.transform_sf_into_required_layout(
            s2, D, inter, (1, FP4_BLOCK), E
        )

        # (3) Apply Mega MoE-specific transforms: L1 gate/up interleave
        # (gran=8 along N) + both SFs UTCCP-transposed.  Returns the
        # final weights that the kernel reads directly each forward.
        (l1_w, l1_sf), (l2_w, l2_sf) = deep_gemm.transform_weights_for_mega_moe(
            (w13, s13_int),
            (w2, s2_int),
        )

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
        # Pre-allocate static output buffer — avoids torch.empty((T, D)) inside the
        # forward, which reallocates on every step and blocks CUDA graph capture.
        # Sized to max_tokens_per_rank; forward slices [:T] for the live batch.
        self._mega_y = torch.empty(
            (max(cfg.max_tokens_per_rank, 1), D),
            dtype=torch.bfloat16,
            device=device,
        )

    def forward(
        self,
        x: torch.Tensor,        # [T, D] BF16 local-rank tokens
        weights: torch.Tensor,  # [T, topk] FP32 router weights
        indices: torch.Tensor,  # [T, topk] int64 GLOBAL expert IDs
    ) -> torch.Tensor:
        """Run the fused DeepGEMM Mega MoE kernel: dispatch + L1 GEMM +
        SwiGLU + L2 GEMM + combine — all fused, symm-mem backed.

        Returns the combined routed-expert output in FP32 (to match the
        contract of ``DeepEPStrategy`` / ``LocalLoopStrategy``).
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
        if T == 0:
            return torch.zeros_like(x, dtype=torch.float32)

        # Per-token FP8 cast with packed UE8M0 group-32 scale — the
        # dispatch side of Mega MoE reads this layout directly.
        # Inline impl avoids deep_gemm's pack_ue8m0_to_int .all() assertion
        # which does a CUDA→CPU sync illegal during stream capture.
        x_fp8, x_sf = _per_token_cast_to_fp8_packed_ue8m0(x.contiguous(), gran_k=32)
        # Fill the symm-mem buffer slots.  Only the first T rows are
        # meaningful; the remainder was zero-initialised at buffer
        # alloc (0 is expert 0, but tokens past T aren't read because
        # the kernel uses y.size(0) as the effective token count).
        buf.x[:T].copy_(x_fp8)
        buf.x_sf[:T].copy_(x_sf)
        buf.topk_idx[:T].copy_(indices.to(torch.int64).contiguous())
        buf.topk_weights[:T].copy_(weights.to(torch.float32).contiguous())

        y = self._mega_y[:T]
        deep_gemm.fp8_fp4_mega_moe(
            y,
            (self._mega_l1_w, self._mega_l1_sf),
            (self._mega_l2_w, self._mega_l2_sf),
            buf,
            recipe=(1, 1, FP4_BLOCK),
            activation="swiglu",
            activation_clamp=self.cfg.swiglu_limit if self.cfg.swiglu_limit > 0 else None,
            fast_math=True,
        )
        return y.float()
