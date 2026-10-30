"""SM90MegaMoEStrategy: DeepGEMM ``sm90_fp8_fp4_mega_moe`` Hopper fused kernel.

Default native SM90 strategy; ``DSV4_MOE_STRATEGY=mega_sm90`` selects it explicitly.
The Hopper kernel fuses dispatch + L1 GEMM + SwiGLU + L2 GEMM + combine over a
``SymmBufferSM90`` collective buffer, but its protocol has no shared-expert
fusion, so the ``MoE`` layer keeps its standalone shared expert
(``routed_includes_shared = False``).

Weights are re-encoded at load time by
``deep_gemm.transform_weights_for_mega_moe_sm90`` into the kernel's processed
MXFP4 triples. That preparation is lossy when an expert's K32 exponent span
exceeds 11 (small groups are requantized to E2M1); it is not a permutation of
the checkpoint bytes. See DeepGEMM ``opt_sm90/h20_mega_moe_sm90.md``.
"""

from __future__ import annotations

from typing import Dict

import torch

from ..mega_sm90_buf import (
    _get_or_create_mega_sm90_buf,
    _get_or_create_mega_sm90_output,
    _sm90_mega_moe_available,
    _sm90_mega_moe_enabled,
    _sm90_mega_moe_unavailable_reason,
    _sm90_shape_supported,
)
from ..mega_sm90_input_packer import get_mega_sm90_input_packer
from .base import MoeCfg, RoutedExpertsStrategy, register_strategy

# The SM90 kernel accepts exactly one routed weight recipe: K32 UE8M0 scales
# with 16 packed E2M1 values per scale group.
_SM90_ROUTED_RECIPE = (1, 1, 32)
_SM90_ACTIVATION = "swiglu"
# L20Z/SM90 decode benchmarking for the V4.1 EP8 geometry shows that the
# register-A tiled transform is consistently faster than the row/shared-memory
# transform for the small token counts served here.  Split-K is intentionally
# left disabled in the kernel call because it regresses this workload.
_SM90_WEIGHT_LAYOUT = "tiled"


@register_strategy
class SM90MegaMoEStrategy(RoutedExpertsStrategy):
    """Run routed MXFP4 experts through the fused Hopper MegaMoE kernel."""

    name = "mega_sm90"
    routed_includes_shared = False

    @classmethod
    def can_handle(cls, cfg: MoeCfg, explicit: bool = False) -> bool:
        """Whether this backend can run ``cfg``.

        ``explicit`` is True when the operator named ``mega_sm90`` directly
        (``DSV4_MOE_STRATEGY=mega_sm90``). Naming it is itself the opt-in, so
        the ``DSV4_USE_MEGA_MOE_SM90`` switch is only required on the
        automatic path.
        """
        if not explicit and not _sm90_mega_moe_enabled():
            return False
        if not _sm90_mega_moe_available():
            return False
        return _sm90_shape_supported(
            cfg.dim,
            cfg.moe_inter_dim,
            cfg.n_routed_experts,
            cfg.ep_size,
            cfg.n_activated_experts,
        )

    def setup_weights(self, layer_weights: Dict) -> None:
        """Re-encode checkpoint MXFP4 experts into the SM90 processed triples.

        Routed weights arrive already EP-sliced as ``[E_local, ...]`` packed
        E2M1 weights plus natural K32 UE8M0 scales. Unlike the SM100 path this
        must NOT go through ``prepare_fp4_weight_scale_for_deepgemm``: the
        Hopper transform consumes natural scales and produces its own opaque
        physical layout.
        """
        import deep_gemm

        from rtp_llm.utils.model_weight import W

        cfg = self.cfg
        E = cfg.n_local_experts
        D = cfg.dim
        inter = cfg.moe_inter_dim
        device = layer_weights[W.v4_routed_w1_w].device

        w13 = torch.empty((E, 2 * inter, D // 2), dtype=torch.int8, device=device)
        s13 = torch.empty(
            (E, 2 * inter, D // 32), dtype=torch.float8_e8m0fnu, device=device
        )
        w13[:, :inter].copy_(layer_weights.pop(W.v4_routed_w1_w))
        s13[:, :inter].copy_(layer_weights.pop(W.v4_routed_w1_s))
        w13[:, inter:].copy_(layer_weights.pop(W.v4_routed_w3_w))
        s13[:, inter:].copy_(layer_weights.pop(W.v4_routed_w3_s))
        w2 = torch.empty((E, D, inter // 2), dtype=torch.int8, device=device)
        s2 = torch.empty((E, D, inter // 32), dtype=torch.float8_e8m0fnu, device=device)
        w2.copy_(layer_weights.pop(W.v4_routed_w2_w))
        s2.copy_(layer_weights.pop(W.v4_routed_w2_s))
        torch.cuda.empty_cache()

        self._mega_l1, self._mega_l2 = deep_gemm.transform_weights_for_mega_moe_sm90(
            (w13, s13), (w2, s2), layout=_SM90_WEIGHT_LAYOUT
        )
        del w13, s13, w2, s2
        torch.cuda.empty_cache()
        self._mega_runtime_device = device

    def setup_runtime(self) -> None:
        """Allocate the collective buffer and output rows outside the weight pool."""
        import torch.distributed as dist

        cfg = self.cfg
        assert (
            dist.is_initialized()
        ), "SM90 MegaMoE requires torch.distributed initialised; can_handle() should have gated this"
        group = dist.group.WORLD
        self._mega_group = group
        self._mega_buf = _get_or_create_mega_sm90_buf(
            group=group,
            num_experts=cfg.n_routed_experts,
            num_max_tokens_per_rank=max(cfg.max_tokens_per_rank, 1),
            num_topk=cfg.n_activated_experts,
            hidden=cfg.dim,
            intermediate_hidden=cfg.moe_inter_dim,
            activation=_SM90_ACTIVATION,
        )
        # DeepGEMM aligns the capacity internally; keep the output rows at
        # least that wide so a kernel write can never exceed the allocation.
        capacity = max(
            int(cfg.max_tokens_per_rank),
            int(getattr(self._mega_buf, "num_max_tokens_per_rank", 0) or 0),
            1,
        )
        self._mega_y = _get_or_create_mega_sm90_output(
            capacity,
            cfg.dim,
            torch.bfloat16,
            self._mega_runtime_device,
        )
        self._input_packer = get_mega_sm90_input_packer()

    def forward(
        self, x: torch.Tensor, weights: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        """Run the fused Hopper kernel and return BF16 ``[T, dim]`` routed rows."""
        import deep_gemm

        if self._mega_buf is None or self._mega_y is None:
            self.setup_runtime()
        T = x.size(0)
        buf = self._mega_buf
        self._input_packer.pack(x, weights, indices, buf, T)
        y = self._mega_y[:T]
        if y.device.index != torch.cuda.current_device():
            raise RuntimeError(
                "SM90 MegaMoE requires the current CUDA device to match the "
                f"collective buffer device ({y.device} vs cuda:{torch.cuda.current_device()})"
            )
        deep_gemm.sm90_fp8_fp4_mega_moe(
            y,
            self._mega_l1,
            self._mega_l2,
            buf,
            recipe=_SM90_ROUTED_RECIPE,
            activation=_SM90_ACTIVATION,
            activation_clamp=(
                self.cfg.swiglu_limit if self.cfg.swiglu_limit > 0 else None
            ),
            fast_math=True,
            # Every EP rank stages the same local row count, so the per-rank
            # count is also the common upper bound the kernel requires.
            num_tokens_bound=T,
        )
        return y


def sm90_mega_moe_unavailable_reason() -> str | None:
    """Expose the gate reason so callers can explain a fallback."""
    return _sm90_mega_moe_unavailable_reason()

