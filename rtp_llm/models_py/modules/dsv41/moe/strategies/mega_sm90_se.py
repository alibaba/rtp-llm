"""SM90 MegaMoE with the local FP8 shared expert fused into the same kernel.

DeepGEMM ``opt_sm90`` (a577b08) folds one local shared expert into
``sm90_fp8_fp4_mega_moe``: the shared GEMMs run on the math warpgroups ahead
of the routed tiles and overlap the dispatch pull, and their BF16 result seeds
the FP32 combine before the routed slots. That removes the standalone
``W13SharedExpert`` launch and its separate output add.

Opt-in through ``DSV4_MOE_STRATEGY=mega_sm90_se``. The routed path, buffer
geometry and staging are inherited unchanged; only the shared weights and the
launch arguments differ.
"""

from __future__ import annotations

import logging
import os

import torch

from .base import MoeCfg, register_strategy
from .mega_sm90 import (
    _SM90_ACTIVATION,
    _SM90_ROUTED_RECIPE,
    SM90MegaMoEStrategy,
)

_SHARED_BLOCK = 128


def _fused_se_enabled() -> bool:
    return os.environ.get("DSV4_MEGA_MOE_FUSED_SE", "1") != "0"


def _has_fused_shared_api() -> bool:
    try:
        import deep_gemm

        buf = deep_gemm.mega_sm90.SymmBufferSM90
    except Exception:  # noqa: BLE001 - older wheel or CPU-only host
        return False
    return "num_shared_experts" in getattr(buf.__init__, "__code__").co_varnames


@register_strategy
class SM90MegaMoEFusedSEStrategy(SM90MegaMoEStrategy):
    """Routed MXFP4 + one local FP8 shared expert in one Hopper kernel."""

    name = "mega_sm90_se"
    routed_includes_shared = True
    # Read by the inherited setup_runtime when it allocates the collective buffer.
    _fused_shared_experts = 1

    @classmethod
    def can_handle(cls, cfg: MoeCfg, explicit: bool = False) -> bool:
        if not _fused_se_enabled() or not _has_fused_shared_api():
            return False
        if cfg.n_shared_experts < 1:
            return False
        # The fused shared GEMM takes 128x128 FP32 block scales; MXFP8 32x32
        # checkpoint scales are folded into the FP8 bytes at weight setup.
        if cfg.shared_fp8_block_size not in (_SHARED_BLOCK, 32):
            return False
        return super().can_handle(cfg, explicit=explicit)

    def setup_weights(self, layer_weights: dict) -> None:
        """Keep the routed transform and add the fused shared weight pairs.

        The kernel wants contiguous FP8 ``[N, K]`` with FP32 ``[N/128, K/128]``
        block scales, and W13 rows ordered gate-then-up. The checkpoint already
        ships that order through the loader-merged ``w13``; only the scale
        dtype/layout has to change.
        """
        from rtp_llm.utils.model_weight import W

        super().setup_weights(layer_weights)

        cfg = self.cfg
        inter, dim = cfg.moe_inter_dim, cfg.dim

        def block_scales(
            weight: torch.Tensor, scale: torch.Tensor, n: int, k: int
        ):
            """Return (fp8 weights, FP32 128x128 scales) for the fused GEMM."""
            if tuple(scale.shape) == (n // _SHARED_BLOCK, k // _SHARED_BLOCK):
                return weight.contiguous(), scale.float().contiguous()
            # MXFP8 32x32 UE8M0: fold the per-block exponent into the FP8 bytes
            # and keep one residual FP32 scale per 128x128 tile.
            import deep_gemm

            folded, residual = deep_gemm.fold_shared_k32_scales(weight, scale)
            logging.info(
                "[SM90 MegaMoE] folded MXFP8 32x32 shared scales to 128x128: "
                "w=%s residual=%s",
                tuple(folded.shape),
                tuple(residual.shape),
            )
            return folded, residual

        w13 = layer_weights.pop(W.v4_shared_w13_w)
        s13 = layer_weights.pop(W.v4_shared_w13_s)
        w2 = layer_weights.pop(W.v4_shared_w2_w)
        s2 = layer_weights.pop(W.v4_shared_w2_s)
        if w13.shape != (2 * inter, dim) or w2.shape != (dim, inter):
            raise ValueError(
                f"fused SM90 shared geometry mismatch: w13={tuple(w13.shape)} "
                f"w2={tuple(w2.shape)} expected {(2 * inter, dim)}/{(dim, inter)}"
            )
        self._shared_l1 = block_scales(w13, s13, 2 * inter, dim)
        self._shared_l2 = block_scales(w2, s2, dim, inter)
        logging.info(
            "[SM90 MegaMoE] fused shared expert enabled: w13=%s w2=%s",
            tuple(w13.shape),
            tuple(w2.shape),
        )

    def setup_runtime(self) -> None:
        super().setup_runtime()
        if getattr(self._mega_buf, "num_shared_experts", 0) != 1:
            raise RuntimeError(
                "fused SM90 shared expert needs a SymmBufferSM90 allocated with "
                "num_shared_experts=1; check deep_gemm >= a577b08"
            )

    def _launch(self, y: torch.Tensor, tokens: int) -> None:
        import deep_gemm

        if y.device.index != torch.cuda.current_device():
            raise RuntimeError(
                "SM90 MegaMoE requires the current CUDA device to match the "
                f"collective buffer device ({y.device} vs cuda:{torch.cuda.current_device()})"
            )
        deep_gemm.sm90_fp8_fp4_mega_moe(
            y,
            self._mega_l1,
            self._mega_l2,
            self._mega_buf,
            recipe=_SM90_ROUTED_RECIPE,
            activation=_SM90_ACTIVATION,
            activation_clamp=(
                self.cfg.swiglu_limit if self.cfg.swiglu_limit > 0 else None
            ),
            fast_math=True,
            num_tokens_bound=tokens,
            shared_l1_weights=self._shared_l1,
            shared_l2_weights=self._shared_l2,
        )
