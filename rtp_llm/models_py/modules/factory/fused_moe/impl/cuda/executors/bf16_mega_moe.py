"""DeepGEMM ``bf16_mega_moe`` fused expert executor.

The BF16 twin of :class:`MegaMoeExecutor`. Everything that is not about
quantization -- the symmetric-memory buffer, its single-layer sharing, the
capacity checks, JIT warmup, and the "every rank must enter the kernel" rule --
is inherited; only four things differ:

* the dispatch carries BF16 activations, so the symmetric buffer is sized for
  ``mma_type="bf16xbf16"`` and the input pack is a plain copy with no
  per-token scale;
* ``transform_weights_for_mega_moe`` takes and returns bare weight tensors
  rather than ``(weight, scale)`` pairs;
* the intermediate width needs no FP4-block padding -- the kernel's L2 K
  dimension is the intermediate itself and its ``block_k`` for BF16 is 64, so
  the checkpoint's 640 is already legal;
* ``deep_gemm.bf16_mega_moe`` replaces ``fp8_fp4_mega_moe``.

Weights are refit in place (see :meth:`refresh_routed_weights`) so the buffers
the captured decode graph recorded keep their addresses across an RL weight
update.
"""

from __future__ import annotations

import logging
import weakref
from typing import Any, Dict

import torch

from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.input_packer import (
    MegaMoEInputPacker,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.warmup_sync import (
    sync_cuda_graph_warmup_ranks,
)
from rtp_llm.utils.model_weight import W

from .fp8_fp4_base import split_moe_w13_gate_up
from .mega_moe import MegaMoeExecutor


# Every live BF16 MegaMoE layer, so a weight refit can re-derive all of them.
# Weak, so a destroyed engine does not keep its executors alive.
_LIVE_EXECUTORS: "weakref.WeakSet" = weakref.WeakSet()


def refresh_all_routed_weights() -> int:
    """Re-derive every live layer's MegaMoE layout from its source weights.

    An RL weight update writes the layers' source tensors in place; the kernel
    reads only the transformed copy, so something has to re-derive it. Nothing
    in the engine notifies a module that a refit committed, and the source
    tensors are inference tensors, which carry no version counter to poll -- so
    the refit path calls this once after COMMIT_UPDATE instead.
    """
    refreshed = 0
    for executor in list(_LIVE_EXECUTORS):
        executor.refresh_routed_weights()
        refreshed += 1
    return refreshed


class Bf16MegaMoEInputPacker(MegaMoEInputPacker):
    """Copy BF16 tokens and routing straight into the symmetric buffer."""

    name = "bf16"

    def pack(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        buf,
        tokens: int,
    ) -> None:
        if x.dtype != torch.bfloat16:
            raise ValueError(
                f"BF16 MegaMoE dispatch requires bfloat16 activations, got {x.dtype}"
            )
        buf.x[:tokens].copy_(x)
        buf.topk_idx[:tokens].copy_(indices)
        buf.topk_weights[:tokens].copy_(weights)


class Bf16MegaMoeExecutor(MegaMoeExecutor):
    mma_type = "bf16xbf16"
    use_fp8_dispatch = False

    @staticmethod
    def _intermediate_size(cfg) -> int:
        # No FP4 block alignment to satisfy: the BF16 kernel's block_k is 64 and
        # the checkpoint intermediate (640 for Qwen3.8-Flash-Next) divides it.
        return int(cfg.moe_inter_dim)

    @property
    def supports_gate_pack(self) -> bool:
        # The fused gate packer writes the FP8 per-token scale plane, which the
        # BF16 buffer does not have.
        return False

    @classmethod
    def check_conditions(cls, checker: Any, config) -> None:
        from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
            MoeConfigResolver,
        )
        from .mega_moe import _mega_moe_available

        checker.check(MoeConfigResolver().get_quant_method(config) is None)
        checker.check(config.ep_size > 1)
        checker.check(config.world_size == config.ep_size)
        checker.check(config.world_rank == config.ep_rank)
        checker.check(_mega_moe_available())

    def _transform_routed_weights(
        self, layer_weights: Dict, deep_gemm, inter: int
    ) -> torch.device:
        """Derive the MegaMoE layout, leaving the source weights in the layer.

        Routed weights arrive already EP-sliced as ``[E_local, 2*inter, dim]``
        and ``[E_local, dim, inter]``. The only reordering is gate|up, which
        ``transform_weights_for_mega_moe`` then interleaves for the kernel.

        Unlike the FP8 executor, these are read and **not popped**. An RL weight
        update writes the layer's live tensors in place, and a key the layer no
        longer holds is rejected outright by the refit transaction
        (``rl_weight_update.py``: ``unexpected or repeated internal weight``).
        Keeping them is what makes this path refittable; the cost is the source
        copy staying resident next to the transformed one.
        """
        cfg = self.cfg
        w13 = layer_weights[W.moe_w1]
        w2 = layer_weights[W.moe_w2]
        if w13.dtype != torch.bfloat16 or w2.dtype != torch.bfloat16:
            raise ValueError(
                "BF16 MegaMoE requires bfloat16 routed weights, got "
                f"{w13.dtype} and {w2.dtype}"
            )
        self._source_l1 = w13
        self._source_l2 = w2
        l1_w, l2_w = self._derive(deep_gemm, w13, w2, inter)
        self._mega_l1_w = l1_w
        self._mega_l1_sf = None
        self._mega_l2_w = l2_w
        self._mega_l2_sf = None
        _LIVE_EXECUTORS.add(self)
        return w13.device

    def _derive(self, deep_gemm, w13, w2, inter: int):
        gate_w, up_w = split_moe_w13_gate_up(w13, inter, self.cfg.moe_w1_layout)
        if self.cfg.moe_w1_layout != "gate_up":
            w13 = torch.cat((gate_w, up_w), dim=-2)
        del gate_w, up_w
        l1_w, l2_w = deep_gemm.transform_weights_for_mega_moe(w13, w2)
        return l1_w, l2_w

    def _make_input_packer(self):
        return Bf16MegaMoEInputPacker()

    @torch.inference_mode()
    def refresh_routed_weights(self) -> None:
        """Re-derive the MegaMoE layout in place when the sources have changed.

        Called by :func:`refresh_all_routed_weights` after a refit commits.
        The captured decode graph recorded the addresses of ``_mega_l1_w`` and
        ``_mega_l2_w``, so the fresh layout is copied into the existing tensors
        rather than rebound. A shape or dtype change would mean the checkpoint
        geometry moved, which no refit may do, so it raises instead of silently
        reallocating and invalidating the graph.
        """
        import deep_gemm

        resident = (self._mega_l1_w, self._mega_l2_w)
        pointers = tuple(tensor.data_ptr() for tensor in resident)
        fresh = self._derive(
            deep_gemm,
            self._source_l1,
            self._source_l2,
            self._intermediate_size(self.cfg),
        )
        for old, new in zip(resident, fresh):
            if tuple(new.shape) != tuple(old.shape) or new.dtype != old.dtype:
                raise RuntimeError(
                    "BF16 MegaMoE refit changed the routed weight layout "
                    f"({tuple(old.shape)} {old.dtype} -> {tuple(new.shape)} "
                    f"{new.dtype}); the captured decode graph cannot follow it"
                )
            old.copy_(new)
        del fresh
        if tuple(t.data_ptr() for t in resident) != pointers:
            raise RuntimeError("BF16 MegaMoE refit moved a resident weight buffer")
        logging.info(
            "[MegaMoE] layer=%d routed weights re-derived in place after refit",
            self.cfg.layer_id,
        )

    def _launch(self, y: torch.Tensor, tokens: int, device: torch.device) -> None:
        import deep_gemm


        sync_cuda_graph_warmup_ranks(
            f"moe.bf16_mega_moe.layer{self.cfg.layer_id}.before_deepgemm",
            device,
        )
        deep_gemm.bf16_mega_moe(
            y,
            self._mega_l1_w,
            self._mega_l2_w,
            self._mega_buf,
            activation="swiglu",
            activation_clamp=(
                self.cfg.swiglu_limit if self.cfg.swiglu_limit > 0 else None
            ),
            # Fixed at build time, so it does not affect batch invariance; it
            # moves the result by ~1e-7 relative and under 2% of the runtime.
            fast_math=True,
        )
