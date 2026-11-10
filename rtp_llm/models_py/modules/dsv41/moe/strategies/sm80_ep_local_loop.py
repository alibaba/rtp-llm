"""Pre-SM100 (A100/sm80, L20/sm89) expert-parallel routed-experts strategy.

MegaMoEStrategy requires SM100 symmetric memory plus DeepGEMM mega kernels;
DeepEP requires SM90+. Neither can run on sm80, and ``select_strategy`` fails
fast for ``ep_size > 1`` without them — which strands pre-Blackwell GPUs even
though V4.1's ~265 GB of packed FP4 experts cannot fit on one card, making EP
mandatory there.

This strategy closes that gap with the replication property the fused
attention path already guarantees: ``_prefill_output_all_reduce`` reduces the
attention output over the TP group, so every rank enters the MoE with the same
activations. Each rank therefore runs the inherited LocalLoop compute (exact
e2m1/ue8m0 -> BF16 dequant references on its ``[local_expert_start,
local_expert_end)`` slice, fixed-shape and CUDA-graph-safe) over ALL tokens,
and a single SUM allreduce over the same TP process group merges the partial
expert contributions. This is mathematically identical to an all-to-all
dispatch/combine: token t's output is the routing-weighted sum over its top-k
experts, and each expert lives on exactly one rank.

Trade-offs vs. dispatch-based EP (documented, deliberate for round 0):
  * every rank reads its full local expert slice per step regardless of how
    few tokens route to it (Mega/DeepEP skip cold experts) — bandwidth-bound,
    optimized in later rounds with a Triton fused dequant grouped GEMM;
  * NCCL allreduce per MoE layer instead of point-to-point combine — fine on
    single-node NVLink, and NCCL on a warmed communicator is CUDA-graph
    capturable (``setup_runtime`` warms it outside capture).

Unsupported shapes raise explicitly instead of silently mis-parallelizing:
``dp_size > 1`` or ``ep_size != world_size`` would make the EP span differ
from the TP group this strategy reduces over.
"""

from __future__ import annotations

import logging

import torch

from rtp_llm.models_py.modules.dsv41.moe.strategies.base import MoeCfg, register_strategy
from rtp_llm.models_py.modules.dsv41.moe.strategies.local_loop import LocalLoopStrategy

logger = logging.getLogger(__name__)


@register_strategy
class Sm80EpLocalLoopStrategy(LocalLoopStrategy):
    name = "sm80_ep_local_loop"

    @classmethod
    def can_handle(cls, cfg: MoeCfg) -> bool:
        from rtp_llm.models_py.modules.dsv41.sm80_support import (
            use_reference_gemm_paths,
        )

        return (
            use_reference_gemm_paths()
            and cfg.ep_size > 1
            and cfg.dp_size <= 1
        )

    def setup_runtime(self) -> None:
        super().setup_runtime()
        cfg = self.cfg
        if cfg.ep_size <= 1:
            return
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            raise RuntimeError(
                f"{self.name} requires an initialized process group for EP={cfg.ep_size}"
            )
        world = torch.distributed.get_world_size(group=self._ep_group())
        if cfg.dp_size > 1 or world != cfg.ep_size:
            raise RuntimeError(
                f"{self.name} supports dp_size==1 with the EP span equal to the TP "
                f"group (world={world}, ep_size={cfg.ep_size}, dp_size={cfg.dp_size}); "
                "other EP shapes are not supported on pre-SM100 GPUs"
            )
        if not torch.cuda.is_current_stream_capturing():
            # Warm the NCCL communicator outside graph capture so the in-graph
            # combine never triggers communicator setup mid-capture.
            probe = torch.zeros(1, device=torch.device("cuda", torch.cuda.current_device()))
            torch.distributed.all_reduce(probe, group=self._ep_group())
            logger.info(
                "[dsv4-sm80] EP combine ready: layer=%d ep_rank=%d local_experts=[%d,%d) "
                "(replicated-input LocalLoop + TP-group SUM allreduce)",
                cfg.layer_id,
                cfg.ep_rank,
                cfg.local_expert_start,
                cfg.local_expert_end,
            )

    def _ep_group(self):
        from rtp_llm.models_py.distributed.collective_torch import Group, _get_group

        return _get_group(Group.TP)

    def forward(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        expert_load=None,
        eplb_layer=None,
    ) -> torch.Tensor:
        self.finalize_routing(indices, weights, eplb_layer)
        y = self._forward_into_buf(
            x,
            weights,
            indices,
            local_start=self.cfg.local_expert_start,
            local_end=self.cfg.local_expert_end,
        )
        if self.cfg.ep_size > 1:
            # Inputs are TP-replicated; SUM folds each rank's local-expert
            # partial sum into the complete routed output on every rank.
            torch.distributed.all_reduce(y, group=self._ep_group())
        return y
