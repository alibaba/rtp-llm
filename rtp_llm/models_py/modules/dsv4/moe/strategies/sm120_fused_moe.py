"""Standalone SM120 routed-MoE strategy.

SM120 in the current CUDA 13 environment has no compatible DeepEP cubin.
This strategy therefore contains no DeepEP dependency: local expert compute
uses the generic FusedMoe executor and communication uses the common Factory
PureCP all-gather/reduce-scatter router for prefill, or a correctness
all-gather/all-reduce fallback for non-CP/CUDA-graph execution.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Dict

import torch

from rtp_llm.models_py.utils.arch import is_sm120

from ..._profiler import record_function_range
from ..sm120_fused_moe import build_sm120_fused_moe
from .base import MoeCfg, RoutedExpertsStrategy, register_strategy
from .grouped_fp4 import GroupedFP4Strategy, _has_fp8_fp4_grouped_kernel
from .local_loop import LocalLoopStrategy

_SM120_CP_FUSED_MOE_LOGGED = False


def _is_sm120_runtime() -> bool:
    # The kernels are compiled for the exact SM120 (12.0) target.  Do not
    # treat other SM12x devices as compatible merely because they share the
    # major capability number.
    return is_sm120()


def _validate_world_collective_topology(cfg: MoeCfg, dist, group) -> tuple[int, int]:
    world = dist.get_world_size(group)
    rank = dist.get_rank(group)
    if world != cfg.ep_size or rank != cfg.ep_rank:
        raise RuntimeError(
            "SM120 WORLD collective topology does not match the expert "
            f"partition: world/rank={world}/{rank}, "
            f"ep_size/ep_rank={cfg.ep_size}/{cfg.ep_rank}. "
            "A dedicated EP process group is required for mixed DP/TP+EP."
        )
    return world, rank


@register_strategy
class Sm120FusedMoeStrategy(RoutedExpertsStrategy):
    """SM120 FusedMoe compute with collective communication routers."""

    name = "sm120_fused_moe"
    requires_synchronized_chunk_schedule = False

    def __init__(self, cfg: MoeCfg):
        super().__init__(cfg)
        # Pure CP guarantees equal local token shapes and uses its own TP
        # router.  Only the variable-size WORLD-collective fallback needs the
        # explicit cross-rank outer chunk schedule.
        self.requires_synchronized_chunk_schedule = not self._is_pure_cp()
        self._chunk_extent_tensor: torch.Tensor | None = None
        self._sm120_grouped = None
        if _is_sm120_runtime() and _has_fp8_fp4_grouped_kernel():
            # GroupedFP4 owns only this EP rank's weights and therefore uses
            # a local [0, n_local_experts) expert-id space.  The FusedMoe
            # executor masks/remaps global ids before invoking it.
            local_cfg = replace(
                cfg,
                ep_size=1,
                ep_rank=0,
                n_routed_experts=cfg.n_local_experts,
                local_expert_start=0,
                local_expert_end=cfg.n_local_experts,
            )
            self._sm120_grouped = GroupedFP4Strategy(local_cfg)
            self._local = self._sm120_grouped
        else:
            self._local = LocalLoopStrategy(cfg)

        self._fused_moe = build_sm120_fused_moe(
            cfg,
            self._local,
            uses_grouped_fp4=self._sm120_grouped is not None,
        )
        # Decode/DP never enters the CP router.  Do not construct an unused
        # second FusedMoe tree (and duplicate the local strategy registration)
        # on that path.
        self._cp_fused_moe = None
        if self._is_pure_cp():
            # This router imports CUDA-only quantization bindings.  Delay the
            # import until exact-SM120 selection so importing the DSV4 strategy
            # package remains valid on ROCm and CPU runtimes.
            from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_cp_router import (
                PureCpRouterNoQuant,
            )

            self._cp_fused_moe = build_sm120_fused_moe(
                cfg,
                self._local,
                uses_grouped_fp4=self._sm120_grouped is not None,
                router_cls=PureCpRouterNoQuant,
            )

    @classmethod
    def can_handle(cls, cfg: MoeCfg) -> bool:
        return cfg.ep_size > 1 and _is_sm120_runtime()

    def setup_weights(self, layer_weights: Dict) -> None:
        self._local.setup_weights(layer_weights)

    def synchronized_chunk_extent(self, local_tokens: int, device: torch.device) -> int:
        """Agree on one eager-prefill chunk count across all EP ranks."""
        dist = torch.distributed
        if not dist.is_initialized():
            raise RuntimeError("SM120 collective MoE requires torch.distributed")
        group = dist.group.WORLD
        _validate_world_collective_topology(self.cfg, dist, group)
        max_tokens = self._chunk_extent_tensor
        if max_tokens is None or max_tokens.device != device:
            max_tokens = torch.empty((1,), dtype=torch.int64, device=device)
            self._chunk_extent_tensor = max_tokens
        max_tokens.fill_(int(local_tokens))
        dist.all_reduce(max_tokens, op=dist.ReduceOp.MAX, group=group)
        return int(max_tokens.item())

    def forward(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        if not x.is_cuda or not is_sm120(x.device):
            raise RuntimeError("Sm120FusedMoeStrategy requires an SM120 CUDA device")
        if self._is_pure_cp() and not torch.cuda.is_current_stream_capturing():
            return self._forward_cp_reduce_scatter(x, weights, indices)
        return self._forward_collective(x, weights, indices)

    def _is_pure_cp(self) -> bool:
        cfg = self.cfg
        return bool(
            cfg.cp_enabled
            and cfg.cp_size > 1
            and cfg.cp_size == cfg.moe_tp_size == cfg.ep_size
        )

    def _forward_cp_reduce_scatter(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """All-gather inputs, run local FusedMoe, reduce-scatter outputs."""

        if self._cp_fused_moe is None:
            raise RuntimeError("SM120 CP FusedMoe was not initialized")

        global _SM120_CP_FUSED_MOE_LOGGED
        if not _SM120_CP_FUSED_MOE_LOGGED:
            _SM120_CP_FUSED_MOE_LOGGED = True
            logging.info(
                "[DSV4 MoE] SM120 FusedMoe with CP all_gather + "
                "reduce_scatter: executor=%s cp_size=%d local_tokens=%d "
                "hidden=%d",
                type(self._cp_fused_moe.fused_experts).__name__,
                self.cfg.cp_size,
                x.size(0),
                x.size(1),
            )

        local_tokens = x.size(0)
        output = torch.empty(
            (local_tokens, x.size(1)),
            dtype=torch.float32,
            device=x.device,
        )
        if local_tokens == 0:
            return output

        # The MoE layer owns the configurable outer chunking.  Keep one
        # collective/compute pass per outer chunk so the common PureCP router
        # does not add another fixed-size split or repeat communication.
        with record_function_range("dsv4.moe.cp.window"):
            with record_function_range("dsv4.moe.cp.fused_moe"):
                reduced = self._cp_fused_moe(x, weights, indices)
        output.copy_(reduced)
        return output

    def _forward_collective(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """SM120 correctness fallback: all-gather + FusedMoe + all-reduce."""

        dist = torch.distributed
        if not dist.is_initialized():
            raise RuntimeError("SM120 collective MoE requires torch.distributed")
        group = dist.group.WORLD
        world, rank = _validate_world_collective_topology(self.cfg, dist, group)
        if torch.cuda.is_current_stream_capturing():
            # Decode graphs use a fixed equal batch shape on every rank.
            counts = [x.size(0)] * world
        else:
            n = torch.full((1,), x.size(0), dtype=torch.int64, device=x.device)
            counts_t = [torch.empty_like(n) for _ in range(world)]
            dist.all_gather(counts_t, n, group=group)
            counts = [int(value.item()) for value in counts_t]
        max_n = max(max(counts), 1)

        def gather_padded(tensor: torch.Tensor, fill: float | int):
            padded = torch.full(
                (max_n, *tensor.shape[1:]),
                fill,
                dtype=tensor.dtype,
                device=tensor.device,
            )
            if tensor.size(0):
                padded[: tensor.size(0)].copy_(tensor)
            gathered = [torch.empty_like(padded) for _ in range(world)]
            dist.all_gather(gathered, padded, group=group)
            return gathered

        xs = gather_padded(x, 0.0)
        ws = gather_padded(weights, 0.0)
        ids = gather_padded(indices.to(torch.int64), -1)
        all_x = torch.cat([tensor[:count] for tensor, count in zip(xs, counts)], 0)
        all_w = torch.cat([tensor[:count] for tensor, count in zip(ws, counts)], 0)
        all_i = torch.cat([tensor[:count] for tensor, count in zip(ids, counts)], 0)

        local_begin = sum(counts[:rank])
        local_end = local_begin + counts[rank]
        if all_x.size(0) == 0:
            return torch.empty(
                (counts[rank], all_x.size(1)),
                dtype=torch.float32,
                device=all_x.device,
            )
        with record_function_range("dsv4.moe.sm120.fused_moe"):
            partial = self._fused_moe(all_x, all_w, all_i)
        dist.all_reduce(partial, op=dist.ReduceOp.SUM, group=group)
        return partial[local_begin:local_end].contiguous()


__all__ = ["Sm120FusedMoeStrategy"]
