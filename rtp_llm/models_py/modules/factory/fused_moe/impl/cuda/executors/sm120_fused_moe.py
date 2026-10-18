"""SM120 FP8-activation x MXFP4-weight MoE executor for EP > 1.

SM120 in the CUDA 13 environment has no compatible DeepEP/Mega kernel. This
executor keeps communication collective and computes the rank-local expert
partition with :class:`Sm120GroupedFp4Executor`:

* pure context-parallel prefill (``cp_size == moe_tp_size == ep_size``):
  all-gather tokens over the TP group, compute a partial routed sum over the
  local experts, reduce-scatter back to the local CP shard;
* every other case (DP/EP, pure TP, CUDA graph capture): all-gather tokens
  over WORLD, compute the local partial, all-reduce, and keep this rank's
  rows. The WORLD fallback requires one EP group spanning WORLD; callers
  synchronize the eager chunk schedule across ranks.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict

import torch

from rtp_llm.models_py.distributed.collective_torch import (
    Group,
    all_gather,
    reduce_scatter,
)
from rtp_llm.models_py.kernels.cuda.quant_layouts import FP8_BLOCK
from rtp_llm.models_py.modules.factory.fused_moe.utils.profiler import (
    record_function_range,
)
from rtp_llm.models_py.utils.arch import is_sm120

from .fp8_fp4_base import Fp8Fp4ExecutorBase
from .sm120_grouped_fp4 import Sm120GroupedFp4Executor, sm120_grouped_fp4_available

_CP_LOGGED = False


def is_pure_cp(config) -> bool:
    cp_size = int(getattr(config, "moe_cp_size", 1))
    return bool(
        getattr(config, "moe_cp_enabled", False)
        and cp_size > 1
        and cp_size == int(getattr(config, "moe_tp_size", 1)) == int(config.ep_size)
    )


def local_expert_config(config):
    """Return a copy of ``config`` describing only this rank's experts."""
    n_local = int(config.n_local_experts)
    local = copy.copy(config)
    local.ep_size = 1
    local.ep_rank = 0
    local.dp_size = 1
    local.dp_rank = 0
    local.expert_num = n_local
    local.n_routed_experts = n_local
    local.physical_expert_num = n_local
    local.has_redundant_experts = False
    local.n_local_experts = n_local
    local.local_expert_start = 0
    local.local_expert_end = n_local
    return local


def mask_to_local_experts(config, weights, indices):
    """Zero non-local slots and remap global ids to ``[0, n_local)``."""
    n_local = int(config.n_local_experts)
    local_ids = indices.to(torch.int64) - int(config.local_expert_start)
    valid = (local_ids >= 0) & (local_ids < n_local)
    local_weights = weights * valid.to(weights.dtype)
    return local_weights, local_ids.clamp_(0, n_local - 1)


def validate_world_ep_topology(config, dist, group) -> tuple[int, int]:
    world = dist.get_world_size(group)
    rank = dist.get_rank(group)
    if world != config.ep_size or rank != config.ep_rank:
        raise RuntimeError(
            "SM120 WORLD collective topology does not match the expert "
            f"partition: world/rank={world}/{rank}, "
            f"ep_size/ep_rank={config.ep_size}/{config.ep_rank}. "
            "A dedicated EP process group is required for mixed DP/TP+EP."
        )
    return world, rank


class Sm120FusedMoeExecutor(Fp8Fp4ExecutorBase):
    """Collective SM120 EP with FlashInfer grouped local experts."""

    # Every rank must enter the collectives, including ranks without tokens.
    execute_empty_inputs = True

    @classmethod
    def check_conditions(cls, checker: Any, config) -> None:
        super().check_conditions(checker, config)
        checker.check(config.ep_size > 1)
        checker.check(config.world_size == config.ep_size)
        checker.check(config.world_rank == config.ep_rank)
        checker.check(config.dim % FP8_BLOCK == 0)
        checker.check(config.moe_inter_dim % FP8_BLOCK == 0)
        checker.check(sm120_grouped_fp4_available())

    @property
    def requires_synchronized_chunk_schedule(self) -> bool:
        # Pure CP guarantees equal local shapes; the WORLD fallback exchanges
        # variable-size payloads and needs one shared outer chunk count.
        return not is_pure_cp(self.cfg)

    def setup_weights(self, layer_weights: Dict) -> None:
        self._local = Sm120GroupedFp4Executor(
            local_expert_config(self.cfg), self.quant_config, layer_weights
        )

    def forward(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        if not x.is_cuda or not is_sm120(x.device):
            raise RuntimeError("Sm120FusedMoeExecutor requires an SM120 CUDA device")
        if is_pure_cp(self.cfg) and not torch.cuda.is_current_stream_capturing():
            return self._forward_cp_reduce_scatter(x, weights, indices)
        return self._forward_world_collective(x, weights, indices)

    def _local_partial(self, x, weights, indices) -> torch.Tensor:
        local_weights, local_ids = mask_to_local_experts(self.cfg, weights, indices)
        return self._local.forward(x, local_weights, local_ids)

    def _forward_cp_reduce_scatter(self, x, weights, indices) -> torch.Tensor:
        global _CP_LOGGED
        if not _CP_LOGGED:
            _CP_LOGGED = True
            logging.info(
                "[SM120 MoE] CP all_gather + reduce_scatter: cp_size=%d "
                "local_tokens=%d hidden=%d",
                int(self.cfg.moe_cp_size),
                x.size(0),
                x.size(1),
            )
        if x.size(0) == 0:
            return torch.empty(x.shape, dtype=torch.float32, device=x.device)
        # The layer owns outer chunking; keep one collective/compute pass per
        # chunk.
        with record_function_range("moe.sm120.cp_fused_moe"):
            x_full = all_gather(x, group=Group.TP)
            weights_full = all_gather(weights, group=Group.TP)
            indices_full = all_gather(indices, group=Group.TP)
            partial = self._local_partial(x_full, weights_full, indices_full)
            return reduce_scatter(partial, group=Group.TP)

    def _forward_world_collective(self, x, weights, indices) -> torch.Tensor:
        dist = torch.distributed
        if not dist.is_initialized():
            raise RuntimeError("SM120 collective MoE requires torch.distributed")
        group = dist.group.WORLD
        world, rank = validate_world_ep_topology(self.cfg, dist, group)
        if torch.cuda.is_current_stream_capturing():
            if int(self.cfg.dp_size) > 1:
                raise RuntimeError(
                    "SM120 DP/EP requires eager execution; disable CUDA graphs"
                )
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
        with record_function_range("moe.sm120.world_fused_moe"):
            partial = self._local_partial(all_x, all_w, all_i)
        dist.all_reduce(partial, op=dist.ReduceOp.SUM, group=group)
        return partial[local_begin:local_end].contiguous()
