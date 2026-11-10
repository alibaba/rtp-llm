"""RocmEPStrategy: expert-parallel MoE for ROCm without DeepEP or Mega.

The CUDA path uses DeepGEMM's fused MegaMoE/DeepEP dispatch.  Neither is
available in the ROCm image, so this strategy implements the same semantics
with plain RCCL collectives:

  1. all-gather the local tokens, top-k ids and top-k weights from every EP
     rank (fixed shapes, so the path is CUDA/HIP-graph capturable);
  2. run the existing per-expert ``LocalLoopStrategy`` compute on the gathered
     tokens, restricted to this rank's contiguous expert slice;
  3. reduce-scatter the partial outputs, which sums the contributions of each
     expert owner and returns only the caller's tokens.

It is mathematically equivalent to EP dispatch+combine: every routed expert
is evaluated exactly once on its owner, and the per-token top-k weights are
applied before the sum.  No weight is quantized or approximated.  For long
prefill the token axis is chunked so the temporary gather stays bounded.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.distributed as dist

from rtp_llm.models_py.distributed.collective_torch import (
    Group,
    all_gather as _capture_all_gather,
    all_reduce as _capture_all_reduce,
)

from .base import MoeCfg, RoutedExpertsStrategy, register_strategy
from .local_loop import LocalLoopStrategy

_DEFAULT_CHUNK = 4096


def _ep_chunk_tokens() -> int:
    try:
        return max(1, int(os.environ.get("DSV4_ROCM_EP_CHUNK", _DEFAULT_CHUNK)))
    except ValueError:
        return _DEFAULT_CHUNK


@register_strategy
class RocmEPStrategy(RoutedExpertsStrategy):
    name = "rocm_ep"

    def __init__(self, cfg: MoeCfg):
        super().__init__(cfg)
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("RocmEPStrategy requires an initialized process group")
        if cfg.ep_size <= 1:
            raise RuntimeError("RocmEPStrategy requires ep_size > 1")
        self._local = LocalLoopStrategy(cfg)
        self._ep_group = dist.group.WORLD
        self._ep_world = dist.get_world_size(self._ep_group)
        self._ep_rank = dist.get_rank(self._ep_group)
        if self._ep_world != cfg.ep_size:
            raise RuntimeError(
                f"RocmEPStrategy expects one EP group spanning world_size={cfg.ep_size}, "
                f"got {self._ep_world}"
            )

    @classmethod
    def can_handle(cls, cfg: MoeCfg) -> bool:
        return cfg.ep_size > 1 and torch.version.hip is not None

    def setup_weights(self, layer_weights) -> None:
        # Same EP-sliced [E_local, ...] stacks LocalLoop consumes.
        self._local.setup_weights(layer_weights)

    def get_eplb_weights(self):
        return self._local.get_eplb_weights()

    def _gather(self, tensor: torch.Tensor) -> torch.Tensor:
        # Route through the capture-aware ROCm collective wrapper.  Raw
        # dist.all_gather_into_tensor crashes RCCL graph capture on MI300.
        # The HIP-graph shim expects a flat contiguous buffer.
        n = tensor.shape[0]
        flat = tensor.contiguous().view(-1)
        out = _capture_all_gather(flat, Group.TP)
        return out.view(self._ep_world * n, *tensor.shape[1:])

    def _forward_chunk(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        n = x.shape[0]
        diag = os.environ.get("DSV41_EP_DIAG", "")
        if diag == "local":
            # Diagnostic only: bypass EP collectives to isolate graph capture.
            return self._local._forward_into_buf(
                x,
                weights,
                indices,
                local_start=self.cfg.local_expert_start,
                local_end=self.cfg.local_expert_end,
            )
        x_all = self._gather(x)
        idx_all = self._gather(indices)
        w_all = self._gather(weights)
        # Local expert ids are the global ids in [local_start, local_end).
        y_all = self._local._forward_into_buf(
            x_all,
            w_all,
            idx_all,
            local_start=self.cfg.local_expert_start,
            local_end=self.cfg.local_expert_end,
        )
        # RCCL has no capture-safe reduce-scatter wrapper here.  All-reduce the
        # partial expert outputs and slice this rank's token block instead; the
        # result is identical and the collective is graph-capturable.
        flat = y_all.contiguous().view(-1)
        y_flat = _capture_all_reduce(flat, Group.TP, inplace=False)
        if diag == "gather":
            return y_all[self._ep_rank * n : (self._ep_rank + 1) * n]
        y_sum = y_flat.view_as(y_all)
        return y_sum[self._ep_rank * n : (self._ep_rank + 1) * n]

    def forward(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        expert_load=None,
        eplb_layer=None,
    ) -> torch.Tensor:
        if x.numel() == 0:
            return torch.zeros((0, self.cfg.dim), dtype=torch.float32, device=x.device)
        if eplb_layer is not None:
            raise NotImplementedError("RocmEPStrategy does not implement EPLB")
        # All ranks must take the same chunk boundaries.
        n = x.shape[0]
        chunk = _ep_chunk_tokens()
        if n <= chunk:
            return self._forward_chunk(x, weights, indices)
        out = torch.empty((n, self.cfg.dim), dtype=torch.float32, device=x.device)
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            out[start:end] = self._forward_chunk(
                x[start:end], weights[start:end], indices[start:end]
            )
        return out
