# SPDX-License-Identifier: Apache-2.0

"""Optional FlashInfer TRT-LLM all-reduce fast path for CUDA TP groups.

This is a small adapter around FlashInfer's existing communication workspace.
It deliberately stays below the model layer: callers keep using the generic
``collective_torch.all_reduce`` API and unsupported shapes/topologies continue
to use NCCL.

The workspace lifecycle and eligibility rules follow vLLM's Apache-2.0
``FlashInferAllReduce`` implementation, adapted to RTP-LLM's process groups.
"""

import logging
import os
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup


_MIB = 1024 * 1024
# Hopper thresholds used by vLLM's FlashInfer all-reduce path.  A larger
# workspace only broadens eligibility; it does not make a large collective
# faster than NCCL, so keep the empirically tuned per-TP limits.
_MAX_WORKSPACE_BYTES = {
    2: 64 * _MIB,
    4: 2 * _MIB,
    8: _MIB // 2,
}


def _env_is_false(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return not default
    return value.strip().lower() in ("0", "false", "off", "no")


def enabled_by_env() -> bool:
    """Use the existing custom-AR switch; main keeps it disabled by default."""
    return _env_is_false("FT_DISABLE_CUSTOM_AR", default=True)


class FlashInferAllReduce:
    def __init__(
        self,
        group: ProcessGroup,
        device: torch.device,
        *,
        single_node: bool,
    ) -> None:
        self.group = group
        self.device = device
        self.world_size = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        self.disabled = True
        self._workspace = None
        self._hidden_dim = 0
        self._dtype: Optional[torch.dtype] = None
        self._max_num_tokens = 0
        self._flashinfer_comm = None

        if not enabled_by_env() or not single_node:
            return
        if self.world_size not in _MAX_WORKSPACE_BYTES:
            logging.info(
                "FlashInfer all-reduce disabled for unsupported TP size %s",
                self.world_size,
            )
            return
        try:
            import flashinfer.comm as flashinfer_comm

            if not hasattr(flashinfer_comm, "allreduce_fusion") or not hasattr(
                flashinfer_comm, "create_allreduce_fusion_workspace"
            ):
                return
            self._flashinfer_comm = flashinfer_comm
            self.disabled = False
        except ImportError:
            logging.info("FlashInfer all-reduce unavailable; using NCCL")

    def _create_workspace(self, tensor: torch.Tensor) -> bool:
        if self.disabled or self._flashinfer_comm is None:
            return False
        if torch.cuda.is_current_stream_capturing():
            # Workspace creation exchanges IPC handles and must happen during
            # the ordinary warmup forward, never from inside graph capture.
            return False
        try:
            from flashinfer.comm.mnnvl import TorchDistBackend

            hidden_dim = tensor.shape[1]
            max_num_tokens = _MAX_WORKSPACE_BYTES[self.world_size] // (
                hidden_dim * tensor.element_size()
            )
            workspace = self._flashinfer_comm.create_allreduce_fusion_workspace(
                backend="trtllm",
                world_size=self.world_size,
                rank=self.rank,
                max_token_num=max_num_tokens,
                hidden_dim=hidden_dim,
                dtype=tensor.dtype,
                comm_backend=TorchDistBackend(group=self.group),
            )
            self._workspace = workspace
            self._hidden_dim = hidden_dim
            self._dtype = tensor.dtype
            self._max_num_tokens = max_num_tokens
            logging.info(
                "Initialized FlashInfer TRT-LLM all-reduce: rank=%s, tp=%s, "
                "hidden=%s, max_tokens=%s",
                self.rank,
                self.world_size,
                hidden_dim,
                max_num_tokens,
            )
            return True
        except Exception as exc:
            # Initialization is collective. A failure means this communicator
            # is unsafe to retry independently on later model calls.
            logging.warning(
                "FlashInfer TRT-LLM all-reduce initialization failed; using NCCL: %s",
                exc,
            )
            self.disabled = True
            return False

    def should_use(self, tensor: torch.Tensor) -> bool:
        if self.disabled:
            return False
        if (
            not tensor.is_cuda
            or tensor.dtype != torch.bfloat16
            or tensor.dim() != 2
            or not tensor.is_contiguous()
        ):
            return False
        if self._workspace is None:
            if torch.cuda.is_current_stream_capturing():
                # Capturing NCCL here and switching to FlashInfer later would
                # make the selected graph depend on warmup order.  The normal
                # model warmup must initialize the IPC workspace first.
                raise RuntimeError(
                    "FlashInfer all-reduce workspace was not initialized before "
                    "CUDA graph capture"
                )
            if not self._create_workspace(tensor):
                return False

        # FlashInfer workspaces are capacity based and may safely serve other
        # 2-D shapes when the backend's own metadata check accepts them.
        if hasattr(self._workspace, "is_buffer_size_sufficient"):
            return self._workspace.is_buffer_size_sufficient(
                self.world_size,
                tensor.shape[0],
                tensor.shape[1],
                tensor.dtype,
            )
        return (
            tensor.shape[1] == self._hidden_dim
            and tensor.dtype == self._dtype
            and tensor.shape[0] <= self._max_num_tokens
        )

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        assert self._workspace is not None
        assert self._flashinfer_comm is not None
        return self._flashinfer_comm.allreduce_fusion(
            input=tensor,
            workspace=self._workspace,
            pattern=self._flashinfer_comm.AllReduceFusionPattern.kAllReduce,
            launch_with_pdl=True,
            # FlashInfer 0.6.9's one-shot Lamport path can expose its output
            # too early when PDL is chained. Completion-at-end is the safe
            # graph-replay contract; a newer dependency can relax it later.
            trigger_completion_at_end=True,
        )

    def destroy(self) -> None:
        if self._workspace is not None:
            self._workspace.destroy()
        self._workspace = None


_communicator: Optional[FlashInferAllReduce] = None


def init_flashinfer_allreduce(
    group: ProcessGroup,
    device: torch.device,
    *,
    single_node: bool,
) -> Optional[FlashInferAllReduce]:
    global _communicator
    communicator = FlashInferAllReduce(group, device, single_node=single_node)
    _communicator = None if communicator.disabled else communicator
    return _communicator


def get_flashinfer_allreduce() -> Optional[FlashInferAllReduce]:
    return _communicator


def destroy_flashinfer_allreduce() -> None:
    global _communicator
    if _communicator is not None:
        _communicator.destroy()
    _communicator = None
