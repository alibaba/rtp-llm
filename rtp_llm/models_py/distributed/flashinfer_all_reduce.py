# SPDX-License-Identifier: Apache-2.0

"""Optional FlashInfer TRT-LLM all-reduce for single-node CUDA TP groups.

The adapter stays behind the generic ``collective_torch.all_reduce`` API.
Unsupported devices, topologies, shapes, or initialization failures fall back
to NCCL before CUDA graph capture starts.
"""

import logging
import os
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup


_MIB = 1024 * 1024
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
    """Use the server-wide custom AllReduce switch."""
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
            return False

        workspace = None
        local_success = False
        local_error: Optional[Exception] = None
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
            local_success = True
        except Exception as exc:
            local_error = exc

        # Every rank must choose the same collective implementation.  A local
        # fallback on only one peer would mismatch FlashInfer and NCCL launches.
        success = torch.tensor(
            int(local_success), dtype=torch.int32, device=self.device
        )
        dist.all_reduce(success, op=dist.ReduceOp.MIN, group=self.group)
        global_success = bool(success.item())
        if not global_success:
            if workspace is not None:
                workspace.destroy()
            self.disabled = True
            logging.warning(
                "FlashInfer TRT-LLM all-reduce initialization failed on at "
                "least one TP rank; using NCCL%s",
                f": {local_error}" if local_error is not None else "",
            )
            return False

        assert workspace is not None
        self._workspace = workspace
        self._hidden_dim = tensor.shape[1]
        self._dtype = tensor.dtype
        self._max_num_tokens = _MAX_WORKSPACE_BYTES[self.world_size] // (
            self._hidden_dim * tensor.element_size()
        )
        logging.info(
            "Initialized FlashInfer TRT-LLM all-reduce: rank=%s, tp=%s, "
            "hidden=%s, max_tokens=%s",
            self.rank,
            self.world_size,
            self._hidden_dim,
            self._max_num_tokens,
        )
        return True

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
                raise RuntimeError(
                    "FlashInfer all-reduce workspace was not initialized before "
                    "CUDA graph capture"
                )
            if not self._create_workspace(tensor):
                return False

        # The workspace has a fixed dtype, hidden width, and token capacity.
        # Reject incompatible calls before entering FlashInfer's validator: in
        # addition to logging on every layer, older FlashInfer releases warn
        # that an oversized launch can access memory beyond the workspace.
        # NCCL remains the generic fallback for large prefill tensors, while
        # decode tensors retain the graph-safe fast path.
        if (
            tensor.shape[1] != self._hidden_dim
            or tensor.dtype != self._dtype
            or tensor.shape[0] > self._max_num_tokens
        ):
            return False

        if hasattr(self._workspace, "is_buffer_size_sufficient"):
            return self._workspace.is_buffer_size_sufficient(
                self.world_size,
                tensor.shape[0],
                tensor.shape[1],
                tensor.dtype,
            )
        return True

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        assert self._workspace is not None
        assert self._flashinfer_comm is not None
        return self._flashinfer_comm.allreduce_fusion(
            input=tensor,
            workspace=self._workspace,
            pattern=self._flashinfer_comm.AllReduceFusionPattern.kAllReduce,
            launch_with_pdl=True,
            # FlashInfer 0.6.9 can expose one-shot output too early when PDL is
            # chained. Completion-at-end is the graph replay safety contract.
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
