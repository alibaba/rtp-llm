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
    """Use the legacy environment form of the custom AllReduce switch."""
    return _env_is_false("FT_DISABLE_CUSTOM_AR", default=True)


def enabled_by_config(disable_custom_all_reduce: Optional[bool] = None) -> bool:
    """Prefer an explicit parsed override and preserve the legacy default."""
    if disable_custom_all_reduce is not None:
        return not disable_custom_all_reduce
    return enabled_by_env()


class FlashInferAllReduce:
    def __init__(
        self,
        group: ProcessGroup,
        device: torch.device,
        *,
        single_node: bool,
        disable_custom_all_reduce: Optional[bool] = None,
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
        self.cpp_small_enabled = False
        # Qwen4 target has a native post-capture callback. Other capture owners
        # must retain NCCL until they provide that lifecycle contract too.
        self.allow_ipc_capture = False

        if not enabled_by_config(disable_custom_all_reduce) or not single_node:
            logging.info(
                "FlashInfer all-reduce fallback: explicit_disable=%s env_disable=%s single_node=%s",
                disable_custom_all_reduce,
                os.getenv("FT_DISABLE_CUSTOM_AR"),
                single_node,
            )
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
            self.cpp_small_enabled = (
                self.world_size == 2
                and torch.cuda.get_device_capability(device) == (12, 0)
                and torch.cuda.device_count() == 2
                and torch.cuda.can_device_access_peer(0, 1)
                and torch.cuda.can_device_access_peer(1, 0)
            )
            logging.info(
                "FlashInfer C++ small TP eligibility: rank=%s tp=%s device=%s eligible=%s",
                self.rank,
                self.world_size,
                self.device,
                self.cpp_small_enabled,
            )
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
            hidden_dim = tensor.shape[1]
            if self.cpp_small_enabled and hidden_dim == 2560:
                from .flashinfer_ipc_all_reduce import FlashInferIpcAllReduce

                large_prefill = os.environ.get("QWEN4_TP_LARGE_PREFILL", "0") == "1"
                workspace = FlashInferIpcAllReduce(
                    self.group,
                    self.device,
                    max_bytes=128 * 1024 * 1024 if large_prefill else 32 * 2560 * 2,
                )
                workspace.large_prefill = large_prefill
            else:
                from flashinfer.comm.mnnvl import TorchDistBackend

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
        self._max_num_tokens = (
            workspace.max_bytes // (self._hidden_dim * tensor.element_size())
            if hasattr(workspace, "max_bytes")
            else _MAX_WORKSPACE_BYTES[self.world_size]
            // (self._hidden_dim * tensor.element_size())
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
            from .flashinfer_ipc_all_reduce import FlashInferIpcAllReduce

            if (
                isinstance(self._workspace, FlashInferIpcAllReduce)
                and torch.cuda.is_current_stream_capturing()
                and not self.allow_ipc_capture
            ):
                return False
            return self._workspace.is_buffer_size_sufficient(
                self.world_size,
                tensor.shape[0],
                tensor.shape[1],
                tensor.dtype,
            )
        return True

    def should_use_registered_graph_input(self, tensor):
        from .flashinfer_ipc_all_reduce import FlashInferIpcAllReduce

        return (
            self.allow_ipc_capture
            and isinstance(self._workspace, FlashInferIpcAllReduce)
            and self._workspace.registered_graph_inputs
            and torch.cuda.is_current_stream_capturing()
            and self.should_use(tensor)
        )

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        assert self._workspace is not None
        assert self._flashinfer_comm is not None
        from .flashinfer_ipc_all_reduce import FlashInferIpcAllReduce

        if isinstance(self._workspace, FlashInferIpcAllReduce):
            return self._workspace.all_reduce(tensor)
        return self._flashinfer_comm.allreduce_fusion(
            input=tensor,
            workspace=self._workspace,
            pattern=self._flashinfer_comm.AllReduceFusionPattern.kAllReduce,
            launch_with_pdl=True,
            # FlashInfer 0.6.9 can expose one-shot output too early when PDL is
            # chained. Completion-at-end is the graph replay safety contract.
            trigger_completion_at_end=True,
        )

    def finish_graph_capture(self):
        from .flashinfer_ipc_all_reduce import FlashInferIpcAllReduce

        if isinstance(self._workspace, FlashInferIpcAllReduce):
            self._workspace.finish_graph_capture()

    def all_reduce_into(self, tensor, output):
        from .flashinfer_ipc_all_reduce import FlashInferIpcAllReduce

        if isinstance(self._workspace, FlashInferIpcAllReduce):
            # The API first copies input to its IPC staging buffer. The kernel
            # consumes staging, so input/output aliasing is safe here too.
            return self._workspace.all_reduce(tensor, output)
        output.copy_(self.all_reduce(tensor))
        return output

    def should_use_cpp_small(self, tensor: torch.Tensor) -> bool:
        """Admit the verified two-rank BF16 hidden-width decode domain only."""
        return (
            self.cpp_small_enabled
            and tensor.ndim == 2
            and 1 <= tensor.shape[0] <= 32
            and tensor.shape[1] == 2560
            and self.should_use(tensor)
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
    disable_custom_all_reduce: Optional[bool] = None,
) -> Optional[FlashInferAllReduce]:
    global _communicator
    communicator = FlashInferAllReduce(
        group,
        device,
        single_node=single_node,
        disable_custom_all_reduce=disable_custom_all_reduce,
    )
    _communicator = None if communicator.disabled else communicator
    return _communicator


def get_flashinfer_allreduce() -> Optional[FlashInferAllReduce]:
    return _communicator


def destroy_flashinfer_allreduce() -> None:
    global _communicator
    if _communicator is not None:
        _communicator.destroy()
    _communicator = None
