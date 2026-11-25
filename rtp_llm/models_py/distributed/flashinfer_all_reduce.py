# SPDX-License-Identifier: Apache-2.0

"""Optional FlashInfer TRT-LLM all-reduce for single-node CUDA TP groups.

The adapter stays behind the generic ``collective_torch.all_reduce`` API.
On high-VRAM GPUs (H200, >=110 GiB), eager prefills exceeding the single-shot
workspace token cap are chunked across FlashInfer launches instead of falling
back to NCCL. Unsupported devices, topologies, dtypes, or initialization
failures fall back to NCCL before CUDA graph capture starts.
"""

import logging
import math
import os
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

_MIB = 1024 * 1024
_H200_MIN_TOTAL_MEMORY_BYTES = 110 * (1024 ** 3)
_MAX_WORKSPACE_BYTES = {
    2: 64 * _MIB,
    4: 2 * _MIB,
    8: _MIB // 2,
}
_HIGH_VRAM_MAX_WORKSPACE_BYTES = {
    2: 64 * _MIB,
    4: 32 * _MIB,
    8: 16 * _MIB,
}


def _is_high_vram_device(device: torch.device) -> bool:
    try:
        props = torch.cuda.get_device_properties(device)
        total_mem = getattr(props, "total_memory", 0)
        return isinstance(total_mem, int) and total_mem >= _H200_MIN_TOTAL_MEMORY_BYTES
    except Exception:
        return False


def _max_workspace_bytes_for_device(world_size: int, device: torch.device) -> int:
    override = os.getenv("RTP_LLM_FLASHINFER_AR_WORKSPACE_MIB", "").strip()
    if override:
        try:
            mib = int(override)
            if mib > 0:
                return mib * _MIB
        except ValueError:
            pass
    default_bytes = _MAX_WORKSPACE_BYTES[world_size]
    if world_size in _HIGH_VRAM_MAX_WORKSPACE_BYTES and _is_high_vram_device(device):
        return _HIGH_VRAM_MAX_WORKSPACE_BYTES[world_size]
    return default_bytes


def enabled_by_config(disable_custom_all_reduce: Optional[bool] = None) -> bool:
    """Only model-owned callers can explicitly enable TRT-LLM AllReduce."""
    return disable_custom_all_reduce is False


class FlashInferAllReduce:
    _h20_max_num_tokens: int = 128
    _allow_eager_chunking: bool = False

    def __init__(
        self,
        group: ProcessGroup,
        device: torch.device,
        *,
        single_node: bool,
        disable_custom_all_reduce: Optional[bool] = None,
        h20_max_num_tokens: int = 128,
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
        self._allow_eager_chunking = False
        if h20_max_num_tokens not in (128, 256, 512, 1024):
            raise ValueError("H20 all-reduce token capacity must be 128, 256, 512, or 1024")
        self._h20_max_num_tokens = h20_max_num_tokens
        self._flashinfer_comm = None
        self._shared_workspace_key = None
        self.disabled_reason = ""
        self.cpp_small_enabled = False
        # Qwen4 target has a native post-capture callback. Other capture owners
        # must retain NCCL until they provide that lifecycle contract too.
        self.allow_ipc_capture = False

        if not enabled_by_config(disable_custom_all_reduce) or not single_node:
            self.disabled_reason = (
                f"disabled_by_config(override={disable_custom_all_reduce}, "
                f"FT_DISABLE_CUSTOM_AR={os.getenv('FT_DISABLE_CUSTOM_AR')!r}, "
                f"single_node={single_node})"
            )
            logging.info(
                "FlashInfer all-reduce fallback: explicit_disable=%s env_disable=%s single_node=%s",
                disable_custom_all_reduce,
                os.getenv("FT_DISABLE_CUSTOM_AR"),
                single_node,
            )
            return
        if self.world_size not in _MAX_WORKSPACE_BYTES:
            self.disabled_reason = f"unsupported_tp_size({self.world_size})"
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
                self.disabled_reason = "flashinfer.comm_missing_allreduce_fusion"
                return
            self._flashinfer_comm = flashinfer_comm
            self.disabled = False
        except ImportError:
            self.disabled_reason = "flashinfer_import_error"
            logging.info("FlashInfer all-reduce unavailable; using NCCL")

    def _create_workspace(self, tensor: torch.Tensor) -> bool:
        if self.disabled or self._flashinfer_comm is None:
            return False
        if torch.cuda.is_current_stream_capturing():
            return False

        workspace = None
        reused_entry = None
        shared_key = None
        allow_eager_chunking = False
        hidden_dim = 0
        local_success = False
        local_error: Optional[Exception] = None
        try:
            hidden_dim = int(tensor.shape[-1])
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

                max_num_tokens = _max_workspace_bytes_for_device(
                    self.world_size, self.device
                ) // (hidden_dim * tensor.element_size())
                # The measured H20 hidden-2560 path benefits on Decode;
                # long Prefill retains its existing collective and workspace.
                is_h20_capped = (
                    self.world_size == 2
                    and hidden_dim == 2560
                    and torch.version.hip is None
                    and torch.cuda.get_device_capability(self.device) == (9, 0)
                    and torch.cuda.get_device_properties(
                        self.device
                    ).multi_processor_count
                    == 78
                )
                if is_h20_capped:
                    max_num_tokens = min(max_num_tokens, self._h20_max_num_tokens)
                allow_eager_chunking = (not is_h20_capped) and _is_high_vram_device(
                    self.device
                )
                device_idx = (
                    self.device.index
                    if self.device.index is not None
                    else torch.cuda.current_device()
                )
                candidate_key = (id(self.group), device_idx, hidden_dim, tensor.dtype)
                existing = _SHARED_TRTLLM_WORKSPACES.get(candidate_key)
                if (
                    existing is not None
                    and existing["group"] is self.group
                    and existing["max_num_tokens"] >= max_num_tokens
                ):
                    workspace = existing["workspace"]
                    reused_entry = existing
                    shared_key = candidate_key
                else:
                    workspace = self._flashinfer_comm.create_allreduce_fusion_workspace(
                        backend="trtllm",
                        world_size=self.world_size,
                        rank=self.rank,
                        max_token_num=max_num_tokens,
                        hidden_dim=hidden_dim,
                        dtype=tensor.dtype,
                        comm_backend=TorchDistBackend(group=self.group),
                    )
                    if existing is None:
                        shared_key = candidate_key
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
            if workspace is not None and reused_entry is None:
                workspace.destroy()
            self.disabled = True
            self.disabled_reason = (
                f"workspace_init_failed({local_error})"
                if local_error is not None
                else "workspace_init_failed_on_peer"
            )
            logging.warning(
                "FlashInfer TRT-LLM all-reduce initialization failed on at "
                "least one TP rank; using NCCL%s",
                f": {local_error}" if local_error is not None else "",
            )
            return False

        assert workspace is not None
        self._workspace = workspace
        self._hidden_dim = hidden_dim
        self._dtype = tensor.dtype
        self._max_num_tokens = (
            workspace.max_bytes // (self._hidden_dim * tensor.element_size())
            if hasattr(workspace, "max_bytes")
            else max_num_tokens
        )
        self._allow_eager_chunking = allow_eager_chunking
        if reused_entry is not None:
            reused_entry["refcount"] += 1
            self._shared_workspace_key = shared_key
            logging.info(
                "Reusing shared FlashInfer TRT-LLM all-reduce workspace: rank=%s, "
                "tp=%s, hidden=%s, max_tokens=%s (shared_capacity=%s, eager_chunk=%s)",
                self.rank,
                self.world_size,
                self._hidden_dim,
                self._max_num_tokens,
                reused_entry["max_num_tokens"],
                self._allow_eager_chunking,
            )
        else:
            if shared_key is not None:
                _SHARED_TRTLLM_WORKSPACES[shared_key] = {
                    "workspace": workspace,
                    "max_num_tokens": max_num_tokens,
                    "refcount": 1,
                    "group": self.group,
                }
                self._shared_workspace_key = shared_key
            logging.info(
                "Initialized FlashInfer TRT-LLM all-reduce: rank=%s, tp=%s, "
                "hidden=%s, max_tokens=%s, eager_chunk=%s",
                self.rank,
                self.world_size,
                self._hidden_dim,
                self._max_num_tokens,
                self._allow_eager_chunking,
            )
        return True

    @staticmethod
    def _tensor_rows_hidden(tensor: torch.Tensor) -> tuple[int, int]:
        if tensor.dim() == 2:
            return int(tensor.shape[0]), int(tensor.shape[1])
        return math.prod(int(s) for s in tensor.shape[:-1]), int(tensor.shape[-1])

    def _can_eager_chunk(self) -> bool:
        return (
            getattr(self, "_allow_eager_chunking", False)
            and self._max_num_tokens > 0
            and not torch.cuda.is_current_stream_capturing()
        )

    def should_use(self, tensor: torch.Tensor) -> bool:
        if self.disabled:
            return False
        if (
            not tensor.is_cuda
            or tensor.dtype != torch.bfloat16
            or tensor.dim() < 2
            or not tensor.is_contiguous()
        ):
            return False
        rows, hidden_dim = self._tensor_rows_hidden(tensor)
        if rows <= 0 or hidden_dim <= 0:
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
        # Reject incompatible calls before entering FlashInfer's validator.
        # On high-VRAM GPUs (H200), oversized eager prefills are chunked across
        # FlashInfer launches instead of falling back to NCCL.
        if hidden_dim != self._hidden_dim or tensor.dtype != self._dtype:
            return False
        if rows > self._max_num_tokens and not self._can_eager_chunk():
            return False

        if hasattr(self._workspace, "is_buffer_size_sufficient"):
            from .flashinfer_ipc_all_reduce import FlashInferIpcAllReduce

            if (
                isinstance(self._workspace, FlashInferIpcAllReduce)
                and torch.cuda.is_current_stream_capturing()
                and not self.allow_ipc_capture
            ):
                return False
            check_tokens = (
                min(rows, self._max_num_tokens)
                if rows > self._max_num_tokens and self._can_eager_chunk()
                else rows
            )
            return self._workspace.is_buffer_size_sufficient(
                self.world_size,
                check_tokens,
                hidden_dim,
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

    def _all_reduce_trtllm_single(self, tensor: torch.Tensor) -> torch.Tensor:
        return self._flashinfer_comm.allreduce_fusion(
            input=tensor,
            workspace=self._workspace,
            pattern=self._flashinfer_comm.AllReduceFusionPattern.kAllReduce,
            launch_with_pdl=True,
            # FlashInfer 0.6.9 can expose one-shot output too early when PDL is
            # chained. Completion-at-end is the graph replay safety contract.
            trigger_completion_at_end=True,
        )

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        assert self._workspace is not None
        assert self._flashinfer_comm is not None
        from .flashinfer_ipc_all_reduce import FlashInferIpcAllReduce

        is_nd = hasattr(tensor, "dim") and hasattr(tensor, "shape") and tensor.dim() > 2
        orig_shape = tensor.shape if is_nd else None
        view = tensor.reshape(-1, self._hidden_dim) if is_nd else tensor

        if isinstance(self._workspace, FlashInferIpcAllReduce):
            reduced = self._workspace.all_reduce(view)
            return reduced.reshape(orig_shape) if is_nd else reduced

        if (
            self._max_num_tokens > 0
            and hasattr(view, "shape")
            and view.shape[0] > self._max_num_tokens
        ):
            out = torch.empty_like(view)
            chunk = self._max_num_tokens
            for start in range(0, view.shape[0], chunk):
                end = min(start + chunk, view.shape[0])
                out[start:end].copy_(self._all_reduce_trtllm_single(view[start:end]))
            return out.reshape(orig_shape) if is_nd else out

        reduced = self._all_reduce_trtllm_single(view)
        return reduced.reshape(orig_shape) if is_nd else reduced

    def finish_graph_capture(self):
        from .flashinfer_ipc_all_reduce import FlashInferIpcAllReduce

        if isinstance(self._workspace, FlashInferIpcAllReduce):
            self._workspace.finish_graph_capture()

    def all_reduce_into(self, tensor, output):
        from .flashinfer_ipc_all_reduce import FlashInferIpcAllReduce

        is_nd = hasattr(tensor, "dim") and hasattr(tensor, "shape") and tensor.dim() > 2
        view_in = tensor.reshape(-1, self._hidden_dim) if is_nd else tensor
        view_out = output.reshape(-1, self._hidden_dim) if is_nd else output

        if isinstance(self._workspace, FlashInferIpcAllReduce):
            # The API first copies input to its IPC staging buffer. The kernel
            # consumes staging, so input/output aliasing is safe here too.
            self._workspace.all_reduce(view_in, view_out)
            return output

        if (
            self._max_num_tokens > 0
            and hasattr(view_in, "shape")
            and view_in.shape[0] > self._max_num_tokens
        ):
            chunk = self._max_num_tokens
            for start in range(0, view_in.shape[0], chunk):
                end = min(start + chunk, view_in.shape[0])
                view_out[start:end].copy_(
                    self._all_reduce_trtllm_single(view_in[start:end])
                )
            return output

        view_out.copy_(self._all_reduce_trtllm_single(view_in))
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

    def warmup(
        self,
        hidden_dim: int,
        dtype: torch.dtype = torch.bfloat16,
        probe_tokens: int = 1,
    ) -> bool:
        """Eagerly allocate and JIT-warm the workspace before CUDA graph capture."""
        if self.disabled or self._flashinfer_comm is None:
            return False
        if self._workspace is not None:
            return True
        if torch.cuda.is_current_stream_capturing():
            return False
        with torch.cuda.device(self.device):
            probe = torch.zeros(
                (max(1, int(probe_tokens)), int(hidden_dim)),
                device=self.device,
                dtype=dtype,
            )
            if not self.should_use(probe):
                return False
            self.all_reduce(probe)
        return True

    def status_summary(self) -> str:
        if self.disabled:
            return f"disabled({self.disabled_reason or 'unknown'})"
        if self._workspace is None:
            return f"enabled_uninitialized(tp={self.world_size})"
        from .flashinfer_ipc_all_reduce import FlashInferIpcAllReduce

        backend = (
            "vllm_ipc"
            if isinstance(self._workspace, FlashInferIpcAllReduce)
            else "trtllm"
        )
        shared_key = getattr(self, "_shared_workspace_key", None)
        shared_entry = (
            _SHARED_TRTLLM_WORKSPACES.get(shared_key)
            if shared_key is not None
            else None
        )
        shared_tag = (
            f",shared_refs={shared_entry['refcount']}"
            if shared_entry is not None and shared_entry["refcount"] > 1
            else ""
        )
        chunk_tag = ",eager_chunk=1" if getattr(self, "_allow_eager_chunking", False) else ""
        return (
            f"ready(backend={backend},tp={self.world_size},"
            f"hidden={self._hidden_dim},max_tokens={self._max_num_tokens}"
            f"{shared_tag}{chunk_tag})"
        )

    def destroy(self) -> None:
        if self._workspace is not None:
            shared_key = getattr(self, "_shared_workspace_key", None)
            if shared_key is not None:
                entry = _SHARED_TRTLLM_WORKSPACES.get(shared_key)
                self._shared_workspace_key = None
                if entry is not None and entry["workspace"] is self._workspace:
                    entry["refcount"] -= 1
                    if entry["refcount"] <= 0:
                        _SHARED_TRTLLM_WORKSPACES.pop(shared_key, None)
                        self._workspace.destroy()
                else:
                    self._workspace.destroy()
            else:
                self._workspace.destroy()
        self._workspace = None


_SHARED_TRTLLM_WORKSPACES: dict = {}
_communicator: Optional[FlashInferAllReduce] = None
_last_status_summary: str = "not_initialized"


def init_flashinfer_allreduce(
    group: ProcessGroup,
    device: torch.device,
    *,
    single_node: bool,
) -> Optional[FlashInferAllReduce]:
    global _communicator, _last_status_summary
    communicator = FlashInferAllReduce(
        group,
        device,
        single_node=single_node,
    )
    _last_status_summary = communicator.status_summary()
    _communicator = None if communicator.disabled else communicator
    return _communicator


def get_flashinfer_allreduce() -> Optional[FlashInferAllReduce]:
    return _communicator


def get_flashinfer_allreduce_status() -> str:
    if _communicator is not None:
        return _communicator.status_summary()
    return _last_status_summary


def destroy_flashinfer_allreduce() -> None:
    global _communicator, _last_status_summary
    if _communicator is not None:
        _communicator.destroy()
    _communicator = None
    _last_status_summary = "not_initialized"
