# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
# Adapted from atrex trt_allreduce for rtp-llm ROCm backend.

import logging
from contextlib import contextmanager
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup


def _get_handle_class():
    from rtp_llm.ops.compute_ops import rtp_llm_ops

    return rtp_llm_ops.TrtllmArFusionHandle


FP8_DTYPE = torch.float8_e4m3fnuz

FP8_MAX_VALUES = {
    torch.float8_e4m3fn: 240,
    torch.float8_e4m3fnuz: 120,
}

FP8_QUANT_TYPE_IDS = {
    torch.float8_e4m3fn: 1,
    torch.float8_e4m3fnuz: 2,
}

FP8_MAX_VALUE = FP8_MAX_VALUES[FP8_DTYPE]
FP8_QUANT_TYPE_ID = FP8_QUANT_TYPE_IDS[FP8_DTYPE]

# Supported hidden_size for trtllm allreduce kernels (pure allreduce).
# Must match the switch cases in allreduce_kernel_launcher_hd (trtllm_allreduce_fusion.cu).
ALLREDUCE_SUPPORTED_HIDDEN_SIZES = frozenset({1024, 2048, 2560, 4096, 5120})

# Supported hidden_size for fused allreduce + residual + rmsnorm kernels.
# Must match the switch cases in allreduce_fusion_kernel_launcher_hd (trtllm_allreduce_fusion.cu).
ALLREDUCE_FUSION_SUPPORTED_HIDDEN_SIZES = frozenset({1024, 2048, 4096})

# If the control group fails, collective two-phase release is impossible.
# Retain local exports until process exit instead of freeing memory that a peer
# may still have mapped.
_unrecoverable_workspaces = []


class TrtllmDistEnv:
    """
    Distributed communication environment for TRT-LLM AllReduce Fusion.

    Manages IPC shared memory, barrier flags, and data buffers for
    cross-GPU allreduce operations. Supports CUDA Graph capture.
    """

    _SUPPORTED_WORLD_SIZES = [2, 4, 8]

    def __init__(
        self,
        group: ProcessGroup,
        control_group: ProcessGroup,
        device_id: int,
        max_size_in_bytes: int = 16384 * 16384,
        comm_ptrs_buf_len: int = 1024 * 256,
    ) -> None:
        # Establish a destructor-safe state before any distributed query or
        # topology validation can raise.
        self.group = group
        self.control_group = control_group
        self.device_id = device_id
        self.max_size_in_bytes = max_size_in_bytes
        self.rank = -1
        self.world_size = 0
        self.handle = None
        self.disabled = False
        self._is_capturing = False
        self._is_captured = False
        self._capture_handles_pending = False
        self.rank = dist.get_rank(group=self.group)
        self.world_size = dist.get_world_size(group=self.group)
        control_rank = dist.get_rank(group=self.control_group)
        control_world_size = dist.get_world_size(group=self.control_group)
        data_ranks = tuple(dist.get_process_group_ranks(self.group))
        control_ranks = tuple(dist.get_process_group_ranks(self.control_group))
        if (control_rank, control_world_size) != (
            self.rank,
            self.world_size,
        ) or control_ranks != data_ranks:
            raise RuntimeError(
                "TRT-LLM control group rank order does not match the data group: "
                f"data=({self.rank}, {self.world_size}), "
                f"control=({control_rank}, {control_world_size}), "
                f"data_ranks={data_ranks}, control_ranks={control_ranks}"
            )
        torch.cuda.set_device(self.device_id)

        if self.world_size == 1:
            self.disabled = True
            return

        if self.world_size not in self._SUPPORTED_WORLD_SIZES:
            self.disabled = True
            return

        barrier_handle = None
        data_handle = None
        try:
            TrtllmArFusionHandle = _get_handle_class()
            self.handle = TrtllmArFusionHandle(
                self.device_id,
                self.rank,
                self.world_size,
                max_size_in_bytes,
                comm_ptrs_buf_len,
            )

            barrier_handle = self.handle.get_barrier_handle()
            data_handle = self.handle.get_data_handle()
        except Exception as e:
            logging.warning(
                "TRT-LLM AllReduce initialization failed (likely insufficient GPU memory, "
                "requested %d bytes for data buffer). Falling back to RCCL. Error: %s",
                max_size_in_bytes * 2,
                e,
            )
            # No IPC handle has been exchanged yet, so local exports cannot be
            # mapped by a peer and are safe to release without a collective
            # handshake. Do not rely on the destructor: it intentionally
            # retains exports once publication may have occurred.
            unpublished_handle = self.handle
            cleanup_failed = False
            if unpublished_handle is not None:
                try:
                    unpublished_handle.close_peer_mappings()
                except Exception:
                    cleanup_failed = True
                    logging.exception(
                        "Failed to close peer mappings for an unpublished "
                        "TRT-LLM workspace"
                    )
                try:
                    unpublished_handle.release_local_exports()
                except Exception:
                    cleanup_failed = True
                    logging.exception(
                        "Failed to release local exports for an unpublished "
                        "TRT-LLM workspace"
                    )
                if cleanup_failed:
                    _unrecoverable_workspaces.append(unpublished_handle)
            self.handle = None
            self.disabled = True
        try:
            self._initialize_ipc_collectively(barrier_handle, data_handle)
        except Exception as exc:
            self._quarantine_workspace_after_control_failure(exc)
            raise RuntimeError(
                "TRT-LLM IPC control group failed during initialization; "
                "the process must be rebuilt before retrying"
            ) from exc

    def _initialize_ipc_collectively(self, barrier_handle, data_handle) -> None:
        local_ready = not self.disabled and self.handle is not None
        readiness = [None] * self.world_size
        dist.all_gather_object(readiness, local_ready, group=self.control_group)
        if not all(readiness):
            failed_ranks = [rank for rank, ready in enumerate(readiness) if not ready]
            logging.warning(
                "TRT-LLM AllReduce disabled because ranks %s failed initialization",
                failed_ranks,
            )
            self._release_workspace_two_phase()
            self.disabled = True
            return

        self._barrier()

        barrier_handle_list = [None] * self.world_size
        data_handle_list = [None] * self.world_size
        dist.all_gather_object(
            barrier_handle_list, barrier_handle, group=self.control_group
        )
        dist.all_gather_object(data_handle_list, data_handle, group=self.control_group)

        open_error = None
        try:
            self.handle.open_barrier_handles(barrier_handle_list)
            self.handle.open_data_handles(data_handle_list)
        except Exception as exc:
            open_error = exc
            logging.warning(
                "TRT-LLM AllReduce failed to open IPC handles on rank %d, "
                "device %d: %s",
                self.rank,
                self.device_id,
                exc,
            )
        open_status = [None] * self.world_size
        dist.all_gather_object(
            open_status, open_error is None, group=self.control_group
        )
        if not all(open_status):
            failed_ranks = [rank for rank, ready in enumerate(open_status) if not ready]
            logging.warning(
                "TRT-LLM AllReduce disabled because ranks %s failed to open IPC handles",
                failed_ranks,
            )
            # Every rank closes imported mappings before any rank releases its
            # exported allocation.  The second barrier makes that release
            # observable before the control group can be destroyed.
            self._release_workspace_two_phase()
            self.disabled = True
            return

        self._barrier()

    def _quarantine_workspace_after_control_failure(self, error: Exception) -> None:
        self.disabled = True
        if self.handle is None:
            return
        try:
            torch.cuda.set_device(self.device_id)
            torch.cuda.synchronize(self.device_id)
            self.handle.close_peer_mappings()
        except Exception:
            logging.exception(
                "Failed to close imported TRT-LLM IPC mappings after control-group failure"
            )
        logging.error(
            "Retaining TRT-LLM local exports until process exit after control-group "
            "failure on rank %d device %d: %s",
            self.rank,
            self.device_id,
            error,
        )
        _unrecoverable_workspaces.append(self.handle)
        self.handle = None

    def _barrier(self):
        torch.cuda.set_device(self.device_id)
        torch.cuda.synchronize(self.device_id)
        dist.barrier(group=self.control_group)

    def _release_workspace_two_phase(self) -> None:
        """Close peer mappings, then collectively release local exports.

        The local device is drained before phase one so no in-flight kernel can
        dereference an imported mapping while it is being closed.
        A barrier failure deliberately leaves ``handle`` alive so teardown can
        be retried without freeing memory that a peer may still have mapped.
        """
        # Once teardown starts this workspace must never service another
        # capture, even when a barrier failure leaves it available for retry.
        self.disabled = True
        torch.cuda.set_device(self.device_id)
        torch.cuda.synchronize(self.device_id)
        if self.handle is not None:
            self.handle.close_peer_mappings()
        self._barrier()
        if self.handle is not None:
            self.handle.release_local_exports()
        self._barrier()
        self.handle = None

    def shutdown(self) -> None:
        self._release_workspace_two_phase()

    def _consume_capture(self):
        if not self._capture_handles_pending:
            return
        self._barrier()
        try:
            handles = self.handle.get_captured_handles()
            offsets = self.handle.get_captured_offsets()
            local_success = True
        except Exception as e:
            handles = []
            offsets = torch.tensor([], dtype=torch.int64)
            local_success = False
            logging.warning(
                "[TrtllmAllreduce] get_captured_handles failed on rank %d: %s. "
                "Will coordinate with other ranks to discard this capture.",
                self.rank,
                e,
            )

        # All ranks must agree on success; if any rank failed, everyone must
        # discard. This prevents used_comm_ptrs_ from diverging across ranks.
        success_flags = [None] * self.world_size
        dist.all_gather_object(success_flags, local_success, group=self.control_group)

        if not all(success_flags):
            failed_ranks = [r for r, ok in enumerate(success_flags) if not ok]
            # Only capture_clear here — no open_captured_handles has executed
            # yet, so there are no IPC slots to roll back.  Calling
            # invalidate_capture() without a prior begin_capture_session()
            # would rewind to snapshot=0 and destroy base IPC slots.
            self.handle.capture_clear()
            self._barrier()
            self._capture_handles_pending = False
            raise RuntimeError(
                f"[TrtllmAllreduce] get_captured_handles failed on rank(s) "
                f"{failed_ranks}. All ranks are discarding this graph capture. "
                f"The caller should re-capture the graph."
            )

        # All ranks must agree on the number of handles before entering the
        # per-handle loop; otherwise a rank with fewer handles exits early
        # while others block on the next all_gather_object, causing a hang.
        local_count = len(handles)
        count_list = [None] * self.world_size
        dist.all_gather_object(count_list, local_count, group=self.control_group)
        if len(set(count_list)) != 1:
            # Same reasoning: no IPC slots opened yet, only clear graph state.
            self.handle.capture_clear()
            self._barrier()
            self._capture_handles_pending = False
            raise RuntimeError(
                f"[TrtllmAllreduce] Handle count mismatch across ranks: "
                f"{count_list}. All ranks are discarding this graph capture. "
                f"The caller should re-capture the graph."
            )
        num_handles = local_count

        # Snapshot the current used_comm_ptrs_ / IPC handle watermarks so
        # that invalidate_capture() can roll back to exactly this point
        # if any handle registration fails mid-loop.
        self.handle.begin_capture_session()

        open_error = None
        for idx in range(num_handles):
            handle_list = [None] * self.world_size
            offset_list = [None] * self.world_size
            dist.all_gather_object(handle_list, handles[idx], group=self.control_group)
            dist.all_gather_object(
                offset_list, int(offsets[idx].item()), group=self.control_group
            )
            self._barrier()

            # open_captured_handles may TORCH_CHECK-fail on some ranks (e.g.
            # stale pointer).  Wrap in try so we can synchronise the failure
            # across all ranks instead of letting the failing rank exit the
            # loop while others block on the next collective.
            local_open_ok = True
            if open_error is None:
                try:
                    self.handle.open_captured_handles(handle_list, offset_list, idx)
                except Exception as e:
                    local_open_ok = False
                    open_error = e
            else:
                # Already failed on a previous iteration — skip but keep
                # participating in collectives so other ranks don't hang.
                local_open_ok = False

            open_ok_flags = [None] * self.world_size
            dist.all_gather_object(
                open_ok_flags, local_open_ok, group=self.control_group
            )

            if not all(open_ok_flags):
                failed_ranks = [r for r, ok in enumerate(open_ok_flags) if not ok]
                if open_error is None:
                    open_error = RuntimeError(
                        f"[TrtllmAllreduce] open_captured_handles failed on "
                        f"rank(s) {failed_ranks} at idx={idx}."
                    )
                # Continue the loop so remaining iterations' collectives are
                # not orphaned — but mark that we need to bail out afterwards.

        self.handle.capture_clear()
        if open_error is not None:
            # Roll back only this session's slots (used_comm_ptrs_, map
            # entries, IPC handles).  Previously committed sessions are safe.
            self.handle.invalidate_capture()
        else:
            # All handles registered successfully — promote pending slots to
            # committed state so future invalidate_capture() won't touch them.
            self.handle.commit_capture()
        self._barrier()
        self._capture_handles_pending = False

        if open_error is not None:
            raise RuntimeError(
                f"[TrtllmAllreduce] Captured IPC handle registration failed. "
                f"All ranks have cleaned up. The caller should discard the "
                f"captured graph and re-capture. Original error: {open_error}"
            )

    @contextmanager
    def capture(self):
        """Context manager for CUDA Graph capture mode."""
        try:
            self._is_capturing = True
            yield
        finally:
            self._is_capturing = False
            if not self.disabled:
                self._consume_capture()

    def _prepare_capture(self, input_tensor: torch.Tensor):
        """Handle graph capture state transitions for input tensor."""
        if torch.cuda.is_current_stream_capturing():
            self._is_captured = True
            self._capture_handles_pending = True
        else:
            if self._is_captured:
                self._consume_capture()
                self._is_captured = False

    def consume_capture_if_needed(self) -> None:
        """Finalize IPC pointers after graph capture if allreduce was used.

        Safe to call unconditionally — only performs the (potentially expensive)
        IPC handle exchange when allreduce was actually invoked during capture.
        Resets the internal capture flag afterwards.
        """
        if self._is_captured:
            self._consume_capture()
            self._is_captured = False

    def finish_capture_session(self) -> None:
        """Collectively finalize one runner-owned capture session.

        Every TP rank enters the same control-group consensus before any rank
        decides whether the handle-exchange collectives are required. A mixed
        pending state invalidates the session on every rank and fails fast.
        """
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("finish_capture_session must run outside stream capture")
        local_pending = bool(self._is_captured or self._capture_handles_pending)
        pending_flags = [None] * self.world_size
        dist.all_gather_object(pending_flags, local_pending, group=self.control_group)
        if len(set(pending_flags)) != 1:
            self.handle.capture_clear()
            self._is_captured = False
            self._capture_handles_pending = False
            self._barrier()
            raise RuntimeError(
                "TRT-LLM capture participation differed across TP ranks: "
                f"{pending_flags}; every rank must discard this captured graph"
            )
        if local_pending:
            self._consume_capture()
            self._is_captured = False

    def __del__(self):
        handle = self.handle
        if handle is not None:
            logging.error(
                "TRT-LLM workspace on rank %d device %d was reclaimed "
                "without collective shutdown; quarantining exports until process exit",
                self.rank,
                self.device_id,
            )
            _unrecoverable_workspaces.append(handle)
        self.handle = None

    def allreduce_add_rms_native(
        self,
        allreduce_in: Tensor,
        residual_in: Tensor,
        rms_weight: Tensor,
        eps: float,
        fp8_out: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Reference implementation using standard ops (for correctness testing)."""

        def rms_norm_forward(
            hidden_states: Tensor, weight: Tensor, epsilon: float
        ) -> Tensor:
            input_dtype = hidden_states.dtype
            variance = hidden_states.float().pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + epsilon)
            hidden_states = hidden_states.to(input_dtype)
            return weight * hidden_states

        dist.all_reduce(allreduce_in, group=self.group)
        residual_out = allreduce_in + residual_in
        norm_out = rms_norm_forward(residual_out, rms_weight, eps)

        if fp8_out:
            norm_out_scale, _ = norm_out.float().abs().max(dim=-1, keepdim=True)
            norm_out_scale = norm_out_scale / FP8_MAX_VALUE
            norm_out = norm_out / norm_out_scale
            norm_out.clamp_(min=-FP8_MAX_VALUE, max=FP8_MAX_VALUE)
            norm_out = norm_out.to(FP8_DTYPE)
            return residual_out, norm_out, norm_out_scale
        else:
            scale_out = torch.empty(
                allreduce_in.shape[0],
                1,
                dtype=torch.float32,
                device=allreduce_in.device,
            )
            return residual_out, norm_out, scale_out

    def allreduce_add_rms_fused(
        self,
        allreduce_in: Tensor,
        residual_in: Tensor,
        rms_weight: Tensor,
        eps: float,
        fp8_out: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Fused AllReduce + Residual Add + RMSNorm kernel."""
        self._prepare_capture(allreduce_in)
        residual_out = torch.empty_like(allreduce_in)

        if fp8_out:
            norm_out = torch.empty_like(allreduce_in, dtype=FP8_DTYPE)
            scale_out = torch.empty(
                allreduce_in.shape[0],
                1,
                dtype=torch.float32,
                device=allreduce_in.device,
            )
        else:
            norm_out = torch.empty_like(allreduce_in)
            scale_out = torch.empty(1, dtype=torch.float32, device=allreduce_in.device)

        self.handle.allreduce_rms(
            allreduce_in,
            residual_in,
            rms_weight,
            residual_out,
            norm_out,
            scale_out,
            eps,
            FP8_QUANT_TYPE_ID if fp8_out else 0,
        )
        return residual_out, norm_out, scale_out

    def allreduce_op(
        self,
        allreduce_in: Tensor,
        allreduce_out: Tensor,
    ) -> None:
        """Pure AllReduce kernel (no fusion)."""
        self._prepare_capture(allreduce_in)
        self.handle.allreduce(
            allreduce_in,
            allreduce_out,
        )


class TrtllmCommManager:
    """Singleton manager for TrtllmDistEnv lifecycle."""

    def __init__(self):
        self.group: Optional[ProcessGroup] = None
        self.control_group: Optional[ProcessGroup] = None
        self.device_id: Optional[int] = None
        self.generation: Optional[int] = None
        self.dist_env: Optional[TrtllmDistEnv] = None
        self.initialized = False

    def initialize(
        self,
        group: ProcessGroup,
        device_id: int,
        generation: Optional[int] = None,
        control_group: Optional[ProcessGroup] = None,
    ):
        actual_control = control_group if control_group is not None else group
        if (
            self.initialized
            and group == self.group
            and actual_control == self.control_group
            and device_id == self.device_id
            and generation == self.generation
        ):
            return

        self.cleanup()

        self.group = group
        self.control_group = actual_control
        self.device_id = device_id
        self.generation = generation
        self.dist_env = TrtllmDistEnv(
            group=self.group,
            control_group=self.control_group,
            device_id=self.device_id,
        )
        self.initialized = True

    def cleanup(self):
        dist_env = self.dist_env
        # A failed control-group barrier must retain the object and its identity
        # so the same collective shutdown can be retried symmetrically.
        self.initialized = False
        if dist_env is not None:
            dist_env.shutdown()
        self.dist_env = None
        self.group = None
        self.control_group = None
        self.device_id = None
        self.generation = None


_trtllm_comm_manager = TrtllmCommManager()


def _workspace_identity_error(group: ProcessGroup, device_id: int) -> RuntimeError:
    return RuntimeError(
        "TRT-LLM AllReduce workspace identity mismatch; refusing to rebuild "
        "captured IPC state: "
        f"requested group={group!r}, device={device_id}; "
        f"initialized group={_trtllm_comm_manager.group!r}, "
        f"device={_trtllm_comm_manager.device_id}, "
        f"generation={_trtllm_comm_manager.generation}"
    )


def _require_workspace_identity(group: ProcessGroup, device_id: int) -> None:
    if _trtllm_comm_manager.initialized and (
        _trtllm_comm_manager.group != group
        or _trtllm_comm_manager.device_id != device_id
    ):
        raise _workspace_identity_error(group, device_id)


def ensure_trtllm_comm_initialized(
    group: ProcessGroup,
    device_id: int,
    generation: Optional[int] = None,
    control_group: Optional[ProcessGroup] = None,
) -> bool:
    """Ensure TrtllmCommManager is initialized with the given parameters."""
    _require_workspace_identity(group, device_id)
    graph_owned = (
        _trtllm_comm_manager.initialized and _trtllm_comm_manager.generation is not None
    )
    if graph_owned and (
        (generation is not None and generation != _trtllm_comm_manager.generation)
        or (
            control_group is not None
            and control_group != _trtllm_comm_manager.control_group
        )
    ):
        raise _workspace_identity_error(group, device_id)
    if (
        _trtllm_comm_manager.initialized
        and _trtllm_comm_manager.group == group
        and _trtllm_comm_manager.device_id == device_id
        and generation is None
        and control_group is None
    ):
        return not _trtllm_comm_manager.dist_env.disabled
    requested_generation = generation
    if (
        not _trtllm_comm_manager.initialized
        or _trtllm_comm_manager.group != group
        or _trtllm_comm_manager.device_id != device_id
        or _trtllm_comm_manager.generation != requested_generation
        or _trtllm_comm_manager.control_group
        != (control_group if control_group is not None else group)
    ):
        _trtllm_comm_manager.initialize(
            group=group,
            device_id=device_id,
            generation=requested_generation,
            control_group=control_group,
        )

    if _trtllm_comm_manager.initialized and _trtllm_comm_manager.dist_env.disabled:
        return False

    return _trtllm_comm_manager.initialized


def allreduce(
    allreduce_in: Tensor,
    group: ProcessGroup,
    device_id: int,
) -> Tensor:
    """Top-level AllReduce using the TRT-LLM fusion kernel.

    Returns a **new** tensor containing the allreduced result. The input
    tensor is not modified. Callers must use the returned tensor.

    This is consistent with the CUDA symm_mem allreduce path which also
    returns a new tensor (see ``TorchSymmMemCommunicator.all_reduce``).

    Automatically initializes the communication workspace on first call.
    Drop-in replacement for ``atrex.allreduce``.

    Returns:
        A new tensor with the allreduced values.
    """
    dist_env = _require_ready_workspace(group, device_id, "AllReduce")

    allreduce_out = torch.empty_like(allreduce_in)
    dist_env.allreduce_op(allreduce_in, allreduce_out)
    return allreduce_out


def _require_ready_workspace(
    group: ProcessGroup, device_id: int, operation: str
) -> TrtllmDistEnv:
    """Return the ready workspace shared by both TRT allreduce entry points."""
    # Graph communication is normally initialized explicitly with its control
    # group and generation. Preserve that identity on the capture hot path,
    # while retaining eager-call compatibility outside capture.
    if not _trtllm_comm_manager.initialized:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"TRT-LLM {operation} workspace must be initialized before stream capture"
            )
        if not ensure_trtllm_comm_initialized(group, device_id):
            raise RuntimeError(f"TRT-LLM {operation} workspace failed to initialize")
    _require_workspace_identity(group, device_id)
    if not is_trt_allreduce_ready():
        raise RuntimeError(f"TRT-LLM {operation} workspace is initialized but disabled")
    return _trtllm_comm_manager.dist_env


def is_trt_allreduce_ready() -> bool:
    """Check if trt_allreduce is initialized and usable.

    Returns True when the TrtllmCommManager singleton has been successfully
    initialized and the underlying TrtllmDistEnv is not disabled (e.g. due
    to insufficient GPU memory during IPC buffer allocation).
    """
    return (
        _trtllm_comm_manager is not None
        and _trtllm_comm_manager.initialized
        and _trtllm_comm_manager.dist_env is not None
        and not _trtllm_comm_manager.dist_env.disabled
    )


def consume_capture() -> None:
    """Notify the TRT-LLM comm manager to finalize IPC pointers after graph capture.

    Only performs the (potentially expensive) IPC handle exchange when
    trtllm allreduce was actually used during the capture session.
    Delegates to TrtllmDistEnv.consume_capture_if_needed() which manages
    the internal capture flag.
    """
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("consume_capture must not run during stream capture.")
    if (
        _trtllm_comm_manager is not None
        and _trtllm_comm_manager.initialized
        and _trtllm_comm_manager.dist_env is not None
        and not _trtllm_comm_manager.dist_env.disabled
    ):
        _trtllm_comm_manager.dist_env.consume_capture_if_needed()


def finish_capture_session() -> None:
    """Finalize capture through a rank-symmetric control-group consensus."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("finish_capture_session must not run during stream capture")
    if (
        _trtllm_comm_manager.initialized
        and _trtllm_comm_manager.dist_env is not None
        and not _trtllm_comm_manager.dist_env.disabled
    ):
        _trtllm_comm_manager.dist_env.finish_capture_session()


def has_pending_capture() -> bool:
    if (
        _trtllm_comm_manager is None
        or not _trtllm_comm_manager.initialized
        or _trtllm_comm_manager.dist_env is None
        or _trtllm_comm_manager.dist_env.disabled
    ):
        return False
    return bool(_trtllm_comm_manager.dist_env._capture_handles_pending)


def cleanup() -> None:
    """Explicitly release IPC workspace state before ProcessGroup teardown."""
    _trtllm_comm_manager.cleanup()


def allreduce_residual_rmsnorm(
    allreduce_in: Tensor,
    residual_in: Tensor,
    rms_weight: Tensor,
    group: ProcessGroup,
    device_id: int,
    eps: float = 1e-6,
    fp8_out: bool = False,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Fused AllReduce + Residual Add + RMSNorm + optional FP8 quantization.

    Args:
        allreduce_in: Input tensor to allreduce.
        residual_in: Residual tensor to add.
        rms_weight: RMSNorm gamma weights.
        group: Process group for communication.
        device_id: Target GPU device id.
        eps: RMSNorm epsilon.
        fp8_out: Whether to quantize output to FP8.

    Returns:
        Tuple of (residual_out, norm_out, scale_out).
    """
    dist_env = _require_ready_workspace(group, device_id, "AllReduce Fusion")
    return dist_env.allreduce_add_rms_fused(
        allreduce_in,
        residual_in,
        rms_weight,
        eps,
        fp8_out,
    )
