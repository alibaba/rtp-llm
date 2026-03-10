# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
# Adapted from atrex trt_allreduce for rtp-llm ROCm backend.

from typing import Optional, Tuple
from contextlib import contextmanager

import torch
from torch import Tensor
import torch.distributed as dist
from torch.distributed import ProcessGroup

from rtp_llm.ops.compute_ops import rtp_llm_ops

TrtllmArFusionHandle = rtp_llm_ops.TrtllmArFusionHandle

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


class TrtllmDistEnv:
    """
    Distributed communication environment for TRT-LLM AllReduce Fusion.

    Manages IPC shared memory, barrier flags, and data buffers for
    cross-GPU allreduce operations. Supports CUDA Graph capture.
    """

    _SUPPORTED_WORLD_SIZES = [2, 4, 8]

    def __init__(
        self,
        group: ProcessGroup = None,
        device_id: int = None,
        max_size_in_bytes: int = 16384 * 16384,
        comm_ptrs_buf_len: int = 1024 * 256,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.group = group
        self.device_id = device_id
        self.rank = dist.get_rank(group=self.group)
        self.world_size = dist.get_world_size(group=self.group)
        self.handle: Optional[TrtllmArFusionHandle] = None
        self.disabled = False
        self._is_capturing = False
        self._is_captured = False
        torch.cuda.set_device(self.device_id)

        if self.world_size == 1:
            return

        if self.world_size not in self._SUPPORTED_WORLD_SIZES:
            return

        try:
            self.handle = rtp_llm_ops.TrtllmArFusionHandle(
                self.device_id, self.rank, self.world_size,
                max_size_in_bytes, comm_ptrs_buf_len
            )

            barrier_handle = self.handle.get_barrier_handle()
            data_handle = self.handle.get_data_handle()
        except Exception as e:
            import logging
            logging.warning(
                "TRT-LLM AllReduce initialization failed (likely insufficient GPU memory, "
                "requested %d bytes for data buffer). Falling back to RCCL. Error: %s",
                max_size_in_bytes * 2, e,
            )
            self.handle = None
            self.disabled = True
            return

        self._barrier()

        barrier_handle_list = [None] * self.world_size
        data_handle_list = [None] * self.world_size
        dist.all_gather_object(barrier_handle_list, barrier_handle, group=self.group)
        dist.all_gather_object(data_handle_list, data_handle, group=self.group)

        self.handle.open_barrier_handles(barrier_handle_list)
        self.handle.open_data_handles(data_handle_list)

        self._barrier()

    def _barrier(self):
        torch.cuda.set_device(self.device_id)
        torch.cuda.synchronize(self.device_id)
        dist.barrier(group=self.group)

    def _consume_capture(self):
        self._barrier()
        handles = self.handle.get_captured_handles()
        offsets = self.handle.get_captured_offsets()
        for idx in range(len(handles)):
            handle_list = [None] * self.world_size
            offset_list = [None] * self.world_size
            dist.all_gather_object(handle_list, handles[idx], group=self.group)
            dist.all_gather_object(offset_list, int(offsets[idx].item()), group=self.group)
            self._barrier()
            self.handle.open_captured_handles(handle_list, offset_list, idx)
        self.handle.capture_clear()
        self._barrier()

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
        else:
            if self._is_captured:
                self._consume_capture()
                self._is_captured = False

    def __del__(self):
        try:
            self.handle = None
        except Exception:
            pass

    def allreduce_add_rms_native(
        self,
        allreduce_in: Tensor,
        residual_in: Tensor,
        rms_weight: Tensor,
        eps: float,
        fp8_out: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Reference implementation using standard ops (for correctness testing)."""
        def rms_norm_forward(hidden_states: Tensor, weight: Tensor, epsilon: float) -> Tensor:
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
                allreduce_in.shape[0], 1,
                dtype=torch.float32, device=allreduce_in.device,
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
                allreduce_in.shape[0], 1,
                dtype=torch.float32, device=allreduce_in.device,
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
        self.device_id: Optional[int] = None
        self.dtype: Optional[torch.dtype] = None
        self.dist_env: Optional[TrtllmDistEnv] = None
        self.initialized = False

    def initialize(
        self,
        group: ProcessGroup,
        device_id: int,
        dtype: torch.dtype,
    ):
        if self.initialized and group == self.group and device_id == self.device_id:
            return

        self.cleanup()

        self.group = group
        self.device_id = device_id
        self.dtype = dtype
        self.dist_env = TrtllmDistEnv(
            group=self.group, device_id=self.device_id, dtype=self.dtype
        )
        self.initialized = True

    def cleanup(self):
        self.dist_env = None
        self.initialized = False


_trtllm_comm_manager = TrtllmCommManager()


def ensure_trtllm_comm_initialized(
    dtype: torch.dtype,
    group: ProcessGroup,
    device_id: int,
) -> bool:
    """Ensure TrtllmCommManager is initialized with the given parameters."""
    if _trtllm_comm_manager is None:
        return False

    if (
        not _trtllm_comm_manager.initialized
        or _trtllm_comm_manager.group != group
        or _trtllm_comm_manager.device_id != device_id
        or _trtllm_comm_manager.dtype != dtype
    ):
        _trtllm_comm_manager.initialize(
            group=group, device_id=device_id, dtype=dtype,
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

    Automatically initializes the communication workspace on first call.
    Drop-in replacement for ``atrex.allreduce``.
    """
    if not ensure_trtllm_comm_initialized(allreduce_in.dtype, group, device_id):
        raise RuntimeError("TRT-LLM AllReduce workspace failed to initialize")

    allreduce_out = torch.empty_like(allreduce_in)
    _trtllm_comm_manager.dist_env.allreduce_op(allreduce_in, allreduce_out)
    return allreduce_out


def consume_capture() -> None:
    """Notify the TRT-LLM comm manager to finalize IPC pointers after graph capture."""
    if _trtllm_comm_manager is not None and _trtllm_comm_manager.initialized:
        _trtllm_comm_manager.dist_env._consume_capture()


def allreduce_residual_rmsnorm(
    allreduce_in: Tensor,
    residual_in: Tensor,
    rms_weight: Tensor,
    group: ProcessGroup,
    device_id: int,
    dtype: torch.dtype = torch.bfloat16,
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
        dtype: Data type for initialization.
        eps: RMSNorm epsilon.
        fp8_out: Whether to quantize output to FP8.

    Returns:
        Tuple of (residual_out, norm_out, scale_out).
    """
    if not ensure_trtllm_comm_initialized(allreduce_in.dtype, group, device_id):
        raise RuntimeError("TRT-LLM AllReduce Fusion workspace is not initialized")
    return _trtllm_comm_manager.dist_env.allreduce_add_rms_fused(
        allreduce_in, residual_in, rms_weight, eps, fp8_out,
    )


