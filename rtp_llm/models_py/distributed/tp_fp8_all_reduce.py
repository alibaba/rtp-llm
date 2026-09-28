"""Opt-in, lossy TP=2 FP8 all-reduce with RTP-LLM's own byte-size policy.

Only eager CUDA BF16 tensors are eligible. Initialization is collective and
fails on every rank when explicitly enabled on an unsupported configuration.
Per-call ineligible tensors use the caller's existing full-precision path.
"""

from __future__ import annotations

import logging
import os
import socket
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist

# Runtime policy, not user tuning knobs. Small messages retain the existing
# latency-oriented path; the upper bound is the preallocated IPC capacity.
_MIN_BYTES = 2 * 1024 * 1024
_MAX_BYTES = 128 * 1024 * 1024
_communicator: Optional["TpFp8AllReduceCommunicator"] = None


@dataclass(frozen=True)
class TpFp8AllReduceConfig:
    enabled: bool = False

    @classmethod
    def from_value(cls, enabled: str) -> "TpFp8AllReduceConfig":
        if enabled not in ("0", "1"):
            raise ValueError("RTP_LLM_TP_FP8_ALLREDUCE must be 0 or 1")
        return cls(enabled == "1")


def read_config_value() -> str:
    return os.environ.get("RTP_LLM_TP_FP8_ALLREDUCE", "0").strip()


def _select_blocks(sm_counts: list[int]) -> int:
    # One block per SM is a conservative residency bound. Native code also
    # checks actual occupancy before launching its device-wide peer barriers.
    return min(64, *sm_counts)


def _gather(value, group):
    values = [None] * dist.get_world_size(group)
    dist.all_gather_object(values, value, group=group)
    return values


def validate_config(group) -> TpFp8AllReduceConfig:
    # Disabled ranks must also participate: otherwise a typo on one rank can
    # send its peer into an IPC kernel while it runs NCCL.
    local = read_config_value()
    values = _gather(local, group)
    if any(value != local for value in values):
        raise ValueError(
            f"TP FP8 all-reduce configuration differs across ranks: {values}"
        )
    return TpFp8AllReduceConfig.from_value(local)


def _check_errors(error: Optional[str], group, phase: str) -> None:
    errors = _gather(error, group)
    if any(item is not None for item in errors):
        raise RuntimeError(f"TP FP8 all-reduce {phase} failed across ranks: {errors}")


@dataclass
class PendingTpFp8AllReduce:
    tensor: torch.Tensor
    event: torch.cuda.Event

    def wait(self) -> torch.Tensor:
        # Every consumer stream needs a fence, including repeated waits.
        stream = torch.cuda.current_stream(self.tensor.device)
        stream.wait_event(self.event)
        self.tensor.record_stream(stream)
        return self.tensor


class TpFp8AllReduceCommunicator:
    """Own IPC allocations and serialize their reuse on one CUDA stream.

    Construction and close must be called by both ranks in the same order.
    Model forwards must issue collectives in the same order on both ranks.
    As with NCCL, dtype and element count must agree. CUDA graph capture must
    encompass the same collectives on both ranks. Layout/alignment differences
    are normalized locally and never select different communication protocols.
    NCCL is used only for bootstrap and teardown, never for an eligible payload.
    """

    def __init__(
        self, group, device, max_bytes=_MAX_BYTES, min_bytes=_MIN_BYTES, blocks=None
    ):
        self.group = group
        self.device = torch.device(device)
        self.max_bytes = max_bytes
        self.min_bytes = min_bytes
        self.calls = 0
        self._native = None
        self._closed = False
        error = None
        try:
            if min_bytes < 64 or max_bytes < min_bytes or max_bytes % 64:
                raise ValueError("invalid TP FP8 internal workspace bounds")
            # Explicit arguments are for kernel tests; serving has one switch.
            if blocks is not None and not 1 <= blocks <= 256:
                raise ValueError("TP FP8 all-reduce blocks must be in [1,256]")
            if dist.get_world_size(group) != 2 or dist.get_backend(group) != "nccl":
                raise ValueError("requires a TP=2 NCCL group")
            if self.device.type != "cuda" or torch.version.hip is not None:
                raise ValueError("requires NVIDIA CUDA")
            if self.device.index is None:
                self.device = torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.get_device_capability(self.device) < (8, 9):
                raise ValueError("requires FP8-capable SM89 or newer")
            self.rank = dist.get_rank(group)
            properties = torch.cuda.get_device_properties(self.device)
            local = (
                socket.gethostname(),
                str(properties.uuid),
                self.max_bytes,
                self.min_bytes,
                blocks,
                properties.multi_processor_count,
            )
        except Exception as exc:
            local = None
            error = str(exc)
        _check_errors(error, group, "device validation")
        identities = _gather(local, group)
        if len({item[0] for item in identities}) != 1:
            raise RuntimeError("TP FP8 all-reduce requires both ranks on the same host")
        if len({item[1] for item in identities}) != 2:
            raise RuntimeError("TP FP8 all-reduce requires two distinct GPUs")
        if len({item[2:5] for item in identities}) != 1:
            raise RuntimeError("TP FP8 internal workspace / blocks differ across ranks")
        self.blocks = (
            _select_blocks([item[5] for item in identities])
            if blocks is None
            else blocks
        )

        # Each phase reports local failures before the next collective. Never
        # silently fall back on just one rank after an explicit enable request.
        error = None
        try:
            from rtp_llm.models_py.kernels.cuda.low_precision_all_reduce import (
                TpFp8AllReduce,
            )

            with torch.cuda.device(self.device):
                self._stream = torch.cuda.Stream(device=self.device)
                self._native = TpFp8AllReduce(
                    max_numel=max_bytes // 2,
                    device_index=self.device.index,
                    rank=self.rank,
                    blocks=self.blocks,
                )
                handle = self._native.get_ipc_handle()
                # Publish initialized barrier memory before peer kernels run.
                torch.cuda.current_stream(self.device).synchronize()
        except Exception as exc:
            error = repr(exc)
            handle = None
        try:
            _check_errors(error, group, "JIT/allocation")
            handles = _gather(handle, group)
            error = None
            try:
                self._native.open_peer(handles[1 - self.rank])
            except Exception as exc:
                error = repr(exc)
            _check_errors(error, group, "peer IPC mapping")
        except Exception:
            if self._native is not None:
                self._native.close_peer()
            dist.barrier(group=group)
            if self._native is not None:
                self._native.close()
            self._closed = True
            raise
        logging.info(
            "TP FP8 all-reduce enabled: rank=%d, device=%s, bytes=[%d,%d], "
            "blocks=%d; BF16 -> dynamic E4M3 -> BF16; independent RTP-LLM policy",
            self.rank,
            self.device,
            self.min_bytes,
            self.max_bytes,
            self.blocks,
        )

    def should_use(self, tensor: torch.Tensor) -> bool:
        if not self._closed and tensor.is_cuda and tensor.device != self.device:
            raise ValueError("TP FP8 tensor device differs from the bound communicator")
        return (
            not self._closed
            and tensor.is_cuda
            and tensor.device == self.device
            and tensor.dtype == torch.bfloat16
            and tensor.numel() % 32 == 0
            and self.min_bytes
            <= tensor.numel() * tensor.element_size()
            <= self.max_bytes
            and not torch.cuda.is_current_stream_capturing()
        )

    def all_reduce_async(self, tensor: torch.Tensor, out=None) -> PendingTpFp8AllReduce:
        if not self.should_use(tensor):
            raise ValueError("Tensor is ineligible for TP FP8 all-reduce")
        if out is None:
            out = torch.empty_like(tensor)
        if (
            out.shape != tensor.shape
            or out.dtype != tensor.dtype
            or out.device != tensor.device
        ):
            raise ValueError("TP FP8 output must match input shape, dtype and device")
        with torch.cuda.device(self.device):
            producer = torch.cuda.current_stream(self.device)
            native_input = tensor
            if not tensor.is_contiguous() or tensor.data_ptr() % 16:
                native_input = torch.empty_like(
                    tensor, memory_format=torch.contiguous_format
                )
                native_input.copy_(tensor)
            native_output = out
            if not out.is_contiguous() or out.data_ptr() % 16:
                native_output = torch.empty_like(
                    out, memory_format=torch.contiguous_format
                )
            self._stream.wait_stream(producer)
            with torch.cuda.stream(self._stream):
                self._native.all_reduce(native_input, native_output)
                if native_output is not out:
                    out.copy_(native_output)
                event = torch.cuda.Event()
                event.record(self._stream)
            # Storage cannot be recycled until the private stream completes.
            tensor.record_stream(self._stream)
            out.record_stream(self._stream)
            native_input.record_stream(self._stream)
            native_output.record_stream(self._stream)
        self.calls += 1
        if self.calls == 1:
            logging.info(
                "TP FP8 all-reduce first payload: rank=%d, shape=%s, bytes=%d",
                self.rank,
                tuple(tensor.shape),
                tensor.numel() * tensor.element_size(),
            )
        return PendingTpFp8AllReduce(out, event)

    def all_reduce(self, tensor: torch.Tensor, out=None) -> torch.Tensor:
        return self.all_reduce_async(tensor, out).wait()

    def close(self) -> None:
        if self._closed:
            return
        self._stream.synchronize()
        # Ensure neither GPU can still be using the other GPU's allocation.
        dist.barrier(group=self.group)
        self._native.close_peer()
        dist.barrier(group=self.group)
        self._native.close()
        self._closed = True
        logging.info(
            "TP FP8 all-reduce closed: rank=%d, calls=%d", self.rank, self.calls
        )


def init_tp_fp8_allreduce(group, device) -> None:
    global _communicator
    config = validate_config(group)
    if not config.enabled or _communicator is not None:
        return
    _communicator = TpFp8AllReduceCommunicator(group, device)


def get_tp_fp8_allreduce() -> Optional[TpFp8AllReduceCommunicator]:
    return _communicator


def destroy_tp_fp8_allreduce() -> None:
    global _communicator
    if _communicator is not None:
        _communicator.close()
        _communicator = None
