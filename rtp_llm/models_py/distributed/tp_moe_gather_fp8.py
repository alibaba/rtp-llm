"""Opt-in TP2 eager-prefill MoE gather, gate, and FP8 transport service.

This module deliberately owns a separate native IPC context.  It is only a
capability provider for the MoE serving path: callers retain their established
path whenever the topology or a particular input is ineligible.
"""

from __future__ import annotations

import logging
import os
import socket
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist

_SWITCH = "RTP_LLM_MOE_TP_FUSED_FP8_AR"
_HIDDEN_SIZE = 2048
_TOP_K = 8
_MIN_TOKENS = 4096
_MAX_BYTES = 128 * 1024 * 1024
_MAX_NUMEL = _MAX_BYTES // 2

_communicator: Optional["TpMoeGatherFp8Communicator"] = None


@dataclass(frozen=True)
class TpMoeGatherFp8Config:
    enabled: bool = False

    @classmethod
    def from_value(cls, value: str) -> "TpMoeGatherFp8Config":
        if value not in ("0", "1"):
            raise ValueError(f"{_SWITCH} must be 0 or 1")
        return cls(enabled=value == "1")


def read_config_value() -> str:
    return os.environ.get(_SWITCH, "0").strip()


def _gather(value, group):
    values = [None] * dist.get_world_size(group)
    dist.all_gather_object(values, value, group=group)
    return values


def _check_errors(error: Optional[str], group, phase: str) -> None:
    errors = _gather(error, group)
    if any(item is not None for item in errors):
        raise RuntimeError(f"TP MoE gather FP8 {phase} failed across ranks: {errors}")


def validate_config(group) -> TpMoeGatherFp8Config:
    """Validate the independent flag on every TP participant.

    Disabled ranks take part as well.  Otherwise one rank could start the IPC
    protocol while its peer executes the ordinary MoE path.
    """

    local = read_config_value()
    values = _gather(local, group)
    if any(value != local for value in values):
        raise ValueError(
            f"TP MoE gather FP8 configuration differs across ranks: {values}"
        )
    return TpMoeGatherFp8Config.from_value(local)


def _support_status(group, device, parallelism_config) -> tuple[bool, str]:
    """Return a collective, non-throwing eligibility decision before JIT.

    A requested but unsupported deployment is intentionally a fallback.  This
    is distinct from a malformed or rank-inconsistent switch, which is an
    error because it could cause ranks to select different collectives.
    """

    device = torch.device(device)
    error = None
    host = None
    uuid = None
    topology = (
        getattr(parallelism_config, "tp_size", None),
        getattr(parallelism_config, "ep_size", None),
        getattr(parallelism_config, "dp_size", None),
    )
    try:
        if dist.get_world_size(group) != 2 or not str(
            dist.get_backend(group)
        ).lower().endswith("nccl"):
            error = "requires a TP=2 NCCL group"
        elif topology != (2, 1, 1):
            error = f"requires pure TP2 / EP1 / DP1, got {topology}"
        elif device.type != "cuda" or torch.version.hip is not None:
            error = "requires NVIDIA CUDA"
        else:
            if device.index is None:
                device = torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.get_device_capability(device) != (12, 0):
                error = "requires SM120"
            else:
                properties = torch.cuda.get_device_properties(device)
                host, uuid = socket.gethostname(), str(properties.uuid)
    except Exception as exc:
        error = repr(exc)

    statuses = _gather((error, host, uuid, topology), group)
    errors = [status[0] for status in statuses if status[0] is not None]
    if errors:
        return False, "; ".join(str(item) for item in errors)
    if len({status[3] for status in statuses}) != 1:
        return False, f"parallelism topology differs across TP ranks: {statuses}"
    if len({status[1] for status in statuses}) != 1:
        return False, "requires both TP ranks on the same host"
    if len({status[2] for status in statuses}) != 2:
        return False, "requires two distinct GPUs"
    return True, ""


class TpMoeGatherFp8Communicator:
    """One ordered IPC workspace bound to the first caller stream.

    Calls on both ranks must keep the same order and tensor metadata, just as
    for NCCL.  The caller makes that routing decision; this class performs no
    per-forward CPU collective or metadata agreement. Input producers must be
    ordered before the caller stream. Alternate callers are joined to the
    first caller's stream, preserving the native context's single-stream rule.
    """

    def __init__(self, group, device):
        self.group = group
        self.device = torch.device(device)
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.rank = dist.get_rank(group)
        self.max_numel = _MAX_NUMEL
        self.blocks = 0
        self.calls = 0
        self._closed = False
        self._native = None
        self._stream = None

        # Probe the actual compiled gather kernel.  Its occupancy is part of
        # the device-wide barrier safety contract, so do not use the native
        # blocks=0 sentinel after the vector-gather register change.
        error = None
        probe = None
        try:
            from rtp_llm.models_py.kernels.cuda.moe_tp_fused_fp8 import load_native

            with torch.cuda.device(self.device):
                native_module = load_native()
                probe = native_module.MoeTpFusedFp8(
                    self.max_numel, self.device.index, self.rank, 1
                )
                capacity = int(
                    probe.launch_info()["gather_gate_push_deferred_max_resident_blocks"]
                )
                if capacity < 1:
                    raise RuntimeError(
                        "native gather deferred kernel has no resident CTA"
                    )
        except Exception as exc:
            error, capacity = repr(exc), None
        finally:
            if probe is not None:
                try:
                    probe.close()
                except Exception as exc:
                    if error is None:
                        error = repr(exc)
        try:
            _check_errors(error, group, "JIT/probe allocation")
            capacities = _gather(capacity, group)
            if any(not isinstance(item, int) or item < 1 for item in capacities):
                raise RuntimeError(f"invalid native gather capacity: {capacities}")
            self.blocks = min(capacities)

            error, handle = None, None
            try:
                with torch.cuda.device(self.device):
                    self._native = native_module.MoeTpFusedFp8(
                        self.max_numel, self.device.index, self.rank, self.blocks
                    )
                    actual_capacity = int(
                        self._native.launch_info()[
                            "gather_gate_push_deferred_max_resident_blocks"
                        ]
                    )
                    if self.blocks > actual_capacity:
                        raise RuntimeError(
                            f"selected blocks={self.blocks} exceeds local "
                            f"deferred gather capacity={actual_capacity}"
                        )
                    handle = self._native.get_ipc_handle()
                    # Publish initialized peer-visible state before open_peer.
                    torch.cuda.current_stream(self.device).synchronize()
            except Exception as exc:
                error = repr(exc)
            _check_errors(error, group, "workspace allocation")
            handles = _gather(handle, group)
            error = None
            try:
                self._native.open_peer(handles[1 - self.rank])
            except Exception as exc:
                error = repr(exc)
            _check_errors(error, group, "peer IPC mapping")
        except Exception:
            if self._native is not None:
                try:
                    self._native.close_peer()
                finally:
                    dist.barrier(group=group)
                    self._native.close()
            else:
                dist.barrier(group=group)
            self._closed = True
            raise

        logging.info(
            "TP MoE gather FP8 enabled: rank=%d device=%s blocks=%d "
            "tokens>=%d max_bytes=%d",
            self.rank,
            self.device,
            self.blocks,
            _MIN_TOKENS,
            _MAX_BYTES,
        )

    def should_use(self, hidden_states: torch.Tensor) -> bool:
        """Return eligibility for eager prefill; no collectives or syncs."""

        if (
            not self._closed
            and hidden_states.is_cuda
            and hidden_states.device != self.device
        ):
            raise ValueError(
                "TP MoE gather FP8 tensor device differs from communicator"
            )
        return (
            not self._closed
            and hidden_states.is_cuda
            and hidden_states.device == self.device
            and hidden_states.dtype == torch.bfloat16
            and hidden_states.ndim == 2
            and hidden_states.shape[0] >= _MIN_TOKENS
            and hidden_states.shape[1] == _HIDDEN_SIZE
            and hidden_states.numel() <= self.max_numel
            and hidden_states.is_contiguous()
            and not torch.cuda.is_current_stream_capturing()
        )

    def _validate_inputs(self, down, ids, weights, index, shared, gate, out) -> int:
        if self._closed:
            raise RuntimeError("TP MoE gather FP8 communicator is closed")
        tensors = (down, ids, weights, index, shared, gate, out)
        if any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
            raise TypeError("TP MoE gather FP8 inputs must be tensors")
        tokens = shared.shape[0] if shared.ndim == 2 else -1
        expected = (
            (
                down.dtype == torch.bfloat16
                and down.ndim == 2
                and down.shape[0] > 0
                and down.shape[1] == _HIDDEN_SIZE
            ),
            (ids.dtype == torch.int64 and tuple(ids.shape) == (tokens, _TOP_K)),
            (
                weights.dtype == torch.float32
                and tuple(weights.shape) == (tokens, _TOP_K)
            ),
            (index.dtype == torch.int64 and tuple(index.shape) == (tokens, _TOP_K)),
            (
                shared.dtype == torch.bfloat16
                and tuple(shared.shape) == (tokens, _HIDDEN_SIZE)
            ),
            (gate.dtype == torch.bfloat16 and tuple(gate.shape) == (tokens, 1)),
            (
                out.dtype == torch.bfloat16
                and tuple(out.shape) == (tokens, _HIDDEN_SIZE)
            ),
        )
        if not all(expected):
            raise ValueError(
                "TP MoE gather FP8 tensor shapes or dtypes violate native contract"
            )
        if tokens < _MIN_TOKENS or out.numel() > self.max_numel:
            raise ValueError(
                "TP MoE gather FP8 output is outside eager prefill capacity"
            )
        if any(
            not tensor.is_cuda
            or tensor.device != self.device
            or not tensor.is_contiguous()
            for tensor in tensors
        ):
            raise ValueError(
                "TP MoE gather FP8 tensors must be contiguous on bound CUDA device"
            )

        # Native rejects every overlap for this API.  This is CPU metadata only
        # and does not synchronize CUDA.
        def overlaps(left: torch.Tensor, right: torch.Tensor) -> bool:
            return max(left.data_ptr(), right.data_ptr()) < min(
                left.data_ptr() + left.nbytes, right.data_ptr() + right.nbytes
            )

        for position, tensor in enumerate(tensors[:-1]):
            if overlaps(tensor, out):
                raise ValueError("TP MoE gather FP8 output may not overlap an input")
            for other in tensors[position + 1 : -1]:
                if overlaps(tensor, other):
                    raise ValueError("TP MoE gather FP8 inputs may not overlap")
        return tokens

    def gather_gate_push(self, down, ids, weights, index, shared, gate, out=None):
        """Enqueue the deferred-ack native protocol and join it to caller stream.

        The return is host-asynchronous.  Its producing event is inserted into
        the caller's current stream before return, so ordinary downstream CUDA
        work observes the completed result without a host synchronize.
        """

        if out is None:
            out = torch.empty_like(shared, memory_format=torch.contiguous_format)
        self._validate_inputs(down, ids, weights, index, shared, gate, out)
        with torch.cuda.device(self.device):
            caller = torch.cuda.current_stream(self.device)
            # Keep the native context's single-stream contract, while allowing
            # the common caller/producer stream to reuse FC2 storage promptly.
            if self._stream is None:
                self._stream = caller
            if self._stream != caller:
                self._stream.wait_stream(caller)
            with torch.cuda.stream(self._stream):
                self._native.gather_gate_push_deferred(
                    down, ids, weights, index, shared, gate, out
                )
                # This is enqueued after the native launch, avoiding a host
                # scalar extraction per layer while preserving sticky failures.
                torch._assert_async(
                    self._native.error_status() == 0,
                    "TP MoE gather FP8 native device error",
                )
                event = torch.cuda.Event()
                event.record(self._stream)
            for tensor in (down, ids, weights, index, shared, gate, out):
                tensor.record_stream(self._stream)
            if caller != self._stream:
                caller.wait_event(event)
            out.record_stream(caller)
        self.calls += 1
        if self.calls == 1:
            logging.info(
                "TP MoE gather FP8 first payload: rank=%d tokens=%d bytes=%d blocks=%d",
                self.rank,
                shared.shape[0],
                out.nbytes,
                self.blocks,
            )
        return out

    def copy_local_packets(self) -> torch.Tensor:
        """Debug-only snapshot of this rank's last encoded wire packets.

        This serializes the native copy behind this rank's bound stream and
        joins it to the current caller stream.  It intentionally does not
        expose the peer-written inbox: after an acknowledgement, that storage
        may already be overwritten by the peer's next call.
        """

        if self._closed:
            raise RuntimeError("TP MoE gather FP8 communicator is closed")
        if self._stream is None:
            raise RuntimeError("no MoE gather invocation has published packets")
        with torch.cuda.device(self.device):
            caller = torch.cuda.current_stream(self.device)
            with torch.cuda.stream(self._stream):
                packets = self._native.copy_local_packets()
                event = torch.cuda.Event()
                event.record(self._stream)
            packets.record_stream(self._stream)
            caller.wait_event(event)
            packets.record_stream(caller)
        return packets

    def close(self) -> None:
        if self._closed:
            return
        if self._stream is not None:
            self._stream.synchronize()
        # Do not release an IPC mapping while its peer may still dereference it.
        dist.barrier(group=self.group)
        self._native.close_peer()
        dist.barrier(group=self.group)
        self._native.close()
        self._closed = True
        logging.info(
            "TP MoE gather FP8 closed: rank=%d calls=%d", self.rank, self.calls
        )


def init_tp_moe_gather_fp8(group, device, parallelism_config) -> None:
    """Create the capability provider only when the whole TP group supports it."""

    global _communicator
    config = validate_config(group)
    if not config.enabled or _communicator is not None:
        return
    supported, reason = _support_status(group, device, parallelism_config)
    if not supported:
        logging.warning(
            "TP MoE gather FP8 requested but unavailable; using existing path: %s",
            reason,
        )
        return
    _communicator = TpMoeGatherFp8Communicator(group, device)


def get_tp_moe_gather_fp8() -> Optional[TpMoeGatherFp8Communicator]:
    return _communicator


def destroy_tp_moe_gather_fp8() -> None:
    global _communicator
    if _communicator is not None:
        _communicator.close()
        _communicator = None
