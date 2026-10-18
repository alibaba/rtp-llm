"""Opt-in structured observations without synchronizing model execution.

Tensor observations must be called on the stream that produces the scalars.
They capture scalar values at that point, before later work on that stream.
"""

import atexit
import json
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from torch import Tensor

_LOGGER = logging.getLogger(__name__)
_PREFIX = "[RTP_SMOKE_EVENT] "
_MAX_PENDING_OBSERVATIONS = 128
_worker: Optional["_ObservationWorker"] = None
_worker_lock = threading.Lock()


def smoke_evidence_enabled() -> bool:
    return os.environ.get("RTP_LLM_SMOKE_EVIDENCE") == "1"


def _write_event(event: str, time_ns: int, fields: dict, *, error=False) -> None:
    payload = dict(fields)
    payload.update(event=event, time_ns=time_ns)
    log = _LOGGER.error if error else _LOGGER.info
    log("%s%s", _PREFIX, json.dumps(payload, sort_keys=True, allow_nan=False))


def emit_smoke_event(event: str, **fields) -> None:
    if smoke_evidence_enabled():
        _write_event(event, time.time_ns(), fields)


def _write_error(event: str, time_ns: int, reason: str, detail: str) -> None:
    _write_event(
        "evidence_error",
        time_ns,
        {"observation_event": event, "reason": reason, "detail": detail},
        error=True,
    )


@dataclass
class _Observation:
    event: str
    time_ns: int
    fields: dict
    sources: List[Any] = field(default_factory=list)
    snapshots: Dict[str, Any] = field(default_factory=dict)
    events: List[Any] = field(default_factory=list)
    fallback_streams: List[Any] = field(default_factory=list)
    error: Optional[str] = None


class _ObservationWorker:
    def __init__(self):
        self.pid = os.getpid()
        self.queue = queue.Queue(maxsize=_MAX_PENDING_OBSERVATIONS)
        # Count in-flight work as well as queued work. Reserve before any D2H
        # copy, so a full queue cannot discard a copy's lifetime references.
        self.slots = threading.BoundedSemaphore(_MAX_PENDING_OBSERVATIONS)
        self.condition = threading.Condition()
        self.pending = 0
        self.thread = threading.Thread(
            target=self._run, name="smoke-evidence", daemon=True
        )
        self.thread.start()

    def reserve(self) -> bool:
        with self.condition:
            if not self.slots.acquire(blocking=False):
                return False
            self.pending += 1
        return True

    def submit(self, observation: _Observation) -> None:
        self.queue.put_nowait(observation)

    def _run(self) -> None:
        while True:
            observation = self.queue.get()
            try:
                # Only this background thread waits for the producer's CUDA
                # events and reads the completed CPU scalar copies.
                completion_errors = []
                for event in observation.events:
                    try:
                        event.synchronize()
                    except Exception as error:
                        completion_errors.append(str(error))
                # An event may fail to record after a copy was already queued.
                # Retain all buffers and let this background thread finish that
                # producer stream before releasing the failed observation.
                for stream in observation.fallback_streams:
                    try:
                        stream.synchronize()
                    except Exception as error:
                        completion_errors.append(str(error))
                if completion_errors:
                    raise RuntimeError("; ".join(completion_errors))
                if observation.error is not None:
                    raise RuntimeError(observation.error)
                fields = dict(observation.fields)
                fields.update(
                    (name, tensor.item())
                    for name, tensor in observation.snapshots.items()
                )
                _write_event(observation.event, observation.time_ns, fields)
            except Exception as error:
                _write_error(
                    observation.event,
                    observation.time_ns,
                    "observation_failed",
                    str(error),
                )
            finally:
                # Drop tensor/event references before marking the queue drained.
                observation = None
                self.queue.task_done()
                self.slots.release()
                with self.condition:
                    self.pending -= 1
                    self.condition.notify_all()

    def flush(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        with self.condition:
            while self.pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    _write_error(
                        "flush",
                        time.time_ns(),
                        "flush_timeout",
                        f"{self.pending} scalar observations remain pending",
                    )
                    return False
                self.condition.wait(remaining)
        return True


def _get_worker() -> _ObservationWorker:
    global _worker
    with _worker_lock:
        if _worker is None or _worker.pid != os.getpid():
            _worker = _ObservationWorker()
        return _worker


def _reset_after_fork() -> None:
    global _worker, _worker_lock
    _worker = None
    _worker_lock = threading.Lock()


def emit_smoke_tensor_event(
    event: str, tensor_fields: Dict[str, "Tensor"], **fields
) -> None:
    """Observe scalar tensors on their current producer streams.

    Each CUDA scalar gets an asynchronous copy into pinned host memory. An
    event on that same stream protects the copies and their source lifetimes;
    the background worker waits for it before decoding the CPU scalars. CUDA
    Graph capture is excluded, and non-scalar tensors are rejected so this
    cannot accidentally copy a model tensor or an entire cache table.
    """
    if not smoke_evidence_enabled():
        return
    sampled_time_ns = time.time_ns()
    try:
        import torch

        for name, tensor in tensor_fields.items():
            if name in fields or name in ("event", "time_ns"):
                raise ValueError(f"duplicate or reserved scalar field {name!r}")
            if not isinstance(tensor, torch.Tensor) or tensor.numel() != 1:
                raise ValueError(f"{name!r} must be a scalar tensor")
            if tensor.is_complex():
                raise ValueError(f"{name!r} must have a real scalar dtype")
            if tensor.device.type not in ("cpu", "cuda"):
                raise ValueError(f"unsupported scalar device: {tensor.device}")
            if tensor.is_cuda:
                with torch.cuda.device(tensor.device):
                    if torch.cuda.is_current_stream_capturing():
                        return
        # Metadata is a JSON snapshot too; caller mutation after sampling must
        # not change the eventual record.
        metadata = json.loads(json.dumps(fields, allow_nan=False))
        worker = _get_worker()
        if not worker.reserve():
            _write_error(
                event,
                sampled_time_ns,
                "queue_full",
                f"capacity {_MAX_PENDING_OBSERVATIONS} is exhausted",
            )
            return
    except Exception as error:
        _write_error(event, sampled_time_ns, "invalid_observation", str(error))
        return

    observation = _Observation(event, sampled_time_ns, metadata)
    streams = {}
    try:
        for name, tensor in tensor_fields.items():
            observation.sources.append(tensor)
            if tensor.is_cuda:
                with torch.cuda.device(tensor.device):
                    stream = torch.cuda.current_stream(tensor.device)
                    streams[tensor.device] = stream
                    snapshot = torch.empty_like(tensor, device="cpu", pin_memory=True)
                    observation.snapshots[name] = snapshot
                    snapshot.copy_(tensor.detach(), non_blocking=True)
            else:
                observation.snapshots[name] = tensor.detach().clone()
    except Exception as error:
        observation.error = str(error)
    finally:
        # Even a partially scheduled copy keeps its buffers alive in the queue
        # until an event recorded after the attempted copies completes.
        for device, stream in streams.items():
            try:
                with torch.cuda.device(device):
                    completed = torch.cuda.Event()
                    completed.record(stream)
                    observation.events.append(completed)
            except Exception as error:
                observation.error = str(error)
                observation.fallback_streams.append(stream)
        worker.submit(observation)


def flush_smoke_evidence(timeout_s: float = 10.0) -> bool:
    """Wait for the observer thread, without synchronizing the CUDA device."""
    worker = _worker
    if worker is None or worker.pid != os.getpid():
        return True
    return worker.flush(timeout_s)


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork)
atexit.register(flush_smoke_evidence)
