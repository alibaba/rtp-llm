# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Lossless tensor snapshots for explicit K3 model/sampler instrumentation.

The runner owns begin/end and CUDA-Graph capture/replay boundaries. This is
not a module hook: callers must record every required intermediate explicitly.
Files describe recorded observations, not proof of model coverage.
"""

from __future__ import annotations

import atexit
import hashlib
import itertools
import json
import os
import queue
import socket
import threading
import time
from contextlib import suppress
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, NoReturn

import torch

_observation_ids = itertools.count()


@dataclass
class _Snapshot:
    name: str
    value: torch.Tensor
    metadata: dict[str, Any]
    nbytes: int
    ready: Any = None


@dataclass
class _Frame:
    metadata: dict[str, Any]
    capture_key: str | None = None
    snapshots: list[_Snapshot] = field(default_factory=list)
    fragment: int = 0


def _json_copy(value: Any) -> Any:
    # Reject tensors, opaque runtime objects, and nonfinite metadata. Tensor
    # values belong in record(), where storage ownership is explicit.
    return json.loads(json.dumps(value, allow_nan=False))


class TensorTrace:
    """One producer thread and one asynchronous, lossless file writer.

    CUDA record() takes device snapshots on the producer stream. Frames spill
    into ordered fragments as the pending budget fills; the producer waits for
    the writer before taking further snapshots. end() marks the final fragment.
    max_pending_bytes bounds live snapshots plus staged CPU copies. Exhaustion
    aborts the trace explicitly instead of dropping or truncating tensors.

    For Graph: begin_capture() BEFORE torch.cuda.graph, record() inside, then
    end_capture() AFTER capture. After EVERY graph.replay() call replay() on
    its launch stream before that stream reuses the graph output buffers.
    Separate graph buckets need distinct capture keys. Dynamic request/step
    metadata is supplied at replay time, never borrowed from capture inputs.
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        identity: dict[str, Any],
        max_pending_bytes: int = 4 * 1024**3,
    ) -> None:
        if max_pending_bytes <= 0:
            raise ValueError("max_pending_bytes must be positive")
        self.identity = _json_copy(identity)
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self._owner = threading.get_ident()
        self._limit = max_pending_bytes
        self._pending = 0
        self._lock = threading.Lock()
        self._json_lock = threading.Lock()
        self._frame: _Frame | None = None
        self._graphs: dict[str, list[_Snapshot]] = {}
        self._copy_streams: dict[int, torch.cuda.Stream] = {}
        self._queue: queue.Queue = queue.Queue()
        self._error: str | None = None
        self._closed = False
        self._next_id = 0
        self._written = 0
        self._write_json("identity.json", self.identity)
        self._index = (self.directory / "index.jsonl").open("x")
        self._worker = threading.Thread(target=self._write_loop, daemon=True)
        self._worker.start()

    def _write_json(self, name: str, value: Any) -> None:
        path = self.directory / name
        temporary = path.with_suffix(path.suffix + ".part")
        with self._json_lock:
            temporary.write_text(
                json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n"
            )
            temporary.replace(path)

    def _fail(self, message: str) -> NoReturn:
        with self._lock:
            if self._error is None:
                self._error = message
        self._write_json("incomplete.json", {"error": self._error})
        raise RuntimeError(message)

    def _check(self) -> None:
        if threading.get_ident() != self._owner:
            raise RuntimeError("TensorTrace producer thread changed")
        if self._closed:
            raise RuntimeError("TensorTrace is closed")
        if self._error:
            raise RuntimeError(f"TensorTrace writer failed: {self._error}")

    def _reserve(self, count: int) -> None:
        with self._lock:
            overflow = self._pending + count > self._limit
            if not overflow:
                self._pending += count
        if overflow:
            self._fail("tensor trace pending-byte budget exceeded; no tensors dropped")

    def handoff(self) -> None:
        """Transfer idle producer ownership under the caller's external lock.

        Startup/capture and serving may use different threads. This method
        does not make concurrent producer calls safe; the runner must prevent
        the old producer from entering again until the transfer is complete.
        """
        if self._frame is not None or self._closed or self._error:
            raise RuntimeError("cannot hand off an active, closed, or failed trace")
        self._owner = threading.get_ident()

    def _release(self, count: int) -> None:
        with self._lock:
            self._pending -= count

    def _make_room(self, count: int) -> None:
        """Drain spillable data before reserving an eager snapshot and its D2H."""
        if self._frame is None:
            self._fail("reserving a snapshot outside a frame")
        if count > self._limit:
            self._fail("tensor trace pending-byte budget exceeded by one tensor")
        with self._lock:
            fits = self._pending + count <= self._limit
        if fits:
            return
        if self._frame.snapshots:
            self._flush_fragment(final=False)
        # The writer never needs the producer lock or the inference stream to
        # enqueue new work: every queued D2H already has its completion event.
        self._queue.join()
        self._check()

    def _flush_fragment(self, *, final: bool) -> None:
        frame = self._frame
        if frame is None:
            self._fail("flushing a fragment outside a frame")
        fragment = _Frame(
            {
                **frame.metadata,
                "trace_fragment": {"index": frame.fragment, "final": final},
            },
            snapshots=frame.snapshots,
        )
        frame.snapshots = []
        frame.fragment += 1
        self._enqueue(fragment)

    def begin(self, metadata: dict[str, Any]) -> None:
        self._check()
        if self._frame is not None:
            self._fail("nested trace frame; runner omitted end()")
        self._frame = _Frame(
            _json_copy(
                {
                    **metadata,
                    "observation_id": next(_observation_ids),
                    "monotonic_ns": time.monotonic_ns(),
                }
            )
        )

    def record(self, name: str, tensor: torch.Tensor | None, **metadata: Any) -> None:
        self._check()
        if self._frame is None:
            self._fail("record outside explicit trace frame")
        if tensor is None:
            return
        if tensor.layout != torch.strided or tensor.is_quantized:
            self._fail(
                "trace requires a dense tensor; "
                "explicitly unpack sparse/quantized values"
            )
        if tensor.device.type not in {"cpu", "cuda"}:
            self._fail(f"unsupported trace device: {tensor.device}")
        if tensor.is_cuda:
            with torch.accelerator.device_index(tensor.device.index):
                capturing = torch.cuda.is_current_stream_capturing()
            if capturing != (self._frame.capture_key is not None):
                self._fail(
                    "Graph capture has no matching begin_capture/end_capture scope"
                )
        meta = _json_copy(metadata)
        meta.update(
            {
                "shape": list(tensor.shape),
                "source_stride": list(tensor.stride()),
                "dtype": str(tensor.dtype),
                "source_device": str(tensor.device),
            }
        )
        nbytes = tensor.numel() * tensor.element_size()
        # Contiguous snapshots avoid retaining holes in a view's storage. The
        # original strides remain in metadata for layout reconstruction.
        # Reserve staging storage up front so flushing cannot itself overflow.
        capturing = self._frame.capture_key is not None
        reserved = nbytes * (2 if tensor.is_cuda and not capturing else 1)
        if not capturing:
            self._make_room(reserved)
        self._reserve(reserved)
        try:
            value = tensor.detach().clone(memory_format=torch.contiguous_format)
            ready = None
            if tensor.is_cuda and self._frame.capture_key is None:
                with torch.accelerator.device_index(tensor.device.index):
                    ready = torch.cuda.Event()
                    ready.record(torch.cuda.current_stream(tensor.device))
        except BaseException:
            self._release(reserved)
            self._fail(f"failed to snapshot {name}")
        self._frame.snapshots.append(_Snapshot(name, value, meta, nbytes, ready))

    def end(self) -> None:
        self._check()
        if self._frame is None or self._frame.capture_key is not None:
            self._fail("end() requires an eager/replay frame")
        self._flush_fragment(final=True)
        self._frame = None

    def _enqueue(self, frame: _Frame) -> None:
        sequence = self._next_id
        self._next_id += 1
        staged = []
        events = []
        reserve = sum(s.nbytes for s in frame.snapshots if s.value.is_cuda)
        try:
            for snapshot in frame.snapshots:
                value = snapshot.value
                if value.is_cuda:
                    device = value.device.index
                    with torch.accelerator.device_index(device):
                        stream = self._copy_streams.get(device)
                        if stream is None:
                            stream = torch.cuda.Stream(device=device)
                            self._copy_streams[device] = stream
                        host = torch.empty_like(value, device="cpu", pin_memory=True)
                        with torch.cuda.stream(stream):
                            stream.wait_event(snapshot.ready)
                            host.copy_(value, non_blocking=True)
                            done = torch.cuda.Event()
                            done.record(stream)
                        value.record_stream(stream)
                        events.append(done)
                else:
                    host = value
                staged.append((snapshot.name, host, snapshot.metadata))
        except BaseException as exc:
            # A staging failure is terminal; no successful close marker can be
            # produced. Keep the first error rather than hiding it in writer IO.
            self._fail(f"trace D2H staging failed: {exc}")
        total = reserve + sum(s.nbytes for s in frame.snapshots)
        self._queue.put(
            (sequence, frame.metadata, staged, events, frame.snapshots, total)
        )

    def begin_capture(self, key: str) -> None:
        self._check()
        if key in self._graphs:
            self._fail(f"capture key already registered: {key}")
        self.begin({"capture_key": key})
        if self._frame is None:
            self._fail("begin_capture did not create a frame")
        self._frame.capture_key = key

    def end_capture(self) -> None:
        self._check()
        if self._frame is None or self._frame.capture_key is None:
            self._fail("end_capture without begin_capture")
        self._graphs[self._frame.capture_key] = self._frame.snapshots
        self._frame = None

    def replay(
        self,
        key: str,
        metadata: dict[str, Any],
        live_tensors: dict[str, torch.Tensor] | None = None,
    ) -> None:
        self._check()
        if key not in self._graphs:
            self._fail(f"unknown graph capture key: {key}")
        self.begin({**metadata, "capture_key": key, "execution": "graph_replay"})
        for name, tensor in (live_tensors or {}).items():
            self.record(name, tensor, origin="live_replay_input")
        for snapshot in self._graphs[key]:
            # The clone runs after replay, before the next replay on the same
            # stream. It decouples the writer from reusable capture storage.
            self.record(
                snapshot.name,
                snapshot.value,
                captured_layout=snapshot.metadata,
                assert_zero=snapshot.metadata.get("assert_zero", False),
            )
        self.end()

    def _write_loop(self) -> None:
        with self._index as index:
            while True:
                item = self._queue.get()
                if item is None:
                    self._queue.task_done()
                    return
                sequence, metadata, staged, events, owners, total = item
                try:
                    for event in events:
                        event.synchronize()
                    for name, host, meta in staged:
                        if meta.get("assert_zero", False) and torch.count_nonzero(host):
                            raise RuntimeError(
                                f"nonzero diagnostic failure flag: {name}"
                            )
                    path = self.directory / f"frame-{sequence:08d}.pt"
                    temporary = path.with_suffix(".pt.part")
                    torch.save(
                        {
                            "schema_version": 1,
                            "sequence": sequence,
                            "metadata": metadata,
                            "tensors": [
                                {"name": name, "value": host, "metadata": meta}
                                for name, host, meta in staged
                            ],
                        },
                        temporary,
                    )
                    digest = hashlib.sha256()
                    with temporary.open("rb") as data:
                        for block in iter(lambda: data.read(1024 * 1024), b""):
                            digest.update(block)
                    size = temporary.stat().st_size
                    temporary.replace(path)
                    index.write(
                        json.dumps(
                            {
                                "sequence": sequence,
                                "path": path.name,
                                "sha256": digest.hexdigest(),
                                "bytes": size,
                                "metadata": metadata,
                                "tensor_count": len(staged),
                            },
                            allow_nan=False,
                        )
                        + "\n"
                    )
                    index.flush()
                    self._written += 1
                except BaseException as exc:
                    with self._lock:
                        self._error = self._error or f"writer: {exc}"
                    with suppress(OSError):
                        self._write_json("incomplete.json", {"error": self._error})
                    # close() still raises, even if the disk is full.
                finally:
                    del owners, staged, events, item
                    self._release(total)
                    self._queue.task_done()

    def close(self) -> None:
        if self._closed:
            return
        if threading.get_ident() != self._owner:
            raise RuntimeError("TensorTrace close must run on its producer thread")
        if self._frame is not None and self._error is None:
            self._error = "trace closed with unfinished frame"
        self._queue.put(None)
        self._worker.join()
        self._closed = True
        self._graphs.clear()
        if self._error:
            self._write_json("incomplete.json", {"error": self._error})
            raise RuntimeError(self._error)
        self._write_json(
            "recorder_closed.json",
            {
                "frames_written": self._written,
                "coverage_verified": False,
                "note": "writer flushed; full-model/stage/rank coverage "
                "requires a separate audit",
            },
        )


# Eager C++/Python boundaries share this runtime through the Python module.
# Model code inside torch.compile/CUDA Graph must use the explicit capture
# interface instead: event() deliberately rejects an unregistered capture.
_runtime_lock = threading.RLock()
_runtimes: dict[tuple[int, int], dict[str, Any]] = {}


def enabled() -> bool:
    return bool(os.environ.get("K3_TRACE_ROOT"))


def _runtime() -> dict[str, Any]:
    key = (os.getpid(), threading.get_ident())
    if key not in _runtimes:
        engine = os.environ["K3_TRACE_ENGINE"]
        if engine not in {"rtp", "vllm"}:
            raise ValueError("K3_TRACE_ENGINE must be rtp or vllm")
        identity = {
            "engine": engine,
            "run_id": os.environ["K3_TRACE_RUN_ID"],
            "host": socket.gethostname(),
            "pid": key[0],
            "thread_id": key[1],
            "rank_environment": {
                name: os.environ[name]
                for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "TP_RANK", "DP_RANK")
                if name in os.environ
            },
        }
        directory = Path(os.environ["K3_TRACE_ROOT"]) / (
            f"{engine}-{identity['host']}-{key[0]}-{key[1]}"
        )
        trace = TensorTrace(
            directory,
            identity=identity,
            max_pending_bytes=int(
                os.environ.get("K3_TRACE_MAX_PENDING_BYTES", str(4 * 1024**3))
            ),
        )
        _runtimes[key] = {"trace": trace, "stack": [], "next_scope": 0}
    return _runtimes[key]


def event(
    name: str,
    tensors: dict[str, torch.Tensor | None],
    metadata: dict[str, Any] | None = None,
) -> None:
    """Snapshot an explicit eager boundary; never infer valid rows or token IDs."""
    if not enabled():
        return
    with _runtime_lock:
        state = _runtime()
        trace = state["trace"]
        trace.begin(
            {
                "event": name,
                "scopes": state["stack"],
                "details": metadata or {},
                "execution": "eager_boundary",
            }
        )
        for tensor_name, tensor in tensors.items():
            trace.record(tensor_name, tensor)
        trace.end()


def enter_scope(name: str, metadata: dict[str, Any] | None = None) -> int:
    if not enabled():
        return -1
    with _runtime_lock:
        state = _runtime()
        scope_id = state["next_scope"]
        state["next_scope"] += 1
        state["stack"].append(
            {"id": scope_id, "name": name, "details": _json_copy(metadata or {})}
        )
        event("scope.begin", {})
        return scope_id


def exit_scope(scope_id: int, success: bool = True) -> None:
    if scope_id == -1:
        return
    with _runtime_lock:
        state = _runtime()
        if not state["stack"] or state["stack"][-1]["id"] != scope_id:
            state["trace"]._fail("trace scope nesting mismatch")
        event("scope.end", {}, {"success": success})
        state["stack"].pop()
        if not success:
            state["trace"]._fail("instrumented execution aborted")


def traced_scope(name: str):
    """Wrap an eager entry point, preserving the original inference exception."""

    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            if not enabled():
                return function(*args, **kwargs)
            scope_id = enter_scope(name)
            try:
                result = function(*args, **kwargs)
            except BaseException:
                with suppress(Exception):
                    exit_scope(scope_id, success=False)
                raise
            exit_scope(scope_id)
            return result

        return wrapped

    return decorate


def close_process() -> None:
    """Flush all eager writers once inference has stopped; fail incomplete runs."""
    errors = []
    with _runtime_lock:
        for (pid, _), state in list(_runtimes.items()):
            if pid != os.getpid():
                continue
            trace = state["trace"]
            # All runtime producer calls take _runtime_lock. Shutdown therefore
            # owns the recorder exclusively even if serving used another thread.
            trace._owner = threading.get_ident()
            if state["stack"]:
                trace._error = trace._error or "trace closed with unfinished scope"
            try:
                trace.close()
            except RuntimeError as exc:
                errors.append(str(exc))
        if errors:
            raise RuntimeError("; ".join(errors))


atexit.register(close_process)
