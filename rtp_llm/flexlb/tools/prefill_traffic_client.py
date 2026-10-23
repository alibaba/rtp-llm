"""Open-loop Schedule RPCs with one completion monitor, independent of queue wait."""

import concurrent.futures
import math
import queue
import threading
import time
from dataclasses import dataclass


@dataclass
class _Query:
    row: dict
    result: concurrent.futures.Future
    rpc: object = None
    stream: object = None
    handled: bool = False
    stream_started_s: float = 0.0
    failure: str = ""


class AsyncQueryClient:
    """Submit immediately; waiting Schedule RPCs never occupy a Python worker.

    The gRPC callback only queues a completed RPC. A single monitor starts the
    existing StreamHandle reader and observes its termination. Client cleanup
    cancels outstanding calls and waits a bounded time for readers to exit.
    """

    def __init__(
        self,
        ops,
        schedule_timeout_s=15.0,
        stream_timeout_s=20.0,
        poll_interval_s=0.005,
        close_timeout_s=5.0,
    ):
        if any(
            not math.isfinite(value) or value <= 0
            for value in (
                schedule_timeout_s,
                stream_timeout_s,
                poll_interval_s,
                close_timeout_s,
            )
        ):
            raise ValueError("client timeouts and polling interval must be positive")
        self.ops = ops
        self.schedule_timeout_s = schedule_timeout_s
        self.stream_timeout_s = stream_timeout_s
        self.poll_interval_s = poll_interval_s
        self.close_timeout_s = close_timeout_s
        self._stub = ops.schedule_pb2_grpc.FlexlbServiceStub(
            ops._channel(ops.master_target())
        )
        self._lock = threading.Lock()
        self._ready = queue.SimpleQueue()
        self._wake = threading.Event()
        self._closing = threading.Event()
        self._entries = {}
        self._request_ids = set()
        self._thread = None
        self._close_deadline = None
        self._errors = []

    def __enter__(self):
        if self._thread is not None:
            raise RuntimeError("traffic client cannot be entered twice")
        self._thread = threading.Thread(
            target=self._monitor, name="traffic-completion-monitor", daemon=True
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def submit(self, item, planned, epoch_ms):
        started = time.monotonic()
        entry = _Query(
            dict(
                item,
                planned_s=planned,
                planned_epoch_ms=epoch_ms,
                started_s=started,
                pacing_lag_s=started - planned,
                success=False,
            ),
            concurrent.futures.Future(),
        )
        request_id = item["request_id"]
        with self._lock:
            if self._thread is None or self._closing.is_set():
                raise RuntimeError("traffic client is not open")
            if request_id in self._request_ids:
                raise ValueError(f"duplicate request id: {request_id}")
            self._request_ids.add(request_id)
            self._entries[request_id] = entry
            try:
                request = self.ops.build_schedule_request(
                    request_id,
                    input_len=item["input_tokens"],
                    output_len=2,
                    block_keys=item["keys"],
                )
                request.session_id = item["session"]
                entry.rpc = self._stub.Schedule.future(
                    request, timeout=self.schedule_timeout_s
                )
                entry.row["schedule_launched_s"] = time.monotonic()
                entry.row["schedule_launch_lag_s"] = (
                    entry.row["schedule_launched_s"] - planned
                )

                def done(rpc):
                    self._ready.put((request_id, rpc, time.monotonic()))
                    self._wake.set()

                entry.rpc.add_done_callback(done)
            except Exception as error:
                entry.failure = repr(error)
        if entry.failure:
            self._finish(request_id, entry)
        self._wake.set()
        return entry.result

    def _finish(self, request_id, entry, ended_s=None):
        if entry.failure:
            entry.row.update(success=False, error=entry.failure)
        entry.row["ended_s"] = ended_s if ended_s is not None else time.monotonic()
        with self._lock:
            if self._entries.get(request_id) is not entry:
                return
            self._entries.pop(request_id)
        if entry.result.set_running_or_notify_cancel():
            entry.result.set_result(entry.row)

    def _schedule_done(self, request_id, rpc, completed_s):
        with self._lock:
            entry = self._entries.get(request_id)
        if entry is None or entry.handled:
            return
        entry.handled = True
        entry.row["schedule_completed_s"] = completed_s
        if self._closing.is_set() or entry.result.cancelled():
            entry.failure = "traffic client closed or query cancelled"
            self._finish(request_id, entry)
            return
        try:
            response = rpc.result()  # The callback only queues completed RPCs.
            entry.row.update(
                code=response.code,
                worker=self.ops.prefill_addr(response),
                enqueued_by_master=response.enqueued_by_master,
            )
            if not response.success or response.code != 200:
                entry.row["error"] = response.error_message
                self._finish(request_id, entry, completed_s)
                return
            generated = self.ops.build_generate_input(
                request_id,
                input_len=entry.row["input_tokens"],
                output_len=2,
                block_keys=entry.row["keys"],
            )
            entry.stream_started_s = time.monotonic()
            entry.stream = self.ops.start_stream(response, request_id, generated)
            launched = time.monotonic()
            entry.row.update(
                stream_started_s=entry.stream_started_s,
                stream_launch_completed_s=launched,
                stream_launch_lag_s=launched - completed_s,
            )
        except Exception as error:
            entry.failure = repr(error)
            self._finish(request_id, entry)

    def _poll(self, request_id, entry, now):
        closing = self._closing.is_set() or entry.result.cancelled()
        if closing and not entry.failure:
            entry.failure = "traffic client closed or query cancelled"
            if entry.rpc is not None:
                entry.rpc.cancel()
            if entry.stream is not None:
                entry.stream.cancel()
                entry.row["stream_cancelled_s"] = now
        if entry.stream is None:
            if closing:
                self._finish(request_id, entry)
            elif now - entry.row["started_s"] > self.schedule_timeout_s + 0.1:
                entry.failure = "Schedule RPC did not terminate by its deadline"
                entry.rpc.cancel()
                self._finish(request_id, entry)
            return
        stream = entry.stream
        if stream.thread.is_alive():
            if (
                not entry.failure
                and now - entry.stream_started_s > self.stream_timeout_s
            ):
                entry.failure = "stream did not terminate before client timeout"
                stream.cancel()
                entry.row["stream_cancelled_s"] = now
            if (
                entry.failure
                and now - entry.row.get("stream_cancelled_s", now)
                > self.close_timeout_s
            ):
                self._errors.append(f"stream reader did not stop: {request_id}")
                self._finish(request_id, entry)
            return
        snap = stream.snap
        entry.row.update(
            success=bool(
                not entry.failure
                and snap.terminated
                and snap.completed
                and not snap.error
                and not snap.stream_error_code
            ),
            error=snap.error,
            stream_error_code=snap.stream_error_code,
            stream_error_message=getattr(snap, "stream_error_message", None),
            stream_terminated=snap.terminated,
        )
        if snap.first_received_s is not None:
            entry.row["first_received_s"] = snap.first_received_s
            entry.row["ttft_s"] = snap.first_received_s - entry.row["started_s"]
        self._finish(request_id, entry, snap.terminated_s)

    def _monitor(self):
        try:
            while True:
                self._wake.wait(self.poll_interval_s)
                self._wake.clear()
                for _ in range(128):
                    try:
                        request_id, rpc, completed = self._ready.get_nowait()
                    except queue.Empty:
                        break
                    self._schedule_done(request_id, rpc, completed)
                if not self._ready.empty():
                    self._wake.set()
                with self._lock:
                    entries = list(self._entries.items())
                now = time.monotonic()
                for request_id, entry in entries:
                    self._poll(request_id, entry, now)
                if self._closing.is_set():
                    with self._lock:
                        remaining = list(self._entries.items())
                    if not remaining:
                        return
                    if now >= self._close_deadline:
                        self._errors.append(
                            "client cleanup timed out: "
                            + repr([request_id for request_id, _ in remaining])
                        )
                        for request_id, entry in remaining:
                            self._finish(request_id, entry)
                        return
        except Exception as error:
            self._errors.append(repr(error))
            self._closing.set()
            with self._lock:
                remaining = list(self._entries.items())
            for request_id, entry in remaining:
                if entry.rpc is not None:
                    entry.rpc.cancel()
                if entry.stream is not None:
                    entry.stream.cancel()
                entry.failure = "completion monitor failed: " + repr(error)
                self._finish(request_id, entry)

    def close(self):
        if self._thread is None:
            return
        if not self._closing.is_set():
            self._close_deadline = time.monotonic() + self.close_timeout_s
            self._closing.set()
        self._wake.set()
        self._thread.join(self.close_timeout_s + 1)
        if self._thread.is_alive():
            raise RuntimeError("traffic completion monitor did not stop")
        if self._errors:
            raise RuntimeError(
                "traffic client cleanup failed: " + "; ".join(self._errors)
            )
