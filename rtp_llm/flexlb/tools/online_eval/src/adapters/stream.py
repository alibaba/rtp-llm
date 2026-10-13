"""Own Fetch stream consumption and terminal observations."""

import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional
import grpc

STREAM_CANCEL_TIMEOUT_S = 5.0

@dataclass
class StreamSnapshot:
    """Collected state from a FetchResponse / GenerateStreamCall stream."""

    outputs: List[object] = field(default_factory=list)
    first_received: bool = False
    completed: bool = False
    error: Optional[str] = None
    terminated: bool = False
    terminated_s: Optional[float] = None  # monotonic time when stream ended
    # Monotonic time when the FIRST output arrived — the client-observed
    # TTFT anchor for graded property P7 (see balance_overload_avoid_prefill;
    # under BATCH dispatch the first FetchResponse message only surfaces
    # after decode completes, so P7 uses the completion-duration口径 there).
    first_received_s: Optional[float] = None
    # In-band typed error frame (GenerateOutputsPB.error_info, RpcErrorPB):
    # the engine terminates failed streams IN-BAND — a frame carrying
    # error_info is the LAST frame, then the stream completes with gRPC
    # status OK (see priority.py _StreamTerminal's A1 note).  The raw code
    # is an int: proto3 open enums surface non-production values (e.g. an
    # injected 8500) as plain ints on the wire.  These fields are PURE
    # additions — snap.error / snap.completed semantics stay untouched, so
    # every existing consumer keeps its exact prior behavior.
    stream_error_code: Optional[int] = None
    stream_error_message: Optional[str] = None


class StreamHandle:
    """A gRPC stream consumed on a background thread."""

    def __init__(self, call, snap: StreamSnapshot):
        self.call = call
        self.snap = snap
        self.thread = threading.Thread(target=self._consume, daemon=True)
        self.thread.start()

    def _consume(self) -> None:
        try:
            for output in self.call:
                if not self.snap.first_received:
                    self.snap.first_received = True
                    self.snap.first_received_s = time.monotonic()
                self.snap.outputs.append(output)
                # Preserve both protocol terminal errors and transport completion.
                try:
                    if output.HasField("error_info"):
                        self.snap.stream_error_code = int(output.error_info.error_code)
                        self.snap.stream_error_message = output.error_info.error_message
                except AttributeError:
                    pass
                finished = output.flatten_output.finished
                if finished and any(finished):
                    self.snap.completed = True
        except grpc.RpcError as exc:
            # Client-side cancellation is not an error (mirrors legacy
            # asyncio.CancelledError handling).
            if exc.code() != grpc.StatusCode.CANCELLED:
                self.snap.error = repr(exc)
        except Exception as exc:
            self.snap.error = repr(exc)
        finally:
            self.snap.terminated = True
            self.snap.terminated_s = time.monotonic()

    def wait_end(self, timeout_s: float = STREAM_CANCEL_TIMEOUT_S) -> bool:
        self.thread.join(timeout_s)
        if self.thread.is_alive():
            self.cancel()
            self.thread.join(5.0)
            return False
        return True

    def cancel(self) -> None:
        try:
            self.call.cancel()
        except Exception:
            pass
