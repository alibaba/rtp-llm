"""Owned stream/event resources; callers retain their scheduling policy.

Execution dependencies and allocator lifetimes are separate contracts. Graph
owners must prepare resources before capture and retain them until all graph
users are drained. Closing rejects submissions; it does not synchronize or
release resources still referenced by captured graphs.
"""

import threading
from contextlib import contextmanager

import torch


def stream_key(stream):
    return str(stream.device), int(stream.cuda_stream)


def record_stream(value, stream):
    """Register nested tensor owners on their consuming stream."""
    if isinstance(value, torch.Tensor):
        if value.is_cuda:
            value.record_stream(stream)
    elif isinstance(value, (tuple, list)):
        for item in value:
            record_stream(item, stream)
    elif isinstance(value, dict):
        for item in value.values():
            record_stream(item, stream)


class OverlapResources:
    """One nonreentrant side stream and two model-lifetime reusable events.

    This object neither selects branches nor chooses when to fork or join.
    Eager users may validate the default caller themselves. Graph users bind
    one actual caller before capture and use separate resources per caller.
    The submission lock protects host enqueue/event reuse, not Graph replay;
    runtime owners must still serialize graphs sharing a workspace.
    """

    def __init__(self, device, *, name="overlap"):
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("overlap resources require a torch.cuda device")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("create overlap resources before capture")
        self.name = name
        with torch.cuda.device(self.device):
            self.side_stream = torch.cuda.Stream(device=self.device)
            self.producer = torch.cuda.Event(enable_timing=False)
            self.done = torch.cuda.Event(enable_timing=False)
        self.lock = threading.Lock()
        self.caller = None
        self.caller_key = None
        self.closed = False

    def storage_signature(self):
        return (
            stream_key(self.side_stream),
            id(self.producer),
            id(self.done),
            self.closed,
        )

    @contextmanager
    def submission(self):
        if not self.lock.acquire(blocking=False):
            raise RuntimeError(f"{self.name} is not reentrant")
        try:
            if self.closed:
                raise RuntimeError(f"{self.name} is closed")
            yield self
        finally:
            self.lock.release()

    def bind_caller(self, caller):
        """Initialize events on the actual Graph caller, outside capture."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("bind overlap caller before capture")
        if torch.device(caller.device) != self.device:
            raise ValueError("overlap caller belongs to another device")
        with self.submission():
            key = stream_key(caller)
            if self.caller_key is not None:
                if key != self.caller_key:
                    raise RuntimeError(
                        "overlap caller changed; use an independent plan"
                    )
                return
            self.fork(caller)
            self.join(caller)
            self.caller, self.caller_key = caller, key

    def fork(self, caller):
        self.producer.record(caller)
        self.side_stream.wait_event(self.producer)

    def record_done(self):
        self.done.record(self.side_stream)

    def wait_done(self, caller):
        caller.wait_event(self.done)

    def join(self, caller):
        self.record_done()
        self.wait_done(caller)

    def close(self):
        """Reject new submissions; the owner retains/drains captured users."""
        if not self.lock.acquire(blocking=False):
            raise RuntimeError(f"cannot close an active {self.name} call")
        try:
            self.closed = True
        finally:
            self.lock.release()
