from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class _WorkerStream:
    worker: str
    frames: asyncio.Queue
    done: asyncio.Future
    abort_sent: bool = False


class SGLangWorkerBridge:
    """Bound streams between the Agent process and tokenizer workers."""

    def __init__(self, send, max_requests, max_buffered_responses, timeout_ms):
        for value in (max_requests, max_buffered_responses, timeout_ms):
            if type(value) is not int or value <= 0:
                raise ValueError("worker stream limits must be positive integers")
        self.send = send
        self.max_requests = max_requests
        self.max_buffered_responses = max_buffered_responses
        self.timeout_ms = timeout_ms
        self.active = {}
        self.closed = False

    def _abort(self, call_id, stream):
        if stream.abort_sent:
            return
        stream.abort_sent = True
        try:
            self.send(stream.worker, {"kind": "abort", "call_id": call_id})
        except Exception:
            logger.exception("Failed to send tokenizer worker cancellation")

    def receive(self, worker, call_id, payload=None, *, error=None, terminal=False):
        stream = self.active.get(call_id)
        if stream is None or stream.worker != worker or stream.done.done():
            return False
        if error is not None:
            stream.done.set_exception(error)
            return True
        try:
            if payload is not None:
                stream.frames.put_nowait(payload)
        except asyncio.QueueFull:
            stream.done.set_exception(
                RuntimeError("tokenizer worker response buffer is full")
            )
            self._abort(call_id, stream)
            return False
        if terminal:
            stream.done.set_result(None)
        return True

    async def generate(self, worker, payload):
        if self.closed:
            raise RuntimeError("tokenizer worker bridge is closed")
        if len(self.active) >= self.max_requests:
            raise RuntimeError("tokenizer worker bridge is at capacity")
        loop = asyncio.get_running_loop()
        call_id = uuid.uuid4().hex
        stream = _WorkerStream(
            worker, asyncio.Queue(self.max_buffered_responses), loop.create_future()
        )
        self.active[call_id] = stream
        deadline = loop.time() + self.timeout_ms / 1000
        getter = None
        completed = False
        try:
            self.send(
                worker, {"kind": "execute", "call_id": call_id, "payload": payload}
            )
            while True:
                if not stream.frames.empty():
                    yield stream.frames.get_nowait()
                    continue
                if stream.done.done():
                    await stream.done
                    completed = True
                    return
                getter = asyncio.create_task(stream.frames.get())
                finished, _ = await asyncio.wait(
                    (getter, stream.done),
                    timeout=max(0, deadline - loop.time()),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if not finished:
                    raise TimeoutError("tokenizer worker request timed out")
                if getter in finished:
                    frame = getter.result()
                    getter = None
                    yield frame
                else:
                    getter.cancel()
                    await asyncio.gather(getter, return_exceptions=True)
                    getter = None
        finally:
            if getter is not None:
                getter.cancel()
                await asyncio.gather(getter, return_exceptions=True)
            if not completed:
                self._abort(call_id, stream)
            self.active.pop(call_id, None)
            if stream.done.done() and not stream.done.cancelled():
                stream.done.exception()
            else:
                stream.done.cancel()

    def fail_worker(self, worker):
        for stream in self.active.values():
            if stream.worker == worker:
                # The target is gone; sending abort would recreate its dead IPC socket.
                stream.abort_sent = True
                if not stream.done.done():
                    stream.done.set_exception(RuntimeError("tokenizer worker exited"))

    def close(self):
        if self.closed:
            return
        self.closed = True
        for call_id, stream in tuple(self.active.items()):
            self._abort(call_id, stream)
            if not stream.done.done():
                stream.done.set_exception(
                    RuntimeError("tokenizer worker bridge is closed")
                )
