from __future__ import annotations

import asyncio
from contextlib import aclosing


class SGLangWorkerExecutor:
    """Execute routed requests in the tokenizer worker that owns their state."""

    def __init__(self, manager, generate, abort, codec, send, max_requests):
        self.manager = manager
        self.generate = generate
        self.abort = abort
        self.codec = codec
        self.send = send
        self.max_requests = max_requests
        self.tasks = {}
        self.closed = False

    def receive(self, message):
        call_id = message["call_id"]
        if message["kind"] == "abort":
            task = self.tasks.get(call_id)
            if task is not None:
                task.cancel()
            return
        if message["kind"] != "execute":
            raise ValueError("unknown tokenizer worker command")
        if call_id in self.tasks:
            raise ValueError("duplicate tokenizer worker call ID")
        if self.closed or len(self.tasks) >= self.max_requests:
            self.send(
                {
                    "kind": "response",
                    "call_id": call_id,
                    "error": self.codec.encode_error(
                        RuntimeError("tokenizer worker is unavailable")
                    ),
                }
            )
            return
        task = asyncio.create_task(self._execute(call_id, message["payload"]))
        self.tasks[call_id] = task
        task.add_done_callback(lambda completed: self._finished(call_id, completed))

    def _finished(self, call_id, task):
        if self.tasks.get(call_id) is task:
            self.tasks.pop(call_id)
        if not task.cancelled():
            task.exception()

    async def _execute(self, call_id, payload):
        obj = None
        try:
            obj = self.codec.decode_request(payload)
            obj.stream = True
            finished = False
            async with aclosing(self.generate(self.manager, obj, None)) as outputs:
                async for output in outputs:
                    finished = (
                        output.get("meta_info", {}).get("finish_reason") is not None
                    )
                    self.send(
                        {
                            "kind": "response",
                            "call_id": call_id,
                            "payload": self.codec.encode_response(output),
                            "terminal": finished,
                        }
                    )
            if not finished:
                raise RuntimeError(
                    "tokenizer worker stream ended without a finish reason"
                )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self.send(
                {
                    "kind": "response",
                    "call_id": call_id,
                    "error": self.codec.encode_error(error),
                }
            )
        finally:
            if obj is not None:
                self.abort(self.manager, obj.rid)

    def close(self):
        self.closed = True
        for task in tuple(self.tasks.values()):
            task.cancel()

    async def wait_closed(self):
        self.close()
        await asyncio.gather(*tuple(self.tasks.values()), return_exceptions=True)
