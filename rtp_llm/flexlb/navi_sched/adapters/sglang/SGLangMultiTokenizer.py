from __future__ import annotations

import asyncio
import logging
import os
import uuid
from contextlib import aclosing
from types import SimpleNamespace

import msgspec

from .SGLangConfig import SGLangConfig
from .SGLangForwardProtocol import SGLangForwardCodec
from .SGLangObserver import SGLangState
from .SGLangRuntime import SGLangRuntime
from .SGLangWorkerBridge import SGLangWorkerBridge
from .SGLangWorkerExecutor import SGLangWorkerExecutor
from .SGLangWorkerMonitor import SGLangWorkerMonitor

logger = logging.getLogger(__name__)


class RouterSocket:
    def __init__(self, socket, receive):
        self.socket = socket
        self.navi_sched_receive = receive

    def __getattr__(self, name):
        return getattr(self.socket, name)


class SGLangRouterRuntime:
    """Own the node Agent in the single tokenizer router process."""

    def __init__(self, router, server_args):
        self.router = router
        self.workers = set()
        self.running = {}
        self.failure = None
        self.worker_monitor = SGLangWorkerMonitor(
            self.worker_exited,
            max(1, getattr(server_args, "tokenizer_worker_num", 1)) * 2,
        )
        manager = SimpleNamespace(
            server_args=server_args,
            auto_create_handle_loop=lambda: None,
            abort_request=self.abort_request,
        )
        self.runtime = SGLangRuntime(manager, self.generate)
        config = self.runtime.config
        self.bridge = SGLangWorkerBridge(
            self.send,
            config.max_inflight_requests,
            config.max_buffered_responses,
            config.request_timeout_ms,
        )

    def send(self, worker, values):
        payload = msgspec.msgpack.encode(values)
        if len(payload) > self.runtime.config.max_payload_bytes:
            raise ValueError("tokenizer worker message exceeds max_payload_bytes")
        self.router.socket_mapping.send_output(
            worker, SGLangState(data=payload), is_tokenizer=True
        )

    def receive_worker(self, message):
        values = msgspec.msgpack.decode(message.data)
        worker = values["worker_ipc"]
        if values["kind"] == "worker_register":
            if self.failure is not None:
                self.send(worker, {"kind": "worker_error", "error": self.failure})
                return
            try:
                self.worker_monitor.register(worker, values["pid"])
            except ProcessLookupError:
                self.worker_exited(worker)
                return
            self.workers.add(worker)
            if self.runtime.ready.is_set():
                self.send(
                    worker,
                    {
                        "kind": "worker_ready",
                        "policy": self.runtime.snapshot["cache_policy"],
                    },
                )
        elif values["kind"] == "worker_unregister":
            self.worker_monitor.remove(worker)
            self.worker_exited(worker)
        elif values["kind"] == "response":
            error = (
                self.runtime.codec.decode_error(values["error"])
                if "error" in values
                else None
            )
            self.bridge.receive(
                worker,
                values["call_id"],
                values.get("payload"),
                error=error,
                terminal=values.get("terminal", False),
            )
        elif values["kind"] == "worker_snapshot":
            self.send(
                worker,
                {
                    "kind": "worker_snapshot_result",
                    "query_id": values["query_id"],
                    "snapshot": self.get_snapshot(),
                },
            )
        else:
            raise ValueError("unknown tokenizer router message")

    def receive_scheduler(self, message):
        was_ready = self.runtime.ready.is_set()
        try:
            self.runtime.receive(message)
        except Exception as error:
            self.failure = str(error)[:4096]
            for worker in self.workers:
                try:
                    self.send(worker, {"kind": "worker_error", "error": self.failure})
                except Exception:
                    logger.exception(
                        "Failed to notify tokenizer worker of Agent failure"
                    )
            self.close()
            raise
        if not was_ready and self.runtime.ready.is_set():
            for worker in self.workers:
                self.send(
                    worker,
                    {
                        "kind": "worker_ready",
                        "policy": self.runtime.snapshot["cache_policy"],
                    },
                )

    async def generate(self, manager, obj, request):
        if not self.workers:
            raise RuntimeError("no tokenizer worker is registered")
        counts = {worker: 0 for worker in self.workers}
        for stream in self.bridge.active.values():
            counts[stream.worker] += 1
        worker = min(counts, key=lambda name: (counts[name], name))
        self.running[obj.rid] = (asyncio.current_task(), worker)
        try:
            payload = self.runtime.codec.encode_request(obj)
            async with aclosing(self.bridge.generate(worker, payload)) as outputs:
                async for output in outputs:
                    yield self.runtime.codec.decode_response(output)
        finally:
            self.running.pop(obj.rid, None)

    def abort_request(self, rid):
        entry = self.running.get(rid)
        if entry is not None and entry[0] is not asyncio.current_task():
            entry[0].cancel()

    def worker_exited(self, worker):
        from sglang.srt.managers.io_struct import AbortReq, async_sock_send

        self.workers.discard(worker)
        self.bridge.fail_worker(worker)
        for rid, (_, owner) in tuple(self.running.items()):
            if owner == worker:
                # A dead tokenizer cannot cancel GPU work itself.
                task = asyncio.create_task(
                    async_sock_send(self.router.send_to_scheduler, AbortReq(rid=rid))
                )
                task.add_done_callback(self._abort_finished)
        self.router.all_worker_ipcs.discard(worker)
        socket = self.router.socket_mapping._mapping.pop(worker, None)
        if socket is not None:
            socket.close(linger=0)

    @staticmethod
    def _abort_finished(task):
        if not task.cancelled() and task.exception() is not None:
            logger.error(
                "Failed to abort GPU request after tokenizer exit",
                exc_info=task.exception(),
            )

    def get_snapshot(self):
        result = dict(
            self.runtime.snapshot_state(),
            tokenizer_worker_count=len(self.workers),
            worker_stream_count=len(self.bridge.active),
        )
        return result

    def close(self):
        self.worker_monitor.close()
        self.bridge.close()
        self.runtime.close()


class SGLangTokenizerRuntime:
    """Keep HTTP client routing separate from worker-local execution."""

    def __init__(self, manager, generate, abort):
        self.manager = manager
        self.original_generate = generate
        self.config = SGLangConfig.from_env()
        self.codec = SGLangForwardCodec(self.config.max_payload_bytes)
        self.ready = asyncio.Event()
        self.transport = None
        self.policy = None
        self.registered = False
        self.closed = False
        self.failure = None
        self.snapshot = {}
        self.snapshot_query = None
        self.snapshot_future = None
        self.snapshot_deadline = 0
        self.executor = SGLangWorkerExecutor(
            manager,
            generate,
            abort,
            self.codec,
            self.send,
            self.config.max_inflight_requests,
        )

    def send(self, values):
        values = dict(values, worker_ipc=self.manager.tokenizer_ipc_name)
        payload = msgspec.msgpack.encode(values)
        if len(payload) > self.config.max_payload_bytes:
            raise ValueError("tokenizer worker message exceeds max_payload_bytes")
        self.manager._dispatch_to_scheduler(SGLangState(data=payload))

    def receive(self, message):
        values = msgspec.msgpack.decode(message.data)
        if values["kind"] == "worker_ready":
            self.policy = values["policy"]
            self.ready.set()
        elif values["kind"] == "worker_error":
            self.failure = values["error"]
            self.ready.set()
        elif values["kind"] == "worker_snapshot_result":
            if (
                values["query_id"] == self.snapshot_query
                and self.snapshot_future is not None
                and not self.snapshot_future.done()
            ):
                self.snapshot_future.set_result(values["snapshot"])
        else:
            self.executor.receive(values)

    async def start(self):
        if self.closed:
            raise RuntimeError("tokenizer worker runtime is closed")
        self.manager.auto_create_handle_loop()
        if not self.registered:
            self.send({"kind": "worker_register", "pid": os.getpid()})
            self.registered = True
        await asyncio.wait_for(self.ready.wait(), self.config.request_timeout_ms / 1000)
        if self.failure is not None:
            raise RuntimeError(self.failure)

    async def generate(self, obj, request):
        from .SGLangTransport import SGLangTransport

        await self.start()
        if self.transport is None:
            self.transport = SGLangTransport(
                self.manager, self.config, page_size=self.policy["page_size"]
            )
            self.snapshot["transport"] = self.transport.snapshot
        async with aclosing(
            self.transport.generate(obj, request, self.original_generate)
        ) as outputs:
            async for output in outputs:
                yield output

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.registered:
            try:
                self.send({"kind": "worker_unregister"})
            except Exception:
                logger.exception("Failed to unregister tokenizer worker")
        if self.snapshot_future is not None and not self.snapshot_future.done():
            self.snapshot_future.cancel()
        self.executor.close()
        if self.transport is not None:
            self.transport.close()

    async def get_snapshot(self):
        loop = asyncio.get_running_loop()
        if (
            self.snapshot_future is None
            or self.snapshot_future.done()
            or loop.time() >= self.snapshot_deadline
        ):
            if self.snapshot_future is not None and not self.snapshot_future.done():
                self.snapshot_future.cancel()
            self.snapshot_query = uuid.uuid4().hex
            self.snapshot_future = loop.create_future()
            self.snapshot_deadline = loop.time() + 5
            self.send({"kind": "worker_snapshot", "query_id": self.snapshot_query})
        return await asyncio.wait_for(
            asyncio.shield(self.snapshot_future),
            max(0, self.snapshot_deadline - loop.time()),
        )
