import asyncio
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import msgspec
from navi_sched.adapters.sglang import SGLangMultiTokenizer as multi
from navi_sched.adapters.sglang.NaviSchedSGLangPlugin import _internal_state
from navi_sched.adapters.sglang.SGLangConfig import SGLangConfig
from navi_sched.adapters.sglang.SGLangObserver import SGLangState


def message(value):
    return SGLangState(data=msgspec.msgpack.encode(value))


class NodeRuntime:
    def __init__(self, manager, generate):
        self.config = SGLangConfig("worker", "master", request_timeout_ms=1000)
        self.ready = asyncio.Event()
        self.snapshot = {}
        self.closed = False

    def receive(self, value):
        raise RuntimeError("agent startup failed")

    def snapshot_state(self):
        return {"agent_workload": {"request_count": 0}, "attention_dp": {"size": 4}}

    def close(self):
        self.closed = True


class SGLangMultiTokenizerTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.sent = []
        socket_mapping = SimpleNamespace(
            _mapping={},
            send_output=lambda worker, value, **kw: self.sent.append((worker, value)),
        )
        router = SimpleNamespace(
            socket_mapping=socket_mapping, all_worker_ipcs=set(), send_to_scheduler=None
        )
        with patch.object(multi, "SGLangRuntime", NodeRuntime):
            self.node = multi.SGLangRouterRuntime(router, SimpleNamespace())
        self.workers = []

    async def asyncTearDown(self):
        for worker in self.workers:
            worker.close()
            await worker.executor.wait_closed()
        self.node.close()

    def worker(self, name):
        manager = SimpleNamespace(
            tokenizer_ipc_name=name,
            auto_create_handle_loop=lambda: None,
            _dispatch_to_scheduler=self.node.receive_worker,
        )
        with patch.object(
            SGLangConfig, "from_env", return_value=self.node.runtime.config
        ):
            worker = multi.SGLangTokenizerRuntime(manager, None, None)
        self.workers.append(worker)
        return worker

    def deliver(self):
        for address, value in self.sent:
            for worker in self.workers:
                if worker.manager.tokenizer_ipc_name == address:
                    worker.receive(value)
        self.sent.clear()

    async def test_agent_start_failure_wakes_registered_workers(self):
        workers = [self.worker("a"), self.worker("b")]
        tasks = [asyncio.create_task(worker.start()) for worker in workers]
        await asyncio.sleep(0)
        with self.assertRaisesRegex(RuntimeError, "agent startup failed"):
            self.node.receive_scheduler(message({"kind": "ready"}))
        self.deliver()
        results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True), 0.2
        )
        self.assertTrue(
            all(isinstance(error, RuntimeError) for error in results), results
        )
        self.assertTrue(all("agent startup failed" in str(error) for error in results))

    async def test_late_worker_sees_startup_failure(self):
        with self.assertRaisesRegex(RuntimeError, "agent startup failed"):
            self.node.receive_scheduler(message({"kind": "ready"}))
        worker = self.worker("late")
        task = asyncio.create_task(worker.start())
        await asyncio.sleep(0)
        self.deliver()
        with self.assertRaisesRegex(RuntimeError, "agent startup failed"):
            await asyncio.wait_for(task, 0.2)

    async def test_concurrent_snapshot_queries_share_one_request(self):
        worker = self.worker("a")
        tasks = [asyncio.create_task(worker.get_snapshot()) for _ in range(2)]
        await asyncio.sleep(0)
        self.assertEqual(len(self.sent), 1)
        self.deliver()
        results = await asyncio.gather(*tasks)
        self.assertEqual(results, [self.node.get_snapshot()] * 2)

    async def test_expired_snapshot_reply_cannot_answer_new_query(self):
        worker = self.worker("a")
        first = asyncio.create_task(worker.get_snapshot())
        await asyncio.sleep(0)
        old = self.sent.pop()[1]
        first.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first
        worker.snapshot_deadline = 0
        second = asyncio.create_task(worker.get_snapshot())
        await asyncio.sleep(0)
        worker.receive(old)
        self.assertFalse(second.done())
        self.deliver()
        self.assertEqual(await second, self.node.get_snapshot())

    async def test_worker_exit_fails_stream_and_aborts_gpu_request(self):
        from unittest.mock import AsyncMock, Mock

        worker = "departed"
        self.node.workers.add(worker)
        self.node.router.all_worker_ipcs.add(worker)
        socket = Mock()
        self.node.router.socket_mapping._mapping[worker] = socket
        self.node.running["rid"] = (asyncio.current_task(), worker)
        stream = self.node.bridge.generate(worker, b"request")
        reader = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        abort = AsyncMock()
        with patch("sglang.srt.managers.io_struct.async_sock_send", abort):
            self.node.worker_exited(worker)
            with self.assertRaisesRegex(RuntimeError, "worker exited"):
                await reader
            await asyncio.sleep(0)
        self.assertFalse(self.node.workers)
        self.assertFalse(self.node.router.all_worker_ipcs)
        self.assertFalse(self.node.bridge.active)
        self.assertEqual(abort.await_args.args[1].rid, "rid")
        socket.close.assert_called_once_with(linger=0)
        self.assertEqual(len(self.sent), 1)  # No cancellation to the dead IPC.
        self.node.running.clear()
        self.node.receive_worker(
            message(
                {
                    "kind": "worker_register",
                    "worker_ipc": "replacement",
                    "pid": os.getpid(),
                }
            )
        )
        self.assertEqual(self.node.workers, {"replacement"})

    async def test_internal_state_preserves_native_dp_states(self):
        native = [{"dp_rank": 0}, {"dp_rank": 1}]

        async def original(manager):
            return native

        async def snapshot():
            return {"attention_dp": {"size": 2}}

        result = await _internal_state(
            original, SimpleNamespace(navi_sched=SimpleNamespace(get_snapshot=snapshot))
        )
        self.assertEqual(result[1], {"dp_rank": 1})
        self.assertEqual(result[0]["navi_sched_node"], {"attention_dp": {"size": 2}})
        self.assertEqual(native, [{"dp_rank": 0}, {"dp_rank": 1}])


if __name__ == "__main__":
    unittest.main()
