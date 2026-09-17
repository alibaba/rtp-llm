import asyncio
import concurrent.futures
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import TestCase, main, mock

import grpc
import torch

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    MultimodalInputsPB,
    StatusVersionPB,
)
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc import (
    MultimodalRpcServiceStub,
    add_MultimodalRpcServiceServicer_to_server,
)
from rtp_llm.multimodal.greennet_hook import GreenNetVerdict
from rtp_llm.multimodal.mm_embedding_cache import MMEmbeddingCacheEntry
from rtp_llm.multimodal.test import multimodal_process_engine_test as helpers
from rtp_llm.server.vit_rpc_server import MultimodalRpcServer, _rpc_timeout_ms


class MMCancellationTest(TestCase):
    _make_input = helpers.AsyncSubmitGetEmbeddingTest._make_input

    def _make_engine(self, **kwargs):
        engine = helpers.AsyncSubmitGetEmbeddingTest._make_engine(self, **kwargs)
        self.addCleanup(engine.stop)
        return engine

    def _block_compute(self, engine, key):
        started, release = threading.Event(), threading.Event()
        calls = []

        def compute(inputs, cache_key, entry, *args, **kwargs):
            calls.append(cache_key)
            if cache_key == key:
                started.set()
                self.assertTrue(release.wait(5))
            entry.set_greennet_verdict(GreenNetVerdict(passed=True))
            entry.complete((torch.ones(1, 1), None))

        engine._async_compute = compute
        self.addCleanup(release.set)
        return started, release, calls

    @contextmanager
    def _rpc(self, engine, inp, release):
        started, done = threading.Event(), threading.Event()
        servicer = MultimodalRpcServer(engine)
        register = servicer._register_rpc_completion

        def observe(context):
            event = register(context)
            context.add_callback(done.set)
            started.set()
            return event

        servicer._register_rpc_completion = observe
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            server = grpc.server(pool)
            add_MultimodalRpcServiceServicer_to_server(servicer, server)
            port = server.add_insecure_port("127.0.0.1:0")
            server.start()
            channel = grpc.insecure_channel(f"127.0.0.1:{port}")
            try:
                grpc.channel_ready_future(channel).result(timeout=2)
                with mock.patch(
                    "rtp_llm.server.vit_rpc_server.trans_mm_input", return_value=[inp]
                ):
                    yield MultimodalRpcServiceStub(channel), started, done
            finally:
                release.set()
                channel.close()
                server.stop(0).wait(timeout=2)

    def test_invalid_later_input_leaves_no_pending_claims(self):
        engine = self._make_engine()
        inp = self._make_input("fake://valid")
        with self.assertRaises(ValueError):
            engine.async_submit([inp, self._make_input("")], request_id=10)
        self.assertIsNone(engine._embedding_cache.peek(inp.cache_key()))
        self.assertEqual(engine._async_tasks, {})
        self.assertEqual(engine._async_request_tasks, {})
        self.assertEqual(engine._async_admitted, 0)

    def test_partial_claim_failure_wakes_other_waiter_and_removes_orphan(self):
        engine = self._make_engine()
        inputs = [self._make_input("fake://a"), self._make_input("fake://b")]
        acquire = engine._async_cache.try_acquire
        claimed = []

        def fail_second(key):
            if claimed:
                entry = claimed[0]
                engine._async_tasks[entry].request_ids.add(20)
                engine._async_request_tasks[20] = {entry}
                raise RuntimeError("claim failed")
            state, entry = acquire(key)
            claimed.append(entry)
            return state, entry

        with mock.patch.object(
            engine._async_cache, "try_acquire", side_effect=fail_second
        ):
            with self.assertRaisesRegex(RuntimeError, "claim failed"):
                engine.async_submit(inputs, request_id=10)
        with self.assertRaisesRegex(RuntimeError, "claim failed"):
            claimed[0].wait_ready(timeout=0)
        self.assertEqual(engine._async_tasks, {})
        self.assertEqual(engine._async_request_tasks, {})
        self.assertEqual(engine._async_admitted, 0)
        self.assertIsNone(engine._embedding_cache.peek(inputs[0].cache_key()))

    def test_preprocess_submit_failure_rolls_back_sync_claim(self):
        engine = self._make_engine()
        inp = self._make_input("fake://submit-failed")
        with mock.patch.object(
            engine.preprocess_executor,
            "submit",
            side_effect=RuntimeError("submit failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "submit failed"):
                engine._create_work_items([inp])
        self.assertIsNone(engine._embedding_cache.peek(inp.cache_key()))
        self.assertEqual(
            engine._embedding_cache.try_acquire(inp.cache_key())[0], "miss"
        )

    def test_cancel_wait_keeps_shared_compute(self):
        for method in ("get_embedding_result", "wait_greennet_verdict"):
            with self.subTest(method=method):
                engine = self._make_engine()
                engine._greennet_provider = helpers._StubGreenNetProvider(
                    GreenNetVerdict(passed=True)
                )
                inp = self._make_input("fake://shared")
                started, release, calls = self._block_compute(engine, inp.cache_key())
                cancelled = threading.Event()
                wait = getattr(engine, method)
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(
                        wait, [inp], request_id=10, cancellation_event=cancelled
                    )
                    try:
                        self.assertTrue(started.wait(2))
                        engine.async_submit([inp], request_id=20)
                        cancelled.set()
                        with self.assertRaises(FtRuntimeException) as raised:
                            future.result(timeout=1)
                        self.assertEqual(
                            raised.exception.exception_type,
                            ExceptionType.CANCELLED_ERROR,
                        )
                        entry = engine._embedding_cache.peek(inp.cache_key())
                        self.assertEqual(engine._async_tasks[entry].request_ids, {20})
                        self.assertFalse(entry.is_done)
                    finally:
                        release.set()
                result = wait([inp], request_id=20, timeout_ms=2000)
                self.assertTrue(
                    result.passed if method == "wait_greennet_verdict" else result
                )
                self.assertEqual(calls, [inp.cache_key()])

    def test_expired_deadline_does_not_submit(self):
        engine = self._make_engine()
        with self.assertRaises(TimeoutError):
            engine.get_embedding_result(
                [self._make_input("fake://expired")], timeout_ms=0
            )
        self.assertEqual(engine._async_tasks, {})

    def test_completed_failure_is_not_mistaken_for_wait_timeout(self):
        entry = MMEmbeddingCacheEntry()
        error = TimeoutError("producer failed")
        entry.fail(error)
        with self.assertRaises(TimeoutError) as raised:
            entry.wait_ready(timeout=10, cancellation_event=threading.Event())
        self.assertIs(raised.exception, error)

    def test_greennet_timeout_cleans_preprocess_and_late_handle(self):
        for late_handle in (False, True):
            with self.subTest(late_handle=late_handle):
                engine = self._make_engine()
                engine._greennet_timeout_s = 0.1
                stopped = threading.Event()

                class Handle(helpers._StubGreenNetHandle):
                    def cancel(self):
                        stopped.set()
                        super().cancel()

                class Provider(helpers._StubGreenNetProvider):
                    async def preprocess_and_submit(self, request, inputs):
                        try:
                            await asyncio.sleep(10)
                        except asyncio.CancelledError:
                            if late_handle:
                                return Handle(inputs, GreenNetVerdict(passed=True))
                            stopped.set()
                            raise

                engine._greennet_provider = Provider(GreenNetVerdict(passed=True))
                with self.assertRaises(concurrent.futures.TimeoutError):
                    engine._begin_greennet([self._make_input("fake://slow")])
                self.assertTrue(stopped.wait(1))

    def test_greennet_stages_share_one_deadline(self):
        engine = self._make_engine()
        engine._greennet_timeout_s = 0.3

        class Provider(helpers._StubGreenNetProvider):
            async def preprocess_and_submit(self, request, inputs):
                await asyncio.sleep(0.15)
                return helpers._StubGreenNetHandle(
                    inputs, GreenNetVerdict(passed=True), delay=0.2
                )

        engine._greennet_provider = Provider(GreenNetVerdict(passed=True))
        _, verdict, handle = engine._begin_greennet([self._make_input("fake://budget")])
        try:
            with self.assertRaises(asyncio.TimeoutError):
                verdict.result(timeout=2)
        finally:
            engine._cancel_greennet(handle)

    def test_rpc_completion_preserves_success_and_cancels_abandoned_work(self):
        cases = [
            (True, "success"),
            (True, "cancel"),
            (True, "deadline"),
            (False, "cancel"),
            (False, "deadline"),
        ]
        for queued, outcome in cases:
            with self.subTest(queued=queued, outcome=outcome):
                engine = self._make_engine(vit_concurrency=1, vit_max_queue_size=4)
                engine._greennet_provider = helpers._StubGreenNetProvider(
                    GreenNetVerdict(passed=True)
                )
                blocker = self._make_input("fake://blocker")
                target = self._make_input("fake://queued") if queued else blocker
                started, release, calls = self._block_compute(
                    engine, blocker.cache_key()
                )
                engine.async_submit([blocker], request_id=1 if queued else 2)
                self.assertTrue(started.wait(2))
                if queued:
                    engine.async_submit([target], request_id=2)
                entry = engine._embedding_cache.peek(target.cache_key())
                future = engine._async_tasks[entry].future
                self.assertEqual(future.running(), not queued)
                if outcome == "success":
                    engine._hash_key_cache.put(
                        target.cache_key(),
                        [torch.tensor([17])],
                        "old-generation",
                        greennet_passed=True,
                    )
                with self._rpc(engine, target, release) as (stub, registered, done):
                    rpc = (
                        stub.WaitGreenNetVerdict
                        if queued
                        else stub.RemoteMultimodalEmbedding
                    )
                    call = rpc.future(
                        MultimodalInputsPB(request_id=2),
                        timeout=0.3 if outcome == "deadline" else 2,
                    )
                    self.assertTrue(registered.wait(2))
                    if outcome == "cancel":
                        self.assertTrue(call.cancel())
                    elif outcome == "deadline":
                        with self.assertRaises(grpc.RpcError) as raised:
                            call.result(timeout=2)
                        self.assertEqual(
                            raised.exception.code(), grpc.StatusCode.DEADLINE_EXCEEDED
                        )
                    else:
                        call.result(timeout=2)
                    self.assertTrue(done.wait(2))
                    # One handler: a status reply proves the waiter exited while compute is blocked.
                    self.assertTrue(
                        stub.GetWorkerStatus(StatusVersionPB(), timeout=1).alive
                    )
                    self.assertFalse(release.is_set())
                    if outcome == "success":
                        self.assertFalse(future.cancelled())
                        release.set()
                        future.result(timeout=2)
                        self.assertEqual(calls.count(target.cache_key()), 1)
                        torch.testing.assert_close(entry.wait()[0], torch.ones(1, 1))
                    elif queued:
                        with self.assertRaises(FtRuntimeException):
                            entry.wait_ready(timeout=1)
                        self.assertTrue(future.cancelled())
                        self.assertNotIn(target.cache_key(), calls)
                    else:
                        self.assertTrue(future.running())

    def test_rpc_timeout_uses_remaining_deadline(self):
        for remaining, expected in ((0.125, 125), (None, 60000), (500, 60000), (-1, 0)):
            with self.subTest(remaining=remaining):
                self.assertEqual(
                    _rpc_timeout_ms(
                        SimpleNamespace(time_remaining=lambda: remaining), 60000
                    ),
                    expected,
                )


class DownloadCancellationTest(TestCase):
    def test_cancel_streaming_download_closes_response(self):
        from rtp_llm.multimodal import multimodal_util as util

        cancelled = threading.Event()
        response = mock.Mock(status_code=200)

        def chunks(**kwargs):
            yield b"a"
            cancelled.set()
            yield b"b"

        response.iter_content.side_effect = chunks
        with mock.patch.object(util, "request_get", return_value=response):
            with self.assertRaises(concurrent.futures.CancelledError):
                util._download_http_content(
                    "https://example.test/image", {}, None, cancellation_event=cancelled
                )
        response.close.assert_called_once()

    def test_download_deadline_reaches_request_timeout(self):
        from rtp_llm.multimodal import multimodal_util as util

        with mock.patch.object(util, "REQUEST_GET") as get:
            util.request_get(
                "https://example.test/image", {}, deadline=time.monotonic() + 0.5
            )
            self.assertGreater(get.call_args.kwargs["timeout"], 0)
            self.assertLessEqual(get.call_args.kwargs["timeout"], 0.5)


if __name__ == "__main__":
    main()
