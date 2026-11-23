import asyncio
import concurrent.futures
import json
import multiprocessing
import os
import pickle
import threading
import time
from functools import partial
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import MagicMock, patch

import torch

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.metrics.kmonitor_metric_reporter import GaugeMetrics
from rtp_llm.multimodal.greennet_hook import (
    GreenNetHandle,
    GreenNetProvider,
    GreenNetVerdict,
)
from rtp_llm.multimodal.mm_async_result import continue_after
from rtp_llm.multimodal.mm_preprocess import (
    MultiprocessPreprocessExecutor,
    PreprocessExecutor,
    PreprocessTaskResult,
    _report_result_path,
)
from rtp_llm.multimodal.mm_preprocess_result import MMPreprocessResult
from rtp_llm.multimodal.mm_work_item import MMWorkItem
from rtp_llm.multimodal.test import multimodal_process_engine_test as helpers
from rtp_llm.multimodal.test.multimodal_process_engine_test import (
    FakeMultiModalEmbeddingInterface,
)


class SourceModel(FakeMultiModalEmbeddingInterface):
    supports_inspection_sources = True

    def __init__(self):
        super().__init__()
        self.forward_urls = []
        self.two_forwards = threading.Event()

    @staticmethod
    def preprocess_input(mm_inputs, vit_config, **kwargs):
        urls = [item.url for item in mm_inputs]
        if kwargs.get("capture_inspection_sources"):
            return MMPreprocessResult(urls, [url.encode() for url in urls])
        return urls

    def get_preprocess_params(self):
        return {}

    def embedding(self, data, **kwargs):
        self.forward_urls.extend(data)
        if len(self.forward_urls) >= 2:
            self.two_forwards.set()
        return torch.tensor([[len(data[0])]], dtype=torch.float32), None


from rtp_llm.multimodal.test.remote_embedding_test_helpers import SourceProvider


class ControlledExecutor(PreprocessExecutor):
    """Deterministic out-of-order CPU completions without starting GPUs."""

    def __init__(self, slow_release, fast_done):
        self.pool = concurrent.futures.ThreadPoolExecutor(2)
        self.slow_release, self.fast_done = slow_release, fast_done

    def submit(self, item):
        def run():
            if "slow" in item.mm_inputs[0].url:
                if not self.slow_release.wait(5):
                    raise TimeoutError("test did not release slow task")
            return [value.url for value in item.mm_inputs]

        def ready(_):
            item.preprocess_ready = True
            item.preprocess_ready_event.set()

        item.future = self.pool.submit(run)
        item.future.add_done_callback(ready)

    def get_result(self, item):
        item.preprocess_result = item.future.result(timeout=item.remaining_timeout_s())
        item.future = None

    def shutdown(self):
        self.slow_release.set()
        self.pool.shutdown()


class WorkerSourceProvider(GreenNetProvider):
    """Real spawn child, controlled audit completion, no model/GPU mocks below IPC."""

    def __init__(self, release, records, cancelled):
        self.release, self.records, self.cancelled = release, records, cancelled

    async def submit_sources(self, request, mm_inputs, sources):
        if getattr(request, "on_image_upload", None) is not None:
            for offset, media in enumerate(mm_inputs):
                request.on_image_upload(
                    request.input_index_offset + offset,
                    f"https://oss.example/{media.url.rsplit('/', 1)[-1]}?sig=test",
                )
        self.records.put(
            (
                os.getpid(),
                request.id,
                request.input_index_offset,
                [m.url for m in mm_inputs],
                list(sources),
            )
        )
        provider = self

        class Handle(GreenNetHandle):
            async def wait_result(self):
                if request.id == "worker-crash":
                    os._exit(17)
                while not provider.release.is_set():
                    await asyncio.sleep(0.005)
                return GreenNetVerdict(
                    request.id != "reject",
                    2 if request.id == "reject" else 1,
                    str(request.input_index_offset),
                )

            def cancel(self):
                provider.cancelled.set()

        return Handle()


class WorkerInspectionTest(TestCase):
    _make_input = helpers.MMProcessEngineGreenNetTest._make_input
    _make_engine = helpers.MMProcessEngineGreenNetTest._make_engine

    def setUp(self):
        context = multiprocessing.get_context("spawn")
        self.release, self.cancelled = context.Event(), context.Event()
        self.records = context.Queue()
        self.factory = partial(
            WorkerSourceProvider, self.release, self.records, self.cancelled
        )
        config = VitConfig()
        config.mm_preprocess_max_workers = 1
        config.url_cache_item_num = config.mm_cache_item_num = 0
        self.executor = MultiprocessPreprocessExecutor(
            context,
            config,
            {},
            SourceModel.preprocess_input,
            inspection_factory=self.factory,
        )

    def tearDown(self):
        self.release.set()
        self.executor.shutdown()
        self.records.close()
        self.records.join_thread()

    def item(self, name, index=0, timeout=30):
        item = MMWorkItem(
            [self._make_input("fake://" + name)], deadline=time.monotonic() + timeout
        )
        item.capture_inspection_sources = True
        item.inspection_request = SimpleNamespace(id=name, input_index_offset=index)
        return item

    def test_upload_reaches_only_its_own_parent_request(self):
        self.release.set()
        urls = [{}, {}]
        items = [self.item("same-id", 3), self.item("same-id", 7)]
        for index, item in enumerate(items):
            item.inspection_request.on_image_upload = urls[index].__setitem__
            self.executor.submit(item)
        for item in items:
            self.executor.get_result(item)
            self.assertTrue(item.inspection_future.result(10).passed)
        url = "https://oss.example/same-id?sig=test"
        self.assertEqual(urls, [{3: url}, {7: url}])

    def test_rgb_and_next_task_ready_before_verdict_without_jpeg_ipc(self):
        first, second = self.item("first", 3), self.item("second", 7)
        self.executor.submit(first)
        payload = first.future.get(timeout=30)
        self.assertEqual(payload.result.inspection_sources, [])
        self.executor.get_result(first)
        self.assertFalse(first.inspection_future.done())
        self.assertIsNone(first.inspection_sources)
        self.executor.submit(second)
        self.executor.get_result(second)
        records = [self.records.get(timeout=5) for _ in range(2)]
        self.assertEqual(records[0][0], records[1][0])
        self.assertNotEqual(records[0][0], os.getpid())
        self.assertEqual([r[2] for r in records], [3, 7])
        self.assertEqual(
            [r[4] for r in records], [[b"fake://first"], [b"fake://second"]]
        )
        self.release.set()
        self.assertEqual(first.inspection_future.result(5).message, "3")
        self.assertEqual(second.inspection_future.result(5).message, "7")

    def test_verdict_may_arrive_before_rgb_is_consumed(self):
        self.release.set()
        item = self.item("early")
        self.executor.submit(item)
        self.assertTrue(item.inspection_future.result(30).passed)
        self.executor.get_result(item)
        self.assertEqual(item.preprocess_result, ["fake://early"])

    def test_cancellation_releases_worker_inspection(self):
        item = self.item("cancel")
        self.executor.submit(item)
        self.executor.get_result(item)
        self.records.get(timeout=5)
        item.inspection_future.cancel()
        self.assertTrue(self.cancelled.wait(3))
        self.release.set()
        later = self.item("later")
        self.executor.submit(later)
        self.executor.get_result(later)
        self.assertTrue(later.inspection_future.result(5).passed)

    def test_worker_death_fails_verdict_and_does_not_leak_capacity(self):
        item = self.item("worker-crash")
        self.executor.submit(item)
        with self.assertRaisesRegex(RuntimeError, "worker exited"):
            item.inspection_future.result(30)
        self.assertEqual(self.executor._inspection_channel._pending, {})
        self.release.set()
        recovered = self.item("after-crash")
        self.executor.submit(recovered)
        self.executor.get_result(recovered)
        self.assertTrue(recovered.inspection_future.result(5).passed)

    def test_shutdown_fails_unfinished_audit(self):
        item = self.item("unfinished")
        self.executor.submit(item)
        self.executor.get_result(item)
        self.executor.shutdown()
        with self.assertRaisesRegex(RuntimeError, "stopped"):
            item.inspection_future.result(1)

    def test_deadline_and_rejection_preserve_result_type_and_image_index(self):
        self.release.set()
        rejected = self.item("reject", 4)
        self.executor.submit(rejected)
        self.executor.get_result(rejected)
        verdict = rejected.inspection_future.result(5)
        self.assertFalse(verdict.passed)
        self.assertEqual((verdict.code, verdict.message), (2, "4"))
        self.release.clear()
        expired = self.item("expired", timeout=0.3)
        self.executor.submit(expired)
        self.executor.get_result(expired)
        with self.assertRaises(concurrent.futures.TimeoutError):
            expired.inspection_future.result(5)

    def test_embedding_and_cache_wait_for_second_reply(self):
        model = SourceModel()
        engine = self._make_engine(model)
        engine._inspection._greennet_provider = SourceProvider()
        engine.vit_config.disable_access_log = False
        engine._access_logger.logger = MagicMock()
        engine._access_logger.query_logger = MagicMock()
        engine.preprocess_executor.shutdown()
        engine.preprocess_executor = self.executor
        engine.mm_preprocess_batch_size = 1
        inputs = [
            self._make_input("fake://image-one"),
            self._make_input("fake://image-two"),
        ]
        try:
            with concurrent.futures.ThreadPoolExecutor(1) as caller:
                result = caller.submit(
                    engine._embed_with_greennet_sync, inputs, request_id=91
                )
                try:
                    self.assertTrue(model.two_forwards.wait(30))
                    self.assertFalse(result.done())
                    self.assertTrue(
                        all(
                            not engine._embedding_cache.peek(m.cache_key()).is_done
                            for m in inputs
                        )
                    )
                    records = [self.records.get(timeout=5) for _ in range(2)]
                    self.assertEqual([r[2] for r in records], [0, 1])
                finally:
                    self.release.set()
                result.result(10)
                engine._access_logger.logger.info.assert_called_once()
                record = json.loads(engine._access_logger.logger.info.call_args.args[0])
                self.assertEqual(record["id"], 91)
                self.assertEqual(
                    record["greennet_oss_urls"],
                    [
                        "https://oss.example/image-one?sig=test",
                        "https://oss.example/image-two?sig=test",
                    ],
                )
        finally:
            self.release.set()
            engine.stop()


class InspectionPipelineTest(TestCase):
    _make_engine = helpers.MMProcessEngineGreenNetTest._make_engine
    _make_input = helpers.MMProcessEngineGreenNetTest._make_input

    def _enable_access_log(self, engine):
        engine.vit_config.disable_access_log = False
        # Keep the real MMAccessLogger serializer, replacing only its sinks.
        engine._access_logger.logger = MagicMock()
        engine._access_logger.query_logger = MagicMock()

    def _assert_access_ids(self, engine, expected):
        for sink in (
            engine._access_logger.query_logger,
            engine._access_logger.logger,
        ):
            records = [json.loads(call.args[0]) for call in sink.info.call_args_list]
            self.assertEqual(len(records), len(expected))
            self.assertEqual(
                {record["id"]: record["external_request_id"] for record in records},
                expected,
            )

    def test_sync_access_ids_match_inspection_single_multi_and_cached(self):
        for count in (1, 2):
            for external_id in ("dash-sync", ""):
                with self.subTest(count=count, external_id=external_id):
                    engine = self._make_engine(SourceModel())
                    provider = SourceProvider()
                    engine._inspection._greennet_provider = provider
                    self._enable_access_log(engine)
                    media = [
                        self._make_input(f"https://example.com/sync-{i}.jpg")
                        for i in range(count)
                    ]
                    try:
                        engine._embed_with_greennet_sync(
                            media, request_id=71, external_request_id=external_id
                        )
                        expected_id = external_id or "71"
                        self.assertTrue(provider.request_ids)
                        self.assertEqual(set(provider.request_ids), {expected_id})
                        self._assert_access_ids(engine, {71: expected_id})
                        # Reusing tensors must not reuse the previous caller's ID.
                        engine._embed_with_greennet_sync(
                            media, request_id=72, external_request_id="dash-cached"
                        )
                        self._assert_access_ids(
                            engine, {71: expected_id, 72: "dash-cached"}
                        )
                    finally:
                        engine.stop()

    def test_concurrent_async_access_ids_match_inspection(self):
        engine = self._make_engine(SourceModel())
        provider = SourceProvider()
        engine._inspection._greennet_provider = provider
        self._enable_access_log(engine)
        try:
            with concurrent.futures.ThreadPoolExecutor(2) as executor:
                futures = [
                    executor.submit(
                        engine.get_embedding_result,
                        [self._make_input(f"https://example.com/async-{rid}.jpg")],
                        request_id=rid,
                        external_request_id=external_id,
                    )
                    for rid, external_id in ((81, "dash-async"), (82, ""))
                ]
                for future in futures:
                    self.assertEqual(len(future.result(timeout=10)), 1)
            self._assert_access_ids(engine, {81: "dash-async", 82: "82"})
            self.assertCountEqual(provider.request_ids, ["dash-async", "82"])
        finally:
            engine.stop()

    def test_embedding_error_keeps_access_ids_in_sync_and_async(self):
        for asynchronous in (False, True):
            with self.subTest(asynchronous=asynchronous):
                engine = self._make_engine(SourceModel())
                engine._inspection._greennet_provider = SourceProvider()
                self._enable_access_log(engine)
                media = [self._make_input("https://example.com/failure.jpg")]
                try:
                    method = (
                        engine.get_embedding_result
                        if asynchronous
                        else engine._embed_with_greennet_sync
                    )
                    with patch.object(
                        engine,
                        "_compute_embeddings",
                        side_effect=ValueError("embedding failed"),
                    ):
                        with self.assertRaisesRegex(ValueError, "embedding failed"):
                            method(
                                media,
                                request_id=91,
                                external_request_id="dash-failed",
                            )
                    self._assert_access_ids(engine, {91: "dash-failed"})
                    record = json.loads(
                        engine._access_logger.logger.info.call_args.args[0]
                    )
                    self.assertEqual(record["exception"], "embedding failed")
                finally:
                    engine.stop()

    def test_typed_preprocessing_errors_survive_inspection_wait(self):
        for code in (
            ExceptionType.MM_WRONG_FORMAT_ERROR,
            ExceptionType.INVALID_PARAMS,
            ExceptionType.MM_DOWNLOAD_FAILED,
            ExceptionType.MM_IMAGE_PIXEL_LIMIT_EXCEEDED,
        ):
            with self.subTest(code=code):
                error = FtRuntimeException(code, "media preprocessing failed")
                with patch.object(SourceModel, "preprocess_input", side_effect=error):
                    engine = self._make_engine(SourceModel())
                    provider = SourceProvider()
                    engine._inspection._greennet_provider = provider
                    try:
                        with self.assertRaises(FtRuntimeException) as raised:
                            engine.wait_greennet_verdict(
                                [self._make_input("http://10.0.0.1/image.png")],
                                timeout_ms=5000,
                                request_id=92,
                            )
                        self.assertIs(raised.exception, error)
                        self.assertEqual(provider.calls, [])
                        self.assertEqual(engine.mm_part.forward_urls, [])
                    finally:
                        engine.stop()

    def test_untyped_preprocessing_failure_keeps_process_error_verdict(self):
        with patch.object(
            SourceModel, "preprocess_input", side_effect=RuntimeError("backend failed")
        ):
            engine = self._make_engine(SourceModel())
            engine._inspection._greennet_provider = SourceProvider()
            try:
                verdict = engine.wait_greennet_verdict(
                    [self._make_input("fake://broken")], timeout_ms=5000
                )
                self.assertFalse(verdict.passed)
                self.assertEqual(verdict.code, 11)
            finally:
                engine.stop()

    def test_cancelled_queued_continuation_finishes_terminal_future(self):
        source, queued = concurrent.futures.Future(), concurrent.futures.Future()
        cancelled = threading.Event()
        executor = SimpleNamespace(submit=lambda fn: queued)
        terminal = continue_after(
            source, executor, lambda value: value, cancelled.set, lambda error: None
        )
        source.set_result(True)
        queued.cancel()
        self.assertTrue(terminal.cancelled())
        self.assertTrue(cancelled.is_set())

    def test_continuation_reports_original_error_even_if_error_hook_fails(self):
        source = concurrent.futures.Future()
        error = ValueError("inspection failure")
        source.set_exception(error)
        with concurrent.futures.ThreadPoolExecutor(1) as executor:

            def error_hook(_):
                raise RuntimeError("report failure")

            terminal = continue_after(
                source, executor, lambda value: value, lambda: None, error_hook
            )
            with self.assertRaisesRegex(ValueError, "inspection failure"):
                terminal.result(timeout=2)

    def test_real_spawn_worker_returns_optional_source_bytes(self):
        config = VitConfig()
        config.mm_preprocess_max_workers = 1
        config.url_cache_item_num = 0
        config.mm_cache_item_num = 0
        executor = MultiprocessPreprocessExecutor(
            multiprocessing.get_context("spawn"),
            config,
            {},
            SourceModel.preprocess_input,
        )
        try:
            item = MMWorkItem([self._make_input("https://example.com/spawn.jpg")])
            item.capture_inspection_sources = True
            executor.submit(item)
            executor.get_result(item)
            self.assertEqual(item.preprocess_result, [item.mm_inputs[0].url])
            self.assertEqual(item.inspection_sources, [item.mm_inputs[0].url.encode()])
            self.assertIsNone(item.future)
        finally:
            executor.shutdown()

    def test_dispatch_and_return_measurements_exclude_consumer_wait(self):
        payload = PreprocessTaskResult(None, 50, 10, [], 10.3, 10.35)
        item = SimpleNamespace(
            preprocess_submitted_at=10.0, preprocess_received_at=10.4
        )
        with patch("rtp_llm.metrics.kmonitor.report") as report:
            _report_result_path(item, payload)
        values = {call.args[0]: call.args[1] for call in report.call_args_list}
        self.assertAlmostEqual(
            values[GaugeMetrics.VIT_PREPROCESS_DISPATCH_RT_METRIC], 300
        )
        self.assertAlmostEqual(values[GaugeMetrics.VIT_PREPROCESS_RETURN_RT_METRIC], 50)

    def test_preprocess_benchmark_logger_failure_does_not_change_result(self):
        payload = PreprocessTaskResult(None, 50, 10, [], 10.3, 10.35)
        item = SimpleNamespace(
            preprocess_submitted_at=10.0, preprocess_received_at=10.4
        )
        with patch.dict(os.environ, {"MM_EMB_BENCHMARK_TIMING": "1"}), patch(
            "rtp_llm.multimodal.mm_preprocess.logging.info",
            side_effect=RuntimeError("sink"),
        ), patch("rtp_llm.metrics.kmonitor.report"):
            _report_result_path(item, payload)

    def test_sources_are_optional_and_survive_worker_pickle(self):
        model = SourceModel()
        media = [self._make_input("https://example.com/a.jpg")]
        plain = model.preprocess_input(media, None)
        captured = model.preprocess_input(media, None, capture_inspection_sources=True)
        restored = pickle.loads(pickle.dumps(captured))
        self.assertEqual(plain, restored.data)
        self.assertEqual(restored.inspection_sources, [media[0].url.encode()])

    def test_source_handoff_and_rejection_gate(self):
        for passed in (True, False):
            engine = self._make_engine(SourceModel())
            provider = SourceProvider(passed)
            engine._inspection._greennet_provider = provider
            media = self._make_input("https://example.com/original.jpg")
            try:
                if passed:
                    result = engine.get_embedding_result([media], request_id=41)
                    self.assertEqual(len(result), 1)
                else:
                    with self.assertRaises(Exception):
                        engine.get_embedding_result([media], request_id=42)
                    self.assertIsNone(engine._embedding_cache.peek(media.cache_key()))
                self.assertEqual(provider.calls, [([media.url], [media.url.encode()])])
                self.assertEqual(engine.mm_part.forward_urls, [media.url])
            finally:
                engine.stop()

    def test_deferred_compute_precedes_verdict_and_strict_publication(self):
        for passed in (False, True):
            with self.subTest(passed=passed):
                engine = self._make_engine(SourceModel())
                provider = SourceProvider(passed, hold=True)
                engine._inspection._greennet_provider = provider
                media = self._make_input("fake://deferred")
                key = media.cache_key()
                tickets = engine._inspection.tickets
                ticket = tickets.open([{"key": key}])
                with concurrent.futures.ThreadPoolExecutor(2) as pool:
                    try:
                        hashes = engine.get_embedding_result(
                            [media],
                            hashes_only=True,
                            inspection_ticket=ticket,
                            timeout_ms=3000,
                        )
                        embeddings = engine.get_embedding_result(
                            [media], inspection_ticket=ticket, timeout_ms=3000
                        )
                        self.assertEqual(len(embeddings[0].embeddings), 1)
                        self.assertTrue(
                            torch.equal(
                                hashes[0].feature_hashes[0],
                                embeddings[0].feature_hashes[0],
                            )
                        )
                        entry = engine._embedding_cache.peek(key)
                        self.assertFalse(entry.is_done)
                        self.assertFalse(engine._hash_key_cache.contains(key))
                        strict = pool.submit(
                            engine.get_embedding_result, [media], request_id=17
                        )
                        query = pool.submit(tickets.wait, ticket, time.monotonic() + 5)
                        with self.assertRaises(concurrent.futures.TimeoutError):
                            query.result(0.05)
                        self.assertFalse(strict.done())
                        provider.release.set()
                        self.assertEqual(query.result(3).passed, passed)
                        if passed:
                            self.assertEqual(len(strict.result(3)), 1)
                        else:
                            with self.assertRaises(FtRuntimeException) as raised:
                                strict.result(3)
                            self.assertEqual(
                                raised.exception.exception_type,
                                ExceptionType.UNSAFE_INPUT_CONTENT,
                            )
                            self.assertFalse(engine._hash_key_cache.contains(key))
                        self.assertEqual(len(provider.calls), 1)
                    finally:
                        provider.release.set()
                        engine.stop()

    def test_hash_and_embedding_rpc_return_while_same_audit_is_pending(self):
        import grpc

        from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
            GreenNetQueryPB,
            MultimodalHashRequestPB,
            MultimodalInputsPB,
        )
        from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc import (
            MultimodalRpcServiceStub,
            add_MultimodalRpcServiceServicer_to_server,
        )
        from rtp_llm.multimodal.multimodal_util import trans_mm_input
        from rtp_llm.server.vit_rpc_server import MultimodalRpcServer

        engine = self._make_engine(SourceModel())
        engine._embedding_cache.resize(0, 0)
        engine._hash_key_cache.resize(0)
        provider = SourceProvider(hold=True)
        engine._inspection._greennet_provider = provider
        with concurrent.futures.ThreadPoolExecutor(4) as pool:
            server = grpc.server(pool)
            add_MultimodalRpcServiceServicer_to_server(
                MultimodalRpcServer(engine), server
            )
            port = server.add_insecure_port("127.0.0.1:0")
            server.start()
            try:
                with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
                    stub = MultimodalRpcServiceStub(channel)
                    wire = MultimodalInputsPB(request_id=721)
                    url = "https://example.com/deferred"
                    wire.multimodal_inputs.add(multimodal_url=url, multimodal_type=1)
                    key = trans_mm_input(wire)[0].cache_key()
                    probe = stub.GetMultimodalHashes(
                        MultimodalHashRequestPB(
                            keys=[key], defer_inspection=True, request_id=721
                        ),
                        timeout=3,
                    )
                    self.assertTrue(probe.inspection_ticket)
                    wire.inspection_ticket = probe.inspection_ticket
                    filled = stub.GetMultimodalHashes(
                        MultimodalHashRequestPB(keys=[key], inputs=wire), timeout=3
                    )
                    self.assertTrue(filled.entries[0].hash_hit)
                    self.assertFalse(filled.entries[0].greennet_passed)
                    self.assertGreater(
                        stub.RemoteMultimodalEmbedding(wire, timeout=3).ByteSize(), 0
                    )
                    self.assertIsNone(engine._embedding_cache.peek(key))
                    query = stub.GetGreenNetVerdict.future(
                        GreenNetQueryPB(
                            inspection_ticket=wire.inspection_ticket, request_id=721
                        ),
                        timeout=3,
                    )
                    with self.assertRaises(grpc.FutureTimeoutError):
                        query.result(timeout=0.05)
                    provider.release.set()
                    query.result(timeout=2)
                    self.assertEqual(len(provider.calls), 1)
                    self.assertEqual(engine.mm_part.forward_urls, [url])
            finally:
                provider.release.set()
                server.stop(0).wait()
                engine.stop()

    def test_ticket_is_bound_to_worker_and_every_media_generation(self):
        from rtp_llm.multimodal.mm_inspection_tickets import InspectionTickets

        store = InspectionTickets(ttl=30, capacity=2)
        ticket = store.open(
            [
                {"key": "a", "entry_generation": "1", "greennet_passed": True},
                {"key": "b"},
            ]
        )
        with self.assertRaises(FtRuntimeException):
            store.open([{"key": "c"}])
        with self.assertRaises(FtRuntimeException):
            store.wait(ticket, time.monotonic() + 1)
        with self.assertRaises(FtRuntimeException):
            InspectionTickets(30).wait(ticket, time.monotonic() + 1)
        decision = concurrent.futures.Future()
        with self.assertRaises(ValueError):
            store.bind(ticket, "c", "2", decision)
        store.bind(ticket, "b", "2", decision)
        cancelled = threading.Event()
        cancelled.set()
        with self.assertRaises(concurrent.futures.CancelledError):
            store.wait(ticket, time.monotonic() + 1, cancelled)
        self.assertFalse(decision.cancelled())
        denied = concurrent.futures.Future()
        denied.set_result(GreenNetVerdict(False, 2))
        store.bind(ticket, "a", "new-generation", denied)
        decision.set_result(GreenNetVerdict(True))
        self.assertEqual(store.wait(ticket, time.monotonic() + 1).code, 2)
        self.assertTrue(store.open([{"key": "c"}]))
        expired = InspectionTickets(ttl=0)
        with self.assertRaises(FtRuntimeException):
            expired.wait(expired.open([{"key": "c"}]), time.monotonic() + 1)

    def test_ticket_sweep_expires_completed_receipts_without_changing_retries(self):
        from rtp_llm.multimodal.mm_inspection_tickets import InspectionTickets

        store = InspectionTickets(ttl=5, capacity=2)
        entry = {"key": "a", "entry_generation": "1", "greennet_passed": True}
        with patch("rtp_llm.multimodal.mm_inspection_tickets.time.monotonic") as clock:
            clock.return_value = 100
            first = store.open([entry], request_id=1)
            approved = store.wait(first, 200, request_id=1)
            clock.return_value = 102
            second = store.open([entry], request_id=2)
            other = store.wait(second, 200, request_id=2)
            clock.return_value = 104
            self.assertIs(store.wait(first, 200, request_id=1), approved)
            with self.assertRaises(FtRuntimeException):
                store.wait(first, 200, request_id=2)

            clock.return_value = 105
            store.open([entry, {"key": "b"}])
            self.assertEqual(list(store._completed), [second])
            self.assertIs(store.wait(second, 200, request_id=2), other)
            with self.assertRaises(FtRuntimeException):
                store.wait(first, 200, request_id=1)
            with self.assertRaises(FtRuntimeException):
                store.open([entry])

            clock.return_value = 110
            store.open([entry, {"key": "b"}])
            self.assertFalse(store._completed)
            self.assertEqual(store._media_count, 2)

            clock.return_value = 120
            receipts = [store.open([entry], request_id=rid) for rid in (1, 2)]
            for rid, receipt in enumerate(receipts, 1):
                store.wait(receipt, 200, request_id=rid)
            latest = store.open([entry], request_id=3)
            store.wait(latest, 200, request_id=3)
            self.assertEqual(list(store._completed), [receipts[-1], latest])
            self.assertEqual(store._media_count, 0)

    def test_refinished_ticket_follows_expiration_order(self):
        from rtp_llm.multimodal.mm_inspection_tickets import InspectionTickets

        store = InspectionTickets(ttl=5, capacity=2)
        verdict = GreenNetVerdict(passed=False)
        with patch("rtp_llm.multimodal.mm_inspection_tickets.time.monotonic") as clock:
            for stamp, key in ((100, "first"), (101, "second"), (102, "first")):
                clock.return_value = stamp
                with store._lock:
                    store._finish(key, 1, verdict)
            clock.return_value = 106
            store.open([{"key": "pending"}])
            self.assertEqual(list(store._completed), ["first"])
            self.assertIs(store.wait("first", 200, request_id=1), verdict)

    def test_preprocess_callbacks_signal_shared_or_independent_event(self):
        class ObservedEvent(threading.Event):
            def wait(self, timeout=None):
                waiting.set()
                return super().wait(timeout)

        for result in (
            None,
            RuntimeError("failed"),
            concurrent.futures.CancelledError(),
        ):
            with self.subTest(result=result):
                waiting = threading.Event()
                ready = ObservedEvent()
                media = self._make_input("fake://ready")
                first = MMWorkItem(
                    [media], mm_timeout_ms=3000, preprocess_ready_event=ready
                )
                second = MMWorkItem([media], preprocess_ready_event=ready)
                independent = MMWorkItem([media])
                self.assertIs(
                    first.preprocess_ready_event, second.preprocess_ready_event
                )
                self.assertIsNot(independent.preprocess_ready_event, ready)

                executor = object.__new__(MultiprocessPreprocessExecutor)
                executor.pool = MagicMock()
                executor._finish_preprocess_task = MagicMock()
                first.future = executor._apply_async(first, task_id=1)
                callbacks = executor.pool.apply_async.call_args.kwargs
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as waiter:
                    pending = waiter.submit(next, executor.iter_ready([first]))
                    self.assertTrue(waiting.wait(timeout=1))
                    callback = "callback" if result is None else "error_callback"
                    callbacks[callback](result)
                    self.assertIs(pending.result(timeout=1), first)
                self.assertTrue(first.preprocess_ready)
                self.assertFalse(independent.preprocess_ready_event.is_set())

    def test_unchecked_local_hit_starts_one_audit_without_reencoding(self):
        engine = self._make_engine(SourceModel())
        provider = SourceProvider(hold=True)
        engine._inspection._greennet_provider = provider

        async def reinspect(request, inputs):
            return await provider.submit_sources(
                request, inputs, [media.url.encode() for media in inputs]
            )

        provider.reinspect_cached = reinspect
        media = self._make_input("fake://unchecked")
        tickets = engine._inspection.tickets
        ticket = tickets.open([{"key": media.cache_key()}])
        try:
            engine.get_embedding_result([media], input_inspection_enabled=False)
            for hashes_only in (True, False):
                engine.get_embedding_result(
                    [media],
                    inspection_ticket=ticket,
                    hashes_only=hashes_only,
                    timeout_ms=3000,
                )
            with self.assertRaises(TimeoutError):
                tickets.wait(ticket, time.monotonic() + 0.05)
            provider.release.set()
            self.assertTrue(tickets.wait(ticket, time.monotonic() + 3).passed)
            self.assertEqual(len(provider.calls), 1)
            self.assertEqual(engine.mm_part.forward_urls, [media.url])
        finally:
            provider.release.set()
            engine.stop()

    def test_source_audit_deadline_preserves_threaded_timeout_contract(self):
        engine = self._make_engine(SourceModel())
        engine._inspection._greennet_provider = SourceProvider(hold=True)
        media = self._make_input("fake://audit-deadline")
        key = media.cache_key()
        _, entry = engine._embedding_cache.try_acquire(key)
        try:
            engine._run_async_compute([media], key, entry, 71, time.monotonic() + 0.1)
            with self.assertRaises(concurrent.futures.TimeoutError):
                entry.wait(timeout=2)
            self.assertIsNone(engine._embedding_cache.peek(key))
        finally:
            engine.stop()

    def test_waiting_for_audit_releases_thread_but_not_admission(self):
        class ReinspectionProvider(SourceProvider):
            async def preprocess_and_submit(self, request, inputs):
                # Once async work finishes, dev re-inspects cached results for
                # the consuming request. The first inspection must still use
                # the original worker sources.
                if request.id not in self.request_ids:
                    raise AssertionError("initial inspection skipped worker sources")
                handle = await self.submit_sources(
                    request, inputs, [item.url.encode() for item in inputs]
                )
                handle.rewritten_inputs = inputs
                return handle

        model, provider = SourceModel(), ReinspectionProvider(hold=True)
        engine = self._make_engine(model)
        engine._inspection._greennet_provider = provider
        engine._async_compute_executor.shutdown()
        engine._async_compute_executor = concurrent.futures.ThreadPoolExecutor(1)
        media = [self._make_input("fake://one"), self._make_input("fake://two")]
        try:
            engine.async_submit(media, request_id=51)
            self.assertTrue(
                model.two_forwards.wait(3), "audit blocked the only compute thread"
            )
            self.assertEqual(engine._async_admitted, 2)
            self.assertTrue(
                all(
                    not engine._embedding_cache.peek(m.cache_key()).is_done
                    for m in media
                )
            )
            provider.release.set()
            self.assertEqual(len(engine.get_embedding_result(media, request_id=51)), 2)
        finally:
            provider.release.set()
            engine.stop()

    def test_duplicate_images_share_original_source_and_matching_cached_verdict(self):
        model, provider = SourceModel(), SourceProvider()
        engine = self._make_engine(model)
        engine._inspection._greennet_provider = provider
        engine.mm_preprocess_batch_size = 1
        media = self._make_input("fake://original-source")
        try:
            first = engine._embed_with_greennet_sync([media, media], request_id=61)
            self.assertEqual(len(first.embeddings), 2)
            self.assertEqual(model.forward_urls, [media.url])
            self.assertEqual(
                provider.calls, [([media.url, media.url], [media.url.encode()] * 2)]
            )
            # A cache hit must not download again and use a changed resource
            # as proof for the previous tensor. Reuse the matching verdict.
            with patch.object(
                engine.preprocess_executor,
                "preprocess_func",
                side_effect=AssertionError("redownload"),
            ):
                second = engine._embed_with_greennet_sync([media, media], request_id=62)
            self.assertEqual(len(second.embeddings), 2)
            self.assertEqual(len(provider.calls), 1)
            self.assertEqual(model.forward_urls, [media.url])
        finally:
            engine.stop()

    def test_fast_image_reaches_gpu_before_slow_image_in_same_request(self):
        model = SourceModel()
        model.supports_inspection_sources = False
        engine = self._make_engine(model)
        engine._inspection._greennet_provider = GreenNetProvider()
        release, fast = threading.Event(), threading.Event()
        original_embedding = model.embedding

        def embedding(data, **kwargs):
            if "fast" in data[0]:
                fast.set()
            return original_embedding(data, **kwargs)

        model.embedding = embedding
        engine.preprocess_executor.shutdown()
        engine.preprocess_executor = ControlledExecutor(release, fast)
        engine.mm_preprocess_batch_size = 1
        media = [self._make_input("fake://slow-image"), self._make_input("fake://fast")]
        try:
            with concurrent.futures.ThreadPoolExecutor(1) as caller:
                result = caller.submit(engine.mm_embedding_impl, media)
                try:
                    self.assertTrue(
                        fast.wait(2), "ready image was held behind slow preprocessing"
                    )
                finally:
                    release.set()
                outputs = result.result(timeout=5).embeddings
            self.assertEqual(
                [tensor.item() for tensor in outputs], [len(m.url) for m in media]
            )
        finally:
            release.set()
            engine.stop()


class InspectionSourceWorkerTest(TestCase):
    def test_remote_cache_sends_neither_original_bytes_nor_digest_to_parent(self):
        import base64
        import io

        from PIL import Image

        from rtp_llm.multimodal.multimodal_mixins.deepseek_v41.deepseek_v41_processor import (
            V41ImageProcessorConfig,
        )
        from rtp_llm.multimodal.multimodal_mixins.deepseek_v41.deepseek_v41_vision import (
            DeepSeekV41VisionEmbedding,
        )

        image = io.BytesIO()
        Image.new("RGB", (42, 42), "red").save(image, format="PNG")
        media = image.getvalue()
        context = multiprocessing.get_context("spawn")
        release, cancelled = context.Event(), context.Event()
        records = context.Queue()
        config = VitConfig()
        config.mm_preprocess_max_workers = 1
        config.mm_remote_cache_enable = True
        config.url_cache_item_num = config.mm_cache_item_num = 0
        executor = MultiprocessPreprocessExecutor(
            context,
            config,
            dict(processor_config=V41ImageProcessorConfig(vision_min_pixels=1764)),
            DeepSeekV41VisionEmbedding.preprocess_input,
            inspection_factory=partial(
                WorkerSourceProvider, release, records, cancelled
            ),
        )
        item = MMWorkItem(
            [
                helpers.MMProcessEngineGreenNetTest._make_input(
                    self, "data:image/png;base64," + base64.b64encode(media).decode()
                )
            ],
            deadline=time.monotonic() + 30,
        )
        item.capture_inspection_sources = True
        item.inspection_request = SimpleNamespace(id="identity", input_index_offset=0)
        try:
            executor.submit(item)
            wire_result = item.future.get(timeout=30).result
            self.assertEqual(wire_result.inspection_sources, [])
            self.assertIsNone(wire_result.source_identity)
            executor.get_result(item)
            self.assertEqual(item.source_identity, wire_result.source_identity)
            self.assertIsNone(item.inspection_sources)
            self.assertFalse(item.inspection_future.done())
            worker_record = records.get(timeout=5)
            self.assertNotEqual(worker_record[0], os.getpid())
            self.assertEqual(worker_record[4], [media])
            release.set()
            self.assertTrue(item.inspection_future.result(5).passed)
        finally:
            release.set()
            executor.shutdown()
            records.close()
            records.join_thread()


if __name__ == "__main__":
    main()
