import asyncio
import concurrent.futures
import multiprocessing
import os
import pickle
import threading
import time
from functools import partial
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch

import torch

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


class SourceProvider(GreenNetProvider):
    def __init__(self, passed=True, hold=False):
        self.passed = passed
        self.release = threading.Event()
        if not hold:
            self.release.set()
        self.calls = []

    def is_enabled(self):
        return True

    async def preprocess_and_submit(self, *args):
        raise AssertionError("worker-source path must not use URL rewriting")

    async def submit_sources(self, request, mm_inputs, sources):
        self.calls.append(([item.url for item in mm_inputs], list(sources)))
        provider = self

        class Handle(GreenNetHandle):
            async def wait_result(self):
                while not provider.release.is_set():
                    await asyncio.sleep(0.001)
                return GreenNetVerdict(provider.passed, 1 if provider.passed else 2)

            def cancel(self):
                pass

        return Handle()


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
        finally:
            self.release.set()
            engine.stop()


class InspectionPipelineTest(TestCase):
    _make_engine = helpers.MMProcessEngineGreenNetTest._make_engine
    _make_input = helpers.MMProcessEngineGreenNetTest._make_input

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
        model, provider = SourceModel(), SourceProvider(hold=True)
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


if __name__ == "__main__":
    main()
