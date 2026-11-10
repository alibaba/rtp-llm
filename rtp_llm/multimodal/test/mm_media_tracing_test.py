"""Trace production CPU/executor/scheduler paths without model weights or CUDA.

Only the model forward and native feature hash are test doubles. Media loading,
cache admission, multiprocessing IPC and asynchronous inspection are real.
"""

import asyncio
import base64
import gc
import multiprocessing
import os
import threading
import time
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from rtp_llm.config.exceptions import ExceptionType
from rtp_llm.config.py_config_modules import ProfilingDebugLoggingConfig, VitConfig
from rtp_llm.multimodal.greennet_hook import (
    GreenNetHandle,
    GreenNetProvider,
    GreenNetVerdict,
)
from rtp_llm.multimodal.mm_batch_pipeline import PrefetchedMMBatch
from rtp_llm.multimodal.mm_preprocess import (
    MultiprocessPreprocessExecutor,
    PreprocessExecutor,
)
from rtp_llm.multimodal.mm_preprocess_result import MMPreprocessResult
from rtp_llm.multimodal.mm_process_engine import MMProcessEngine
from rtp_llm.multimodal.mm_work_item import MMWorkItem
from rtp_llm.multimodal.multimodal_mixins.multimodal_common import (
    MultiModalEmbeddingInterface,
)
from rtp_llm.multimodal.multimodal_util import get_bytes_io_from_url
from rtp_llm.ops import MMPreprocessConfig, MultimodalInput
from rtp_llm.telemetry import attributes as trace_attrs
from rtp_llm.telemetry import tracing, vit_media
from rtp_llm.utils.base_model_datatypes import MMUrlType


class CPUModel(MultiModalEmbeddingInterface):
    supports_inspection_sources = True

    def __init__(self):
        self.forward_started = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.forward_count = 0

    @staticmethod
    def preprocess_input(mm_inputs, vit_config, **kwargs):
        data = []
        for index, item in enumerate(mm_inputs):
            with vit_media.input_scope(index):
                data.append(get_bytes_io_from_url(item.url).getvalue())
        if kwargs.get("capture_inspection_sources"):
            return MMPreprocessResult(data, list(data))
        return data

    def embedding(self, data, **kwargs):
        self.forward_started.set()
        if not self.release.wait(10):
            raise TimeoutError("test forward was not released")
        self.forward_count += 1
        return torch.tensor([[len(data[0])]], dtype=torch.float32), None


class InspectionProvider(GreenNetProvider):
    def is_enabled(self):
        return True

    async def submit_sources(self, request, mm_inputs, sources):
        class Handle(GreenNetHandle):
            async def wait_result(self):
                await asyncio.sleep(0)
                return GreenNetVerdict(True, 1)

            def cancel(self):
                pass

        return Handle()


class MediaPipelineTraceTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(tracing.reset_telemetry_for_test())
        self.exporter = InMemorySpanExporter()
        self.assertTrue(tracing.init_telemetry_for_test(self.exporter, role="vit"))
        self.root = tracing.start_server_span("rpc", {})
        self.addCleanup(tracing.reset_telemetry_for_test)
        self.addCleanup(self.root.finish)
        hashes = patch(
            "rtp_llm.multimodal.mm_work_item._feature_hashes_from_result",
            return_value=[torch.tensor([7], dtype=torch.int32)],
        )
        hashes.start()
        self.addCleanup(hashes.stop)

    def make_input(self, data=b"image"):
        url = "data:image/png;base64," + base64.b64encode(data).decode()
        return MultimodalInput(
            url,
            MMUrlType.IMAGE,
            torch.empty(0),
            MMPreprocessConfig(-1, -1, -1, -1, -1, -1, -1, [], 30000),
        )

    def engine(self, **options):
        config = VitConfig()
        config.use_local_preprocess = True
        config.mm_cache_gpu_max_bytes = 0
        config.mm_cache_cpu_max_bytes = config.mm_hash_key_cache_max_bytes = 1024 * 1024
        config.mm_cache_async_publish = False
        config.disable_access_log = True
        for key, value in options.items():
            setattr(config, key, value)
        model = CPUModel()
        model_config = SimpleNamespace(
            mm_model_config=SimpleNamespace(mm_position_ids_style=0),
            mm_related_params=SimpleNamespace(preprocess_batch_size=1),
        )
        engine = MMProcessEngine(
            model, model_config, config, ProfilingDebugLoggingConfig(), device="cpu"
        )
        self.addCleanup(engine.stop)
        self.addCleanup(model.release.set)
        return engine

    def spans(self):
        tracing._provider.force_flush()
        return self.exporter.get_finished_spans()

    def test_engine_two_inputs_raw_urls_and_cache_reuse(self):
        engine = self.engine()
        inputs = [self.make_input(b"first"), self.make_input(b"second")]
        self.assertEqual(len(engine.get_embedding_result(inputs, request_id=11)), 2)
        # Drain callbacks before examining asynchronous producer spans.
        engine._async_compute_executor.shutdown(wait=True)
        first = self.spans()
        media = [s for s in first if s.name == vit_media.PREFIX + "media"]
        self.assertEqual(
            {s.attributes[vit_media.MEDIA + "index"] for s in media}, {0, 1}
        )
        self.assertEqual(
            {s.attributes[vit_media.MEDIA + "url"] for s in media},
            {i.url for i in inputs},
        )
        for parent in media:
            children = [s for s in first if s.parent.span_id == parent.context.span_id]
            names = {s.name.removeprefix(vit_media.PREFIX) for s in children}
            self.assertTrue(
                {
                    "preprocess.queue",
                    "preprocess",
                    "embedding.queue",
                    "embedding",
                }.issubset(names),
                names,
            )
            self.assertTrue("download" in names or "download.cache" in names)
            self.assertFalse(parent.links, "producer must not link to itself")
        self.assertEqual(len(engine.get_embedding_result(inputs, request_id=12)), 2)
        later = self.spans()[len(first) :]
        self.assertEqual(engine.mm_part.forward_count, 2)
        self.assertFalse(any(s.name.endswith(".embedding") for s in later))
        self.assertEqual(
            len([s for s in later if s.name.endswith(".media") and s.links]), 2
        )

    def test_async_submit_rpc_finishes_before_media(self):
        engine = self.engine()
        engine.mm_part.release.clear()
        inp = self.make_input(b"async")
        engine.async_submit([inp], request_id=21)
        self.assertTrue(engine.mm_part.forward_started.wait(5))
        self.root.finish()
        self.assertFalse(any(s.name.endswith(".media") for s in self.spans()))
        engine.mm_part.release.set()
        engine._async_compute_executor.shutdown(wait=True)
        spans = self.spans()
        rpc = next(s for s in spans if s.name == "rpc")
        media = next(s for s in spans if s.name.endswith(".media"))
        self.assertEqual(media.parent.span_id, rpc.context.span_id)
        self.assertEqual(media.context.trace_id, rpc.context.trace_id)
        self.assertGreater(media.end_time, rpc.end_time)

    def test_resident_cache_does_not_retain_finished_sdk_spans(self):
        for async_publish in (False, True):
            with self.subTest(async_publish=async_publish):
                engine = self.engine(mm_cache_async_publish=async_publish)
                inp = self.make_input(b"retained-cache")
                sdk_span_refs = []
                new_media = vit_media._new_media

                def capture(*args, **kwargs):
                    media = new_media(*args, **kwargs)
                    sdk_span_refs.extend(weakref.ref(member.span) for member in media)
                    return media

                with patch.object(vit_media, "_new_media", side_effect=capture):
                    result = engine.get_embedding_result([inp], request_id=81)
                engine._async_compute_executor.shutdown(wait=True)
                engine._embedding_cache.flush()
                self.assertEqual(len(result), 1)
                entry = engine._embedding_cache.peek(inp.cache_key())
                self.assertIsNotNone(entry)
                producer_context = entry._vit_media_context
                retained_handle = entry._pending_before_publish.__self__.media_traces[0]
                self.assertEqual(
                    retained_handle.span.get_span_context(), producer_context
                )
                self.spans()
                gc.collect()
                self.assertEqual(len(sdk_span_refs), 1)
                self.assertIsNone(sdk_span_refs[0]())
                self.assertFalse(retained_handle.span.is_recording())

                # Reuse the cached tensor from a different request trace. Its
                # media keeps that request's trace ID and links to the producer.
                other = tracing.start_server_span("cached_rpc", {})
                try:
                    engine.get_embedding_result([inp], request_id=82)
                finally:
                    other.finish()
                observer = [
                    span
                    for span in self.spans()
                    if span.name.endswith(".media")
                    and span.attributes["request_id"] == "82"
                ][-1]
                self.assertNotEqual(
                    observer.context.trace_id, producer_context.trace_id
                )
                self.assertEqual(
                    observer.attributes[trace_attrs.RTP_LLM_REQUEST_TRACE_ID],
                    format(observer.context.trace_id, "032x"),
                )
                self.assertEqual(observer.links[0].context, producer_context)
                self.assertEqual(engine.mm_part.forward_count, 1)

    def test_preprocess_failure_finishes_media_with_original_error(self):
        engine = self.engine()
        inp = self.make_input()
        inp.url = "/definitely-missing-vit-trace-test/input.png"
        with self.assertRaises(Exception):
            engine.get_embedding_result([inp])
        engine._async_compute_executor.shutdown(wait=True)
        spans = self.spans()
        media = next(s for s in spans if s.name.endswith(".media"))
        self.assertEqual(media.status.status_code, StatusCode.ERROR)
        self.assertEqual(
            media.attributes[vit_media.PREFIX + "failure_stage"], "download"
        )
        self.assertTrue(
            any(
                s.name.endswith(".download")
                and s.status.status_code == StatusCode.ERROR
                for s in spans
            )
        )
        self.assertFalse(any(s.name.endswith(".embedding") for s in spans))

    def test_parent_inspection_is_attributed_to_media(self):
        engine = self.engine()
        engine._inspection._greennet_provider = InspectionProvider()
        engine.get_embedding_result([self.make_input(b"audit")], request_id=31)
        engine._async_compute_executor.shutdown(wait=True)
        spans = self.spans()
        green = [
            s
            for s in spans
            if s.name
            in (vit_media.PREFIX + "greennet", vit_media.PREFIX + "greennet.total")
        ]
        self.assertEqual(len(green), 2)
        self.assertTrue(
            all(s.attributes[vit_media.PREFIX + "greennet.passed"] for s in green)
        )

    def test_cancelled_queued_producer_finishes_once(self):
        engine = self.engine(vit_concurrency=1)
        engine.mm_part.release.clear()
        engine.async_submit([self.make_input(b"running")], request_id=51)
        self.assertTrue(engine.mm_part.forward_started.wait(5))
        engine.async_submit([self.make_input(b"queued")], request_id=52)
        self.assertEqual(engine.cancel_queued_request(52), 1)
        engine.mm_part.release.set()
        engine._async_compute_executor.shutdown(wait=True)
        spans = self.spans()
        cancelled = [
            s
            for s in spans
            if s.name.endswith(".media") and s.attributes["request_id"] == "52"
        ]
        self.assertEqual(len(cancelled), 1)
        self.assertEqual(cancelled[0].status.status_code, StatusCode.ERROR)
        children = [
            s for s in spans if s.parent.span_id == cancelled[0].context.span_id
        ]
        self.assertEqual(
            [s.name for s in children], [vit_media.PREFIX + "compute.queue"]
        )
        self.assertEqual(children[0].status.status_code, StatusCode.ERROR)
        self.assertEqual(
            cancelled[0].attributes[vit_media.PREFIX + "failure_stage"], "compute.queue"
        )
        self.assertEqual(
            cancelled[0].attributes[vit_media.PREFIX + "failure_reason"], "cancelled"
        )
        self.assertEqual(cancelled[0].attributes[vit_media.PREFIX + "error_code"], 499)

    def test_expired_compute_queue_is_error_not_success(self):
        engine = self.engine()
        inputs = [self.make_input()]
        member = vit_media._new_media(inputs, 91)[0]
        entry = SimpleNamespace(_vit_media_producer=member)
        member.begin("compute.queue")
        with patch.object(
            engine,
            "_fail_async_compute",
            side_effect=lambda key, entry, error: vit_media.finish_producer(
                entry, error
            ),
        ) as fail, patch.object(engine, "_async_compute") as compute:
            engine._run_async_compute(inputs, "key", entry, 91, time.monotonic() - 1)
        compute.assert_not_called()
        self.assertEqual(
            fail.call_args.args[2].exception_type, ExceptionType.GENERATE_TIMEOUT
        )
        spans = [s for s in self.spans() if s.name.startswith(vit_media.PREFIX)]
        self.assertEqual(len(spans), 2)
        for span in spans:
            self.assertEqual(span.status.status_code, StatusCode.ERROR)
            self.assertEqual(
                span.attributes[vit_media.PREFIX + "failure_stage"], "compute.queue"
            )
            self.assertEqual(
                span.attributes[vit_media.PREFIX + "failure_reason"], "timeout"
            )
            self.assertEqual(span.attributes[vit_media.PREFIX + "error_code"], 603)

    def test_worker_wait_timeout_does_not_invent_download_stage(self):
        item = MMWorkItem([self.make_input()])
        item.media_traces = vit_media._new_media(item.mm_inputs, 92)
        item.future = object()
        item.preprocess_ready_event = Mock(wait=Mock(return_value=False))
        with self.assertRaises(TimeoutError) as caught:
            list(PreprocessExecutor().iter_ready([item]))
        item.media_traces[0].finish(caught.exception)
        spans = [s for s in self.spans() if s.name.startswith(vit_media.PREFIX)]
        self.assertEqual(len(spans), 1)
        self.assertEqual(
            spans[0].attributes[vit_media.PREFIX + "failure_stage"], "preprocess.worker"
        )
        self.assertEqual(
            spans[0].attributes[vit_media.PREFIX + "failure_reason"], "timeout"
        )

    def test_preprocess_and_embedding_failures_keep_original_exception(self):
        for stage in ("preprocess", "embedding"):
            with self.subTest(stage=stage):
                engine = self.engine()
                if stage == "preprocess":
                    owner, method = engine.preprocess_executor, "preprocess_func"
                else:
                    owner, method = engine.mm_part, "batched_embedding"
                error = ValueError("injected-stage-failure")
                with patch.object(owner, method, side_effect=error), self.assertRaises(
                    Exception
                ):
                    engine.get_embedding_result(
                        [self.make_input(stage.encode())], request_id=93
                    )
                engine._async_compute_executor.shutdown(wait=True)
                root = [s for s in self.spans() if s.name.endswith(".media")][-1]
                self.assertEqual(
                    root.attributes[vit_media.PREFIX + "failure_stage"], stage
                )
                self.assertEqual(root.attributes["error.type"], "ValueError")
                self.assertEqual(root.status.status_code, StatusCode.ERROR)

    def test_worker_lost_without_timing_sidecar(self):
        executor = MultiprocessPreprocessExecutor.__new__(
            MultiprocessPreprocessExecutor
        )
        executor._pool_lock = threading.Lock()
        executor._preprocess_queue_lock = threading.Lock()
        executor._pending_preprocess_tasks = set()
        executor._rebuild_pool = Mock()
        item = MMWorkItem([self.make_input()])
        item.media_traces = vit_media._new_media(item.mm_inputs, 97)
        error = EOFError("worker IPC closed")
        item.future = Mock(get=Mock(side_effect=error))
        with self.assertRaises(EOFError) as caught:
            executor.get_result(item)
        self.assertIs(caught.exception, error)
        executor._rebuild_pool.assert_called_once()
        item.media_traces[0].finish(error)
        root = [s for s in self.spans() if s.name.endswith(".media")][-1]
        self.assertEqual(
            root.attributes[vit_media.PREFIX + "failure_stage"], "preprocess.worker"
        )
        self.assertEqual(
            root.attributes[vit_media.PREFIX + "failure_reason"], "worker_lost"
        )

    def test_inspection_rejection_preserves_business_code(self):
        class RejectingProvider(InspectionProvider):
            async def submit_sources(self, request, mm_inputs, sources):
                class Handle(GreenNetHandle):
                    async def wait_result(self):
                        return GreenNetVerdict(False, 2)

                    def cancel(self):
                        pass

                return Handle()

        engine = self.engine()
        engine._inspection._greennet_provider = RejectingProvider()
        with self.assertRaises(Exception) as caught:
            engine.get_embedding_result([self.make_input(b"rejected")], request_id=98)
        self.assertEqual(
            caught.exception.exception_type, ExceptionType.UNSAFE_INPUT_CONTENT
        )
        engine._async_compute_executor.shutdown(wait=True)
        root = [s for s in self.spans() if s.name.endswith(".media")][-1]
        self.assertEqual(
            root.attributes[vit_media.PREFIX + "failure_stage"], "greennet"
        )
        self.assertEqual(
            root.attributes[vit_media.PREFIX + "failure_reason"], "rejected"
        )
        self.assertEqual(root.attributes[vit_media.PREFIX + "error_code"], 908)

    def test_prefetch_and_publication_failures_are_attributed(self):
        for stage in ("embedding.prefetch", "cache.publish"):
            with self.subTest(stage=stage):
                item = MMWorkItem([self.make_input(stage.encode())])
                item.media_traces = vit_media._new_media(item.mm_inputs, 94)
                error = ValueError("injected-boundary-failure")
                with self.assertRaises(ValueError) as caught:
                    if stage == "embedding.prefetch":
                        batch = PrefetchedMMBatch([SimpleNamespace(work_items=[item])])
                        batch.prepare(
                            SimpleNamespace(
                                prepare_embedding_batch=Mock(side_effect=error)
                            )
                        )
                    else:
                        item.embedding_cache = SimpleNamespace(
                            complete=Mock(side_effect=error)
                        )
                        item.cache_key, item.cache_entry = "key", object()
                        item.complete_cache((torch.ones(1, 1), None))
                self.assertIs(caught.exception, error)
                item.media_traces[0].finish(error)
                root = [s for s in self.spans() if s.name.endswith(".media")][-1]
                self.assertEqual(
                    root.attributes[vit_media.PREFIX + "failure_stage"], stage
                )

    def test_trace_on_off_outputs_are_bitwise_identical(self):
        outputs = []
        for enabled in (False, True):
            engine = self.engine()
            inputs = [self.make_input(data) for data in (b"a", b"medium", b"a")]
            with patch.object(tracing, "is_telemetry_active", return_value=enabled):
                cold = engine.get_embedding_result(inputs, request_id=95)
                warm = engine.get_embedding_result(inputs, request_id=96)
                engine._async_compute_executor.shutdown(wait=True)
            self.assertEqual(engine.mm_part.forward_count, 2)
            outputs.append((cold, warm))
        for cold_or_warm in range(2):
            before, after = (result[cold_or_warm] for result in outputs)
            self.assertEqual(len(before), len(after))
            for left, right, expected in zip(before, after, (1, 6, 1)):
                for field in (
                    "embeddings",
                    "position_ids",
                    "extra_input",
                    "feature_hashes",
                ):
                    lhs, rhs = getattr(left, field), getattr(right, field)
                    self.assertEqual(len(lhs), len(rhs))
                    for a, b in zip(lhs, rhs):
                        self.assertEqual(a.shape, b.shape)
                        self.assertEqual(a.dtype, b.dtype)
                        self.assertEqual(a.device, b.device)
                        self.assertTrue(torch.isfinite(b).all())
                        self.assertTrue(torch.equal(a, b))
                self.assertEqual(right.embeddings[0].item(), expected)

    def test_two_request_traces_share_one_real_scheduler_batch(self):
        engine = self.engine(
            gpu_max_batch_size=2, gpu_max_batch_images=2, gpu_batch_wait_ms=200
        )
        barrier = threading.Barrier(2)
        preprocess = engine.preprocess_executor.preprocess_func

        def simultaneous(*args, **kwargs):
            barrier.wait(timeout=5)
            return preprocess(*args, **kwargs)

        engine.preprocess_executor.preprocess_func = simultaneous
        engine.async_submit([self.make_input(b"batch-one")], request_id=61)
        other = tracing.start_server_span("other_rpc", {})
        try:
            engine.async_submit([self.make_input(b"batch-two")], request_id=62)
        finally:
            other.finish()
        engine._async_compute_executor.shutdown(wait=True)
        spans = self.spans()
        batches = [s for s in spans if s.name.endswith("embedding.batch")]
        media = [s for s in spans if s.name.endswith(".media")]
        self.assertEqual(len(batches), 1)
        self.assertEqual(
            batches[0].attributes[vit_media.PREFIX + "batch.media_count"], 2
        )
        self.assertEqual(len({s.context.trace_id for s in media}), 2)
        self.assertEqual(
            {link.context.trace_id for link in batches[0].links},
            {s.context.trace_id for s in media},
        )
        for span in spans:
            if span.name.startswith(vit_media.PREFIX):
                self.assertEqual(
                    span.attributes[trace_attrs.RTP_LLM_REQUEST_TRACE_ID],
                    format(span.context.trace_id, "032x"),
                )

    def test_hash_metadata_filter_preserves_original_input_index(self):
        from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import MultimodalInputsPB
        from rtp_llm.multimodal.multimodal_util import trans_mm_input
        from rtp_llm.server.mm_cache_metadata import get_mm_cache_metadata

        engine = self.engine()
        request = MultimodalInputsPB(request_id=71)
        for data in (b"cached", b"uncached"):
            request.multimodal_inputs.add(
                multimodal_url=self.make_input(data).url,
                multimodal_type=int(MMUrlType.IMAGE),
            )
        converted = trans_mm_input(request)
        engine.get_embedding_result(converted[:1], request_id=70)
        get_mm_cache_metadata(
            engine, [item.cache_key() for item in converted], inputs=request
        )
        engine._async_compute_executor.shutdown(wait=True)
        media = [
            s
            for s in self.spans()
            if s.name.endswith(".media") and s.attributes["request_id"] == "71"
        ]
        self.assertEqual(len(media), 1)
        self.assertEqual(media[0].attributes[vit_media.MEDIA + "index"], 1)
        self.assertEqual(media[0].attributes[vit_media.MEDIA + "url"], converted[1].url)

    def test_spawn_preprocess_and_inspection_timestamps_return_to_parent(self):
        context = multiprocessing.get_context("spawn")
        config = VitConfig()
        config.mm_preprocess_max_workers = 1
        config.url_cache_item_num = config.mm_cache_item_num = 0
        executor = MultiprocessPreprocessExecutor(
            context,
            config,
            {},
            CPUModel.preprocess_input,
            inspection_factory=InspectionProvider,
        )
        self.addCleanup(executor.shutdown)
        item = MMWorkItem([self.make_input(b"spawn")], deadline=time.monotonic() + 30)
        item.media_traces = vit_media._new_media(item.mm_inputs, 41)
        item.capture_inspection_sources = True
        item.inspection_request = SimpleNamespace(id="41", input_index_offset=0)
        executor.submit(item)
        executor.get_result(item)
        self.assertTrue(item.inspection_future.result(timeout=10).passed)
        item.media_traces[0].finish()
        spans = self.spans()
        names = {s.name.removeprefix(vit_media.PREFIX) for s in spans}
        self.assertTrue(
            {
                "download",
                "preprocess.queue",
                "preprocess",
                "greennet.queue",
                "greennet",
            }.issubset(names),
            names,
        )
        worker_spans = [
            s for s in spans if vit_media.PREFIX + "worker.pid" in s.attributes
        ]
        self.assertTrue(worker_spans)
        self.assertTrue(
            all(
                s.attributes[vit_media.PREFIX + "worker.pid"] != os.getpid()
                for s in worker_spans
            )
        )


if __name__ == "__main__":
    unittest.main()
