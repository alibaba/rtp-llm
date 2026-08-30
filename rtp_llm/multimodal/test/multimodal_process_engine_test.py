import asyncio
import concurrent.futures
import io
import json
import os
import sys
import threading
import time
import types
from types import SimpleNamespace
from typing import List
from unittest import TestCase, main
from unittest.mock import MagicMock, patch

import PIL
import pillow_avif
import pillow_heif
import torch
from PIL import Image, ImageFile

from rtp_llm.access_logger.access_logger import MMAccessLogger
from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.py_config_modules import (
    ProfilingDebugLoggingConfig,
    PyEnvConfigs,
    VitConfig,
)
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    MultimodalInputPB,
    MultimodalInputsPB,
)
from rtp_llm.metrics.kmonitor_metric_reporter import AccMetrics, GaugeMetrics
from rtp_llm.multimodal.greennet_hook import (
    GreenNetHandle,
    GreenNetProvider,
    GreenNetVerdict,
)
from rtp_llm.multimodal.mm_error_messages import MMErr
from rtp_llm.multimodal.mm_process_engine import (
    MMEmbeddingAsyncCache,
    MMEmbeddingCacheEntry,
    MMProcessEngine,
    MMWorkItem,
)
from rtp_llm.multimodal.multimodal_mixins.multimodal_common import (
    MultiModalEmbeddingInterface,
)
from rtp_llm.multimodal.multimodal_mixins.qwen2_vl.image_processing_qwen2_vl import (
    Qwen2VLImageProcessor,
)
from rtp_llm.multimodal.multimodal_mixins.qwen2_vl.qwen2_vl_mixin import (
    Qwen2_VLImageEmbedding,
)
from rtp_llm.multimodal.multimodal_util import vit_emb_cache_
from rtp_llm.ops import MMPreprocessConfig, MultimodalInput
from rtp_llm.utils.base_model_datatypes import MMUrlType


class FakeMultiModalEmbeddingInterface(Qwen2_VLImageEmbedding):
    def __init__(self, config: ModelConfig = ModelConfig()):
        self.data_type = config.compute_dtype
        self.image_processor: Qwen2VLImageProcessor = (
            Qwen2VLImageProcessor.from_pretrained(
                "./rtp_llm/multimodal/test/testdata/qwen2_vl/"
            )
        )
        self.spatial_merge_size = 2

    @torch.inference_mode()
    def embedding(self, data, **kwargs):
        return torch.tensor(0), None

    @staticmethod
    def preprocess_input(
        mm_inputs: List[MultimodalInput], vit_config: VitConfig, **kwargs
    ):
        return mm_inputs, kwargs

    def get_preprocess_params(self):
        return {}


class PreprcoesException(Exception):
    pass


class FakeMultiModalEmbeddingInterfacePreprocessException(
    FakeMultiModalEmbeddingInterface
):
    @staticmethod
    def preprocess_input(
        mm_inputs: List[MultimodalInput], vit_config: VitConfig, **kwargs
    ):
        raise PreprcoesException(kwargs)

    def get_preprocess_params(self):
        return {"test": "hello"}


class FakeMultiModalEmbeddingInterfaceSlow(FakeMultiModalEmbeddingInterface):
    """Preprocess function that sleeps to guarantee timeout."""

    @staticmethod
    def preprocess_input(
        mm_inputs: List[MultimodalInput], vit_config: VitConfig, **kwargs
    ):
        time.sleep(5)
        return mm_inputs, kwargs

    def get_preprocess_params(self):
        return {}


class FakeMultiModalEmbeddingInterfaceSlowEmbedding(FakeMultiModalEmbeddingInterface):
    """batched_embedding sleeps, to exercise the embedding-level timeout on the
    default (non-gpu-batch) serial path."""

    @torch.inference_mode()
    def batched_embedding(self, data_list, mm_types, **kwargs):
        time.sleep(5)
        return [(torch.tensor(0), None) for _ in data_list]


class FakeMultiModalEmbeddingInterfaceProcessCrash(FakeMultiModalEmbeddingInterface):
    """Preprocess function that crashes the worker process to trigger BrokenProcessPool."""

    @staticmethod
    def preprocess_input(
        mm_inputs: List[MultimodalInput], vit_config: VitConfig, **kwargs
    ):
        os._exit(1)

    def get_preprocess_params(self):
        return {}


class FakeMultiModalEmbeddingInterfaceBadCount(FakeMultiModalEmbeddingInterface):
    """batched_embedding returns the wrong number of outputs."""

    @torch.inference_mode()
    def batched_embedding(self, data_list, mm_types, **kwargs):
        # One fewer than requested, to trip the count guard.
        return [(torch.tensor(0), None) for _ in range(len(data_list) - 1)]


class FakeEmbeddingLengthInterface(FakeMultiModalEmbeddingInterface):
    """Return distinct token counts so request-level aggregation is testable."""

    @staticmethod
    def preprocess_input(
        mm_inputs: List[MultimodalInput], vit_config: VitConfig, **kwargs
    ):
        return torch.tensor([len(mm_inputs[0].url)])

    @torch.inference_mode()
    def batched_embedding(self, data_list, mm_types, **kwargs):
        return [
            (torch.zeros((int(data.reshape(-1)[0]), 4)), None) for data in data_list
        ]


class FakeModel:
    def __init__(self, mm_part: MultiModalEmbeddingInterface = None):
        self.model_config = ModelConfig()
        self.model_config.mm_model_config.mm_position_ids_style = 2
        self.mm_part = mm_part


class MMProcessEngineTest(TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = FakeModel(FakeMultiModalEmbeddingInterface())
        self.mm_process_engine = MMProcessEngine(
            self.model.mm_part,
            self.model.model_config,
            VitConfig(),
            ProfilingDebugLoggingConfig(),
        )

    def test_embedding(self):
        res = self.mm_process_engine.mm_embedding_cpp(
            ["./rtp_llm/multimodal/test/testdata/qwen2_vl/1.jpg"],
            [MMUrlType.IMAGE],
            [torch.empty(0)],
            [[-1, -1, -1, -1, -1, -1, -1, [], 30000]],
        )
        self.assertEqual(res.embeddings, [torch.tensor(0)])
        self.assertEqual(res.position_ids, [])

        mm_inputs = MultimodalInputsPB()
        mm_input = MultimodalInputPB()
        mm_input.multimodal_url = "./rtp_llm/multimodal/test/testdata/qwen2_vl/1.jpg"
        mm_input.multimodal_type = MMUrlType.IMAGE
        mm_input.mm_preprocess_config.mm_timeout_ms = 30000
        mm_inputs.multimodal_inputs.append(mm_input)
        res = self.mm_process_engine.mm_embedding_rpc(mm_inputs)
        self.assertEqual(res.embeddings, [torch.tensor(0)])
        self.assertEqual(res.position_ids, [])

    @patch("rtp_llm.multimodal.mm_process_engine.kmonitor.report")
    def test_embedding_length_metric_sums_a_request(self, report):
        """A multi-image request emits one gauge containing all visual tokens."""
        model = FakeModel(FakeEmbeddingLengthInterface())
        vit_config = VitConfig()
        vit_config.use_local_preprocess = True
        vit_config.use_gpu_batch = True
        vit_config.gpu_batch_wait_ms = 100
        vit_config.mm_cache_item_num = 0
        engine = MMProcessEngine(
            model.mm_part,
            model.model_config,
            vit_config,
            ProfilingDebugLoggingConfig(),
        )
        config = MMPreprocessConfig(-1, -1, -1, -1, -1, -1, -1, [], 30000)
        inputs = [
            MultimodalInput("a", MMUrlType.IMAGE, torch.empty(0), config),
            MultimodalInput("bb", MMUrlType.IMAGE, torch.empty(0), config),
        ]
        try:
            result = engine.mm_embedding_impl(inputs)
        finally:
            engine.stop()

        self.assertEqual(
            [embedding.shape[0] for embedding in result.embeddings], [1, 2]
        )
        lengths = [
            call.args[1]
            for call in report.call_args_list
            if call.args and call.args[0] == GaugeMetrics.VIT_EMBEDDING_LENGTH_METRIC
        ]
        self.assertEqual(lengths, [3])

        image_counts = [
            call.args[1]
            for call in report.call_args_list
            if call.args and call.args[0] == GaugeMetrics.VIT_IMAGE_COUNT_METRIC
        ]
        self.assertEqual(image_counts, [2])

    def test_timeout(self):
        model = FakeModel(FakeMultiModalEmbeddingInterfaceSlow())
        vit_config = VitConfig()
        vit_config.mm_cache_item_num = 0
        engine = MMProcessEngine(
            model.mm_part,
            model.model_config,
            vit_config,
            ProfilingDebugLoggingConfig(),
        )
        with self.assertRaises(TimeoutError):
            engine.mm_embedding_cpp(
                ["./rtp_llm/multimodal/test/testdata/qwen2_vl/1.jpg"],
                [MMUrlType.IMAGE],
                [torch.empty(0)],
                [
                    [-1, -1, -1, -1, -1, -1, -1, [], 1],
                ],
            )
        engine.stop()

    def test_preprocess(self):
        model = FakeModel(FakeMultiModalEmbeddingInterfacePreprocessException())
        vit_config = VitConfig()
        vit_config.mm_cache_item_num = 0
        mm_process_engine = MMProcessEngine(
            model.mm_part,
            model.model_config,
            vit_config,
            ProfilingDebugLoggingConfig(),
        )
        try:
            mm_process_engine.mm_embedding_cpp(
                ["./rtp_llm/multimodal/test/testdata/qwen2_vl/1.jpg"],
                [MMUrlType.IMAGE],
                [torch.empty(0)],
                [
                    [-1, -1, -1, -1, -1, -1, -1, [], 30000],
                ],
            )
        except PreprcoesException as e:
            self.assertEqual(str(e), "{'test': 'hello'}")

    @patch("rtp_llm.multimodal.mm_process_engine.kmonitor.report")
    def test_error_qps_is_reported_for_preprocess_failure(self, report):
        model = FakeModel(FakeMultiModalEmbeddingInterfacePreprocessException())
        vit_config = VitConfig()
        vit_config.use_local_preprocess = True
        vit_config.mm_cache_item_num = 0
        engine = MMProcessEngine(
            model.mm_part,
            model.model_config,
            vit_config,
            ProfilingDebugLoggingConfig(),
        )
        try:
            with self.assertRaises(PreprcoesException):
                engine.mm_embedding_cpp(
                    ["fake://error-qps-preprocess"],
                    [MMUrlType.IMAGE],
                    [torch.empty(0)],
                    [[-1, -1, -1, -1, -1, -1, -1, [], 30000]],
                )
        finally:
            engine.stop()

        error_reports = [
            call
            for call in report.call_args_list
            if call.args and call.args[0] == AccMetrics.VIT_ERROR_QPS_METRIC
        ]
        self.assertEqual(len(error_reports), 1)

    def test_local_preprocess_mode(self):
        """LocalPreprocessExecutor path: use_local_preprocess=True bypasses the worker pool."""
        model = FakeModel(FakeMultiModalEmbeddingInterface())
        vit_config = VitConfig()
        vit_config.use_local_preprocess = True
        engine = MMProcessEngine(
            model.mm_part,
            model.model_config,
            vit_config,
            ProfilingDebugLoggingConfig(),
        )
        res = engine.mm_embedding_cpp(
            ["./rtp_llm/multimodal/test/testdata/qwen2_vl/1.jpg"],
            [MMUrlType.IMAGE],
            [torch.empty(0)],
            [[-1, -1, -1, -1, -1, -1, -1, [], 30000]],
        )
        self.assertEqual(res.embeddings, [torch.tensor(0)])
        engine.stop()

    def test_query_counter(self):
        self.assertEqual(self.mm_process_engine.get_query_num(), 0)
        self.mm_process_engine.inc_query_num()
        self.mm_process_engine.inc_query_num()
        self.assertEqual(self.mm_process_engine.get_query_num(), 2)
        self.mm_process_engine.dec_query_num()
        self.assertEqual(self.mm_process_engine.get_query_num(), 1)
        self.mm_process_engine.dec_query_num()
        self.assertEqual(self.mm_process_engine.get_query_num(), 0)

    def test_work_item_rejects_empty_inputs(self):
        with self.assertRaises(ValueError):
            MMWorkItem([])

    def test_work_item_uses_global_timeout_when_request_timeout_is_unset(self):
        preprocess_config = MMPreprocessConfig(-1, -1, -1, -1, -1, -1, -1, [], -1)
        mm_input = MultimodalInput(
            "", MMUrlType.IMAGE, torch.empty(0), preprocess_config
        )

        self.assertEqual(preprocess_config.mm_timeout_ms, -1)
        self.assertEqual(
            MMWorkItem([mm_input], mm_timeout_ms=123000).mm_timeout_ms, 123000
        )

    def test_multimodal_cache_key_ignores_timeout(self):
        def make_input(width: int, timeout_ms: int) -> MultimodalInput:
            preprocess_config = MMPreprocessConfig(
                width, 480, 100, 1000, 2, 1, 64, [0.25, 0.75], timeout_ms
            )
            return MultimodalInput(
                "https://example.com/image.jpg",
                MMUrlType.IMAGE,
                torch.empty(0),
                preprocess_config,
            )

        self.assertEqual(
            make_input(640, 30000).cache_key(),
            make_input(640, 120000).cache_key(),
        )
        self.assertNotEqual(
            make_input(640, 30000).cache_key(),
            make_input(800, 30000).cache_key(),
        )

    def test_embedding_timeout_default_path(self):
        """Default (non-gpu-batch) serial path enforces an embedding-level timeout.

        Regression guard for the timeout semantic migrated from the old inline
        path: a slow batched_embedding must surface as TimeoutError, not hang.
        """
        model = FakeModel(FakeMultiModalEmbeddingInterfaceSlowEmbedding())
        vit_config = VitConfig()
        vit_config.use_local_preprocess = True  # fast preprocess; isolate embedding
        vit_config.mm_cache_item_num = 0  # no cache hit to short-circuit the forward
        engine = MMProcessEngine(
            model.mm_part,
            model.model_config,
            vit_config,
            ProfilingDebugLoggingConfig(),
        )
        try:
            with self.assertRaises(TimeoutError):
                engine.mm_embedding_cpp(
                    ["./rtp_llm/multimodal/test/testdata/qwen2_vl/1.jpg"],
                    [MMUrlType.IMAGE],
                    [torch.empty(0)],
                    [[-1, -1, -1, -1, -1, -1, -1, [], 100]],  # mm_timeout_ms=100
                )
        finally:
            engine.stop()

    def test_batched_embedding_count_mismatch(self):
        """Serial-mode scheduler path fails fast when batched_embedding returns wrong count."""
        model = FakeModel(FakeMultiModalEmbeddingInterfaceBadCount())
        vit_config = VitConfig()
        vit_config.use_local_preprocess = True  # local preprocess, serial scheduler
        engine = MMProcessEngine(
            model.mm_part,
            model.model_config,
            vit_config,
            ProfilingDebugLoggingConfig(),
        )
        try:
            with self.assertRaises(RuntimeError):
                engine.mm_embedding_cpp(
                    ["url0", "url1"],
                    [MMUrlType.IMAGE, MMUrlType.IMAGE],
                    [torch.empty(0), torch.empty(0)],
                    [[-1, -1, -1, -1, -1, -1, -1, [], 30000]] * 2,
                )
        finally:
            engine.stop()

    def test_worker_crash_recovery(self):
        """Pool rebuilds after worker process crash and subsequent requests succeed."""
        model = FakeModel(FakeMultiModalEmbeddingInterfaceProcessCrash())
        vit_config = VitConfig()
        vit_config.mm_cache_item_num = 0
        engine = MMProcessEngine(
            model.mm_part,
            model.model_config,
            vit_config,
            ProfilingDebugLoggingConfig(),
        )

        # First call crashes the worker — should raise but pool rebuilds internally
        with self.assertRaises(Exception):
            engine.mm_embedding_cpp(
                ["./rtp_llm/multimodal/test/testdata/qwen2_vl/1.jpg"],
                [MMUrlType.IMAGE],
                [torch.empty(0)],
                [[-1, -1, -1, -1, -1, -1, -1, [], 5000]],
            )

        # Swap to a working mm_part so the rebuilt pool can serve requests
        working_model = FakeModel(FakeMultiModalEmbeddingInterface())
        engine.preprocess_executor.preprocess_func = (
            working_model.mm_part.preprocess_input
        )
        engine.preprocess_executor._rebuild_pool()

        # Subsequent request should succeed after pool recovery
        res = engine.mm_embedding_cpp(
            ["./rtp_llm/multimodal/test/testdata/qwen2_vl/1.jpg"],
            [MMUrlType.IMAGE],
            [torch.empty(0)],
            [[-1, -1, -1, -1, -1, -1, -1, [], 30000]],
        )
        self.assertEqual(res.embeddings, [torch.tensor(0)])
        engine.stop()

    def test_consecutive_timeout_triggers_rebuild(self):
        """Pool rebuilds after consecutive timeouts reach the threshold."""
        from rtp_llm.multimodal.mm_process_engine import MultiprocessPreprocessExecutor

        model = FakeModel(FakeMultiModalEmbeddingInterfaceSlow())
        vit_config = VitConfig()
        vit_config.mm_preprocess_max_workers = 2
        vit_config.mm_cache_item_num = 0
        engine = MMProcessEngine(
            model.mm_part,
            model.model_config,
            vit_config,
            ProfilingDebugLoggingConfig(),
        )

        executor = engine.preprocess_executor
        if not isinstance(executor, MultiprocessPreprocessExecutor):
            self.skipTest("Not using multiprocess executor")

        old_pool = executor.pool

        # Simulate consecutive timeouts reaching the threshold
        executor._consecutive_timeouts = executor._max_consecutive_timeouts - 1

        # This timeout should trigger a rebuild
        with self.assertRaises(TimeoutError):
            engine.mm_embedding_cpp(
                ["./rtp_llm/multimodal/test/testdata/qwen2_vl/1.jpg"],
                [MMUrlType.IMAGE],
                [torch.empty(0)],
                [[-1, -1, -1, -1, -1, -1, -1, [], 1]],
            )

        # Pool should have been rebuilt
        self.assertIsNot(executor.pool, old_pool)
        self.assertEqual(executor._consecutive_timeouts, 0)
        engine.stop()


class PreprocessMetricTest(TestCase):
    @patch("rtp_llm.multimodal.mm_process_engine.kmonitor.report")
    def test_preprocess_queue_metric_tracks_pending_tasks(self, report):
        from rtp_llm.multimodal.mm_process_engine import MultiprocessPreprocessExecutor

        class FakePool:
            def __init__(self):
                self.callbacks = []

            def apply_async(self, *args, **kwargs):
                self.callbacks.append((kwargs["callback"], kwargs["error_callback"]))
                return object()

        executor = object.__new__(MultiprocessPreprocessExecutor)
        executor.pool = FakePool()
        executor._pool_lock = threading.Lock()
        executor._preprocess_queue_lock = threading.Lock()
        executor._pending_preprocess_tasks = set()
        executor._next_preprocess_task_id = 0

        config = MMPreprocessConfig(-1, -1, -1, -1, -1, -1, -1, [], 30000)
        work_items = [
            MMWorkItem(
                [
                    MultimodalInput(
                        f"fake://queue-{index}",
                        MMUrlType.IMAGE,
                        torch.empty(0),
                        config,
                    )
                ],
                mm_timeout_ms=30000,
            )
            for index in range(2)
        ]

        executor.submit(work_items[0])
        executor.submit(work_items[1])
        depth_values = [
            call.args[1]
            for call in report.call_args_list
            if call.args
            and call.args[0] == GaugeMetrics.VIT_PREPROCESS_QUEUE_SIZE_METRIC
        ]
        self.assertEqual(depth_values[-1], 2)

        executor.pool.callbacks[0][0](None)
        executor.pool.callbacks[1][1](RuntimeError("preprocess failed"))
        depth_values = [
            call.args[1]
            for call in report.call_args_list
            if call.args
            and call.args[0] == GaugeMetrics.VIT_PREPROCESS_QUEUE_SIZE_METRIC
        ]
        self.assertEqual(depth_values[-1], 0)


class FakeSlowEmbeddingInterface(FakeMultiModalEmbeddingInterface):
    """Embedding that takes a configurable delay, for testing async concurrency."""

    delay = 0.3

    @torch.inference_mode()
    def embedding(self, data, **kwargs):
        time.sleep(self.delay)
        return torch.tensor(1), None


class MMEmbeddingCacheEntryTest(TestCase):
    def test_complete_then_wait(self):
        entry = MMEmbeddingCacheEntry()
        self.assertFalse(entry.is_done)
        entry.complete("result_value")
        self.assertTrue(entry.is_done)
        self.assertEqual(entry.wait(), "result_value")

    def test_wait_blocks_until_complete(self):
        entry = MMEmbeddingCacheEntry()
        result_holder = [None]

        def setter():
            time.sleep(0.1)
            entry.complete(42)

        threading.Thread(target=setter, daemon=True).start()
        result_holder[0] = entry.wait(timeout=5.0)
        self.assertEqual(result_holder[0], 42)

    def test_wait_timeout(self):
        entry = MMEmbeddingCacheEntry()
        with self.assertRaises(TimeoutError):
            entry.wait(timeout=0.05)

    def test_fail_then_wait_raises(self):
        entry = MMEmbeddingCacheEntry()
        entry.fail(ValueError("boom"))
        self.assertTrue(entry.is_done)
        with self.assertRaises(ValueError):
            entry.wait()


class MMEmbeddingAsyncCacheTest(TestCase):
    def test_miss_then_complete_then_hit(self):
        cache = MMEmbeddingAsyncCache(max_size=10)
        state, entry = cache.try_acquire("key1")
        self.assertEqual(state, "miss")
        self.assertFalse(entry.is_done)

        entry.complete("val1")

        state2, entry2 = cache.try_acquire("key1")
        self.assertEqual(state2, "complete")
        self.assertIs(entry2, entry)
        self.assertEqual(entry2.wait(), "val1")

    def test_in_progress_state(self):
        cache = MMEmbeddingAsyncCache(max_size=10)
        state, entry = cache.try_acquire("key1")
        self.assertEqual(state, "miss")

        state2, entry2 = cache.try_acquire("key1")
        self.assertEqual(state2, "in_progress")
        self.assertIs(entry2, entry)

    def test_remove(self):
        cache = MMEmbeddingAsyncCache(max_size=10)
        _, entry = cache.try_acquire("key1")
        entry.complete("v")
        cache.remove("key1")

        state, entry2 = cache.try_acquire("key1")
        self.assertEqual(state, "miss")
        self.assertIsNot(entry2, entry)

    def test_eviction(self):
        cache = MMEmbeddingAsyncCache(max_size=2)
        _, e1 = cache.try_acquire("k1")
        e1.complete("v1")
        _, e2 = cache.try_acquire("k2")
        e2.complete("v2")
        _, e3 = cache.try_acquire("k3")

        # Eviction ran when k3 was inserted (3 > max_size=2),
        # removing k1 (oldest done entry). k3 is in_progress.
        self.assertEqual(len(cache._entries), 2)
        self.assertNotIn("k1", cache._entries)
        state_k3, _ = cache.try_acquire("k3")
        self.assertEqual(state_k3, "in_progress")

    def test_resize(self):
        cache = MMEmbeddingAsyncCache(max_size=5)
        cache.resize(20)
        self.assertEqual(cache._max_size, 20)

    def test_weighted_lru_evicts_by_actual_tensor_bytes(self):
        cache = MMEmbeddingAsyncCache(max_size=1, max_bytes=32)
        _, e1 = cache.try_acquire("k1")
        e1.complete((torch.zeros((2, 2)), None))  # 16 bytes
        _, e2 = cache.try_acquire("k2")
        e2.complete((torch.zeros((2, 2)), None))  # 16 bytes

        # Weighted mode can retain more entries than the legacy count cap.
        self.assertEqual(cache.stats()["resident_entries"], 2)
        self.assertEqual(cache.try_acquire("k1")[0], "complete")

        _, e3 = cache.try_acquire("k3")
        e3.complete((torch.zeros((2, 2)), None))

        # k1 was touched, so k2 is the least-recently-used completed entry.
        self.assertNotIn("k2", cache._entries)
        self.assertIn("k1", cache._entries)
        self.assertIn("k3", cache._entries)
        stats = cache.stats()
        self.assertEqual(stats["resident_bytes"], 32)
        self.assertEqual(stats["resident_tokens"], 4)
        self.assertEqual(stats["eviction"], 1)

    @patch("rtp_llm.multimodal.mm_embedding_cache.kmonitor.report")
    def test_cache_token_metric_tracks_eviction_and_one_dimensional_embedding(
        self, report
    ):
        from rtp_llm.multimodal.mm_embedding_cache import _embedding_result_cost

        # The cache stores embedding vectors; a [hidden] tensor represents one
        # token rather than hidden scalar tokens.
        self.assertEqual(
            _embedding_result_cost((torch.zeros(4), None))[0],
            1,
        )

        cache = MMEmbeddingAsyncCache(max_size=1, report_metrics=True)
        _, first = cache.try_acquire("first")
        first.complete((torch.zeros((2, 4)), None))

        # Inserting a pending entry evicts the completed entry. The resident
        # gauge must drop immediately instead of waiting for the next result.
        cache.try_acquire("second")
        token_values = [
            call.args[1]
            for call in report.call_args_list
            if call.args
            and call.args[0] == GaugeMetrics.VIT_EMBEDDING_CACHE_TOKENS_METRIC
        ]
        self.assertEqual(token_values, [2, 0])


class VitErrorReportingTest(TestCase):
    @patch("rtp_llm.multimodal.mm_process_engine.kmonitor.report")
    def test_proxy_worker_reports_each_error_once(self, report):
        engine = object.__new__(MMProcessEngine)
        engine.is_proxy_mode = True
        error = RuntimeError("worker preprocessing failed")

        engine.report_vit_error(error)
        engine.report_vit_error(error)

        error_reports = [
            call
            for call in report.call_args_list
            if call.args and call.args[0] == AccMetrics.VIT_ERROR_QPS_METRIC
        ]
        self.assertEqual(len(error_reports), 1)


class AsyncSubmitGetEmbeddingTest(TestCase):
    def _make_engine(self, mm_part=None, vit_concurrency=64, vit_max_queue_size=64):
        model = FakeModel(mm_part or FakeMultiModalEmbeddingInterface())
        vit_config = VitConfig()
        vit_config.use_local_preprocess = True
        vit_config.vit_concurrency = vit_concurrency
        vit_config.vit_max_queue_size = vit_max_queue_size
        return MMProcessEngine(
            model.mm_part,
            model.model_config,
            vit_config,
            ProfilingDebugLoggingConfig(),
        )

    def _make_input(self, url):
        return MultimodalInput(
            url,
            MMUrlType.IMAGE,
            torch.empty(0),
            MMPreprocessConfig(-1, -1, -1, -1, -1, -1, -1, [], 30000),
        )

    def test_async_submit_returns_keys(self):
        engine = self._make_engine()
        inp = self._make_input("./rtp_llm/multimodal/test/testdata/qwen2_vl/1.jpg")
        keys = engine.async_submit([inp])
        self.assertEqual(len(keys), 1)
        self.assertIsInstance(keys[0], str)
        self.assertTrue(len(keys[0]) > 0)
        engine.stop()

    def test_submit_then_get(self):
        engine = self._make_engine()
        inp = self._make_input("./rtp_llm/multimodal/test/testdata/qwen2_vl/1.jpg")
        engine.async_submit([inp])
        results = engine.get_embedding_result([inp])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].embeddings, [torch.tensor(0)])
        engine.stop()

    def test_get_without_submit_computes_synchronously(self):
        engine = self._make_engine()
        inp = self._make_input("./rtp_llm/multimodal/test/testdata/qwen2_vl/1.jpg")
        results = engine.get_embedding_result([inp])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].embeddings, [torch.tensor(0)])
        engine.stop()

    def test_cache_hit_is_fast(self):
        engine = self._make_engine()
        inp = self._make_input("./rtp_llm/multimodal/test/testdata/qwen2_vl/1.jpg")
        engine.get_embedding_result([inp])

        t0 = time.time()
        results = engine.get_embedding_result([inp])
        elapsed = time.time() - t0
        self.assertLess(elapsed, 0.05)
        self.assertEqual(results[0].embeddings, [torch.tensor(0)])
        engine.stop()

    def test_duplicate_submit_no_recompute(self):
        engine = self._make_engine()
        inp = self._make_input("./rtp_llm/multimodal/test/testdata/qwen2_vl/1.jpg")
        keys1 = engine.async_submit([inp])
        keys2 = engine.async_submit([inp])
        self.assertEqual(keys1, keys2)
        engine.stop()

    def test_async_compute_concurrency_is_bounded(self):
        engine = self._make_engine(vit_concurrency=2)
        release = threading.Event()
        saturated = threading.Event()
        completed = threading.Event()
        lock = threading.Lock()
        active = 0
        max_active = 0
        completed_count = 0

        def blocked_compute(mm_inputs, cache_key, entry, request_id=0):
            nonlocal active, max_active, completed_count
            with lock:
                active += 1
                max_active = max(max_active, active)
                if active == 2:
                    saturated.set()
            try:
                release.wait(timeout=5)
                entry.complete((torch.tensor(0), None))
            finally:
                with lock:
                    active -= 1
                    completed_count += 1
                    if completed_count == 8:
                        completed.set()

        engine._async_compute = blocked_compute
        try:
            for index in range(8):
                engine.async_submit([self._make_input(f"fake://bounded-{index}")])

            self.assertTrue(saturated.wait(timeout=2))
            time.sleep(0.1)
            self.assertEqual(max_active, 2)

            release.set()
            self.assertTrue(completed.wait(timeout=5))
            self.assertEqual(max_active, 2)
        finally:
            release.set()
            engine.stop()

    @patch("rtp_llm.multimodal.mm_process_engine.kmonitor.report")
    def test_async_compute_queue_rejects_over_capacity(self, report):
        engine = self._make_engine(vit_concurrency=1, vit_max_queue_size=1)
        release = threading.Event()
        started = threading.Event()
        completed = threading.Event()
        lock = threading.Lock()
        completed_count = 0

        def blocked_compute(mm_inputs, cache_key, entry, request_id=0):
            nonlocal completed_count
            started.set()
            release.wait(timeout=5)
            entry.complete((torch.tensor(0), None))
            with lock:
                completed_count += 1
                if completed_count == 2:
                    completed.set()

        engine._async_compute = blocked_compute
        first = self._make_input("fake://queue-running")
        queued = self._make_input("fake://queue-waiting")
        rejected = self._make_input("fake://queue-rejected")
        try:
            engine.async_submit([first])
            self.assertTrue(started.wait(timeout=2))
            engine.async_submit([queued])

            with self.assertRaises(FtRuntimeException) as raised:
                engine.async_submit([rejected])
            self.assertEqual(
                raised.exception.exception_type,
                ExceptionType.CONCURRENCY_LIMIT_ERROR,
            )
            self.assertIsNone(engine._embedding_cache.peek(rejected.cache_key()))
            error_reports = [
                call
                for call in report.call_args_list
                if call.args and call.args[0] == AccMetrics.VIT_ERROR_QPS_METRIC
            ]
            self.assertEqual(len(error_reports), 1)

            release.set()
            self.assertTrue(completed.wait(timeout=5))
        finally:
            release.set()
            engine.stop()

    def test_cancel_request_removes_only_queued_work(self):
        engine = self._make_engine(vit_concurrency=1, vit_max_queue_size=1)
        release = threading.Event()
        first_started = threading.Event()
        first_done = threading.Event()
        started_urls = []

        def blocked_compute(mm_inputs, cache_key, entry, request_id=0):
            started_urls.append(mm_inputs[0].url)
            first_started.set()
            release.wait(timeout=5)
            entry.complete((torch.tensor(0), None))
            first_done.set()

        engine._async_compute = blocked_compute
        running = self._make_input("fake://cancel-running")
        queued = self._make_input("fake://cancel-queued")
        try:
            engine.async_submit([running], request_id=101)
            self.assertTrue(first_started.wait(timeout=2))
            engine.async_submit([queued], request_id=102)

            self.assertEqual(engine.cancel_queued_request(102), 1)
            self.assertIsNone(engine._embedding_cache.peek(queued.cache_key()))
            self.assertEqual(engine._async_admitted, 1)
            self.assertEqual(engine.cancel_queued_request(101), 0)

            release.set()
            self.assertTrue(first_done.wait(timeout=5))
            self.assertEqual(started_urls, [running.url])
            running_entry = engine._embedding_cache.peek(running.cache_key())
            self.assertIsNotNone(running_entry)
            self.assertTrue(running_entry.is_done)
        finally:
            release.set()
            engine.stop()

    def test_cancel_keeps_queued_work_owned_by_another_request(self):
        engine = self._make_engine(vit_concurrency=1, vit_max_queue_size=1)
        release = threading.Event()
        first_started = threading.Event()
        first_done = threading.Event()
        started_urls = []

        def blocked_compute(mm_inputs, cache_key, entry, request_id=0):
            started_urls.append(mm_inputs[0].url)
            first_started.set()
            release.wait(timeout=5)
            entry.complete((torch.tensor(0), None))
            first_done.set()

        engine._async_compute = blocked_compute
        running = self._make_input("fake://shared-running")
        shared = self._make_input("fake://shared-queued")
        try:
            engine.async_submit([running], request_id=201)
            self.assertTrue(first_started.wait(timeout=2))
            engine.async_submit([shared], request_id=202)
            engine.async_submit([shared], request_id=203)

            self.assertEqual(engine.cancel_queued_request(202), 0)
            self.assertIsNotNone(engine._embedding_cache.peek(shared.cache_key()))
            self.assertEqual(engine.cancel_queued_request(203), 1)
            self.assertIsNone(engine._embedding_cache.peek(shared.cache_key()))

            release.set()
            self.assertTrue(first_done.wait(timeout=5))
            self.assertEqual(started_urls, [running.url])
        finally:
            release.set()
            engine.stop()

    def test_already_cancelled_request_is_not_submitted(self):
        engine = self._make_engine(vit_concurrency=1, vit_max_queue_size=1)
        cancellation_event = threading.Event()
        cancellation_event.set()
        inp = self._make_input("fake://cancel-before-submit")
        try:
            with self.assertRaises(FtRuntimeException) as raised:
                engine.get_embedding_result(
                    [inp],
                    request_id=301,
                    cancellation_event=cancellation_event,
                )
            self.assertEqual(
                raised.exception.exception_type, ExceptionType.CANCELLED_ERROR
            )
            self.assertIsNone(engine._embedding_cache.peek(inp.cache_key()))
            self.assertEqual(engine._async_admitted, 0)
        finally:
            engine.stop()

    def test_get_submits_all_inputs_before_waiting(self):
        engine = self._make_engine(vit_concurrency=2, vit_max_queue_size=0)
        release = threading.Event()
        both_started = threading.Event()
        lock = threading.Lock()
        started_count = 0
        result = None
        error = None

        def blocked_compute(mm_inputs, cache_key, entry, request_id=0):
            nonlocal started_count
            with lock:
                started_count += 1
                if started_count == 2:
                    both_started.set()
            release.wait(timeout=5)
            entry.complete((torch.tensor(0), None))

        def get_results():
            nonlocal result, error
            try:
                result = engine.get_embedding_result(
                    [
                        self._make_input("fake://parallel-get-0"),
                        self._make_input("fake://parallel-get-1"),
                    ]
                )
            except Exception as caught:
                error = caught

        engine._async_compute = blocked_compute
        thread = threading.Thread(target=get_results)
        try:
            thread.start()
            self.assertTrue(both_started.wait(timeout=2))
            release.set()
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
            self.assertIsNone(error)
            self.assertEqual(len(result), 2)
        finally:
            release.set()
            thread.join(timeout=5)
            engine.stop()

    def test_multiple_inputs_independent(self):
        engine = self._make_engine()
        inp1 = self._make_input("./rtp_llm/multimodal/test/testdata/qwen2_vl/1.jpg")
        inp2 = self._make_input("./rtp_llm/multimodal/test/testdata/qwen2_vl/1.jpg")
        # Same URL → same cache key
        keys = engine.async_submit([inp1, inp2])
        self.assertEqual(len(keys), 2)
        self.assertEqual(keys[0], keys[1])

        results = engine.get_embedding_result([inp1, inp2])
        self.assertEqual(len(results), 2)
        engine.stop()

    def test_concurrent_get_same_key(self):
        engine = self._make_engine(FakeSlowEmbeddingInterface())
        inp = self._make_input("./rtp_llm/multimodal/test/testdata/qwen2_vl/1.jpg")
        results = [None, None]
        errors = [None, None]

        def worker(idx):
            try:
                results[idx] = engine.get_embedding_result([inp])
            except Exception as e:
                errors[idx] = e

        t0 = time.time()
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        elapsed = time.time() - t0

        for e in errors:
            self.assertIsNone(e)
        for r in results:
            self.assertIsNotNone(r)
            self.assertEqual(len(r), 1)
        # Both should finish in roughly one embedding time, not two
        self.assertLess(elapsed, FakeSlowEmbeddingInterface.delay * 2)
        engine.stop()

    def test_async_and_sync_same_key_share_one_embedding(self):
        class CountingSlowEmbedding(FakeMultiModalEmbeddingInterface):
            def __init__(self):
                super().__init__()
                self.calls = 0
                self.lock = threading.Lock()

            @torch.inference_mode()
            def embedding(self, data, **kwargs):
                with self.lock:
                    self.calls += 1
                time.sleep(0.2)
                return torch.tensor([[1.0]]), None

        mm_part = CountingSlowEmbedding()
        engine = self._make_engine(mm_part)
        inp = self._make_input("fake://sync-async-dedup")

        engine.async_submit([inp])
        sync_result = engine.mm_embedding_cpp(
            [inp.url],
            [MMUrlType.IMAGE],
            [torch.empty(0)],
            [[-1, -1, -1, -1, -1, -1, -1, [], 30000]],
        )
        async_result = engine.get_embedding_result([inp])

        self.assertEqual(mm_part.calls, 1)
        self.assertEqual(sync_result.embeddings[0].item(), 1.0)
        self.assertEqual(async_result[0].embeddings[0].item(), 1.0)
        self.assertGreaterEqual(engine._embedding_cache.stats()["inflight_dedup"], 1)
        engine.stop()

    @patch("rtp_llm.multimodal.mm_process_engine.kmonitor.report")
    def test_error_clears_cache(self, report):
        engine = self._make_engine(
            FakeMultiModalEmbeddingInterfacePreprocessException()
        )
        # Use a unique URL so the global vit_emb_cache_ won't have a hit
        # from earlier tests (which would skip preprocessing entirely).
        inp = self._make_input(
            "./rtp_llm/multimodal/test/testdata/qwen2_vl/1.jpg?error_test"
        )

        with self.assertRaises(PreprcoesException):
            engine.get_embedding_result([inp])

        # After error, cache entry should be removed — next call should re-attempt
        state, _ = engine._async_cache.try_acquire(inp.cache_key())
        self.assertEqual(state, "miss")
        error_reports = [
            call
            for call in report.call_args_list
            if call.args and call.args[0] == AccMetrics.VIT_ERROR_QPS_METRIC
        ]
        self.assertEqual(len(error_reports), 1)
        engine.stop()

    def test_empty_url_raises(self):
        engine = self._make_engine()
        inp = self._make_input("")
        with self.assertRaises(ValueError):
            engine.async_submit([inp])
        with self.assertRaises(ValueError):
            engine.get_embedding_result([inp])
        engine.stop()


# ----------------------------------------------------------------------------
# GreenNet (content safety) integration
# ----------------------------------------------------------------------------


class _StubGreenNetHandle(GreenNetHandle):
    def __init__(self, rewritten_inputs, verdict, delay=0.0):
        self.rewritten_inputs = rewritten_inputs
        self._verdict = verdict
        self._delay = delay
        self.cancelled = False

    async def wait_result(self) -> GreenNetVerdict:
        if self._delay:
            await asyncio.sleep(self._delay)
        return self._verdict

    def cancel(self) -> None:
        self.cancelled = True


class _StubGreenNetProvider(GreenNetProvider):
    """Records inputs and returns a programmable verdict. Optionally rewrites
    each input's url so we can assert the rewritten inputs reach ViT."""

    def __init__(self, verdict, rewrite_suffix=None, delay=0.0):
        self._verdict = verdict
        self._rewrite_suffix = rewrite_suffix
        self._delay = delay
        self.calls = 0
        self.last_handle = None
        self.request_ids = []

    def is_enabled(self) -> bool:
        return True

    async def preprocess_and_submit(self, request, mm_inputs):
        self.calls += 1
        self.request_ids.append(str(request.id))
        if self._rewrite_suffix is not None:
            rewritten = [
                MultimodalInput(
                    mi.url + self._rewrite_suffix,
                    mi.mm_type,
                    torch.empty(0),
                    mi.mm_preprocess_config,
                )
                for mi in mm_inputs
            ]
        else:
            rewritten = list(mm_inputs)
        handle = _StubGreenNetHandle(rewritten, self._verdict, self._delay)
        self.last_handle = handle
        return handle


class _UrlRecordingEmbedding(FakeMultiModalEmbeddingInterface):
    """Records the urls preprocess_input actually received (to verify the
    greennet-rewritten inputs are what ViT consumes)."""

    seen_urls: List[str] = []

    @staticmethod
    def preprocess_input(mm_inputs, vit_config, **kwargs):
        _UrlRecordingEmbedding.seen_urls.extend(mi.url for mi in mm_inputs)
        return mm_inputs, kwargs


class MMProcessEngineGreenNetTest(TestCase):
    def _make_engine(self, mm_part=None):
        model = FakeModel(mm_part or FakeMultiModalEmbeddingInterface())
        vit_config = VitConfig()
        vit_config.use_local_preprocess = True
        return MMProcessEngine(
            model.mm_part,
            model.model_config,
            vit_config,
            ProfilingDebugLoggingConfig(),
        )

    def _make_input(self, url):
        return MultimodalInput(
            url,
            MMUrlType.IMAGE,
            torch.empty(0),
            MMPreprocessConfig(-1, -1, -1, -1, -1, -1, -1, [], 30000),
        )

    def test_default_provider_is_noop(self):
        # No internal_source in the open-source test env → no-op provider,
        # so greennet is disabled and the engine behaves exactly as before.
        engine = self._make_engine()
        self.assertFalse(engine._greennet_enabled())
        verdict = engine.wait_greennet_verdict(
            [self._make_input("./rtp_llm/multimodal/test/testdata/qwen2_vl/1.jpg")]
        )
        self.assertTrue(verdict.passed)
        engine.stop()

    def test_request_id_reaches_greennet_and_vit_access_logs(self):
        engine = self._make_engine()
        provider = _StubGreenNetProvider(GreenNetVerdict(passed=True, code=1))
        engine._greennet_provider = provider
        engine._access_logger = MagicMock(spec=MMAccessLogger)
        request_id = 987654321

        engine.mm_embedding_cpp(
            ["./rtp_llm/multimodal/test/testdata/qwen2_vl/1.jpg?request_id"],
            [MMUrlType.IMAGE],
            [torch.empty(0)],
            [[-1, -1, -1, -1, -1, -1, -1, [], 30000]],
            request_id,
        )

        self.assertEqual(provider.request_ids, [str(request_id)])
        self.assertEqual(
            engine._access_logger.log_query_access.call_args.args[1], request_id
        )
        self.assertEqual(
            engine._access_logger.log_success_access.call_args.args[2], request_id
        )
        engine.stop()

    def test_local_path_passes_when_verdict_passes(self):
        engine = self._make_engine()
        engine._greennet_provider = _StubGreenNetProvider(
            GreenNetVerdict(passed=True, code=1)
        )
        res = engine.mm_embedding_cpp(
            ["./rtp_llm/multimodal/test/testdata/qwen2_vl/1.jpg?gn_pass"],
            [MMUrlType.IMAGE],
            [torch.empty(0)],
            [[-1, -1, -1, -1, -1, -1, -1, [], 30000]],
        )
        self.assertEqual(res.embeddings, [torch.tensor(0)])
        engine.stop()

    def test_local_path_raises_when_verdict_fails(self):
        engine = self._make_engine()
        engine._greennet_provider = _StubGreenNetProvider(
            GreenNetVerdict(passed=False, code=2, message="blocked")
        )
        with self.assertRaises(FtRuntimeException) as ctx:
            engine.mm_embedding_cpp(
                ["./rtp_llm/multimodal/test/testdata/qwen2_vl/1.jpg?gn_fail"],
                [MMUrlType.IMAGE],
                [torch.empty(0)],
                [[-1, -1, -1, -1, -1, -1, -1, [], 30000]],
            )
        self.assertEqual(
            ctx.exception.exception_type, ExceptionType.UNSAFE_INPUT_CONTENT
        )
        self.assertIn("blocked", ctx.exception.message)
        engine.stop()

    def test_rewritten_inputs_reach_vit(self):
        _UrlRecordingEmbedding.seen_urls = []
        engine = self._make_engine(_UrlRecordingEmbedding())
        engine._greennet_provider = _StubGreenNetProvider(
            GreenNetVerdict(passed=True, code=1), rewrite_suffix="#rewritten"
        )
        engine.mm_embedding_cpp(
            ["./rtp_llm/multimodal/test/testdata/qwen2_vl/1.jpg?gn_rw"],
            [MMUrlType.IMAGE],
            [torch.empty(0)],
            [[-1, -1, -1, -1, -1, -1, -1, [], 30000]],
        )
        self.assertTrue(
            any(u.endswith("#rewritten") for u in _UrlRecordingEmbedding.seen_urls),
            f"ViT did not see rewritten url: {_UrlRecordingEmbedding.seen_urls}",
        )
        engine.stop()

    def test_rewritten_sync_and_async_share_original_key(self):
        class SlowRecordingEmbedding(_UrlRecordingEmbedding):
            calls = 0
            lock = threading.Lock()

            @torch.inference_mode()
            def embedding(self, data, **kwargs):
                with self.lock:
                    self.calls += 1
                time.sleep(0.2)
                return torch.tensor([[1.0]]), None

        _UrlRecordingEmbedding.seen_urls = []
        part = SlowRecordingEmbedding()
        engine = self._make_engine(part)
        provider = _StubGreenNetProvider(
            GreenNetVerdict(passed=True, code=1), rewrite_suffix="#rewritten"
        )
        engine._greennet_provider = provider
        inp = self._make_input(
            "./rtp_llm/multimodal/test/testdata/qwen2_vl/1.jpg?gn_dedup"
        )

        engine.async_submit([inp])
        sync_result = engine.mm_embedding_cpp(
            [inp.url],
            [MMUrlType.IMAGE],
            [torch.empty(0)],
            [[-1, -1, -1, -1, -1, -1, -1, [], 30000]],
        )
        async_result = engine.get_embedding_result([inp])

        self.assertEqual(part.calls, 1)
        self.assertEqual(provider.calls, 1)
        self.assertEqual(sync_result.embeddings[0].item(), 1.0)
        self.assertEqual(async_result[0].embeddings[0].item(), 1.0)
        self.assertTrue(
            any(u.endswith("#rewritten") for u in _UrlRecordingEmbedding.seen_urls)
        )
        engine.stop()

    def test_wait_verdict_pass_after_async_submit(self):
        engine = self._make_engine()
        engine._greennet_provider = _StubGreenNetProvider(
            GreenNetVerdict(passed=True, code=1)
        )
        inp = self._make_input(
            "./rtp_llm/multimodal/test/testdata/qwen2_vl/1.jpg?gn_wait_pass"
        )
        engine.async_submit([inp])
        verdict = engine.wait_greennet_verdict([inp])
        self.assertTrue(verdict.passed)
        engine.stop()

    @patch("rtp_llm.multimodal.mm_process_engine.kmonitor.report")
    def test_wait_verdict_fail_after_async_submit(self, report):
        engine = self._make_engine()
        engine._greennet_provider = _StubGreenNetProvider(
            GreenNetVerdict(passed=False, code=2, message="nsfw")
        )
        inp = self._make_input(
            "./rtp_llm/multimodal/test/testdata/qwen2_vl/1.jpg?gn_wait_fail"
        )
        engine.async_submit([inp])
        verdict = engine.wait_greennet_verdict([inp])
        self.assertFalse(verdict.passed)
        self.assertEqual(verdict.code, 2)
        # The embedding entry must also surface the violation.
        with self.assertRaises(FtRuntimeException) as ctx:
            engine.get_embedding_result([inp])
        self.assertEqual(
            ctx.exception.exception_type, ExceptionType.UNSAFE_INPUT_CONTENT
        )
        self.assertTrue(
            any(
                call.args and call.args[0] == AccMetrics.VIT_ERROR_QPS_METRIC
                for call in report.call_args_list
            )
        )
        engine.stop()

    def test_wait_verdict_kicks_compute_on_miss(self):
        # wait_greennet_verdict called without a prior async_submit must still
        # produce a verdict (kick compute itself).
        engine = self._make_engine()
        engine._greennet_provider = _StubGreenNetProvider(
            GreenNetVerdict(passed=False, code=2, message="bad")
        )
        inp = self._make_input(
            "./rtp_llm/multimodal/test/testdata/qwen2_vl/1.jpg?gn_miss"
        )
        verdict = engine.wait_greennet_verdict([inp])
        self.assertFalse(verdict.passed)
        engine.stop()


class MMAccessLoggerRequestIdTest(TestCase):
    def test_request_id_is_serialized_at_top_level(self):
        access_logger = MMAccessLogger.__new__(MMAccessLogger)
        access_logger.query_logger = MagicMock()
        mm_input = MagicMock()
        mm_input.to_string.return_value = "image://test"

        access_logger.log_query_access([mm_input], request_id=123456)

        payload = json.loads(access_logger.query_logger.info.call_args.args[0])
        self.assertEqual(payload["id"], 123456)
        self.assertEqual(payload["query"], ["image://test"])


_DEFAULT_CONFIG = [-1, -1, -1, -1, -1, -1, -1, [], 30000]


class FakeBatchMMPart(MultiModalEmbeddingInterface):
    """mm_part returning identity-encoded (emb, pos, extra) tuples.

    Each input carries an index in its url ("fake://<i>"); embedding echoes that
    index into all three output tensors so tests can assert ordering, and counts
    embedding/batched_embedding invocations to observe cache hits and batching.
    """

    def __init__(self):
        self.embedding_calls = 0
        self.batch_sizes: List[int] = []
        self._lock = threading.Lock()

    @staticmethod
    def preprocess_input(mm_inputs, vit_config, **kwargs):
        # Carry the inputs through; embedding derives identity from the url.
        return mm_inputs, kwargs

    def get_preprocess_params(self):
        return {}

    @torch.inference_mode()
    def embedding(self, data, **kwargs):
        mm_inputs, _ = data
        idx = float(int(mm_inputs[0].url.split("://")[1]))
        with self._lock:
            self.embedding_calls += 1
        emb = torch.tensor([[idx]])  # (1, 1) -> one embedding per work item
        pos = torch.tensor([[idx]])  # (1, 1)
        extra = torch.tensor([idx])  # (1,) -> one flat extra tensor
        return emb, pos, extra

    def batched_embedding(self, data_list, mm_types, **kwargs):
        with self._lock:
            self.batch_sizes.append(len(data_list))
        return super().batched_embedding(data_list, mm_types, **kwargs)


class MMProcessEngineGpuBatchTest(TestCase):
    def setUp(self):
        # vit_emb_cache_ is a process-global; isolate it so cache state never
        # leaks between these tests (or into other test classes in the process).
        vit_emb_cache_.resize_cache(0)

    def tearDown(self):
        vit_emb_cache_.resize_cache(0)

    def _make_engine(self, **vit_overrides):
        model = FakeModel(FakeBatchMMPart())
        vit_config = VitConfig()
        vit_config.use_gpu_batch = True
        # Local preprocess keeps the test in-process and deterministic.
        vit_config.use_local_preprocess = True
        # Cache off by default; the cache test opts in explicitly.
        vit_config.mm_cache_item_num = 0
        for key, value in vit_overrides.items():
            setattr(vit_config, key, value)
        engine = MMProcessEngine(
            model.mm_part,
            model.model_config,
            vit_config,
            ProfilingDebugLoggingConfig(),
        )
        self.addCleanup(engine.stop)
        return engine, model.mm_part

    def _embed(self, engine, urls):
        n = len(urls)
        return engine.mm_embedding_cpp(
            urls,
            [MMUrlType.IMAGE] * n,
            [torch.empty(0)] * n,
            [list(_DEFAULT_CONFIG) for _ in range(n)],
        )

    def test_gpu_batch_order_and_outputs(self):
        """Single multi-image request: emb/pos/extra preserve input order."""
        engine, _ = self._make_engine()
        urls = [f"fake://{i}" for i in range(4)]
        res = self._embed(engine, urls)

        self.assertEqual([e.item() for e in res.embeddings], [0, 1, 2, 3])
        self.assertEqual([p.item() for p in res.position_ids], [0, 1, 2, 3])
        self.assertEqual([x.item() for x in res.extra_input], [0, 1, 2, 3])

    def test_gpu_batch_multi_request(self):
        """Concurrent requests are batched yet each gets its own correct result."""
        engine, part = self._make_engine(gpu_batch_wait_ms=400, gpu_max_batch_size=16)
        n = 5
        results: List[float] = [None] * n

        def run(i: int):
            res = self._embed(engine, [f"fake://{i}"])
            results[i] = res.embeddings[0].item()

        threads = [threading.Thread(target=run, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(results, [0.0, 1.0, 2.0, 3.0, 4.0])
        # The wait window should let at least one forward serve >1 request.
        self.assertGreaterEqual(max(part.batch_sizes), 2)

    def test_gpu_batch_cache_hit(self):
        """A repeated url is served from cache without a second embedding call."""
        engine, part = self._make_engine(mm_cache_item_num=10)
        # tearDown restores the global cache to disabled for other tests.

        url = "fake://7"
        r1 = self._embed(engine, [url])
        r2 = self._embed(engine, [url])

        self.assertEqual(r1.embeddings[0].item(), 7)
        self.assertEqual(r2.embeddings[0].item(), 7)
        self.assertEqual(part.embedding_calls, 1)


class MiniMaxM3VLPreprocessTest(TestCase):
    def assert_mm_error(self, callable_, message):
        with self.assertRaises(FtRuntimeException) as context:
            callable_()
        self.assertEqual(
            context.exception.exception_type, ExceptionType.MM_WRONG_FORMAT_ERROR
        )
        self.assertEqual(context.exception.message, message)

    def test_default_file_size_limits(self):
        config = VitConfig()
        self.assertEqual(config.mm_image_max_file_size_kb, 100 * 1024)
        self.assertEqual(config.mm_video_max_file_size_kb, 2 * 1024 * 1024)

    def test_image_open_error(self):
        from rtp_llm.multimodal.multimodal_mixins.minimax_m3_vl.minimax_m3_vl_mixin import (
            MiniMaxM3VLImageEmbedding,
        )

        mm_input = SimpleNamespace(url="invalid-image")
        config = VitConfig()
        with patch(
            "rtp_llm.multimodal.multimodal_mixins.minimax_m3_vl."
            "minimax_m3_vl_mixin.get_bytes_io_from_url",
            return_value=io.BytesIO(b"not an image"),
        ):
            self.assert_mm_error(
                lambda: MiniMaxM3VLImageEmbedding._preprocess_image(
                    mm_input, config, SimpleNamespace()
                ),
                MMErr.IMG_OPEN,
            )

    def test_image_resize_accepts_small_and_extreme_aspect_inputs(self):
        from rtp_llm.multimodal.multimodal_mixins.minimax_m3_vl.image_processor import (
            smart_resize,
        )

        self.assertEqual(smart_resize(9, 100), (28, 112))
        self.assertEqual(smart_resize(10, 3000), (28, 2996))

    def test_image_resize_too_small_after_max_pixels(self):
        from rtp_llm.multimodal.multimodal_mixins.minimax_m3_vl.image_processor import (
            smart_resize,
        )

        self.assert_mm_error(
            lambda: smart_resize(100, 100, max_pixels=1),
            MMErr.IMG_TOO_SMALL,
        )

    def test_minimax_long_side_resize_rules(self):
        from rtp_llm.multimodal.multimodal_mixins.minimax_m3_vl.image_processor import (
            IMAGE_MAX_TOTAL_PIXELS,
            MIN_SHORT_SIDE_PIXEL,
            smart_resize,
        )

        self.assertEqual(
            smart_resize(
                2048,
                1024,
                factor=28,
                max_long_side_pixel=1008,
                max_total_pixels=IMAGE_MAX_TOTAL_PIXELS,
            ),
            (1008, 504),
        )
        resized = smart_resize(
            200,
            40,
            factor=28,
            max_long_side_pixel=1008,
            max_total_pixels=IMAGE_MAX_TOTAL_PIXELS,
        )
        self.assertEqual(min(resized), MIN_SHORT_SIDE_PIXEL)

        with self.assertRaises(FtRuntimeException) as context:
            smart_resize(
                4000,
                4000,
                factor=28,
                max_long_side_pixel=4000,
                max_total_pixels=IMAGE_MAX_TOTAL_PIXELS,
            )
        self.assertIn("exceeds max_total_pixels", context.exception.message)

        with self.assertRaises(FtRuntimeException) as context:
            smart_resize(
                4000,
                4000,
                factor=28,
                max_pixels=20_000_000,
                max_total_pixels=IMAGE_MAX_TOTAL_PIXELS,
            )
        self.assertIn("exceeds max_total_pixels", context.exception.message)

    def test_minimax_request_media_limits_and_fps_range(self):
        from rtp_llm.multimodal.multimodal_mixins.minimax_m3_vl.minimax_m3_vl_mixin import (
            MiniMaxM3VLImageEmbedding,
        )

        embedding = object.__new__(MiniMaxM3VLImageEmbedding)

        def make_input(mm_type, fps=-1.0):
            return SimpleNamespace(
                mm_type=mm_type,
                mm_preprocess_config=SimpleNamespace(fps=fps),
            )

        embedding.validate_inputs(
            [make_input(MMUrlType.IMAGE) for _ in range(200)]
            + [make_input(MMUrlType.VIDEO, 0.2) for _ in range(10)]
            + [make_input(MMUrlType.VIDEO, 5.0) for _ in range(10)]
        )

        for inputs, expected in (
            (
                [make_input(MMUrlType.IMAGE) for _ in range(201)],
                "at most 200 images",
            ),
            (
                [make_input(MMUrlType.VIDEO) for _ in range(21)],
                "at most 20 videos",
            ),
            ([make_input(MMUrlType.VIDEO, 0.19)], "fps must be in [0.2, 5.0]"),
            ([make_input(MMUrlType.VIDEO, 5.01)], "fps must be in [0.2, 5.0]"),
        ):
            with self.subTest(expected=expected), self.assertRaises(
                FtRuntimeException
            ) as context:
                embedding.validate_inputs(inputs)
            self.assertIn(expected, context.exception.message)

    def _run_video_preprocess(
        self,
        *,
        total_frames,
        video_fps,
        requested_fps=0,
        video_reader_error=None,
        configured_max_frames=64,
        source_height=28,
        source_width=28,
        max_long_side_pixel=-1,
    ):
        from rtp_llm.multimodal.multimodal_mixins.minimax_m3_vl.minimax_m3_vl_mixin import (
            MiniMaxM3VLImageEmbedding,
        )

        captured = {}

        class _VideoReader:
            def __init__(self, _data, width=None, height=None, **_kwargs):
                if video_reader_error is not None:
                    raise video_reader_error
                self.width = width or source_width
                self.height = height or source_height

            def __len__(self):
                return total_frames

            def get_avg_fps(self):
                return video_fps

            def __getitem__(self, _index):
                return SimpleNamespace(shape=(source_height, source_width, 3))

            def get_batch(self, indices):
                captured.setdefault("indices", []).extend(indices)
                return torch.zeros(
                    (len(indices), self.height, self.width, 3),
                    dtype=torch.uint8,
                )

        decord = types.ModuleType("decord")
        decord.VideoReader = _VideoReader
        decord.bridge = SimpleNamespace(set_bridge=lambda _name: None)

        mm_input = SimpleNamespace(
            url="video",
            mm_preprocess_config=SimpleNamespace(
                fps=requested_fps,
                max_frames=0,
                max_long_side_pixel=max_long_side_pixel,
            ),
        )
        config = VitConfig()
        config.mm_video_max_frames = configured_max_frames
        processor = SimpleNamespace(patch_size=14, max_pixels=451584)
        tokenizer = SimpleNamespace(encode=lambda *_args, **_kwargs: [1])

        with patch.dict(sys.modules, {"decord": decord}), patch(
            "rtp_llm.multimodal.multimodal_mixins.minimax_m3_vl."
            "minimax_m3_vl_mixin.get_bytes_io_from_url",
            return_value=io.BytesIO(b"video"),
        ):
            result = MiniMaxM3VLImageEmbedding._preprocess_video(
                mm_input,
                config,
                processor,
                tokenizer,
                video_fps=1.0,
                video_max_frames=768,
                merge_size=2,
                temporal_patch_size=2,
            )
        return result, captured

    def test_invalid_video(self):
        self.assert_mm_error(
            lambda: self._run_video_preprocess(
                total_frames=0,
                video_fps=0,
                video_reader_error=OSError("invalid"),
            ),
            MMErr.VIDEO_INVALID,
        )

    def test_video_duration_is_not_rejected(self):
        for total_frames in (30, 1500):
            with self.subTest(total_frames=total_frames):
                result, captured = self._run_video_preprocess(
                    total_frames=total_frames, video_fps=30
                )
                frames, _, _ = result
                self.assertGreater(frames.shape[0], 0)
                self.assertGreater(len(captured["indices"]), 0)

    def test_video_sampling_is_capped_at_64_frames(self):
        result, captured = self._run_video_preprocess(
            total_frames=1200,
            video_fps=30,
            requested_fps=5,
        )
        frames, _, timestamps = result
        self.assertEqual(len(captured["indices"]), 64)
        self.assertEqual(frames.shape[0], 64)
        self.assertEqual(len(timestamps), 32)

    def test_video_sampling_supports_single_frame_limit(self):
        _, captured = self._run_video_preprocess(
            total_frames=1200,
            video_fps=30,
            requested_fps=5,
            configured_max_frames=1,
        )
        self.assertEqual(captured["indices"], [0])

    def test_video_sampling_supports_fractional_fps(self):
        _, captured = self._run_video_preprocess(
            total_frames=300,
            video_fps=30,
            requested_fps=0.2,
        )
        self.assertEqual(captured["indices"], [0, 150, 299])

    def test_video_total_pixels_are_rejected_before_batch_decode(self):
        with self.assertRaises(FtRuntimeException) as context:
            self._run_video_preprocess(
                total_frames=1200,
                video_fps=30,
                requested_fps=5,
                source_height=3584,
                source_width=3584,
                max_long_side_pixel=3584,
            )
        self.assertIn("exceeds max_total_pixels", context.exception.message)


if __name__ == "__main__":
    main()
