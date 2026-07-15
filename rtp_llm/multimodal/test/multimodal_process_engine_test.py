import asyncio
import concurrent.futures
import io
import os
import sys
import threading
import time
import types
from types import SimpleNamespace
from typing import List
from unittest import TestCase, main
from unittest.mock import patch

import PIL
import pillow_avif
import pillow_heif
import torch
from PIL import Image, ImageFile

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

    def test_timeout(self):
        model = FakeModel(FakeMultiModalEmbeddingInterfaceSlow())
        engine = MMProcessEngine(
            model.mm_part,
            model.model_config,
            VitConfig(),
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
        mm_process_engine = MMProcessEngine(
            model.mm_part,
            model.model_config,
            VitConfig(),
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
        engine = MMProcessEngine(
            model.mm_part,
            model.model_config,
            VitConfig(),
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


class AsyncSubmitGetEmbeddingTest(TestCase):
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

    def test_error_clears_cache(self):
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

    def is_enabled(self) -> bool:
        return True

    async def preprocess_and_submit(self, request, mm_inputs):
        self.calls += 1
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

    def test_wait_verdict_fail_after_async_submit(self):
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
        self.assertEqual(config.mm_image_min_dimension, 10)
        self.assertEqual(config.mm_image_max_aspect_ratio, 200.0)

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

    def test_image_dimension_errors(self):
        from rtp_llm.multimodal.multimodal_mixins.minimax_m3_vl.image_processor import (
            smart_resize,
        )

        self.assert_mm_error(
            lambda: smart_resize(9, 100),
            MMErr.IMG_HW.format("height:9 or width:100 must be larger than 10"),
        )
        self.assert_mm_error(
            lambda: smart_resize(100, 100, max_pixels=1),
            MMErr.IMG_TOO_SMALL,
        )

    def test_image_dimension_limits_are_configurable(self):
        from rtp_llm.multimodal.multimodal_mixins.minimax_m3_vl.image_processor import (
            smart_resize,
        )

        self.assert_mm_error(
            lambda: smart_resize(19, 100, min_image_dimension=20),
            MMErr.IMG_HW.format("height:19 or width:100 must be larger than 20"),
        )
        self.assert_mm_error(
            lambda: smart_resize(20, 100, max_image_aspect_ratio=4),
            MMErr.IMG_HW.format(
                "absolute aspect ratio must be smaller than 4, got 20 / 100"
            ),
        )

    def _run_video_preprocess(
        self,
        *,
        total_frames,
        video_fps,
        requested_fps=0,
        video_reader_error=None,
        configured_max_frames=64,
    ):
        from rtp_llm.multimodal.multimodal_mixins.minimax_m3_vl.minimax_m3_vl_mixin import (
            MiniMaxM3VLImageEmbedding,
        )

        captured = {}

        class _VideoReader:
            def __init__(self, _data):
                if video_reader_error is not None:
                    raise video_reader_error

            def __len__(self):
                return total_frames

            def get_avg_fps(self):
                return video_fps

            def get_batch(self, indices):
                captured["indices"] = list(indices)
                return torch.zeros(
                    (len(indices), 28, 28, 3),
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
            requested_fps=30,
        )
        frames, _, timestamps = result
        self.assertEqual(len(captured["indices"]), 64)
        self.assertEqual(frames.shape[0], 64)
        self.assertEqual(len(timestamps), 32)

    def test_video_sampling_supports_single_frame_limit(self):
        _, captured = self._run_video_preprocess(
            total_frames=1200,
            video_fps=30,
            requested_fps=30,
            configured_max_frames=1,
        )
        self.assertEqual(captured["indices"], [0])


if __name__ == "__main__":
    main()
