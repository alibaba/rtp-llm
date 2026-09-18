import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.py_config_modules import ProfilingDebugLoggingConfig, VitConfig
from rtp_llm.multimodal.kvcm.tensor_object import pack_object, plan_object
from rtp_llm.multimodal.mm_embedding_cache import MMEmbeddingCache
from rtp_llm.multimodal.mm_process_engine import MMProcessEngine
from rtp_llm.multimodal.mm_remote_embedding_cache import MMRemoteEmbeddingCache
from rtp_llm.multimodal.test.multimodal_process_engine_test import (
    FakeEmbeddingLengthInterface,
)
from rtp_llm.ops import MMPreprocessConfig, MultimodalInput


class Store:
    def __init__(self, objects=None):
        self.objects = {} if objects is None else objects
        self.saved = threading.Event()
        self.read_started = threading.Event()
        self.read_gate = threading.Event()
        self.read_gate.set()
        self.write_gate = threading.Event()
        self.write_gate.set()
        self.closed = False
        self.fail_read = self.fail_write = False
        self.instance_id = "kve_test"
        self.instance_group = "kve_test_group"

    def save_one(self, key, value):
        self.write_gate.wait(3)
        if self.fail_write:
            raise RuntimeError("unknown save outcome")
        self.objects.setdefault(key, value.clone())
        self.saved.set()

    def object_size(self, key, **kwargs):
        value = self.objects.get(key)
        return None if value is None else value.numel()

    def load_one(self, key, value):
        self.read_started.set()
        self.read_gate.wait(3)
        if self.fail_read:
            raise RuntimeError("read unavailable")
        value.copy_(self.objects[key])

    def close(self):
        self.closed = True


class RemoteCacheTest(unittest.TestCase):
    def remote(self, store, **kwargs):
        options = dict(
            max_object_bytes=65536,
            max_inflight_bytes=1024 * 1024,
            max_pending=8,
            read_timeout_ms=500,
        )
        options.update(kwargs)
        result = MMRemoteEmbeddingCache(store, **options)
        self.addCleanup(result.close)
        return result

    def test_cpu_eviction_whole_result_shared_with_second_worker(self):
        store = Store()
        writer = self.remote(store)
        cache = MMEmbeddingCache(0, 128, on_cpu_evict=writer.submit_eviction)
        result = (
            torch.arange(8, dtype=torch.float32).reshape(2, 4),
            torch.arange(4).reshape(2, 2),
            [torch.tensor([7], dtype=torch.int32), torch.tensor([0.5])],
        )
        # Tensor payload and pool alignment both fit one entry, not two.
        for key, value in [("url_config_a", result), ("url_config_b", result)]:
            state, entry = cache.try_acquire(key)
            self.assertEqual(state, "miss")
            cache.complete(key, entry, value)
        self.assertTrue(store.saved.wait(2))
        self.assertNotIn("url_config_a", cache.resident_tiers())
        reader = self.remote(Store(store.objects))
        loaded = reader.load("url_config_a")
        self.assertIsInstance(loaded, tuple)
        self.assertIsInstance(loaded[2], list)
        for actual, expected in [
            (loaded[0], result[0]),
            (loaded[1], result[1]),
            (loaded[2][0], result[2][0]),
            (loaded[2][1], result[2][1]),
        ]:
            self.assertEqual(actual.dtype, expected.dtype)
            self.assertTrue(torch.equal(actual, expected))
        self.assertIsNone(reader.load("url_config_missing"))
        self.assertEqual(reader.stats()["hit"], 1)
        cache.clear()

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_gpu_cpu_remote_gpu_survives_pool_reuse_and_source_mutation(self):
        store = Store()
        store.write_gate.clear()
        writer = self.remote(store)
        cache = MMEmbeddingCache(128, 128, on_cpu_evict=writer.submit_eviction)
        self.addCleanup(cache.clear)
        source = (
            torch.arange(16, dtype=torch.float32, device="cuda").reshape(4, 4),
            torch.arange(4, dtype=torch.int64, device="cuda"),
            [torch.arange(8, dtype=torch.float32, device="cuda")],
        )
        expected = [source[0].cpu(), source[1].cpu(), source[2][0].cpu()]
        for index in range(5):
            key = f"existing_key_{index}"
            _, entry = cache.try_acquire(key)
            cache.complete(key, entry, source)
        self.assertNotIn("existing_key_0", cache.resident_tiers())
        # Pool slots and the original producer tensors can be reused while
        # the remote write still waits; its detached snapshot must stay intact.
        for tensor in (source[0], source[1], source[2][0]):
            tensor.zero_()
        store.write_gate.set()
        self.assertTrue(store.saved.wait(2))
        writer.close()
        reader = self.remote(Store(store.objects), cuda_device=torch.device("cuda", 0))
        loaded = reader.load("existing_key_0")
        self.assertIsNotNone(loaded)
        for actual, value in zip([loaded[0], loaded[1], loaded[2][0]], expected):
            self.assertEqual(actual.device, torch.device("cuda", 0))
            self.assertEqual(actual.dtype, value.dtype)
            self.assertTrue(torch.equal(actual.cpu(), value))

    def test_timeout_retains_budget_until_actual_io_completion(self):
        store = Store({"key": pack_object(plan_object((torch.ones(3, 4), None)))})
        store.read_gate.clear()
        remote = self.remote(store, read_timeout_ms=15)
        self.assertIsNone(remote.load("key"))
        self.assertTrue(store.read_started.is_set())
        self.assertGreater(remote.stats()["inflight_bytes"], 0)
        store.read_gate.set()
        remote.close()
        self.assertEqual(remote.stats()["inflight_bytes"], 0)
        self.assertEqual(remote.stats()["inflight_tasks"], 0)

    def test_failed_or_corrupt_read_is_not_published(self):
        data = pack_object(plan_object((torch.ones(3, 4), None)))
        store = Store({"key": data})
        remote = self.remote(store)
        store.fail_read = True
        self.assertIsNone(remote.load("key"))
        store.fail_read = False
        data[-1] ^= 1
        self.assertIsNone(remote.load("key"))
        self.assertEqual(remote.stats()["read_error"], 2)
        self.assertEqual(remote.stats()["inflight_bytes"], 0)

    def test_write_budget_dedup_and_unknown_failure_do_not_remove(self):
        store = Store()
        store.write_gate.clear()
        remote = self.remote(store, max_pending=1)
        entry = SimpleNamespace(
            pool_owners=[],
            result=(torch.ones(2, 4), None),
            error=None,
            original_devices=(torch.device("cpu"), None),
        )
        remote.submit_eviction("same", entry)
        remote.submit_eviction("same", entry)
        remote.submit_eviction("different", entry)
        self.assertEqual(remote.stats()["inflight_tasks"], 1)
        self.assertEqual(remote.stats()["admission_skip"], 1)
        store.fail_write = True
        store.write_gate.set()
        remote.close()
        self.assertEqual(remote.stats()["write_error"], 1)
        self.assertEqual(remote.stats()["inflight_bytes"], 0)
        self.assertEqual(store.objects, {})

    def test_clear_does_not_upload_or_delete_shared_values(self):
        store = Store()
        remote = self.remote(store)
        cache = MMEmbeddingCache(0, 256, on_cpu_evict=remote.submit_eviction)
        _, entry = cache.try_acquire("key")
        cache.complete("key", entry, (torch.ones(2, 4), None))
        cache.clear()
        self.assertEqual(remote.stats()["inflight_tasks"], 0)
        self.assertEqual(store.objects, {})

    def test_oversized_read_and_pinned_snapshot_bypass(self):
        store = Store({"key": torch.empty(1024, dtype=torch.uint8)})
        remote = self.remote(store, max_object_bytes=128)
        self.assertIsNone(remote.load("key"))
        self.assertFalse(store.read_started.is_set())
        entry = SimpleNamespace(pool_owners=[object()], result=(torch.ones(2), None))
        remote.submit_eviction("key", entry)
        self.assertEqual(remote.stats()["inflight_tasks"], 0)


class EngineRemoteCacheTest(unittest.TestCase):
    def engine(self, store):
        config = VitConfig()
        config.use_local_preprocess = True
        config.mm_cache_gpu_max_bytes = 0
        config.mm_cache_cpu_max_bytes = 1024
        config.mm_hash_key_cache_max_bytes = 4096
        config.mm_remote_cache_enable = True
        config.mm_remote_cache_read_timeout_ms = 500
        model = ModelConfig()
        model.mm_related_params.preprocess_batch_size = 1
        interface = FakeEmbeddingLengthInterface()
        with patch(
            "rtp_llm.multimodal.kvcm.RtpKvMetaObjectClient.from_env", return_value=store
        ):
            engine = MMProcessEngine(
                interface, model, config, ProfilingDebugLoggingConfig()
            )
        self.addCleanup(engine.stop)
        return engine

    def test_sync_hit_preserves_extras_and_skips_preprocess_and_forward(self):
        mm_input = MultimodalInput(
            "existing_input_key_url",
            1,
            torch.empty(0),
            MMPreprocessConfig(-1, -1, -1, -1, -1.0, -1, -1, [], 30000),
        )
        result = (
            torch.arange(12, dtype=torch.float32).reshape(3, 4),
            torch.arange(6).reshape(3, 2),
            [torch.tensor([10, 20], dtype=torch.int32)],
        )
        store = Store({mm_input.cache_key(): pack_object(plan_object(result))})
        engine = self.engine(store)
        with patch.object(
            engine.preprocess_executor,
            "preprocess_func",
            side_effect=AssertionError("preprocess called"),
        ), patch.object(
            engine._scheduler,
            "submit_and_wait",
            side_effect=AssertionError("forward called"),
        ):
            response = engine.mm_embedding_impl([mm_input])
        self.assertTrue(torch.equal(response.embeddings[0], result[0]))
        self.assertTrue(torch.equal(response.position_ids[0], result[1]))
        self.assertTrue(torch.equal(response.extra_input[0], result[2][0]))
        self.assertEqual(engine._remote_embedding_cache.stats()["hit"], 1)

    def test_async_remote_failure_falls_back_to_normal_compute(self):
        store = Store()
        engine = self.engine(store)
        mm_input = MultimodalInput(
            "abcd",
            1,
            torch.empty(0),
            MMPreprocessConfig(-1, -1, -1, -1, -1.0, -1, -1, [], 30000),
        )
        response = engine.get_embedding_result([mm_input], request_id=101)
        self.assertEqual(response[0].embeddings[0].shape, (4, 4))
        self.assertEqual(engine._remote_embedding_cache.stats()["miss"], 1)

    def test_remote_hit_does_not_bypass_greennet_rejection(self):
        from rtp_llm.config.exceptions import FtRuntimeException
        from rtp_llm.multimodal.greennet_hook import GreenNetVerdict
        from rtp_llm.multimodal.test.multimodal_process_engine_test import (
            _StubGreenNetProvider,
        )

        mm_input = MultimodalInput(
            "rejected",
            1,
            torch.empty(0),
            MMPreprocessConfig(-1, -1, -1, -1, -1.0, -1, -1, [], 30000),
        )
        store = Store(
            {mm_input.cache_key(): pack_object(plan_object((torch.ones(3, 4), None)))}
        )
        engine = self.engine(store)
        engine._greennet_provider = _StubGreenNetProvider(
            GreenNetVerdict(passed=False, code=2, message="blocked"), delay=0.05
        )
        with self.assertRaises(FtRuntimeException):
            engine.get_embedding_result([mm_input], request_id=202, user_id="test-user")
        self.assertIsNone(engine._embedding_cache.peek(mm_input.cache_key()))
        self.assertFalse(engine._hash_key_cache.contains(mm_input.cache_key()))
        self.assertFalse(store.saved.is_set())

    def test_disabled_never_constructs_optional_client(self):
        config = VitConfig()
        config.use_local_preprocess = True
        config.mm_cache_gpu_max_bytes = 0
        config.mm_cache_cpu_max_bytes = 0
        config.mm_hash_key_cache_max_bytes = 0
        with patch("rtp_llm.multimodal.kvcm.RtpKvMetaObjectClient.from_env") as create:
            engine = MMProcessEngine(
                FakeEmbeddingLengthInterface(),
                ModelConfig(),
                config,
                ProfilingDebugLoggingConfig(),
            )
            self.addCleanup(engine.stop)
            self.assertIsNone(engine._remote_embedding_cache)
            create.assert_not_called()


if __name__ == "__main__":
    unittest.main()
