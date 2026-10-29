import json
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.py_config_modules import ProfilingDebugLoggingConfig, VitConfig
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import MultimodalInputsPB
from rtp_llm.multimodal.kvcm.tensor_object import pack_object, plan_object
from rtp_llm.multimodal.mm_embedding_cache import MMEmbeddingCache
from rtp_llm.multimodal.mm_process_engine import MMProcessEngine
from rtp_llm.multimodal.mm_remote_embedding_cache import MMRemoteEmbeddingCache
from rtp_llm.multimodal.multimodal_util import trans_mm_input
from rtp_llm.multimodal.test.multimodal_process_engine_test import (
    FakeEmbeddingLengthInterface,
)
from rtp_llm.ops import MMPreprocessConfig, MultimodalInput
from rtp_llm.utils.grpc_util import trans_from_tensor


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

    def save_one(self, key, value, **kwargs):
        self.write_gate.wait(3)
        if self.fail_write:
            raise RuntimeError("unknown save outcome")
        self.objects.setdefault(key, value.clone())
        self.saved.set()

    def object_size(self, key, **kwargs):
        value = self.objects.get(key)
        return None if value is None else value.numel()

    def load_one(self, key, value, **kwargs):
        self.read_started.set()
        self.read_gate.wait(3)
        if self.fail_read:
            raise RuntimeError("read unavailable")
        value.copy_(self.objects[key])

    def close(self):
        self.closed = True


class FakeClient:
    instance_id = "kve_unit"

    def __init__(self, storage):
        self.storage = storage
        self.fail_load = self.fail_save = False
        self.closed = False

    def object_size(self, key, **kwargs):
        return self.storage[key].numel() if key in self.storage else None

    def load_one(self, key, output, **kwargs):
        output.copy_(self.storage[key])
        if self.fail_load:
            output.fill_(23)
            raise TimeoutError("partial destination")

    def save_one(self, key, value, **kwargs):
        if self.fail_save:
            raise TimeoutError("write failed")
        self.storage[key] = value.clone()

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
            writer.bind_entry(entry, key, "writer")
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
            writer.bind_entry(entry, key, "writer")
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
            generation="test-generation",
        )
        remote.bind_entry(entry, "same", "writer")
        remote.submit_eviction("same", entry)
        remote.submit_eviction("same", entry)
        other = SimpleNamespace(**vars(entry))
        remote.bind_entry(other, "different", "writer")
        remote.submit_eviction("different", other)
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
        remote.bind_entry(entry, "key", "writer")
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
        config.mm_remote_cache_namespace = "test-model-weights-preprocessor"
        config.mm_remote_cache_read_timeout_ms = 500
        model = ModelConfig()
        model.mm_related_params.preprocess_batch_size = 1
        model.ckpt_path = "/nonexistent/test-model"
        interface = FakeEmbeddingLengthInterface()
        factory = patch(
            "rtp_llm.multimodal.kvcm.RtpKvMetaObjectClient.from_env", return_value=store
        )
        factory.start()
        self.addCleanup(factory.stop)
        engine = MMProcessEngine(
            interface, model, config, ProfilingDebugLoggingConfig()
        )
        self.addCleanup(engine.stop)
        return engine

    def test_sync_hit_preserves_extras_and_skips_preprocess_and_forward(self):
        mm_input = MultimodalInput(
            "data:image/jpeg;base64,AQID",
            1,
            torch.empty(0),
            MMPreprocessConfig(-1, -1, -1, -1, -1, -1, -1, [], 30000),
        )
        result = (
            torch.arange(12, dtype=torch.float32).reshape(3, 4),
            torch.arange(6).reshape(3, 2),
            [torch.tensor([10, 20], dtype=torch.int32)],
        )
        store = Store()
        engine = self.engine(store)
        key = engine._remote_embedding_cache.key([mm_input])
        store.objects[key] = pack_object(plan_object(result))
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
            "data:image/jpeg;base64,AQID",
            1,
            torch.empty(0),
            MMPreprocessConfig(-1, -1, -1, -1, -1, -1, -1, [], 30000),
        )
        response = engine.get_embedding_result([mm_input], request_id=101)
        self.assertEqual(response[0].embeddings[0].shape, (len(mm_input.url), 4))
        self.assertEqual(engine._remote_embedding_cache.stats()["miss"], 1)

    def test_remote_hit_does_not_bypass_greennet_rejection(self):
        from rtp_llm.config.exceptions import FtRuntimeException
        from rtp_llm.multimodal.greennet_hook import GreenNetVerdict
        from rtp_llm.multimodal.test.multimodal_process_engine_test import (
            _StubGreenNetProvider,
        )

        mm_input = MultimodalInput(
            "data:image/jpeg;base64,AQIE",
            1,
            torch.empty(0),
            MMPreprocessConfig(-1, -1, -1, -1, -1, -1, -1, [], 30000),
        )
        store = Store()
        engine = self.engine(store)
        key = engine._remote_embedding_cache.key([mm_input])
        store.objects[key] = pack_object(plan_object((torch.ones(3, 4), None)))
        engine._inspection._greennet_provider = _StubGreenNetProvider(
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


class RemoteIdentityAndRecoveryTest(unittest.TestCase):
    def setUp(self):
        self.storage = {}
        self.clients = []

        def factory():
            client = FakeClient(self.storage)
            self.clients.append(client)
            return client

        self.factory = factory
        self.cache = self.make_cache()
        self.value = (
            torch.arange(24, dtype=torch.bfloat16).reshape(3, 8),
            torch.arange(9, dtype=torch.int64).reshape(3, 3),
            torch.arange(72, dtype=torch.bfloat16),
        )

    def make_cache(self, namespace="model-weights-processor-v1", max_bytes=65536):
        value = MMRemoteEmbeddingCache(
            None,
            client_factory=self.factory,
            namespace=namespace,
            model_identity={"dtype": "torch.bfloat16"},
            max_object_bytes=max_bytes,
            max_inflight_bytes=1024 * 1024,
            read_timeout_ms=500,
        )
        self.addCleanup(value.close)
        return value

    def item(self, url="data:image/jpeg;base64,AQID"):
        return SimpleNamespace(
            url=url,
            mm_type=1,
            tensor=torch.empty(0),
            mm_preprocess_config=SimpleNamespace(
                width=-1,
                height=-1,
                min_pixels=256,
                max_pixels=1024,
                fps=-1,
                min_frames=-1,
                max_frames=-1,
                crop_positions=[],
            ),
        )

    def evict(self, remote, key="key", value=None):
        value = self.value if value is None else value
        entry = SimpleNamespace(
            pool_owners=[],
            result=value,
            error=None,
            original_devices=tuple(
                t.device if isinstance(t, torch.Tensor) else None for t in value
            ),
            generation="unit-generation",
        )
        remote.bind_entry(entry, key, "origin")
        remote.submit_eviction("local-key", entry)
        deadline = time.monotonic() + 3
        while remote.stats()["inflight_tasks"] and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertEqual(remote.stats()["inflight_tasks"], 0)

    def test_complete_eviction_result_roundtrip_and_fresh_client(self):
        key = self.cache.key([self.item()])
        self.assertIsNone(self.cache.load(key, trace_id="writer"))
        with self.assertLogs(level="INFO") as logs:
            self.evict(self.cache, key)
            reader = self.make_cache()
            actual = reader.load(key, trace_id="reader")
        events = [
            json.loads(line.split("MM_EMB_REMOTE_JSON ")[1])
            for line in logs.output
            if "MM_EMB_REMOTE_JSON " in line
        ]
        stored = next(e for e in events if e["event"] == "store_success")
        loaded = next(e for e in events if e["event"] == "load_hit")
        self.assertEqual(stored["payload_sha256"], loaded["payload_sha256"])
        self.assertEqual(stored["write_source"], "cpu_eviction")
        for a, expected in zip(actual, self.value):
            self.assertEqual(a.dtype, expected.dtype)
            self.assertTrue(torch.equal(a, expected))

    def test_retired_client_drains_concurrent_write_without_blocking_new_reader(self):
        blob = pack_object(plan_object(self.value))
        self.storage["existing"] = blob
        started, release, closing = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )

        class DrainingClient(FakeClient):
            def save_one(client, key, value, **kwargs):
                started.set()
                release.wait(3)
                super(DrainingClient, client).save_one(key, value, **kwargs)

            def load_one(client, key, output, **kwargs):
                output.fill_(9)
                raise TimeoutError("native partial transfer")

            def close(client):
                closing.set()
                release.wait(3)
                client.closed = True

        failed = DrainingClient(self.storage)
        self.cache._client = failed
        self.cache.read_timeout_ms = 20
        entry = SimpleNamespace(
            pool_owners=[],
            result=self.value,
            error=None,
            original_devices=tuple(t.device for t in self.value),
            generation="concurrent",
        )
        self.cache.bind_entry(entry, "pending-write", "writer")
        self.cache.submit_eviction("local", entry)
        self.assertTrue(started.wait(1))
        try:
            self.assertIsNone(self.cache.load("existing"))
            self.assertFalse(closing.is_set())
            self.assertFalse(failed.closed)
            # The failed read releases its IO worker, while the old client
            # stays alive for the writer. New requests use a fresh client.
            self.cache.read_timeout_ms = 500
            actual = self.cache.load("existing")
            self.assertTrue(torch.equal(actual[0], self.value[0]))
        finally:
            release.set()
            self.cache.close()
        self.assertTrue(closing.is_set())
        self.assertTrue(failed.closed)
        self.assertEqual(self.cache.stats()["inflight_tasks"], 0)
        self.assertEqual(self.cache.stats()["inflight_bytes"], 0)

    def test_metadata_failure_and_blocking_close_remain_bounded_off_request_thread(
        self,
    ):
        release = threading.Event()
        retired = []

        class SlowCleanup(FakeClient):
            def object_size(client, key, **kwargs):
                raise TimeoutError("metadata deadline")

            def close(client):
                retired.append(client)
                release.wait(3)
                client.closed = True

        self.cache._factory = lambda: SlowCleanup(self.storage)
        self.cache.read_timeout_ms = 20
        self.cache.max_pending = 2
        try:
            started = time.monotonic()
            self.assertIsNone(self.cache.load("one"))
            self.assertIsNone(self.cache.load("two"))
            self.assertLess(time.monotonic() - started, 0.5)
            self.assertEqual(len(retired), 2)
            self.assertFalse(any(client.closed for client in retired))
            self.assertEqual(self.cache.stats()["inflight_tasks"], 2)
            self.assertIsNone(self.cache.load("not-admitted"))
            self.assertEqual(self.cache.stats()["admission_skip"], 1)
            self.assertEqual(len(retired), 2)
        finally:
            release.set()
            self.cache.close()
        self.assertTrue(all(client.closed for client in retired))
        self.assertEqual(self.cache.stats()["inflight_tasks"], 0)
        self.assertEqual(self.cache.stats()["inflight_bytes"], 0)
        self.assertEqual(self.cache._client_users, {})

    def test_slow_client_factory_is_also_inside_read_deadline(self):
        started, release = threading.Event(), threading.Event()

        def factory():
            started.set()
            release.wait(3)
            return FakeClient(self.storage)

        self.cache._factory = factory
        self.cache.read_timeout_ms = 20
        try:
            before = time.monotonic()
            self.assertIsNone(self.cache.load("missing"))
            self.assertLess(time.monotonic() - before, 0.5)
            self.assertTrue(started.is_set())
            self.assertEqual(self.cache.stats()["inflight_tasks"], 1)
        finally:
            release.set()
            self.cache.close()
        self.assertEqual(self.cache.stats()["inflight_tasks"], 0)
        self.assertEqual(self.cache._client_users, {})

    def test_partial_native_failure_discards_buffer_and_recreates_client(self):
        self.evict(self.cache)
        failed = self.clients[-1]
        failed.fail_load = True
        self.assertIsNone(self.cache.load("key"))
        self.assertTrue(failed.closed)
        restored = self.cache.load("key")
        self.assertEqual(len(self.clients), 2)
        self.assertTrue(torch.equal(restored[0], self.value[0]))

    def test_failed_eviction_write_has_no_implicit_retry(self):
        self.cache.load("missing")
        self.clients[-1].fail_save = True
        self.evict(self.cache)
        self.assertNotIn("key", self.storage)
        self.assertEqual(self.cache.stats()["write_error"], 1)
        self.assertEqual(len(self.clients), 1)
        self.evict(self.cache)
        self.assertIsNotNone(self.cache.load("key"))

    def test_oversize_admission_before_native_client(self):
        tiny = self.make_cache(max_bytes=20)
        self.evict(tiny)
        self.assertFalse(self.clients)
        self.assertEqual(tiny.stats()["encode_skip"], 1)

    def test_unbound_entry_cannot_upload_mutable_local_key(self):
        entry = SimpleNamespace(pool_owners=[], result=self.value, error=None)
        self.cache.submit_eviction("https://mutable", entry)
        self.assertEqual(self.cache.stats()["inflight_tasks"], 0)
        self.assertFalse(self.clients)

    def test_benchmark_logger_failure_preserves_io_and_releases_resources(self):
        def log(marker, *args, **kwargs):
            if marker.startswith("MM_EMB_BENCHMARK_JSON"):
                raise RuntimeError("diagnostic sink unavailable")

        with patch.dict("os.environ", {"MM_EMB_BENCHMARK_TIMING": "1"}), patch(
            "rtp_llm.multimodal.mm_remote_embedding_cache.logging.info", side_effect=log
        ):
            self.evict(self.cache)
            value = self.cache.load("key")
        self.assertTrue(torch.equal(value[0], self.value[0]))
        self.assertEqual(self.cache.stats()["inflight_tasks"], 0)
        self.assertEqual(self.cache.stats()["inflight_bytes"], 0)
        self.assertEqual(self.cache._client_users, {})

    def test_immutable_key_namespace_and_all_preprocessing_fields(self):
        item = self.item()
        key = self.cache.key([item])
        self.assertEqual(key, self.make_cache().key([self.item()]))
        self.assertNotEqual(key, self.make_cache(namespace="new-weights").key([item]))
        item.mm_preprocess_config.crop_positions = [0.123456789]
        self.assertNotEqual(key, self.cache.key([item]))
        first_crop = self.cache.key([item])
        item.mm_preprocess_config.crop_positions = [0.123456790]
        self.assertNotEqual(first_crop, self.cache.key([item]))
        item.url = "data:image/jpeg;base64,AQIE"
        self.assertNotEqual(first_crop, self.cache.key([item]))

    def test_mutable_urls_tensor_inputs_and_batched_groups_bypass(self):
        self.assertIsNone(self.cache.key([self.item("https://host/mutable.jpg")]))
        self.assertIsNone(self.cache.key([self.item("/tmp/mutable.jpg")]))
        item = self.item()
        item.tensor = torch.ones(3)
        self.assertIsNone(self.cache.key([item]))
        self.assertIsNone(self.cache.key([self.item(), self.item()]))
        self.assertIsNone(self.cache.load(None))
        self.assertFalse(self.clients)

    def test_inline_identity_hashes_decoded_bytes_and_reports_rejections(self):
        self.assertEqual(
            self.cache.key([self.item("data:image/jpeg;base64,AQID")]),
            self.cache.key([self.item("data:image/png;base64,AQID")]),
        )
        self.assertIsNone(self.cache.key([self.item("data:image/jpeg;base64,!")]))
        self.assertIsNone(self.cache.key([self.item("data:image/jpeg;base64,")]))
        with self.assertLogs(level="INFO") as logs:
            item = self.item()
            item.tensor = torch.empty(1)
            self.assertIsNone(self.cache.key([item], trace_id="reject"))
        event = json.loads(logs.output[0].split("MM_EMB_REMOTE_JSON ")[1])
        self.assertEqual(event["reason"], "nonempty_tensor")
        self.assertEqual(event["tensor_shape"], [1])
        self.assertEqual(event["trace_id"], "reject")
        self.assertNotIn("AQID", logs.output[0])

    def test_backend_registration_failure_falls_back(self):
        def fail():
            raise RuntimeError("unavailable")

        self.cache._factory = fail
        self.assertIsNone(self.cache.load("key"))
        self.evict(self.cache)
        self.assertEqual(self.cache.stats()["read_error"], 1)
        self.assertEqual(self.cache.stats()["write_error"], 1)


class VitRpcRemoteIdentityTest(unittest.TestCase):
    def setUp(self):
        self.cache = MMRemoteEmbeddingCache(
            None,
            client_factory=lambda: self.fail("identity checks must not open a client"),
            namespace="deepseek-v41-test-weight-revision",
            model_identity={"model_type": "deepseek_v41"},
            max_object_bytes=65536,
            max_inflight_bytes=1024 * 1024,
        )
        self.addCleanup(self.cache.close)

    def decode(self, url="data:image/jpeg;base64,AQID", tensor=None, mm_type=1):
        wire = MultimodalInputsPB(request_id=123)
        image = wire.multimodal_inputs.add()
        image.multimodal_url = url
        image.multimodal_type = mm_type
        if tensor is not None:
            image.multimodal_tensor.CopyFrom(trans_from_tensor(tensor))
        # Exercise the actual protobuf wire decoder used by the ViT RPC server.
        decoded = MultimodalInputsPB.FromString(wire.SerializeToString())
        return trans_mm_input(decoded)

    def test_url_only_empty_tensor_wire_is_admitted_for_deepseek_v41(self):
        for tensor in (None, torch.empty(0)):
            with self.subTest(tensor_present=tensor is not None):
                inputs = self.decode(tensor=tensor)
                self.assertEqual(inputs[0].tensor.numel(), 0)
                self.assertIsNotNone(self.cache.key(inputs))

    def test_real_jpeg_bytes_are_admitted_after_wire_roundtrip(self):
        import base64
        import io

        from PIL import Image

        content = io.BytesIO()
        Image.new("RGB", (3, 2), (11, 22, 33)).save(content, format="JPEG")
        image_bytes = content.getvalue()
        with Image.open(io.BytesIO(image_bytes)) as image:
            self.assertEqual(image.size, (3, 2))
        url = "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode()
        inputs = self.decode(url=url, tensor=torch.empty(0))
        self.assertEqual(inputs[0].tensor.numel(), 0)
        self.assertEqual(inputs[0].url, url)
        key = self.cache.key(inputs)
        self.assertIsNotNone(key)
        self.assertEqual(key, self.cache.key(self.decode(url=url)))

    def test_renderer_png_and_jpeg_octet_stream_keep_original_image_key(self):
        import base64
        import io

        from PIL import Image

        from rtp_llm.config.generate_config import GenerateConfig
        from rtp_llm.cpp.model_rpc.model_rpc_client import trans_input
        from rtp_llm.openai.renderers.deepseekv41_renderer import _image_records_to_urls
        from rtp_llm.utils.base_model_datatypes import GenerateInput, MMUrlType

        for image_format, mime in (("PNG", "image/png"), ("JPEG", "image/jpeg")):
            with self.subTest(image_format=image_format):
                content = io.BytesIO()
                Image.new("RGB", (3, 2), (11, 22, 33)).save(
                    content, format=image_format
                )
                image_bytes = content.getvalue()
                encoded = base64.b64encode(image_bytes).decode("ascii")
                original_url = f"data:{mime};base64,{encoded}"
                original_key = self.cache.key(self.decode(url=original_url))
                self.assertIsNotNone(original_key)
                for record in (
                    {"source": {"url": original_url}},
                    {"data": image_bytes},
                ):
                    urls = _image_records_to_urls([1234], [record], 1234)
                    self.assertEqual(
                        urls, ["data:application/octet-stream;base64," + encoded]
                    )
                    # Invoke the actual frontend serializer, not a hand-built
                    # absent TensorPB. URL-only requests must reach C++ with
                    # has_multimodal_tensor() false.
                    frontend = trans_input(
                        GenerateInput(
                            request_id=123,
                            token_ids=torch.tensor([1234], dtype=torch.int32),
                            mm_inputs=[
                                MultimodalInput(
                                    url=urls[0],
                                    mm_type=MMUrlType.IMAGE,
                                    tensor=torch.empty(0),
                                    mm_preprocess_config=MMPreprocessConfig(),
                                )
                            ],
                            generate_config=GenerateConfig(max_new_tokens=1),
                        )
                    )
                    frontend = type(frontend).FromString(frontend.SerializeToString())
                    self.assertEqual(len(frontend.multimodal_inputs), 1)
                    image = frontend.multimodal_inputs[0]
                    self.assertEqual(image.multimodal_url, urls[0])
                    self.assertEqual(image.multimodal_type, MMUrlType.IMAGE)
                    self.assertFalse(image.HasField("multimodal_tensor"))

                    # QueryConverterMultimodalInputTest independently exercises
                    # the actual C++ converter for this absent-TensorPB contract.
                    # Reproduce its present zero-length tensor on the ViT wire,
                    # then invoke the actual Python ViT decoder and cache key.
                    wire = MultimodalInputsPB(request_id=123)
                    forwarded = wire.multimodal_inputs.add()
                    forwarded.CopyFrom(image)
                    forwarded.multimodal_tensor.CopyFrom(
                        trans_from_tensor(torch.empty(0))
                    )
                    self.assertTrue(forwarded.HasField("multimodal_tensor"))
                    self.assertEqual(list(forwarded.multimodal_tensor.shape), [0])
                    inputs = trans_mm_input(
                        MultimodalInputsPB.FromString(wire.SerializeToString())
                    )
                    self.assertEqual(inputs[0].tensor.numel(), 0)
                    # Compare the same serialized preprocessing defaults.
                    expected = [
                        MultimodalInput(
                            url=original_url,
                            mm_type=MMUrlType.IMAGE,
                            tensor=torch.empty(0),
                            mm_preprocess_config=inputs[0].mm_preprocess_config,
                        )
                    ]
                    expected_key = self.cache.key(expected)
                    self.assertIsNotNone(expected_key)
                    self.assertEqual(self.cache.key(inputs), expected_key)

    def test_octet_stream_requires_image_media_type(self):
        url = "data:application/octet-stream;base64,AQID"
        for mm_type in (0, 2, 3, 4):
            with self.subTest(mm_type=mm_type):
                self.assertIsNone(self.cache.key(self.decode(url=url, mm_type=mm_type)))

    def test_other_data_media_and_parameters_still_bypass(self):
        for prefix in (
            "data:application/json;base64,",
            "data:application/pdf;base64,",
            "data:application/octet-stream-extra;base64,",
            "data:text/plain;base64,",
            "data:audio/wav;base64,",
            "data:video/mp4;base64,",
            "data:application/octet-stream;charset=utf-8;base64,",
            "data:application/octet-stream,",
        ):
            with self.subTest(prefix=prefix):
                self.assertIsNone(self.cache.key(self.decode(url=prefix + "AQID")))

    def test_octet_stream_base64_and_nonempty_tensor_fail_closed(self):
        prefix = "data:application/octet-stream;base64,"
        for payload in ("", "!", "AQI", "AQID\n", "AQ ID"):
            with self.subTest(payload=payload):
                self.assertIsNone(self.cache.key(self.decode(url=prefix + payload)))
        expected = torch.tensor([42.0])
        inputs = self.decode(url=prefix + "AQID", tensor=expected)
        self.assertIsNone(self.cache.key(inputs))
        self.assertTrue(torch.equal(inputs[0].tensor, expected))
        self.assertIsNone(
            self.cache.key(self.decode(url=prefix + "AQID") + self.decode())
        )

    def test_octet_stream_keeps_preprocessing_and_source_bytes_in_identity(self):
        url = "data:application/octet-stream;base64,AQID"
        inputs = self.decode(url=url)
        original_key = self.cache.key(inputs)
        self.assertIsNotNone(original_key)
        inputs[0].mm_preprocess_config.width = 256
        self.assertNotEqual(self.cache.key(inputs), original_key)
        self.assertNotEqual(
            self.cache.key(
                self.decode(url="data:application/octet-stream;base64,AQIE")
            ),
            original_key,
        )

    def test_explicit_one_float_is_preserved_and_bypasses_remote_cache(self):
        expected = torch.tensor([42.0])
        inputs = self.decode(tensor=expected)
        self.assertTrue(torch.equal(inputs[0].tensor, expected))
        self.assertIsNone(self.cache.key(inputs))
        self.assertTrue(torch.equal(inputs[0].tensor, expected))

    def test_mutable_urls_and_grouped_inputs_still_bypass(self):
        for url in (
            "https://host/image.jpg",
            "/tmp/image.jpg",
            "data:image/jpeg;base64,!",
        ):
            with self.subTest(url=url):
                self.assertIsNone(self.cache.key(self.decode(url=url)))
        self.assertIsNone(self.cache.key(self.decode() + self.decode()))

    def test_tensor_only_rpc_stays_a_tensor_and_bypasses_remote_cache(self):
        expected = torch.arange(6, dtype=torch.int32).reshape(2, 3)
        inputs = self.decode(url="", tensor=expected, mm_type=4)
        self.assertTrue(torch.equal(inputs[0].tensor, expected))
        self.assertIsNone(self.cache.key(inputs))


class DownloadedSourceModel(FakeEmbeddingLengthInterface):
    supports_inspection_sources = True
    supports_remote_source_identity = True

    def __init__(self):
        super().__init__()
        self.forward_count = 0

    def get_preprocess_params(self):
        from rtp_llm.multimodal.multimodal_mixins.deepseek_v41.deepseek_v41_processor import (
            V41ImageProcessorConfig,
        )

        return dict(processor_config=V41ImageProcessorConfig(vision_min_pixels=1764))

    @staticmethod
    def preprocess_input(mm_inputs, vit_config, **kwargs):
        from rtp_llm.multimodal.multimodal_mixins.deepseek_v41.deepseek_v41_vision import (
            DeepSeekV41VisionEmbedding,
        )

        return DeepSeekV41VisionEmbedding.preprocess_input(
            mm_inputs, vit_config, **kwargs
        )

    def batched_embedding(self, data_list, mm_types, **kwargs):
        self.forward_count += len(data_list)
        return [
            (torch.full((2, 4), float(data.rgb[0, 0, 0])), None) for data in data_list
        ]


class DownloadedImageCacheTest(unittest.TestCase):
    def image_bytes(self, color="red"):
        import io

        from PIL import Image

        stream = io.BytesIO()
        Image.new("RGB", (42, 42), color).save(stream, format="PNG")
        return stream.getvalue()

    def item(self, url="https://example.invalid/mutable.png"):
        return MultimodalInput(
            url,
            1,
            torch.empty(0),
            MMPreprocessConfig(-1, -1, -1, -1, -1, -1, -1, [], 30000),
        )

    def engine(self, payload, *, enabled=True, passed=True, hold=False):
        import io

        from rtp_llm.multimodal.test.mm_inspection_pipeline_test import SourceProvider

        config = VitConfig()
        config.use_local_preprocess = True
        config.mm_cache_gpu_max_bytes = 0
        config.mm_cache_cpu_max_bytes = 1024 * 1024
        config.mm_hash_key_cache_max_bytes = 4096
        config.mm_remote_cache_enable = enabled
        config.mm_remote_cache_namespace = "downloaded-source-test-model-v1"
        config.mm_remote_cache_read_timeout_ms = 500
        model = ModelConfig()
        model.model_type = "deepseek_v41"
        model.ckpt_path = "/nonexistent/test-model"
        model.mm_related_params.preprocess_batch_size = 1
        store = Store()
        factory = patch(
            "rtp_llm.multimodal.kvcm.RtpKvMetaObjectClient.from_env", return_value=store
        )
        factory.start()
        self.addCleanup(factory.stop)
        interface = DownloadedSourceModel()
        engine = MMProcessEngine(
            interface, model, config, ProfilingDebugLoggingConfig(), device="cpu"
        )
        self.addCleanup(engine.stop)
        provider = SourceProvider(passed, hold)
        engine._inspection._greennet_provider = provider
        self.addCleanup(provider.release.set)
        download = patch(
            "rtp_llm.multimodal.multimodal_mixins.deepseek_v41.deepseek_v41_vision.load_image_source",
            side_effect=lambda *args, **kw: io.BytesIO(payload),
        )
        download.start()
        self.addCleanup(download.stop)
        return engine, store, provider, interface, download

    def seed(self, engine, store, payload, value=None):
        import base64

        item = self.item("data:image/png;base64," + base64.b64encode(payload).decode())
        key = engine._remote_embedding_cache.key([item])
        value = (torch.full((2, 4), 37.0), None) if value is None else value
        store.objects[key] = pack_object(plan_object(value))
        return key, value

    def test_fresh_inspection_remote_hit_waits_for_matching_verdict_without_encoder(
        self,
    ):
        payload = self.image_bytes()
        engine, store, provider, model, _ = self.engine(payload, hold=True)
        item = self.item()
        key, value = self.seed(engine, store, payload)
        engine.async_submit([item], request_id=901)
        self.assertTrue(store.read_started.wait(5))
        limit = time.monotonic() + 5
        while not provider.calls and time.monotonic() < limit:
            time.sleep(0.005)
        self.assertEqual(provider.calls, [([item.url], [payload])])
        entry = engine._embedding_cache.peek(item.cache_key())
        self.assertIsNotNone(entry)
        self.assertFalse(entry.is_done)
        self.assertFalse(store.saved.is_set())
        self.assertEqual(model.forward_count, 0)
        provider.release.set()
        result = engine.get_embedding_result([item], request_id=901)
        self.assertTrue(torch.equal(result[0].embeddings[0], value[0]))
        self.assertTrue(entry.greennet_passed)
        self.assertEqual(entry.remote_cache_context["key"], key)
        self.assertEqual(engine._remote_embedding_cache.stats()["hit"], 1)

    def test_fresh_remote_hit_rejection_never_publishes_or_returns(self):
        from rtp_llm.config.exceptions import FtRuntimeException

        payload = self.image_bytes()
        engine, store, provider, model, _ = self.engine(payload, passed=False)
        self.seed(engine, store, payload)
        item = self.item()
        with self.assertRaises(FtRuntimeException):
            engine.get_embedding_result([item], request_id=902)
        self.assertEqual(provider.calls, [([item.url], [payload])])
        self.assertEqual(engine._remote_embedding_cache.stats()["hit"], 1)
        self.assertEqual(model.forward_count, 0)
        self.assertIsNone(engine._embedding_cache.peek(item.cache_key()))
        self.assertFalse(store.saved.is_set())

    def test_fresh_remote_hit_inspection_timeout_never_publishes(self):
        import concurrent.futures

        payload = self.image_bytes()
        engine, store, provider, model, _ = self.engine(payload, hold=True)
        self.seed(engine, store, payload)
        item = self.item()
        key = item.cache_key()
        _, entry = engine._embedding_cache.try_acquire(key)
        engine._run_async_compute([item], key, entry, 903, time.monotonic() + 0.3)
        with self.assertRaises(concurrent.futures.TimeoutError):
            entry.wait(timeout=5)
        self.assertEqual(engine._remote_embedding_cache.stats()["hit"], 1)
        self.assertEqual(model.forward_count, 0)
        self.assertIsNone(engine._embedding_cache.peek(key))
        self.assertFalse(store.saved.is_set())

    def test_same_url_new_bytes_do_not_reuse_old_identity_or_url_verdict(self):
        from rtp_llm.config.exceptions import FtRuntimeException

        old, new = self.image_bytes("red"), self.image_bytes("blue")
        engine, store, provider, model, _ = self.engine(new, passed=False)
        old_key, _ = self.seed(engine, store, old)
        item = self.item()
        with patch.object(engine._hash_key_cache, "greennet_passed", return_value=True):
            verdict = engine.wait_greennet_verdict([item], request_id=904)
            self.assertFalse(verdict.passed)
            with self.assertRaises(FtRuntimeException):
                engine.get_embedding_result([item], request_id=904)
        self.assertTrue(provider.calls)
        self.assertTrue(all(sources == [new] for _, sources in provider.calls))
        self.assertGreaterEqual(engine._remote_embedding_cache.stats()["miss"], 1)
        self.assertEqual(engine._remote_embedding_cache.stats().get("hit", 0), 0)
        self.assertGreaterEqual(model.forward_count, 1)
        self.assertEqual(list(store.objects), [old_key])
        self.assertFalse(store.saved.is_set())

    def test_remote_transfer_failure_uses_already_decoded_source_and_encoder(self):
        payload = self.image_bytes()
        engine, store, provider, model, _ = self.engine(payload)
        self.seed(engine, store, payload)
        store.fail_read = True
        result = engine.get_embedding_result([self.item()], request_id=905)
        self.assertEqual(model.forward_count, 1)
        self.assertEqual(result[0].embeddings[0][0, 0].item(), 255)
        self.assertEqual(provider.calls[0][1], [payload])
        self.assertEqual(engine._remote_embedding_cache.stats()["read_error"], 1)

    def test_remote_disabled_keeps_encoder_and_has_no_optional_client(self):
        payload = self.image_bytes()
        engine, store, provider, model, _ = self.engine(payload, enabled=False)
        result = engine.get_embedding_result([self.item()], request_id=906)
        self.assertEqual(model.forward_count, 1)
        self.assertEqual(result[0].embeddings[0][0, 0].item(), 255)
        self.assertEqual(provider.calls[0][1], [payload])
        self.assertIsNone(engine._remote_embedding_cache)
        self.assertFalse(store.read_started.is_set())
        self.assertFalse(store.saved.is_set())

    def test_http_and_inline_actual_bytes_share_key_but_changed_bytes_do_not(self):
        import base64

        from rtp_llm.multimodal.mm_preprocess_result import ImageSourceIdentity

        payload = self.image_bytes()
        engine, _, _, _, _ = self.engine(payload)
        remote = engine._remote_embedding_cache
        url = self.item()
        identity = ImageSourceIdentity.from_bytes(payload)
        http_key = remote.key([url], source_identity=identity)
        inline = self.item(
            "data:application/octet-stream;base64," + base64.b64encode(payload).decode()
        )
        self.assertEqual(http_key, remote.key([inline]))
        self.assertNotEqual(
            http_key,
            remote.key(
                [url],
                source_identity=ImageSourceIdentity.from_bytes(
                    self.image_bytes("blue")
                ),
            ),
        )
        self.assertIsNone(remote.key([url]))
        self.assertIsNone(
            remote.key(
                [inline], source_identity=ImageSourceIdentity.from_bytes(b"mismatch")
            )
        )
        for media_type in (2, 3, 4):
            url.mm_type = media_type
            self.assertIsNone(remote.key([url], source_identity=identity))
        url.mm_type = 1
        url.tensor = torch.ones(1)
        self.assertIsNone(remote.key([url], source_identity=identity))


if __name__ == "__main__":
    unittest.main()
