import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.config.exceptions import FtRuntimeException
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.py_config_modules import ProfilingDebugLoggingConfig, VitConfig
from rtp_llm.multimodal.greennet_hook import GreenNetVerdict
from rtp_llm.multimodal.kvcm.tensor_object import (
    pack_object,
    plan_object,
    unpack_object,
)
from rtp_llm.multimodal.mm_embedding_cache import (
    MMEmbeddingCache,
    MMEmbeddingCacheEntry,
)
from rtp_llm.multimodal.mm_process_engine import MMProcessEngine
from rtp_llm.multimodal.mm_remote_embedding_cache import MMRemoteEmbeddingCache
from rtp_llm.multimodal.test.multimodal_process_engine_test import (
    FakeEmbeddingLengthInterface,
    _StubGreenNetProvider,
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

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_gpu_only_eviction_writes_snapshot_after_pool_reuse(self):
        store = Store()
        store.write_gate.clear()
        remote = self.remote(store)
        cache = MMEmbeddingCache(32, 0, on_gpu_evict=remote.submit_eviction)
        self.addCleanup(cache.clear)
        self.addCleanup(store.write_gate.set)

        source = torch.arange(8, dtype=torch.float32, device="cuda").reshape(2, 4)
        expected = source.cpu()
        _, first = cache.try_acquire("first")
        verdict = GreenNetVerdict(True, 1, "GPU eviction")
        first.set_greennet_verdict(verdict)
        cache.complete("first", first, (source, None))
        self.assertEqual(cache.resident_tiers(), {"first": "gpu"})

        with torch.cuda.stream(torch.cuda.Stream()):
            _, second = cache.try_acquire("second")
            cache.complete("second", second, (torch.zeros_like(source), None))
        source.zero_()
        self.assertEqual(cache.resident_tiers(), {"second": "gpu"})
        self.assertEqual(cache.stats()["cpu_pool_capacity_bytes"], 0)
        held = first.wait()[0]
        self.assertEqual(held.device, source.device)
        self.assertTrue(torch.equal(held.cpu(), expected))
        held.zero_()

        store.write_gate.set()
        self.assertTrue(store.saved.wait(2))
        restored = MMEmbeddingCacheEntry()
        loaded, positions = remote.load("first", entry=restored)
        self.assertEqual(restored.wait_greennet(0), verdict)
        self.assertTrue(restored.greennet_passed)
        self.assertIsNone(positions)
        self.assertEqual(loaded.device, source.device)
        self.assertTrue(torch.equal(loaded.cpu(), expected))
        cache.clear()
        remote.close()
        self.assertEqual(set(store.objects), {"first"})

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_gpu_only_fragmentation_writes_only_the_lru_victim(self):
        store = Store()
        remote = self.remote(store)
        cache = MMEmbeddingCache(32, 0, on_gpu_evict=remote.submit_eviction)
        self.addCleanup(cache.clear)
        for index, key in enumerate(("a", "b", "c", "d")):
            _, entry = cache.try_acquire(key)
            value = torch.full((2,), float(index), device="cuda")
            cache.complete(key, entry, (value, None))
        cache.remove("b")
        cache.remove("d")
        _, entry = cache.try_acquire("large")
        cache.complete("large", entry, (torch.ones(4, device="cuda"), None))
        self.assertEqual(cache.resident_tiers(), {"c": "gpu", "large": "gpu"})
        remote.close()
        self.assertEqual(set(store.objects), {"a"})

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

    def test_malformed_or_rejected_remote_verdict_cannot_approve_an_entry(self):
        store = Store()
        remote = self.remote(store)
        inspection = dict(
            checked=True, passed=True, code=1, message="", request_scoped=False
        )
        for field, invalid in (
            ("checked", "true"),
            ("request_scoped", 0),
            ("code", True),
            ("message", 0),
            ("passed", False),
        ):
            with self.subTest(field=field):
                value = dict(
                    version=1,
                    result=(torch.ones(2, 4), None),
                    greennet={**inspection, field: invalid},
                )
                store.objects["bad"] = pack_object(plan_object(value))
                entry = MMEmbeddingCacheEntry()
                self.assertIsNone(remote.load("bad", entry=entry))
                self.assertFalse(entry.is_greennet_decided)
        self.assertEqual(remote.stats()["read_error"], 5)

    def test_write_budget_dedup_and_unknown_failure_do_not_remove(self):
        store = Store()
        store.write_gate.clear()
        remote = self.remote(store, max_pending=1)
        entry = MMEmbeddingCacheEntry()
        entry.complete((torch.ones(2, 4), None))
        entry.original_devices = (torch.device("cpu"), None)
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
    def engine(self, store, *, gpu_bytes=0, cpu_bytes=1024, request_scoped=False):
        config = VitConfig()
        config.use_local_preprocess = True
        config.mm_cache_gpu_max_bytes = gpu_bytes
        config.mm_cache_cpu_max_bytes = cpu_bytes
        config.mm_hash_key_cache_max_bytes = 4096
        config.mm_remote_cache_enable = True
        config.mm_remote_cache_read_timeout_ms = 500
        model = ModelConfig()
        model.mm_related_params.preprocess_batch_size = 1
        interface = FakeEmbeddingLengthInterface()
        interface.inspection_verdict_is_request_scoped = request_scoped
        with patch(
            "rtp_llm.multimodal.kvcm.RtpKvMetaObjectClient.from_env", return_value=store
        ):
            engine = MMProcessEngine(
                interface, model, config, ProfilingDebugLoggingConfig()
            )
        self.addCleanup(engine.stop)
        return engine

    def test_remote_requires_at_least_one_local_tier(self):
        with self.assertRaisesRegex(ValueError, "nonzero CPU or GPU embedding cache"):
            self.engine(Store(), gpu_bytes=0, cpu_bytes=0)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_engine_wires_gpu_only_eviction_to_remote(self):
        store = Store()
        engine = self.engine(store, gpu_bytes=32, cpu_bytes=0)
        config = MMPreprocessConfig(-1, -1, -1, -1, -1, -1, -1, [], 30000)
        first, second = [
            MultimodalInput(url, 1, torch.empty(0), config) for url in ("a", "bb")
        ]

        def embed(data_list, mm_types, **kwargs):
            return [
                (
                    torch.arange(8, device="cuda").float().reshape(2, 4) + data.item(),
                    None,
                )
                for data in data_list
            ]

        with patch.object(
            engine.mm_part, "batched_embedding", side_effect=embed
        ) as forward:
            expected = engine.mm_embedding_impl([first]).embeddings[0].clone()
            engine._embedding_cache.flush()
            engine.mm_embedding_impl([second])
            engine._embedding_cache.flush()
            self.assertTrue(store.saved.wait(2))
            self.assertEqual(set(store.objects), {first.cache_key()})
            loaded = engine.mm_embedding_impl([first]).embeddings[0]
            self.assertEqual(forward.call_count, 2)
        self.assertEqual(loaded.device, expected.device)
        self.assertTrue(torch.equal(loaded, expected))
        self.assertEqual(engine._remote_embedding_cache.stats()["hit"], 1)

    @staticmethod
    def input(url):
        config = MMPreprocessConfig(-1, -1, -1, -1, -1, -1, -1, [], 30000)
        return MultimodalInput(url, 1, torch.empty(0), config)

    def test_checked_result_crosses_workers_without_reinspection(self):
        # One 32-byte slot forces the first URL to be evicted to KVCM.
        first, second = self.input("aa"), self.input("bb")
        store = Store()
        writer = self.engine(store, cpu_bytes=32)
        verdict = GreenNetVerdict(True, 1, "accepted by writer")
        writer._inspection._greennet_provider = _StubGreenNetProvider(verdict)
        expected = writer._embed_with_greennet_sync([first]).embeddings[0].clone()
        writer._embedding_cache.flush()
        writer._embed_with_greennet_sync([second])
        writer._embedding_cache.flush()
        self.assertTrue(store.saved.wait(2))
        for sync in (True, False):
            with self.subTest(sync=sync):
                reader = self.engine(Store(store.objects))
                provider = _StubGreenNetProvider(
                    GreenNetVerdict(False, 2, "unexpected")
                )
                reader._inspection._greennet_provider = provider
                with patch.object(
                    reader.preprocess_executor,
                    "preprocess_func",
                    side_effect=AssertionError("unexpected download"),
                ), patch.object(
                    reader.mm_part,
                    "batched_embedding",
                    side_effect=AssertionError("unexpected forward"),
                ):
                    if sync:
                        result = reader._embed_with_greennet_sync([first])
                    else:
                        result = reader.get_embedding_result([first], request_id=71)[0]
                self.assertTrue(torch.equal(result.embeddings[0], expected))
                entry = reader._embedding_cache.peek(first.cache_key())
                self.assertEqual(entry.greennet_snapshot(), (verdict, True))
                self.assertEqual(provider.calls, 0)

    def test_followers_of_remote_lookup_respect_producer_inspection_override(self):
        item = self.input("inflight")
        value = (torch.ones(2, 4), None)
        for producer_checks in (True, False):
            with self.subTest(producer_checks=producer_checks):
                store = Store({item.cache_key(): pack_object(plan_object(value))})
                store.read_gate.clear()
                engine = self.engine(store)
                provider = _StubGreenNetProvider(GreenNetVerdict(True))
                engine._inspection._greennet_provider = provider
                engine.async_submit(
                    [item],
                    request_id=1,
                    user_id="producer",
                    input_inspection_enabled=producer_checks,
                )
                self.assertTrue(store.read_started.wait(2))
                # Release the producer only after the follower has joined the same lookup.
                claim = engine._claim_and_submit_async

                def join(*args, **kwargs):
                    entries = claim(*args, **kwargs)
                    store.read_gate.set()
                    return entries

                with patch.object(engine, "_claim_and_submit_async", side_effect=join):
                    engine.get_embedding_result(
                        [item],
                        request_id=2,
                        user_id="follower",
                        input_inspection_enabled=True,
                    )
                self.assertEqual(
                    provider.user_ids, ["producer" if producer_checks else "follower"]
                )
                self.assertEqual(engine._remote_embedding_cache.stats()["hit"], 1)

    def test_source_batch_inspects_only_unchecked_remote_results(self):
        from rtp_llm.multimodal.mm_preprocess import MMPreprocessResult

        checked, unchecked, legacy = [self.input(url) for url in ("ok", "off", "old")]
        store = Store()
        writer = self.engine(store)
        cache = writer._embedding_cache
        value = (torch.ones(2, 4), None)
        for item, approved in ((checked, True), (unchecked, False)):
            _, entry = cache.try_acquire(item.cache_key())
            entry.set_greennet_verdict(
                GreenNetVerdict(True, 1, "stored"), checked=approved
            )
            cache.complete(item.cache_key(), entry, value)
        cache.flush()
        cache.resize(0, 0)
        writer._remote_embedding_cache.close()
        store.objects[legacy.cache_key()] = pack_object(plan_object(value))
        unchecked_value = unpack_object(store.objects[unchecked.cache_key()])
        self.assertFalse(unchecked_value["greennet"]["checked"])
        reader = self.engine(Store(store.objects))
        reader.mm_part.supports_inspection_sources = True
        provider = _StubGreenNetProvider(GreenNetVerdict(True))
        reader._inspection._greennet_provider = provider

        def prepare(inputs, config, **kwargs):
            self.assertTrue(kwargs["capture_inspection_sources"])
            return MMPreprocessResult(None, [item.url.encode() for item in inputs])

        with patch.object(
            reader.preprocess_executor, "preprocess_func", side_effect=prepare
        ) as preprocess, patch.object(
            reader.mm_part,
            "batched_embedding",
            side_effect=AssertionError("unexpected forward"),
        ):
            result = reader._embed_with_greennet_sync([checked, unchecked, legacy])
        self.assertEqual(len(result.embeddings), 3)
        self.assertEqual(preprocess.call_count, 2)
        self.assertEqual(provider.source_bytes, [b"off", b"old"])
        self.assertEqual(
            reader._embedding_cache.peek(checked.cache_key()).wait_greennet(0).message,
            "stored",
        )
        self.assertTrue(
            reader._embedding_cache.peek(unchecked.cache_key()).greennet_passed
        )

    def test_request_scoped_writer_or_reader_requires_fresh_verdict(self):
        item = self.input("scoped")
        for scoped_writer, scoped_reader in ((True, False), (False, True)):
            with self.subTest(scoped_writer=scoped_writer, scoped_reader=scoped_reader):
                store = Store()
                writer = self.engine(store, request_scoped=scoped_writer)
                writer._inspection._greennet_provider = _StubGreenNetProvider(
                    GreenNetVerdict(True)
                )
                writer._embed_with_greennet_sync([item])
                writer._embedding_cache.flush()
                writer._embedding_cache.resize(0, 0)
                self.assertTrue(store.saved.wait(2))
                reader = self.engine(Store(store.objects), request_scoped=scoped_reader)
                provider = _StubGreenNetProvider(GreenNetVerdict(False, 2, "blocked"))
                reader._inspection._greennet_provider = provider
                with self.assertRaisesRegex(FtRuntimeException, "blocked"):
                    reader.get_embedding_result([item], user_id="new-user")
                self.assertEqual(provider.user_ids, ["new-user"])
                self.assertIsNone(reader._embedding_cache.peek(item.cache_key()))

    def test_sync_hit_preserves_extras_and_skips_preprocess_and_forward(self):
        mm_input = MultimodalInput(
            "existing_input_key_url",
            1,
            torch.empty(0),
            MMPreprocessConfig(-1, -1, -1, -1, -1, -1, -1, [], 30000),
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
            MMPreprocessConfig(-1, -1, -1, -1, -1, -1, -1, [], 30000),
        )
        response = engine.get_embedding_result([mm_input], request_id=101)
        self.assertEqual(response[0].embeddings[0].shape, (4, 4))
        self.assertEqual(engine._remote_embedding_cache.stats()["miss"], 1)

    def test_remote_hit_does_not_bypass_greennet_rejection(self):
        mm_input = MultimodalInput(
            "rejected",
            1,
            torch.empty(0),
            MMPreprocessConfig(-1, -1, -1, -1, -1, -1, -1, [], 30000),
        )
        store = Store(
            {mm_input.cache_key(): pack_object(plan_object((torch.ones(3, 4), None)))}
        )
        engine = self.engine(store)
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


if __name__ == "__main__":
    unittest.main()
