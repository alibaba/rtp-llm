import concurrent.futures
import threading
import unittest
from unittest.mock import patch

import torch

from rtp_llm.multimodal.mm_embedding_cache import MMEmbeddingCache, MMHashKeyCache


class EmbeddingCapacityTest(unittest.TestCase):
    def test_variable_sizes_and_lru(self):
        cache = MMEmbeddingCache(gpu_max_bytes=0, cpu_max_bytes=64)
        entries = {}
        for key, rows in (("a", 2), ("b", 4), ("c", 2)):
            _, entry = cache.try_acquire(key)
            entry.complete((torch.full((rows, 2), float(rows)), None))
            entries[key] = entry
        self.assertEqual(cache.stats()["resident_bytes"], 64)
        cache.try_acquire("a")
        _, latest = cache.try_acquire("d")
        latest.complete((torch.zeros(4, 2), None))
        self.assertIsNone(cache.peek("b"))
        self.assertIs(cache.peek("a"), entries["a"])
        self.assertIs(cache.peek("c"), entries["c"])
        self.assertEqual(cache.stats()["resident_bytes"], 64)

    def test_backing_storage_and_extra_outputs_are_charged(self):
        cache = MMEmbeddingCache(gpu_max_bytes=0, cpu_max_bytes=512)
        batch = torch.ones(100)
        position = torch.ones(2, dtype=torch.int64)
        extra = torch.ones(8)
        _, entry = cache.try_acquire("view")
        # Two small, distinct views share and retain the 400-byte batch.
        result = ([batch[:2].view(1, 2), batch[10:12].view(1, 2)], position, extra)
        entry.complete(result)
        self.assertEqual(cache.stats()["resident_bytes"], 400 + 16 + 32)
        self.assertEqual(cache.stats()["resident_tokens"], 2)
        cache.remove("view")
        self.assertEqual(cache.stats()["resident_bytes"], 0)
        # Evicting the index entry does not invalidate a waiting request.
        self.assertIs(entry.wait(), result)

    def test_oversized_result_bypasses_cache_without_flushing_small_results(self):
        cache = MMEmbeddingCache(gpu_max_bytes=0, cpu_max_bytes=32)
        _, small = cache.try_acquire("small")
        small.complete(torch.ones(4))
        _, large = cache.try_acquire("large")
        result = torch.arange(100)
        large.complete(result)
        self.assertIs(cache.peek("small"), small)
        self.assertIsNone(cache.peek("large"))
        self.assertEqual(cache.stats()["resident_bytes"], 16)
        self.assertIs(large.wait(), result)

    def test_pending_dedup_survives_resize_and_disabled_completion(self):
        cache = MMEmbeddingCache(gpu_max_bytes=0, cpu_max_bytes=16)
        _, pending = cache.try_acquire("pending")
        self.assertIs(cache.try_acquire("pending")[1], pending)
        _, ready = cache.try_acquire("ready")
        ready.complete(torch.ones(4))
        cache.resize(0, 0)
        self.assertIsNone(cache.peek("ready"))
        self.assertEqual(cache.stats()["resident_bytes"], 0)
        pending.complete(torch.ones(2))
        self.assertIsNone(cache.peek("pending"))
        self.assertEqual(pending.wait().numel(), 2)
        cache.resize(0, 16)
        self.assertEqual(cache.try_acquire("pending")[0], "miss")

    def test_concurrent_acquire_and_completion(self):
        cache = MMEmbeddingCache(gpu_max_bytes=0, cpu_max_bytes=128)
        barrier = threading.Barrier(8)

        def acquire(_):
            barrier.wait(timeout=5)
            return cache.try_acquire("shared")

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            claims = list(pool.map(acquire, range(8)))
        self.assertEqual(sum(state == "miss" for state, _ in claims), 1)
        entry = claims[0][1]
        self.assertTrue(all(claimed is entry for _, claimed in claims))
        result = torch.ones(4)
        entry.complete(result)
        self.assertTrue(all(claimed.wait() is result for _, claimed in claims))

        def complete(i):
            _, current = cache.try_acquire(str(i))
            value = torch.full((i % 8 + 1,), i, dtype=torch.int32)
            current.complete(value)
            self.assertTrue(torch.equal(current.wait(), value))
            self.assertLessEqual(cache.stats()["resident_bytes"], 128)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(complete, range(80)))
        self.assertEqual(cache.stats()["pending_entries"], 0)

    def test_removed_pending_completion_cannot_charge_replacement(self):
        cache = MMEmbeddingCache(gpu_max_bytes=0, cpu_max_bytes=32)
        _, old = cache.try_acquire("key")
        cache.remove("key")
        _, new = cache.try_acquire("key")
        old.complete(torch.ones(100))
        new.complete(torch.ones(2))
        self.assertIs(cache.peek("key"), new)
        self.assertEqual(cache.stats()["resident_bytes"], 8)


@unittest.skipUnless(
    torch.cuda.is_available(), "requires CUDA for actual tier transfers"
)
class EmbeddingGpuCpuTest(unittest.TestCase):
    def add(self, cache, key, count=8):
        _, entry = cache.try_acquire(str(key))
        value = torch.full((count,), float(key), device="cuda")
        entry.complete(value)
        return entry

    def test_gpu_spill_cpu_promotion_and_cpu_lru_eviction(self):
        cache = MMEmbeddingCache(gpu_max_bytes=32, cpu_max_bytes=64)
        hashes = MMHashKeyCache(max_bytes=8192)
        a = self.add(cache, 1)
        hashes.put("1", [torch.tensor([91])], a.generation)
        b = self.add(cache, 2)
        c = self.add(cache, 3)
        self.assertEqual(a.result.device.type, "cpu")
        self.assertEqual(c.result.device.type, "cuda")
        before = cache.stats()
        self.assertTrue(hashes.metadata(["1"], cache)["entries"][0]["hit"])
        self.assertEqual(cache.stats(), before)
        self.assertEqual(cache.try_acquire("1")[0], "complete")
        restored = a.wait()
        self.assertEqual(restored.device, torch.device("cuda:0"))
        self.assertEqual(restored.tolist(), [1.0] * 8)
        self.assertEqual(c.result.device.type, "cpu")
        self.assertEqual(cache.stats()["promotion"], 1)
        self.assertEqual(hashes.get("1", a.generation)[0].tolist(), [91])
        self.add(cache, 4)
        self.assertIsNone(cache.peek("2"))
        self.assertEqual(cache.stats()["gpu_resident_bytes"], 32)
        self.assertEqual(cache.stats()["cpu_resident_bytes"], 64)
        # Both an already-returned tensor and an evicted entry remain usable.
        self.assertEqual(restored.tolist(), [1.0] * 8)
        self.assertEqual(b.wait().tolist(), [2.0] * 8)
        self.assertIsNone(cache.peek("2"))

    def test_mixed_devices_nested_outputs_views_and_dtypes_round_trip(self):
        cache = MMEmbeddingCache(gpu_max_bytes=512, cpu_max_bytes=2048)
        batch = torch.arange(128, device="cuda", dtype=torch.bfloat16)
        embedding = batch[8:16].view(2, 4)
        position = torch.arange(4, dtype=torch.int64)
        extra = torch.arange(6, device="cuda", dtype=torch.int16).reshape(2, 3).t()
        result = ([embedding, embedding], position, {"extra": extra, "none": None})
        _, entry = cache.try_acquire("mixed")
        entry.complete(result)
        generation = entry.generation
        self.add(cache, 2, count=128)
        self.assertEqual(entry.result[0][0].device.type, "cpu")
        self.assertIs(entry.result[0][0], entry.result[0][1])
        self.assertIs(entry.result[1], position)
        actual = entry.wait()
        self.assertEqual(entry.generation, generation)
        for expected, got in (
            (embedding, actual[0][0]),
            (position, actual[1]),
            (extra, actual[2]["extra"]),
        ):
            self.assertEqual(got.device, expected.device)
            self.assertEqual(got.dtype, expected.dtype)
            self.assertEqual(got.shape, expected.shape)
            self.assertTrue(torch.equal(got, expected))
        self.assertIs(actual[0][0], actual[0][1])
        self.assertIsNone(actual[2]["none"])
        self.assertLessEqual(cache.stats()["gpu_resident_bytes"], 512)
        self.assertLessEqual(cache.stats()["cpu_resident_bytes"], 2048)

    def test_disable_either_tier_and_oversized_results(self):
        for gpu_bytes in (0, 16):
            cache = MMEmbeddingCache(gpu_max_bytes=gpu_bytes, cpu_max_bytes=64)
            entry = self.add(cache, 1)
            self.assertEqual(entry.tier, "cpu")
            self.assertEqual(entry.wait().device.type, "cuda")
            self.assertEqual(entry.tier, "cpu")
            self.assertEqual(cache.stats()["gpu_resident_bytes"], 0)
            huge = self.add(cache, 2, count=100)
            self.assertIsNone(cache.peek("2"))
            self.assertIs(cache.peek("1"), entry)
            self.assertEqual(huge.wait().tolist(), [2.0] * 100)
        cache = MMEmbeddingCache(gpu_max_bytes=32, cpu_max_bytes=0)
        first = self.add(cache, 1)
        self.add(cache, 2)
        self.assertIsNone(cache.peek("1"))
        self.assertEqual(cache.stats()["cpu_resident_bytes"], 0)
        self.assertEqual(first.wait().device.type, "cuda")

    def test_concurrent_cpu_hit_copies_once(self):
        cache = MMEmbeddingCache(gpu_max_bytes=32, cpu_max_bytes=64)
        entry = self.add(cache, 1)
        self.add(cache, 2)
        barrier = threading.Barrier(8)

        def read(_):
            _, claimed = cache.try_acquire("1")
            barrier.wait(timeout=5)
            return claimed.wait(timeout=5)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(read, range(8)))
        self.assertTrue(all(value is results[0] for value in results))
        self.assertEqual(cache.stats()["promotion"], 1)
        self.assertEqual(entry.tier, "gpu")
        self.assertEqual(results[0].tolist(), [1.0] * 8)

    def test_concurrent_insert_hit_evict_and_clear_preserve_each_value(self):
        cache = MMEmbeddingCache(gpu_max_bytes=128, cpu_max_bytes=256)

        def request(i):
            key = str(i % 13)
            state, entry = cache.try_acquire(key)
            if state == "miss":
                entry.complete(torch.full((8,), float(key), device="cuda"))
            value = entry.wait(timeout=10)
            self.assertEqual(value.device.type, "cuda")
            self.assertEqual(value.tolist(), [float(key)] * 8)
            stats = cache.stats()
            self.assertLessEqual(stats["gpu_resident_bytes"], 128)
            self.assertLessEqual(stats["cpu_resident_bytes"], 256)
            if i % 19 == 0:
                cache.clear()

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(request, range(160)))
        cache.clear()
        self.assertEqual(cache.stats()["resident_bytes"], 0)

    def test_promotion_failure_keeps_cpu_generation_for_retry(self):
        cache = MMEmbeddingCache(gpu_max_bytes=32, cpu_max_bytes=64)
        entry = self.add(cache, 1)
        self.add(cache, 2)
        generation = entry.generation
        with patch(
            "rtp_llm.multimodal.mm_embedding_cache._copy_result_to_devices",
            side_effect=torch.OutOfMemoryError("test H2D allocation"),
        ):
            with self.assertRaises(torch.OutOfMemoryError):
                entry.wait()
        self.assertIs(cache.peek("1"), entry)
        self.assertEqual(entry.generation, generation)
        self.assertEqual(entry.result.device.type, "cpu")
        self.assertEqual(entry.wait().tolist(), [1.0] * 8)
        self.assertEqual(entry.tier, "gpu")

    def test_spill_failure_does_not_fail_completed_computation(self):
        cache = MMEmbeddingCache(gpu_max_bytes=32, cpu_max_bytes=64)
        a = self.add(cache, 1)
        with patch(
            "rtp_llm.multimodal.mm_embedding_cache._copy_result_to_devices",
            side_effect=MemoryError("test host allocation"),
        ):
            b = self.add(cache, 2)
        self.assertIsNone(cache.peek("1"))
        self.assertIs(cache.peek("2"), b)
        self.assertEqual(b.wait().tolist(), [2.0] * 8)
        self.assertEqual(a.wait().tolist(), [1.0] * 8)
        self.assertEqual(cache.stats()["transfer_error"], 1)

    def test_metadata_and_hot_reads_do_not_wait_for_an_unrelated_spill(self):
        from rtp_llm.multimodal.mm_embedding_cache import _copy_result_to_devices

        cache = MMEmbeddingCache(gpu_max_bytes=64, cpu_max_bytes=64)
        a = self.add(cache, 1)
        b = self.add(cache, 2)
        entered, release = threading.Event(), threading.Event()

        def copy(result, devices):
            entered.set()
            if not release.wait(timeout=5):
                raise TimeoutError("test did not release spill")
            return _copy_result_to_devices(result, devices)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            with patch(
                "rtp_llm.multimodal.mm_embedding_cache._copy_result_to_devices", copy
            ):
                inserting = pool.submit(self.add, cache, 3)
                try:
                    self.assertTrue(entered.wait(timeout=5))

                    def probe():
                        self.assertIs(cache.peek("1"), a)
                        self.assertEqual(
                            cache.try_acquire("2")[1].wait().tolist(), [2.0] * 8
                        )
                        cache.stats()

                    pool.submit(probe).result(timeout=2)
                finally:
                    release.set()
                inserting.result(timeout=5)
        self.assertEqual(a.tier, "cpu")
        self.assertEqual(b.tier, "gpu")

    def test_resize_spills_and_old_waiter_cannot_replace_new_generation(self):
        cache = MMEmbeddingCache(gpu_max_bytes=64, cpu_max_bytes=64)
        a = self.add(cache, 1)
        b = self.add(cache, 2)
        cache.resize(0, 64)
        self.assertEqual(a.tier, "cpu")
        self.assertEqual(b.tier, "cpu")
        cache.resize(0, 32)
        self.assertIsNone(cache.peek("1"))
        cache.resize(32, 64)
        replacement = self.add(cache, 1)
        self.assertNotEqual(replacement.generation, a.generation)
        a.wait()
        self.assertIs(cache.peek("1"), replacement)
        cache.resize(0, 0)
        self.assertEqual(cache.stats()["resident_bytes"], 0)
        self.assertEqual(cache.stats()["resident_entries"], 0)
        self.assertEqual(b.wait().tolist(), [2.0] * 8)

    def test_wait_timeout_includes_transfer_queue(self):
        cache = MMEmbeddingCache(gpu_max_bytes=32, cpu_max_bytes=64)
        entry = self.add(cache, 1)
        self.add(cache, 2)
        with cache._transfer_lock:
            with self.assertRaises(TimeoutError):
                entry.wait(timeout=0.01)
        self.assertEqual(entry.wait().tolist(), [1.0] * 8)

    def test_cpu_metadata_on_gpu_entries_counts_against_cpu_budget(self):
        cache = MMEmbeddingCache(gpu_max_bytes=64, cpu_max_bytes=8)
        for key in ("1", "2"):
            _, entry = cache.try_acquire(key)
            entry.complete((torch.ones(4, device="cuda"), torch.ones(2)))
        self.assertIsNone(cache.peek("1"))
        self.assertEqual(cache.stats()["gpu_resident_bytes"], 16)
        self.assertEqual(cache.stats()["cpu_resident_bytes"], 8)

    def test_non_default_producer_stream_and_storage_release(self):
        import weakref

        cache = MMEmbeddingCache(gpu_max_bytes=4096, cpu_max_bytes=8192)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            value = torch.arange(1024, device="cuda", dtype=torch.float32) * 3
            ref = weakref.ref(value)
            _, entry = cache.try_acquire("stream")
            entry.complete(value)
        del value
        self.add(cache, 2, count=1024)
        self.assertIsNone(ref())
        with torch.cuda.stream(torch.cuda.Stream()):
            actual = entry.wait()
            expected = torch.arange(1024, device="cuda", dtype=torch.float32) * 3
            self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(entry.tier, "gpu")


class HashCapacityTest(unittest.TestCase):
    def test_capacity_accounts_for_hash_length_and_python_metadata(self):
        cache = MMHashKeyCache(max_bytes=5120)
        for i in range(3):
            cache.put(str(i), [torch.ones(1)], "g")
        self.assertEqual(cache.stats()["resident_entries"], 3)
        self.assertGreater(cache.stats()["resident_bytes"], 3 * 4)
        cache.put("big", [torch.ones(1024)], "g")
        self.assertEqual(cache.keys(), ["big"])
        self.assertLessEqual(cache.stats()["resident_bytes"], 5120)
        self.assertGreater(cache.stats()["resident_bytes"], 1024 * 4)

    def test_replacement_resize_and_clear_balance_bytes(self):
        cache = MMHashKeyCache(max_bytes=8192)
        cache.put("a", [torch.ones(8)], "g")
        first_size = cache.stats()["resident_bytes"]
        cache.put("a", [torch.ones(32)], "g")
        self.assertEqual(cache.stats()["resident_bytes"], first_size + 24 * 4)
        cache.put("a", [torch.ones(8)], "g")
        self.assertEqual(cache.stats()["resident_bytes"], first_size)
        cache.put("b", [torch.ones(8)], "g")
        cache.resize(first_size)
        self.assertEqual(cache.keys(), ["b"])
        self.assertEqual(cache.stats()["resident_bytes"], first_size)
        cache.clear()
        self.assertEqual(cache.stats()["resident_bytes"], 0)
        self.assertEqual(cache.keys(), [])

    def test_real_hits_touch_lru_but_probes_and_wrong_generation_do_not(self):
        cache = MMHashKeyCache(max_bytes=8192)
        embeddings = MMEmbeddingCache(gpu_max_bytes=0, cpu_max_bytes=32)
        for key in ("a", "b"):
            _, entry = embeddings.try_acquire(key)
            entry.complete(torch.ones(1, 2))
            cache.put(key, [torch.ones(1)], entry.generation)
        limit = cache.stats()["resident_bytes"]
        cache.resize(limit)
        self.assertTrue(cache.metadata(["a"], embeddings)["entries"][0]["hit"])
        self.assertIsNone(cache.get("a", "wrong-generation"))
        self.assertEqual(cache.keys(), ["a", "b"])
        self.assertIsNotNone(cache.get("a", embeddings.peek("a").generation))
        self.assertEqual(cache.keys(), ["b", "a"])
        cache.put("c", [torch.ones(1)], embeddings.peek("a").generation)
        self.assertEqual(cache.keys(), ["a", "c"])

    def test_cpu_owned_int32_copies_and_generation(self):
        cache = MMHashKeyCache(max_bytes=8192)
        original = torch.tensor([-1, 2147483647], dtype=torch.int64)
        cache.put("a", [original], "g1")
        original.zero_()
        hashes = cache.get("a", "g1")
        self.assertEqual(hashes[0].tolist(), [-1, 2147483647])
        self.assertEqual(hashes[0].dtype, torch.int32)
        self.assertEqual(hashes[0].device.type, "cpu")
        self.assertIsNone(cache.get("a", "g2"))

    def test_oversized_update_drops_stale_generation_only(self):
        cache = MMHashKeyCache(max_bytes=4096)
        cache.put("old", [torch.ones(1)], "g1")
        cache.put("keep", [torch.ones(1)], "g1")
        cache.put("old", [torch.ones(4096)], "g2")
        self.assertEqual(cache.keys(), ["keep"])
        # Keys and generation strings count even without a hash payload.
        cache.put("x" * 8192, [], "g")
        self.assertEqual(cache.keys(), ["keep"])

    def test_embedding_eviction_invalidates_metadata_but_keeps_hash_history(self):
        cache = MMHashKeyCache(max_bytes=8192)
        embeddings = MMEmbeddingCache(gpu_max_bytes=0, cpu_max_bytes=8)
        _, entry = embeddings.try_acquire("a")
        entry.complete(torch.ones(1, 2))
        cache.put("a", [torch.ones(1)], entry.generation)
        self.assertTrue(cache.metadata(["a"], embeddings)["entries"][0]["hit"])
        embeddings.resize(0, 0)
        self.assertEqual(cache.keys(), ["a"])
        self.assertFalse(cache.metadata(["a"], embeddings)["entries"][0]["hit"])

    def test_concurrent_puts_replacements_and_resize_stay_bounded(self):
        cache = MMHashKeyCache(max_bytes=8192)

        def write(worker):
            for i in range(30):
                cache.put(str((worker + i) % 12), [torch.arange(i * 8)], str(i))
                self.assertLessEqual(cache.stats()["resident_bytes"], 8192)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(write, range(8)))
        cache.resize(0)
        cache.put("disabled", [torch.ones(1)])
        self.assertEqual(cache.stats()["resident_bytes"], 0)
        self.assertEqual(cache.keys(), [])

    def test_disabled_and_negative_budgets(self):
        with self.assertRaises(ValueError):
            MMHashKeyCache(max_bytes=-1)
        cache = MMHashKeyCache(max_bytes=0)
        self.assertFalse(cache.enabled)
        with self.assertRaises(ValueError):
            cache.resize(-1)
        for gpu, cpu in ((-1, 0), (0, -1)):
            with self.assertRaises(ValueError):
                MMEmbeddingCache(gpu_max_bytes=gpu, cpu_max_bytes=cpu)
        cache = MMEmbeddingCache(gpu_max_bytes=0, cpu_max_bytes=0)
        self.assertFalse(cache.enabled)
        with self.assertRaises(ValueError):
            cache.resize(0, -1)


if __name__ == "__main__":
    unittest.main()
