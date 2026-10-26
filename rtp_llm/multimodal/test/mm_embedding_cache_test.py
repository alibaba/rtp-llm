import concurrent.futures
import gc
import random
import threading
import unittest
import weakref
from contextlib import contextmanager
from unittest.mock import patch

import torch

from rtp_llm.multimodal import mm_embedding_cache as cache_module
from rtp_llm.multimodal.mm_embedding_cache import (
    MMEmbeddingCache,
    MMEmbeddingCacheEntry,
    MMHashKeyCache,
    _current_cuda_streams,
    _PoolCapacityError,
    _TensorMemoryPool,
)


class TensorMemoryPoolFreeListTest(unittest.TestCase):
    def test_alignment_and_two_sided_coalescing(self):
        pool = _TensorMemoryPool(64, torch.device("cpu"))
        first, middle, last = [pool.reserve(9) for _ in range(3)]
        self.assertEqual([b.offset for b in (first, middle, last)], [0, 16, 32])
        pool.release(first, [])
        pool.release(last, [])
        self.assertEqual(pool._free, [(0, 16), (25, 39)])
        pool.release(middle, [])
        self.assertEqual(pool._free, [(0, 64)])
        self.assertEqual(pool.reserve(64).offset, 0)
        self.assertEqual(pool._free, [])

    def test_random_operations_match_full_sort_and_coalesce(self):
        # Differential check against the previous allocator: identical first-fit
        # placement, failures and free blocks imply identical fragmentation.
        for seed in (11, 29, 47):
            rng = random.Random(seed)
            capacity = 4096
            pool = _TensorMemoryPool(capacity, torch.device("cpu"))
            reference = [(0, capacity)]
            live = []

            def insert(offset, size):
                if not size:
                    return
                reference.append((offset, size))
                reference.sort()
                merged = []
                for start, count in reference:
                    if merged and sum(merged[-1]) == start:
                        merged[-1] = (merged[-1][0], merged[-1][1] + count)
                    else:
                        merged.append((start, count))
                reference[:] = merged

            def reserve(size):
                if size == 0:
                    return None
                for index, (offset, count) in enumerate(reference):
                    aligned = (offset + 7) // 8 * 8
                    if count - (aligned - offset) >= size:
                        reference.pop(index)
                        insert(offset, aligned - offset)
                        insert(aligned + size, offset + count - aligned - size)
                        return aligned
                raise _PoolCapacityError("reference full")

            for _ in range(3000):
                if live and rng.random() < 0.47:
                    block = live.pop(rng.randrange(len(live)))
                    insert(block.offset, block.size)
                    pool.release(block, [])
                else:
                    size = rng.randrange(0, 600)
                    try:
                        expected = reserve(size)
                    except _PoolCapacityError:
                        with self.assertRaises(_PoolCapacityError):
                            pool.reserve(size)
                    else:
                        actual = pool.reserve(size)
                        if expected is None:
                            self.assertIsNone(actual)
                        else:
                            self.assertEqual(actual.offset, expected)
                            self.assertEqual(actual.offset % 8, 0)
                            self.assertEqual(actual.size, size)
                            live.append(actual)
                self.assertEqual(pool._free, reference)
                used = sum(block.size for block in live)
                self.assertEqual(pool.stats()["used_bytes"], used)
                self.assertEqual(used + sum(n for _, n in pool._free), capacity)
                intervals = sorted(
                    [(block.offset, block.size) for block in live] + pool._free
                )
                self.assertEqual(intervals[0][0], 0)
                self.assertEqual(sum(intervals[-1]), capacity)
                for left, right in zip(intervals, intervals[1:]):
                    self.assertEqual(sum(left), right[0])
            rng.shuffle(live)
            for block in live:
                pool.release(block, [])
            self.assertEqual(pool._free, [(0, capacity)])

    def test_pending_event_delays_reuse_and_coalesces_after_wait(self):
        class Event:
            ready = False
            waits = 0

            def query(self):
                return self.ready

            def synchronize(self):
                self.waits += 1
                self.ready = True

        pool = _TensorMemoryPool(64, torch.device("cpu"))
        left, right = pool.reserve(32), pool.reserve(32)
        event = Event()
        pool.release(left, [event])
        pool.release(right, [])
        self.assertEqual(pool._free, [(32, 32)])
        self.assertEqual(pool.stats()["pending_bytes"], 32)
        whole = pool.reserve(64)
        self.assertEqual((whole.offset, whole.size), (0, 64))
        self.assertEqual(event.waits, 1)
        self.assertEqual(pool.stats()["pending_bytes"], 0)
        pool.release(whole, [])
        self.assertEqual(pool._free, [(0, 64)])


def _assert_read_copies_are_released(testcase, entry, device_type):
    gc_enabled = gc.isenabled()
    gc.disable()
    try:
        refs = []
        for _ in range(16):
            result = entry.wait()
            testcase.assertEqual(result.device.type, device_type)
            testcase.assertEqual(result.shape, (16384,))
            testcase.assertEqual(result[0].item(), 7.0)
            refs.append(weakref.ref(result))
            del result
        testcase.assertTrue(all(ref() is None for ref in refs))
    finally:
        if gc_enabled:
            gc.enable()


class UncachedEntryReadinessTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_wait_observes_output_from_another_stream(self):
        for stream_ordered in (False, True):
            with self.subTest(stream_ordered=stream_ordered):
                entry = MMEmbeddingCacheEntry()
                producer = torch.cuda.Stream()
                consumer = torch.cuda.Stream()
                with torch.cuda.stream(producer):
                    result = torch.full((128, 128), 7.0, device="cuda")
                    entry.producer_streams = _current_cuda_streams(result)
                entry.complete(result)
                with torch.cuda.stream(consumer):
                    actual = entry.wait(wait_on_current_stream=stream_ordered)
                    self.assertTrue(
                        torch.equal(actual.cpu(), torch.full((128, 128), 7.0))
                    )
                    self.assertTrue(
                        all(stream.query() for stream in entry.producer_streams)
                    )


class EmbeddingCapacityTest(unittest.TestCase):
    def test_before_publish_runs_once_after_storage_before_ready(self):
        cache = MMEmbeddingCache(gpu_max_bytes=0, cpu_max_bytes=64)
        _, entry = cache.try_acquire("a")
        called = []

        def publish():
            self.assertEqual(entry.tier, "cpu")
            self.assertFalse(entry.is_done)
            self.assertEqual(cache.try_acquire("a")[0], "in_progress")
            with self.assertRaises(TimeoutError):
                entry.wait_ready(timeout=0)
            called.append(True)

        self.assertTrue(
            cache.complete("a", entry, torch.ones(4), before_publish=publish)
        )
        self.assertFalse(
            cache.complete("a", entry, torch.zeros(4), before_publish=publish)
        )
        self.assertEqual(called, [True])
        self.assertTrue(torch.equal(entry.wait(timeout=0), torch.ones(4)))

    def test_before_publish_error_rolls_back_and_wakes_waiters(self):
        cache = MMEmbeddingCache(gpu_max_bytes=0, cpu_max_bytes=64)
        _, entry = cache.try_acquire("a")

        def publish():
            raise RuntimeError("hash publication failed")

        with self.assertRaisesRegex(RuntimeError, "hash publication failed"):
            cache.complete("a", entry, torch.ones(4), before_publish=publish)
        with self.assertRaisesRegex(RuntimeError, "hash publication failed"):
            entry.wait(timeout=0)
        self.assertIsNone(cache.peek("a"))
        self.assertEqual(entry.pool_owners, [])
        self.assertEqual(cache.stats()["cpu_pool_used_bytes"], 0)

    def test_cache_reads_release_outputs_without_cyclic_gc(self):
        cache = MMEmbeddingCache(gpu_max_bytes=0, cpu_max_bytes=65536)
        _, entry = cache.try_acquire("image")
        entry.complete(torch.full((16384,), 7.0))
        _assert_read_copies_are_released(self, entry, "cpu")

    def test_completion_failure_rolls_back_before_and_after_admission(self):
        for after_admission in (False, True):
            with self.subTest(after_admission=after_admission):
                cache = MMEmbeddingCache(gpu_max_bytes=0, cpu_max_bytes=64)
                original = cache._on_complete
                refs = []

                def fail_completion(key, entry, result):
                    if after_admission:
                        original(key, entry, result)
                    raise RuntimeError("cache completion failed")

                with patch.object(cache, "_on_complete", new=fail_completion):
                    for index in range(8):
                        key = f"failed-{index}"
                        _, entry = cache.try_acquire(key)
                        result = torch.ones(4)
                        refs.append(weakref.ref(result))
                        with self.assertRaisesRegex(RuntimeError, "completion failed"):
                            entry.complete(result)
                        with self.assertRaisesRegex(RuntimeError, "completion failed"):
                            entry.wait(timeout=0)
                        self.assertTrue(entry.is_done)
                        self.assertIsNone(entry.result)
                        self.assertIsNone(cache.peek(key))
                        self.assertEqual(entry.pool_owners, [])
                        del result, entry
                gc.collect()
                self.assertTrue(all(ref() is None for ref in refs))
                stats = cache.stats()
                self.assertEqual(stats["pending_entries"], 0)
                self.assertEqual(stats["resident_bytes"], 0)
                self.assertEqual(stats["cpu_pool_used_bytes"], 0)
                state, replacement = cache.try_acquire("failed-0")
                self.assertEqual(state, "miss")
                replacement.complete(torch.ones(4))
                self.assertTrue(torch.equal(replacement.wait(), torch.ones(4)))

    def test_completion_is_not_published_while_callback_is_running(self):
        cache = MMEmbeddingCache(gpu_max_bytes=0, cpu_max_bytes=64)
        _, entry = cache.try_acquire("pending")
        started, release = threading.Event(), threading.Event()
        original = cache._on_complete

        def slow_completion(key, current, result):
            started.set()
            self.assertTrue(release.wait(timeout=2))
            original(key, current, result)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            with patch.object(cache, "_on_complete", new=slow_completion):
                completed = pool.submit(entry.complete, torch.ones(4))
                try:
                    self.assertTrue(started.wait(timeout=2))
                    self.assertFalse(entry.is_done)
                    self.assertEqual(cache.try_acquire("pending")[0], "in_progress")
                    self.assertFalse(entry.complete(torch.zeros(4)))
                    self.assertFalse(entry.fail(RuntimeError("duplicate transition")))
                    with self.assertRaises(TimeoutError):
                        entry.wait_ready(timeout=0)
                finally:
                    release.set()
                self.assertTrue(completed.result(timeout=2))
        self.assertTrue(entry.is_done)
        self.assertTrue(torch.equal(entry.wait(), torch.ones(4)))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_active_tensor_survives_demotion_eviction_and_same_key_replacement(self):
        cache = MMEmbeddingCache(gpu_max_bytes=128, cpu_max_bytes=128)
        _, entry = cache.try_acquire("image")
        expected = torch.arange(32, dtype=torch.float32, device="cuda").reshape(4, 8)
        entry.complete((expected, None))
        held = entry.wait()[0]
        cache.resize(0, 128)
        self.assertEqual(entry.result[0].device.type, "cpu")
        cache.resize(0, 0)
        cache.resize(128, 128)
        _, replacement = cache.try_acquire("image")
        replacement.complete((torch.full((4, 8), 99.0, device="cuda"), None))
        self.assertNotEqual(held.data_ptr(), replacement.wait()[0].data_ptr())
        self.assertTrue(
            torch.equal(
                held, torch.arange(32, dtype=torch.float32, device="cuda").reshape(4, 8)
            )
        )
        cache.clear()
        self.assertTrue(torch.equal(held, expected))

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
        # Pool insertion compacts views instead of retaining the 400-byte batch.
        result = ([batch[:2].view(1, 2), batch[10:12].view(1, 2)], position, extra)
        entry.complete(result)
        self.assertEqual(cache.stats()["resident_bytes"], 8 + 8 + 16 + 32)
        self.assertEqual(cache.stats()["resident_tokens"], 2)
        cache.remove("view")
        self.assertEqual(cache.stats()["resident_bytes"], 0)
        # Evicting the index entry does not invalidate a waiting request.
        actual = entry.wait()
        self.assertEqual(actual[0][0].tolist(), [[1.0, 1.0]])
        self.assertEqual(actual[0][1].tolist(), [[1.0, 1.0]])

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
        self.assertTrue(
            all(torch.equal(claimed.wait(), result) for _, claimed in claims)
        )

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

    def test_cpu_pool_reuses_one_arena_without_mutating_returned_values(self):
        cache = MMEmbeddingCache(gpu_max_bytes=0, cpu_max_bytes=32)
        arena_ptr = cache._cpu_pool._storage.data_ptr()
        _, first = cache.try_acquire("first")
        first.complete(torch.arange(8, dtype=torch.float32))
        held = first.wait()
        cache.remove("first")

        _, second = cache.try_acquire("second")
        second.complete(torch.full((8,), 9.0))
        self.assertEqual(cache._cpu_pool._storage.data_ptr(), arena_ptr)
        self.assertEqual(held.tolist(), list(map(float, range(8))))
        self.assertEqual(second.wait().tolist(), [9.0] * 8)
        stats = cache.stats()
        self.assertEqual(stats["cpu_pool_capacity_bytes"], 32)
        self.assertEqual(stats["cpu_pool_used_bytes"], 32)

    def test_cpu_pool_evicts_lru_to_coalesce_fragmented_space(self):
        cache = MMEmbeddingCache(gpu_max_bytes=0, cpu_max_bytes=32)
        entries = {}
        for key in ("a", "b", "c", "d"):
            _, entry = cache.try_acquire(key)
            entry.complete(torch.ones(2))
            entries[key] = entry
        cache.remove("b")
        cache.remove("d")

        _, large = cache.try_acquire("large")
        large.complete(torch.arange(4, dtype=torch.float32))
        self.assertIsNone(cache.peek("a"))
        self.assertIs(cache.peek("c"), entries["c"])
        self.assertIs(cache.peek("large"), large)
        self.assertEqual(large.wait().tolist(), [0.0, 1.0, 2.0, 3.0])
        self.assertEqual(cache.stats()["cpu_pool_used_bytes"], 24)


@unittest.skipUnless(
    torch.cuda.is_available(), "requires CUDA for actual tier transfers"
)
class EmbeddingGpuCpuTest(unittest.TestCase):
    def test_gpu_reads_release_outputs_without_cyclic_gc(self):
        for gpu_bytes, evict_gpu in ((65536, False), (0, False), (65536, True)):
            with self.subTest(gpu_bytes=gpu_bytes, evict_gpu=evict_gpu):
                cache = MMEmbeddingCache(gpu_bytes, 131072)
                _, entry = cache.try_acquire("image")
                entry.complete(torch.full((16384,), 7.0, device="cuda"))
                if evict_gpu:
                    self.add(cache, 2, count=16384)
                before = torch.cuda.memory_allocated()
                _assert_read_copies_are_released(self, entry, "cuda")
                self.assertLessEqual(torch.cuda.memory_allocated(), before)

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
        self.assertIsNot(entry.result[1], position)
        self.assertTrue(torch.equal(entry.result[1], position))
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

    def test_concurrent_cpu_hits_promote_once_and_return_independent_values(self):
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
        self.assertTrue(all(value.tolist() == [1.0] * 8 for value in results))
        self.assertEqual(len({value.data_ptr() for value in results}), len(results))
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
            "rtp_llm.multimodal.mm_embedding_cache._copy_pooled_result_to_devices",
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
        from rtp_llm.multimodal.mm_embedding_cache import _pool_result

        cache = MMEmbeddingCache(gpu_max_bytes=32, cpu_max_bytes=64)
        a = self.add(cache, 1)

        calls = 0

        def fail_first(result, devices, gpu_pool, cpu_pool):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise MemoryError("test host allocation")
            return _pool_result(result, devices, gpu_pool, cpu_pool)

        with patch(
            "rtp_llm.multimodal.mm_embedding_cache._pool_result",
            side_effect=fail_first,
        ):
            b = self.add(cache, 2)
        self.assertIsNone(cache.peek("1"))
        self.assertIs(cache.peek("2"), b)
        self.assertEqual(b.wait().tolist(), [2.0] * 8)
        self.assertEqual(a.wait().tolist(), [1.0] * 8)
        self.assertEqual(cache.stats()["transfer_error"], 1)

    def test_metadata_and_hot_reads_do_not_wait_for_an_unrelated_spill(self):
        from rtp_llm.multimodal.mm_embedding_cache import _pool_result

        cache = MMEmbeddingCache(gpu_max_bytes=64, cpu_max_bytes=64)
        a = self.add(cache, 1)
        b = self.add(cache, 2)
        entered, release = threading.Event(), threading.Event()

        def copy(result, devices, gpu_pool, cpu_pool):
            entered.set()
            if not release.wait(timeout=5):
                raise TimeoutError("test did not release spill")
            return _pool_result(result, devices, gpu_pool, cpu_pool)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            with patch("rtp_llm.multimodal.mm_embedding_cache._pool_result", copy):
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
    def test_live_generation_publication_is_idempotent_without_tensor_copies(self):
        hashes = MMHashKeyCache(max_bytes=8192)
        embeddings = MMEmbeddingCache(gpu_max_bytes=0, cpu_max_bytes=64)
        _, entry = embeddings.try_acquire("a")
        entry.complete(torch.ones(4))
        value = [torch.tensor([1, 2], dtype=torch.int32)]
        hashes.put_for_entry("a", value, entry, embeddings)
        stored = hashes._entries["a"]
        before = hashes.stats()
        with patch.object(
            torch.Tensor, "clone", side_effect=AssertionError("duplicate copy")
        ):
            for _ in range(20):
                hashes.put_for_entry("a", value, entry, embeddings)
        self.assertIs(hashes._entries["a"], stored)
        self.assertEqual(hashes.stats(), before)

    def test_live_generation_approval_never_downgrades_or_crosses_generation(self):
        from rtp_llm.multimodal.greennet_hook import GreenNetVerdict

        hashes = MMHashKeyCache(max_bytes=8192)
        embeddings = MMEmbeddingCache(gpu_max_bytes=0, cpu_max_bytes=64)
        _, old = embeddings.try_acquire("a")
        old.complete(torch.ones(4))
        value = [torch.tensor([1], dtype=torch.int32)]
        hashes.put_for_entry("a", value, old, embeddings)
        self.assertFalse(hashes.greennet_passed("a"))
        old.set_greennet_verdict(GreenNetVerdict(passed=True))
        hashes.put_for_entry("a", value, old, embeddings)
        old.set_greennet_verdict(GreenNetVerdict(passed=True), checked=False)
        hashes.put_for_entry("a", value, old, embeddings)
        self.assertTrue(hashes.greennet_passed("a"))
        embeddings.remove("a")
        _, new = embeddings.try_acquire("a")
        new.complete(torch.ones(4))
        hashes.put_for_entry("a", [torch.tensor([2])], new, embeddings)
        old.set_greennet_verdict(GreenNetVerdict(passed=True))
        hashes.put_for_entry("a", value, old, embeddings)
        hashes.put_for_entry("a", [torch.ones(8192)], old, embeddings)
        self.assertFalse(hashes.greennet_passed("a"))
        self.assertEqual(hashes.get("a", new.generation)[0].tolist(), [2])

    def test_stale_writer_during_cpu_copy_cannot_replace_new_generation(self):
        hashes = MMHashKeyCache(max_bytes=8192)
        embeddings = MMEmbeddingCache(gpu_max_bytes=0, cpu_max_bytes=64)
        _, old = embeddings.try_acquire("a")
        old.complete(torch.ones(4))
        cloning, release = threading.Event(), threading.Event()
        original_clone = torch.Tensor.clone

        def clone(tensor, *args, **kwargs):
            if threading.current_thread().name.startswith("old-publisher"):
                cloning.set()
                if not release.wait(timeout=5):
                    raise TimeoutError("publisher not released")
            return original_clone(tensor, *args, **kwargs)

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="old-publisher"
        ) as pool:
            with patch.object(torch.Tensor, "clone", new=clone):
                future = pool.submit(
                    hashes.put_for_entry, "a", [torch.tensor([1])], old, embeddings
                )
                try:
                    self.assertTrue(cloning.wait(timeout=5))
                    embeddings.remove("a")
                    _, new = embeddings.try_acquire("a")
                    new.complete(torch.ones(4))
                    hashes.put_for_entry("a", [torch.tensor([2])], new, embeddings)
                finally:
                    release.set()
                future.result(timeout=5)
        self.assertIsNone(hashes.get("a", old.generation))
        self.assertEqual(hashes.get("a", new.generation)[0].tolist(), [2])

    def test_concurrent_publications_allocate_only_once(self):
        hashes = MMHashKeyCache(max_bytes=8192)
        embeddings = MMEmbeddingCache(gpu_max_bytes=0, cpu_max_bytes=64)
        _, entry = embeddings.try_acquire("a")
        entry.complete(torch.ones(4))
        cloning, release = threading.Event(), threading.Event()
        original_clone = torch.Tensor.clone

        def clone(tensor, *args, **kwargs):
            cloning.set()
            if not release.wait(timeout=5):
                raise TimeoutError("publisher not released")
            return original_clone(tensor, *args, **kwargs)

        with patch.object(torch.Tensor, "clone", new=clone), patch.object(
            hashes._pool, "reserve", wraps=hashes._pool.reserve
        ) as reserve:
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                args = ("a", [torch.tensor([1, 2])], entry, embeddings)
                first = pool.submit(hashes.put_for_entry, *args)
                try:
                    self.assertTrue(cloning.wait(timeout=5))
                    futures = [
                        pool.submit(hashes.put_for_entry, *args) for _ in range(7)
                    ]
                    for future in futures:
                        future.result(timeout=5)
                    self.assertGreater(hashes.stats()["building_bytes"], 0)
                finally:
                    release.set()
                first.result(timeout=5)
            self.assertEqual(reserve.call_count, 1)
        self.assertEqual(hashes.get("a", entry.generation)[0].tolist(), [1, 2])

    def test_approval_follows_hash_lifetime_and_replacement(self):
        hashes = MMHashKeyCache(max_bytes=8192)
        embeddings = MMEmbeddingCache(gpu_max_bytes=0, cpu_max_bytes=32)
        _, entry = embeddings.try_acquire("a")
        entry.complete(torch.ones(2))
        hashes.put("a", [torch.tensor([1, 2])], entry.generation, greennet_passed=True)
        embeddings.clear()
        item = hashes.metadata(["a"], embeddings)["entries"][0]
        self.assertTrue(item["greennet_passed"])
        self.assertFalse(item["embedding_hit"])
        self.assertTrue(hashes.greennet_passed("a"))
        hashes.put("a", [torch.tensor([3])], "new-uninspected")
        self.assertFalse(hashes.greennet_passed("a"))
        hashes.put("a", [torch.tensor([4])], "approved", greennet_passed=True)
        hashes.resize(0)
        self.assertFalse(hashes.greennet_passed("a"))
        self.assertFalse(
            hashes.metadata(["a"], embeddings)["entries"][0]["greennet_passed"]
        )

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

    def test_hash_pool_reuse_cannot_overwrite_a_returned_hash(self):
        cache = MMHashKeyCache(max_bytes=8192)
        arena_ptr = cache._pool._storage.data_ptr()
        cache.put("a", [torch.tensor([1, 2, 3])], "g1")
        held = cache.get("a", "g1")[0]
        cache.clear()
        cache.put("b", [torch.tensor([7, 8, 9])], "g2")
        self.assertEqual(cache._pool._storage.data_ptr(), arena_ptr)
        self.assertEqual(held.tolist(), [1, 2, 3])
        self.assertEqual(cache.get("b", "g2")[0].tolist(), [7, 8, 9])
        self.assertLessEqual(
            cache.stats()["pool_used_bytes"], cache.stats()["pool_capacity_bytes"]
        )

    def test_oversized_update_drops_stale_generation_only(self):
        cache = MMHashKeyCache(max_bytes=4096)
        cache.put("old", [torch.ones(1)], "g1")
        cache.put("keep", [torch.ones(1)], "g1")
        cache.put("old", [torch.ones(4096)], "g2")
        self.assertEqual(cache.keys(), ["keep"])
        # Keys and generation strings count even without a hash payload.
        cache.put("x" * 8192, [], "g")
        self.assertEqual(cache.keys(), ["keep"])

    def test_hash_metadata_survives_embedding_eviction(self):
        cache = MMHashKeyCache(max_bytes=8192)
        embeddings = MMEmbeddingCache(gpu_max_bytes=0, cpu_max_bytes=8)
        _, entry = embeddings.try_acquire("a")
        entry.complete(torch.ones(1, 2))
        cache.put("a", [torch.ones(1)], entry.generation)
        self.assertTrue(cache.metadata(["a"], embeddings)["entries"][0]["hit"])
        embeddings.resize(0, 0)
        self.assertEqual(cache.keys(), ["a"])
        metadata = cache.metadata(["a"], embeddings)["entries"][0]
        self.assertTrue(metadata["hit"])
        self.assertTrue(metadata["hash_hit"])
        self.assertFalse(metadata["embedding_hit"])
        self.assertIsNone(metadata["embedding_tier"])
        self.assertEqual(metadata["feature_hashes"], [1])
        self.assertEqual(metadata["entry_generation"], entry.generation)

    def test_hash_and_embedding_availability_are_independent(self):
        hashes = MMHashKeyCache(max_bytes=8192)
        embeddings = MMEmbeddingCache(gpu_max_bytes=0, cpu_max_bytes=32)
        _, ready = embeddings.try_acquire("ready")
        ready.complete(torch.ones(2))
        hashes.put("history", [torch.tensor([71, 72])], "old-generation")
        embeddings.try_acquire("history")
        before = embeddings.stats(), hashes.keys(), hashes.stats()
        entries = hashes.metadata(["ready", "history", "absent"], embeddings)["entries"]
        self.assertFalse(entries[0]["hash_hit"])
        self.assertTrue(entries[0]["embedding_hit"])
        self.assertEqual(entries[0]["embedding_tier"], "cpu")
        self.assertTrue(entries[1]["hash_hit"])
        self.assertFalse(entries[1]["embedding_hit"])
        self.assertEqual(entries[1]["feature_hashes"], [71, 72])
        self.assertFalse(entries[2]["hash_hit"])
        self.assertFalse(entries[2]["embedding_hit"])
        self.assertEqual(before, (embeddings.stats(), hashes.keys(), hashes.stats()))

    def test_residency_snapshot_excludes_pending_and_does_not_touch_lru(self):
        embeddings = MMEmbeddingCache(gpu_max_bytes=0, cpu_max_bytes=16)
        for key in ("a", "b"):
            _, entry = embeddings.try_acquire(key)
            entry.complete(torch.ones(2))
        embeddings.try_acquire("pending")
        before = embeddings.stats()
        self.assertEqual(embeddings.resident_tiers(limit=1), {"b": "cpu"})
        self.assertEqual(
            embeddings.resident_tiers(["a", "pending", "absent"]), {"a": "cpu"}
        )
        self.assertEqual(embeddings.resident_tiers(limit=0), {})
        self.assertEqual(before, embeddings.stats())
        _, latest = embeddings.try_acquire("c")
        latest.complete(torch.ones(2))
        self.assertIsNone(embeddings.peek("a"))

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


class CacheLockIsolationTest(unittest.TestCase):
    @contextmanager
    def paused(self, target, method):
        entered, release = threading.Event(), threading.Event()
        original = getattr(target, method)

        def slow(*args, **kwargs):
            if threading.current_thread().name.startswith("cache-slow"):
                entered.set()
                if not release.wait(10):
                    raise TimeoutError("test did not release slow operation")
            return original(*args, **kwargs)

        with patch.object(
            target, method, new=slow
        ), concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="cache-slow"
        ) as pool:
            try:
                yield pool, entered
            finally:
                release.set()

    def quick(self, function, *args):
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(function, *args).result(timeout=2)

    def test_read_and_metadata_leases_survive_clear_and_pool_reuse(self):
        for metadata in (False, True):
            with self.subTest(metadata=metadata):
                hashes = MMHashKeyCache(8192)
                embeddings = MMEmbeddingCache(0, 64)
                hashes.put("a", [torch.tensor([1, 2, 3])], "old", True)
                hashes.put("b", [torch.tensor([4])], "other")
                with self.paused(torch.Tensor, "clone") as (pool, entered):
                    read = (
                        pool.submit(hashes.metadata, ["a"], embeddings)
                        if metadata
                        else pool.submit(hashes.get, "a")
                    )
                    self.assertTrue(entered.wait(5))
                    self.assertEqual(self.quick(hashes.get, "b")[0].tolist(), [4])
                    self.quick(hashes.clear)
                    self.assertGreater(hashes.stats()["retired_bytes"], 0)
                    self.quick(hashes.put, "a", [torch.tensor([9, 8, 7])], "new")
                    self.assertEqual(self.quick(hashes.get, "a")[0].tolist(), [9, 8, 7])
                result = read.result(5)
                if metadata:
                    row = result["entries"][0]
                    self.assertEqual(row["feature_hashes"], [1, 2, 3])
                    self.assertEqual(row["entry_generation"], "old")
                    self.assertTrue(row["greennet_passed"])
                else:
                    self.assertEqual(result[0].tolist(), [1, 2, 3])
                self.assertEqual(hashes.stats()["retired_bytes"], 0)

    def test_pinned_bytes_are_not_reallocated_after_shrink(self):
        hashes = MMHashKeyCache(8192)
        hashes.put("a", [torch.arange(256)], "g")
        limit = hashes.stats()["resident_bytes"]
        hashes.resize(limit)
        with self.paused(torch.Tensor, "clone") as (pool, entered):
            read = pool.submit(hashes.get, "a")
            self.assertTrue(entered.wait(5))
            self.quick(hashes.resize, 0)
            self.quick(hashes.resize, limit)
            self.quick(hashes.put, "b", [torch.ones(256)], "g")
            self.assertFalse(hashes.contains("b"))
            self.assertEqual(hashes.stats()["retired_bytes"], limit)
        self.assertEqual(read.result(5)[0].tolist(), list(range(256)))
        hashes.put("b", [torch.ones(256)], "g")
        self.assertTrue(hashes.contains("b"))

    def test_growth_waits_for_reader_without_holding_index(self):
        hashes = MMHashKeyCache(4096)
        hashes.put("a", [torch.tensor([17])])
        old_pool = hashes._pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as resizer:
            with self.paused(torch.Tensor, "clone") as (pool, entered):
                read = pool.submit(hashes.get, "a")
                self.assertTrue(entered.wait(5))
                drained = threading.Event()
                original_release = hashes._release_retired

                def release(values):
                    original_release(values)
                    drained.set()

                with patch.object(hashes, "_release_retired", side_effect=release):
                    growth = resizer.submit(hashes.resize, 16384)
                    self.assertTrue(drained.wait(5))
                    self.assertFalse(growth.done())
                    self.assertEqual(self.quick(hashes.keys), [])
                    self.assertIs(hashes._pool, old_pool)
                    self.quick(hashes.put, "b", [torch.tensor([18])])
                    self.assertFalse(hashes.contains("b"))
            self.assertEqual(read.result(5)[0].tolist(), [17])
            growth.result(5)
        self.assertEqual(hashes.stats()["pool_capacity_bytes"], 16384)
        self.assertEqual(hashes.stats()["retired_bytes"], 0)

    def test_pool_copy_is_unlocked_and_clear_invalidates_builder(self):
        hashes = MMHashKeyCache(8192)
        hashes.put("b", [torch.tensor([4])])
        with self.paused(cache_module, "_pool_result") as (pool, entered):
            write = pool.submit(hashes.put, "a", [torch.tensor([1])], "old")
            self.assertTrue(entered.wait(5))
            self.assertEqual(self.quick(hashes.get, "b")[0].tolist(), [4])
            self.assertGreater(hashes.stats()["building_bytes"], 0)
            self.quick(hashes.clear)
            self.quick(hashes.put, "a", [torch.tensor([2])], "new")
        write.result(5)
        self.assertEqual(hashes.get("a", "new")[0].tolist(), [2])
        self.assertEqual(hashes.stats()["building_bytes"], 0)
        self.assertEqual(hashes.stats()["retired_bytes"], 0)

    def test_resize_invalidates_builder_and_releases_reservation(self):
        hashes = MMHashKeyCache(8192)
        with self.paused(cache_module, "_pool_result") as (pool, entered):
            write = pool.submit(hashes.put, "a", [torch.arange(256)])
            self.assertTrue(entered.wait(5))
            self.quick(hashes.resize, 0)
        write.result(5)
        stats = hashes.stats()
        self.assertEqual(stats["building_bytes"], 0)
        self.assertEqual(stats["pool_used_bytes"], 0)
        self.assertEqual(hashes.keys(), [])

    def test_copy_exceptions_release_leases_and_builder_budget(self):
        hashes = MMHashKeyCache(8192)
        hashes.put("a", [torch.tensor([1])])
        with patch.object(
            torch.Tensor, "clone", side_effect=RuntimeError("copy failed")
        ):
            with self.assertRaisesRegex(RuntimeError, "copy failed"):
                hashes.get("a")
        hashes.clear()
        with patch.object(
            cache_module, "_pool_result", side_effect=RuntimeError("copy failed")
        ):
            with self.assertRaisesRegex(RuntimeError, "copy failed"):
                hashes.put("a", [torch.tensor([1])])
        stats = hashes.stats()
        for key in (
            "building_bytes",
            "retired_bytes",
            "resident_bytes",
            "pool_used_bytes",
        ):
            self.assertEqual(stats[key], 0, key)
        self.assertFalse(hashes._pending)

    def test_waiting_for_embedding_index_does_not_own_hash_index(self):
        hashes = MMHashKeyCache(8192)
        embeddings = MMEmbeddingCache(0, 64)
        _, entry = embeddings.try_acquire("a")
        entry.complete(torch.ones(2))
        hashes.put("b", [torch.tensor([4])])
        attempting = threading.Event()
        original = embeddings._with_current_entry

        def publish(*args):
            attempting.set()
            return original(*args)

        with patch.object(
            embeddings, "_with_current_entry", side_effect=publish
        ), concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            with embeddings._lock:
                write = pool.submit(
                    hashes.put_for_entry, "a", [torch.tensor([1])], entry, embeddings
                )
                self.assertTrue(attempting.wait(5))
                self.assertEqual(self.quick(hashes.get, "b")[0].tolist(), [4])
            write.result(5)
        self.assertEqual(hashes.get("a")[0].tolist(), [1])

    def test_detach_does_not_hold_embedding_or_hash_index(self):
        hashes = MMHashKeyCache(8192)
        embeddings = MMEmbeddingCache(0, 64)
        _, old = embeddings.try_acquire("a")
        old.complete(torch.tensor([7.0, 8.0]))
        _, other = embeddings.try_acquire("b")
        other.complete(torch.ones(2))
        with self.paused(embeddings, "_detach_pool_storage") as (pool, entered):
            remove = pool.submit(embeddings.remove, "a")
            self.assertTrue(entered.wait(5))
            self.assertIsNone(self.quick(embeddings.peek, "a"))
            self.assertIs(self.quick(embeddings.peek, "b"), other)
            self.quick(
                hashes.put_for_entry, "b", [torch.tensor([9])], other, embeddings
            )
            self.assertEqual(self.quick(hashes.get, "b")[0].tolist(), [9])
            self.quick(hashes.put_for_entry, "a", [torch.tensor([7])], old, embeddings)
            self.assertFalse(hashes.contains("a"))
        remove.result(5)
        _, replacement = embeddings.try_acquire("a")
        replacement.complete(torch.zeros(2))
        self.assertEqual(old.wait(2).tolist(), [7.0, 8.0])

    def test_remote_eviction_callback_can_probe_index(self):
        seen = []
        cache = MMEmbeddingCache(
            0,
            16,
            on_cpu_evict=lambda key, entry: seen.append(
                (key, cache.peek(key), entry.result.tolist())
            ),
        )
        _, entry = cache.try_acquire("a")
        entry.complete(torch.tensor([3.0, 4.0]))
        self.quick(cache.resize, 0, 0)
        self.assertEqual(seen, [("a", None, [3.0, 4.0])])


if __name__ == "__main__":
    unittest.main()
