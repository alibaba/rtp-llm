import unittest
from types import SimpleNamespace
from unittest.mock import patch

from navi_sched.adapters.sglang.SGLangCache import SGLangCacheState
from sglang.srt.disaggregation.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    StorageMedium,
)


class _Bridge:
    def __init__(self):
        self.inserted = []
        self.removed = []
        self.updates = []

    def report_cache_inserted(self, hashes):
        self.inserted.extend(hashes)

    def report_cache_removed(self, hashes):
        self.removed.extend(hashes)

    def report_cache_updated(self, *values):
        self.updates.append(values)


def stored(hashes, size=4, medium=StorageMedium.GPU):
    return BlockStored(
        block_hashes=hashes,
        parent_block_hash=None,
        token_ids=list(range(size)),
        block_size=size,
        medium=medium,
        lora_id=None,
    )


class SGLangCacheTest(unittest.TestCase):
    def test_hybrid_placement_does_not_imply_tail_readiness(self):
        bridge = _Bridge()
        state = SGLangCacheState(bridge, 2, 4, hybrid_tail_required=True)
        state.on_event(stored([10, 20]))
        self.assertEqual(bridge.updates, [([10, 20], [], [10, 20], [], [], [])])

        class Key:
            extra_key = None
            cache_salt = None

            def __getitem__(self, item):
                return [1, 2, 3, 4]

        result = SimpleNamespace(
            host_hit_length=0,
            swa_host_hit_length=0,
            mamba_host_hit_length=1,
            device_prefix_len=4,
        )
        state.on_match(Key(), result)
        self.assertEqual(len(bridge.updates), 1)
        result.mamba_host_hit_length = 0
        with patch(
            "navi_sched.adapters.sglang.SGLangCache.prompt_block_hashes",
            return_value=[10],
        ):
            state.on_match(Key(), result)
            state.on_match(Key(), result)
        self.assertEqual(bridge.updates[-1], ([], [], [], [], [10], []))
        self.assertEqual(len(bridge.updates), 2)
        state.invalidate_tails()
        self.assertEqual(bridge.updates[-1], ([], [], [], [], [], [10]))
        self.assertEqual(state._hashes, {10, 20})

    def test_request_hashes_equal_real_namespaced_sglang_events(self):
        from navi_sched.adapters.sglang.SGLangCache import prompt_block_hashes
        from sglang.srt.mem_cache.events import KVCacheEventRecorder
        from sglang.srt.mem_cache.radix_cache import RadixKey

        for extra_key, salt in (
            (None, None),
            (None, ""),
            ("", None),
            ("tenant", None),
            (None, "salt"),
            ("tenant", "salt"),
        ):
            recorder = KVCacheEventRecorder(
                enabled=True, page_size=4, dynamo_format=True
            )
            root = SimpleNamespace(
                parent=None, key=RadixKey([]), hash_value=[], event_hash_value=None
            )
            first = SimpleNamespace(
                parent=root,
                key=RadixKey([1, 2, 3, 4], extra_key, cache_salt=salt),
                hash_value=None,
                event_hash_value=None,
            )
            second = SimpleNamespace(
                parent=first,
                key=RadixKey([5, 6, 7, 8], extra_key, cache_salt=salt),
                hash_value=None,
                event_hash_value=None,
            )
            recorder.record_store(first)
            recorder.record_store(second)
            event_hashes = [
                int(value) & ((1 << 64) - 1)
                for event in recorder.take()
                for value in event.block_hashes
            ]
            self.assertEqual(
                prompt_block_hashes(
                    list(range(1, 9)), 4, extra_key=extra_key, cache_salt=salt
                ),
                event_hashes,
            )

    def test_loading_pages_are_unavailable_until_copy_completion(self):
        bridge = _Bridge()
        state = SGLangCacheState(bridge, 2, 4)
        state.on_event(stored([1, 2]), device_ready=False)
        self.assertEqual(bridge.inserted, [])
        self.assertEqual(state._hashes, set())
        self.assertEqual(state._loading_hashes, {1, 2})
        state.on_event(BlockRemoved(block_hashes=[1], medium=StorageMedium.GPU))
        state.finish_loads()
        self.assertEqual(bridge.inserted, [2])
        self.assertEqual(state._hashes, {2})
        self.assertEqual(state._loading_hashes, set())
        state.on_event(stored([3]), device_ready=False)
        state.on_event(AllBlocksCleared())
        state.finish_loads()
        self.assertEqual(bridge.inserted, [2])
        self.assertEqual(bridge.removed, [2])

    def test_hybrid_native_key_and_match_result_use_the_same_page_identity(self):
        from navi_sched.adapters.sglang.SGLangCache import prompt_block_hashes
        from sglang.srt.mem_cache.base_prefix_cache import MatchResult
        from sglang.srt.mem_cache.radix_cache import RadixKey

        key = RadixKey([1, 2, 3, 4, 5, 6, 7, 8], "tenant", cache_salt="salt")
        hashes = prompt_block_hashes(
            key, 4, extra_key=key.extra_key, cache_salt=key.cache_salt
        )
        bridge = _Bridge()
        state = SGLangCacheState(bridge, 2, 4, hybrid_tail_required=True)
        state.on_event(stored(hashes))
        result = MatchResult(8, None, None, None)
        state.on_match(key, result._replace(swa_host_hit_length=4))
        self.assertEqual(len(bridge.updates), 1)
        state.on_match(key, result)
        self.assertEqual(bridge.updates[-1], ([], [], [], [], [hashes[1]], []))
        self.assertEqual(state._tail_hashes, {hashes[1]})

    def test_partial_pages_and_cpu_pages_are_not_gpu_hits(self):
        bridge = _Bridge()
        state = SGLangCacheState(bridge, 2, 4)
        state.on_event(stored([11], size=3))
        state.on_event(stored([12], medium=StorageMedium.CPU))
        self.assertEqual(bridge.inserted, [])
        state.on_event(stored([13, -1]))
        state.on_event(stored([13]))
        self.assertEqual(bridge.inserted, [13, (1 << 64) - 1])
        state.on_event(BlockRemoved(block_hashes=[13], medium=StorageMedium.CPU))
        self.assertEqual(bridge.removed, [])
        state.on_event(BlockRemoved(block_hashes=[13, 99], medium=StorageMedium.GPU))
        self.assertEqual(bridge.removed, [13])
        state.on_event(AllBlocksCleared())
        self.assertEqual(bridge.removed, [13, (1 << 64) - 1])

    def test_capacity_and_page_size_are_checked(self):
        state = SGLangCacheState(_Bridge(), 1, 4)
        state.on_event(stored([1]))
        with self.assertRaisesRegex(ValueError, "capacity"):
            state.on_event(stored([2]))
        with self.assertRaisesRegex(ValueError, "page size"):
            state.on_event(stored([2], size=8))


if __name__ == "__main__":
    unittest.main()
