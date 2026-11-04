import unittest
from types import SimpleNamespace
from unittest import mock

from navi_sched.adapters.vllm import VLLMImageCacheMap as mapping_module
from navi_sched.adapters.vllm.VLLMImageCacheMap import VLLMImageCacheMap


def block(value):
    return value.to_bytes(32, "big")


def image(offset, length):
    return SimpleNamespace(modality="image", mm_position=SimpleNamespace(offset=offset, length=length))


class VLLMImageCacheMapTest(unittest.TestCase):
    def setUp(self):
        self.cache = SimpleNamespace(_full_ready=set(), _tail_ready=set())
        self.cache._flat_ready = self.cache._full_ready
        self.mapping = VLLMImageCacheMap(self.cache, False)
        self.prediction = SimpleNamespace(prompt_token_ids=list(range(12)), mm_features=[image(5, 1)])
        self.actual = SimpleNamespace(
            prompt_token_ids=list(range(19)), mm_features=[image(5, 8)],
            block_hashes=[block(i) for i in range(10, 14)],
        )
        self.hashes = mock.patch.object(
            mapping_module, "calculate_vllm_block_hashes", return_value=[1, 2, 3],
        )
        self.hashes.start()
        self.addCleanup(self.hashes.stop)

    def store(self, indexes, tail=False):
        hashes = [self.actual.block_hashes[i] for i in indexes]
        (self.cache._tail_ready if tail else self.cache._full_ready).update(hashes)
        return self.mapping.update(hashes)

    def test_entire_image_required_before_placeholder_and_suffix_are_confirmed(self):
        self.assertTrue(self.mapping.register(71, self.prediction, self.actual, 4).empty())
        # First logical block is text-only; block 2 contains the one-token image.
        record = next(iter(self.mapping._records.values()))
        self.assertEqual((4,), record.required_blocks)
        self.assertEqual((block(2),), record.logical_hashes)
        self.assertTrue(self.store([0, 1, 3]).empty())
        confirmed = self.store([2])
        self.assertEqual((block(2),), confirmed.added)
        self.assertEqual((), confirmed.added_full_ready)
        self.mapping.finish(71)
        self.assertEqual({}, self.mapping._owners)
        # Completion retains confirmed cache, but eviction revokes it.
        self.cache._full_ready.remove(block(11))
        removed = self.mapping.update([block(11)])
        self.assertEqual((block(2),), removed.removed)
        self.assertEqual({}, self.mapping._records)
        self.assertEqual({}, self.mapping._by_hash)
        self.assertEqual(0, self.mapping._references)

    def test_suffix_blocks_wait_for_the_image_even_when_suffix_arrives_first(self):
        self.prediction.prompt_token_ids = list(range(16))
        self.actual.prompt_token_ids = list(range(23))
        self.actual.block_hashes.append(block(14))
        self.hashes.stop()
        with mock.patch.object(mapping_module, "calculate_vllm_block_hashes", return_value=[1, 2, 3, 4]):
            self.mapping.register(71, self.prediction, self.actual, 4)
        self.assertTrue(self.store([0, 2, 3, 4]).empty())
        delta = self.store([1])
        self.assertEqual({block(2), block(3)}, set(delta.added))
        self.cache._full_ready.remove(block(11))
        delta = self.mapping.update([block(11)])
        self.assertEqual({block(2), block(3)}, set(delta.removed))

    def test_completion_and_abort_without_confirmations_leave_no_mapping(self):
        for owner in range(50):
            self.mapping.register(owner, self.prediction, self.actual, 4)
            self.mapping.finish(owner)
            self.assertEqual({}, self.mapping._records)
            self.assertEqual({}, self.mapping._owners)
            self.assertEqual({}, self.mapping._by_hash)
            self.assertEqual(0, self.mapping._references)

    def test_two_images_gate_their_own_suffix_without_per_token_mapping(self):
        self.prediction.prompt_token_ids = list(range(20))
        self.prediction.mm_features = [image(5, 1), image(10, 1)]
        self.actual.prompt_token_ids = list(range(30))
        self.actual.mm_features = [image(5, 8), image(17, 4)]
        self.actual.block_hashes = [block(i) for i in range(10, 17)]
        with mock.patch.object(mapping_module, "calculate_vllm_block_hashes",
                               return_value=[1, 2, 3, 4, 5]):
            self.mapping.register(71, self.prediction, self.actual, 4)
        self.assertEqual((4, 6, 7), next(iter(self.mapping._records.values())).required_blocks)
        self.assertEqual((block(2),), self.store(range(4)).added)
        self.assertTrue(self.store([6]).empty())
        self.assertTrue(self.store([4]).empty())
        self.assertEqual({block(3), block(4)}, set(self.store([5]).added))

    def test_oversized_mapping_is_skipped_without_leaving_owner_or_indices(self):
        self.mapping = VLLMImageCacheMap(self.cache, False, max_block_references=4)
        self.assertTrue(self.mapping.register(71, self.prediction, self.actual, 4).empty())
        self.assertEqual({}, self.mapping._records)
        self.assertEqual({}, self.mapping._owners)
        self.assertEqual({}, self.mapping._by_hash)
        self.assertEqual(0, self.mapping._references)

    def test_hybrid_tail_must_be_ready_at_the_corresponding_native_boundary(self):
        self.mapping = VLLMImageCacheMap(self.cache, True)
        self.mapping.register(71, self.prediction, self.actual, 4)
        delta = self.store(range(4))
        self.assertEqual((), delta.added)
        self.assertEqual((block(2),), delta.added_full_ready)
        self.assertEqual((block(2),), self.store([3], tail=True).added_tail_ready)
        self.cache._tail_ready.remove(block(13))
        delta = self.mapping.update([block(13)])
        self.assertEqual((), delta.removed)
        self.assertEqual((block(2),), delta.removed_tail_ready)
        self.assertEqual((), delta.removed_full_ready)

    def test_duplicate_requests_share_mapping_and_finish_independently(self):
        self.mapping.register(71, self.prediction, self.actual, 4)
        self.mapping.register(72, self.prediction, self.actual, 4)
        self.assertEqual(1, len(self.mapping._records))
        self.mapping.finish(71)
        self.assertEqual({72}, next(iter(self.mapping._records.values())).owners)
        self.assertEqual((block(2),), self.store(range(4)).added)
        self.mapping.finish(72)
        self.assertEqual(1, len(self.mapping._records))
        delta = self.mapping.update((), cleared=True)
        self.assertEqual((block(2),), delta.removed)
        self.assertEqual({}, self.mapping._records)
        self.assertEqual({}, self.mapping._owners)
        self.assertEqual({}, self.mapping._definitions)

    def test_capacity_eviction_revokes_alias_and_drops_reverse_indices(self):
        self.mapping = VLLMImageCacheMap(self.cache, False, max_records=1)
        self.cache._full_ready.update(self.actual.block_hashes)
        self.mapping.register(71, self.prediction, self.actual, 4)
        replacement = SimpleNamespace(
            prompt_token_ids=self.actual.prompt_token_ids, mm_features=self.actual.mm_features,
            block_hashes=[block(i) for i in range(20, 24)],
        )
        delta = self.mapping.register(72, self.prediction, replacement, 4)
        self.assertEqual((block(2),), delta.removed)
        self.assertNotIn(71, self.mapping._owners)
        self.assertEqual(1, len(self.mapping._records))
        self.assertEqual(set(replacement.block_hashes), set(self.mapping._by_hash))
        self.mapping.finish(72)
        self.assertEqual(0, self.mapping._references)

    def test_multiple_native_variants_keep_alias_until_last_cached_variant_is_evicted(self):
        self.cache._full_ready.update(self.actual.block_hashes)
        self.assertEqual((block(2),), self.mapping.register(
            71, self.prediction, self.actual, 4,
        ).added)
        other = SimpleNamespace(
            prompt_token_ids=self.actual.prompt_token_ids, mm_features=self.actual.mm_features,
            block_hashes=[block(i) for i in range(20, 24)],
        )
        self.cache._full_ready.update(other.block_hashes)
        self.assertTrue(self.mapping.register(72, self.prediction, other, 4).empty())
        self.cache._full_ready.remove(block(11))
        self.assertTrue(self.mapping.update([block(11)]).empty())
        self.cache._full_ready.remove(block(21))
        self.assertEqual((block(2),), self.mapping.update([block(21)]).removed)
        self.mapping.finish(71)
        self.mapping.finish(72)
        self.assertEqual({}, self.mapping._records)
        self.assertEqual({}, self.mapping._by_hash)
        self.assertEqual(0, self.mapping._references)

    def test_cache_clear_drops_active_owners_and_all_metadata(self):
        self.mapping.register(71, self.prediction, self.actual, 4)
        self.mapping.update((), cleared=True)
        self.mapping.finish(71)
        self.assertEqual({}, self.mapping._records)
        self.assertEqual({}, self.mapping._owners)
        self.assertEqual({}, self.mapping._definitions)
        self.assertEqual({}, self.mapping._by_hash)
        self.assertEqual(0, self.mapping._references)

    def test_duplicate_or_tail_event_does_not_rescan_cached_prefix(self):
        class CountingSet(set):
            probes = 0

            def __contains__(self, value):
                self.probes += 1
                return super().__contains__(value)

        self.prediction.prompt_token_ids = list(range(10000))
        self.actual.prompt_token_ids = list(range(10007))
        self.actual.block_hashes = [block(i) for i in range(100, 2601)]
        cached = CountingSet(self.actual.block_hashes)
        self.cache._flat_ready = cached
        self.mapping = VLLMImageCacheMap(self.cache, False)
        with mock.patch.object(mapping_module, "calculate_vllm_block_hashes",
                               return_value=list(range(2500))):
            self.mapping.register(71, self.prediction, self.actual, 4)
        cached.probes = 0
        self.assertTrue(self.mapping.update([self.actual.block_hashes[0]]).empty())
        self.assertLessEqual(cached.probes, 2)


if __name__ == "__main__":
    unittest.main()
