"""CPU state-machine coverage; this is not CUDA numerical validation."""

import gc
import unittest
import weakref
from unittest.mock import Mock, patch

import torch

from rtp_llm.multimodal.multimodal_mixins.deepseek_v41.packed_graph import (
    PackedVisionGraph,
    _Entry,
)
from rtp_llm.utils.cuda_graph_gate import cuda_graph_gate


class GraphCacheTest(unittest.TestCase):
    def setUp(self):
        self.inference = torch.inference_mode()
        self.inference.__enter__()
        self.addCleanup(self.inference.__exit__, None, None, None)
        self.cache = PackedVisionGraph(256, max_entries=2, max_bytes=4096)
        # A Mock would itself retain request tensors in call_args, defeating
        # the weak-reference test of the cache's ownership below.
        self.eligibility = patch.object(self.cache, "_eligible", new=lambda *_: True)
        self.eligibility.start()
        self.addCleanup(self.eligibility.stop)
        self.memory = patch(
            "torch.cuda.mem_get_info", return_value=(16 * 1024**3, 16 * 1024**3)
        )
        self.memory.start()
        self.addCleanup(self.memory.stop)
        self.snapshot = patch("torch.cuda.memory_snapshot", return_value=[])
        self.snapshot.start()
        self.addCleanup(self.snapshot.stop)
        self.stream = patch("torch.cuda.current_stream", return_value=Mock())
        self.stream.start()
        self.addCleanup(self.stream.stop)
        self.capture = patch.object(
            self.cache, "_capture", side_effect=self.fake_capture
        )
        self.capture_mock = self.capture.start()
        self.addCleanup(self.capture.stop)

    @staticmethod
    def forward(parts, grids):
        return [x + h + w for x, (h, w) in zip(parts, grids)]

    @staticmethod
    def consume(outputs):
        return [x.clone() for x in outputs]

    def fake_capture(self, key, estimate, forward, prepare):
        inputs = [torch.zeros(shape, dtype=dtype) for shape, _, dtype, _ in key[3]]
        outputs = self.forward(inputs, key[2])

        def replay():
            for dst, src in zip(outputs, self.forward(inputs, key[2])):
                dst.copy_(src)

        return _Entry(Mock(replay=replay), inputs, outputs, Mock(), estimate)

    def run_shape(self, grids=((2, 3),), value=1, context=()):
        parts = [torch.full((h * w, 2), float(value)) for h, w in grids]
        return self.cache.run(
            parts,
            grids,
            self.forward,
            self.consume,
            context=context,
            estimated_bytes=128,
        )

    def test_first_miss_eager_deduplicates_without_retaining_request(self):
        source = torch.ones(6, 2)
        weak = weakref.ref(source)
        result = self.cache.run([source], [(2, 3)], self.forward, self.consume)
        del source
        gc.collect()
        self.assertIsNone(weak())
        self.assertTrue(torch.equal(result[0], torch.full((6, 2), 6.0)))
        self.run_shape()
        self.assertEqual(self.cache.stats()["pending"], 1)
        self.capture_mock.assert_not_called()
        self.assertEqual(self.cache.stats()["captures"], 0)

    def test_two_mixed_shapes_replay_changed_input_and_own_outputs(self):
        for shape in (((2, 3),), ((2, 3), (3, 3))):
            self.run_shape(shape)
            self.assertTrue(self.cache.build_one(self.forward))
        previous = self.run_shape(value=3)[0]
        saved = previous.clone()
        changed = self.run_shape(value=7)[0]
        self.assertTrue(torch.equal(previous, saved))
        self.assertFalse(torch.equal(previous, changed))
        self.assertEqual(self.cache.stats()["entries"], 2)
        self.assertEqual(self.cache.stats()["replays"], 2)

    def test_lru_evicts_only_completed_entry(self):
        self.run_shape()
        self.cache.build_one(self.forward)
        first = next(iter(self.cache._entries.values()))
        self.run_shape(((3, 3),))
        self.cache.build_one(self.forward)
        self.run_shape()
        self.run_shape(((4, 4),))
        self.cache.build_one(self.forward)
        self.assertTrue(any(entry is first for entry in self.cache._entries.values()))
        self.assertEqual(self.cache.stats()["evictions"], 1)

    def test_one_off_tail_cannot_evict_hot_full_batch(self):
        self.cache.max_entries = 1
        self.run_shape(((8, 8),))
        self.assertTrue(self.cache.build_one(self.forward))
        full_key = next(iter(self.cache._entries))
        self.run_shape(((8, 8),))
        self.run_shape(((8, 8),))
        self.run_shape(((7, 8),))
        self.assertFalse(self.cache.build_one(self.forward))
        self.assertEqual(list(self.cache._entries), [full_key])
        replays = self.cache.stats()["replays"]
        self.run_shape(((8, 8),))
        self.assertEqual(self.cache.stats()["replays"], replays + 1)
        self.assertEqual(self.cache.stats()["evictions"], 0)

    def test_repeated_candidate_eventually_replaces_cold_entry(self):
        self.cache.max_entries = 1
        self.run_shape(((8, 8),))
        self.cache.build_one(self.forward)
        for _ in range(2):
            self.run_shape(((7, 8),))
        self.assertTrue(self.cache.build_one(self.forward))
        self.assertEqual(next(iter(self.cache._entries))[2], ((7, 8),))
        self.assertEqual(self.cache.stats()["evictions"], 1)

    def test_actual_charge_does_not_evict_hot_entry(self):
        self.cache.max_entries = 2
        self.cache.max_bytes = 1024
        self.run_shape(((8, 8),))
        self.cache.build_one(self.forward)
        hot_key = next(iter(self.cache._entries))
        self.run_shape(((8, 8),))
        self.run_shape(((8, 8),))

        def larger_capture(key, estimate, forward, prepare):
            entry = self.fake_capture(key, estimate, forward, prepare)
            entry.charged_bytes = 800
            return entry

        self.capture_mock.side_effect = larger_capture
        self.run_shape(((7, 8),))
        self.assertFalse(self.cache.build_one(self.forward))
        self.assertEqual(list(self.cache._entries), [hot_key])
        self.assertEqual(self.cache.stats()["evictions"], 0)

    def test_free_check_failure_does_not_evict_ready(self):
        self.cache.max_entries = 1
        self.cache.max_bytes = 32 * 1024**3
        self.run_shape(((8, 8),))
        self.assertTrue(self.cache.build_one(self.forward))
        ready_key = next(iter(self.cache._entries))
        self.capture_mock.reset_mock()
        self.run_shape(((7, 8),))
        self.run_shape(((7, 8),))
        pending_key = next(iter(self.cache._pending))
        self.cache._pending[pending_key] = 15 * 1024**3
        with patch(
            "torch.cuda.mem_get_info", return_value=(1024**3, 16 * 1024**3)
        ):
            self.assertFalse(self.cache.build_one(self.forward))
        self.assertEqual(list(self.cache._entries), [ready_key])
        self.assertEqual(self.cache.stats()["evictions"], 0)
        self.assertEqual(self.cache.stats()["pending"], 0)
        self.capture_mock.assert_not_called()

    def test_unallocatable_hot_pending_does_not_starve_smaller_pending(self):
        self.cache.max_bytes = 32 * 1024**3
        self.run_shape(((8, 8),))
        self.run_shape(((8, 8),))
        self.run_shape(((2, 3),))
        large_key = next(key for key in self.cache._pending if key[2] == ((8, 8),))
        small_key = next(key for key in self.cache._pending if key[2] == ((2, 3),))
        self.cache._pending[large_key] = 15 * 1024**3
        with patch(
            "torch.cuda.mem_get_info", return_value=(1024**3, 16 * 1024**3)
        ):
            self.assertFalse(self.cache.build_one(self.forward))
            self.assertTrue(self.cache.build_one(self.forward))
        self.assertNotIn(large_key, self.cache._pending)
        self.assertIn(small_key, self.cache._entries)
        self.assertEqual(self.capture_mock.call_count, 1)

    def test_pending_budget_and_context_are_bounded(self):
        for size in range(2, 20):
            self.run_shape(((size, size),))
        self.assertEqual(self.cache.stats()["pending"], 2)
        self.assertGreater(self.cache.stats()["deferred"], 0)
        self.assertFalse(self.cache.build_one(self.forward, context=("other_backend",)))
        self.assertEqual(self.cache.stats()["pending"], 0)
        parts = [torch.zeros(6, 2)]
        self.cache.run(
            parts, [(2, 3)], self.forward, self.consume, estimated_bytes=8192
        )
        self.assertEqual(self.cache.stats()["budget_skips"], 1)
        self.assertEqual(self.cache.stats()["pending"], 0)

    def test_busy_gate_defers_build_and_does_not_request_writer_priority(self):
        import concurrent.futures

        self.run_shape()
        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            with cuda_graph_gate.operation():
                self.assertFalse(
                    pool.submit(self.cache.build_one, self.forward).result(5)
                )
        self.assertEqual(self.cache.stats()["pending"], 1)
        self.capture_mock.assert_not_called()

    def test_failure_stops_cache_no_hidden_eager_retry(self):
        self.run_shape()
        self.capture_mock.side_effect = RuntimeError("capture failed")
        with self.assertRaisesRegex(RuntimeError, "capture failed"):
            self.cache.build_one(self.forward)
        with self.assertRaisesRegex(RuntimeError, "stopped"):
            self.run_shape()
        self.assertTrue(self.cache.stats()["failed"])

    def test_clear_discards_pending_and_ready_with_fence(self):
        self.run_shape()
        self.cache.build_one(self.forward)
        entry = next(iter(self.cache._entries.values()))
        self.run_shape(((3, 3),))
        self.cache.clear()
        entry.done.synchronize.assert_called_once()
        self.assertEqual(self.cache.stats()["entries"], 0)
        self.assertEqual(self.cache.stats()["pending"], 0)
        self.assertEqual(self.cache.stats()["observed_shapes"], 0)

    def test_observation_metadata_is_bounded(self):
        for size in range(2, 80):
            self.run_shape(((size, size),))
        self.assertLessEqual(
            self.cache.stats()["observed_shapes"], self.cache._observation_limit
        )

    def test_same_patch_count_different_geometry_has_distinct_key(self):
        self.run_shape(((2, 3),))
        self.run_shape(((3, 2),))
        self.assertEqual(self.cache.stats()["pending"], 2)

    def test_oversized_actual_capture_is_not_published(self):
        self.run_shape()
        self.capture_mock.side_effect = lambda *args: _Entry(
            Mock(), [], [], Mock(), 8192
        )
        self.assertFalse(self.cache.build_one(self.forward))
        self.assertEqual(self.cache.stats()["entries"], 0)
        self.assertEqual(self.cache.stats()["budget_skips"], 1)


if __name__ == "__main__":
    unittest.main()
