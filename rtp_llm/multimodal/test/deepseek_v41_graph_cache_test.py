"""CPU state-machine coverage; this is not CUDA numerical validation."""

import gc
import threading
import unittest
import weakref
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, PropertyMock, patch

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
        self.cache = PackedVisionGraph(
            256,
            max_entries=2,
            max_bytes=4096,
            observation_window=getattr(self, "observation_window", None),
        )
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
        with patch("torch.cuda.mem_get_info", return_value=(1024**3, 16 * 1024**3)):
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
        with patch("torch.cuda.mem_get_info", return_value=(1024**3, 16 * 1024**3)):
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


class RecentUseGraphCacheTest(GraphCacheTest):
    observation_window = 16

    def test_online_capture_requires_four_real_uses_and_builds_one_job(self):
        with patch.object(
            cuda_graph_gate, "capture", wraps=cuda_graph_gate.capture
        ) as acquire:
            for _ in range(3):
                self.run_shape()
            self.assertFalse(
                self.cache.build_one(self.forward, min_observations=4, priority=True)
            )
            acquire.assert_not_called()
            self.assertEqual(self.cache.stats()["pending"], 1)
            self.run_shape()
            for _ in range(4):
                self.run_shape(((3, 3),))
            self.assertTrue(
                self.cache.build_one(self.forward, min_observations=4, priority=True)
            )
            self.assertEqual(self.cache.stats()["entries"], 1)
            self.assertEqual(self.cache.stats()["pending"], 1)
            self.assertEqual(self.cache.stats()["online_attempts"], 1)
            self.assertEqual(self.cache.stats()["online_captures"], 1)
            self.assertTrue(
                self.cache.build_one(self.forward, min_observations=4, priority=True)
            )
            self.assertEqual(acquire.call_count, 2)
            self.assertEqual(self.capture_mock.call_count, 2)
            self.assertEqual(self.cache.stats()["online_captures"], 2)

    def test_online_losing_candidate_does_not_queue_a_priority_writer(self):
        self.cache.max_entries = 1
        for _ in range(8):
            self.run_shape(((8, 8),))
        self.assertTrue(self.cache.build_one(self.forward))
        for _ in range(4):
            self.run_shape()
        with patch.object(cuda_graph_gate, "capture") as acquire:
            self.assertFalse(
                self.cache.build_one(self.forward, min_observations=4, priority=True)
            )
            acquire.assert_not_called()
        self.assertEqual(self.cache.stats()["pending"], 1)
        self.assertEqual(self.cache.stats()["online_attempts"], 0)
        self.assertEqual(self.cache.stats()["online_captures"], 0)

    def test_online_large_loser_does_not_starve_admissible_smaller_shape(self):
        self.cache.max_bytes = 1024
        for _ in range(9):
            self.run_shape()
        self.assertTrue(self.cache.build_one(self.forward))
        hot = next(iter(self.cache._entries))
        self.run_shape(((2, 2),))
        self.assertTrue(self.cache.build_one(self.forward))
        for _ in range(4):
            self.run_shape(((3, 3),))
        large = next(iter(self.cache._pending))
        # This shape would have to evict both the cold and the hotter entry.
        self.cache._pending[large] = 1000
        for _ in range(4):
            self.run_shape(((2, 4),))
        self.assertTrue(
            self.cache.build_one(self.forward, min_observations=4, priority=True)
        )
        self.assertIn(hot, self.cache._entries)
        self.assertIn(large, self.cache._pending)
        self.assertEqual(next(reversed(self.cache._entries))[2], ((2, 4),))
        self.assertEqual(self.cache.stats()["online_captures"], 1)

    def test_online_wait_releases_mutex_and_rechecks_reload_generation(self):
        for _ in range(4):
            self.run_shape()
        entered = threading.Event()
        original_capture = cuda_graph_gate.capture

        @contextmanager
        def announce_capture():
            entered.set()
            with original_capture():
                yield

        with patch.object(cuda_graph_gate, "capture", new=announce_capture):
            with ThreadPoolExecutor(2) as pool:
                with cuda_graph_gate.operation():
                    capture = pool.submit(
                        self.cache.build_one,
                        self.forward,
                        min_observations=4,
                        priority=True,
                    )
                    self.assertTrue(entered.wait(5))
                    # A capture waiter must not retain the cache mutex needed
                    # by a reader/reload which already owns an operation scope.
                    self.assertEqual(
                        pool.submit(self.cache.stats).result(5)["pending"], 1
                    )
                    self.cache.clear()
                self.assertFalse(capture.result(5))
        self.capture_mock.assert_not_called()
        self.assertEqual(self.cache.stats()["online_attempts"], 0)

    def test_online_rechecks_admission_after_acquiring_gate(self):
        for _ in range(4):
            self.run_shape()
        original_capture = cuda_graph_gate.capture

        @contextmanager
        def lower_budget():
            with original_capture():
                with self.cache._lock:
                    self.cache.max_bytes = 0
                yield

        with patch.object(cuda_graph_gate, "capture", new=lower_budget), patch(
            "torch.cuda.mem_get_info"
        ) as memory:
            self.assertFalse(
                self.cache.build_one(self.forward, min_observations=4, priority=True)
            )
            memory.assert_not_called()
        self.capture_mock.assert_not_called()

    def test_online_capture_failure_is_not_an_eager_retry(self):
        for _ in range(4):
            self.run_shape()
        self.capture_mock.side_effect = RuntimeError("online capture failed")
        with self.assertRaisesRegex(RuntimeError, "online capture failed"):
            self.cache.build_one(self.forward, min_observations=4, priority=True)
        self.assertTrue(self.cache.stats()["failed"])
        self.assertEqual(self.cache.stats()["online_attempts"], 1)
        self.assertEqual(self.cache.stats()["online_captures"], 0)
        with self.assertRaisesRegex(RuntimeError, "stopped"):
            self.run_shape()

    def test_expired_pending_is_not_captured_into_an_available_slot(self):
        self.run_shape(((8, 8),))
        expired = next(iter(self.cache._pending))
        for _ in range(16):
            self.run_shape(((2, 2),))
        self.assertNotIn(expired, self.cache._observations)
        self.assertTrue(self.cache.build_one(self.forward))
        # Before the fix this second idle build selected the obsolete8x8 job
        # and raised KeyError while publishing its missing observation.
        self.assertFalse(self.cache.build_one(self.forward))
        self.assertNotIn(expired, self.cache._pending)
        self.assertEqual(self.capture_mock.call_count, 1)
        self.assertEqual(self.cache.stats()["cached_observations"], [16])
        self.assertEqual(self.cache.stats()["expired_pending"], 1)

    def test_expired_pending_cannot_capture_or_evict_a_hot_entry(self):
        self.cache.max_entries = 1
        self.run_shape(((8, 8),))
        self.assertTrue(self.cache.build_one(self.forward))
        ready = next(iter(self.cache._entries.values()))
        self.run_shape(((9, 9),))
        expired = next(iter(self.cache._pending))
        for _ in range(16):
            self.run_shape(((8, 8),))
        self.assertNotIn(expired, self.cache._pending)
        self.assertFalse(self.cache.build_one(self.forward))
        self.assertIs(next(iter(self.cache._entries.values())), ready)
        self.assertEqual(self.capture_mock.call_count, 1)
        ready.done.synchronize.assert_not_called()

    def test_expiring_same_key_is_reobserved_and_queued_without_fake_counts(self):
        cache = PackedVisionGraph(
            256, max_entries=2, max_bytes=4096, observation_window=1
        )
        parts = [torch.ones(6, 2)]
        with patch.object(cache, "_eligible", return_value=True), patch.object(
            cache, "_capture", side_effect=self.fake_capture
        ):
            for _ in range(2):
                cache.run(parts, [(2, 3)], self.forward, self.consume)
            self.assertEqual(list(cache._observations.values()), [1])
            self.assertEqual(len(cache._pending), 1)
            self.assertEqual(cache.stats()["expired_pending"], 1)
            self.assertTrue(cache.build_one(self.forward))
            self.assertEqual(cache.stats()["cached_observations"], [1])

    def test_new_phase_replaces_long_lived_hot_larger_shape(self):
        self.cache.max_entries = 1
        self.run_shape(((8, 8),))
        self.assertTrue(self.cache.build_one(self.forward))
        for _ in range(100):
            self.run_shape(((8, 8),))
        old = next(iter(self.cache._entries.values()))
        self.assertEqual(self.cache.stats()["cached_observations"], [16])
        self.run_shape(((2, 2),))
        self.assertFalse(self.cache.build_one(self.forward))
        for _ in range(8):
            self.run_shape(((2, 2),))
        self.assertTrue(self.cache.build_one(self.forward))
        self.assertEqual(next(iter(self.cache._entries))[2], ((2, 2),))
        self.assertEqual(self.cache.stats()["evictions"], 1)
        self.assertEqual(self.cache.stats()["cached_observations"], [9])
        old.done.synchronize.assert_called_once()

    def test_clear_discards_recent_phase_history(self):
        for _ in range(40):
            self.run_shape(((8, 8),))
        self.cache.clear()
        self.run_shape(((2, 2),))
        self.assertTrue(self.cache.build_one(self.forward))
        self.assertEqual(self.cache.stats()["observation_window"], 16)
        self.assertEqual(self.cache.stats()["cached_observations"], [1])


class OnlineVisionPolicyTest(unittest.TestCase):
    def test_online_capability_requires_ppu_eval_cuda_and_graph_budget(self):
        from rtp_llm.multimodal.multimodal_mixins.deepseek_v41.deepseek_v41_vision import (
            DeepSeekV41VisionEmbedding,
        )

        module = DeepSeekV41VisionEmbedding.__module__
        cases = (
            (True, False, "cuda", 4, 4096, True),
            (False, False, "cuda", 4, 4096, False),
            (True, True, "cuda", 4, 4096, False),
            (True, False, "cpu", 4, 4096, False),
            (True, False, "cuda", 0, 4096, False),
            (True, False, "cuda", 4, 0, False),
        )
        for ppu, training, device, entries, budget, expected in cases:
            with self.subTest(
                ppu=ppu,
                training=training,
                device=device,
                entries=entries,
                budget=budget,
            ), patch(module + ".is_ppu", return_value=ppu):
                owner = SimpleNamespace(
                    training=training,
                    _device=torch.device(device),
                    _packed_graph=SimpleNamespace(
                        max_entries=entries, max_bytes=budget
                    ),
                )
                self.assertIs(
                    DeepSeekV41VisionEmbedding.supports_embedding_online_work.fget(
                        owner
                    ),
                    expected,
                )

    def test_online_hook_requests_one_four_use_priority_job(self):
        from rtp_llm.multimodal.multimodal_mixins.deepseek_v41.deepseek_v41_vision import (
            DeepSeekV41VisionEmbedding,
        )

        owner = SimpleNamespace(
            supports_embedding_online_work=True,
            _packed_graph=Mock(),
            _forward_packed=Mock(),
            _graph_context=lambda: ("fa2",),
            _prepare_graph_forward=Mock(),
        )
        DeepSeekV41VisionEmbedding.run_embedding_online_work(owner)
        owner._packed_graph.build_one.assert_called_once_with(
            owner._forward_packed,
            context=("fa2",),
            prepare_forward=owner._prepare_graph_forward,
            min_observations=4,
            priority=True,
        )


class StablePackedOrderTest(unittest.TestCase):
    """CPU ordering/budget contracts; the forward fixture is not PPU math."""

    def test_grid_permutation_is_stable_and_invertible(self):
        from rtp_llm.multimodal.multimodal_mixins.deepseek_v41.deepseek_v41_vision import (
            _stable_grid_permutation,
        )

        grids = [(3, 2), (1, 2), (2, 2), (1, 2), (1, 2)]
        order, inverse = _stable_grid_permutation(grids)
        self.assertEqual(order, [1, 3, 4, 2, 0])
        self.assertEqual([order[index] for index in inverse], list(range(len(grids))))
        self.assertEqual(grids, [(3, 2), (1, 2), (2, 2), (1, 2), (1, 2)])
        for identity in ([], [(2, 3)], [(2, 3)] * 4):
            expected = list(range(len(identity)))
            self.assertEqual(_stable_grid_permutation(identity), (expected, expected))

    def _run_class_cpu_contract(self, enable_ordering):
        from rtp_llm.multimodal.multimodal_mixins.deepseek_v41.deepseek_v41_vision import (
            DeepSeekV41VisionEmbedding,
            _RuntimeImage,
        )
        from rtp_llm.multimodal.multimodal_mixins.multimodal_common import (
            MMWorkEstimate,
        )

        owner = object.__new__(DeepSeekV41VisionEmbedding)
        torch.nn.Module.__init__(owner)
        owner._packed_max_images = 3
        owner._resource_lock = threading.Lock()
        owner._resource_budget = MMWorkEstimate(input_patches=10)
        owner._workspace_per_patch = 1
        owner._packed_stats_lock = threading.Lock()
        owner._packed_batch_histogram = {}
        owner.processor_config = SimpleNamespace(vision_downsample_ratio=1)
        for name, value in (
            ("image_start", -1),
            ("image_newline", -2),
            ("image_end", -3),
        ):
            setattr(
                owner,
                name,
                torch.nn.Parameter(torch.tensor([value], dtype=torch.bfloat16)),
            )
        owner._packed_graph = PackedVisionGraph(3, max_entries=2, max_bytes=4096)
        owner._graph_context = lambda: ()
        forwards = []

        def forward(patches, grids):
            forwards.append(([int(part[0, 0]) for part in patches], list(grids)))
            return [part.clone() for part in patches]

        owner._forward_packed = forward
        grids = [(3, 2), (1, 2), (2, 2), (1, 2), (1, 2)]
        images = [
            _RuntimeImage(torch.full((h * w, 1), index, dtype=torch.bfloat16), h, w)
            for index, (h, w) in enumerate(grids, 1)
        ]
        owner.eval()
        # Only the enabled policy is forced for this CPU contract. The real
        # chunk selection, graph.run eager miss, assembly and inverse all run;
        # the disabled case retains the class's actual default CPU policy.
        policy = (
            patch.object(
                DeepSeekV41VisionEmbedding,
                "supports_embedding_online_work",
                new_callable=PropertyMock,
                return_value=True,
            )
            if enable_ordering
            else nullcontext()
        )
        with policy:
            outputs = owner.encode_images_packed(images)
        self.assertEqual(
            [int(image.patches[0, 0]) for image in images], [1, 2, 3, 4, 5]
        )
        self.assertEqual(len(outputs), 5)
        for index, (output, (height, width)) in enumerate(zip(outputs, grids), 1):
            self.assertEqual(tuple(output.shape), (height * (width + 1) + 2, 1))
            self.assertEqual(int(output[1, 0]), index)
        self.assertEqual(owner._packed_batch_histogram, {2: 1, 3: 1})
        self.assertEqual(owner._packed_graph.stats()["eager"], 2)
        return forwards

    def test_real_class_sorts_real_patches_only_within_existing_bounded_chunks(self):
        self.assertEqual(
            self._run_class_cpu_contract(True),
            [([2, 1], [(1, 2), (3, 2)]), ([4, 5, 3], [(1, 2), (1, 2), (2, 2)])],
        )

    def test_real_class_keeps_original_order_when_policy_is_disabled(self):
        self.assertEqual(
            self._run_class_cpu_contract(False),
            [([1, 2], [(3, 2), (1, 2)]), ([3, 4, 5], [(2, 2), (1, 2), (1, 2)])],
        )


if __name__ == "__main__":
    unittest.main()
