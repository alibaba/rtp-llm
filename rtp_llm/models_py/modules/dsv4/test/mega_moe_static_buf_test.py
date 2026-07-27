"""Stream B — Mega MoE static output buffer (CUDA graph safety).

Tests that:
  1. A pre-allocated tensor slice shares storage with the original tensor
     (no reallocation) — this is the key property for CUDA graph capture.
  2. The slice `self._mega_y[:T]` is correctly sized for the live batch.
  3. The slice shares data_ptr with the full buffer (graph-safe).

These tests validate the buffer-slice design without requiring deep_gemm or
torch.distributed (needed for the full MoE init path). Full integration is
covered by the SM100_ARM smoke suite.
"""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, "..", "..", "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


class TestStaticBufferSliceProperty(unittest.TestCase):
    """Pre-allocated buffer slice is a view (no reallocation) — key for CUDA graphs."""

    def test_slice_shares_data_ptr(self):
        """y = buf[:T] must have the same data_ptr as buf (no new allocation)."""
        D = 128
        max_T = 64
        T = 16
        buf = torch.empty((max_T, D), dtype=torch.bfloat16)
        y = buf[:T]
        # Same underlying storage.
        self.assertEqual(y.data_ptr(), buf.data_ptr())
        # Not a copy.
        buf[0, 0] = 42.0
        self.assertEqual(y[0, 0].item(), 42.0)

    def test_slice_shape(self):
        D = 256
        max_T = 100
        buf = torch.empty((max_T, D), dtype=torch.bfloat16)
        for T in (1, 32, 99, 100):
            y = buf[:T]
            self.assertEqual(y.shape, (T, D))

    def test_write_through_slice_visible_in_buf(self):
        """Kernel writes to slice are visible in the pre-allocated buffer
        (same memory region); simulates deep_gemm writing to y = buf[:T]."""
        D = 64
        max_T = 32
        T = 8
        buf = torch.zeros((max_T, D), dtype=torch.bfloat16)
        y = buf[:T]
        y.fill_(1.0)  # simulate kernel writing to y
        # Rows 0..T-1 in buf should now be 1.0.
        self.assertTrue(torch.all(buf[:T] == 1.0))
        # Rows T..max_T should remain 0.
        self.assertTrue(torch.all(buf[T:] == 0.0))


class TestMegaMoeBufCodeChange(unittest.TestCase):
    """Verify the Stream B code change semantics in isolation."""

    def test_pre_alloc_avoids_realloc(self):
        """Simulate the forward: buf already allocated, y = buf[:T] is stable."""
        D = 512
        max_T = 384  # typical max_tokens_per_rank on SM100
        T = 37
        # This represents self._mega_y (pre-allocated in _setup_mega_moe).
        mega_y = torch.empty((max_T, D), dtype=torch.bfloat16)
        ptr_before = mega_y.data_ptr()

        # This is what the new code does on every forward step.
        y = mega_y[:T]

        # No new allocation: y.data_ptr() == mega_y.data_ptr()
        self.assertEqual(y.data_ptr(), ptr_before)
        # Shape is correct for the batch.
        self.assertEqual(y.shape, (T, D))

    def test_float_cast_on_slice_does_not_change_buf_dtype(self):
        """y.float() returns a new tensor; does not alter mega_y's bf16 storage."""
        D = 64
        mega_y = torch.zeros((32, D), dtype=torch.bfloat16)
        T = 10
        y = mega_y[:T]
        y.fill_(1.0)
        out = y.float()  # this is what return y.float() does
        self.assertEqual(out.dtype, torch.float32)
        self.assertEqual(mega_y.dtype, torch.bfloat16)
        self.assertEqual(out.shape, (T, D))

    def test_transform_cache_trim_failure_is_best_effort(self):
        from rtp_llm.models_py.modules.dsv4.moe.strategies import mega

        error = RuntimeError("CUDA error: invalid argument")
        with patch.object(
            mega.torch.cuda, "empty_cache", side_effect=error
        ) as empty_cache, self.assertLogs(level="WARNING") as logs:
            mega._best_effort_empty_cache("test stage")

        empty_cache.assert_called_once_with()
        self.assertTrue(
            any(
                "test stage" in line and "invalid argument" in line
                for line in logs.output
            )
        )


class TestMegaSymmBufferRelease(unittest.TestCase):
    class _FakeBuffer:
        def __init__(self, name, error=None):
            self.name = name
            self.error = error
            self.buffer = torch.empty(1, dtype=torch.uint8)
            self.destroy_calls = 0

        def destroy(self):
            self.destroy_calls += 1
            if self.error is not None:
                raise self.error

    class _FakeStrategy:
        pass

    def setUp(self):
        from rtp_llm.models_py.modules.dsv4.moe import mega_buf

        self.mega_buf = mega_buf
        self._saved_graph_baked = mega_buf._MEGA_BUFFERS_GRAPH_BAKED
        self._saved_generation = mega_buf._MEGA_CUDA_GRAPH_GENERATION
        mega_buf._MEGA_BUFFERS_GRAPH_BAKED = False
        mega_buf._MEGA_BUF_CACHE.clear()
        mega_buf._MEGA_OUTPUT_CACHE.clear()
        self.strategies = []

    def tearDown(self):
        for strategy in self.strategies:
            self.mega_buf._MEGA_STRATEGY_REGISTRY.discard(strategy)
        self.mega_buf._MEGA_BUF_CACHE.clear()
        self.mega_buf._MEGA_OUTPUT_CACHE.clear()
        self.mega_buf._MEGA_CUDA_GRAPH_GENERATION = self._saved_generation
        self.mega_buf._MEGA_BUFFERS_GRAPH_BAKED = self._saved_graph_baked

    def _add_strategy(self, buf):
        strategy = self._FakeStrategy()
        strategy._mega_buf = buf
        strategy._mega_y = object()
        strategy._mega_group = object()
        self.strategies.append(strategy)
        self.mega_buf._MEGA_STRATEGY_REGISTRY.add(strategy)
        return strategy

    def test_release_destroys_cached_and_strategy_owned_buffers(self):
        shared = self._FakeBuffer("shared")
        strategy_only = self._FakeBuffer("strategy-only")
        self.mega_buf._MEGA_BUF_CACHE["first"] = shared
        self.mega_buf._MEGA_BUF_CACHE["alias"] = shared
        shared_owner = self._add_strategy(shared)
        strategy_only_owner = self._add_strategy(strategy_only)

        freed_gib = self.mega_buf.release_mega_symm_buffers()

        self.assertGreater(freed_gib, 0.0)
        self.assertEqual(shared.destroy_calls, 1)
        self.assertEqual(strategy_only.destroy_calls, 1)
        self.assertEqual(self.mega_buf._MEGA_BUF_CACHE, {})
        for strategy in (shared_owner, strategy_only_owner):
            self.assertIsNone(strategy._mega_buf)
            self.assertIsNone(strategy._mega_y)
            self.assertIsNone(strategy._mega_group)

    def test_partial_failure_still_destroys_remaining_and_clears_owners(self):
        failed = self._FakeBuffer("failed", RuntimeError("first destroy failed"))
        succeeded = self._FakeBuffer("succeeded")
        failed_owner = self._add_strategy(failed)
        succeeded_owner = self._add_strategy(succeeded)
        self.mega_buf._MEGA_BUF_CACHE["failed"] = failed
        self.mega_buf._MEGA_BUF_CACHE["succeeded"] = succeeded

        with self.assertRaises(RuntimeError):
            self.mega_buf.release_mega_symm_buffers()

        self.assertEqual(failed.destroy_calls, 1)
        self.assertEqual(succeeded.destroy_calls, 1)
        self.assertEqual(self.mega_buf._MEGA_BUF_CACHE, {})
        self.assertIsNone(failed_owner._mega_buf)
        self.assertIsNone(succeeded_owner._mega_buf)

    def test_destroy_failures_are_aggregated_in_final_error(self):
        first = self._FakeBuffer("first", ValueError("alpha"))
        second = self._FakeBuffer("second", OSError("beta"))
        self.mega_buf._MEGA_BUF_CACHE["first"] = first
        self.mega_buf._MEGA_BUF_CACHE["second"] = second

        with self.assertRaises(RuntimeError) as context:
            self.mega_buf.release_mega_symm_buffers()

        message = str(context.exception)
        self.assertIn("2 error(s)", message)
        self.assertIn("cache['first']", message)
        self.assertIn("ValueError: alpha", message)
        self.assertIn("cache['second']", message)
        self.assertIn("OSError: beta", message)

    def test_release_is_idempotent(self):
        buf = self._FakeBuffer("once")
        strategy = self._add_strategy(buf)
        self.mega_buf._MEGA_BUF_CACHE["once"] = buf

        self.mega_buf.release_mega_symm_buffers()
        self.mega_buf.release_mega_symm_buffers()

        self.assertEqual(buf.destroy_calls, 1)
        self.assertIsNone(strategy._mega_buf)
        self.assertEqual(self.mega_buf._MEGA_BUF_CACHE, {})


class TestMegaBufferGraphBakedGate(unittest.TestCase):
    """Sleep-time buffer releases must be NO-OPS when CUDA graphs are captured.

    Both Mega MoE buffers (symm ``_mega_buf`` + output staging ``_mega_y``) have
    their device pointers baked into the captured decode graph, so freeing them at
    sleep dangles the VA a post-wake replay writes into (illegal access, both
    ranks -> SIGABRT). When graphs are NOT captured (prefill) they must still be
    freed as before. GPU-free: seeds fake cache/registry entries and toggles the
    flag directly.
    """

    def setUp(self):
        from rtp_llm.models_py.modules.dsv4.moe import mega_buf

        self.mega_buf = mega_buf
        self._saved = mega_buf.mega_buffers_graph_baked()
        self._saved_generation = mega_buf._MEGA_CUDA_GRAPH_GENERATION
        self._saved_invalidated = mega_buf._MEGA_CUDA_GRAPH_INVALIDATED
        self._saved_owners = set(mega_buf._MEGA_CUDA_GRAPH_OWNERS)
        self._saved_invalidated_owners = set(
            mega_buf._MEGA_CUDA_GRAPH_INVALIDATED_OWNERS
        )
        mega_buf._MEGA_CUDA_GRAPH_OWNERS.clear()
        mega_buf._MEGA_CUDA_GRAPH_INVALIDATED_OWNERS.clear()
        mega_buf._MEGA_CUDA_GRAPH_INVALIDATED = False
        self._key = ("cpu", 8, torch.bfloat16)
        self._buf = torch.empty((4, 8), dtype=torch.bfloat16)
        mega_buf._MEGA_OUTPUT_CACHE.clear()
        mega_buf._MEGA_OUTPUT_CACHE[self._key] = self._buf

        class _FakeStrat:
            pass

        self.strat = _FakeStrat()
        self.strat._mega_y = self._buf
        self.strat._mega_buf = SimpleNamespace(destroy=lambda: None)
        mega_buf._MEGA_STRATEGY_REGISTRY.add(self.strat)

    def tearDown(self):
        self.mega_buf._MEGA_CUDA_GRAPH_OWNERS.clear()
        self.mega_buf._MEGA_CUDA_GRAPH_OWNERS.update(self._saved_owners)
        self.mega_buf._MEGA_CUDA_GRAPH_INVALIDATED_OWNERS.clear()
        self.mega_buf._MEGA_CUDA_GRAPH_INVALIDATED_OWNERS.update(
            self._saved_invalidated_owners
        )
        self.mega_buf._MEGA_CUDA_GRAPH_GENERATION = self._saved_generation
        self.mega_buf._MEGA_CUDA_GRAPH_INVALIDATED = self._saved_invalidated
        self.mega_buf._MEGA_BUFFERS_GRAPH_BAKED = self._saved
        self.mega_buf._MEGA_BUF_CACHE.clear()
        self.mega_buf._MEGA_OUTPUT_CACHE.clear()
        try:
            self.mega_buf._MEGA_STRATEGY_REGISTRY.discard(self.strat)
        except Exception:
            pass

    def test_output_release_is_noop_when_graph_baked(self):
        self.mega_buf.set_mega_buffers_graph_baked(True)
        self.assertEqual(self.mega_buf.release_mega_output_buffers(), (0, 0.0))
        # Buffer + per-strategy ref stay resident across sleep.
        self.assertEqual(len(self.mega_buf._MEGA_OUTPUT_CACHE), 1)
        self.assertIsNotNone(self.strat._mega_y)
        # Footprint is still reportable for the sleep-reclaim note.
        self.assertGreater(self.mega_buf.mega_output_buffer_gib(), 0.0)

    def test_output_release_frees_when_not_graph_baked(self):
        self.mega_buf.set_mega_buffers_graph_baked(False)
        entries, gib = self.mega_buf.release_mega_output_buffers()
        self.assertEqual(entries, 1)
        self.assertGreater(gib, 0.0)
        self.assertEqual(len(self.mega_buf._MEGA_OUTPUT_CACHE), 0)
        self.assertIsNone(self.strat._mega_y)

    def test_symm_release_is_noop_when_graph_baked(self):
        self.mega_buf.set_mega_buffers_graph_baked(True)
        # Returns early before dereferencing the opaque _mega_buf / cache.
        self.assertEqual(self.mega_buf.release_mega_symm_buffers(), 0.0)
        self.assertIsNotNone(self.strat._mega_buf)

    def test_symm_release_rebinds_rebuilt_process_group(self):
        from rtp_llm.models_py.modules.dsv4.moe.strategies import mega as strategy

        self.mega_buf.set_mega_buffers_graph_baked(False)
        generation = self.mega_buf.mega_buffer_generation()
        old_group = object()
        new_group = object()
        self.strat._mega_group = old_group
        self.strat._mega_buffer_generation = generation
        self.strat._mega_buf_kwargs = {}
        self.strat._mega_out_capacity_tokens = 4
        self.strat._mega_out_hidden = 8
        self.strat._mega_out_device = torch.device("cpu")

        self.mega_buf.release_mega_symm_buffers()

        self.assertEqual(self.mega_buf.mega_buffer_generation(), generation + 1)
        self.assertIsNone(self.strat._mega_group)

        new_symm = SimpleNamespace(num_max_tokens_per_rank=8)
        new_output = object()
        with patch.object(
            strategy, "_get_or_create_mega_buf", return_value=new_symm
        ) as create_symm, patch.object(
            strategy, "_get_or_create_mega_output", return_value=new_output
        ), patch.object(
            strategy.torch.distributed, "is_initialized", return_value=True
        ), patch.object(
            strategy.torch.distributed,
            "group",
            SimpleNamespace(WORLD=new_group),
        ):
            strategy.MegaMoEStrategy._ensure_mega_buffers(self.strat)

        self.assertIs(create_symm.call_args.kwargs["group"], new_group)
        self.assertIs(self.strat._mega_group, new_group)
        self.assertIs(self.strat._mega_buf, new_symm)
        self.assertIs(self.strat._mega_y, new_output)

    def test_l3_invalidation_rebuilds_graph_baked_buffers_for_two_cycles(self):
        from rtp_llm.models_py.modules.dsv4.moe.strategies import mega as strategy

        self.mega_buf.set_mega_buffers_graph_baked(True)
        expected_generation = self.mega_buf.mega_buffer_generation()

        for _ in range(2):
            self.strat._mega_buf = SimpleNamespace(destroy=lambda: None)
            self.strat._mega_y = self._buf
            self.strat._mega_buffer_generation = expected_generation
            self.mega_buf._MEGA_OUTPUT_CACHE[self._key] = self._buf

            generation = self.mega_buf.invalidate_mega_cuda_graph_resources()
            expected_generation += 1
            self.assertEqual(generation, expected_generation)
            self.assertTrue(self.mega_buf.mega_cuda_graph_resources_invalidated())
            self.assertIsNone(self.strat._mega_buf)
            self.assertIsNone(self.strat._mega_y)
            self.assertEqual(self.mega_buf._MEGA_OUTPUT_CACHE, {})

            self.assertEqual(
                self.mega_buf.invalidate_mega_cuda_graph_resources(), generation
            )

            new_symm = SimpleNamespace(num_max_tokens_per_rank=8)
            new_output = object()
            old_group = object()
            new_group = object()
            self.strat._mega_group = old_group
            self.strat._mega_buf_kwargs = {}
            self.strat._mega_out_capacity_tokens = 4
            self.strat._mega_out_hidden = 8
            self.strat._mega_out_device = torch.device("cpu")
            with patch.object(
                strategy, "_get_or_create_mega_buf", return_value=new_symm
            ) as create_symm, patch.object(
                strategy, "_get_or_create_mega_output", return_value=new_output
            ) as create_output, patch.object(
                strategy.torch.distributed, "is_initialized", return_value=True
            ), patch.object(
                strategy.torch.distributed,
                "group",
                SimpleNamespace(WORLD=new_group),
            ):
                strategy.MegaMoEStrategy._ensure_mega_buffers(self.strat)

            create_symm.assert_called_once()
            self.assertIs(create_symm.call_args.kwargs["group"], new_group)
            self.assertIs(self.strat._mega_group, new_group)
            create_output.assert_called_once()
            self.assertIs(self.strat._mega_buf, new_symm)
            self.assertIs(self.strat._mega_y, new_output)
            self.assertEqual(self.strat._mega_buffer_generation, generation)

            self.mega_buf.mark_mega_cuda_graph_resources_recaptured()
            self.assertFalse(self.mega_buf.mega_cuda_graph_resources_invalidated())

    def test_level_three_rejects_opt_in_fused_mega(self):
        from rtp_llm.model_loader import weight_memory_saver
        from rtp_llm.models_py.modules.dsv4.moe import mega_fused_buf

        with patch.dict(os.environ, {"DSV4_USE_MEGA_MOE_FUSED": "1"}), patch.object(
            weight_memory_saver, "is_enabled", return_value=True
        ), patch.object(weight_memory_saver, "sleep_mode_level", return_value=3):
            with self.assertRaisesRegex(RuntimeError, "does not support"):
                mega_fused_buf._mega_moe_fused_enabled()

    def test_l3_waits_for_every_mtp_graph_owner_before_release(self):
        self.mega_buf.register_mega_cuda_graph_owner(101)
        self.mega_buf.register_mega_cuda_graph_owner(202)
        generation = self.mega_buf.mega_buffer_generation()

        self.assertEqual(
            self.mega_buf.invalidate_mega_cuda_graph_resources(101), generation
        )
        self.assertIsNotNone(self.strat._mega_buf)
        self.assertIsNotNone(self.strat._mega_y)

        self.assertEqual(
            self.mega_buf.invalidate_mega_cuda_graph_resources(202), generation + 1
        )
        self.assertTrue(self.mega_buf.mega_cuda_graph_resources_invalidated())
        self.assertIsNone(self.strat._mega_buf)
        self.assertIsNone(self.strat._mega_y)

        self.mega_buf.mark_mega_cuda_graph_resources_recaptured(101)
        self.assertTrue(self.mega_buf.mega_cuda_graph_resources_invalidated())
        self.mega_buf.mark_mega_cuda_graph_resources_recaptured(101)
        self.assertTrue(self.mega_buf.mega_cuda_graph_resources_invalidated())

        self.mega_buf.mark_mega_cuda_graph_resources_recaptured(202)
        self.assertFalse(self.mega_buf.mega_cuda_graph_resources_invalidated())


if __name__ == "__main__":
    unittest.main()
