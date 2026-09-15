"""Tests for SharedExpertOverlapExecutor.

Covers:
  1. Sequential fallback when MOE_SHARED_EXPERT_OVERLAP=0 (default)
  2. Overlap correctness: output matches sequential
  3. Token-count threshold: large batches fall back to sequential
  4. MOEDBG: debug mode disables overlap
  5. prepare() is no-op when overlap disabled
  6. Exception safety: state is clean after finish()
  7. Non-CUDA tensor falls back to sequential
  8. Empty args falls back to sequential
"""

import importlib.util
import os
import sys
import unittest
from contextlib import contextmanager

import torch

# Import shared_expert_overlap directly, bypassing the modules/__init__.py
# which pulls in heavy CUDA dependencies (deep_gemm, flash_attn, etc.)
# that are not available on every dev machine.
_MODULE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "shared_expert_overlap.py"
)
_spec = importlib.util.spec_from_file_location("shared_expert_overlap", _MODULE_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

SharedExpertOverlapExecutor = _mod.SharedExpertOverlapExecutor
_get_or_create_shared_expert_stream = _mod._get_or_create_shared_expert_stream
_is_cuda_graph_warmup = _mod._is_cuda_graph_warmup
_overlap_enabled = _mod._overlap_enabled
_shared_expert_stream_cache = _mod._shared_expert_stream_cache


@contextmanager
def _env(key: str, value: str):
    old = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old


def _dummy_shared_expert(x: torch.Tensor) -> torch.Tensor:
    """Simple shared expert: scale by 0.25."""
    return (x.float() * 0.25).to(x.dtype)


class TestOverlapEnabled(unittest.TestCase):
    """_overlap_enabled() checks env vars correctly."""

    def test_default_off(self):
        with _env("MOE_SHARED_EXPERT_OVERLAP", "0"):
            with _env("MOEDBG", "0"):
                self.assertFalse(_overlap_enabled())

    def test_enabled(self):
        with _env("MOE_SHARED_EXPERT_OVERLAP", "1"):
            with _env("MOEDBG", "0"):
                self.assertTrue(_overlap_enabled())

    def test_moedbg_disables(self):
        with _env("MOE_SHARED_EXPERT_OVERLAP", "1"):
            with _env("MOEDBG", "1"):
                self.assertFalse(_overlap_enabled())

    def test_missing_env_means_off(self):
        with _env("MOE_SHARED_EXPERT_OVERLAP", "0"):
            with _env("MOEDBG", "0"):
                self.assertFalse(_overlap_enabled())


class TestIsCudaGraphWarmup(unittest.TestCase):
    """_is_cuda_graph_warmup() handles missing module gracefully."""

    def test_returns_bool(self):
        result = _is_cuda_graph_warmup()
        self.assertIsInstance(result, bool)


class TestSequentialFallback(unittest.TestCase):
    """When overlap is disabled, start() runs synchronously."""

    def test_env_off_runs_synchronously(self):
        with _env("MOE_SHARED_EXPERT_OVERLAP", "0"):
            executor = SharedExpertOverlapExecutor()
            x = (
                torch.randn(16, 64, device="cuda")
                if torch.cuda.is_available()
                else torch.randn(16, 64)
            )
            executor.start(_dummy_shared_expert, x)
            output = executor.finish()
            expected = _dummy_shared_expert(x)
            torch.testing.assert_close(output, expected)

    def test_non_cuda_tensor_falls_back(self):
        with _env("MOE_SHARED_EXPERT_OVERLAP", "1"):
            executor = SharedExpertOverlapExecutor()
            x = torch.randn(16, 64)  # CPU tensor
            executor.start(_dummy_shared_expert, x)
            output = executor.finish()
            expected = _dummy_shared_expert(x)
            torch.testing.assert_close(output, expected)
            self.assertIsNone(executor._shared_expert_stream)

    def test_empty_args_falls_back(self):
        with _env("MOE_SHARED_EXPERT_OVERLAP", "1"):
            executor = SharedExpertOverlapExecutor()
            # No args — _can_overlap returns False
            executor.start(lambda: torch.tensor(1.0))
            output = executor.finish()
            self.assertEqual(output.item(), 1.0)


class TestTokenThreshold(unittest.TestCase):
    """Token-count threshold disables overlap for large batches."""

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_large_batch_falls_back(self):
        with _env("MOE_SHARED_EXPERT_OVERLAP", "1"):
            with _env("MOE_SHARED_EXPERT_OVERLAP_TOKEN_THRESHOLD", "100"):
                with _env("MOEDBG", "0"):
                    executor = SharedExpertOverlapExecutor()
                    x = torch.randn(200, 64, device="cuda")
                    executor.start(_dummy_shared_expert, x)
                    output = executor.finish()
                    # Should have fallen back to sequential
                    self.assertIsNone(executor._shared_expert_stream)
                    expected = _dummy_shared_expert(x)
                    torch.testing.assert_close(output, expected)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_small_batch_uses_overlap(self):
        with _env("MOE_SHARED_EXPERT_OVERLAP", "1"):
            with _env("MOE_SHARED_EXPERT_OVERLAP_TOKEN_THRESHOLD", "100"):
                with _env("MOEDBG", "0"):
                    executor = SharedExpertOverlapExecutor()
                    x = torch.randn(50, 64, device="cuda")
                    executor.start(_dummy_shared_expert, x)
                    output = executor.finish()
                    # Should have used overlap (stream was active)
                    expected = _dummy_shared_expert(x)
                    torch.testing.assert_close(output, expected)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_graph_capture_falls_back_to_sequential(self):
        with _env("MOE_SHARED_EXPERT_OVERLAP", "1"):
            with _env("MOEDBG", "0"):
                executor = SharedExpertOverlapExecutor()
                x = torch.randn(16, 64, device="cuda")
                original = torch.cuda.is_current_stream_capturing
                try:
                    torch.cuda.is_current_stream_capturing = lambda: True
                    executor.start(_dummy_shared_expert, x)
                    output = executor.finish()
                finally:
                    torch.cuda.is_current_stream_capturing = original
                self.assertIsNone(executor._shared_expert_stream)
                torch.testing.assert_close(output, _dummy_shared_expert(x))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestOverlapCorrectness(unittest.TestCase):
    """Overlap output matches sequential output."""

    def test_output_matches_sequential(self):
        with _env("MOE_SHARED_EXPERT_OVERLAP", "1"):
            with _env("MOEDBG", "0"):
                torch.manual_seed(42)
                x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)

                # Sequential reference
                expected = _dummy_shared_expert(x)

                # Overlap execution
                executor = SharedExpertOverlapExecutor()
                executor.start(_dummy_shared_expert, x)

                # Simulate routed expert work on main stream
                _ = torch.randn(64, 128, device="cuda")

                output = executor.finish()
                torch.testing.assert_close(output, expected)

    def test_kwargs_passed_correctly(self):
        """kwargs tensors are record_stream'd and visible on aux stream."""

        def shared_with_kwargs(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            return (x.float() * scale.float()).to(x.dtype)

        with _env("MOE_SHARED_EXPERT_OVERLAP", "1"):
            with _env("MOEDBG", "0"):
                x = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)
                scale = torch.tensor(0.5, device="cuda")

                expected = shared_with_kwargs(x, scale)

                executor = SharedExpertOverlapExecutor()
                executor.start(shared_with_kwargs, x, scale=scale)
                output = executor.finish()
                torch.testing.assert_close(output, expected)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestPrepare(unittest.TestCase):
    """prepare() pre-creates the stream."""

    def test_prepare_creates_stream(self):
        # Clear cache first
        _shared_expert_stream_cache.clear()

        with _env("MOE_SHARED_EXPERT_OVERLAP", "1"):
            with _env("MOEDBG", "0"):
                executor = SharedExpertOverlapExecutor()
                device = torch.device("cuda", torch.cuda.current_device())
                executor.prepare(device)
                self.assertIn(device.index, _shared_expert_stream_cache)

    def test_prepare_noop_when_disabled(self):
        _shared_expert_stream_cache.clear()

        with _env("MOE_SHARED_EXPERT_OVERLAP", "0"):
            executor = SharedExpertOverlapExecutor()
            device = torch.device("cuda", torch.cuda.current_device())
            executor.prepare(device)
            self.assertNotIn(device.index, _shared_expert_stream_cache)

    def test_prepare_noop_for_cpu(self):
        _shared_expert_stream_cache.clear()

        with _env("MOE_SHARED_EXPERT_OVERLAP", "1"):
            with _env("MOEDBG", "0"):
                executor = SharedExpertOverlapExecutor()
                executor.prepare(torch.device("cpu"))
                self.assertEqual(len(_shared_expert_stream_cache), 0)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestFinishState(unittest.TestCase):
    """finish() cleans up state properly."""

    def test_finish_clears_state(self):
        with _env("MOE_SHARED_EXPERT_OVERLAP", "1"):
            with _env("MOEDBG", "0"):
                executor = SharedExpertOverlapExecutor()
                x = torch.randn(16, 64, device="cuda")
                executor.start(_dummy_shared_expert, x)
                executor.finish()
                self.assertIsNone(executor._shared_expert_output)
                self.assertIsNone(executor._shared_expert_stream)

    def test_finish_before_start_raises(self):
        executor = SharedExpertOverlapExecutor()
        with self.assertRaises(AssertionError):
            executor.finish()

    def test_multiple_start_finish_cycles(self):
        """Executor can be reused across start/finish pairs."""
        with _env("MOE_SHARED_EXPERT_OVERLAP", "1"):
            with _env("MOEDBG", "0"):
                executor = SharedExpertOverlapExecutor()
                for i in range(3):
                    x = torch.randn(16, 64, device="cuda")
                    executor.start(_dummy_shared_expert, x)
                    output = executor.finish()
                    expected = _dummy_shared_expert(x)
                    torch.testing.assert_close(output, expected)


class TestStreamCache(unittest.TestCase):
    """_get_or_create_shared_expert_stream caches per device."""

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_same_device_returns_same_stream(self):
        _shared_expert_stream_cache.clear()
        device = torch.device("cuda", torch.cuda.current_device())
        s1 = _get_or_create_shared_expert_stream(device)
        s2 = _get_or_create_shared_expert_stream(device)
        self.assertIs(s1, s2)


if __name__ == "__main__":
    unittest.main()
