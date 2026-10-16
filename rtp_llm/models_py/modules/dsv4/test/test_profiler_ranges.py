"""Keep opt-in traces visible without per-layer dispatcher calls in serving."""

import importlib.util
import unittest
from pathlib import Path
from unittest import mock

import torch

_SPEC = importlib.util.spec_from_file_location(
    "dsv4_profiler_under_test", Path(__file__).parents[1] / "_profiler.py"
)
profiler = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(profiler)


class ProfilerRangesTest(unittest.TestCase):
    def test_inactive_does_not_enter_torch_dispatcher(self):
        with mock.patch.object(
            torch.profiler,
            "record_function",
            side_effect=AssertionError("inactive range"),
        ):
            with profiler.record_function_range("dsv4.inactive"):
                pass

    def test_profiling_can_start_after_serving(self):
        for _ in range(2):
            with profiler.record_function_range("dsv4.inactive"):
                pass
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU]
            ) as trace:
                with profiler.record_function_range("dsv4.active"):
                    with profiler.record_function_range("dsv4.nested"):
                        torch.ones(2).add_(1)
            names = {event.name for event in trace.events()}
            self.assertTrue({"dsv4.active", "dsv4.nested"} <= names)
            self.assertNotIn("dsv4.inactive", names)
        self.test_inactive_does_not_enter_torch_dispatcher()

    def test_exception_propagates(self):
        with self.assertRaisesRegex(ValueError, "forward failed"):
            with profiler.record_function_range("dsv4.error"):
                raise ValueError("forward failed")


if __name__ == "__main__":
    unittest.main()
