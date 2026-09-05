import json
import tempfile
import unittest
from pathlib import Path

from rtp_llm.models_py.modules.factory.fused_moe.utils.timeline import check_timeline


class TimelineCheckerTest(unittest.TestCase):
    def _timeline(self, events) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "trace.json"
        path.write_text(json.dumps({"traceEvents": events}))
        return path

    def _empty_timeline(self) -> Path:
        return self._timeline([])

    def test_missing_hot_ranges_fail_by_default(self):
        self.assertEqual(check_timeline(self._empty_timeline()), 1)

    def test_missing_hot_ranges_can_be_explicitly_allowed(self):
        self.assertEqual(check_timeline(self._empty_timeline(), allow_empty=True), 0)

    def test_hot_range_without_gpu_kernel_fails_by_default(self):
        path = self._timeline(
            [
                {
                    "ph": "X",
                    "name": "moe.routed_experts",
                    "cat": "gpu_user_annotation",
                    "ts": 0,
                    "dur": 10,
                }
            ]
        )
        self.assertEqual(check_timeline(path), 1)
        self.assertEqual(check_timeline(path, allow_empty=True), 0)

    def test_hot_range_with_gpu_kernel_succeeds(self):
        path = self._timeline(
            [
                {
                    "ph": "X",
                    "name": "moe.routed_experts",
                    "cat": "gpu_user_annotation",
                    "ts": 0,
                    "dur": 10,
                },
                {"ph": "X", "name": "deep_gemm", "cat": "kernel", "ts": 1, "dur": 2},
            ]
        )
        self.assertEqual(check_timeline(path), 0)


if __name__ == "__main__":
    unittest.main()
