import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


spec = importlib.util.spec_from_file_location(
    "k3_launch_bf16",
    Path(__file__).resolve().parents[2] / "example/k3/main_migration/launch_bf16.py",
)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class GpuCapacityTest(unittest.TestCase):
    def probe(self, *, shared=False, count=8, low_gpu=None, processes="", minimum=250):
        gpus = "\n".join(f"{i}, GPU-{i}, {249 * 1024 if i == low_gpu else 270 * 1024}"
                         for i in range(count))
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(launcher.subprocess, "check_output", side_effect=[gpus, processes]):
                launcher.require_gpu_capacity(directory, shared, minimum)
            return json.loads((Path(directory) / "gpu-preflight.json").read_text())

    def test_idle_full_profile(self):
        result = self.probe()
        self.assertFalse(result["allow_shared_accuracy"])
        self.assertFalse(result["performance_validated"])

    def test_sharing_requires_explicit_accuracy_opt_in(self):
        processes = "GPU-0, 42, external training, 1000\nGPU-7, 50, health probe, 1\n"
        with self.assertRaisesRegex(RuntimeError, "occupied"):
            self.probe(processes=processes)
        result = self.probe(processes=processes, shared=True)
        self.assertEqual(result["compute_processes"], processes)
        self.assertTrue(result["allow_shared_accuracy"])
        self.assertFalse(result["performance_validated"])

    def test_sharing_does_not_waive_capacity_on_any_rank(self):
        for rank in range(8):
            with self.subTest(rank=rank), self.assertRaisesRegex(RuntimeError, "Insufficient"):
                self.probe(shared=True, low_gpu=rank)

    def test_missing_rank_rejects(self):
        with self.assertRaisesRegex(RuntimeError, "Insufficient"):
            self.probe(shared=True, count=7)

    def test_unselected_gpu_does_not_block_tp8(self):
        self.probe(count=9, processes="GPU-8, 42, external training, 1000\n")

    def test_invalid_capacity_rejects(self):
        for value in (0, -1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.probe(shared=True, minimum=value)

    def test_failure_preserves_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(launcher.subprocess, "check_output", side_effect=["0, GPU-0, 1", ""]):
                with self.assertRaises(RuntimeError):
                    launcher.require_gpu_capacity(directory, True)
            self.assertEqual(json.loads((Path(directory) / "gpu-preflight.json").read_text())["gpus"],
                             "0, GPU-0, 1")


if __name__ == "__main__":
    unittest.main()
