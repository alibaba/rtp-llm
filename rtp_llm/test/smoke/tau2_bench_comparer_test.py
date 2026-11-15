import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from smoke.common_def import SmokeException
from smoke.tau2_bench_comparer import Tau2BenchComparer


class Tau2BenchComparerTest(unittest.TestCase):
    def run_gate(self, scored_tasks, score, threshold=0.60):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks = root / "tasks.json"
            tasks.write_text(json.dumps({"retail": list(map(str, range(22)))}))
            report = root / "report.json"
            report.write_text(
                json.dumps(
                    {
                        "score": score,
                        "metrics": [
                            {"name": "mean_acc", "score": score, "num": scored_tasks}
                        ],
                    }
                )
            )
            comparer = object.__new__(Tau2BenchComparer)
            comparer.server_manager = SimpleNamespace(port=12345)
            comparer.qr_info = {"tau2_threshold": threshold}
            # Exercise the real gate and report parser; installation and the
            # benchmark subprocess are replaced by its on-disk report contract.
            with patch.multiple(
                comparer,
                _prepare_tau2=Mock(return_value=str(root)),
                _install_evalscope=Mock(),
                _install_tau2_from_tarball=Mock(),
                _install_litellm=Mock(),
                _prepare_tau2_data=Mock(return_value=None),
                _resolve_task_ids_file=Mock(return_value=str(tasks)),
                _resolve_script_file=Mock(return_value="unused.py"),
                _run_tau2_script=Mock(return_value=f"Dump report to: {report}"),
            ):
                comparer.run()

    def test_complete_report_passes_at_threshold(self):
        self.run_gate(22, 0.60)

    def test_one_surviving_task_cannot_hide_21_failures(self):
        with self.assertRaises(SmokeException):
            self.run_gate(1, 1.0)

    def test_ignored_errors_lower_the_gate_score(self):
        # The real CI report scored 17/20 after two simulations raised.
        # Across the selected 22 tasks this is 17/22, below an 80% gate.
        with self.assertRaises(SmokeException):
            self.run_gate(20, 0.85, threshold=0.80)

    def test_partial_report_can_still_meet_the_fixed_threshold(self):
        self.run_gate(20, 0.85)


if __name__ == "__main__":
    unittest.main()
