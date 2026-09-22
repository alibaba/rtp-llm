import json
import tempfile
import unittest
from pathlib import Path

from analysis.compare_ab import main


class MonitorCompareTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def run_data(self, name, value, *, valid=True, traffic="same", duration="20"):
        folder = self.root / name
        folder.mkdir()
        data = {
            "monitor_backend": "prometheus",
            "summary": {"test_valid": valid},
            "errors": [] if valid else [{"query": "missing", "error": "monitor series absent"}],
            "collection_gaps": {},
            "series": {"0/mock/throughput/{}": [[5, value], [6, value]]},
            "statistic_sources": {"0/mock/throughput/{}": {"promql": "sum(rate(x[1m]))"}},
            "run_meta": {"workload": {"inputs": [{"sha256": traffic}]},
                         "configuration": {"master_config.json": {"name": "m"},
                                           "mode_plan.json": {"name": "p"},
                                           "client_env.json": {"DURATION_S": duration,
                                                               "OUTPUT_DIR": str(folder),
                                                               "SHARD_INDEX": name}}},
        }
        (folder / "aggregate.json").write_text(json.dumps(data))
        return folder

    def test_valid_monitor_archive_is_descriptive(self):
        a = self.run_data("a", 10)
        b = self.run_data("b", 12)
        output = self.root / "ab.json"
        self.assertEqual(0, main(["--run-a", str(a), "--run-b", str(b),
                                  "--out", str(output), "--html"]))
        result = json.loads(output.read_text())
        self.assertEqual("DESCRIPTIVE_ONLY", result["verdict"])
        self.assertAlmostEqual(20, result["changes"][0]["relative_pct"])
        self.assertTrue((self.root / "reports/comparison/stress-monitor-ab/report.html").exists())

    def test_invalid_or_different_experiment_fails_closed(self):
        a = self.run_data("a", 10)
        invalid = self.run_data("invalid", 12, valid=False)
        changed = self.run_data("changed", 12, traffic="different")
        for b in (invalid, changed):
            self.assertEqual(2, main(["--run-a", str(a), "--run-b", str(b),
                                      "--out", str(self.root / "ab.json")]))
            self.assertFalse((self.root / "ab.json").exists())

    def test_no_paired_samples_fails_closed(self):
        a = self.run_data("a", 10)
        b = self.run_data("b", 12)
        self.assertEqual(2, main(["--run-a", str(a), "--run-b", str(b),
                                  "--steady-lo", "10", "--steady-hi", "11", "--out", "-"]))
