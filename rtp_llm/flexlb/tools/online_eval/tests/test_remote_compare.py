import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from remote_compare import Incomparable, compare_exports, load_export


def fixture(values, *, traffic="same"):
    return {
        "schema_version": 1,
        "provenance": {
            "deployment": "unit", "hippo_app": "unit", "traffic_fingerprint": traffic,
            "master_mode": "sb", "fetch_output_stream": True,
            "prefill_engines": 2, "decode_engines": 4,
            "start_ms": 0, "end_ms": 10000, "granularity_ms": 1000,
            "collection_source": "kmonitor-export",
        },
        "series": {"ttft": {
            "metric": "py_rtp_response_first_token_rt", "role": "frontend",
            "unit": "ms", "spatial_aggregation": "avg",
            "temporal_aggregation": "avg",
            "points": [{"t_ms": t, "value": v} for t, v in values],
        }},
    }


class RemoteCompareTest(unittest.TestCase):
    def test_paired_samples_and_gaps(self):
        a = fixture([(1000, 10), (2000, 20), (3000, 30), (4000, 40)])
        b = fixture([(1000, 12), (2500, 50), (3000, 32), (4000, 42)])
        report, chart = compare_exports(a, b, steady_lo_ms=1000, steady_hi_ms=4000)
        metric = report["metrics"][0]
        self.assertEqual(metric["paired_samples"], 3)
        self.assertEqual(metric["delta_mean"], 2)
        self.assertEqual(chart["panels"][0]["series"][0]["data"],
                         [10, 20, None, 30, 40])

    def test_different_traffic_rejected(self):
        with self.assertRaisesRegex(Incomparable, "traffic_fingerprint"):
            compare_exports(fixture([(1000, 1)]), fixture([(1000, 2)], traffic="other"),
                            steady_lo_ms=1000, steady_hi_ms=4000)

    def test_incomplete_metric_semantics_rejected(self):
        with tempfile.TemporaryDirectory() as dirname:
            path = Path(dirname) / "export.json"
            document = fixture([(1000, 1)])
            del document["series"]["ttft"]["spatial_aggregation"]
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(Incomparable, "semantics"):
                load_export(path)


if __name__ == "__main__":
    unittest.main()
