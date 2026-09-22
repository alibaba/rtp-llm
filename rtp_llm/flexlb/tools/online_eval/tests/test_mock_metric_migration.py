"""Exercise the renamed series through collection, consolidation and aggregation."""

import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.consolidate import parse_grouped_prometheus_timeseries
from test_cache_hit_metrics import _write_full_run, _run, AGGREGATE, T0


class MetricMigrationTest(unittest.TestCase):
    def test_new_names_survive_entire_analysis_chain(self):
        labels = '{role="prefill",engine_ip="10.1.1.1",engine_name="p1"}'
        metrics = {
            "rtp_llm_running_stream_size": 2,
            "rtp_llm_wait_stream_size": 3,
            "rtp_llm_kv_cache_pool_total_blocks": 100,
            "rtp_llm_kv_cache_pool_available_blocks": 70,
            "mock_engine_held_blocks": 20,
            "mock_engine_referenced_blocks": 10,
        }
        body = "\n".join(f"{name}{labels} {value}" for name, value in metrics.items())
        body += "\nmock_engine_running" + labels + " 99\n"
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_full_run(root)
            prom = root / "metrics.prom"
            # Archived text parsing remains a read-only compatibility utility.
            prom.write_text(
                f"# ts={T0}\n"
                + "\n".join(
                    f"{name}{labels} {value}" for name, value in metrics.items()
                )
                + "\n"
            )
            rows = parse_grouped_prometheus_timeseries(prom)
            self.assertEqual(
                set(rows[0]["metrics"]), {name + labels for name in metrics}
            )
            rows += [{**rows[0], "ts": T0 + 1000}, {**rows[0], "ts": T0 + 2000}]
            with gzip.open(root / "mock_per_engine_timeseries.json.gz", "wt") as f:
                json.dump(rows, f)
            aggregate = json.loads(_run([AGGREGATE], root).stdout)
            self.assertTrue(aggregate["queue_top_bottom_ts"]["p_running"]["top"])
            self.assertTrue(aggregate["queue_top_bottom_ts"]["p_waiting"]["top"])
            kv = aggregate["kv_blocks_ts_by_role"]["prefill"][0]
            self.assertEqual(kv["total_blocks"], 100)
            self.assertEqual(kv["available_blocks"], 70)
            self.assertEqual(kv["held_blocks"] + kv["referenced_blocks"], 30)


if __name__ == "__main__":
    unittest.main()
