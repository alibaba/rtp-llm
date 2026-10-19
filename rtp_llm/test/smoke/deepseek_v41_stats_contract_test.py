"""TP4/EP4 routing-statistics contract, with intentional Stats-on eager fallback.

The separate graph contract is unchanged and must still pass with stats off.
All generation/cache/full-prefill comparisons from that contract are retained.
"""
import json
import logging
import os
import re
import unittest
from unittest.mock import patch

from rtp_llm.test.smoke import deepseek_v41_contract_test as contract


class DeepSeekV41StatsContractTest(contract.DeepSeekV41ContractTest):
    def server_args(self):
        return super().server_args() + " --eplb_mode STATS"

    def start_servers(self):
        with patch.dict(os.environ, {"EPLB_STATS_VALIDATE": "1"}):
            super().start_servers()

    def assert_decode_graph_replayed_and_record_coverage(self):
        with open(self.graph_server.log_file_path, "rb") as reader:
            reader.seek(self.log_offset)
            runtime = reader.read().decode(errors="replace")
        produced = re.findall(
            r"EPLB_STATS_PRODUCED layers=(\d+) ep=(\d+) assignments=(\d+) layer0=(\[[^\n]*?\])", runtime
        )
        reported = re.findall(
            r"EPLB_STATS_REPORTED steps=(\d+) layers=(\d+) ep=(\d+) assignments=(\d+) ib_layer_mean=([0-9.]+) tp_replication=(\d+)", runtime
        )
        self.assertGreater(len(produced), 0, "No model-side stats validated")
        self.assertGreater(len(reported), 1, "Need multiple consumed/reset reporting windows")
        for layers, ep, assignments, layer0 in produced:
            self.assertEqual(int(layers), 40)
            self.assertEqual(int(ep), 4)
            self.assertGreater(int(assignments), 0)
            self.assertEqual(sum(json.loads(layer0)) * int(layers), int(assignments))
        for steps, layers, ep, assignments, imbalance, replication in reported:
            self.assertEqual((int(steps), int(layers), int(ep)), (10, 40, 4))
            self.assertGreater(int(assignments), 0)
            self.assertTrue(1 <= float(imbalance) <= 4)
            self.assertEqual(int(replication), 1)  # raw dispatched work, no unverified TP dedup
        self.assertNotIn("using CUDA graph forward", runtime)
        self.assertNotIn("Discarding EPLB reporting window", runtime)
        (self.output_dir / "v41_stats_coverage.json").write_text(json.dumps({
            "requests": self.observed, "producer_validations": len(produced),
            "report_windows": reported, "stats_on_execution": "eager",
            "graph_flag": 1, "count_semantics": "raw dispatch assignments including TP replicas",
        }, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
