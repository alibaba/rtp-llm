import json
import tempfile
import unittest
from pathlib import Path

from flexlb_test_framework.workload.evidence import join_evidence


class EvidenceJoinTest(unittest.TestCase):
    def test_same_request_id_in_other_environment_is_not_joined(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            environments = {}
            for epoch in ("1", "2"):
                path = root / epoch
                path.mkdir()
                (path / "engine_events.jsonl").write_text(
                    json.dumps(dict(rid=7, batch_id=epoch, event="prefill_done")) + "\n"
                )
                environments[epoch] = str(path)
            payload = dict(
                instance_id="test",
                clock_anchor=dict(epoch_s=100, monotonic_s=10),
                phases=[dict(event="start", env_epoch=1, monotonic_s=10, stage="load")],
                request_resources=[
                    dict(
                        resource=dict(env_epoch=1, kind="requests"),
                        records=[dict(wire_request_id=7, attempt=1, issued_s=11)],
                    )
                ],
            )
            result = join_evidence(payload, environments)
            self.assertEqual(result["requests"][0]["batch_ids"], ["1"])
            self.assertEqual(result["requests"][0]["phase"], "load")
            self.assertIsNone(result["requests"][0]["endpoint_generation"])

    def test_attempt_crossing_restart_is_not_assigned_a_false_generation(self):
        from flexlb_test_framework.workload.evidence import master_incarnation

        history = [
            dict(target="A", generation=1, started_epoch_ms=100),
            dict(target="A", generation=2, started_epoch_ms=200),
        ]
        self.assertEqual(master_incarnation(history, "A", 110, 190)["generation"], 1)
        self.assertEqual(master_incarnation(history, "A", 210, 220)["generation"], 2)
        self.assertIsNone(master_incarnation(history, "A", 190, 210))
        self.assertIsNone(master_incarnation(history, "B", 210, 220))
