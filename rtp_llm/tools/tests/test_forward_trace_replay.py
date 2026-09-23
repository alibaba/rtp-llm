import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("replay", Path(__file__).parents[1] / "forward_trace_replay.py")
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


def fixture():
    return {"rtp_forward_metadata": {"schema_version": 1, "complete": True, "forwards": [{
        "forward_id": 1, "parent_forward_id": 0, "status": "ok", "kind": "model",
        "lengths_complete": True, "phase": "prefill_target", "logical_sequences": 2,
        "q_lens": [8192, 128], "prefix_lens": [0, 32640], "kv_lens": [8192, 32768],
        "total_q_tokens": 8320, "physical_tokens": 8320, "physical_requests": 2,
        "tp_size": 8, "recorded_inputs": {},
    }]}}


class ReplayTest(unittest.TestCase):
    def test_heterogeneous_batch(self):
        case = replay.export_case(fixture(), 1)
        self.assertEqual(case["total_q_tokens"], 8320)
        self.assertEqual(case["requests"][1]["prefix_len"], 32640)
        self.assertEqual(case["requests"][1]["kv_len"], 32768)

    def test_chunk_inherits_config_and_keeps_row_mapping(self):
        trace = fixture()
        child = copy.deepcopy(trace["rtp_forward_metadata"]["forwards"][0])
        child.update(forward_id=2, parent_forward_id=1, kind="chunk", chunk_index=3)
        child.pop("tp_size")
        child["recorded_inputs"] = {"original_batch_indices": [5, 2]}
        trace["rtp_forward_metadata"]["forwards"].append(child)
        case = replay.export_case(trace, 2)
        self.assertEqual(case["model_config"]["tp_size"], 8)
        self.assertEqual([r["original_batch_row"] for r in case["requests"]], [5, 2])

    def test_reject_incomplete_and_inconsistent_records(self):
        for mutation in ("incomplete", "kv", "sum", "size", "failed"):
            with self.subTest(mutation=mutation):
                trace = fixture()
                record = trace["rtp_forward_metadata"]["forwards"][0]
                if mutation == "incomplete": trace["rtp_forward_metadata"]["complete"] = False
                if mutation == "kv": record["kv_lens"][1] += 1
                if mutation == "sum": record["total_q_tokens"] += 1
                if mutation == "size": record["logical_sequences"] += 1
                if mutation == "failed": record["status"] = "error"
                with self.assertRaises(ValueError): replay.export_case(trace, 1)

    def test_reject_parent_cycle(self):
        trace = fixture()
        trace["rtp_forward_metadata"]["forwards"][0]["parent_forward_id"] = 1
        with self.assertRaises(ValueError): replay.export_case(trace, 1)

    def test_large_batch_is_not_truncated(self):
        trace = fixture()
        record = trace["rtp_forward_metadata"]["forwards"][0]
        record.update(logical_sequences=257, q_lens=[1]*257, prefix_lens=list(range(257)),
                      kv_lens=list(range(1, 258)), total_q_tokens=257)
        self.assertEqual(len(replay.export_case(trace, 1)["requests"]), 257)


if __name__ == "__main__":
    unittest.main()
