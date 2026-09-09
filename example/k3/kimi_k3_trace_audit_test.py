"""Reject corrupt or incomplete persisted traces before numerical comparison."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


HERE = Path(__file__).resolve().parent
audit = load("k3_audit_test", HERE / "kimi_k3_trace_audit.py")
recorder = load(
    "k3_audit_recorder_test", HERE.parents[1] / "rtp_llm/utils/k3_tensor_trace.py"
)


class TraceAuditTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.directory = Path(temp.name) / "trace"
        trace = recorder.TensorTrace(
            self.directory, identity={"rank": 0}, max_pending_bytes=8
        )
        trace.begin({"step": 0})
        for layer in range(3):
            trace.record(f"layer.{layer}", torch.full((2,), float(layer)))
        trace.end()
        trace.close()

    def test_valid_fragments_have_one_observation_and_preserve_all_outputs(self):
        result = audit.audit_recorder(self.directory)
        self.assertEqual(result["fragment_count"], 3)
        self.assertEqual(result["observation_count"], 1)
        self.assertEqual(result["tensor_bytes"], 24)
        self.assertEqual(set(result["outputs"]), {f"layer.{i}" for i in range(3)})
        self.assertTrue(result["integrity_verified"])
        self.assertFalse(result["coverage_verified"])

    def test_corrupt_tensor_file_is_rejected_before_loading(self):
        path = self.directory / "frame-00000000.pt"
        data = bytearray(path.read_bytes())
        data[len(data) // 2] ^= 1
        path.write_bytes(data)
        with self.assertRaisesRegex(audit.AuditError, "digest mismatch"):
            audit.audit_recorder(self.directory)

    def test_missing_final_fragment_is_rejected_even_with_consistent_index(self):
        (self.directory / "frame-00000002.pt").unlink()
        index = self.directory / "index.jsonl"
        index.write_text("\n".join(index.read_text().splitlines()[:2]) + "\n")
        (self.directory / "recorder_closed.json").write_text(
            json.dumps({"frames_written": 2})
        )
        with self.assertRaisesRegex(audit.AuditError, "missing its final fragment"):
            audit.audit_recorder(self.directory)

    def test_live_recorder_requires_explicit_opt_in_and_is_not_closed(self):
        (self.directory / "recorder_closed.json").unlink()
        with self.assertRaisesRegex(audit.AuditError, "no shutdown flush marker"):
            audit.audit_recorder(self.directory)
        result = audit.audit_recorder(self.directory, require_closed=False)
        self.assertFalse(result["recorder_closed"])
        self.assertFalse(result["coverage_verified"])


if __name__ == "__main__":
    unittest.main()
