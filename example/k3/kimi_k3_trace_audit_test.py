"""Reject corrupt or incomplete persisted traces before numerical comparison."""

import importlib.util
import json
import sys
import tempfile
import unittest
import zipfile
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
native_buffers = load(
    "k3_native_trace_buffers_test", HERE / "native_trace/k3_trace_buffers.py"
)


class NativeTraceBufferTest(unittest.TestCase):
    def test_unwritten_routes_are_masked_and_snapshot_survives_buffer_reuse(self):
        buffers = native_buffers.K3TraceBuffers(2, 3, 2, 128, "cpu")
        buffers.buffer.fill_(255)
        buffers.reset()
        self.assertEqual(buffers.overflow.item(), 0)
        for name, value in buffers.tensors.items():
            value[1, 2, 1].fill_(7 if name == "expert_ids" else 3)
        snapshot = buffers.snapshot()
        buffers.buffer.zero_()
        expected_valid = torch.zeros((2, 3, 2), dtype=torch.bool)
        expected_valid[1, 2, 1] = True
        torch.testing.assert_close(snapshot["valid"], expected_valid)
        self.assertEqual(snapshot["expert_ids"][1, 2, 1], 7)
        self.assertTrue(torch.all(snapshot["expert_ids"][~expected_valid] == -1))
        for name, value in snapshot.items():
            if name in {"valid", "expert_ids"}:
                continue
            self.assertTrue(torch.all(value[1, 2, 1] == 3))
            self.assertEqual(torch.count_nonzero(value[~expected_valid]), 0)


class TraceAuditTest(unittest.TestCase):
    compression = "none"

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.directory = Path(temp.name) / "trace"
        trace = recorder.TensorTrace(
            self.directory,
            identity={"rank": 0},
            max_pending_bytes=8,
            compression=self.compression,
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


class CompressedTraceAuditTest(TraceAuditTest):
    compression = "deflate"

    def test_compressed_storage_preserves_raw_bits_through_auto_reader(self):
        directory = self.directory.parent / "native-outputs"
        values = {
            "fc1": torch.tensor([0x7FC00001, -2147483648], dtype=torch.int32).view(
                torch.float32
            ),
            "bf16": torch.arange(256, dtype=torch.int16).view(torch.bfloat16),
            "fp8": torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn),
            "scale": torch.arange(256, dtype=torch.uint8),
            "inactive": torch.zeros(65536, dtype=torch.float32),
        }
        trace = recorder.TensorTrace(
            directory, identity={"rank": 0}, compression="deflate"
        )
        trace.begin({"step": 0})
        for name, value in values.items():
            trace.record(name, value)
        trace.end()
        trace.close()
        result = audit.audit_recorder(directory)
        self.assertEqual(result["tensor_count"], len(values))
        self.assertLess(result["file_bytes"], result["tensor_bytes"] // 10)
        path = directory / "frame-00000000.pt"
        with zipfile.ZipFile(path) as archive:
            self.assertTrue(
                all(m.compress_type == zipfile.ZIP_DEFLATED for m in archive.infolist())
            )
        frame = audit.load_frame(path)
        for item in frame["tensors"]:
            expected = values[item["name"]]
            actual = item["value"]
            self.assertEqual(actual.dtype, expected.dtype)
            self.assertTrue(
                torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
            )
        self.assertFalse(result["coverage_verified"])


if __name__ == "__main__":
    unittest.main()
