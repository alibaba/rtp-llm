"""Causal-prefix checks reject candidate omission and accidental future history."""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from kimi_k3_trace_audit import AuditError
from kimi_k3_verify_history import audit_history, check_history

RECORDER_PATH = Path(__file__).resolve().parents[2] / "rtp_llm/utils/k3_tensor_trace.py"
spec = importlib.util.spec_from_file_location(
    "verify_history_recorder_test", RECORDER_PATH
)
recorder = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = recorder
spec.loader.exec_module(recorder)


class VerifyHistoryTest(unittest.TestCase):
    def fixture(self):
        # Nontrivial duplicate in the proposal: position 2 must contain two 7s.
        streams = [
            (
                {
                    "stream_id": 19,
                    "first_row": 0,
                    "row_count": 4,
                    "committed_length": 2,
                },
                torch.tensor([[4, 5]]),
            )
        ]
        tokens = torch.tensor([5, 7, 7, 9])
        history = torch.tensor(
            [
                [4, 5, -1, -1, -1, -1],
                [4, 5, 7, -1, -1, -1],
                [4, 5, 7, 7, -1, -1],
                [4, 5, 7, 7, 9, -1],
            ]
        )
        return history, torch.tensor([2, 3, 4, 5]), tokens, streams

    def test_correct_prefixes_and_unused_padding(self):
        args = self.fixture()
        result = check_history(*args, 3)
        self.assertTrue(result["causal_history_matches"])
        self.assertEqual(result["rows_checked"], 4)

    def test_committed_only_rows_with_maximum_lengths_fail(self):
        history, lengths, tokens, streams = self.fixture()
        history[:, 2:] = 0  # Plausible allocator residue, also a valid vocabulary ID.
        lengths[:] = 5
        result = check_history(history, lengths, tokens, streams, 3)
        self.assertFalse(result["causal_history_matches"])
        self.assertEqual(len(result["failures"]), 4)

    def test_future_candidate_is_not_valid_history_for_earlier_position(self):
        history, lengths, tokens, streams = self.fixture()
        history[1, 3] = 9
        lengths[1] = 4
        result = check_history(history, lengths, tokens, streams, 3)
        self.assertEqual([f["spec_position"] for f in result["failures"]], [1])

    def test_same_token_set_with_wrong_duplicate_counts_fails(self):
        history, lengths, tokens, streams = self.fixture()
        history[3, :5] = torch.tensor([4, 5, 7, 9, 9])
        result = check_history(history, lengths, tokens, streams, 3)
        self.assertEqual(result["failures"][0]["first_difference"], 3)

    def test_unproven_input_layout_is_rejected(self):
        history, lengths, tokens, streams = self.fixture()
        tokens[0] = 11
        with self.assertRaisesRegex(AuditError, "last committed token"):
            check_history(history, lengths, tokens, streams, 3)

    def test_persisted_fragmented_events_join_within_their_scope(self):
        history, lengths, tokens, streams = self.fixture()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "trace"
            trace = recorder.TensorTrace(
                root, identity={"rank": 0}, max_pending_bytes=224
            )
            scopes = [{"name": "mtp.gather_verify_sampler", "id": 7}]
            trace.begin(
                {
                    "event": "mtp.verify_history_rows",
                    "scopes": scopes,
                    "details": streams[0][0],
                }
            )
            trace.record("committed_token_ids", streams[0][1])
            trace.end()
            trace.begin(
                {
                    "event": "mtp.verify_sampler_gathered",
                    "scopes": scopes,
                    "details": {"propose_step": 3},
                }
            )
            trace.record("history", history)
            trace.record("sequence_lengths", lengths)
            trace.record("model_input_tokens", tokens)
            trace.end()
            trace.close()
            report = audit_history(root)
            self.assertTrue(report["causal_history_matches"])
            self.assertEqual(report["batches"][0]["rows_checked"], 4)
            self.assertFalse(report["coverage_verified"])
            (root / "recorder_closed.json").unlink()
            with self.assertRaisesRegex(AuditError, "shutdown flush"):
                audit_history(root)


if __name__ == "__main__":
    unittest.main()
