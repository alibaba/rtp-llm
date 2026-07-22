import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from rtp_llm.models_py.utils import prefill_trace


class PrefillTraceTest(unittest.TestCase):
    def setUp(self) -> None:
        prefill_trace._reset_for_test()

    def _tempdir(self):
        return tempfile.TemporaryDirectory(
            dir=os.environ.get("RTP_LLM_TEST_TMPDIR") or None
        )

    def test_records_metadata_and_only_short_last_hidden(self) -> None:
        with self._tempdir() as trace_dir, patch.dict(
            os.environ,
            {
                "RTP_LLM_PREFILL_TRACE": "1",
                "RTP_LLM_PREFILL_TRACE_DIR": trace_dir,
                "RTP_LLM_PREFILL_TRACE_SHORT_MAX_INPUT_LEN": "8",
                "RTP_LLM_PREFILL_TRACE_LOG": "0",
            },
            clear=False,
        ):
            request_ids = torch.tensor([101, 202, 303], dtype=torch.int64)
            executed = torch.tensor([3, 4, 2], dtype=torch.int32)
            reused = torch.tensor([2, 10, 1], dtype=torch.int32)
            hidden = torch.arange(9 * 4, dtype=torch.float32).reshape(9, 4)
            prefill_trace.record_prefill(
                hidden, request_ids, executed, reused, False, 7, 0
            )

            metadata_files = list(Path(trace_dir).glob("*.jsonl"))
            self.assertEqual(len(metadata_files), 1)
            record = json.loads(metadata_files[0].read_text().strip())
            self.assertEqual(record["batch_exec_tokens"], 9)
            self.assertEqual(record["batch_reuse_tokens"], 13)
            self.assertEqual(
                [item["request_id"] for item in record["requests"]],
                [101, 202, 303],
            )
            self.assertEqual(
                [item["input_len"] for item in record["requests"]], [5, 14, 3]
            )
            self.assertIn("last_hidden_sha256", record["requests"][0])
            self.assertNotIn("last_hidden_sha256", record["requests"][1])
            self.assertIn("last_hidden_sha256", record["requests"][2])

            payload = torch.load(Path(trace_dir) / record["hidden_file"])
            torch.testing.assert_close(payload["request_ids"], torch.tensor([101, 303]))
            torch.testing.assert_close(
                payload["last_hidden_states"],
                hidden.index_select(0, torch.tensor([2, 8])),
            )

    def test_cp_gathered_last_rows_and_batch_limit(self) -> None:
        with self._tempdir() as trace_dir, patch.dict(
            os.environ,
            {
                "RTP_LLM_PREFILL_TRACE": "1",
                "RTP_LLM_PREFILL_TRACE_DIR": trace_dir,
                "RTP_LLM_PREFILL_TRACE_SHORT_MAX_INPUT_LEN": "8",
                "RTP_LLM_PREFILL_TRACE_MAX_BATCHES": "1",
                "RTP_LLM_PREFILL_TRACE_LOG": "0",
            },
            clear=False,
        ):
            hidden = torch.arange(3 * 2, dtype=torch.float32).reshape(3, 2)
            args = (
                hidden,
                torch.tensor([11, 22, 33]),
                torch.tensor([3, 4, 2]),
                torch.tensor([2, 10, 1]),
                True,
                9,
                0,
            )
            prefill_trace.record_prefill(*args)
            prefill_trace.record_prefill(*args)

            records = []
            for path in Path(trace_dir).glob("*.jsonl"):
                records.extend(
                    json.loads(line) for line in path.read_text().splitlines()
                )
            self.assertEqual(len(records), 1)
            payload = torch.load(Path(trace_dir) / records[0]["hidden_file"])
            torch.testing.assert_close(
                payload["last_hidden_states"],
                hidden.index_select(0, torch.tensor([0, 2])),
            )

    def test_disabled_has_no_side_effect(self) -> None:
        with self._tempdir() as parent:
            trace_dir = Path(parent) / "must_not_exist"
            with patch.dict(
                os.environ,
                {
                    "RTP_LLM_PREFILL_TRACE": "0",
                    "RTP_LLM_PREFILL_TRACE_DIR": str(trace_dir),
                },
                clear=False,
            ):
                prefill_trace.record_prefill(
                    torch.ones((1, 2)),
                    torch.tensor([1]),
                    torch.tensor([1]),
                    torch.tensor([0]),
                    True,
                    0,
                    0,
                )
            self.assertFalse(trace_dir.exists())


if __name__ == "__main__":
    unittest.main()
