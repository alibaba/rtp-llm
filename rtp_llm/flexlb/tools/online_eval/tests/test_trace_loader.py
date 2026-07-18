from __future__ import annotations

import sys
import unittest
from pathlib import Path

TOOL_DIR = Path(__file__).resolve().parents[1]
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from online_eval.rt_model import INT64_MAX, INT64_MIN
from online_eval.trace_loader import parse_record


class TraceLoaderTest(unittest.TestCase):
    def test_preserves_string_ids_and_normalizes_int64_fields(self) -> None:
        raw = {
            "request_id": "frontend-request-id",
            "trace_id": "trace-id-string",
            "request_id_int": 2**63 + 7,
            "ts": 1,
            "il": 2048,
            "ol": 8,
            "bh": [2**63 + 5, 42],
        }

        req = parse_record(
            raw, zero_output_policy="skip", include_token_ids=False, block_size=1024
        )

        self.assertEqual("frontend-request-id", req.source_rid)
        self.assertEqual("trace-id-string", req.trace_id)
        self.assertTrue(INT64_MIN <= req.request_id <= INT64_MAX)
        self.assertEqual(-(2**63) + 7, req.request_id)
        self.assertEqual([-(2**63) + 5, 42], req.block_keys)


if __name__ == "__main__":
    unittest.main()
