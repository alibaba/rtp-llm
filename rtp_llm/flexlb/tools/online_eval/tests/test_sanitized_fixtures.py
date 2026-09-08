import json
import re
import unittest
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "online_logs"
SENSITIVE_TEXT = re.compile(
    r"authorization|x-dashscope|request_id|traceid|user_id|workspace_id|"
    r"/home/admin|(?:[0-9]{1,3}\.){3}[0-9]{1,3}",
    re.IGNORECASE,
)


class SanitizedFixturesTest(unittest.TestCase):
    def test_access_fixture_is_minimal_and_pseudonymous(self):
        path = DATA_DIR / "sample_access.json"
        content = path.read_text(encoding="utf-8")
        fixture = json.loads(content)

        self.assertTrue(fixture["sanitized"])
        self.assertEqual("sanitized_access_shape", fixture["fixture_type"])
        self.assertEqual(
            "mock-model", fixture["request_controls"]["ds_header_attributes"]["model"]
        )
        self.assertEqual(29_699, len(fixture["input_ids"]))
        self.assertIsNone(SENSITIVE_TEXT.search(content))

    def test_trace_has_relative_time_and_no_request_identity(self):
        records = [
            json.loads(line)
            for line in (DATA_DIR / "trace_30min.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line
        ]

        self.assertEqual(8_332, len(records))
        self.assertEqual(0, min(record["ts"] for record in records))
        self.assertLess(max(record["ts"] for record in records), 1_000_000_000)
        for record in records:
            self.assertNotIn("rid", record)
            self.assertNotIn("request_id", record)
            self.assertRegex(record["pep"], r"^prefill-\d+$")
            self.assertRegex(record["dep"], r"^decode-\d+$")

    def test_arrivals_contain_only_relative_numeric_data(self):
        lines = DATA_DIR / "pod1_arrivals.tsv"
        content = lines.read_text(encoding="utf-8")
        rows = [line for line in content.splitlines() if not line.startswith("#")]

        self.assertEqual(2_413, len(rows))
        self.assertIsNone(SENSITIVE_TEXT.search(content))
        for row in rows:
            timestamp, input_tokens = row.split("\t")
            self.assertGreaterEqual(int(timestamp), 0)
            self.assertGreaterEqual(int(input_tokens), 0)


if __name__ == "__main__":
    unittest.main()
