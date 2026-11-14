"""Portable regression and publication contracts used by internal CI."""

import json
import tempfile
import unittest
from pathlib import Path


class PublicPolicyExtractionTest(unittest.TestCase):
    def test_public_baseline_schema_is_available(self):
        try:
            from rtp_llm.test.perf_test.regression.baseline import parse_baseline
        except ModuleNotFoundError:
            self.fail("public baseline parser is missing")

        entries = parse_baseline(
            {
                "schema_version": 1,
                "entries": [
                    {
                        "case_key": "decode-" + "a" * 24,
                        "metric": "latency_ms.total",
                        "value": 10.0,
                        "direction": "lower_is_better",
                        "threshold": 0.05,
                    }
                ],
            }
        )
        self.assertEqual(1, len(entries))
        self.assertEqual(10.0, entries[0].value)

    def test_public_golden_loader_validates_json_object(self):
        try:
            from rtp_llm.test.perf_test.regression.golden import load_result
        except ModuleNotFoundError:
            self.fail("public golden module is missing")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps({"run_id": "run-001"}), encoding="utf-8")
            self.assertEqual({"run_id": "run-001"}, load_result(path))
            path.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "expected an object"):
                load_result(path)

    def test_public_publish_policy_records_optional_failure(self):
        try:
            from rtp_llm.test.perf_test.publish.policy import (
                ConfiguredPublisher,
                run_publishers,
            )
        except ModuleNotFoundError:
            self.fail("public publication policy is missing")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "results").mkdir()
            (root / "manifest.json").write_text('{"state":"completed"}')
            (root / "results/summary.json").write_text('{"status":"completed"}')
            publication = run_publishers(
                root,
                [
                    ConfiguredPublisher(
                        name="archive",
                        required=False,
                        publish=lambda: (_ for _ in ()).throw(RuntimeError("offline")),
                    )
                ],
            )
            self.assertEqual("failed_optional", publication["status"])
            self.assertEqual(
                "completed",
                json.loads((root / "results/summary.json").read_text())["status"],
            )
            self.assertEqual(
                publication,
                json.loads((root / "manifest.json").read_text())["publication"],
            )


if __name__ == "__main__":
    unittest.main()
