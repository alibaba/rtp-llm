"""Portable publication outcome policy contracts."""

import json
import tempfile
import unittest
from pathlib import Path

from rtp_llm.test.perf_test.errors import PublisherError
from rtp_llm.test.perf_test.publish.policy import ConfiguredPublisher, run_publishers


class PublisherPolicyTest(unittest.TestCase):
    def _run_directory(self, root):
        (root / "results" / "cases").mkdir(parents=True)
        manifest = {
            "schema_version": 1,
            "run_id": "run-001",
            "state": "completed",
        }
        summary = {
            "schema_version": 1,
            "run_id": "run-001",
            "status": "completed",
            "planned_cases": 1,
            "successful_cases": 1,
            "failed_cases": 0,
        }
        case = {"case_id": "prefill-case", "status": "ok"}
        result = {"run_id": "run-001", "status": "completed", "cases": [case]}
        (root / "manifest.json").write_text(json.dumps(manifest))
        (root / "results" / "summary.json").write_text(json.dumps(summary))
        (root / "results" / "result.json").write_text(json.dumps(result))
        (root / "results" / "cases" / "prefill-case.json").write_text(json.dumps(case))

    def _measurement_bytes(self, root):
        return {
            path.relative_to(root).as_posix(): path.read_bytes()
            for path in [
                root / "results" / "result.json",
                root / "results" / "cases" / "prefill-case.json",
            ]
        }

    def test_optional_failure_preserves_measurement_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._run_directory(root)
            before = self._measurement_bytes(root)

            publication = run_publishers(
                root,
                [
                    ConfiguredPublisher(
                        name="oss",
                        required=False,
                        publish=lambda: (_ for _ in ()).throw(RuntimeError("offline")),
                    ),
                    ConfiguredPublisher(
                        name="odps",
                        required=False,
                        publish=lambda: {"status": "succeeded", "destination": "table"},
                    ),
                ],
            )

            self.assertEqual("failed_optional", publication["status"])
            self.assertFalse(publication["required_failure"])
            self.assertEqual(before, self._measurement_bytes(root))
            manifest = json.loads((root / "manifest.json").read_text())
            summary = json.loads((root / "results" / "summary.json").read_text())
            self.assertEqual("completed", manifest["state"])
            self.assertEqual("completed", summary["status"])
            self.assertEqual(publication, manifest["publication"])
            self.assertEqual(publication, summary["publication"])

    def test_required_failure_is_recorded_then_raises_exit_code_six(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._run_directory(root)
            before = self._measurement_bytes(root)

            with self.assertRaisesRegex(PublisherError, "required publisher") as raised:
                run_publishers(
                    root,
                    [
                        ConfiguredPublisher(
                            name="oss",
                            required=True,
                            publish=lambda: (_ for _ in ()).throw(
                                RuntimeError("upload failed")
                            ),
                        )
                    ],
                )

            self.assertEqual(6, raised.exception.exit_code)
            self.assertEqual(before, self._measurement_bytes(root))
            manifest = json.loads((root / "manifest.json").read_text())
            summary = json.loads((root / "results" / "summary.json").read_text())
            self.assertEqual("failed_required", manifest["publication"]["status"])
            self.assertTrue(summary["publication"]["required_failure"])
            self.assertEqual("failed", summary["publication"]["outcomes"][0]["status"])


if __name__ == "__main__":
    unittest.main()
