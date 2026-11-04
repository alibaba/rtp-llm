"""Capture assembly rejects mixed builds, missing rows and altered histories."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from rtp_llm.tools.v41.capture_logits import assemble, digest, load_fixture


class CaptureTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.fixture = self.root / "fixture.json"
        self.fixture.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "vocab_size": 8,
                    "cases": {
                        "case": {"input_ids": [2, 3], "continuation_tokens": [4, 5]}
                    },
                }
            )
        )
        self.requests = self.root / "requests"
        (self.requests / "case").mkdir(parents=True)
        self.identity = dict(
            snapshot_id="snapshot",
            public_commit="public",
            internal_commit="internal",
            source_sha256="source",
        )
        report = dict(
            self.identity,
            case_id="case",
            request_id=123,
            finished=True,
            output_ids=[4, 5],
            fixture_sha256=digest(self.fixture),
            single_engine=False,
        )
        (self.requests / "case/summary.json").write_text(json.dumps(report))
        self.captures = []
        for row, role in enumerate(("PREFILL", "DECODE")):
            directory = self.root / role / "capture"
            data = directory / "case-request-123"
            data.mkdir(parents=True)
            (directory.parent / "owner.json").write_text(
                json.dumps(dict(self.identity, role=role.lower()))
            )
            (directory.parent / "processes.json").write_text(
                json.dumps({"99": {"start_ticks": 1}})
            )
            metadata = dict(
                schema_version=1,
                dtype="float32",
                vocab_size=8,
                row=row,
                context_length=2 + row,
                target_token=4 + row,
                request_id=123,
                case_id="case",
                source="native_lm_head_before_sampling",
                pid=99,
                role=role,
            )
            (data / f"row-{row}.json").write_text(json.dumps(metadata))
            np.arange(8, dtype=np.float32).tofile(data / f"row-{row}.f32")
            self.captures.append(directory)

    def run_assembly(self):
        assemble(
            SimpleNamespace(
                fixture=self.fixture,
                requests=self.requests,
                capture=self.captures,
                output=self.root / "candidate.npz",
            )
        )

    def test_complete_rows_preserve_raw_values_and_frozen_ids(self):
        self.run_assembly()
        with np.load(self.root / "candidate.npz") as arrays:
            np.testing.assert_array_equal(
                arrays["case__logits_fp32"], np.tile(np.arange(8), (2, 1))
            )
            np.testing.assert_array_equal(arrays["case__continuation_tokens"], [4, 5])
            np.testing.assert_array_equal(arrays["case__context_lengths"], [2, 3])

    def test_missing_row_and_wrong_context_cannot_be_assembled(self):
        path = self.captures[1] / "case-request-123/row-1.json"
        metadata = json.loads(path.read_text())
        metadata["context_length"] = 9
        path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "metadata mismatch"):
            self.run_assembly()
        path.unlink()
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.run_assembly()

    def test_owner_build_and_pid_must_match_the_actual_request(self):
        path = self.captures[1].parent / "owner.json"
        metadata = json.loads(path.read_text())
        metadata["source_sha256"] = "other-source"
        path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "owner identity"):
            self.run_assembly()
        metadata["source_sha256"] = self.identity["source_sha256"]
        path.write_text(json.dumps(metadata))
        (self.captures[1].parent / "processes.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "unowned"):
            self.run_assembly()

    def test_fixture_rejects_out_of_vocabulary_ids(self):
        fixture = json.loads(self.fixture.read_text())
        fixture["cases"]["case"]["continuation_tokens"][1] = 8
        self.fixture.write_text(json.dumps(fixture))
        with self.assertRaisesRegex(ValueError, "token IDs"):
            load_fixture(self.fixture)


if __name__ == "__main__":
    unittest.main()
