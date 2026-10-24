import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rtp_llm.models_py.modules.qwen4_exp.test.upstream_fixture import (
    uncovered_fixture,
    upstream_source_path,
)


class UpstreamFixtureTest(unittest.TestCase):
    def test_content_pin_requires_explicit_unverified_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "modeling_qwen4_exp.py"
            source.write_text("# reference fixture\n")
            manifest = root / "manifest.json"
            pin = {
                "source_kind": "content_addressed",
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "commit": None,
                "commit_status": "unverified",
            }
            env = {
                "QWEN4_EXP_REQUIRE_PARITY": "1",
                "QWEN4_EXP_UPSTREAM_SRC": str(source),
                "QWEN4_EXP_UPSTREAM_MANIFEST": str(manifest),
                "QWEN4_EXP_CKPT_DIR": directory,
            }
            with patch.dict(os.environ, env):
                manifest.write_text(json.dumps(pin))
                self.assertEqual(upstream_source_path(), str(source))
                source.write_text("# modified reference\n")
                with self.assertRaisesRegex(RuntimeError, "differs"):
                    upstream_source_path()
                source.write_text("# reference fixture\n")
                for field in ("source_kind", "commit_status"):
                    manifest.write_text(
                        json.dumps({k: v for k, v in pin.items() if k != field})
                    )
                    with self.assertRaisesRegex(RuntimeError, "provenance"):
                        upstream_source_path()
                manifest.write_text(json.dumps({**pin, "source_kind": "git"}))
                with self.assertRaisesRegex(RuntimeError, "provenance"):
                    upstream_source_path()
                manifest.write_text(
                    json.dumps({**pin, "source_kind": "git", "commit": "a" * 40})
                )
                self.assertEqual(upstream_source_path(), str(source))

    def test_required_coverage_defaults_to_vendored_fixture(self):
        # Required qualification without explicit env falls back to the vendored
        # in-repo copy; only the real-weight checkpoint stays an external fixture.
        with patch.dict(os.environ, {"QWEN4_EXP_REQUIRE_PARITY": "1"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "checkpoint fixture"):
                upstream_source_path()
        with patch.dict(
            os.environ,
            {
                "QWEN4_EXP_REQUIRE_PARITY": "1",
                "QWEN4_EXP_CKPT_DIR": tempfile.gettempdir(),
            },
            clear=True,
        ):
            source = upstream_source_path()
            self.assertTrue(Path(source).is_file())
            self.assertTrue(source.endswith("modeling_qwen4_exp.py"))

    def test_missing_download_cannot_skip_required_qualification(self):
        with patch.dict(os.environ, {"QWEN4_EXP_REQUIRE_PARITY": "1"}):
            with self.assertRaisesRegex(RuntimeError, "uncovered.*missing shard"):
                uncovered_fixture("missing shard")
        with patch.dict(os.environ, {"QWEN4_EXP_REQUIRE_PARITY": "0"}):
            with self.assertRaises(unittest.SkipTest):
                uncovered_fixture("missing optional shard")


if __name__ == "__main__":
    unittest.main()
