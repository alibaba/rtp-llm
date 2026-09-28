"""Exercise the independent feat PD flow audit on recorded HTTP responses."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "evidence/kmerge-20260929/feat-nocp-r11-prefill-logs.tar.gz"
AUDITOR = ROOT / "audit_feat_profile_flow.py"


class AuditFeatFlowTest(unittest.TestCase):
    def run_audit(self, port):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "audit.json"
            run = subprocess.run(
                [sys.executable, str(AUDITOR), "--prefill-tar", str(ARCHIVE),
                 "--output", str(output), "--decode-port", str(port)],
                capture_output=True, text=True, check=False,
            )
            result = json.loads(output.read_text()) if output.exists() else None
        return run, result

    def test_accepts_recorded_no_cp_flow_on_its_actual_decode_port(self):
        run, result = self.run_audit(26600)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertTrue(result["passed"])
        self.assertEqual(result["checked"], 10)
        self.assertEqual(result["replacement_characters_observed"], 0)
        self.assertFalse(result["semantic_answer_claim"])

    def test_rejects_same_responses_if_decode_port_is_wrong(self):
        run, result = self.run_audit(26400)
        self.assertEqual(run.returncode, 1, run.stderr)
        self.assertFalse(result["passed"])
        self.assertEqual(result["checked"], 10)
        self.assertTrue(any("Decode route differs" in error for error in result["errors"]))


if __name__ == "__main__":
    unittest.main()
