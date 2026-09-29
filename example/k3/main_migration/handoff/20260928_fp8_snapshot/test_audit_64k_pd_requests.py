"""Audit fixed 64K HTTP records from both current and legacy RTP PD."""

import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest


ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "evidence/kmerge-20260929"
AUDITOR = ROOT / "audit_64k_pd_requests.py"


class Audit64kPdRequestsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = tempfile.TemporaryDirectory(prefix="k3-64k-audit-")
        cls.root = Path(cls.work.name)
        archives = {
            "legacy": EVIDENCE / "feat-r12-114115/feat-r12-114115-timeline-requests.tar.gz",
            "integrated": EVIDENCE / "pagedmeta-b24-r2-timeline-requests.tar.gz",
        }
        for kind, path in archives.items():
            destination = cls.root / kind
            destination.mkdir()
            with tarfile.open(path, "r:gz") as archive:
                for member in archive:
                    if not member.isfile():
                        continue
                    parts = Path(member.name).parts
                    suffix = parts if kind == "legacy" else parts[1:]
                    if not suffix or any(part in ("", "..") for part in suffix):
                        raise ValueError("invalid archive member")
                    target = destination.joinpath(*suffix)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    source = archive.extractfile(member)
                    assert source is not None
                    target.write_bytes(source.read())

    @classmethod
    def tearDownClass(cls):
        cls.work.cleanup()

    def run_audit(self, kind, ip, port):
        directory = self.root / kind
        result = subprocess.run(
            [sys.executable, str(AUDITOR), str(directory),
             "--decode-ip", ip, "--decode-port", str(port)],
            capture_output=True, text=True, check=False,
        )
        path = directory / "independent-request-audit.json"
        report = json.loads(path.read_text()) if path.exists() else None
        return result, report

    def test_legacy_feat_64k_requests_pass_without_inventing_mtp_counters(self):
        result, report = self.run_audit("legacy", "11.163.39.115", 26600)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(report["passed"])
        self.assertEqual((report["warmup_count"], report["profiled_count"]), (11, 16))
        self.assertEqual(report["replacement_characters_observed"], 0)
        self.assertFalse(report["mtp_execution_claim"])
        self.assertFalse(report["decode_handoff_length_claim"])

    def test_integrated_records_keep_strict_draft_and_handoff_checks(self):
        result, report = self.run_audit("integrated", "11.163.39.112", 28200)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(report["passed"])
        self.assertTrue(report["mtp_execution_claim"])
        self.assertTrue(report["decode_handoff_length_claim"])

    def test_legacy_route_mismatch_fails(self):
        result, report = self.run_audit("legacy", "11.163.39.112", 26600)
        self.assertEqual(result.returncode, 1)
        self.assertFalse(report["passed"])
        self.assertTrue(any("Decode route differs" in error for error in report["errors"]))


if __name__ == "__main__":
    unittest.main()
