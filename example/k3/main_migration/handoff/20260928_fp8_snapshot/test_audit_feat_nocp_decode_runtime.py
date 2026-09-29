"""Checks the task-local no-CP verifier against archived Decode evidence."""

import gzip
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "evidence/kmerge-20260929"
AUDITOR = HERE / "audit_feat_nocp_decode_runtime.py"


class NoCpDecodeRuntimeAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = tempfile.TemporaryDirectory(prefix="k3-nocp-audit-test-")
        cls.root = Path(cls.work.name)
        with tarfile.open(EVIDENCE / "feat-nocp-r11-decode-logs.tar.gz", "r:gz") as archive:
            env_file = archive.extractfile("decode/service.env")
            assert env_file is not None
            cls.env_text = env_file.read().decode()
        cls.engine_text = gzip.decompress(
            (EVIDENCE / "feat-nocp-r11-decode-engine.log.gz").read_bytes()
        ).decode(errors="replace")

    @classmethod
    def tearDownClass(cls):
        cls.work.cleanup()

    def run_audit(self, env_text=None, engine_text=None):
        env = self.root / "service.env"
        engine = self.root / "engine.log"
        report = self.root / "report.json"
        env.write_text(self.env_text if env_text is None else env_text)
        engine.write_text(self.engine_text if engine_text is None else engine_text)
        result = subprocess.run(
            [
                "python3", str(AUDITOR), "--service-env", str(env),
                "--engine-log", str(engine), "--tp-size", "8",
                "--proposal-tokens", "3", "--output", str(report),
            ],
            capture_output=True, text=True, check=False,
        )
        return result, json.loads(report.read_text()) if report.exists() else None

    def test_real_r11_nocp_flow_has_all_rank_capture_and_request_mtp_path(self):
        result, report = self.run_audit()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["graph_buckets"], [2, 4, 8])
        for bucket in ("2", "4", "8"):
            self.assertEqual(report["graph_capture_ranks"][bucket], list(range(8)))
        self.assertEqual(report["mtp_device_input_ranks"], list(range(8)))
        self.assertFalse(report["graph_replay_proven"])

    def test_cp8_environment_cannot_pass_as_nocp(self):
        env = self.env_text.replace("PREFILL_CP_SIZE=1", "PREFILL_CP_SIZE=8")
        result, report = self.run_audit(env_text=env)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("PREFILL_CP_SIZE", " ".join(report["errors"]))

    def test_missing_rank_capture_cannot_pass(self):
        engine = "\n".join(
            line for line in self.engine_text.splitlines()
            if not ("[RANK 7]" in line and "captured batch size" in line)
        )
        result, report = self.run_audit(engine_text=engine)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("rank 7", " ".join(report["errors"]))

    def test_no_request_time_mtp_path_cannot_pass(self):
        engine = "\n".join(
            line for line in self.engine_text.splitlines()
            if "[mtp-device-input] RTP_LLM_DEVICE_INPUT=1 -> enabled=1" not in line
        )
        result, report = self.run_audit(engine_text=engine)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("MTP", " ".join(report["errors"]))


if __name__ == "__main__":
    unittest.main()
