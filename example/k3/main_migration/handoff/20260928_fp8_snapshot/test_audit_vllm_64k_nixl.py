import pathlib
import tarfile
import tempfile
import unittest

from audit_vllm_64k_nixl import audit


HERE = pathlib.Path(__file__).resolve().parent
EVIDENCE = HERE / "evidence/vllm_3df4_r2"


class VllmNixlAuditTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run_dir = pathlib.Path(self.temp.name)
        with tarfile.open(EVIDENCE / "requests-64k-vllm-3df4-r2.tar.gz") as archive:
            archive.extractall(self.run_dir, filter="data")

    def check(self):
        return audit(
            self.run_dir,
            EVIDENCE / "vllm-pd-3df4-r2-traces.tar.gz",
            EVIDENCE / "vllm-pd-3df4-r2-consumer-kv-metrics.txt",
            HERE / "timeline-input-64k",
        )

    def test_recorded_pd_trace_passes_with_http_warmup_exception(self):
        report = self.check()
        self.assertTrue(report["passed"], report["errors"])
        self.assertFalse(report["http_warmup_stable"])
        self.assertEqual(report["prefill_tokens_per_rank"], 65536)
        self.assertEqual(report["nixl_transfer_count"], 136)

    def test_changed_response_bytes_fail_digest_check(self):
        response = self.run_dir / "requests/profiled-01.json"
        response.write_bytes(response.read_bytes() + b" ")
        report = self.check()
        self.assertFalse(report["passed"])
        self.assertTrue(any("profiled-01" in error and "digest" in error
                            for error in report["errors"]))


if __name__ == "__main__":
    unittest.main()
