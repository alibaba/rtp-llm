"""CI must use the traditional smoke's requests and strict acceptance checks."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rtp_llm.test.smoke.kimi_k3 import audit_k31_full_smoke as audit
from rtp_llm.test.smoke.kimi_k3 import server_launcher as shared_launcher
from rtp_llm.test.smoke.kimi_k3 import worker as shared_worker
from rtp_llm.test.smoke.kimi_k3 import k31_server_launcher as server_launcher
from rtp_llm.test.smoke.kimi_k31_bazel import worker


class SharedSmokeTest(unittest.TestCase):
    def test_ci_exports_the_same_configuration_and_worker_as_traditional_smoke(self):
        from rtp_llm.test.smoke.kimi_k31_bazel import k31_ci
        from rtp_llm.test.smoke.kimi_k3 import k31_smoke

        self.assertIs(k31_ci.K31Worker, k31_smoke.K31Worker)
        self.assertIs(k31_ci.smoke_config, k31_smoke.smoke_config)

    def test_split_large_cohort_still_fails_the_shared_strict_gate(self):
        stage = dict(name="orthogonal_decode_00_batch_63",
                     case_names=[str(i) for i in range(63)])
        ids = {str(i): [i] for i in range(63)}
        events = {rank: [] for rank in range(8)}
        with self.assertRaisesRegex(ValueError, "same-step complete"):
            audit.large_batch_evidence([stage], ids, events, 4)

    def test_ci_uses_the_shared_worker_lifecycle(self):
        self.assertIs(worker.Worker.requests, shared_worker.Worker.requests)
        self.assertIs(worker.Worker.close, shared_worker.Worker.close)
        self.assertIs(worker.Worker.start, shared_worker.Worker.start)

    def test_ci_server_does_not_replace_model_or_kernel_configuration(self):
        args = SimpleNamespace(model_type="kimi_k31", core_dump_limit_mib=64)
        command = ["server", "--tp_size", "8", "--dp_size", "1"]
        with (patch.object(shared_launcher, "launch_config",
                           return_value=({"ACT_TYPE": "BF16"}, command)) as configure,
              patch.object(server_launcher, "configure_core_dump_limit", return_value={})):
            environment, actual = server_launcher.launch_config(args)
        configure.assert_called_once_with(args)
        self.assertIs(actual, command)
        self.assertEqual(environment["ACT_TYPE"], "BF16")


if __name__ == "__main__":
    unittest.main()
