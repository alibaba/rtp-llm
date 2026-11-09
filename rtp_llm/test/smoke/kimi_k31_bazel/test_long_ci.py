"""Long CI must dispatch the traditional runner and both raw/native audits."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rtp_llm.test.smoke.kimi_k3 import k31_long_prefix_smoke, k31_two_host_smoke
from rtp_llm.test.smoke.kimi_k31_bazel import k31_ci


class LongCiTest(unittest.TestCase):
    def worker(self, root):
        worker = k31_ci.K31Worker(None)
        self.addCleanup(worker.locks.close)
        worker.role = "prefill"
        worker.task = root / "k31-pd-longtest" / "prefill"
        worker.task.mkdir(parents=True)
        base = dict(run_id="base", external_revision="a" * 40,
                    prefill=dict(ip="192.0.2.1", port=27188, checkpoint="/target",
                                 draft_checkpoint="/draft", test_block_num=96),
                    decode=dict(ip="192.0.2.2", port=28188, test_block_num=192))
        worker.config = k31_ci.smoke_config(base)
        worker._request_phase = "long"
        worker.host = worker.config["prefill"]
        return worker

    def test_long_phase_uses_original_runner_and_shared_dp2_topology(self):
        with tempfile.TemporaryDirectory() as tmp:
            worker = self.worker(Path(tmp))
            with (patch.object(worker, "status"),
                  patch.object(k31_ci, "runtime_geometry", return_value={}) as geometry):
                command = worker.request_command()
            self.assertEqual(Path(command[1]).name, "k31_long_prefix_smoke.py")
            self.assertEqual(command[2], "run")
            self.assertEqual(command.count("--decode-role-addr"), 1)
            self.assertIn("192.0.2.2:28188:28189", command)
            self.assertEqual(geometry.call_args.args[-1], "a" * 40)
            self.assertEqual(worker.config["decode_dp"], 2)
            options = worker.launcher_options()
            self.assertEqual(options[options.index("--linear-pool-blocks") + 1], "128")
            self.assertEqual(
                options[options.index("--prefill-token-capacity") + 1], "65536"
            )
            self.assertEqual(options[options.index("--max-seq-len") + 1], "1052672")
            self.assertEqual(worker.config["decode"]["test_block_num"], 272)
            self.assertNotIn("request_capacity", worker.config["decode"])
            self.assertNotIn("decode_capture_batches", worker.config["decode"])
            worker.host = worker.config["decode"]
            self.assertNotIn("--request-capacity", worker.launcher_options())
            self.assertNotIn("--decode-capture-batches", worker.launcher_options())

    def test_long_pass_requires_shared_raw_verification_and_native_route(self):
        with tempfile.TemporaryDirectory() as tmp:
            worker = self.worker(Path(tmp))
            (worker.task / "requests").mkdir()
            proof = dict(passed=True, cases=dict(seed={}, continuation={}))
            geometry = dict(page_size=1024, checkpoint_span=8192, reuse_cache_tail_blocks=5)
            (worker.task / "geometry.json").write_text(json.dumps(geometry))
            inputs = ({0: []}, {0: []}, worker.task / "engine.log")
            with (
                patch.object(
                    k31_ci.Worker, "requests", return_value={"requests_passed": True}
                ),
                patch.object(worker, "execution_inputs", return_value=inputs),
                patch.object(worker, "long_phase_events", return_value=inputs[:2]),
                patch.object(
                    k31_long_prefix_smoke, "verify", return_value=proof
                ) as raw,
                patch.object(
                    k31_two_host_smoke, "audit_long_execution", return_value=proof
                ) as native,
            ):
                self.assertTrue(worker.run_request_phase({})["requests_passed"])
            raw.assert_called_once_with(worker.task / "requests/long-prefix", worker.task / "geometry.json")
            native.assert_called_once_with(
                proof, inputs[0], inputs[1], geometry, decode_tp=4, max_chunk_tokens=65536,
            )
            verdict = json.loads(
                (worker.task / "requests/long-execution-audit.json").read_text()
            )
            self.assertEqual(verdict["counts"], dict(formal=1, preparation=1))

    def test_long_raw_failure_cannot_be_reported_as_passing(self):
        with tempfile.TemporaryDirectory() as tmp:
            worker = self.worker(Path(tmp))
            with (patch.object(k31_ci.Worker, "requests", return_value={"requests_passed": True}),
                  patch.object(worker, "execution_inputs", return_value=({}, {}, None)),
                  patch.object(k31_long_prefix_smoke, "verify", side_effect=ValueError("wrong prefix")),
                  self.assertRaisesRegex(ValueError, "wrong prefix")):
                worker.run_request_phase({})
            self.assertFalse((worker.task / "requests/execution-audit.json").exists())


if __name__ == "__main__":
    unittest.main()
