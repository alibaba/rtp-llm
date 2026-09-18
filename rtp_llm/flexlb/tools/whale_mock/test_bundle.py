import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import bundle


class BundleConfigurationTest(unittest.TestCase):
    def launch_to_process_boundary(self, extra):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory) / "runtime"
            env = {
                "RTP_LLM_MOCK_BUNDLE": "1",
                "FETCH_OUTPUT_STREAM": "0",
                "MOCK_BUNDLE_RUN_DIR": str(runtime),
                "POD_IP": "10.0.0.1",
            }
            env.update(extra)
            with patch.dict(os.environ, env, clear=True), patch.object(
                bundle.signal, "signal"
            ), patch.object(
                bundle.subprocess, "Popen", side_effect=RuntimeError("process boundary")
            ) as start:
                with self.assertRaisesRegex(RuntimeError, "process boundary"):
                    bundle.run()
            performance = runtime / "performance.json"
            return start.call_args.args[0], (
                json.loads(performance.read_text()) if performance.exists() else None
            )

    def test_default_performance_remains_file_based(self):
        command, performance = self.launch_to_process_boundary({})
        self.assertIsNone(performance)
        self.assertEqual(command[command.index("--block-size") + 1], "1024")
        self.assertEqual(command[command.index("--auto-fetch") + 1], "true")
        self.assertEqual(command[command.index("--unique-engine-ips") + 1], "true")
        self.assertNotIn("--events-file", command)

    def test_event_log_is_explicit(self):
        command, _ = self.launch_to_process_boundary({"MOCK_EVENT_LOG_ENABLED": "1"})
        self.assertIn("--events-file", command)

    def test_frontend_fetch_controls_engine_continuation(self):
        command, _ = self.launch_to_process_boundary({"FETCH_OUTPUT_STREAM": "1"})
        self.assertEqual(command[command.index("--auto-fetch") + 1], "false")
        self.assertEqual(command[command.index("--unique-engine-ips") + 1], "false")

    def test_invalid_fetch_mode_cannot_launch(self):
        with self.assertRaisesRegex(ValueError, "FETCH_OUTPUT_STREAM must be 0 or 1"):
            self.launch_to_process_boundary({"FETCH_OUTPUT_STREAM": "invalid"})

    def test_glm_config_and_eos_are_written_without_losing_coefficients(self):
        profile = {
            "bundle_overrides": {
                "block_size": 64, "prefill_block_size": 512,
                "decode_block_size": 64, "prefill": 3, "decode": 4,
                "decode_max_concurrency": 8, "decode_kv_pool_blocks": 128,
            },
            "performance": {
                "block_size": 64, "prefill": {"scale": 1},
                "decode": {"step_base_ms": 20, "eos": {"enabled": True}},
            },
        }
        command, performance = self.launch_to_process_boundary(
            {
                "MOCK_BUNDLE_OVERRIDES_YAML": json.dumps(profile["bundle_overrides"]),
                "MOCK_PERFORMANCE_CONFIG_JSON": json.dumps(profile["performance"]),
                "MOCK_EOS_CONFIG_JSON": '{"enabled": false}',
            }
        )
        self.assertEqual(command[command.index("--block-size") + 1], "64")
        self.assertEqual(command[command.index("--prefill-block-size") + 1], "512")
        self.assertEqual(command[command.index("--decode-block-size") + 1], "64")
        self.assertEqual(command[command.index("--n-prefill") + 1], "3")
        self.assertEqual(command[command.index("--n-decode") + 1], "4")
        self.assertEqual(command[command.index("--decode-max-concurrency") + 1], "8")
        self.assertEqual(command[command.index("--decode-kv-pool-blocks") + 1], "128")
        self.assertEqual(
            performance["decode"]["step_base_ms"],
            profile["performance"]["decode"]["step_base_ms"],
        )
        self.assertEqual(performance["decode"]["eos"], {"enabled": False})

    def test_invalid_capacity_cannot_launch(self):
        for value in [0, -1, True, "64"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.launch_to_process_boundary(
                    {"MOCK_BUNDLE_OVERRIDES_YAML": json.dumps({"block_size": value})}
                )

    def test_block_override_alone_cannot_disagree_with_default_performance(self):
        with self.assertRaisesRegex(ValueError, "block_size must match"):
            self.launch_to_process_boundary(
                {"MOCK_BUNDLE_OVERRIDES_YAML": '{"block_size":64}'}
            )

    def test_mismatched_performance_block_size_cannot_launch(self):
        with self.assertRaisesRegex(ValueError, "block_size must match"):
            self.launch_to_process_boundary(
                {"MOCK_PERFORMANCE_CONFIG_JSON": '{"block_size":64}'}
            )


if __name__ == "__main__":
    unittest.main()
