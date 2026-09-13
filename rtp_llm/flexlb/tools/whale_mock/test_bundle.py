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

    def test_glm_config_and_eos_are_written_without_losing_coefficients(self):
        profile = json.loads((bundle.ROOT / "glm53-calibration.json").read_text())
        command, performance = self.launch_to_process_boundary(
            {
                "MOCK_BUNDLE_OVERRIDES_YAML": json.dumps(profile["bundle_overrides"]),
                "MOCK_PERFORMANCE_CONFIG_JSON": json.dumps(profile["performance"]),
                "MOCK_EOS_CONFIG_JSON": '{"enabled": false}',
            }
        )
        self.assertEqual(command[command.index("--block-size") + 1], "64")
        self.assertEqual(command[command.index("--n-prefill") + 1], "55")
        self.assertEqual(command[command.index("--n-decode") + 1], "320")
        self.assertEqual(command[command.index("--decode-max-concurrency") + 1], "64")
        self.assertEqual(command[command.index("--decode-kv-pool-blocks") + 1], "46157")
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

    def test_mismatched_performance_block_size_cannot_launch(self):
        with self.assertRaisesRegex(ValueError, "block_size must match"):
            self.launch_to_process_boundary(
                {"MOCK_PERFORMANCE_CONFIG_JSON": '{"block_size":64}'}
            )


if __name__ == "__main__":
    unittest.main()
