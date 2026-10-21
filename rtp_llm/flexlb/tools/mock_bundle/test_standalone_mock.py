import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import standalone_mock


class StandaloneMockTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.env = {
            "RTP_LLM_MOCK_BUNDLE": "1",
            "POD_IP": "10.64.34.208",
            "MOCK_BUNDLE_RUN_DIR": self.directory.name,
            "MOCK_PREFILL_COUNT": "2",
            "MOCK_DECODE_COUNT": "3",
            "MOCK_PREFILL_KV_POOL_BLOCKS": "113282",
            "MOCK_DECODE_KV_POOL_BLOCKS": "113282",
        }

    def test_five_engines_use_one_java_process_and_preserve_capacity(self):
        with patch.dict(os.environ, self.env, clear=True), patch.object(
            standalone_mock.os, "execv"
        ) as execute:
            standalone_mock.run()
        executable, args = execute.call_args.args
        self.assertEqual(executable, "/opt/taobao/java/bin/java")
        self.assertEqual(args.count("-jar"), 1)
        for flag, value in (("--n-prefill", "2"), ("--n-decode", "3"),
                            ("--base-grpc-port", "29130"),
                            ("--unique-engine-ips", "false"),
                            ("--auto-fetch", "false"), ("--kmonitor", "true"),
                            ("--prefill-kv-pool-blocks", "113282"),
                            ("--decode-kv-pool-blocks", "113282")):
            self.assertEqual(args[args.index(flag) + 1], value)
        self.assertNotIn("master.jar", " ".join(args))
        self.assertEqual(json.loads((Path(self.directory.name) / "master-config.json").read_text()), {})

    def test_environment_json_and_quoted_jvm_options(self):
        self.env.update({
            "MOCK_JAVA_OPTS": '-Xmx3g "-Ddescription=mock engine"',
            "FETCH_OUTPUT_STREAM": "0", "MOCK_KMONITOR_ENABLED": "false",
            "FLEXLB_CONFIG": '{"schemaVersion":3}',
            "MOCK_PERFORMANCE_CONFIG_JSON": '{"block_size":1024,"sleep_scale":2}',
        })
        args = standalone_mock.command(self.env)
        self.assertIn("-Ddescription=mock engine", args)
        self.assertEqual(args[args.index("--auto-fetch") + 1], "true")
        self.assertEqual(args[args.index("--kmonitor") + 1], "false")
        self.assertEqual(json.loads((Path(self.directory.name) / "performance.json").read_text())["sleep_scale"], 2)

    def test_invalid_configuration_cannot_exec(self):
        cases = [
            {"RTP_LLM_MOCK_BUNDLE": "0"},
            {"MOCK_PREFILL_COUNT": "0", "MOCK_DECODE_COUNT": "0"},
            {"MOCK_PREFILL_COUNT": "-1"},
            {"MOCK_GRPC_BASE_PORT": "65534"},
            {"MOCK_DECODE_KV_POOL_BLOCKS": "zero"},
            {"POD_IP": "127.0.0.1"}, {"POD_IP": "0.0.0.0"},
            {"FETCH_OUTPUT_STREAM": "maybe"},
            {"MOCK_PERFORMANCE_CONFIG_JSON": "[]"},
            {"MOCK_PERFORMANCE_CONFIG_JSON": '{"block_size":128}'},
            {"FLEXLB_CONFIG": "[]"},
        ]
        for override in cases:
            with self.subTest(override=override), patch.dict(
                os.environ, {**self.env, **override}, clear=True
            ), patch.object(standalone_mock.os, "execv") as execute:
                with self.assertRaises(ValueError):
                    standalone_mock.run()
                execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
