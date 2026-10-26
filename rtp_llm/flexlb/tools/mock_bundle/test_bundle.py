import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import bundle


class BundleConfigurationTest(unittest.TestCase):
    def launch_to_process_boundary(self, extra):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory) / "runtime"
            env = {"RTP_LLM_MOCK_BUNDLE": "1", "FETCH_OUTPUT_STREAM": "0", "MOCK_BUNDLE_RUN_DIR": str(runtime), "POD_IP": "10.0.0.1"}
            env.update(extra)
            with (
                patch.dict(os.environ, env, clear=True),
                patch.object(bundle.signal, "signal"),
                patch.object(bundle.subprocess, "Popen", side_effect=RuntimeError("process boundary")) as start,
            ):
                with self.assertRaisesRegex(RuntimeError, "process boundary"):
                    bundle.run()
            performance = runtime / "performance.json"
            return start.call_args.args[0], (json.loads(performance.read_text()) if performance.exists() else None)

    def test_common_counts_override_bundle_defaults(self):
        command, _ = self.launch_to_process_boundary(
            {"MOCK_DEPLOYMENT_MODE": "embedded", "MOCK_PREFILL_COUNT": "2", "MOCK_DECODE_COUNT": "3"}
        )
        self.assertEqual(command[command.index("--mock.n_prefill") + 1], "2")
        self.assertEqual(command[command.index("--mock.n_decode") + 1], "3")
        for override in ({"MOCK_PREFILL_COUNT": "0"}, {"MOCK_PREFILL_COUNT": "2", "MOCK_BUNDLE_OVERRIDES_YAML": "{prefill: 3}"}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                self.launch_to_process_boundary(override)

    def test_vit_discovery_keeps_dynamic_pd_and_pinned_vit(self):
        with tempfile.TemporaryDirectory() as directory:
            source, destination = Path(directory) / "pd.json", Path(directory) / "combined.json"
            source.write_text(json.dumps({"prefill": ["10.0.0.1:1000"], "decode": []}))
            bundle.publish_vit_discovery(source, destination, "10.0.0.1:1200")
            first = destination.stat().st_mtime_ns
            bundle.publish_vit_discovery(source, destination, "10.0.0.1:1200")
            self.assertEqual(destination.stat().st_mtime_ns, first)
            source.write_text(json.dumps({"prefill": [], "decode": ["10.0.0.1:1001"]}))
            bundle.publish_vit_discovery(source, destination, "10.0.0.1:1200")
            self.assertEqual(
                json.loads(destination.read_text()),
                {"prefill": [], "decode": ["10.0.0.1:1001"], "mock.vit.hosts.address": ["10.0.0.1:1200"]},
            )
            source.write_text("broken")
            with self.assertRaises(ValueError):
                bundle.publish_vit_discovery(source, destination, "10.0.0.1:1200")
            self.assertEqual(json.loads(destination.read_text())["decode"], ["10.0.0.1:1001"])

    def test_vit_rejects_legacy_and_overlapping_port(self):
        cfg = {"prefill": 2, "decode": 3}
        with self.assertRaisesRegex(ValueError, "modern Master"):
            bundle.vit_configuration(cfg, {"MOCK_VIT_ENABLED": "1"}, 1000, True)
        for port in ("1004", "65536", "-1", "abc"):
            with self.subTest(port=port), self.assertRaises(ValueError):
                bundle.vit_configuration(cfg, {"MOCK_VIT_ENABLED": "1", "MOCK_VIT_PORT": port}, 1000, False)
        self.assertEqual(bundle.vit_configuration(cfg, {"MOCK_VIT_ENABLED": "1"}, 1000, False)["port"], 1005)

    def test_integer_heap_override_reaches_jvm_as_bytes(self):
        command, _ = self.launch_to_process_boundary({"MOCK_BUNDLE_OVERRIDES_YAML": "{mock_heap: 536870912, master_heap: 536870912}"})
        self.assertIn("-Xmx536870912", command)

    def test_invalid_heap_is_rejected_before_launch(self):
        for value in (None, True, 0, -1, 1.5, "0g", "-1g", "1g -Dunsafe=true"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "mock_heap"):
                self.launch_to_process_boundary({"MOCK_BUNDLE_OVERRIDES_YAML": json.dumps({"mock_heap": value})})

    def test_default_performance_remains_file_based(self):
        command, performance = self.launch_to_process_boundary({})
        self.assertEqual(performance["calibration_id"], "dsv4_l20_legacy_mock")
        calibration = bundle.ROOT.parent / "online_eval/data/performance/dsv4_l20_mock_calibration.json"
        self.assertEqual(performance["calibration_sha256"], hashlib.sha256(calibration.read_bytes()).hexdigest())
        self.assertEqual(set(performance["mock"]["forward"]["decode"]), {"scale", "step_base_ms", "step_per_running_ms", "tokens_per_step"})
        self.assertEqual(
            command[command.index("--mock.calibration") + 1],
            str(Path(command[command.index("--mock.engine_config") + 1]).parent / "performance.json"),
        )
        self.assertIn("--mock.engine_config", command)
        self.assertEqual(command[command.index("--mock.auto_fetch") + 1], "true")
        self.assertEqual(command[command.index("--mock.unique_engine_ips") + 1], "true")
        self.assertNotIn("--mock.events_file", command)

    def test_java_home_works_when_runtime_path_has_no_java(self):
        with tempfile.TemporaryDirectory() as directory:
            java = Path(directory) / "bin" / "java"
            java.parent.mkdir()
            java.write_text("mock java")
            command, _ = self.launch_to_process_boundary({"JAVA_HOME": directory, "PATH": ""})
            self.assertEqual(command[0], str(java))

    def test_event_log_is_explicit(self):
        command, _ = self.launch_to_process_boundary({"MOCK_EVENT_LOG_ENABLED": "1"})
        self.assertIn("--mock.events_file", command)

    def test_frontend_fetch_controls_engine_continuation(self):
        command, _ = self.launch_to_process_boundary({"FETCH_OUTPUT_STREAM": "1"})
        self.assertEqual(command[command.index("--mock.auto_fetch") + 1], "false")
        self.assertEqual(command[command.index("--mock.unique_engine_ips") + 1], "false")

    def test_invalid_fetch_mode_cannot_launch(self):
        with self.assertRaisesRegex(ValueError, "FETCH_OUTPUT_STREAM must be 0 or 1"):
            self.launch_to_process_boundary({"FETCH_OUTPUT_STREAM": "invalid"})

    def test_engine_behavior_and_eos_calibration_are_separate(self):
        engine = json.loads((bundle.ROOT / "engine-config.json").read_text())
        engine["roles"]["decode"]["scheduler"]["max_generate_batch_size"] = 8
        command, performance = self.launch_to_process_boundary(
            {
                "MOCK_ENGINE_CONFIG_JSON": json.dumps(engine),
                "MOCK_CALIBRATION_CONFIG_JSON": '{"mock":{"forward":{"decode":{"step_base_ms":20}}}}',
                "MOCK_EOS_CONFIG_JSON": '{"enabled":false}',
            }
        )
        self.assertIn("--mock.engine_config", command)
        self.assertNotIn("--decode-max-concurrency", command)
        self.assertEqual(performance["mock"]["forward"]["decode"]["step_base_ms"], 20)
        self.assertEqual(performance["mock"]["forward"]["decode"]["eos"], {"enabled": False})

    def test_invalid_capacity_cannot_launch(self):
        for value in [0, -1, True, "64"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.launch_to_process_boundary({"MOCK_BUNDLE_OVERRIDES_YAML": json.dumps({"block_size": value})})

    def test_block_override_alone_cannot_disagree_with_default_performance(self):
        with self.assertRaises(ValueError):
            self.launch_to_process_boundary({"MOCK_BUNDLE_OVERRIDES_YAML": '{"block_size":64}'})

    def test_mismatched_performance_block_size_cannot_launch(self):
        with self.assertRaises(ValueError):
            self.launch_to_process_boundary({"MOCK_PERFORMANCE_CONFIG_JSON": '{"block_size":64}'})


if __name__ == "__main__":
    unittest.main()
