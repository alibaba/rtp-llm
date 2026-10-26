import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import bundle
import deployment
import standalone_mock


class DeploymentTest(unittest.TestCase):
    def test_three_contracts(self):
        for mode, entrypoint, discovery, remote in (
            ("embedded", "bundle", "file", False),
            ("single_pod", "mock", "vipserver", False),
            ("per_engine", "mock", "vipserver", True),
        ):
            with self.subTest(mode=mode):
                plan = deployment.resolve({"MOCK_DEPLOYMENT_MODE": mode})
                self.assertEqual((plan.entrypoint, plan.master_discovery, plan.remote_decode), (entrypoint, discovery, remote))

    def test_legacy_entrypoints_and_deprecation(self):
        self.assertEqual(deployment.resolve({}).mode, "embedded")
        self.assertEqual(deployment.resolve({}, "mock").mode, "single_pod")
        with contextlib.redirect_stderr(io.StringIO()) as stderr:
            plan = deployment.resolve({"MOCK_BUNDLE_ROLE": "mock", "MOCK_REMOTE_DECODE": "true"})
        self.assertEqual(plan.mode, "per_engine")
        self.assertIn("Deprecated MOCK_REMOTE_DECODE", stderr.getvalue())

    def test_conflicting_or_inapplicable_flags_fail_closed(self):
        cases = [
            {"MOCK_DEPLOYMENT_MODE": "typo"},
            {"MOCK_DEPLOYMENT_MODE": "embedded", "MOCK_BUNDLE_ROLE": "mock"},
            {"MOCK_DEPLOYMENT_MODE": "single_pod", "MOCK_REMOTE_DECODE": "1"},
            {"MOCK_DEPLOYMENT_MODE": "per_engine", "MOCK_REMOTE_DECODE": "0"},
            {"MOCK_DEPLOYMENT_MODE": "embedded", "MOCK_BUNDLE_FILE_DISCOVERY": "0"},
            {"MOCK_DEPLOYMENT_MODE": "single_pod", "MOCK_BUNDLE_FILE_DISCOVERY": "1"},
            {"MOCK_DEPLOYMENT_MODE": "single_pod", "MOCK_BUNDLE_OVERRIDES_YAML": "{}"},
            {"MOCK_DEPLOYMENT_MODE": "per_engine", "MOCK_BUNDLE_LEGACY_MASTER": "1"},
            {"MOCK_BUNDLE_ROLE": "typo"},
            {"MOCK_REMOTE_DECODE": "maybe"},
            {"MOCK_REMOTE_DECODE": "true"},
        ]
        for env in cases:
            with self.subTest(env=env), self.assertRaises(ValueError):
                deployment.resolve(env)

    def test_dispatch_uses_plan(self):
        for mode, module in (("embedded", bundle), ("single_pod", standalone_mock), ("per_engine", standalone_mock)):
            with patch.dict(os.environ, {"MOCK_DEPLOYMENT_MODE": mode}, clear=True), patch.object(module, "run") as run:
                deployment.main()
                run.assert_called_once_with()

    def test_vit_deployment_uses_role_metadata_launcher_then_same_cluster_entry(self):
        import vit
        with patch.dict(os.environ, {"MOCK_DEPLOYMENT_MODE":"per_engine", "ROLE_TYPE":"VIT"}, clear=True), \
                patch.object(vit,"command",return_value=["java","cluster"]) as command, patch("os.execv") as execute:
            deployment.main()
        command.assert_called_once()
        execute.assert_called_once_with("java",["java","cluster"])

    def test_wrong_direct_entrypoint_fails_before_launch(self):
        with (
            patch.dict(os.environ, {"RTP_LLM_MOCK_BUNDLE": "1", "MOCK_DEPLOYMENT_MODE": "single_pod"}, clear=True),
            patch.object(bundle.subprocess, "Popen") as start,
        ):
            with self.assertRaisesRegex(ValueError, "launcher conflicts"):
                bundle.run()
            start.assert_not_called()

    def test_embedded_master_discovery_is_file_for_both_versions(self):
        endpoints = {"env": {"MODEL_SERVICE_CONFIG": '{"discovery_file":"/run/discovery.json"}'}}
        current = bundle.master_discovery_environment(endpoints, False)
        legacy = bundle.master_discovery_environment(endpoints, True)
        self.assertEqual(json.loads(current["MODEL_SERVICE_CONFIG"])["discovery_file"], "/run/discovery.json")
        self.assertEqual(legacy["MOCK_DISCOVERY_FILE"], "/run/discovery.json")
        self.assertNotIn("discovery_file", json.loads(legacy["MODEL_SERVICE_CONFIG"]))
        self.assertFalse(any(key.startswith("DOMAIN_ADDRESS:") for key in legacy))

    def test_per_engine_role_and_single_pod_vip_port_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            base = {
                "RTP_LLM_MOCK_BUNDLE": "1",
                "POD_IP": "10.0.0.5",
                "MOCK_BUNDLE_RUN_DIR": directory,
                "START_PORT": "7001",
                "MOCK_ENGINE_CONFIG_JSON": (Path(__file__).parent / "engine-config.json").read_text(),
            }
            for role in ("PREFILL", "DECODE", "VIT"):
                args = standalone_mock.command({**base, "MOCK_DEPLOYMENT_MODE": "per_engine", "ROLE_TYPE": role})
                self.assertEqual(args[args.index("--mock.whale_bundle") + 1], "false")
                self.assertNotIn("--mock.discovery_file", args)
                plan = json.loads(Path(directory, "deployment.json").read_text())
                self.assertEqual(plan["vip_grpc_endpoints"], [{"role": role.lower(), "ip": "10.0.0.5", "port": 7002}])
            args = standalone_mock.command(
                {**base, "MOCK_DEPLOYMENT_MODE": "single_pod", "MOCK_PREFILL_COUNT": "2", "MOCK_DECODE_COUNT": "3"}
            )
            self.assertEqual(args[args.index("--mock.whale_bundle") + 1], "true")
            plan = json.loads(Path(directory, "deployment.json").read_text())
            self.assertEqual([e["port"] for e in plan["vip_grpc_endpoints"]], list(range(7002, 7007)))
            self.assertEqual([e["role"] for e in plan["vip_grpc_endpoints"]], ["prefill"] * 2 + ["decode"] * 3)
            for override in (
                {"MOCK_DEPLOYMENT_MODE": "per_engine"},
                {"MOCK_DEPLOYMENT_MODE": "per_engine", "ROLE_TYPE": "PREFILL", "MOCK_DECODE_COUNT": "1"},
                {"MOCK_DEPLOYMENT_MODE": "single_pod", "ROLE_TYPE": "PREFILL"},
                {"MOCK_DEPLOYMENT_MODE": "single_pod", "MOCK_GRPC_BASE_PORT": "8000"},
            ):
                with self.subTest(override=override), self.assertRaises(ValueError):
                    standalone_mock.command({**base, **override})
