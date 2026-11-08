"""Transport contracts for the shared traditional/CI K3.1 workflow."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rtp_llm.test.smoke.kimi_k3 import (
    k31_runtime_bootstrap,
    k31_server_launcher,
    k31_smoke,
    k31_ssh_smoke,
)


def config():
    value = dict(
        run_id="unified", external_revision="a" * 40, internal_revision="b" * 40
    )
    for role, ip, port in (
        ("prefill", "192.0.2.1", 25000),
        ("decode", "192.0.2.2", 26000),
    ):
        value[role] = dict(
            ssh_target=role,
            container="test",
            cluster="test",
            ip=ip,
            port=port,
            repo_root="/runtime/source",
            server="/runtime/server",
            artifact_root="/owned/output",
            checkpoint="/target/model",
            draft_checkpoint="/draft/model",
            checkpoint_root="/target",
            draft_checkpoint_root="/draft",
        )
    return value


class SSHSmokeTest(unittest.TestCase):
    def test_bundled_cutlass_precedes_a_global_copy_in_spawned_python(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            server = root / "server"
            packages = root / "server.runfiles/pip_cutlass/site-packages"
            bundled = packages / "nvidia_cutlass_dsl/python_packages"
            old = root / "global-site"
            for path, declaration in (
                (bundled, "OperandMajorMode = 'bundled'\n"),
                (old, "# Old global CUTLASS has no OperandMajorMode.\n"),
            ):
                package = path
                for name in ("cutlass", "cute", "nvgpu"):
                    package /= name
                    package.mkdir(parents=True, exist_ok=True)
                    (package / "__init__.py").write_text("")
                (package / "__init__.py").write_text(declaration)
            pythonpath = k31_runtime_bootstrap.runtime_pythonpath(root, server)
            environment = os.environ.copy()
            environment["PYTHONPATH"] = pythonpath + os.pathsep + str(old)
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from cutlass.cute.nvgpu import OperandMajorMode; "
                    "assert OperandMajorMode == 'bundled'",
                ],
                env=environment,
                check=True,
                timeout=30,
            )

    def test_transports_share_worker_config_and_two_independent_checkpoint_roots(self):
        selected = k31_ssh_smoke.runtime_config(config())
        expected = k31_smoke.smoke_config(config())
        for role in ("prefill", "decode"):
            for key in ("test_block_num", "linear_pool_blocks", "max_seq_len"):
                self.assertEqual(selected[role][key], expected[role][key])
            self.assertNotIn("container", selected[role])
            self.assertEqual(selected[role]["allow_checkpoint_root"], "/target")
            self.assertEqual(selected[role]["draft_checkpoint_root"], "/draft")
        self.assertEqual(selected["decode_dp"], 2)
        self.assertEqual(selected["prefill"]["reuse_cache_tail_blocks"], 5)
        self.assertEqual(selected["prefill_kv_workspace_budget_gib"], 8)

    def test_standalone_output_must_be_owned_and_inside_its_explicit_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            child = root / "run"
            child.mkdir()
            args = SimpleNamespace(traditional_artifact_root=root)
            k31_server_launcher.validate_destination(child, args)
            with self.assertRaisesRegex(ValueError, "escapes"):
                k31_server_launcher.validate_destination(root.parent, args)
            child.chmod(0o777)
            with self.assertRaisesRegex(ValueError, "private"):
                k31_server_launcher.validate_destination(child, args)

    def test_bazel_output_gate_is_not_weakened(self):
        args = SimpleNamespace(traditional_artifact_root=None)
        with patch.object(
            k31_server_launcher.shared, "require_bazel_destination"
        ) as gate:
            k31_server_launcher.validate_destination(Path("/selected/output"), args)
            gate.assert_called_once_with(Path("/selected/output"))

    def run_transport(self, *, request_error=False, close_error=False):
        events = []

        class Connection:
            def __init__(self, base, role, output):
                self.role = role

            def call(self, operation, **fields):
                events.append((self.role, operation))
                if operation == "status":
                    return {"healthy": True}
                if operation == "requests":
                    if request_error:
                        raise RuntimeError("regular execution audit failed")
                    return {"audit": {"passed": True}}
                return {"artifact_dir": "/owned/run/" + self.role}

            def close(self):
                events.append((self.role, "close-channel"))
                if close_error and self.role == "prefill":
                    raise RuntimeError("channel close failed")

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "result"
            with patch.object(k31_ssh_smoke, "RoleConnection", Connection):
                code = k31_ssh_smoke.run(config(), output, 1)
            result = json.loads((output / "RESULT.json").read_text())
        return code, result, events

    def test_one_lifecycle_runs_the_combined_requests_and_closes_both_peers(self):
        code, result, events = self.run_transport()
        self.assertEqual(code, 0)
        self.assertTrue(result["passed"])
        self.assertEqual(
            [role for role, operation in events if operation == "start"],
            ["decode", "prefill"],
        )
        self.assertEqual(
            [role for role, operation in events if operation == "requests"], ["prefill"]
        )
        self.assertEqual(
            {role for role, operation in events if operation == "close"},
            {"prefill", "decode"},
        )

    def test_request_or_transport_failure_cannot_skip_the_other_peer_cleanup(self):
        for errors in ({"request_error": True}, {"close_error": True}):
            with self.subTest(errors=errors):
                code, result, events = self.run_transport(**errors)
                self.assertEqual(code, 1)
                self.assertFalse(result["passed"])
                self.assertEqual(
                    {role for role, operation in events if operation == "close"},
                    {"prefill", "decode"},
                )


if __name__ == "__main__":
    unittest.main()
