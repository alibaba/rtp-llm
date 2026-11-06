"""The opt-in check compares each PD peer's private cache after real requests."""

import argparse
import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from rtp_llm.test.smoke.kimi_k3 import two_host_launcher as launcher
from rtp_llm.test.smoke.kimi_k3.k31_smoke_capacity import cache_geometry, fastsafetensors_proof


def config():
    value = {
        "run_id": "jit-test",
        "container_user": "tester",
        "check_jit_warmup": True,
        "request_argv": ["python3", "existing_smoke.py"],
    }
    for role, ip in (("prefill", "10.0.0.1"), ("decode", "10.0.0.2")):
        value[role] = dict(
            ssh_target=role,
            container="test",
            ip=ip,
            cluster="test",
            repo_root="/data/repo",
            artifact_root="/data/runs",
            port=25000,
            checkpoint="/data/model",
            draft_checkpoint="/data/draft",
        )
    return value


class TwoHostJitWarmupTest(unittest.TestCase):
    @unittest.skipUnless(Path("/proc/self/stat").is_file(), "requires Linux /proc")
    def test_stop_reaps_child_created_during_shutdown(self):
        code = """
import signal, subprocess, sys, time
from pathlib import Path

def shutdown(*_):
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])
    Path(sys.argv[1]).write_text(str(child.pid))
    raise SystemExit(0)

signal.signal(signal.SIGTERM, shutdown)
print('ready', flush=True)
time.sleep(120)
"""
        with tempfile.TemporaryDirectory() as tmp:
            child_file = Path(tmp) / "child.pid"
            run_id = "jit-late-child"
            with subprocess.Popen(
                [sys.executable, "-c", code, str(child_file)],
                env={**os.environ, "K31_PD_SMOKE_RUN_ID": run_id},
                start_new_session=True, text=True, stdout=subprocess.PIPE,
            ) as leader:
                try:
                    self.assertEqual(leader.stdout.readline().strip(), "ready")
                    identity = dict(uid=os.getuid(), pid_namespace=os.readlink("/proc/self/ns/pid"),
                                    server_pid=leader.pid, pgid=leader.pid, run_id=run_id,
                                    server_start_ticks=launcher.start_ticks(leader.pid))
                    launcher.stop_owned_group(identity)
                    child = Path("/proc") / child_file.read_text()
                    if (child / "stat").is_file():
                        self.assertEqual((child / "stat").read_text().rsplit(")", 1)[1].split()[0], "Z")
                finally:
                    try:
                        os.killpg(leader.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    def test_realized_capacity_requires_draft_and_generation_headroom(self):
        def log(blocks, draft_layer=93, linear_blocks=135):
            return "".join(
                f"CP sharded KV cache enabled: cp_rank={rank}, cp_size=8, "
                "block_size=1024, virtual_block_size=8192\n"
                f"[RANK {rank}] cache manager initialized with config CacheConfig{{\n"
                "  layer_num=93\n  layer_all_num=94\n  linear_step=8\n"
                f"  group_block_nums={blocks},{blocks},{linear_blocks}\n"
                "    kernel_seq_size_per_block=128\n"
                "  group_tags=[full,indexer_kv,linear]\n"
                f"  layers_by_group=[[3,{draft_layer}],[3],[0]]\n"
                for rank in range(8)
            )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "engine.log"
            path.write_text(log(131))
            proof = cache_geometry(path, "prefill", 8, long_case=True)
            self.assertEqual(proof["usable_pages"]["7"]["full"], 130)
            path.write_text(log(130))
            with self.assertRaisesRegex(ValueError, "capacity"):
                cache_geometry(path, "prefill", 8, long_case=True)
            path.write_text(log(131, linear_blocks=134))
            with self.assertRaisesRegex(ValueError, "capacity"):
                cache_geometry(path, "prefill", 8, long_case=True)
            path.write_text(log(131, draft_layer=92))
            with self.assertRaisesRegex(ValueError, "draft"):
                cache_geometry(path, "prefill", 8, long_case=True)

    def test_all_ranks_must_report_target_and_draft_loader_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs = Path(tmp)
            marker = "load method: fastsafetensors, finally choose load method: fastsafetensors\n"
            for rank in range(8):
                (logs / f"main_{rank}.log").write_text(marker * 2)
            self.assertEqual(fastsafetensors_proof(logs)["7"], [1, 2])
            (logs / "main_7.log").write_text(marker + marker.replace("fastsafetensors", "scratch"))
            with self.assertRaisesRegex(ValueError, "unexpected checkpoint loader"):
                fastsafetensors_proof(logs)

    def test_default_and_explicit_prefill_workspace_budget(self):
        value = config()
        self.assertEqual(
            launcher.prefill_workspace_environment(value),
            {"RTP_PREFILL_KV_WORKSPACE_BUDGET_GIB": "8"},
        )
        value["prefill_kv_workspace_budget_gib"] = 10
        self.assertIs(launcher.validate(value), value)
        self.assertEqual(
            launcher.prefill_workspace_environment(value),
            {"RTP_PREFILL_KV_WORKSPACE_BUDGET_GIB": "10"},
        )
        for invalid in (True, "10", 0, -1, float("inf"), float("nan"), 1e-20):
            value["prefill_kv_workspace_budget_gib"] = invalid
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "GiB"):
                launcher.validate(value)

    def test_target_only_does_not_require_or_preflight_draft_paths(self):
        value = config()
        value["target_only"] = True
        for role in launcher.ROLES:
            value[role].pop("draft_checkpoint")
        self.assertIs(launcher.validate(value), value)
        self.assertEqual(launcher.checkpoint_fields(value), ("checkpoint",))
        value["target_only"] = False
        with self.assertRaises(KeyError):
            launcher.validate(value)

    def test_matrix_options_require_boolean_values(self):
        for field, value in (("target_only", "false"), ("enable_cuda_graph", 0)):
            candidate = config()
            if field == "target_only":
                candidate[field] = value
            else:
                candidate["decode"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError, "boolean"
            ):
                launcher.validate(candidate)
        candidate = config()
        candidate["decode"]["enable_cuda_graph"] = False
        self.assertIs(launcher.validate(candidate), candidate)

    def test_structural_cli_can_reuse_cases_without_mtp(self):
        path = Path(__file__).with_name("k31_structural_smoke.py")
        spec = importlib.util.spec_from_file_location("k31_structural_cli_test", path)
        module = importlib.util.module_from_spec(spec)
        dependency = ModuleType("text_smoke")
        dependency.prefill_cache_tiers = None
        with patch.dict(sys.modules, {spec.name: module, "text_smoke": dependency}):
            spec.loader.exec_module(module)

        def run_group(args, cases, output):
            module.check_draft_execution(0, target_only=args.target_only)
            return [{"target_only": args.target_only} for _ in cases]

        with tempfile.TemporaryDirectory() as tmp:
            argv = [
                str(path),
                "--prefill-rpc",
                "127.0.0.1:25000",
                "--decode-ip",
                "127.0.0.2",
                "--decode-port",
                "26000",
                "--decode-tp",
                "4",
                "--output",
                str(Path(tmp) / "result"),
                "--target-only",
                "--first-request-only",
            ]
            with patch.object(sys, "argv", argv), patch.object(
                module, "run_group", side_effect=run_group
            ):
                module.main()
            summary = json.loads(
                (Path(tmp) / "result" / "structural-summary.json").read_text()
            )
            self.assertTrue(summary["structural_passed"])
            self.assertEqual(summary["requests"], 1)

    def test_opt_in_requires_existing_request_runner(self):
        value = config()
        value.pop("request_argv")
        with self.assertRaisesRegex(ValueError, "request_argv"):
            launcher.validate(value)
        value["check_jit_warmup"] = False
        self.assertIs(launcher.validate(value), value)

    def test_private_cache_is_per_run_and_cannot_be_reused(self):
        value = config()
        with tempfile.TemporaryDirectory() as tmp:
            value["prefill"]["artifact_root"] = tmp
            directory = launcher.run_dir(value, "prefill")
            directory.mkdir(parents=True)
            cache = launcher.prepare_jit_cache(value, "prefill")
            self.assertEqual(cache, directory / "jit")
            self.assertTrue((cache / "deep-gemm").is_dir())
            with self.assertRaises(FileExistsError):
                launcher.prepare_jit_cache(value, "prefill")

    def test_default_still_reuses_the_existing_shared_cache(self):
        value = config()
        value.pop("check_jit_warmup")
        with tempfile.TemporaryDirectory() as tmp:
            value["prefill"]["artifact_root"] = tmp
            first = launcher.prepare_jit_cache(value, "prefill")
            second = launcher.prepare_jit_cache(value, "prefill")
            self.assertEqual(first, Path(tmp) / "jit")
            self.assertEqual(first, second)

    def test_role_snapshot_uses_receipt_bound_private_cache(self):
        value = config()
        value["decode"]["repo_root"] = str(Path(__file__).resolve().parents[4])
        with tempfile.TemporaryDirectory() as tmp:
            value["decode"]["artifact_root"] = tmp
            directory = launcher.run_dir(value, "decode")
            directory.mkdir(parents=True)
            cache = launcher.prepare_jit_cache(value, "decode")
            native = cache / "deep-gemm" / "kernel.cubin"
            native.write_bytes(b"native")
            identity = dict(
                uid=launcher.os.getuid(),
                run_id=value["run_id"],
                environment={
                    "DG_JIT_CACHE_DIR": str(cache / "deep-gemm"),
                    "TRITON_CACHE_DIR": str(cache / "triton"),
                },
            )
            launcher.atomic(directory / "instance.json", identity)
            snapshot = launcher.jit_keys(value, "decode")
            self.assertEqual(snapshot["role"], "decode")
            self.assertEqual(snapshot["cache_root"], str(cache))
            self.assertEqual(set(snapshot["keys"]), {"deep-gemm/kernel.cubin"})
            identity["environment"]["DG_JIT_CACHE_DIR"] = "/shared/cache"
            launcher.atomic(directory / "instance.json", identity)
            with self.assertRaisesRegex(ValueError, "receipt"):
                launcher.jit_keys(value, "decode")

    def launch(self, value, output, *, changed_role=None, empty_role=None, failed_stop=None):
        calls = {role: 0 for role in launcher.ROLES}

        def remote(_config, role, operation):
            if operation == "status":
                return {"healthy": True}
            if operation == "requests":
                return {"exit_code": 0}
            if operation == "stop-role":
                if role == failed_stop:
                    raise OSError("remote stop failed")
                return {"stop_requested": True}
            if operation != "jit-keys":
                raise AssertionError(operation)
            calls[role] += 1
            keys = {
                "deep-gemm/cache/sm100_mqa_logits/kernel.cubin": "native",
                "triton/one/kernel.cubin": "triton",
            }
            if role == changed_role and calls[role] == 2:
                keys["deep-gemm/cache/sm100_paged_mqa_logits/new.cubin"] = "new"
            if role == empty_role:
                keys = {}
            return dict(
                role=role,
                hostname=role + "-host",
                cache_root=f"/data/runs/jit-test/{role}/jit",
                keys=keys,
            )

        args = argparse.Namespace(
            dry_run=False,
            operation="launch",
            output=output,
            startup_timeout=30,
            keep_services=False,
        )
        with patch.object(
            launcher, "ssh", return_value=subprocess.CompletedProcess([], 0, "image\n")
        ), patch.object(launcher, "remote", side_effect=remote):
            return launcher.controller(args, value)

    def test_request_runner_is_checked_against_both_remote_caches(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "control"
            self.assertEqual(self.launch(config(), output), 0)
            evidence = json.loads((output / "jit-warmup.json").read_text())
            self.assertTrue(evidence["passed"])
            self.assertEqual(set(evidence["before"]), {"prefill", "decode"})
            self.assertEqual(evidence["after"]["decode"]["hostname"], "decode-host")
            self.assertEqual(evidence["changes"]["decode"]["added"], [])

    def test_new_decode_kernel_fails_and_preserves_peer_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "control"
            with self.assertRaisesRegex(RuntimeError, "JIT"):
                self.launch(config(), output, changed_role="decode")
            evidence = json.loads((output / "jit-warmup.json").read_text())
            self.assertFalse(evidence["passed"])
            self.assertEqual(evidence["changes"]["prefill"]["added"], [])
            self.assertEqual(
                evidence["changes"]["decode"]["added"],
                ["deep-gemm/cache/sm100_paged_mqa_logits/new.cubin"],
            )

    def test_empty_decode_cache_cannot_vacuously_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "control"
            with self.assertRaisesRegex(RuntimeError, "JIT"):
                self.launch(config(), output, empty_role="decode")
            self.assertTrue((output / "jit-warmup-before.json").exists())

    def test_stop_failure_does_not_skip_the_other_role(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "control"
            with self.assertRaisesRegex(RuntimeError, "Failed to stop"):
                self.launch(config(), output, failed_stop="decode")
            cleanup = json.loads((output / "stop-receipts.json").read_text())
            self.assertIn("remote stop failed", cleanup["decode"]["error"])
            self.assertTrue(cleanup["prefill"]["stop_requested"])


if __name__ == "__main__":
    unittest.main()
