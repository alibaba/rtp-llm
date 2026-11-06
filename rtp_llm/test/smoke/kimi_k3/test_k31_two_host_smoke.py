"""CPU coverage for the default two-stage K3.1 smoke entry."""

import io
import json
import os
import shlex
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rtp_llm.test.smoke.kimi_k3 import k31_two_host_smoke as smoke


def base_config():
    config = {"run_id": "test-k31", "fp8_gemm": True, "fp8_kv_cache": False}
    for role, ip, port in (
        ("prefill", "10.0.0.1", 25000),
        ("decode", "10.0.0.2", 26000),
    ):
        config[role] = dict(
            ssh_target=role,
            container="test",
            ip=ip,
            cluster="test",
            repo_root="/data/repo",
            artifact_root="/data/runs",
            port=port,
            checkpoint="/data/model",
            draft_checkpoint="/data/draft",
        )
    config["decode"]["test_block_num"] = 103
    return config


class K31TwoHostSmokeTest(unittest.TestCase):
    def test_regular_only_honors_tp8_decode_and_excludes_one_million_stage(self):
        base = {**base_config(), "decode_dp": 1}
        configs = smoke.stage_configs(base, ("regular",))
        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0]["k31_smoke_stage"], "regular")
        self.assertEqual(configs[0]["decode_dp"], 1)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result"
            record = dict(passed=True, cleanup_complete=True,
                          request_counts=dict(total=304, formal=267, preparation=37))
            with patch.object(smoke, "run_stage", return_value=record) as run_stage:
                self.assertEqual(smoke.run(base, output, 1, ("regular",)), 0)
            run_stage.assert_called_once()
            result = json.loads((output / "RESULT.json").read_text())
            self.assertEqual(list(result["stages"]), ["regular"])
            self.assertEqual(result["expected_counts"], record["request_counts"])

    def test_direct_role_launcher_bootstraps_policy_without_pythonpath(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, smoke.launcher.__file__, "--help"],
                cwd=directory, env={**os.environ, "PYTHONPATH": ""},
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_nvfp4_config_resolution_survives_remote_environment(self):
        with patch.dict(os.environ, {}, clear=True):
            regular, long_case = smoke.stage_configs(base_config())
            self.assertTrue(regular["kda_nvfp4"] and regular["dsa_nvfp4"])
            self.assertTrue(long_case["kda_nvfp4"] and long_case["dsa_nvfp4"])
        with patch.dict(os.environ, KDA_NVFP4="0", DSA_NVFP4="1"):
            regular, _ = smoke.stage_configs(base_config())
        with patch.dict(os.environ, KDA_NVFP4="1", DSA_NVFP4="0"):
            received = smoke.launcher.validate(regular)
            self.assertFalse(received["kda_nvfp4"])
            self.assertTrue(received["dsa_nvfp4"])
        for key in ("kda_nvfp4", "dsa_nvfp4"):
            for value in (None, 1, "0"):
                with self.assertRaisesRegex(ValueError, "must be boolean"):
                    smoke.stage_configs({**base_config(), key: value})

    def test_precision_audit_requires_executed_fp4_and_native_draft(self):
        from .nvfp4_evidence import audit_projection_precision, MARKER

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "logs").mkdir()
            rows = []
            config = dict(kda_nvfp4=True, dsa_nvfp4=False)
            for label in ("input", "forget", "output"):
                row = dict(model="target", rank=0, layer=0, module="kda." + label,
                           linear="CudaFp4GEMMLinear", weight_dtype="torch.uint8",
                           weight_precision="NVFP4", activation_precision="NVFP4",
                           activation_scheme="dynamic", **config)
                rows += [dict(row, event="initialized"), dict(row, event="executed",
                         input_dtype="torch.bfloat16", output_dtype="torch.bfloat16")]
            rows.append(dict(model="draft", rank=0, layer=0, module="dsa.input",
                             linear="CudaF16Linear", activation_scheme="native",
                             weight_dtype="torch.bfloat16", weight_precision="BF16",
                             activation_precision="BF16", kda_nvfp4=False, dsa_nvfp4=False))
            log = root / "logs/main_0.log"
            def write(receipts):
                log.write_text("\n".join(MARKER + json.dumps(row) for row in receipts))
            write(rows)
            self.assertTrue(audit_projection_precision(root, config, layers=1, ranks=1)["passed"])
            rows[0]["execution_fragments"] = ["q_gate", "kv"]
            rows[1]["fragment"] = "q_gate"
            write(rows)
            with self.assertRaisesRegex(ValueError, "never executed"):
                audit_projection_precision(root, config, layers=1, ranks=1)
            rows.append(dict(rows[1], fragment="kv"))
            write(rows)
            self.assertTrue(audit_projection_precision(root, config, layers=1, ranks=1)["passed"])
            rows.pop()
            rows[0].pop("execution_fragments")
            rows[1].pop("fragment")
            write([row for row in rows if row.get("event") != "executed"])
            with self.assertRaisesRegex(ValueError, "never executed"):
                audit_projection_precision(root, config, layers=1, ranks=1)
            rows[-1]["weight_dtype"] = "torch.uint8"
            write(rows)
            with self.assertRaisesRegex(ValueError, "draft precision"):
                audit_projection_precision(root, config, layers=1, ranks=1)

            config = dict(kda_nvfp4=False, dsa_nvfp4=False)
            rows[-1]["weight_dtype"] = "torch.bfloat16"
            fp8_rows = [dict(row, linear="CudaFp8GEMMLinear", weight_dtype="torch.float8_e4m3fn",
                             weight_precision="FP8", activation_precision="FP8",
                             activation_scheme="native", **config)
                        for row in rows[:-1] if row["event"] == "initialized"]
            write(fp8_rows + rows[-1:])
            receipt = audit_projection_precision(root, config, layers=1, ranks=1)
            self.assertEqual(receipt["initialized_by_precision"], {"FP8": 3})
            self.assertEqual(receipt["nvfp4_executed"], 0)
            fp8_rows[0]["activation_precision"] = "BF16"
            write(fp8_rows + rows[-1:])
            with self.assertRaisesRegex(ValueError, "projection precision"):
                audit_projection_precision(root, config, layers=1, ranks=1)

    def test_direct_entry_preserves_runfiles_package_and_finds_smoke_modules(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "rtp_llm"
            package.mkdir()
            (package / "__init__.py").touch()
            source = Path(smoke.__file__).resolve()
            code = """
import runpy, sys
from pathlib import Path
import rtp_llm
runfiles_package = rtp_llm.__file__
sys.path.insert(0, str(Path(sys.argv[1]).parent))
runpy.run_path(sys.argv[1])
from rtp_llm.test.smoke.kimi_k3 import mtp_contract
assert rtp_llm.__file__ == runfiles_package
"""
            result = subprocess.run(
                [sys.executable, "-c", code, str(source)],
                cwd=directory,
                env={**os.environ, "PYTHONPATH": directory},
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_default_phases_only_change_capacity_and_decode_topology(self):
        base = base_config()
        regular, long_case = smoke.stage_configs(base)
        self.assertEqual(base, base_config())
        self.assertEqual(regular["run_id"], "test-k31-regular")
        self.assertEqual(long_case["run_id"], "test-k31-long")
        self.assertEqual(smoke.stage_configs(base, ("long", "regular")), [long_case, regular])
        self.assertEqual((regular["decode_dp"], long_case["decode_dp"]), (2, 1))
        self.assertEqual(regular["prefill"]["test_block_num"], 96)
        self.assertEqual(regular["decode"]["test_block_num"], 103)
        self.assertEqual(long_case["prefill"]["test_block_num"], 140)
        self.assertEqual(long_case["decode"]["test_block_num"], 140)
        self.assertEqual(long_case["prefill"]["linear_pool_blocks"], 1032)
        self.assertEqual(long_case["prefill"]["request_capacity"], 1)
        self.assertEqual(long_case["prefill"]["block_tree_transfer_worker_count"], 1)
        self.assertEqual(long_case["prefill"]["prefill_token_capacity"], 8192)
        for role in ("prefill", "decode"):
            self.assertEqual(long_case[role]["max_seq_len"], 1_052_672)
        self.assertNotIn("linear_pool_blocks", regular["prefill"])
        self.assertNotIn("block_tree_transfer_worker_count", regular["prefill"])
        self.assertNotIn("linear_pool_blocks", long_case["decode"])
        self.assertEqual(long_case["decode"]["request_capacity"], 1)
        self.assertEqual(long_case["decode"]["decode_capture_batches"], "1")
        for stage in (regular, long_case):
            self.assertIs(stage["target_only"], False)
            self.assertIs(stage["fp8_gemm"], True)
            self.assertIs(stage["fp8_kv_cache"], False)
            self.assertEqual(stage["prefill_kv_workspace_budget_gib"], 8)
            for role in ("prefill", "decode"):
                self.assertEqual(stage[role]["checkpoint"], base[role]["checkpoint"])
                self.assertEqual(
                    stage[role]["draft_checkpoint"], base[role]["draft_checkpoint"]
                )
            self.assertEqual(
                stage["request_argv"][-1],
                f"/data/runs/{stage['run_id']}/prefill/control.json",
            )

    def test_both_phases_keep_explicit_workspace_budget(self):
        base = {**base_config(), "prefill_kv_workspace_budget_gib": 10}
        for stage in smoke.stage_configs(base):
            self.assertEqual(stage["prefill_kv_workspace_budget_gib"], 10)

    def test_default_phases_require_native_mtp(self):
        with self.assertRaisesRegex(ValueError, "native MTP"):
            smoke.stage_configs({**base_config(), "target_only": True})

    def test_loader_evidence_requires_all_ranks_and_the_final_method(self):
        with tempfile.TemporaryDirectory() as directory:
            logs = Path(directory)
            fast = "load method: fastsafetensors, finally choose load method: fastsafetensors\n"
            for rank in range(8):
                (logs / f"main_{rank}.log").write_text(fast * 2)
            self.assertEqual(
                set(smoke.fastsafetensors_proof(logs)), {str(rank) for rank in range(8)}
            )
            path = logs / "main_5.log"
            path.write_text(
                fast
                + "load method: fastsafetensors, finally choose load method: scratch\n"
            )
            with self.assertRaisesRegex(
                ValueError, "main_5.*unexpected checkpoint loader"
            ):
                smoke.fastsafetensors_proof(logs)

    def test_startup_audit_uses_the_ready_snapshot_after_shutdown(self):
        config, _ = smoke.stage_configs(base_config())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "launch").mkdir()
            ready = {}
            for role in ("prefill", "decode"):
                root = output / role
                logs = root / "logs"
                logs.mkdir(parents=True)
                (root / "control.json").write_text(json.dumps(config))
                # The remote status command can replace its own readiness receipt while stopping.
                (root / "readiness.json").write_text('{"healthy": false}')
                geometry = {"runtime_verified": True, "role": role}
                (root / "runtime-geometry.json").write_text(json.dumps(geometry))
                ready[role] = {
                    "healthy": True,
                    "runtime_geometry": geometry,
                    "rank_prefill_kv_workspace_budget": {
                        str(i): {"gib": str(config["prefill_kv_workspace_budget_gib"])} for i in range(8)
                    },
                }
                for checkpoint in ("checkpoint", "draft_checkpoint"):
                    report = dict(
                        passed=True,
                        load_method="fastsafetensors",
                        config_sha256=checkpoint,
                        index_sha256=checkpoint,
                    )
                    (root / f"{checkpoint}-preflight.json").write_text(
                        json.dumps(report)
                    )
                for i in range(8):
                    (logs / f"main_{i}.log").write_text(
                        "load method: fastsafetensors, finally choose load method: fastsafetensors\n"
                        * 2
                    )
            (output / "launch" / "readiness.json").write_text(json.dumps(ready))
            proof = smoke.audit_startup(config, output)
            self.assertEqual(len(proof["decode"]["fastsafetensors_lines"]), 8)
            for budget in (2, 10):
                config["prefill_kv_workspace_budget_gib"] = budget
                for role in ("prefill", "decode"):
                    (output / role / "control.json").write_text(json.dumps(config))
                    for receipt in ready[role]["rank_prefill_kv_workspace_budget"].values():
                        receipt["gib"] = str(budget)
                (output / "launch/readiness.json").write_text(json.dumps(ready))
                smoke.audit_startup(config, output)
                ready["prefill"]["rank_prefill_kv_workspace_budget"]["7"]["gib"] = str(budget + 1)
                (output / "launch/readiness.json").write_text(json.dumps(ready))
                with self.assertRaisesRegex(ValueError, "all eight.*workspace"):
                    smoke.audit_startup(config, output)

    def test_routes_follow_existing_decode_owner_endpoint_stride(self):
        regular, long_case = smoke.stage_configs(base_config())
        root = Path("/data/runs/prefill")
        regular_args = smoke.request_args(regular, root)
        long_args = smoke.request_args(long_case, root)
        self.assertIn("10.0.0.2:26000:26001", regular_args)
        self.assertIn("10.0.0.2:26036:26037", regular_args)
        self.assertEqual(long_args.count("--decode-role-addr"), 1)
        self.assertIn("10.0.0.2:26000:26001", long_args)

    def test_archive_rejects_symlinks_and_path_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for name, kind in (
                ("../escaped", tarfile.REGTYPE),
                ("logs/link", tarfile.SYMTYPE),
            ):
                with self.subTest(name=name):
                    archive = output / "evidence.tar.gz"
                    with tarfile.open(archive, "w:gz") as stream:
                        entry = tarfile.TarInfo(name)
                        entry.type = kind
                        entry.linkname = "../escaped"
                        stream.addfile(entry, io.BytesIO())
                    with self.assertRaisesRegex(ValueError, "unsafe evidence archive"):
                        smoke.extract_evidence(archive, output / "evidence")

    def test_empty_archive_is_not_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            archive = output / "empty.tar.gz"
            with tarfile.open(archive, "w:gz"):
                pass
            with self.assertRaisesRegex(ValueError, "control.json"):
                smoke.extract_evidence(archive, output / "evidence")
            self.assertFalse((output / "evidence").exists())

    def test_transfer_reads_the_personal_host_run_directory(self):
        config, _ = smoke.stage_configs(base_config())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)

            def host_ssh(argv, *, stdout, stderr, **kwargs):
                self.assertEqual(argv[-2], "prefill")
                command = shlex.split(argv[-1])
                self.assertEqual(command[0:2], ["bash", "-c"])
                self.assertEqual(command[4], "/data/runs/test-k31-regular/prefill")
                self.assertIn("control.json", command[5:])
                self.assertIn("request-evidence", command[5:])
                self.assertNotIn("jit", command[5:])
                stderr.write(b"personal host receipt\n")
                with tarfile.open(fileobj=stdout, mode="w:gz") as archive:
                    for name, data in (
                        ("control.json", json.dumps(config).encode()),
                        ("logs/main_0.log", b"prefill rank 0\n"),
                    ):
                        member = tarfile.TarInfo(name)
                        member.size = len(data)
                        archive.addfile(member, io.BytesIO(data))

            with patch.object(smoke.subprocess, "run", side_effect=host_ssh):
                smoke.transfer_evidence(config, "prefill", output)
            self.assertEqual(
                (output / "prefill/logs/main_0.log").read_text(), "prefill rank 0\n"
            )
            self.assertEqual(
                (output / "prefill-transfer.log").read_text(), "personal host receipt\n"
            )
            self.assertFalse((output / "prefill-evidence.tar.gz").exists())

    def test_failed_transfer_preserves_binary_and_stderr(self):
        config, _ = smoke.stage_configs(base_config())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)

            def failed_ssh(argv, *, stdout, stderr, **kwargs):
                stdout.write(b"partial archive")
                stderr.write(b"read failed")
                raise subprocess.CalledProcessError(1, argv)

            with patch.object(smoke.subprocess, "run", side_effect=failed_ssh):
                with self.assertRaises(subprocess.CalledProcessError):
                    smoke.transfer_evidence(config, "decode", output)
            self.assertEqual(
                (output / "decode-evidence.tar.gz").read_bytes(), b"partial archive"
            )
            self.assertEqual(
                (output / "decode-transfer.log").read_bytes(), b"read failed"
            )
            self.assertFalse((output / "decode").exists())

    def test_failed_regular_audit_keeps_evidence_and_does_not_erase_long_result(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result"
            stages = [
                {
                    "passed": False,
                    "cleanup_complete": True,
                    "errors": ["regular audit failed"],
                },
                {
                    "passed": True,
                    "cleanup_complete": True,
                    "request_counts": dict(total=2, formal=1, preparation=1),
                },
            ]
            with patch.object(smoke, "run_stage", side_effect=stages) as run_stage:
                self.assertEqual(smoke.run(base_config(), output, startup_timeout=1), 1)
            self.assertEqual(
                [call.args[0]["k31_smoke_stage"] for call in run_stage.call_args_list],
                ["regular", "long"],
            )
            result = json.loads((output / "RESULT.json").read_text())
            self.assertIs(result["passed"], False)
            self.assertIn("regular audit failed", result["stages"]["regular"]["errors"])
            self.assertIs(result["stages"]["long"]["passed"], True)

    def test_aggregate_counts_only_pass_after_both_audits(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result"
            stages = [
                {
                    "passed": True,
                    "cleanup_complete": True,
                    "request_counts": dict(total=304, formal=267, preparation=37),
                },
                {
                    "passed": True,
                    "cleanup_complete": True,
                    "request_counts": dict(total=2, formal=1, preparation=1),
                },
            ]
            with patch.object(smoke, "run_stage", side_effect=stages):
                self.assertEqual(smoke.run(base_config(), output, startup_timeout=1), 0)
            result = json.loads((output / "RESULT.json").read_text())
            self.assertIs(result["passed"], True)
            self.assertEqual(
                result["request_counts"], dict(total=306, formal=268, preparation=38)
            )


if __name__ == "__main__":
    unittest.main()
