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
    def test_request_programs_run_in_order_without_restarting_services(self):
        from rtp_llm.test.smoke.kimi_k3 import audit_k31_full_smoke, k31_full_smoke, k31_long_prefix_smoke

        for regular_code in (0, 1):
            with self.subTest(regular_code=regular_code), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                config, = smoke.stage_configs(base_config())
                config["prefill"].update(repo_root=directory, server=str(root / "server"))
                config["source_commit"] = "a" * 40
                (root / "server").touch()
                control = root / "control.json"
                control.write_text(json.dumps(config))
                calls = []

                def client(phase, argv):
                    calls.append((phase, argv))
                    output = Path(argv[argv.index("--output") + 1])
                    (output.parent if phase == "regular" else output).mkdir(parents=True)
                    return regular_code if phase == "regular" else 0

                with (patch.object(smoke.launcher, "validate", side_effect=lambda value: value),
                      patch.object(smoke.launcher, "run_dir", return_value=root),
                      patch.object(k31_full_smoke, "main", side_effect=lambda argv: client("regular", argv)),
                      patch.object(k31_long_prefix_smoke, "main", side_effect=lambda argv: client("long", argv)),
                      patch.object(audit_k31_full_smoke, "read_events", return_value={rank: [] for rank in range(8)})):
                    self.assertEqual(smoke.run_requests(control), regular_code)
                self.assertEqual([phase for phase, _ in calls], ["regular", "long"] if not regular_code else ["regular"])
                self.assertEqual(calls[0][1].count("--decode-role-addr"), 2)
                if not regular_code:
                    self.assertEqual(calls[1][1].count("--decode-role-addr"), 1)
                    self.assertTrue((root / smoke.REQUEST_EVIDENCE / "long/event-end.json").is_file())

    def test_phase_audit_does_not_mix_regular_and_long_events(self):
        from rtp_llm.test.smoke.kimi_k3.audit_k31_full_smoke import read_events

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [dict(time_ns=n, event="mtp_target_prefill_forward", request_ids=[n]) for n in (1, 2)]
            (root / "main_0.log").write_text("".join("[K3_SMOKE_EVENT] " + json.dumps(row) + "\n" for row in rows))
            (root / "engine.log").touch()
            boundary = {"0": dict(count=1, last=rows[0])}
            self.assertEqual(read_events(root, root / "engine.log", (0,), end=boundary)[0], (rows[0],))
            self.assertEqual(read_events(root, root / "engine.log", (0,), start=boundary)[0], (rows[1],))
            boundary["0"]["last"] = rows[1]
            with self.assertRaisesRegex(ValueError, "changed across"):
                read_events(root, root / "engine.log", (0,), end=boundary)

    def test_crc_audit_checks_all_rank_slots_and_the_selected_transfer_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "logs").mkdir()
            path = root / "logs/engine.log"
            rows = [
                f"[RANK {rank}] BlockTree CRC32C enabled: device={rank} "
                "batch=4 slots=2 device_total=428.62 MiB\n"
                for rank in range(8)
            ]
            path.write_text("".join(rows))
            self.assertEqual(len(smoke.audit_crc_workspace(root, 1, 4)), 8)
            with self.assertRaisesRegex(ValueError, "CRC transfer batch"):
                smoke.audit_crc_workspace(root, 1, 8)
            path.write_text("".join(rows[:-1]))
            with self.assertRaisesRegex(ValueError, "eight enabled CRC receipts"):
                smoke.audit_crc_workspace(root, 1, 4)

    def test_service_ports_cannot_race_with_outgoing_ephemeral_connections(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'ip_local_port_range').write_text('32768 60999\n')
            (root/'ip_local_reserved_ports').write_text('')
            self.assertTrue(smoke.launcher.service_port_preflight(30400, root)['passed'])
            with self.assertRaisesRegex(ValueError, 'overlaps unreserved ephemeral'):
                smoke.launcher.service_port_preflight(60400, root)
            (root/'ip_local_reserved_ports').write_text('60400-60471,32000')
            self.assertTrue(smoke.launcher.service_port_preflight(60400, root)['passed'])
            (root/'ip_local_reserved_ports').write_text('60400-60470')
            with self.assertRaisesRegex(ValueError, '60471'):
                smoke.launcher.service_port_preflight(60400, root)

    def test_isolated_probe_rejects_wrong_chunk_or_baseline_mode(self):
        base = {**base_config(), 'last_n_checkpoints': 5, 'last_n_tier_probe': 'HOST'}
        with self.assertRaisesRegex(ValueError, '2K chunks'):
            smoke.stage_configs(base, ('regular',))
        base['prefill']['prefill_token_capacity'] = 2048
        self.assertEqual(smoke.stage_configs(base, ('regular',))[0]['last_n_tier_probe'], 'HOST')
        base['last_n_calibration'] = True
        with self.assertRaisesRegex(ValueError, 'no calibration'):
            smoke.stage_configs(base, ('regular',))

    def test_isolated_probe_cannot_reuse_a_64k_precision_contract(self):
        import time
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference, contract = root/'reference.json', root/'contract.json'
            reference.write_text('{}')
            contract.write_text(json.dumps(dict(schema=1, source_commit='a'*40,
                                created_ns=time.time_ns()-1, prefill_chunk_tokens=65536)))
            base = {**base_config(), 'last_n_checkpoints': 5, 'last_n_tier_probe': 'DEVICE',
                    'source_commit': 'a'*40, 'precision_reference': str(reference),
                    'precision_contract': str(contract)}
            base['prefill']['prefill_token_capacity'] = 2048
            with patch.object(smoke, 'run_stage') as launch:
                with self.assertRaisesRegex(ValueError, 'same Prefill chunk'):
                    smoke.run(base, root/'output', 1, ('regular',))
                launch.assert_not_called()

    def test_last_five_keeps_304_slots_and_tp8_endpoints(self):
        base = {**base_config(), "last_n_checkpoints": 5}
        configs = smoke.stage_configs(base, ("regular",))
        self.assertEqual(len(configs), 1)
        config = configs[0]
        self.assertEqual(config["decode_dp"], 1)
        self.assertEqual(config["prefill"]["test_block_num"], 1032)
        self.assertEqual(config["decode"]["test_block_num"], 1032)
        self.assertEqual(config["prefill"]["linear_pool_blocks"], 96)
        self.assertEqual(config['decode']['linear_pool_blocks'], 512)
        self.assertEqual(config["prefill"]["memory_cache_size_mb"], 8192)
        self.assertEqual(config["prefill"]["disk_cache_size_mb"], 65536)
        args = smoke.request_args(config, Path("/data/runs/test"))
        self.assertEqual(args[args.index("--last-n-checkpoints")+1], "5")
        with self.assertRaisesRegex(ValueError, "304 regular slots"):
            smoke.stage_configs(base)
        with self.assertRaisesRegex(ValueError, "TP8/EP8"):
            smoke.stage_configs({**base, "decode_dp": 2}, ("regular",))
        with self.assertRaisesRegex(ValueError, "Last-5"):
            smoke.stage_configs({**base, "last_n_checkpoints": 1}, ("regular",))

    def test_last5_cannot_launch_without_prelocked_precision_inputs(self):
        base = {**base_config(), 'last_n_checkpoints': 5}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'run'
            with patch.object(smoke, 'run_stage') as launch, self.assertRaisesRegex(ValueError, 'precision_reference'):
                smoke.run(base, output, 1, ('regular',))
            launch.assert_not_called()
            self.assertFalse(output.exists())

    def test_calibration_reuses_six_inputs_without_reporting_a_complete_smoke(self):
        base = {**base_config(), 'last_n_checkpoints': 5, 'last_n_calibration': True}
        stage = smoke.stage_configs(base, ('regular',))[0]
        self.assertEqual(stage['prefill']['linear_pool_blocks'], 256)
        self.assertEqual(stage['decode']['linear_pool_blocks'], 64)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'calibration'
            receipt = dict(passed=True, cleanup_complete=True,
                           request_counts=dict(total=6, formal=2, preparation=4))
            with patch.object(smoke, 'run_stage', return_value=receipt):
                self.assertEqual(smoke.run(base, output, 1, ('regular',)), 0)
            result = json.loads((output/'RESULT.json').read_text())
            self.assertEqual(result['request_counts']['total'], 6)
            self.assertIs(result['complete_smoke_passed'], False)

    def test_runtime_jit_compilers_are_available_paths_and_explicit_choices_survive(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory)/'server'
            (Path(str(binary)+'.runfiles')/'pip_gpu_cuda13_torch_torch/site-packages/torch/lib').mkdir(parents=True)
            with patch.dict(smoke.launcher.os.environ, {}, clear=True), \
                 patch.object(smoke.launcher.shutil, 'which', side_effect=lambda command: '/usr/bin/'+command):
                env = smoke.launcher.runtime_environment(Path(directory), binary)
            self.assertEqual((env['CC'], env['CXX']), ('/usr/bin/gcc', '/usr/bin/g++'))
            with patch.dict(smoke.launcher.os.environ, dict(CC='/opt/compiler/gcc', CXX='/opt/compiler/g++'), clear=True):
                env = smoke.launcher.runtime_environment(Path(directory), binary)
            self.assertEqual(env['CC'], '/opt/compiler/gcc')

    def test_regular_only_keeps_ci_topology_and_excludes_one_million_stage(self):
        base = base_config()
        configs = smoke.stage_configs(base, ("regular",))
        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0]["k31_smoke_stage"], "regular")
        self.assertEqual(configs[0]["decode_dp"], 2)
        self.assertEqual(configs[0]["request_stages"], ["regular"])
        with self.assertRaisesRegex(ValueError, "DP2/TP4"):
            smoke.stage_configs({**base, "decode_dp": 1}, ("regular",))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result"
            record = dict(passed=True, cleanup_complete=True,
                          request_counts=dict(total=304, formal=267, preparation=37))
            record["stages"] = {"regular": dict(passed=True, request_counts=record["request_counts"])}
            with patch.object(smoke, "run_stage", return_value=record) as run_stage:
                self.assertEqual(smoke.run(base, output, 1, ("regular",)), 0)
            run_stage.assert_called_once()
            result = json.loads((output / "RESULT.json").read_text())
            self.assertEqual(list(result["stages"]), ["regular"])
            self.assertEqual(result["request_counts"], record["request_counts"])

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
            regular, = smoke.stage_configs(base_config())
        with patch.dict(os.environ, KDA_NVFP4="1", DSA_NVFP4="1"):
            received = smoke.launcher.validate(regular)
            self.assertFalse(received["kda_nvfp4"])
            self.assertFalse(received["dsa_nvfp4"])
        with patch.dict(os.environ, KDA_NVFP4="0", DSA_NVFP4="1"):
            regular, = smoke.stage_configs(base_config())
        with patch.dict(os.environ, KDA_NVFP4="1", DSA_NVFP4="0"):
            received = smoke.launcher.validate(regular)
            self.assertFalse(received["kda_nvfp4"])
            self.assertTrue(received["dsa_nvfp4"])
        for key in ("kda_nvfp4", "dsa_nvfp4"):
            for value in (None, 1, "0"):
                with self.assertRaisesRegex(ValueError, "must be boolean"):
                    smoke.stage_configs({**base_config(), key: value})

    def test_precision_audit_requires_executed_fp4_and_native_draft(self):
        from rtp_llm.test.smoke.kimi_k3.nvfp4_evidence import audit_projection_precision, MARKER

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

    def test_default_phases_share_the_ci_capacity_and_one_service_lifecycle(self):
        base = base_config()
        regular, = smoke.stage_configs(base)
        self.assertEqual(base, base_config())
        self.assertEqual(regular["run_id"], "test-k31")
        self.assertEqual(regular["request_stages"], ["regular", "long"])
        self.assertEqual(smoke.stage_configs(base, ("long", "regular")), [regular])
        self.assertEqual(regular["decode_dp"], 2)
        self.assertEqual(regular["prefill"]["test_block_num"], 140)
        self.assertEqual(regular["decode"]["test_block_num"], 272)
        self.assertEqual(regular["prefill"]["linear_pool_blocks"], 128)
        self.assertEqual(regular["prefill"]["reuse_cache_tail_blocks"], 5)
        self.assertEqual(regular["prefill"]["prefill_token_capacity"], 65536)
        for role in ("prefill", "decode"):
            self.assertEqual(regular[role]["max_seq_len"], 1_052_672)
        self.assertEqual(regular["decode"]["linear_pool_blocks"], 192)
        self.assertEqual(regular["prefill_kv_workspace_budget_gib"], 8)
        for stage in (regular,):
            self.assertIs(stage["target_only"], False)
            self.assertIs(stage["fp8_gemm"], True)
            self.assertIs(stage["fp8_kv_cache"], False)
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

    def test_shared_stage_entry_keeps_last_five_geometry_for_ci_callers(self):
        base = {**base_config(), "last_n_checkpoints": 5}
        shared = smoke.launcher.validate(smoke.stage_config(base, "regular"))
        traditional, = smoke.stage_configs(base, ("regular",))
        traditional.pop("request_argv")
        self.assertEqual(shared, traditional)
        self.assertEqual(shared["prefill_kv_workspace_budget_gib"], 8)
        with self.assertRaisesRegex(ValueError, "304 regular slots"):
            smoke.stage_config(base, "long")

    def test_shared_stage_entry_preserves_long_budget_and_explicit_override(self):
        base = base_config()
        self.assertEqual(
            smoke.stage_config(base, "long")["prefill_kv_workspace_budget_gib"], 8
        )
        base["prefill_kv_workspace_budget_gib"] = 4.5
        self.assertEqual(smoke.stage_config(base, "long")["prefill_kv_workspace_budget_gib"], 4.5)

    def test_default_phases_require_native_mtp(self):
        with self.assertRaisesRegex(ValueError, "native MTP"):
            smoke.stage_configs({**base_config(), "target_only": True})

    def test_default_phases_require_rdma(self):
        for stage in smoke.stage_configs(base_config()):
            self.assertIs(stage["cache_store_rdma_mode"], True)
        with self.assertRaisesRegex(ValueError, "requires RDMA"):
            smoke.stage_configs({**base_config(), "cache_store_rdma_mode": False})

    def test_stage_configs_record_effective_fp8_defaults(self):
        base = base_config()
        del base["fp8_gemm"], base["fp8_kv_cache"]
        for stage in smoke.stage_configs(base):
            self.assertIs(stage["fp8_gemm"], True)
            self.assertIs(stage["fp8_kv_cache"], True)
        last5, = smoke.stage_configs({**base, "last_n_checkpoints": 5}, ("regular",))
        self.assertIs(last5["fp8_gemm"], True)
        self.assertIs(last5["fp8_kv_cache"], True)

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
        config, = smoke.stage_configs({**base_config(), "prefill_kv_workspace_budget_gib": 8})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "launch").mkdir()
            ready = {}
            for role in ("prefill", "decode"):
                root = output / role
                logs = root / "logs"
                logs.mkdir(parents=True)
                (root / "control.json").write_text(json.dumps(config))
                (root / "rdma-preflight.json").write_text('{"cache_store_transport": "rdma"}')
                (logs / "engine.log").write_text(
                    "".join(f"[RANK {rank}] rdma messager init success\n" for rank in range(8))
                )
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
            self.assertEqual(proof["decode"]["rdma_ranks"], list(range(8)))
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
            (output / "prefill" / "logs" / "engine.log").write_text(
                "".join(f"[RANK {rank}] rdma messager init success\n" for rank in range(7))
            )
            with self.assertRaisesRegex(ValueError, "prefill: expected native RDMA on all eight ranks"):
                smoke.audit_startup(config, output)

    def test_routes_follow_existing_decode_owner_endpoint_stride(self):
        regular, = smoke.stage_configs(base_config())
        long_case = {**regular, "k31_smoke_stage": "long"}
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
        config, = smoke.stage_configs(base_config())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)

            def host_ssh(argv, *, stdout, stderr, **kwargs):
                self.assertEqual(argv[-2], "prefill")
                command = shlex.split(argv[-1])
                self.assertEqual(command[0:2], ["bash", "-c"])
                self.assertEqual(command[4], "/data/runs/test-k31/prefill")
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
        config, = smoke.stage_configs(base_config())
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

    def test_failed_phase_cannot_be_hidden_by_the_other_phase(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result"
            record = dict(passed=False, cleanup_complete=True, errors=["regular audit failed"],
                          stages={"regular": dict(passed=False), "long": dict(passed=True,
                                  request_counts=dict(total=2, formal=1, preparation=1))})
            with patch.object(smoke, "run_stage", return_value=record) as run_stage:
                self.assertEqual(smoke.run(base_config(), output, startup_timeout=1), 1)
            run_stage.assert_called_once()
            result = json.loads((output / "RESULT.json").read_text())
            self.assertIs(result["passed"], False)
            self.assertIn("regular audit failed", result["stages"]["regular"]["errors"])
            self.assertIs(result["stages"]["long"]["passed"], False)

    def test_aggregate_counts_only_pass_after_both_audits(self):
        from rtp_llm.test.smoke.kimi_k3.k31_full_smoke import cache_pressure_plan, request_budget

        plan = cache_pressure_plan(139, 8192)
        regular_budget = request_budget("smoke", plan["requests"])
        self.assertEqual(regular_budget, dict(total=309, formal=267, preparation=42))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result"
            stages = [
                {
                    "passed": True,
                    "cleanup_complete": True,
                    "request_counts": regular_budget,
                },
                {
                    "passed": True,
                    "cleanup_complete": True,
                    "request_counts": dict(total=2, formal=1, preparation=1),
                },
            ]
            record = dict(passed=True, cleanup_complete=True,
                          stages=dict(zip(smoke.STAGES, stages)))
            with patch.object(smoke, "run_stage", return_value=record) as run_stage:
                self.assertEqual(smoke.run(base_config(), output, startup_timeout=1), 0)
            run_stage.assert_called_once()
            result = json.loads((output / "RESULT.json").read_text())
            self.assertIs(result["passed"], True)
            self.assertEqual(result["server_lifecycles"], 1)
            self.assertEqual(
                result["request_counts"], dict(total=311, formal=268, preparation=43)
            )


if __name__ == "__main__":
    unittest.main()
