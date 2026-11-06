"""CPU-only integration contracts; no accelerator allocation or compiled ops."""

import copy
import dataclasses
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from rtp_llm.config.v41_runtime import V41RuntimeConfig, select_v41_backend
from rtp_llm.tools.v41.launch import preset_env
from rtp_llm.tools.v41.preflight import (
    WHEELS,
    checkpoint_metadata,
    compare_reports,
    rdma_inventory,
    service_ports,
)


class IntegrationTest(unittest.TestCase):
    def test_encoder_recovery_is_explicit_and_requires_bounded_decoder_replay(self):
        with self.assertRaisesRegex(ValueError, "requires --ced-mode bounded"):
            preset_env("prefill", "/checkpoint", parent={}, encoder_swa_replay=True)
        for role in ("prefill", "decode"):
            env = preset_env(
                role,
                "/checkpoint",
                parent={},
                ced_mode="bounded",
                encoder_swa_replay=True,
            )
            runtime = V41RuntimeConfig.from_env(env)
            self.assertTrue(runtime.encoder_replay)
            self.assertTrue(runtime.bounded_replay)
            self.assertEqual(env["REUSE_CACHE"], "1")
            env["DSV41_SWA_BOUNDED_REPLAY"] = "0"
            with self.assertRaisesRegex(ValueError, "encoder SWA recovery requires"):
                V41RuntimeConfig.from_env(env)

    def test_offline_dependency_checks_pin_template_and_all_files(self):
        from rtp_llm.tools.v41.native_build_precheck import staged_vipserver

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root, source = base / "snapshot", base / "vipserver"
            (root / "internal_source").mkdir(parents=True)
            (root / "3rdparty/vipserver").mkdir(parents=True)
            source.mkdir()
            pin = {"name": "vipserver", "remote": "test-source", "commit": "fixed-pin"}
            declaration = (
                "new_git_repository("
                + ",".join(f"{key}={value!r}" for key, value in pin.items())
                + ")"
            )
            (root / "internal_source/impls.bzl").write_text(declaration)
            (root / "3rdparty/vipserver/vipserver.BUILD").write_text("# template\n")
            (source / "BUILD.bazel").write_text("# template\n")
            (source / "WORKSPACE").write_text("# workspace\n")
            files = {
                path.name: {
                    "kind": "file",
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "executable": False,
                }
                for path in source.iterdir()
            }
            manifest = dict(pin, files=files)
            metadata = source / ".v41_dependency.json"
            metadata.write_text(json.dumps(manifest))
            self.assertEqual(staged_vipserver(root, source)["commit"], "fixed-pin")
            (source / "WORKSPACE").write_text("modified")
            with self.assertRaisesRegex(ValueError, "file hashes"):
                staged_vipserver(root, source)
            (source / "WORKSPACE").write_text("# workspace\n")
            manifest["commit"] = "another-pin"
            metadata.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "declared pin"):
                staged_vipserver(root, source)
            metadata.write_text(json.dumps(dict(pin, files=files)))
            (root / "3rdparty/vipserver/vipserver.BUILD").write_text(
                "# changed template\n"
            )
            with self.assertRaisesRegex(ValueError, "committed template"):
                staged_vipserver(root, source)

    def test_build_precheck_preserves_idle_jvm_and_detects_real_client(self):
        from rtp_llm.tools.v41.native_build_precheck import owns_bazel_client

        paths = [Path("/task/cache"), Path("/task/output")]
        server = [
            "bazel(snapshot)",
            "-jar",
            "/task/install/A-server.jar",
            "--output_user_root=/task/cache",
            "--output_base=/task/output",
        ]
        self.assertFalse(owns_bazel_client(server, paths))
        self.assertTrue(
            owns_bazel_client(["bazel", "--output_base", "/task/output"], paths)
        )
        self.assertTrue(
            owns_bazel_client(["bazelisk", "--output_user_root=/task/cache"], paths)
        )
        self.assertFalse(
            owns_bazel_client(["bazel", "--output_base=/task/output-other"], paths)
        )

    def test_rdma_inventory_rejects_cgroup_denial_despite_file_permissions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            port = root / "sys/erdma_0/ports/1"
            port.mkdir(parents=True)
            (port / "state").write_text("4: ACTIVE")
            nodes = root / "dev"
            nodes.mkdir()
            (nodes / "uverbs0").touch(mode=0o666)
            with patch(
                "rtp_llm.tools.v41.preflight.os.open",
                side_effect=PermissionError("cgroup denied"),
            ):
                report = rdma_inventory(root / "sys", nodes)
            self.assertTrue(report["active_port"])
            self.assertEqual(report["accessible_uverbs"], [])
            self.assertIn(
                "cgroup denied", report["device_open_errors"][str(nodes / "uverbs0")]
            )

    def test_role_presets_fix_inherited_topology_and_strategies(self):
        for role, tp, dp, cp, port in (
            ("prefill", 8, 1, 8, 8090),
            ("decode", 1, 8, 1, 9090),
        ):
            with self.subTest(role=role):
                env = preset_env(
                    role,
                    "/weights",
                    parent={
                        "WORLD_SIZE": "16",
                        "TP_SIZE": "8",
                        "ENABLE_CUDA_GRAPH": "1",
                        "SP_TYPE": "dspark",
                        "DSV41_CED": "1",
                        "REUSE_CACHE": "1",
                        "GRPC_CLIENT_CHANNEL_BACKUP_POLL_INTERVAL_MS": "5000",
                    },
                )
                runtime = V41RuntimeConfig.from_env(env)
                self.assertEqual(
                    (runtime.physical_tp, runtime.attention_dp, runtime.attention_cp),
                    (tp, dp, cp),
                )
                self.assertEqual(runtime.attention_tp, 1)
                self.assertEqual(runtime.source_cp, 8)
                self.assertEqual(env["START_PORT"], str(port))
                self.assertEqual(env["REUSE_CACHE"], "0")
                self.assertEqual(env["CACHE_STORE_RDMA_MODE"], "0")
                self.assertEqual(env["DECODE_ENTRANCE"], "0")
                self.assertEqual(
                    env["GRPC_CLIENT_CHANNEL_BACKUP_POLL_INTERVAL_MS"], "500"
                )
        self.assertEqual(
            preset_env("decode", "/weights", rdma=True)["CACHE_STORE_RDMA_MODE"], "1"
        )

    def test_reject_unsupported_release_configurations(self):
        runtime = V41RuntimeConfig.from_env(preset_env("decode", "/weights", parent={}))
        mutations = (
            {"world_size": 16},
            {"physical_tp": 8},
            {"source_cp": 1},
            {"attention_dp": 1},
            {"ep": 1},
            {"enp": 1},
            {"max_seq_len": -1},
            {"max_seq_len": 0},
            {"kv_sharded": True},
            {"cuda_graph": True},
            {"speculative": "MTP"},
            {"speculative": "DSPARK", "cuda_graph": True},
            {"eplb": "DYNAMIC"},
            {"low_latency": True},
            {"bounded_replay": True},
            {"cp_method": "ALL_GATHER"},
            {"strategy": "mega"},
            {"redundant_experts": 1},
            {"normal_ep": False},
            {"all_gather_moe": True},
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                dataclasses.replace(runtime, **mutation).validate()

    def test_long_context_runtime_accepts_explicit_and_inherited_lengths(self):
        for role in ("prefill", "decode"):
            for parallel in ("cp", "dp"):
                for length in (32769, 65536, 131072, 262144):
                    with self.subTest(role=role, parallel=parallel, length=length):
                        options = dict(prefill_parallel=parallel)
                        inherited = preset_env(
                            role,
                            "/weights",
                            parent={"MAX_SEQ_LEN": str(length)},
                            **options,
                        )
                        explicit = preset_env(
                            role,
                            "/weights",
                            parent={"MAX_SEQ_LEN": "2048"},
                            max_seq_len=length,
                            **options,
                        )
                        self.assertEqual(
                            V41RuntimeConfig.from_env(inherited).max_seq_len, length
                        )
                        self.assertEqual(
                            V41RuntimeConfig.from_env(explicit).max_seq_len, length
                        )

    def test_dspark_presets_keep_role_budget_and_existing_topologies(self):
        for gamma in (1, 3, 5):
            for parallel in ("cp", "dp"):
                for role in ("prefill", "decode"):
                    env = preset_env(
                        role,
                        "/weights",
                        parent={},
                        dspark_gamma=gamma,
                        dspark_confidence="bounded",
                        dspark_verify_mode="packed",
                        dspark_verify_budget=gamma // 2,
                        dspark_budget_mode="adaptive",
                        prefill_parallel=parallel,
                    )
                    runtime = V41RuntimeConfig.from_env(env)
                    self.assertEqual(runtime.speculative, "DSPARK")
                    self.assertEqual(env["GEN_NUM_PER_CIRCLE"], str(gamma))
                    self.assertEqual(env["SP_MODEL_TYPE"], "deepseek_v41_dspark")
                    self.assertEqual(
                        env["SP_DSPARK_VERIFY_MODE"],
                        "packed" if role == "decode" else "dense",
                    )
                    self.assertEqual(env["ENABLE_CUDA_GRAPH"], "0")

    def test_launcher_prints_the_effective_long_context_capacity(self):
        from rtp_llm.tools.v41.launch import main

        common = [
            "launch",
            "prefill",
            "--checkpoint-path",
            "/weights",
            "--print-command",
        ]
        for extra, expected in (([], "65536"), (["--max-seq-len", "131072"], "131072")):
            with self.subTest(extra=extra), patch.dict(
                os.environ, {"MAX_SEQ_LEN": "65536"}, clear=True
            ), patch.object(sys, "argv", common + extra), redirect_stdout(
                io.StringIO()
            ) as stream:
                self.assertEqual(main(), 0)
            self.assertIn(f"MAX_SEQ_LEN={expected}", stream.getvalue())

    def test_pd_handoff_accepts_long_matching_history_and_rejects_invalid_lengths(self):
        from rtp_llm.models_py.modules.dsv41.ppu.pd_handoff import (
            PROTOCOL,
            validate_handoff,
        )

        identity = {"snapshot_id": "long-context-test"}
        pd = {
            "protocol": PROTOCOL,
            "identity": identity,
            "model_signature": "model",
            "run_id": "run",
            "owner": 0,
            "target_dp": 8,
            "target_tp": 1,
            "source_layout": "replicated-logical-cp-cache",
            "histories": {"0": [2] * 131072},
            "seed_tokens": {"0": 3},
            "owner_map": {"0": 0},
        }
        manifest = {
            "pool_layouts": {"pool": {}},
            "meta": {"pd": pd},
            "source_cp": 8,
            "producer_snapshot": {"snapshot": identity["snapshot_id"]},
            "lengths": {"0": 131072},
        }
        options = dict(
            identity=identity, model_signature="model", run_id="run", owner=0, world=8
        )
        self.assertIs(validate_handoff(manifest, **options), pd)
        for length in (131071, 0, -1):
            manifest["lengths"]["0"] = length
            with self.subTest(length=length), self.assertRaisesRegex(
                ValueError, "history length"
            ):
                validate_handoff(manifest, **options)

    def test_pd_check_only_accepts_long_context_arguments(self):
        from rtp_llm.tools.v41.pd_joint import main

        with tempfile.TemporaryDirectory() as root, patch(
            "rtp_llm.models_py.modules.dsv41.ppu.pd_handoff.runtime_identity",
            return_value={"snapshot_id": "context-check"},
        ), redirect_stdout(io.StringIO()) as stream:
            self.assertEqual(
                main(
                    [
                        "prefill",
                        "--run-id",
                        "long-context",
                        "--synthetic",
                        "--tokens",
                        "131072",
                        "--steps",
                        "129",
                        "--run-root",
                        root,
                        "--check-only",
                    ]
                ),
                0,
            )
        report = json.loads(stream.getvalue())
        self.assertEqual(report["input_tokens_per_request"], 131072)
        self.assertEqual(report["generated_tokens_per_request"], 130)

    def test_dp8_prefill_and_decode_use_unsharded_cp1_source(self):
        for role in ("prefill", "decode"):
            with self.subTest(role=role):
                env = preset_env(role, "/weights", parent={}, prefill_parallel="dp")
                runtime = V41RuntimeConfig.from_env(env)
                self.assertEqual(
                    (runtime.physical_tp, runtime.attention_dp, runtime.attention_cp),
                    (1, 8, 1),
                )
                self.assertEqual(
                    (runtime.ep, runtime.enp, runtime.source_cp), (8, 8, 1)
                )
                self.assertEqual(runtime.cp_method, "DISABLED")
                self.assertFalse(runtime.kv_sharded)
                with self.assertRaises(ValueError):
                    dataclasses.replace(runtime, source_cp=8).validate()
        with self.assertRaisesRegex(ValueError, "prefill_parallel"):
            preset_env("prefill", "/weights", prefill_parallel="typo")

    def test_dp8_prefill_routes_to_every_p_and_d_owner(self):
        for role in ("prefill", "decode"):
            env = preset_env(
                role,
                "/weights",
                prefill_parallel="dp",
                prefill_address="192.0.2.10:18090",
                decode_address="192.0.2.20:19090",
                parent={},
            )
            endpoints = json.loads(env["MODEL_SERVICE_CONFIG"])["role_endpoints"][0]
            for name, host, port in (
                ("prefill", "192.0.2.10", 18090),
                ("decode", "192.0.2.20", 19090),
            ):
                self.assertEqual(
                    endpoints[f"{name}_endpoint"]["address"].split(","),
                    [f"{host}:{port + rank * 9}" for rank in range(8)],
                )

    def test_parsed_dp_prefill_topology_uses_actual_server_values(self):
        config = NS(
            parallelism_config=NS(
                world_size=8,
                local_world_size=8,
                tp_size=1,
                dp_size=8,
                ep_size=8,
                prefill_cp_config=NS(
                    method="DISABLED", prefill_cp_size=1, kv_cache_sharded=False
                ),
            ),
            role_config=NS(role_type="PREFILL"),
            model_args=NS(max_seq_len=32768),
            py_hw_kernel_config=NS(enable_cuda_graph=False),
            sp_config=NS(type="NONE"),
            eplb_config=NS(eplb_mode="NONE", redundant_expert=0),
            moe_config=NS(
                use_deepep_low_latency=False, use_deepep_moe=True, use_all_gather=False
            ),
        )
        with patch.dict(os.environ, {"TP_SIZE": "8", "DP_SIZE": "1"}, clear=True):
            runtime = V41RuntimeConfig.from_server(config)
        self.assertEqual((runtime.attention_dp, runtime.attention_cp), (8, 1))

    def test_decode_graph_requires_explicit_megamoe_w4a4(self):
        env = preset_env("decode", "/weights", parent={})
        env.update(ENABLE_CUDA_GRAPH="1", RTP_V41_PPU_MOE_BACKEND="megamoe")
        runtime = V41RuntimeConfig.from_env(env)
        self.assertTrue(runtime.cuda_graph)
        self.assertEqual(
            (runtime.moe_backend, runtime.moe_precision), ("megamoe", "w4a4")
        )
        with self.assertRaisesRegex(ValueError, "PPU Graph"):
            dataclasses.replace(runtime, moe_backend="grouped").validate()
        with self.assertRaisesRegex(ValueError, "PPU Graph"):
            dataclasses.replace(runtime, moe_precision="w4a8").validate()
        env = preset_env("prefill", "/weights", parent={})
        env.update(ENABLE_CUDA_GRAPH="1", RTP_V41_PPU_MOE_BACKEND="megamoe")
        with self.assertRaisesRegex(ValueError, "PPU Graph"):
            V41RuntimeConfig.from_env(env)

    def test_native_pd_route_preserves_http_endpoints_and_peer_discovery(self):
        options = dict(
            prefill_address="192.0.2.10:18090",
            decode_address="192.0.2.20:19090",
            parent={},
        )
        for role, local_port, peer_host, peer_port in (
            ("prefill", "18090", "192.0.2.20", "19090"),
            ("decode", "19090", "192.0.2.10", "18090"),
        ):
            env = preset_env(role, "/weights", **options)
            self.assertEqual(env["START_PORT"], local_port)
            self.assertEqual(env["REMOTE_RPC_SERVER_IP"], peer_host)
            self.assertEqual(env["REMOTE_SERVER_PORT"], peer_port)
            route = json.loads(env["MODEL_SERVICE_CONFIG"])
            self.assertTrue(route["use_local"])
            group = route["role_endpoints"][0]
            self.assertEqual(
                group["prefill_endpoint"]["address"], options["prefill_address"]
            )
            self.assertEqual(
                group["decode_endpoint"]["address"].split(","),
                [f"192.0.2.20:{19090 + rank * 9}" for rank in range(8)],
            )
            self.assertEqual(env["WORKER_INFO_PORT_NUM"], "9")
            self.assertEqual(env["WARM_UP"], "0")
            self.assertEqual(env["MODEL_WARM_UP"], "0")
        for mutation in (
            {"prefill_address": None},
            {"decode_address": "http://host"},
            {"start_port": 9090},
        ):
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                preset_env("decode", "/weights", **dict(options, **mutation))

    def test_parsed_values_take_precedence_over_environment(self):
        config = NS(
            parallelism_config=NS(
                world_size=8,
                local_world_size=8,
                tp_size=1,
                dp_size=8,
                ep_size=8,
                prefill_cp_config=NS(
                    method="PREFILL_CP", prefill_cp_size=8, kv_cache_sharded=False
                ),
            ),
            role_config=NS(role_type="DECODE"),
            model_args=NS(max_seq_len=32768),
            py_hw_kernel_config=NS(enable_cuda_graph=False),
            sp_config=NS(type="NONE"),
            eplb_config=NS(eplb_mode="NONE", redundant_expert=0),
            moe_config=NS(
                use_deepep_low_latency=False, use_deepep_moe=True, use_all_gather=False
            ),
        )
        with patch.dict(os.environ, {"TP_SIZE": "8", "DP_SIZE": "1"}, clear=True):
            self.assertEqual(V41RuntimeConfig.from_server(config).attention_dp, 8)
            config.parallelism_config.tp_size = 8
            with self.assertRaises(ValueError):
                V41RuntimeConfig.from_server(config)
            config.parallelism_config.tp_size = 1
            config.model_args.max_seq_len = None
            with self.assertRaisesRegex(ValueError, "explicit"):
                V41RuntimeConfig.from_server(config)

    def test_device_selection_has_no_implicit_fallback(self):
        self.assertEqual(select_v41_backend("Ppu", {}), "ppu")
        self.assertEqual(select_v41_backend("Cuda", {}), "cuda")
        self.assertEqual(
            select_v41_backend("Cpu", {"RTP_V41_BACKEND": "reference"}), "reference"
        )
        for device, env in (
            ("Cpu", {}),
            ("ROCm", {}),
            ("Ppu", {"RTP_V41_BACKEND": "cuda"}),
        ):
            with self.subTest(device=device), self.assertRaises(RuntimeError):
                select_v41_backend(device, env)
        with self.assertRaises(ValueError):
            select_v41_backend("Ppu", {"RTP_V41_BACKEND": "typo"})

    def test_port_coverage_includes_negative_dp_rendezvous(self):
        ports = service_ports(9090, dp_size=8)
        for rank in range(8):
            for offset in (0, 1, 2, 4, 5, 7, 8):
                self.assertIn(9090 + rank * 9 + offset, ports)
            self.assertIn(9090 - rank * 11 - 2, ports)
            self.assertIn(9090 - rank * 11 - 5, ports)
        self.assertIn(9090 - 11, ports)
        with self.assertRaises(ValueError):
            service_ports(65530)

    def test_checkpoint_metadata_does_not_claim_weight_content_signature(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(
                json.dumps(
                    {
                        "text_config": {
                            "n_routed_experts": 384,
                            "num_experts_per_tok": 6,
                            "engram_head_dim": 256,
                            "engram_num_embeddings": [1000, 2000],
                        }
                    }
                )
            )
            signature = hashlib.sha256((root / "config.json").read_bytes()).hexdigest()
            (root / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {"total_size": 16},
                        "weight_map": {"weight": "one.safetensors"},
                    }
                )
            )
            (root / "tokenizer.json").write_text("{}")
            (root / "one.safetensors").write_bytes(b"fixture")
            result = checkpoint_metadata(root, signature)
            self.assertEqual(result["engram_table_bytes"], 3000 * 264)
            self.assertEqual(result["full_weight_signature"], "not_measured")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                checkpoint_metadata(root, "invalid")
            (root / "one.safetensors").unlink()
            with self.assertRaisesRegex(ValueError, "shards missing"):
                checkpoint_metadata(root, signature)

    def test_environment_comparison_rejects_mismatches_and_missing_identity(self):
        checks = [
            {"name": "sdk", "status": "pass", "details": "sdk"},
            {
                "name": "checkpoint_metadata",
                "status": "pass",
                "details": {"config": "same"},
            },
            {"name": "topology", "status": "pass", "details": {"role": "PREFILL"}},
        ]
        checks.extend(
            {"name": f"wheel:{name}", "status": "pass", "details": {"version": version}}
            for name, version in WHEELS.items()
        )
        checks.append(
            {
                "name": "native_cache_protocol",
                "status": "pass",
                "details": {"version": 2},
            }
        )
        left = {
            "checks": checks,
            "prerequisites_ok": True,
            "moe_precision": "w4a8",
            "moe_backend": "grouped",
        }
        right = copy.deepcopy(left)
        right["moe_backend"] = "megamoe"
        self.assertFalse(compare_reports(left, right)["identities_match"])
        right["moe_backend"] = "grouped"
        right["checks"][2]["details"]["role"] = "DECODE"
        self.assertTrue(compare_reports(left, right)["identities_match"])
        right["moe_precision"] = "w4a4"
        self.assertFalse(compare_reports(left, right)["identities_match"])
        right["moe_precision"] = "w4a8"
        right["checks"][-1]["details"]["version"] = 1
        self.assertFalse(compare_reports(left, right)["identities_match"])
        right["checks"][-1]["details"]["version"] = 2
        right["checks"][1]["details"]["config"] = "different"
        self.assertFalse(compare_reports(left, right)["identities_match"])
        right["checks"][1]["status"] = "fail"
        self.assertFalse(compare_reports(left, right)["identities_match"])

    def test_dispatch_and_registry_in_fresh_process_without_nvidia_imports(self):
        script = r"""
import importlib.abc
import sys
import types
from enum import IntEnum
class DeviceType(IntEnum):
    Cpu = 0
    Cuda = 1
    Ppu = 5
device = types.ModuleType("rtp_llm.device.device_type")
device.DeviceType = DeviceType
device.get_device_type = lambda: DeviceType.Ppu
sys.modules[device.__name__] = device
blocked = ("deep_gemm", "tilelang", "rtp_llm.models_py.modules.dsv41.moe",
           "rtp_llm.models_py.model_desc.deepseek_v41_base_model",
           "rtp_llm.models_py.model_desc.deepseek_v41_cuda_model")
class RejectNvidia(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if any(fullname == p or fullname.startswith(p + ".") for p in blocked):
            raise AssertionError("eager NVIDIA import: " + fullname)
sys.meta_path.insert(0, RejectNvidia())
from rtp_llm.models_py.model_desc import deepseek_v41_model as entry
from rtp_llm.models_py.modules.dsv41 import backend as b
from rtp_llm.config.v41_runtime import PPU_STRATEGY
try:
    entry.get_v41_model_class()
except RuntimeError as error:
    assert "not integrated" in str(error), error
else:
    raise AssertionError("missing provider was accepted")
class FakeModel: pass
class FakeWeight: pass
ops = types.SimpleNamespace(**{name: lambda *a, **kw: None for name in b.OP_NAMES})
provider = b.V41Backend("ppu", PPU_STRATEGY, ops, FakeModel, FakeWeight, lambda runtime: None)
b.register_v41_backend(provider)
b.register_v41_backend(provider)
from rtp_llm.models_py.model_desc.deepseek_v41_model import DeepSeekV41Model
assert DeepSeekV41Model is FakeModel
try:
    b.register_v41_backend(b.V41Backend("ppu", PPU_STRATEGY, ops, FakeModel, FakeWeight, lambda _: None))
except RuntimeError: pass
else: raise AssertionError("duplicate provider accepted")
for mutation in ({"contract_version": 999}, {"ops": object()}, {"strategy": "mega"}):
    import dataclasses
    try: dataclasses.replace(provider, **mutation).validate()
    except ValueError: pass
    else: raise AssertionError("bad provider accepted: " + str(mutation))
class IncompleteOps(b.V41BackendOps): pass
try: dataclasses.replace(provider, ops=IncompleteOps()).validate()
except ValueError: pass
else: raise AssertionError("inherited Protocol placeholders accepted")
from rtp_llm.tools.v41.launch import preset_env
from rtp_llm.config.v41_runtime import V41RuntimeConfig
runtime = V41RuntimeConfig.from_env(preset_env("decode", "/weights"))
try: dataclasses.replace(provider, check_capabilities=lambda _: False).check_runtime(runtime)
except RuntimeError: pass
else: raise AssertionError("False capability result accepted")
assert not any(name in sys.modules for name in blocked)
"""
        subprocess.run([sys.executable, "-c", script], check=True)


if __name__ == "__main__":
    unittest.main()
