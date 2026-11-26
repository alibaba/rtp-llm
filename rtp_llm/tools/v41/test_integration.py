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

from rtp_llm.config.v41_runtime import (
    DSPARK_DENSE_DECODE_DEFAULTS,
    DSPARK_CROSS_ROUND_OVERLAP_ENV,
    DSPARK_OVERLAP_ENV,
    DSPARK_OVERLAP_FLAGS,
    DSPARK_SIMULATED_ACCEPT_ENV,
    MEGAMOE_DSPARK_ASYNC_FLAGS,
    V41RuntimeConfig,
    ZERO_OVERHEAD_ENV,
    select_v41_backend,
    validate_v41_server_config,
    zero_overhead_flags,
)
from rtp_llm.tools.v41.launch import preset_env
from rtp_llm.tools.v41.preflight import (
    WHEELS,
    checkpoint_metadata,
    compare_reports,
    host_memory_budget,
    local_cache_policy,
    rdma_inventory,
    service_ports,
)


class IntegrationTest(unittest.TestCase):
    def test_zero_overhead_scheduler_profile(self):
        options = dict(shared_decode_graph=True, dspark_gamma=3)
        parent = {"RTP_V41_PPU_MOE_BACKEND": "megamoe"}
        for switch in (None, "0", "1", "true"):
            with self.subTest(switch=switch):
                env = dict(parent)
                if switch is not None:
                    env[ZERO_OVERHEAD_ENV] = switch
                decode = preset_env("decode", "/weights", parent=env, **options)
                normalized = "0" if switch == "0" else "1"
                self.assertEqual(decode.get(ZERO_OVERHEAD_ENV), normalized)
                if normalized == "1":
                    for key in (
                        DSPARK_OVERLAP_ENV,
                        DSPARK_CROSS_ROUND_OVERLAP_ENV,
                        "RTP_V41_PPU_DSPARK_GPU_VERIFY_RECEIPT",
                        "RTP_V41_PPU_DSPARK_FOLD_PROPOSAL_SAMPLER",
                        "RTP_V41_PPU_VERIFY_DEFERRED_STAGE",
                    ):
                        self.assertEqual(decode[key], "1")
        env = {**parent, ZERO_OVERHEAD_ENV: "1"}
        self.assertEqual(
            zero_overhead_flags(env, role="PREFILL", speculative="DSPARK", graph=False,
                                single_host=True, gamma=3, verify_mode="dense", confidence="off"),
            {ZERO_OVERHEAD_ENV: "1"},
        )
        for overrides in ({"dspark_gamma": 0}, {"dspark_gamma": 1}, {"shared_decode_graph": False}):
            with self.assertRaisesRegex(ValueError, ZERO_OVERHEAD_ENV):
                preset_env("decode", "/weights", parent=env, **{**options, **overrides})
        with self.assertRaisesRegex(ValueError, "simulated acceptance 3.25"):
            preset_env("decode", "/weights", parent={**env, DSPARK_SIMULATED_ACCEPT_ENV: "3"}, **options)
        with self.assertRaisesRegex(ValueError, "boolean"):
            preset_env("decode", "/weights", parent={ZERO_OVERHEAD_ENV: "maybe"})

    def test_zero_overhead_scheduler_default_uses_effective_server_config(self):
        config = NS(
            model_args=NS(model_type="deepseek_v41", max_seq_len=32768),
            role_config=NS(role_type="DECODE"),
            parallelism_config=NS(
                world_size=8, local_world_size=8, tp_size=1, dp_size=8, ep_size=8,
                prefill_cp_config=NS(method="DISABLED", prefill_cp_size=1, kv_cache_sharded=False),
            ),
            py_hw_kernel_config=NS(enable_cuda_graph=True),
            sp_config=NS(type="DSPARK", gen_num_per_cycle=3),
            eplb_config=NS(eplb_mode="NONE", redundant_expert=0),
            moe_config=NS(use_deepep_low_latency=False, use_deepep_moe=True, use_all_gather=False),
        )
        for backend, size, switch, expected in (
            ("ppu", 8, None, "1"), ("ppu", 8, "0", "0"),
            ("ppu", 32, None, "1"), ("ppu", 32, "1", "1"), ("cuda", 8, None, None),
        ):
            with self.subTest(backend=backend, size=size, switch=switch):
                config.parallelism_config.world_size = size
                config.parallelism_config.dp_size = size
                config.parallelism_config.ep_size = size
                env = {"RTP_V41_PPU_MOE_BACKEND": "megamoe", "RTP_V41_ENP_SIZE": str(size)}
                if switch is not None:
                    env[ZERO_OVERHEAD_ENV] = switch
                with patch.dict(os.environ, env, clear=True), patch(
                    "rtp_llm.config.v41_runtime.v41_backend_for_config", return_value=backend
                ), patch("rtp_llm.models_py.modules.dsv41.backend.get_v41_backend"):
                    validate_v41_server_config(config)
                    self.assertEqual(os.environ.get(ZERO_OVERHEAD_ENV), expected)
                    self.assertEqual(
                        os.environ.get(DSPARK_CROSS_ROUND_OVERLAP_ENV), "1" if expected == "1" else None
                    )
                    if expected == "1":
                        graph_vote = "1" if size == 8 else "0"
                        self.assertEqual(os.environ["RTP_V41_PPU_DSPARK_HOST_GRAPH_VOTE"], graph_vote)
                        self.assertEqual(os.environ["RTP_V41_PPU_DSPARK_EARLY_VERIFY_VOTE"], graph_vote)

    def test_host_budget_admits_all_eight_l2_pools(self):
        gib = 1024**3
        metadata = {"engram_table_bytes": 189 * gib}
        env = {"WORLD_SIZE": "8", "ENABLE_MEMORY_CACHE": "1", "MEMORY_CACHE_SIZE_MB": "65536"}
        # The old ENP-only check admitted 442 GiB, omitting 512 GiB of L2.
        with self.assertRaisesRegex(ValueError, "ENP\\+staging\\+L2"):
            host_memory_budget(env, metadata, 442 * gib)
        budget = host_memory_budget(env, metadata, 954 * gib)
        self.assertEqual(budget["host_l2_per_rank_bytes"], 64 * gib)
        self.assertEqual(budget["host_l2_total_bytes"], 512 * gib)
        self.assertEqual(budget["required_bytes"], 954 * gib)
        with self.assertRaises(ValueError):
            host_memory_budget(env, metadata, 954 * gib - 1)
        # Pool creation is independent of reuse; disabling REUSE alone does
        # not authorize excluding the configured host backing from admission.
        env["REUSE_CACHE"] = "0"
        self.assertEqual(host_memory_budget(env, metadata)["required_bytes"], 954 * gib)
        env["ENABLE_MEMORY_CACHE"] = "0"
        self.assertEqual(host_memory_budget(env, metadata, 442 * gib)["required_bytes"], 442 * gib)

    def test_host_budget_rejects_missing_and_invalid_l2_sizes(self):
        metadata = {"engram_table_bytes": 0}
        for value in (None, "0", "-1", "1.5", "bad"):
            env = {"ENABLE_MEMORY_CACHE": "1"}
            if value is not None:
                env["MEMORY_CACHE_SIZE_MB"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                host_memory_budget(env, metadata)
        with self.assertRaisesRegex(ValueError, "eight local"):
            host_memory_budget({"WORLD_SIZE": "16"}, metadata)

    def test_cache_policy_explains_decode_lookup_and_request_switch(self):
        env = {"ROLE_TYPE": "DECODE", "REUSE_CACHE": "1", "ENABLE_DEVICE_CACHE": "1",
               "ENABLE_MEMORY_CACHE": "1"}
        policy = local_cache_policy(env)
        self.assertEqual(policy["initial_prefix_lookup"], "disabled_for_opaque_decode")
        self.assertFalse(policy["pd_handoff_counts_as_historical_hit"])
        self.assertTrue(policy["request_reuse_cache_required"])
        env["RTP_LLM_IGNORE_REQUEST_CACHE_SWITCHES"] = "1"
        self.assertFalse(local_cache_policy(env)["request_reuse_cache_required"])
        env["ROLE_TYPE"] = "PREFILL"
        self.assertEqual(local_cache_policy(env)["initial_prefix_lookup"], "request_policy")
        env["REUSE_CACHE"] = "0"
        self.assertEqual(local_cache_policy(env)["initial_prefix_lookup"], "disabled")

    def test_serving_budgets_override_preset_and_cover_decode_capacity(self):
        env = preset_env(
            "decode", "/weights", parent={
                "RTP_V41_PPU_MODEL_PATH": "shared",
                "RTP_V41_PPU_MOE_BACKEND": "megamoe",
                "RTP_V41_PPU_MOE_PRECISION": "w4a4",
            }, shared_decode_graph=True, kv_cache_mem_mb=26000,
            load_cache_timeout_ms=600000, concurrency_limit=8,
            max_batch_tokens=32768, max_inited_kv_cache_streams=8,
            reuse_cache=True, capture_batches="1,2,4,8",
            enable_device_cache=True, enable_memory_cache=True, memory_cache_size_mb=65536,
        )
        for key, value in {
            "KV_CACHE_MEM_MB": "26000", "LOAD_CACHE_TIMEOUT_MS": "600000",
            "CONCURRENCY_LIMIT": "8", "MAX_BATCH_TOKENS_SIZE": "32768",
            "MAX_BATCH_TOKENS_WITHOUT_CACHE": "32768",
            "MAX_INITED_KV_CACHE_STREAMS": "8", "REUSE_CACHE": "1",
            "DECODE_CAPTURE_CONFIG": "1,2,4,8", "ENABLE_CUDA_GRAPH": "1",
            "ENABLE_DEVICE_CACHE": "1", "ENABLE_MEMORY_CACHE": "1", "MEMORY_CACHE_SIZE_MB": "65536",
        }.items():
            self.assertEqual(env[key], value)

    def test_serving_budgets_reject_invalid_or_incompatible_options(self):
        for name in ("kv_cache_mem_mb", "load_cache_timeout_ms", "concurrency_limit",
                     "max_batch_tokens", "max_inited_kv_cache_streams", "memory_cache_size_mb"):
            for value in (0, -1, True, 1.5):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    preset_env("decode", "/weights", parent={}, **{name: value})
        with self.assertRaisesRegex(ValueError, "requires cache reuse"):
            preset_env("prefill", "/weights", parent={}, ced_mode="bounded",
                       encoder_swa_replay=True, reuse_cache=False)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            preset_env("decode", "/weights", parent={}, enable_memory_cache=True)
        for batches in ("", "0,1", "2,4", "1,1,2", "1,4,2", "1,-2", "1,x"):
            with self.subTest(batches=batches), self.assertRaises(ValueError):
                preset_env("decode", "/weights", parent={}, shared_decode_graph=True,
                           capture_batches=batches)
        with self.assertRaisesRegex(ValueError, "explicit shared Decode Graph"):
            preset_env("prefill", "/weights", parent={}, capture_batches="1,2")
        with self.assertRaisesRegex(ValueError, "per-DP concurrency"):
            preset_env("decode", "/weights", parent={}, shared_decode_graph=True,
                       concurrency_limit=8, capture_batches="1,2,4")
        with self.assertRaisesRegex(ValueError, "per-DP concurrency"):
            preset_env("decode", "/weights", parent={"CONCURRENCY_LIMIT": "8",
                       "RTP_V41_PPU_MOE_BACKEND": "megamoe"},
                       shared_decode_graph=True, capture_batches="1,2,4")

    def test_print_command_preserves_explicit_serving_budgets_without_secrets(self):
        from rtp_llm.tools.v41.launch import main

        output = io.StringIO()
        argv = ["launch", "decode", "--checkpoint-path", "/weights", "--print-command",
                "--shared-decode-graph", "--kv-cache-mem-mb", "26000",
                "--concurrency-limit", "8", "--decode-capture-batches", "1,2,4,8",
                "--max-batch-tokens", "32768", "--reuse-cache"]
        with patch.dict(os.environ, {
            "RTP_V41_PPU_MOE_BACKEND": "megamoe", "RTP_V41_PPU_MOE_PRECISION": "w4a4",
            "PRIVATE_TEST_SENTINEL": "do-not-print",
            "ENABLE_DEVICE_CACHE": "1", "ENABLE_MEMORY_CACHE": "1", "MEMORY_CACHE_SIZE_MB": "65536",
            "FRONTEND_SERVER_COUNT": "4",
        }), patch.object(sys, "argv", argv), redirect_stdout(output):
            self.assertEqual(main(), 0)
        for assignment in ("KV_CACHE_MEM_MB=26000", "CONCURRENCY_LIMIT=8",
                           "DECODE_CAPTURE_CONFIG=1,2,4,8", "REUSE_CACHE=1",
                           "MAX_BATCH_TOKENS_WITHOUT_CACHE=32768", "FRONTEND_SERVER_COUNT=4"):
            self.assertIn(assignment, output.getvalue())
        for assignment in ("ENABLE_DEVICE_CACHE=1", "ENABLE_MEMORY_CACHE=1", "MEMORY_CACHE_SIZE_MB=65536"):
            self.assertIn(assignment, output.getvalue())
        self.assertNotIn("do-not-print", output.getvalue())

    def test_print_command_preserves_effective_fake_balance_without_environment_leak(
        self,
    ):
        from rtp_llm.tools.v41.launch import main

        for flags, expected in (([], "1"), (["--no-fake-balance-expert"], "0")):
            output = io.StringIO()
            argv = [
                "launch",
                "decode",
                "--checkpoint-path",
                "/weights",
                "--print-command",
                *flags,
            ]
            with patch.dict(
                os.environ,
                {"FAKE_BALANCE_EXPERT": "1", "PRIVATE_TEST_SENTINEL": "do-not-print"},
            ), patch.object(sys, "argv", argv), redirect_stdout(output):
                self.assertEqual(main(), 0)
            self.assertIn("FAKE_BALANCE_EXPERT=" + expected, output.getvalue())
            self.assertNotIn("PRIVATE_TEST_SENTINEL", output.getvalue())
            self.assertNotIn("do-not-print", output.getvalue())

    def test_fake_balance_default_environment_and_explicit_override(self):
        for role in ("prefill", "decode"):
            self.assertEqual(
                preset_env(role, "/weights", parent={})["FAKE_BALANCE_EXPERT"], "0"
            )
            self.assertEqual(
                preset_env(role, "/weights", parent={"FAKE_BALANCE_EXPERT": "1"})[
                    "FAKE_BALANCE_EXPERT"
                ],
                "1",
            )
            self.assertEqual(
                preset_env(
                    role,
                    "/weights",
                    parent={"FAKE_BALANCE_EXPERT": "1"},
                    fake_balance_expert=False,
                )["FAKE_BALANCE_EXPERT"],
                "0",
            )
        for role in ("prefill", "decode"):
            env = preset_env(
                role, "/weights",
                parent={"RTP_V41_PPU_MOE_BACKEND": "megamoe",
                        "RTP_V41_PPU_MOE_PRECISION": "w4a4"},
                dspark_gamma=3, fake_balance_expert=True,
                shared_decode_graph=role == "decode",
            )
            self.assertEqual(env["FAKE_BALANCE_EXPERT"], "1")
            self.assertEqual(env["GEN_NUM_PER_CIRCLE"], "3")
            if role == "decode":
                self.assertEqual(env["ENABLE_CUDA_GRAPH"], "1")
                self.assertEqual(env[DSPARK_OVERLAP_ENV], "1")
                for key, value in DSPARK_OVERLAP_FLAGS.items():
                    self.assertEqual(env[key], value)

    def test_frontend_worker_count_can_override_preset(self):
        for role in ("prefill", "decode"):
            with self.subTest(role=role):
                default_env = preset_env(role, "/weights", parent={})
                self.assertEqual(default_env["FRONTEND_SERVER_COUNT"], "1")
                configured = preset_env(role, "/weights", parent={"FRONTEND_SERVER_COUNT": "4"})
                self.assertEqual(configured["FRONTEND_SERVER_COUNT"], "4")

    def test_zero_overhead_default_respects_decode_profile_and_legacy_opt_out(self):
        flags = {"RTP_V41_PPU_MOE_BACKEND": "megamoe"}
        options = dict(parent=flags, shared_decode_graph=True, dspark_gamma=3)
        default = preset_env("decode", "/weights", **options)
        self.assertEqual(default[DSPARK_CROSS_ROUND_OVERLAP_ENV], "1")
        for env in (
            preset_env("prefill", "/weights", parent=flags, dspark_gamma=3),
            preset_env("decode", "/weights", parent=flags, dspark_gamma=3),
            preset_env("decode", "/weights", **dict(options, dspark_gamma=1)),
        ):
            self.assertNotIn(DSPARK_CROSS_ROUND_OVERLAP_ENV, env)
            self.assertNotIn(ZERO_OVERHEAD_ENV, env)

        flags[DSPARK_CROSS_ROUND_OVERLAP_ENV] = "false"
        disabled = preset_env("decode", "/weights", **options)
        self.assertEqual(disabled[DSPARK_CROSS_ROUND_OVERLAP_ENV], "0")
        self.assertEqual(disabled[DSPARK_OVERLAP_ENV], "1")
        self.assertNotIn(ZERO_OVERHEAD_ENV, disabled)

    def test_zero_overhead_scheduler_simulated_acceptance(self):
        parent = {
            "RTP_V41_PPU_MOE_BACKEND": "megamoe",
            DSPARK_SIMULATED_ACCEPT_ENV: "3.25",
        }
        options = dict(shared_decode_graph=True, dspark_gamma=3)
        for switch in (None, "0", "1"):
            with self.subTest(switch=switch):
                env = dict(parent)
                if switch is not None:
                    env[ZERO_OVERHEAD_ENV] = switch
                decode = preset_env("decode", "/weights", parent=env, **options)
                self.assertEqual(decode[ZERO_OVERHEAD_ENV], switch or "1")
                self.assertEqual(decode[DSPARK_SIMULATED_ACCEPT_ENV], "3.25")
                self.assertEqual(decode[DSPARK_OVERLAP_ENV], "1")
                self.assertEqual(decode[DSPARK_CROSS_ROUND_OVERLAP_ENV], "1")

    def test_dspark_cross_round_overlap_honors_explicit_flags(self):
        for role in ("prefill", "decode"):
            with self.subTest(role=role):
                flags = {DSPARK_OVERLAP_ENV: "0", "RTP_V41_PPU_MOE_BACKEND": "megamoe"}
                graph = role == "decode"
                options = dict(parent=flags, shared_decode_graph=graph, dspark_gamma=3)
                baseline = preset_env(role, "/weights", **options)
                self.assertNotIn(DSPARK_CROSS_ROUND_OVERLAP_ENV, baseline)
                flags[DSPARK_CROSS_ROUND_OVERLAP_ENV] = "true"
                enabled = preset_env(role, "/weights", **options)
                self.assertEqual(enabled[DSPARK_CROSS_ROUND_OVERLAP_ENV], "1")
                self.assertEqual(enabled[DSPARK_OVERLAP_ENV], "1" if graph else "0")
                self.assertEqual(V41RuntimeConfig.from_env(enabled).role, role.upper())

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

    def test_kv_sharding_requires_cp8_source(self):
        for role in ("prefill", "decode"):
            with self.subTest(role=role):
                options = dict(parent={}, kv_cache_sharded=True)
                env = preset_env(role, "/weights", **options)
                runtime = V41RuntimeConfig.from_env(env)
                self.assertEqual(runtime.source_cp, 8)
                self.assertTrue(runtime.kv_sharded)
                with self.assertRaisesRegex(ValueError, "CP KV sharding requires CP8"):
                    preset_env(
                        role,
                        "/weights",
                        prefill_parallel="dp",
                        **options,
                    )

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
        for extra, expected in (
            ([], "65536"),
            (["--max-seq-len", "131072"], "131072"),
            (["--dspark-gamma", "3"], "65536"),
        ):
            with self.subTest(extra=extra), patch.dict(
                os.environ, {"MAX_SEQ_LEN": "65536"}, clear=True
            ), patch.object(sys, "argv", common + extra), redirect_stdout(
                io.StringIO()
            ) as stream:
                self.assertEqual(main(), 0)
            self.assertIn(f"MAX_SEQ_LEN={expected}", stream.getvalue())
            self.assertTrue(stream.getvalue().strip().endswith("-m rtp_llm.start_server"))

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

    def test_dp32_launch_routes_each_local_rank_and_keeps_prefill_cp8(self):
        nodes = ("192.0.2.20", "192.0.2.21", "192.0.2.22", "192.0.2.23")
        options = dict(
            parent={"RTP_V41_PPU_MOE_BACKEND": "megamoe"},
            prefill_address="192.0.2.10:18090",
            decode_address="192.0.2.20:19090",
            decode_nodes=nodes,
            max_seq_len=98304,
            dspark_gamma=3,
        )
        prefill = preset_env("prefill", "/weights", **options)
        self.assertEqual((prefill["WORLD_SIZE"], prefill["TP_SIZE"], prefill["EP_SIZE"]),
                         ("8", "8", "8"))
        for node_rank in range(4):
            with self.subTest(node_rank=node_rank):
                env = preset_env("decode", "/weights", decode_node_rank=node_rank,
                                 shared_decode_graph=True, capture_batches="1,4,8,16,32",
                                 concurrency_limit=32, **options)
                runtime = V41RuntimeConfig.from_env(env)
                self.assertEqual((runtime.world_size, runtime.attention_dp, runtime.ep, runtime.enp),
                                 (32, 32, 32, 32))
                self.assertEqual((env["LOCAL_WORLD_SIZE"], env["WORLD_RANK"]),
                                 ("8", str(node_rank * 8)))
                self.assertIn("name:decode_part0,ip:192.0.2.20,port:19090",
                              env["GANG_CONFIG_STRING"])
                self.assertEqual(runtime.speculative, "DSPARK")
                self.assertEqual(env[ZERO_OVERHEAD_ENV], "1")
                self.assertEqual(env[DSPARK_CROSS_ROUND_OVERLAP_ENV], "1")
                self.assertEqual(env["RTP_V41_PPU_DSPARK_HOST_GRAPH_VOTE"], "0")
                self.assertEqual(env["RTP_V41_PPU_DSPARK_EARLY_VERIFY_VOTE"], "0")
                with self.assertRaisesRegex(ValueError, "world_size=EP=ENP=32"):
                    V41RuntimeConfig.from_env(dict(env, RTP_V41_ENP_SIZE="8"))
                with self.assertRaisesRegex(ValueError, "EP32 PPU Decode requires MegaMoE"):
                    V41RuntimeConfig.from_env(dict(env, RTP_V41_PPU_MOE_BACKEND="grouped"))
        for role_env in (prefill, env):
            endpoints = json.loads(role_env["MODEL_SERVICE_CONFIG"])["role_endpoints"][0]
            self.assertEqual(endpoints["decode_endpoint"]["address"].split(","),
                             [f"{host}:{19090 + rank * 9}" for host in nodes for rank in range(8)])
        with self.assertRaisesRegex(ValueError, "first Decode node"):
            preset_env("prefill", "/weights", **dict(options, decode_address="192.0.2.21:19090"))
        host_parent = dict(options["parent"], RTP_V41_PPU_DSPARK_HOST_GRAPH_VOTE="1")
        host_env = preset_env(
            "decode", "/weights", shared_decode_graph=True, **dict(options, parent=host_parent)
        )
        self.assertEqual(host_env["RTP_V41_PPU_DSPARK_HOST_GRAPH_VOTE"], "1")
        self.assertEqual(host_env["RTP_V41_PPU_DSPARK_EARLY_VERIFY_VOTE"], "1")
        disabled = preset_env(
            "decode", "/weights", shared_decode_graph=True,
            **dict(options, parent={**options["parent"], ZERO_OVERHEAD_ENV: "0"}),
        )
        self.assertEqual(disabled[ZERO_OVERHEAD_ENV], "0")
        self.assertNotIn(DSPARK_CROSS_ROUND_OVERLAP_ENV, disabled)

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

    def test_megamoe_dspark_graph_uses_synchronous_dense_verify(self):
        parent = {
            "RTP_V41_PPU_MOE_BACKEND": "megamoe",
            DSPARK_OVERLAP_ENV: "0",
        }
        parent.update(dict.fromkeys(MEGAMOE_DSPARK_ASYNC_FLAGS, "1"))
        parent["GENERATION_PREFILL_CAPTURE_CONFIG"] = "512"
        options = dict(shared_decode_graph=True, dspark_gamma=3, parent=parent)
        env = preset_env("decode", "/weights", **options)
        runtime = V41RuntimeConfig.from_env(env)
        self.assertEqual(
            (runtime.speculative, runtime.cuda_graph, runtime.moe_precision),
            ("DSPARK", True, "w4a4"),
        )
        self.assertNotIn("GENERATION_PREFILL_CAPTURE_CONFIG", env)
        for flag in DSPARK_DENSE_DECODE_DEFAULTS:
            self.assertEqual(env[flag], "1")
        for flag in MEGAMOE_DSPARK_ASYNC_FLAGS:
            self.assertEqual(env[flag], "0")
            with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, "synchronous dense"):
                V41RuntimeConfig.from_env(dict(env, **{flag: "1"}))
        with self.assertRaisesRegex(ValueError, "synchronous dense"):
            preset_env(
                "decode",
                "/weights",
                dspark_confidence="bounded",
                dspark_verify_mode="packed",
                dspark_verify_budget=1,
                **options,
            )
        with self.assertRaisesRegex(ValueError, "synchronous dense"):
            V41RuntimeConfig.from_env(dict(env, SP_DSPARK_VERIFY_MODE="packed"))

    def test_print_command_includes_megamoe_dspark_graph_selection(self):
        from rtp_llm.tools.v41.launch import main

        output = io.StringIO()
        argv = [
            "launch", "decode", "--checkpoint-path", "/weights",
            "--shared-decode-graph", "--dspark-gamma", "3", "--print-command",
        ]
        with patch.dict(os.environ, {"RTP_V41_PPU_MOE_BACKEND": "megamoe"}, clear=True), patch.object(
            sys, "argv", argv
        ), redirect_stdout(output):
            self.assertEqual(main(), 0)
        self.assertIn("RTP_V41_PPU_MOE_BACKEND=megamoe", output.getvalue())
        self.assertIn("SP_TYPE=dspark", output.getvalue())
        self.assertIn(f"{ZERO_OVERHEAD_ENV}=1", output.getvalue())
        self.assertIn(f"{DSPARK_CROSS_ROUND_OVERLAP_ENV}=1", output.getvalue())

    def test_preflight_uses_effective_graph_selection(self):
        from rtp_llm.tools.v41.launch import main

        argv = [
            "launch", "decode", "--checkpoint-path", "/weights",
            "--shared-decode-graph", "--dspark-gamma", "3", "--check-only",
        ]
        parent = {
            "RTP_V41_PPU_MOE_BACKEND": "megamoe",
            "RTP_V41_PPU_SHARED_GROUPED_GRAPH": "1",
        }
        with patch.dict(os.environ, parent, clear=True), patch.object(
            sys, "argv", argv
        ), patch("rtp_llm.tools.v41.launch.collect_report", return_value={"prerequisites_ok": True}) as report:
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(), 0)
        runtime = V41RuntimeConfig.from_env(report.call_args.args[0])
        self.assertFalse(runtime.shared_grouped_graph)
        self.assertEqual(runtime.speculative, "DSPARK")

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
            config.model_args.max_seq_len = 32768
            config.py_hw_kernel_config.enable_cuda_graph = True
            config.sp_config = NS(type="DSPARK", sp_dspark_verify_mode="dense")
            with patch.dict(os.environ, {"RTP_V41_PPU_MOE_BACKEND": "megamoe"}):
                runtime = V41RuntimeConfig.from_server(config)
                self.assertEqual((runtime.speculative, runtime.moe_precision), ("DSPARK", "w4a4"))
                config.sp_config.sp_dspark_verify_mode = "packed"
                with self.assertRaisesRegex(ValueError, "synchronous dense"):
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
            self.assertEqual(result["declared_weight_bytes"], 16)
            # The qualified Qlean PTPC index has no declared byte count.
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"metadata": {}, "weight_map": {"weight": "one.safetensors"}})
            )
            result = checkpoint_metadata(root, signature)
            self.assertIsNone(result["declared_weight_bytes"])
            self.assertEqual(result["shard_count"], 1)
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
