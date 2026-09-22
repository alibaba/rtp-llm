import argparse
import json
import os
import pathlib
import shlex
import subprocess
import sys
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest import mock

from example.k3 import kimi_k3_full_model_two_host_pd_smoke_driver as driver


class ForwardedOptionalEnvironmentTest(unittest.TestCase):
    def test_role_specific_ssm_storage_dtype(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            for role in ("prefill", "decode"):
                self.assertEqual(driver.forwarded_optional_environment(role)["SSM_STATE_DTYPE"], "fp32")
        with mock.patch.dict(os.environ, {"PREFILL_SSM_STATE_DTYPE": "bf16", "DECODE_SSM_STATE_DTYPE": "fp32"}, clear=True):
            self.assertEqual(driver.forwarded_optional_environment("prefill")["SSM_STATE_DTYPE"], "bf16")
            self.assertEqual(driver.forwarded_optional_environment("decode")["SSM_STATE_DTYPE"], "fp32")

    def test_draft_mode_defaults_and_explicit_overrides(self) -> None:
        for settings, expected in (
            ({}, ("mtp", "kimi_k3_mtp")),
            ({"SP_TYPE": "", "SP_MODEL_TYPE": ""}, ("mtp", "kimi_k3_mtp")),
            ({"SP_TYPE": "mtp"}, ("mtp", "kimi_k3_mtp")),
            (
                {"SP_TYPE": "mtp", "SP_MODEL_TYPE": "kimi_k3_mtp"},
                ("mtp", "kimi_k3_mtp"),
            ),
        ):
            with self.subTest(settings=settings), mock.patch.dict(
                os.environ, settings, clear=True
            ):
                for role in ("prefill", "decode"):
                    env = driver.forwarded_optional_environment(role)
                    self.assertEqual((env["SP_TYPE"], env["SP_MODEL_TYPE"]), expected)

    def test_invalid_draft_modes_fail_before_remote_launch(self) -> None:
        for settings in (
            {"SP_TYPE": "none"},
            {"SP_TYPE": "eagle3"},
            {"SP_TYPE": "eagle3", "SP_MODEL_TYPE": "kimi_k3_mla_swa_eagle3"},
            {"SP_MODEL_TYPE": "kimi_k3_mla_swa_eagle3"},
            {"SP_TYPE": "eagle3", "SP_MODEL_TYPE": "kimi_k3_mtp"},
        ):
            with self.subTest(settings=settings), mock.patch.dict(
                os.environ, settings, clear=True
            ):
                for role in ("prefill", "decode"):
                    with self.assertRaises(ValueError):
                        driver.forwarded_optional_environment(role)

    def test_legacy_aux_layers_are_not_forwarded(self) -> None:
        with mock.patch.dict(
            os.environ, {"KIMI_K3_EAGLE3_AUX_LAYER_IDS": "0,44,88"}, clear=True
        ):
            for role in ("prefill", "decode"):
                env = driver.forwarded_optional_environment(role)
                self.assertEqual(env["SP_TYPE"], "mtp")
                self.assertEqual(env["SP_MODEL_TYPE"], "kimi_k3_mtp")
                self.assertNotIn("KIMI_K3_EAGLE3_AUX_LAYER_IDS", env)

    def test_forwards_dcp_padding_regression_to_both_roles(self) -> None:
        with mock.patch.dict(
            os.environ, {"SMOKE_DCP_PADDING_REGRESSION": "1"}, clear=True
        ):
            for role in ("prefill", "decode"):
                self.assertEqual(
                    driver.forwarded_optional_environment(role).get(
                        "SMOKE_DCP_PADDING_REGRESSION"
                    ),
                    "1",
                )

    def test_forwards_prefill_page_rr_multi_launch_profile_to_both_roles(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"SMOKE_PREFILL_PAGE_RR_MULTI_LAUNCH": "1"},
            clear=True,
        ):
            for role in ("prefill", "decode"):
                self.assertEqual(
                    driver.forwarded_optional_environment(role)[
                        "SMOKE_PREFILL_PAGE_RR_MULTI_LAUNCH"
                    ],
                    "1",
                )

    def test_removed_page_rr_switches_are_not_forwarded(self) -> None:
        for value in ("0", "1"):
            settings = {"SMOKE_PAGE_RR": value, "SMOKE_DECODE_PAGE_RR": value}
            with mock.patch.dict(os.environ, settings, clear=True):
                for role in ("prefill", "decode"):
                    forwarded = driver.forwarded_optional_environment(role)
                    for key in settings:
                        self.assertNotIn(key, forwarded)

    def test_forwards_decode_q_replicated_to_both_roles(self) -> None:
        with mock.patch.dict(
            os.environ, {"SMOKE_DECODE_Q_REPLICATED": "1"}, clear=True
        ):
            for role in ("prefill", "decode"):
                self.assertEqual(
                    driver.forwarded_optional_environment(role)[
                        "SMOKE_DECODE_Q_REPLICATED"
                    ],
                    "1",
                )

    def test_decode_q_replicated_defaults_to_absent(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            for role in ("prefill", "decode"):
                self.assertNotIn(
                    "SMOKE_DECODE_Q_REPLICATED",
                    driver.forwarded_optional_environment(role),
                )

    def test_forwards_service_environment_isolation_to_both_roles(self) -> None:
        settings = {"PYTHONNOUSERSITE": "1", "NCCL_GRAPH_REGISTER": "0"}
        with mock.patch.dict(os.environ, settings, clear=True):
            for role in ("prefill", "decode"):
                forwarded = driver.forwarded_optional_environment(role)
                for key, value in settings.items():
                    self.assertEqual(forwarded[key], value)

    def test_precision_modes_and_prefix_budget_reach_both_role_commands(self) -> None:
        args = argparse.Namespace(
            prefill_repo_root="/data1/prefill",
            decode_repo_root="/data1/decode",
            prefill_checkpoint_path="/data1/target",
            decode_checkpoint_path="/data1/target",
            prefill_sp_checkpoint_path="/data1/draft",
            decode_sp_checkpoint_path="/data1/draft",
            prefill_container_runtime="docker",
            decode_container_runtime="docker",
            container="lhc_GPU",
            container_user="luohaocheng.lhc",
            prefill_endpoint="prefill:27188",
            decode_endpoint="decode:28188",
            run_id="precision-test",
            suite="all",
            result_endpoint=None,
        )
        for weight in ("0", "1"):
            for mla in ("0", "1"):
                for budget in ("0", "0.25"):
                    settings = {
                        "FP8_GEMM": weight,
                        "FP8_KV_CACHE": mla,
                        "FP8_MLA": mla,
                        "KIMI_K3_MLA_PREFILL_EXPANDED_KV_BUDGET_GIB": budget,
                        "LOAD_METHOD": "fastsafetensors",
                    }
                    with self.subTest(
                        weight=weight, mla=mla, budget=budget
                    ), mock.patch.dict(os.environ, settings, clear=True):
                        for role in ("prefill", "decode"):
                            command = driver.build_remote_command(args, role)
                            inner = shlex.split(command)[-1]
                            tokens = shlex.split(inner)
                            for key, value in settings.items():
                                self.assertIn(f"{key}={value}", tokens)
                            self.assertIn("SP_TYPE=mtp", tokens)
                            self.assertIn("SP_MODEL_TYPE=kimi_k3_mtp", tokens)
                            self.assertEqual(tokens[-2:], [driver.ROLE_SCRIPT, role])

    def test_forwards_mtp_mode_to_both_roles(self) -> None:
        mode = {
            "SP_TYPE": "mtp",
            "SP_MODEL_TYPE": "kimi_k3_mtp",
            "TP_SIZE": "4",
            "EP_SIZE": "4",
            "KIMI_K3_TP_SIZE": "4",
            "KIMI_K3_EP_SIZE": "4",
            "GEN_NUM_PER_CIRCLE": "1",
        }
        with mock.patch.dict(os.environ, mode, clear=True):
            for role in ("prefill", "decode"):
                forwarded = driver.forwarded_optional_environment(role)
                for key, value in mode.items():
                    self.assertEqual(forwarded[key], value)

    def test_role_artifact_roots_override_shared_default(self) -> None:
        settings = {
            "SMOKE_ARTIFACT_ROOT": "/tmp/shared-default",
            "PREFILL_SMOKE_ARTIFACT_ROOT": "/data4/user/mtp-smoke",
            "DECODE_SMOKE_ARTIFACT_ROOT": "/data5/user/mtp-smoke",
        }
        with mock.patch.dict(os.environ, settings, clear=True):
            for role in ("prefill", "decode"):
                self.assertEqual(
                    driver.forwarded_optional_environment(role)["SMOKE_ARTIFACT_ROOT"],
                    settings[f"{role.upper()}_SMOKE_ARTIFACT_ROOT"],
                )
            del os.environ["PREFILL_SMOKE_ARTIFACT_ROOT"]
            self.assertEqual(
                driver.forwarded_optional_environment("prefill")["SMOKE_ARTIFACT_ROOT"],
                settings["SMOKE_ARTIFACT_ROOT"],
            )

    def test_forwards_explicit_rdma_hca_allowlist_to_both_roles(self) -> None:
        value = "mlx5_bond_0,mlx5_bond_1"
        with mock.patch.dict(os.environ, {"SMOKE_ACCL_USE_NICS": value}, clear=True):
            for role in ("prefill", "decode"):
                with self.subTest(role=role):
                    self.assertEqual(
                        driver.forwarded_optional_environment(role)[
                            "SMOKE_ACCL_USE_NICS"
                        ],
                        value,
                    )

    def test_does_not_invent_driver_override_when_unset(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertNotIn(
                "SMOKE_ACCL_USE_NICS",
                driver.forwarded_optional_environment("prefill"),
            )

    def test_forwards_keep_cluster_mode_to_both_roles(self) -> None:
        with mock.patch.dict(
            os.environ, {"SMOKE_KEEP_CLUSTER_ON_SUCCESS": "1"}, clear=True
        ):
            for role in ("prefill", "decode"):
                self.assertEqual(
                    driver.forwarded_optional_environment(role)[
                        "SMOKE_KEEP_CLUSTER_ON_SUCCESS"
                    ],
                    "1",
                )


class KimiK3FullModelTwoHostPdSmokeDriverTest(unittest.TestCase):
    def test_role_profile_pins_runtime_defaults_and_drops_stale_mla_selector(self):
        role_script = pathlib.Path(driver.__file__).with_name(
            "kimi_k3_full_model_two_host_pd_smoke.sh"
        )
        script = role_script.read_text(encoding="utf-8")

        self.assertIn("export RESERVE_BLOCK_RATIO=5", script)
        self.assertIn("export RTP_LLM_MTP_ASYNC_PREPARE=0", script)
        self.assertNotIn("RTP_MLA_DECODE_KERNEL", script)

    def test_role_script_rejects_non_native_mtp_before_host_validation(self):
        role_script = pathlib.Path(driver.__file__).with_name(
            "kimi_k3_full_model_two_host_pd_smoke.sh"
        )
        for role in ("prefill", "decode"):
            for settings, expected in (
                ({"SP_TYPE": "eagle3"}, "SP_TYPE=mtp"),
                (
                    {"SP_TYPE": "eagle3", "SP_MODEL_TYPE": "kimi_k3_mla_swa_eagle3"},
                    "SP_TYPE=mtp",
                ),
                ({"SP_TYPE": "none"}, "SP_TYPE=mtp"),
                (
                    {"SP_TYPE": "mtp", "SP_MODEL_TYPE": "kimi_k3_mla_swa_eagle3"},
                    "SP_MODEL_TYPE=kimi_k3_mtp",
                ),
            ):
                with self.subTest(role=role, settings=settings):
                    result = subprocess.run(
                        [role_script, role],
                        env={"PATH": os.environ["PATH"], **settings},
                        text=True,
                        capture_output=True,
                        check=False,
                        timeout=10,
                    )
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertIn(expected, result.stderr)

    def test_dcp_padding_regression_resolves_and_validates_profile(self):
        source = pathlib.Path(driver.__file__).with_name(
            "kimi_k3_full_model_two_host_pd_smoke.sh"
        ).read_text()
        # Execute real argument/profile handling, stopping before host access.
        prefix = source.split('[[ "$(id -u)" != "0" ]]', 1)[0]
        command = prefix + "\nprintf '%s %s %s' \"$smoke_dcp_padding_regression\" \"${GEN_NUM_PER_CIRCLE:-unset}\" \"${SMOKE_SUITE:-unset}\"\n"
        for settings, expected in (
            ({}, "0 unset unset"),
            ({"SMOKE_DCP_PADDING_REGRESSION": "0"}, "0 unset unset"),
            ({"SMOKE_DCP_PADDING_REGRESSION": "1"}, "1 2 all"),
            ({"SMOKE_DECODE_PAGE_RR": "1", "GEN_NUM_PER_CIRCLE": "3"}, "0 3 unset"),
            ({"SMOKE_DCP_PADDING_REGRESSION": "1", "GEN_NUM_PER_CIRCLE": "3"}, None),
            ({"SMOKE_DCP_PADDING_REGRESSION": "1", "SMOKE_DECODE_PAGE_RR": "0"}, None),
            ({"SMOKE_DCP_PADDING_REGRESSION": "1", "SMOKE_SUITE": "flow"}, None),
            ({"SMOKE_DCP_PADDING_REGRESSION": "bad"}, None),
        ):
            for role in ("prefill", "decode"):
                with self.subTest(settings=settings, role=role):
                    result = subprocess.run(
                        ["bash", "-c", command, "smoke-profile-test", role],
                        env={"PATH": os.environ["PATH"], **settings},
                        capture_output=True, text=True, timeout=10,
                    )
                    if expected is None:
                        self.assertEqual(result.returncode, 2, result.stderr)
                        self.assertIn("SMOKE_DCP_PADDING_REGRESSION", result.stderr)
                    else:
                        self.assertEqual(result.returncode, 0, result.stderr)
                        self.assertEqual(result.stdout, expected)

    def test_role_script_rejects_deployment_profile_drift(self):
        script = pathlib.Path(driver.__file__).with_name(
            "kimi_k3_full_model_two_host_pd_smoke.sh"
        )
        for settings, message in (
            ({"TP_SIZE": "16", "EP_SIZE": "16"}, "integers in 1..8"),
            ({"SMOKE_CHUNKWISE_RDMA": "0"}, "SMOKE_CHUNKWISE_RDMA=1"),
        ):
            for role in ("prefill", "decode"):
                with self.subTest(settings=settings, role=role):
                    result = subprocess.run(
                        [script, role],
                        env={"PATH": os.environ["PATH"], **settings},
                        text=True,
                        capture_output=True,
                        check=False,
                        timeout=10,
                    )
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertIn(message, result.stderr)

    def test_page_rr_profiles_override_inherited_ktp_and_replicated_layouts(self):
        source = pathlib.Path(driver.__file__).with_name(
            "kimi_k3_full_model_two_host_pd_smoke.sh"
        ).read_text()
        definitions = []
        for name in ("common", "prefill", "decode"):
            marker = f"apply_validated_{name}_profile() {{"
            definitions.append(marker + source.split(marker, 1)[1].split("\n}", 1)[0] + "\n}")
        settings = {
            "PATH": os.environ["PATH"],
            "SMOKE_RUN_ID": "profile-test",
            "checkpoint_real": "/data/target",
            "sp_checkpoint_real": "/data/draft",
            "PREFILL_ENDPOINT": "prefill:27188",
            "DECODE_ENDPOINT": "decode:28188",
            "role_dir": "/tmp/profile-test",
            "smoke_block_size": "128",
            "smoke_kernel_block_size": "128",
            "smoke_linear_step": "1",
            "smoke_chunkwise_rdma": "1",
            "smoke_accl_use_nics": "",
            "smoke_sp_type": "mtp",
            "smoke_sp_model_type": "kimi_k3_mtp",
            "smoke_proposal_tokens": "3",
            "smoke_tp_size": "8",
            "smoke_ep_size": "8",
            "smoke_chunk_tokens": "65536",
            "smoke_mega_tokens": "8192",
            "smoke_shared_expert_shard": "1",
            "smoke_decode_topology": "legacy",
            "smoke_decode_q_replicated": "0",
            "smoke_prefill_kv_cache_mem_mb": "56000",
            "smoke_decode_kv_cache_mem_mb": "29000",
            "smoke_decode_kda_pool_blocks": "32",
        }
        # Stale shell settings must not select the old DP8/KTP8 path.
        inherited = {
            "DP_SIZE": "8", "KTP_SIZE": "8",
            "KIMI_K3_DECODE_TOPOLOGY": "dp8_ktp8_ep8",
            "PREFILL_CP_KV_CACHE_SHARDED": "0",
            "DECODE_CP_KV_CACHE_SHARDED": "0",
            "PREFILL_CP_SIZE": "16", "CP_ROTATE_METHOD": "ALLTOALL",
            "NCCL_GRAPH_REGISTER": "1",
            "MM_CACHE_GPU_MAX_BYTES": "21474836480",
            "ENABLE_SP_PREFILL_CUDA_GRAPH": "0",
            "RTP_LLM_MTP_ASYNC_PREPARE": "0",
        }
        for role, tp, dp, source_tp in (
            ("prefill", 8, 1, 8), ("prefill", 4, 1, 4),
            ("decode", 8, 1, 8), ("decode", 8, 1, 4), ("decode", 4, 2, 8),
        ):
            for overrides in ({}, inherited):
                with self.subTest(role=role, inherited=bool(overrides)):
                    command = "\n".join(definitions) + (
                        f"\napply_validated_common_profile\napply_validated_{role}_profile\n"
                        + shlex.join([sys.executable, "-c", "import json,os; print(json.dumps(dict(os.environ)))"])
                    )
                    result = subprocess.run(
                        ["bash", "-eu", "-c", command],
                        env={**settings, **overrides, "role": role,
                             "smoke_tp_size": str(tp), "smoke_dp_size": str(dp),
                             "smoke_world_size": str(tp * dp), "smoke_ep_size": str(tp * dp),
                             "smoke_prefill_tp_size": str(source_tp)},
                        text=True, capture_output=True, check=True,
                    )
                    env = json.loads(result.stdout)
                    for key, value in {"TP_SIZE": str(tp), "EP_SIZE": str(tp * dp), "DP_SIZE": str(dp), "KTP_SIZE": "1", "WORLD_SIZE": str(tp * dp), "LOCAL_WORLD_SIZE": str(tp * dp)}.items():
                        self.assertEqual(env[key], value)
                    self.assertNotIn("CP_ROTATE_METHOD", env)
                    self.assertEqual(env["KV_CACHE_MEM_MB"], "56000" if role == "prefill" else "29000")
                    if role == "prefill":
                        self.assertEqual(env["RTP_LLM_MTP_ASYNC_PREPARE"], "0")
                        self.assertNotIn("ENABLE_SP_PREFILL_CUDA_GRAPH", env)
                        self.assertEqual(env["PREFILL_CP_KV_CACHE_SHARDED"], "1")
                        self.assertEqual(env["REUSE_CACHE"], "1")
                        self.assertEqual(env["MM_CACHE_GPU_MAX_BYTES"], "1073741824")
                        self.assertNotIn("PREFILL_CP_SIZE", env)
                        self.assertNotIn("DECODE_CP_KV_CACHE_SHARDED", env)
                    else:
                        self.assertEqual(env["RTP_LLM_MTP_ASYNC_PREPARE"], "1")
                        self.assertEqual(env["ENABLE_SP_PREFILL_CUDA_GRAPH"], "1")
                        self.assertNotIn("MM_CACHE_GPU_MAX_BYTES", env)
                        self.assertEqual(env["KIMI_K3_DECODE_TOPOLOGY"], "legacy")
                        self.assertEqual(env["DECODE_CP_KV_CACHE_SHARDED"], "1")
                        self.assertEqual(env["PREFILL_CP_SIZE"], str(source_tp))
                        self.assertEqual(env["NCCL_GRAPH_REGISTER"], "0")
                        self.assertEqual(env["DECODE_CAPTURE_CONFIG"], "1,2,4,8")
                        self.assertNotIn("PREFILL_CP_KV_CACHE_SHARDED", env)

    def test_page_rr_geometry_and_default_mixed_owners(self):
        source = pathlib.Path(driver.__file__).with_name(
            "kimi_k3_full_model_two_host_pd_smoke.sh"
        ).read_text()
        topology = source[source.index('smoke_prefill_tp_size='):source.index('[[ "${SMOKE_CHUNKWISE_RDMA')]
        geometry = source[source.index('smoke_block_size='):source.index('smoke_proposal_tokens=')]
        for settings, expected in (
            ({}, "1024 128 legacy 2"),
            ({"SMOKE_DECODE_TP_SIZE": "4", "SMOKE_DECODE_DP_SIZE": "2"}, "1024 128 legacy 2"),
            ({"SMOKE_PREFILL_TP_SIZE": "4"}, "1024 128 legacy 2"),
            ({"SMOKE_DECODE_DP_SIZE": "2"}, "1024 128 legacy 2"),
            ({"SMOKE_DECODE_TP_SIZE": "8"}, "1024 128 legacy 1"),
            ({"SMOKE_BLOCK_SIZE": "256"}, "256 128 legacy 2"),
            ({"SMOKE_BLOCK_SIZE": "4096"}, "4096 128 legacy 2"),
            ({"SMOKE_BLOCK_SIZE": "256", "SMOKE_KERNEL_BLOCK_SIZE": "256"}, None),
            ({"SMOKE_BLOCK_SIZE": "384"}, None),
            ({"SMOKE_BLOCK_SIZE": "0"}, None),
        ):
            with self.subTest(settings=settings):
                command = 'die() { echo "$*" >&2; exit 2; }\n' + 'role=prefill\n' + topology + geometry
                command += '\nprintf "%s %s %s %s" "$smoke_block_size" "$smoke_kernel_block_size" "$smoke_decode_topology" "$smoke_decode_dp_size"'
                result = subprocess.run(
                    ["bash", "-eu", "-c", command],
                    env={"PATH": os.environ["PATH"], **settings},
                    text=True, capture_output=True,
                )
                if expected is None:
                    self.assertEqual(result.returncode, 2, result.stderr)
                else:
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(result.stdout, expected)

    def test_parse_args_requires_both_draft_checkpoints(self):
        argv = [
            "driver",
            "--prefill-ssh-target",
            "prefill-host",
            "--decode-ssh-target",
            "decode-host",
            "--prefill-repo-root",
            "/prefill/repo",
            "--decode-repo-root",
            "/decode/repo",
            "--prefill-checkpoint-path",
            "/prefill/checkpoint",
            "--decode-checkpoint-path",
            "/decode/checkpoint",
            "--prefill-sp-checkpoint-path",
            "/prefill/mtp",
            "--decode-sp-checkpoint-path",
            "/decode/mtp",
            "--prefill-endpoint",
            "10.0.0.1:27188",
            "--decode-endpoint",
            "10.0.0.2:28188",
            "--run-id",
            "projection-ktp",
        ]
        with mock.patch.object(sys, "argv", argv), mock.patch.dict(
            os.environ, {}, clear=True
        ):
            args = driver.parse_args()
        self.assertEqual(args.prefill_sp_checkpoint_path, "/prefill/mtp")
        self.assertEqual(args.decode_sp_checkpoint_path, "/decode/mtp")

    def test_parse_args_rejects_missing_decode_sp_checkpoint(self):
        argv = [
            "driver",
            "--prefill-ssh-target",
            "prefill-host",
            "--decode-ssh-target",
            "decode-host",
            "--prefill-repo-root",
            "/prefill/repo",
            "--decode-repo-root",
            "/decode/repo",
            "--prefill-checkpoint-path",
            "/prefill/checkpoint",
            "--decode-checkpoint-path",
            "/decode/checkpoint",
            "--prefill-sp-checkpoint-path",
            "/mtp/checkpoint",
            "--prefill-endpoint",
            "10.0.0.1:27188",
            "--decode-endpoint",
            "10.0.0.2:28188",
            "--run-id",
            "projection-ktp",
        ]
        with mock.patch.object(sys, "argv", argv), mock.patch.dict(
            os.environ, {}, clear=True
        ):
            with self.assertRaises(SystemExit):
                driver.parse_args()

    def test_role_command_exports_role_local_sp_checkpoint(self):
        args = argparse.Namespace(
            prefill_repo_root="/prefill/repo",
            decode_repo_root="/decode/repo",
            prefill_checkpoint_path="/prefill/checkpoint",
            decode_checkpoint_path="/decode/checkpoint",
            prefill_sp_checkpoint_path="/prefill/mtp",
            decode_sp_checkpoint_path="/decode/mtp",
            prefill_endpoint="10.0.0.1:27188",
            decode_endpoint="10.0.0.2:28188",
            run_id="projection-ktp",
            suite="flow",
            result_endpoint=None,
            container="lhc_GPU",
            prefill_container_runtime="docker",
            decode_container_runtime="docker",
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            _, _, _, command = driver.role_launch_parts(args, "decode")
        self.assertIn("SP_CHECKPOINT_PATH=/decode/mtp", command)

    def test_role_command_preserves_long_prefix_overrides_for_both_hosts(self):
        args = argparse.Namespace(
            prefill_repo_root="/prefill/repo",
            decode_repo_root="/decode/repo",
            prefill_checkpoint_path="/prefill/checkpoint",
            decode_checkpoint_path="/decode/checkpoint",
            prefill_sp_checkpoint_path="/prefill/mtp",
            decode_sp_checkpoint_path="/decode/mtp",
            prefill_endpoint="10.0.0.1:27188",
            decode_endpoint="10.0.0.2:28188",
            run_id="long-prefix-overrides",
            suite="all",
            result_endpoint=None,
            container="lhc_GPU",
            prefill_container_runtime="docker",
            decode_container_runtime="docker",
        )
        settings = {
            "SMOKE_LONG_PREFIX_TARGET_TOKENS": "960000",
            "SMOKE_LONG_PREFIX_TP_SIZE": "1",
        }
        with mock.patch.dict(os.environ, settings, clear=True):
            for role in ("prefill", "decode"):
                with self.subTest(role=role):
                    _, _, _, command = driver.role_launch_parts(args, role)
                    for key, value in settings.items():
                        self.assertIn(f"{key}={value}", command)

    def test_role_command_forwards_core_dump_diagnostic_override(self):
        args = argparse.Namespace(
            prefill_repo_root="/prefill/repo",
            decode_repo_root="/decode/repo",
            prefill_checkpoint_path="/prefill/checkpoint",
            decode_checkpoint_path="/decode/checkpoint",
            prefill_sp_checkpoint_path="/prefill/mtp",
            decode_sp_checkpoint_path="/decode/mtp",
            prefill_endpoint="10.0.0.1:27188",
            decode_endpoint="10.0.0.2:28188",
            run_id="projection-ktp",
            suite="flow",
            result_endpoint=None,
            container="lhc_GPU",
            prefill_container_runtime="docker",
            decode_container_runtime="docker",
        )
        with mock.patch.dict(
            os.environ,
            {"FT_CORE_DUMP_ON_EXCEPTION": "0"},
            clear=True,
        ):
            _, _, _, command = driver.role_launch_parts(args, "prefill")
        self.assertIn("FT_CORE_DUMP_ON_EXCEPTION=0", command)

    def test_role_command_forwards_decode_cache_and_prewarm_timeout(self):
        args = argparse.Namespace(
            prefill_repo_root="/prefill/repo",
            decode_repo_root="/decode/repo",
            prefill_checkpoint_path="/prefill/checkpoint",
            decode_checkpoint_path="/decode/checkpoint",
            prefill_sp_checkpoint_path="/prefill/mtp",
            decode_sp_checkpoint_path="/decode/mtp",
            prefill_endpoint="10.0.0.1:27188",
            decode_endpoint="10.0.0.2:28188",
            run_id="projection-ktp",
            suite="all",
            result_endpoint=None,
            container="lhc_GPU",
            prefill_container_runtime="docker",
            decode_container_runtime="docker",
        )
        with mock.patch.dict(
            os.environ,
            {
                "SMOKE_PREFILL_KV_CACHE_MEM_MB": "56000",
                "SMOKE_DECODE_KV_CACHE_MEM_MB": "26000",
                "SMOKE_DECODE_ROLE_ADDRS": (
                    "10.0.0.2:28188:28189,10.0.0.2:28197:28198"
                ),
                "SMOKE_RDMA_PREWARM_TIMEOUT_S": "300",
            },
            clear=True,
        ):
            _, _, _, command = driver.role_launch_parts(args, "decode")
        self.assertIn("SMOKE_PREFILL_KV_CACHE_MEM_MB=56000", command)
        self.assertIn("SMOKE_DECODE_KV_CACHE_MEM_MB=26000", command)
        self.assertIn(
            "SMOKE_DECODE_ROLE_ADDRS=10.0.0.2:28188:28189,10.0.0.2:28197:28198",
            command,
        )
        self.assertIn("SMOKE_RDMA_PREWARM_TIMEOUT_S=300", command)

    def test_start_remote_roles_launches_prefill_without_decode_health_gate(self):
        events = []

        class FakeRole:
            def __init__(self, role):
                self.role = role

            def start(self):
                events.append(f"start:{self.role}")

        args = SimpleNamespace(prefill_start_delay_s=3.5)
        roles = {
            "decode": FakeRole("decode"),
            "prefill": FakeRole("prefill"),
        }
        with mock.patch.object(
            driver.time,
            "sleep",
            side_effect=lambda seconds: events.append(f"sleep:{seconds}"),
        ):
            driver.start_remote_roles(args, roles)

        self.assertEqual(
            events,
            ["start:decode", "sleep:3.5", "start:prefill"],
        )

    def _run_detached_poll_fixture(self, results, clock):
        args = SimpleNamespace(prefill_start_delay_s=0, overall_timeout=100)
        with ExitStack() as stack:
            for name in ("build_short_ssh_command", "build_detached_remote_command",
                         "build_detached_control_command", "fetch_detached_log", "show_tail"):
                stack.enter_context(mock.patch.object(driver, name, return_value="stub"))
            stack.enter_context(mock.patch.object(driver, "detached_control_paths",
                                                   return_value={"status": "/tmp/status"}))
            process = mock.Mock(returncode=0)
            process.communicate.return_value = ("", None)
            stack.enter_context(mock.patch.object(driver.subprocess, "Popen", return_value=process))
            stack.enter_context(mock.patch.object(driver, "run_short_ssh", side_effect=results))
            stack.enter_context(mock.patch.object(driver.time, "sleep"))
            stack.enter_context(mock.patch.object(driver.time, "monotonic", side_effect=clock))
            stop = stack.enter_context(mock.patch.object(driver, "stop_detached_role"))
            status = driver.run_detached(args, pathlib.Path("/unused"))
            return status, stop.call_args_list

    def test_detached_survives_auth_outage_longer_than_twelve_polls(self):
        unavailable = subprocess.CompletedProcess([], 255, "", "auth expired")
        done = subprocess.CompletedProcess([], 0, "0\n", "")
        status, stops = self._run_detached_poll_fixture(
            [unavailable] * 26 + [done, done], [0] * 15,
        )
        self.assertEqual(status, 0)
        self.assertEqual(stops, [])

    def test_detached_auth_outage_still_obeys_overall_deadline(self):
        unavailable = subprocess.CompletedProcess([], 255, "", "auth expired")
        status, stops = self._run_detached_poll_fixture(
            [unavailable, unavailable], [0, 0, 101],
        )
        self.assertEqual(status, 1)
        self.assertEqual([call.args[1] for call in stops], ["decode", "prefill"])

    def test_detached_real_worker_failure_still_stops_unfinished_peer(self):
        failed = subprocess.CompletedProcess([], 0, "1\n", "")
        running = subprocess.CompletedProcess([], 3, "", "")
        status, stops = self._run_detached_poll_fixture([failed, running], [0, 0])
        self.assertEqual(status, 1)
        self.assertEqual([call.args[1] for call in stops], ["prefill"])

    def test_detached_control_operations_run_inside_role_container(self):
        args = SimpleNamespace(
            prefill_repo_root="/prefill/repo",
            decode_repo_root="/decode/repo",
            prefill_checkpoint_path="/prefill/checkpoint",
            decode_checkpoint_path="/decode/checkpoint",
            prefill_sp_checkpoint_path="/prefill/mtp",
            decode_sp_checkpoint_path="/decode/mtp",
            prefill_endpoint="10.0.0.1:27188",
            decode_endpoint="10.0.0.2:28188",
            run_id="projection-ktp",
            suite="flow",
            result_endpoint=None,
            container="lhc_GPU",
            container_user="19357313:100",
            prefill_container_runtime="docker",
            decode_container_runtime="docker",
        )

        command = driver.build_detached_control_command(
            args, "decode", "cat /tmp/projection-ktp/decode.status"
        )

        self.assertEqual(
            shlex.split(command),
            [
                "docker",
                "exec",
                "-u",
                "19357313:100",
                "lhc_GPU",
                "bash",
                "-lc",
                "cat /tmp/projection-ktp/decode.status",
            ],
        )

    def test_role_specific_containers_override_shared_container(self):
        args = SimpleNamespace(
            prefill_repo_root="/prefill/repo",
            decode_repo_root="/decode/repo",
            prefill_checkpoint_path="/prefill/checkpoint",
            decode_checkpoint_path="/decode/checkpoint",
            prefill_sp_checkpoint_path="/prefill/mtp",
            decode_sp_checkpoint_path="/decode/mtp",
            prefill_endpoint="10.0.0.1:27188",
            decode_endpoint="10.0.0.2:28188",
            run_id="projection-ktp",
            suite="flow",
            result_endpoint=None,
            container="lhc_GPU",
            prefill_container="lhc_GPU_prefill",
            decode_container="lhc_GPU_decode",
            prefill_container_runtime="pouch",
            decode_container_runtime="pouch",
            container_user="luohaocheng.lhc",
        )

        self.assertEqual(
            driver.role_launch_parts(args, "prefill")[2], "lhc_GPU_prefill"
        )
        self.assertEqual(driver.role_launch_parts(args, "decode")[2], "lhc_GPU_decode")


if __name__ == "__main__":
    unittest.main()
