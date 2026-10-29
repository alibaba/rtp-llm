"""Offline checks for the K3 Bazel action launcher."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

LAUNCHER = Path(__file__).with_name("server_launcher.py")
spec = importlib.util.spec_from_file_location("k3_server_launcher", LAUNCHER)
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)


class CheckpointSourceTest(unittest.TestCase):
    def test_merge_profiles_select_role_executor_and_fp8_collectives(self):
        with tempfile.TemporaryDirectory() as tmp:
            target, draft = Path(tmp) / "target", Path(tmp) / "draft"
            target.mkdir()
            draft.mkdir()
            (target / "config.json").write_text('{"num_hidden_layers":93}')
            for role in ("PREFILL", "DECODE"):
                for fp8 in (False, True):
                    args = SimpleNamespace(
                        checkpoint=target,
                        draft_checkpoint=draft,
                        start_port=28000,
                        peer_port=29000,
                        peer_ip="127.0.0.1",
                        role=role,
                        server=sys.executable,
                        fp8_gemm=fp8,
                        orthogonal_smoke=True,
                    )
                    with self.subTest(role=role, fp8=fp8), patch.dict(
                        launch.os.environ, {}, clear=True
                    ):
                        env, argv = launch.launch_config(args)
                    options = dict(zip(argv[1::2], argv[2::2]))
                    self.assertEqual(
                        options["--moe_strategy"],
                        "mega_moe_k3" if role == "PREFILL" else "mega_moe_k3_se",
                    )
                    self.assertEqual(options["--prefill_cp_size"], "8")
                    self.assertEqual(env["MEGA_MOE_INPUT_PACKER_IMPL"], "optimized")
                    self.assertEqual(env["RTP_LLM_FP8_AG_GEMM"], str(int(fp8)))
                    self.assertEqual(env["RTP_LLM_FP8_GEMM_RS"], str(int(fp8)))
                    self.assertEqual(
                        env["KIMI_K3_SHARED_EXPERT_WEIGHT_SHARD"],
                        "1" if role == "PREFILL" else "0",
                    )
                    grpc = json.loads(env["GRPC_CONFIG_JSON"])
                    self.assertEqual(grpc["max_server_pollers"], 4)
                    self.assertEqual(
                        grpc["server_config"]["grpc.max_receive_message_length"],
                        1073741824,
                    )

    def test_default_cpu_tp_socket_location_is_unchanged(self):
        environment = launch.cpu_tp_socket_environment("/data/private/run")
        self.assertEqual(
            environment["RTP_LLM_CPU_TP_BROADCASTER_DIR"], "/data/private/run/uds"
        )
        self.assertEqual(environment["RTP_LLM_CPU_TP_BROADCASTER_ID"], "k3")
        with self.assertRaisesRegex(ValueError, "shorter than 108 bytes"):
            launch.cpu_tp_socket_environment("/data/" + "x" * 100)

    def test_both_role_launchers_accept_short_tmp_without_moving_logs_or_ipc(self):
        for role in ("PREFILL", "DECODE"):
            with self.subTest(role=role), tempfile.TemporaryDirectory(
                dir="/tmp"
            ) as tmp:
                root = Path(tmp)
                target, draft = root / "target", root / "draft"
                target.mkdir()
                draft.mkdir()
                (target / "config.json").write_text('{"num_hidden_layers":93}')
                ipc = root / "ipc"
                ipc.mkdir(mode=0o700)
                runtime_tmp = root / "runtime"
                runtime_tmp.mkdir(mode=0o700)
                artifacts = root / ("long-artifact-" * 8)
                artifacts.mkdir()
                run = artifacts / role
                argv = [
                    str(LAUNCHER),
                    "--role",
                    role,
                    "--checkpoint",
                    str(target),
                    "--draft-checkpoint",
                    str(draft),
                    "--peer-ip",
                    "127.0.0.1",
                    "--peer-port",
                    "28000",
                    "--start-port",
                    "29000",
                    "--server",
                    sys.executable,
                    "--allow-checkpoint-root",
                    str(root),
                    "--run-dir",
                    str(run),
                    "--cpu-tp-socket-dir",
                    str(ipc),
                    "--runtime-tmp-dir",
                    str(runtime_tmp),
                ]
                with patch.object(sys, "argv", argv), patch.object(
                    launch, "require_bazel_destination"
                ), patch.object(launch, "require_bazel_temporary"), patch.object(
                    launch.os, "geteuid", return_value=ipc.stat().st_uid
                ), patch.object(
                    launch,
                    "cpu_tp_socket_environment",
                    wraps=launch.cpu_tp_socket_environment,
                ) as environment, patch.object(
                    launch,
                    "require_rdma_device",
                ), patch.object(
                    launch, "rdma_hca_environment", return_value={}
                ), patch.object(
                    launch, "require_gpu_capacity"
                ), patch.object(
                    launch.subprocess, "run"
                ), patch.object(
                    launch.socket, "socket"
                ) as sockets, patch.object(
                    launch.os, "execve"
                ) as execute, patch.dict(
                    launch.os.environ,
                    {
                        "TMPDIR": "/parent-tmp",
                        "PYTORCH_ALLOC_CONF": "backend:cudaMallocAsync",
                        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
                        "GRPC_CLIENT_CHANNEL_BACKUP_POLL_INTERVAL_MS": "5000",
                    },
                    clear=True,
                ):
                    sockets.return_value.getsockname.return_value = ("127.0.0.1", 0)
                    launch.main()
                    self.assertEqual(launch.os.environ["TMPDIR"], "/parent-tmp")
                    self.assertEqual(
                        launch.os.environ["PYTORCH_ALLOC_CONF"],
                        "backend:cudaMallocAsync",
                    )
                    self.assertEqual(
                        launch.os.environ["PYTORCH_CUDA_ALLOC_CONF"],
                        "expandable_segments:False",
                    )
                    self.assertEqual(
                        launch.os.environ[
                            "GRPC_CLIENT_CHANNEL_BACKUP_POLL_INTERVAL_MS"
                        ],
                        "5000",
                    )
                environment.assert_called_once_with(run.resolve(), str(ipc))
                actual = execute.call_args.args[2]
                self.assertEqual(actual["TMPDIR"], str(runtime_tmp))
                self.assertEqual(actual["LOG_PATH"], str(run / "logs"))
                self.assertEqual(actual["RTP_LLM_CPU_TP_BROADCASTER_DIR"], str(ipc))
                recorded = json.loads((run / "launch.json").read_text())["environment"]
                poll_interval = "GRPC_CLIENT_CHANNEL_BACKUP_POLL_INTERVAL_MS"
                self.assertEqual(actual[poll_interval], "500")
                self.assertEqual(recorded[poll_interval], actual[poll_interval])
                for alias in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
                    self.assertEqual(
                        actual[alias],
                        "expandable_segments:True,large_segment_size_mb:128",
                    )
                    self.assertEqual(recorded[alias], actual[alias])
                self.assertTrue(run.is_dir())
                self.assertFalse((run / "uds").exists())
                self.assertFalse((run / "tmp").exists())
                self.assertTrue((run / "logs").is_dir())
                self.assertTrue(ipc.is_dir())
                self.assertTrue(runtime_tmp.is_dir())

    def test_allocator_profile_is_fixed_for_absent_blank_or_conflicting_aliases(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            target, draft = Path(tmp) / "target", Path(tmp) / "draft"
            target.mkdir()
            draft.mkdir()
            (target / "config.json").write_text('{"num_hidden_layers":93}')
            for role in ("PREFILL", "DECODE"):
                args = SimpleNamespace(
                    checkpoint=str(target),
                    draft_checkpoint=str(draft),
                    start_port=28000,
                    peer_port=29000,
                    peer_ip="127.0.0.1",
                    role=role,
                    server=sys.executable,
                )
                for inherited in (
                    {},
                    {"PYTORCH_CUDA_ALLOC_CONF": ""},
                    {
                        "PYTORCH_ALLOC_CONF": "expandable_segments:False",
                        "PYTORCH_CUDA_ALLOC_CONF": "backend:cudaMallocAsync",
                    },
                ):
                    with self.subTest(role=role, inherited=inherited), patch.dict(
                        launch.os.environ, inherited, clear=True
                    ):
                        environment, _ = launch.launch_config(args)
                        self.assertEqual(dict(launch.os.environ), inherited)
                    for alias in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
                        self.assertEqual(
                            environment[alias],
                            "expandable_segments:True,large_segment_size_mb:128",
                        )

    def test_pd_backup_poll_interval_is_fixed_before_grpc_initialization(self):
        key = "GRPC_CLIENT_CHANNEL_BACKUP_POLL_INTERVAL_MS"
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            target, draft = Path(tmp) / "target", Path(tmp) / "draft"
            target.mkdir()
            draft.mkdir()
            (target / "config.json").write_text('{"num_hidden_layers":93}')
            for role in ("PREFILL", "DECODE"):
                args = SimpleNamespace(
                    checkpoint=str(target),
                    draft_checkpoint=str(draft),
                    start_port=28000,
                    peer_port=29000,
                    peer_ip="127.0.0.1",
                    role=role,
                    server=sys.executable,
                )
                for inherited in ({}, {key: ""}, {key: "5000"}):
                    with self.subTest(role=role, inherited=inherited), patch.dict(
                        launch.os.environ, inherited, clear=True
                    ):
                        environment, _ = launch.launch_config(args)
                        self.assertEqual(dict(launch.os.environ), inherited)
                    self.assertEqual(environment[key], "500")

    def test_runtime_tmp_default_and_maximum_pid_path_budget(self):
        self.assertEqual(
            launch.runtime_tmp_environment("/data/run"), {"TMPDIR": "/data/run/tmp"}
        )
        run = "/" + "x" * 82
        endpoint = Path(run) / "tmp/symm_mem-2147483647"
        self.assertEqual(len(launch.os.fsencode(endpoint)), 107)
        launch.runtime_tmp_environment(run)
        with self.assertRaisesRegex(ValueError, "SymmetricMemory.*108 bytes"):
            launch.runtime_tmp_environment(run + "x")

    def test_long_default_runtime_tmp_fails_before_hardware_or_model_loading(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            ipc = Path(tmp) / "ipc"
            ipc.mkdir(mode=0o700)
            run = Path(tmp) / ("long-artifact-" * 8) / "service"
            argv = [
                str(LAUNCHER),
                "--role",
                "PREFILL",
                "--checkpoint",
                "/target",
                "--draft-checkpoint",
                "/draft",
                "--peer-ip",
                "127.0.0.1",
                "--peer-port",
                "28000",
                "--start-port",
                "29000",
                "--server",
                "/server",
                "--allow-checkpoint-root",
                str(Path(tmp)),
                "--run-dir",
                str(run),
                "--cpu-tp-socket-dir",
                str(ipc),
            ]
            with patch.object(sys, "argv", argv), patch.object(
                launch, "launch_config", return_value=({}, ["/server"])
            ), patch.object(launch, "require_bazel_destination"), patch.object(
                launch, "require_bazel_temporary"
            ), patch.object(
                launch, "require_rdma_device"
            ) as rdma, patch.object(
                launch.subprocess, "run"
            ) as preflight, patch.object(
                launch.os, "execve"
            ) as execute:
                with self.assertRaisesRegex(ValueError, "SymmetricMemory.*108 bytes"):
                    launch.main()
            rdma.assert_not_called()
            preflight.assert_not_called()
            execute.assert_not_called()
            self.assertFalse(run.exists())

    def test_explicit_runtime_tmp_requires_private_owner_and_real_directory(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            directory = Path(tmp) / "runtime"
            directory.mkdir(mode=0o700)
            self.assertEqual(
                launch.runtime_tmp_environment("/unused", directory),
                {"TMPDIR": str(directory.resolve())},
            )
            with patch.object(
                launch.os, "geteuid", return_value=directory.stat().st_uid + 1
            ):
                with self.assertRaisesRegex(ValueError, "owned private directory"):
                    launch.runtime_tmp_environment("/unused", directory)
            directory.chmod(0o755)
            with self.assertRaisesRegex(ValueError, "owned private directory"):
                launch.runtime_tmp_environment("/unused", directory)
            directory.chmod(0o700)
            link = Path(tmp) / "link"
            link.symlink_to(directory)
            with self.assertRaisesRegex(ValueError, "owned private directory"):
                launch.runtime_tmp_environment("/unused", link)

    def test_explicit_cpu_tp_directory_requires_private_owner_and_real_directory(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            directory = Path(tmp) / "ipc"
            directory.mkdir(mode=0o700)
            self.assertEqual(
                launch.cpu_tp_socket_environment("/data/run", directory)[
                    "RTP_LLM_CPU_TP_BROADCASTER_DIR"
                ],
                str(directory.resolve()),
            )
            with patch.object(
                launch.os, "geteuid", return_value=directory.stat().st_uid + 1
            ):
                with self.assertRaisesRegex(ValueError, "owned private directory"):
                    launch.cpu_tp_socket_environment("/data/run", directory)
            directory.chmod(0o755)
            with self.assertRaisesRegex(ValueError, "owned private directory"):
                launch.cpu_tp_socket_environment("/data/run", directory)
            directory.chmod(0o700)
            link = Path(tmp) / "link"
            link.symlink_to(directory)
            with self.assertRaisesRegex(ValueError, "owned private directory"):
                launch.cpu_tp_socket_environment("/data/run", link)

    def test_decode_merge_profile_disables_replay_and_allows_explicit_probe(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            root = Path(directory)
            target, draft = root / "target", root / "draft"
            target.mkdir()
            draft.mkdir()
            (target / "config.json").write_text('{"num_hidden_layers":93}')
            args = SimpleNamespace(
                checkpoint=str(target),
                draft_checkpoint=str(draft),
                start_port=28000,
                peer_port=29000,
                peer_ip="127.0.0.1",
                role="DECODE",
                server=__file__,
                moe_strategy="mega_moe",
                orthogonal_smoke=False,
            )
            with patch.dict(launch.os.environ):
                launch.os.environ.pop("ENABLE_LINEAR_REPLAY", None)
                environment, _ = launch.launch_config(args)
                self.assertEqual(environment["ENABLE_LINEAR_REPLAY"], "0")

                launch.os.environ["ENABLE_LINEAR_REPLAY"] = "1"
                environment, _ = launch.launch_config(args)
                self.assertEqual(environment["ENABLE_LINEAR_REPLAY"], "1")

    def test_each_pd_role_publishes_http_endpoints_for_frontend_routing(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            root = Path(directory)
            target, draft = root / "target", root / "draft"
            target.mkdir()
            draft.mkdir()
            (target / "config.json").write_text('{"num_hidden_layers":93}')
            for role in ("PREFILL", "DECODE"):
                args = SimpleNamespace(
                    checkpoint=str(target),
                    draft_checkpoint=str(draft),
                    start_port=28000,
                    peer_port=29000,
                    peer_ip="127.0.0.2",
                    role=role,
                    server=sys.executable,
                    moe_strategy="mega_moe",
                    orthogonal_smoke=False,
                )
                environment, _ = launch.launch_config(args)
                route = json.loads(environment["MODEL_SERVICE_CONFIG"])
                self.assertTrue(route["use_local"])
                endpoints = route["role_endpoints"][0]
                own = role.lower()
                peer = "decode" if role == "PREFILL" else "prefill"
                self.assertEqual(
                    endpoints[own + "_endpoint"]["address"], "127.0.0.1:28000"
                )
                self.assertEqual(
                    endpoints[peer + "_endpoint"]["address"], "127.0.0.2:29000"
                )
                self.assertEqual(endpoints["decode_endpoint"]["protocol"], "http")

    def test_legacy_bf16_decode_reuse_setting_is_unchanged(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            target, draft = root / "target", root / "draft"
            target.mkdir()
            draft.mkdir()
            (target / "config.json").write_text('{"num_hidden_layers":93}')
            args = SimpleNamespace(
                checkpoint=str(target),
                draft_checkpoint=str(draft),
                start_port=28000,
                peer_port=29000,
                peer_ip="127.0.0.1",
                role="DECODE",
                server=sys.executable,
                moe_strategy="mega_moe",
                orthogonal_smoke=False,
            )
            _, command = launch.launch_config(args)
            options = dict(zip(command[1::2], command[2::2]))
            self.assertEqual(options["--reuse_cache"], "1")

    def test_asymmetric_pd_topologies_and_sixteen_card_planning(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            target = root / "target"
            draft = root / "draft"
            target.mkdir()
            draft.mkdir()
            (target / "config.json").write_text(json.dumps({"num_hidden_layers": 93}))
            for role, tp, dp, source_tp, capture in (
                ("PREFILL", 4, 1, 4, None),
                ("DECODE", 8, 1, 4, "1,2,4,8,16,32,64"),
                ("PREFILL", 8, 1, 8, None),
                ("DECODE", 4, 2, 8, "1,2,4,8,16,32"),
                ("DECODE", 8, 2, 8, "1,2,4,8,16,32"),
                ("DECODE", 4, 4, 8, "1,2,4,8,16,32"),
            ):
                args = SimpleNamespace(
                    checkpoint=str(target),
                    draft_checkpoint=str(draft),
                    start_port=28000,
                    peer_port=29000,
                    peer_ip="127.0.0.1",
                    role=role,
                    server=sys.executable,
                    moe_strategy="mega_moe",
                    orthogonal_smoke=True,
                    tp_size=tp,
                    dp_size=dp,
                    ep_size=tp * dp,
                    prefill_source_tp_size=source_tp,
                )
                env, command = launch.launch_config(args)
                options = dict(zip(command[1::2], command[2::2]))
                self.assertEqual(options["--world_size"], str(tp * dp))
                self.assertEqual(options["--ep_size"], str(tp * dp))
                self.assertEqual(env["LOCAL_WORLD_SIZE"], str(tp * dp))
                self.assertEqual(len(env["CUDA_VISIBLE_DEVICES"].split(",")), tp * dp)
                if role == "DECODE":
                    self.assertEqual(options["--prefill_cp_size"], str(source_tp))
                    self.assertEqual(options["--decode_capture_config"], capture)
                    self.assertEqual(
                        int(options["--max_context_batch_size"]),
                        max(map(int, capture.split(","))),
                    )

    def test_mla_precision_is_explicit_for_both_pd_roles(self):
        with tempfile.TemporaryDirectory() as tmp:
            target, draft = Path(tmp) / "target", Path(tmp) / "draft"
            target.mkdir()
            draft.mkdir()
            (target / "config.json").write_text('{"num_hidden_layers":93}')
            for role in ("PREFILL", "DECODE"):
                for enabled in (False, True):
                    with self.subTest(role=role, fp8_mla=enabled):
                        args = SimpleNamespace(
                            checkpoint=str(target),
                            draft_checkpoint=str(draft),
                            start_port=28000,
                            peer_port=29000,
                            peer_ip="127.0.0.1",
                            role=role,
                            server=sys.executable,
                            fp8_kv_cache=enabled,
                            orthogonal_smoke=True,
                        )
                        with patch.dict(
                            launch.os.environ, {"FP8_MLA": str(int(not enabled))}
                        ):
                            env, _ = launch.launch_config(args)
                        self.assertEqual(env["FP8_MLA"], str(int(enabled)))
                        self.assertEqual(env["FP8_KV_CACHE"], str(int(enabled)))
                        self.assertEqual(env["SP_MODEL_TYPE"], "kimi_k3_mtp")
                        self.assertEqual(env["SP_ACT_TYPE"], "BF16")
                        self.assertEqual(env["SP_QUANTIZATION"], "")
                        if role == "PREFILL":
                            self.assertEqual(env["PREFILL_CHUNK_SIZE"], "65536")
                        else:
                            for key in (
                                "RTP_LLM_STREAM_ASYNC",
                                "RTP_LLM_DROP_BROAD_SYNC",
                                "RTP_LLM_DEVICE_INPUT",
                                "RTP_LLM_MTP_ASYNC_PREPARE",
                            ):
                                self.assertEqual(env[key], "1")

    def test_gpu_process_placeholder_is_not_an_occupier(self):
        gpu_rows = "".join(f"{index}, GPU-{index}, 274114\n" for index in range(8))
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(
                launch.subprocess,
                "check_output",
                side_effect=[
                    gpu_rows,
                    "GPU-0, [N/A], [N/A]\n",
                ],
            ):
                launch.require_gpu_capacity(Path(tmp), min_free_gib=250)
            self.assertTrue((Path(tmp) / "gpu-preflight.json").is_file())

    def test_orthogonal_profile_enables_cache_and_graph_by_role(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            target = root / "target"
            draft = root / "draft"
            target.mkdir()
            draft.mkdir()
            (target / "config.json").write_text(json.dumps({"num_hidden_layers": 93}))
            for role in ("PREFILL", "DECODE"):
                args = SimpleNamespace(
                    checkpoint=str(target),
                    draft_checkpoint=str(draft),
                    start_port=28000,
                    peer_port=29000,
                    peer_ip="127.0.0.1",
                    role=role,
                    server=sys.executable,
                    moe_strategy="mega_moe",
                    orthogonal_smoke=True,
                    fp8_gemm=True,
                    fp8_kv_cache=True,
                    memory_cache_size_mb=8192,
                    kv_cache_mem_mb=4096,
                )
                env, command = launch.launch_config(args)
                options = dict(zip(command[1::2], command[2::2]))
                self.assertEqual(env["RTP_MLA_PREFILL_EXPANDED_KV_BUDGET_GIB"], "6.0")
                self.assertEqual(env["QUANTIZATION"], "FP8_PER_BLOCK")
                self.assertEqual(env["FP8_KV_CACHE"], "1")
                self.assertEqual(env["FP8_MLA"], "1")
                self.assertEqual(env["SP_TYPE"], "mtp")
                self.assertEqual(env["SP_ACT_TYPE"], "BF16")
                self.assertEqual(options["--moe_strategy"], "mega_moe")
                self.assertEqual(options["--max_seq_len"], "2097152")
                self.assertEqual(options["--max_batch_tokens_size"], "262144")
                self.assertEqual(options["--max_batch_tokens_without_cache"], "65536")
                self.assertEqual(options["--concurrency_limit"], "64")
                self.assertEqual(options["--kv_cache_mem_mb"], "4096")
                self.assertEqual(
                    options["--enable_memory_cache"], "1" if role == "PREFILL" else "0"
                )
                self.assertEqual(
                    options["--reuse_cache"], "1" if role == "PREFILL" else "0"
                )
                self.assertEqual(
                    options["--enable_cuda_graph"], "0" if role == "PREFILL" else "1"
                )
                self.assertEqual(options["--fp8_kv_cache"], "1")
                if role == "PREFILL":
                    self.assertEqual(options["--max_context_batch_size"], "64")
                    self.assertEqual(options["--memory_cache_size_mb"], "8192")
                    self.assertEqual(options["--prefill_cp_kv_cache_sharded"], "1")
                else:
                    self.assertEqual(env["NCCL_GRAPH_REGISTER"], "0")
                    self.assertEqual(env["NCCL_MAX_CTAS"], "8")
                    self.assertEqual(options["--prefill_cp_kv_cache_sharded"], "1")
                    self.assertEqual(options["--prefill_cp_size"], "8")
                    self.assertEqual(
                        options["--decode_capture_config"], "1,2,4,8,16,32,64"
                    )

    def test_complete_prefill_smoke_bounds_device_cache_for_host_demotion(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            target, draft = root / "target", root / "draft"
            target.mkdir()
            draft.mkdir()
            (target / "config.json").write_text('{"num_hidden_layers":93}')
            for role in ("PREFILL", "DECODE"):
                args = SimpleNamespace(
                    checkpoint=str(target),
                    draft_checkpoint=str(draft),
                    start_port=28000,
                    peer_port=29000,
                    peer_ip="127.0.0.1",
                    role=role,
                    server=sys.executable,
                    moe_strategy="mega_moe",
                    orthogonal_smoke=True,
                )
                _, command = launch.launch_config(args)
                options = dict(zip(command[1::2], command[2::2]))
                if role == "PREFILL":
                    self.assertEqual(options["--kv_cache_mem_mb"], "4096")
                else:
                    self.assertEqual(options["--kv_cache_mem_mb"], "34000")

    def test_rpc_self_address_bypasses_proxy(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            target = root / "target"
            draft = root / "draft"
            target.mkdir()
            draft.mkdir()
            (target / "config.json").write_text(json.dumps({"num_hidden_layers": 93}))
            args = SimpleNamespace(
                checkpoint=str(target),
                draft_checkpoint=str(draft),
                start_port=28000,
                peer_port=29000,
                peer_ip="192.0.2.2",
                role="PREFILL",
                server=sys.executable,
                moe_strategy="mega_moe",
            )
            with patch.object(launch.socket, "socket") as route_socket:
                route_socket.return_value.getsockname.return_value = (
                    "192.0.2.1",
                    54321,
                )
                environment, _ = launch.launch_config(args)
            for key in ("NO_PROXY", "no_proxy"):
                self.assertIn("192.0.2.1", environment[key].split(","))
                self.assertIn("192.0.2.2", environment[key].split(","))

    def test_moe_input_packer_can_keep_safe_finite_check(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            target = root / "target"
            draft = root / "draft"
            target.mkdir()
            draft.mkdir()
            (target / "config.json").write_text(json.dumps({"num_hidden_layers": 93}))
            args = SimpleNamespace(
                checkpoint=str(target),
                draft_checkpoint=str(draft),
                start_port=28000,
                peer_port=29000,
                peer_ip="127.0.0.1",
                role="PREFILL",
                server=sys.executable,
                moe_strategy="mega_moe",
            )
            with patch.dict(
                launch.os.environ, {"MEGA_MOE_INPUT_PACKER_IMPL": "optimized"}
            ):
                environment, _ = launch.launch_config(args)
            self.assertEqual(environment["MEGA_MOE_INPUT_PACKER_IMPL"], "optimized")


if __name__ == "__main__":
    unittest.main()
