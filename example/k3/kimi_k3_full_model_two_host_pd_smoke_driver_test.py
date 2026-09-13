import argparse
import csv
import fcntl
import json
import os
import pathlib
import runpy
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from example.k3 import kimi_k3_full_model_two_host_pd_smoke_driver as driver

class ForwardedOptionalEnvironmentTest(unittest.TestCase):
    def test_draft_mode_defaults_and_explicit_overrides(self) -> None:
        for settings, expected in (
            ({}, ("mtp", "kimi_k3_mtp")),
            ({"SP_TYPE": "", "SP_MODEL_TYPE": ""}, ("mtp", "kimi_k3_mtp")),
            ({"SP_TYPE": "mtp"}, ("mtp", "kimi_k3_mtp")),
            ({"SP_TYPE": "eagle3"}, ("eagle3", "kimi_k3_mla_swa_eagle3")),
        ):
            with self.subTest(settings=settings), mock.patch.dict(os.environ, settings, clear=True):
                for role in ("prefill", "decode"):
                    env = driver.forwarded_optional_environment(role)
                    self.assertEqual((env["SP_TYPE"], env["SP_MODEL_TYPE"]), expected)

    def test_invalid_draft_modes_fail_before_remote_launch(self) -> None:
        for settings in (
            {"SP_TYPE": "none"},
            {"SP_MODEL_TYPE": "kimi_k3_mla_swa_eagle3"},
            {"SP_TYPE": "eagle3", "SP_MODEL_TYPE": "kimi_k3_mtp"},
        ):
            with self.subTest(settings=settings), mock.patch.dict(os.environ, settings, clear=True):
                for role in ("prefill", "decode"):
                    with self.assertRaises(ValueError):
                        driver.forwarded_optional_environment(role)

    def test_driver_role_launcher_capacity_profile_reaches_real_process(self):
        fixture_parent = pathlib.Path(
            os.environ.get("TEST_TMPDIR", tempfile.gettempdir())
        ).resolve()
        if not (
            fixture_parent.parts[1].startswith("data")
            or fixture_parent.parts[1] == "ssd"
        ):
            self.skipTest("native K3 role requires checkpoints on a local data disk")
        with tempfile.TemporaryDirectory(dir=fixture_parent) as tmp:
            root = pathlib.Path(tmp)
            repo = root / "repo"
            scripts = repo / "example/k3"
            scripts.mkdir(parents=True)
            original = pathlib.Path(driver.__file__).parent
            for name in (
                "kimi_k3_full_model_two_host_pd_smoke.sh",
                "start_kimi_k3_pd.sh",
                "kimi_k3_full_model_pd_cases.py",
                "kimi_k3_long_prefix_case.py",
                "kimi_k3_logits_compare.py",
                "kimi_k3_full_model_pd_nic_selection.py",
                "kimi_k3_rdma_readiness.py",
                "kimi_k3_cache_evidence.py",
            ):
                shutil.copy(original / name, scripts / name)
            image = pathlib.Path("rtp_llm/multimodal/test/testdata/qwen2_vl/1.jpg")
            (repo / image).parent.mkdir(parents=True)
            shutil.copy(original.parents[1] / image, repo / image)
            model, draft = root / "model", root / "draft"
            model.mkdir()
            draft.mkdir()
            (model / "config.json").write_text('{"num_hidden_layers":93}\n')
            (model / "model.safetensors.index.json").write_text("{}\n")
            (draft / "config.json").write_text('{"num_hidden_layers":1}\n')
            fake_server = root / "server"
            fake_server.write_text(
                f"#!{sys.executable}\nimport json, os, sys\nfrom pathlib import Path\n"
                "Path(os.environ['K3_TEST_ARGS_OUTPUT']).write_text(json.dumps({'args': sys.argv[1:], "
                "'moe_tokens': os.environ.get('MEGA_MOE_MAX_TOKENS_PER_RANK'), "
                "'test_blocks': os.environ.get('TEST_BLOCK_NUM'), "
                "'backup_poll': os.environ.get('GRPC_CLIENT_CHANNEL_BACKUP_POLL_INTERVAL_MS')}))\n"
                "sys.exit(17)\n"
            )
            fake_server.chmod(0o755)
            port_manager = runpy.run_path(
                str(original.parents[1] / "rtp_llm/test/utils/port_util.py")
            )["PortManager"]
            locks = []
            try:
                endpoints = []
                for _ in range(3):
                    ports, allocated = port_manager().get_consecutive_ports(201)
                    locks.extend(allocated)
                    endpoints.append(f"127.0.0.1:{ports[100]}")
                args = SimpleNamespace(
                    prefill_repo_root=str(repo),
                    decode_repo_root=str(repo),
                    prefill_checkpoint_path=str(model),
                    decode_checkpoint_path=str(model),
                    prefill_sp_checkpoint_path=str(draft),
                    decode_sp_checkpoint_path=str(draft),
                    prefill_container_runtime="docker",
                    decode_container_runtime="docker",
                    container="p",
                    decode_container="d",
                    container_user=str(os.getuid()),
                    prefill_endpoint=endpoints[0],
                    decode_endpoint=endpoints[1],
                    result_endpoint=endpoints[2],
                    suite="all",
                    run_id="profile",
                )
                for role in ("prefill", "decode"):
                    for prefill_sharded, decode_sharded, valid, topology in (
                        ("0", "0", True, "legacy"),
                        ("0", "1", True, "legacy"),
                        ("1", "0", True, "legacy"),
                        ("1", "1", True, "legacy"),
                        ("1", "1", False, "legacy"),
                        ("0", "0", True, "dp8_ktp8_ep8"),
                    ):
                        args.run_id = (
                            f"{role}-{prefill_sharded}-{decode_sharded}-{valid}-{topology}"
                        )
                        output = root / f"{args.run_id}.json"
                        settings = dict(
                            SMOKE_ARTIFACT_ROOT=str(root / "artifacts"),
                            RTP_LLM_SKIP_BUILD="1",
                            RTP_LLM_SERVER_BINARY=str(fake_server),
                            RTP_LLM_DRY_RUN="0",
                            K3_TEST_ARGS_OUTPUT=str(output),
                            SMOKE_SUITE="all",
                            SMOKE_MAX_CONCURRENCY="64",
                            KIMI_K3_DECODE_TOPOLOGY=topology,
                            KV_CACHE_MEM_MB="-1",
                            KIMI_K3_KDA_POOL_BLOCKS="0",
                            TEST_BLOCK_NUM="0",
                            MAX_BATCH_TOKENS_SIZE="",
                            TP_SIZE="8",
                            EP_SIZE="8",
                            SMOKE_EXPECTED_LAYERS="93",
                            SP_TYPE="eagle3",
                            SP_MODEL_TYPE="kimi_k3_mla_swa_eagle3",
                            PREFILL_CP_KV_CACHE_SHARDED=prefill_sharded,
                            DECODE_CP_KV_CACHE_SHARDED=decode_sharded,
                            SMOKE_STARTUP_TIMEOUT_S="5",
                            SMOKE_REQUEST_TIMEOUT_S="5",
                            SMOKE_RESULT_TIMEOUT_S="5",
                            DECODE_CAPTURE_CONFIG=(
                                "1,2,3,4,5,6,7,8,16,32,64" if valid else "1,2,2"
                            ),
                        )
                        with mock.patch.dict(os.environ, settings):
                            _, _, _, command = driver.role_launch_parts(args, role)
                            result = subprocess.run(
                                command,
                                cwd=repo,
                                capture_output=True,
                                text=True,
                                timeout=20,
                            )
                        # The fixture deliberately exits before health: this test covers configuration, not inference.
                        self.assertNotEqual(result.returncode, 0)
                        if not valid:
                            self.assertFalse(
                                output.exists(), result.stdout + result.stderr
                            )
                            self.assertIn(
                                "capture", (result.stdout + result.stderr).lower()
                            )
                            continue
                        self.assertTrue(output.exists(), result.stdout + result.stderr)
                        captured = json.loads(output.read_text())
                        self.assertEqual(captured["test_blocks"], "0")
                        self.assertEqual(captured["backup_poll"], "500")
                        expected_moe_tokens = "8192" if role == "prefill" else ("448" if topology == "dp8_ktp8_ep8" else "56")
                        self.assertEqual(captured["moe_tokens"], expected_moe_tokens)
                        params = captured["args"]
                        expected = {
                            "--concurrency_limit": "64",
                            "--kv_cache_mem_mb": "-1",
                            "--kimi_k3_kda_pool_blocks": "0",
                            "--max_batch_tokens_size": "67633601",
                            "--tp_size": "1" if role == "decode" and topology == "dp8_ktp8_ep8" else "8",
                            "--ep_size": "8",
                            "--seq_size_per_block": "4096",
                            "--kernel_seq_size_per_block": "128",
                            "--linear_step": "1",
                            "--prefill_cp_kv_cache_sharded": prefill_sharded,
                            "--decode_cp_kv_cache_sharded": decode_sharded,
                            "--enable_cuda_graph": "0" if role == "prefill" else "1",
                        }
                        if role == "decode":
                            expected["--decode_capture_config"] = settings[
                                "DECODE_CAPTURE_CONFIG"
                            ]
                        for name, value in expected.items():
                            self.assertEqual(
                                params[params.index(name) + 1], value, name
                            )
            finally:
                for lock in locks:
                    lock.__exit__(None, None, None)

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
            decode_container=None,
            container_user="luohaocheng.lhc",
            prefill_endpoint="prefill:27188",
            decode_endpoint="decode:28188",
            run_id="precision-test",
            suite="all",
            result_endpoint=None,
        )
        for weight in ("none", "fp8_per_block"):
            for mla in ("0", "1"):
                for budget in ("0", "268435456"):
                    settings = {
                        "KIMI_K3_ATTENTION_QUANTIZATION": weight,
                        "KIMI_K3_MLA_FP8": mla,
                        "KIMI_K3_MLA_FP8_Q_SCALE": "0.5",
                        "KIMI_K3_MLA_FP8_KV_SCALE": "0.5",
                        "KIMI_K3_MLA_PREFILL_EXPANDED_KV_BUDGET_BYTES": budget,
                        "LOAD_METHOD": "fastsafetensors",
                    }
                    with (
                        self.subTest(weight=weight, mla=mla, budget=budget),
                        mock.patch.dict(os.environ, settings, clear=True),
                    ):
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

    def test_role_commands_preserve_independent_cache_sharding(self) -> None:
        required = {
            "PREFILL_SSH_TARGET": "p-host",
            "DECODE_SSH_TARGET": "d-host",
            "PREFILL_REPO_ROOT": "/repo",
            "DECODE_REPO_ROOT": "/repo",
            "PREFILL_CHECKPOINT_PATH": "/model",
            "DECODE_CHECKPOINT_PATH": "/model",
            "PREFILL_SP_CHECKPOINT_PATH": "/draft",
            "DECODE_SP_CHECKPOINT_PATH": "/draft",
            "PREFILL_ENDPOINT": "p-host:27188",
            "DECODE_ENDPOINT": "d-host:28188",
            "SMOKE_RUN_ID": "role-flags",
        }
        for prefill, decode in (
            (None, None), ("0", "0"), ("0", "1"), ("1", "0"), ("1", "1")
        ):
            env = dict(required)
            if prefill is not None:
                env["PREFILL_CP_KV_CACHE_SHARDED"] = prefill
                env["DECODE_CP_KV_CACHE_SHARDED"] = decode
            with (
                self.subTest(prefill=prefill, decode=decode),
                mock.patch.dict(os.environ, env, clear=True),
                mock.patch.object(driver.sys, "argv", ["driver"]),
            ):
                args = driver.parse_args()
                for role in ("prefill", "decode"):
                    command = driver.role_launch_parts(args, role)[3]
                    self.assertIn(
                        f"PREFILL_CP_KV_CACHE_SHARDED={prefill or '0'}", command
                    )
                    self.assertIn(
                        f"DECODE_CP_KV_CACHE_SHARDED={decode or '0'}", command
                    )
                    self.assertFalse(
                        any(part.startswith("CP_ROTATE_METHOD=") for part in command)
                    )
                    args.decode_container = "decode-runtime"
                    self.assertEqual(
                        driver.role_launch_parts(args, "decode")[2], "decode-runtime"
                    )
                    self.assertIn(
                        "decode-runtime",
                        shlex.split(
                            driver.build_detached_control_command(args, "decode", "true")
                        ),
                    )
                    self.assertIn(
                        args.container,
                        shlex.split(
                            driver.build_detached_control_command(args, "prefill", "true")
                        ),
                    )

    def test_validation_parameters_reach_both_role_environments(self):
        values = {
            "SMOKE_MAX_CONCURRENCY": "32",
            "DECODE_CAPTURE_CONFIG": "1,2,3,4,5,6,7,8,16,32",
            "SMOKE_REQUEST_NAMESPACE": "matched",
            "kmonitorEnableLogFileSink": "true",
            "kmonitorNormalSamplePeriod": "1",
        }
        with mock.patch.dict(os.environ, values, clear=True):
            for role in ("prefill", "decode"):
                forwarded = driver.forwarded_optional_environment(role)
                for key, value in values.items():
                    self.assertEqual(forwarded[key], value)

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

    def test_forwards_explicit_graph_debug_without_inventing_a_default(self):
        with mock.patch.dict(os.environ, {"ENABLE_CUDA_GRAPH_DEBUG_MODE": "1"}, clear=True):
            self.assertEqual(driver.forwarded_optional_environment("decode")["ENABLE_CUDA_GRAPH_DEBUG_MODE"], "1")
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertNotIn("ENABLE_CUDA_GRAPH_DEBUG_MODE", driver.forwarded_optional_environment("decode"))

    def test_real_port_blocks_are_disjoint_held_and_released(self):
        repo = pathlib.Path(driver.__file__).resolve().parents[2]
        processes = []
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            try:
                allocations = []
                for role in ("prefill", "decode"):
                    directory = root / role
                    directory.mkdir()
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            driver.__file__,
                            "--hold-ports",
                            str(repo),
                            str(directory),
                            "30",
                        ],
                        env=dict(os.environ, TMPDIR=tmp),
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    processes.append((process, directory))
                    deadline = time.monotonic() + 10
                    while not (directory / "allocation.json").exists():
                        if process.poll() is not None or time.monotonic() > deadline:
                            self.fail(f"{role} port allocation did not complete")
                        time.sleep(0.02)
                    allocation = json.loads((directory / "allocation.json").read_text())
                    allocations.append(allocation)
                    self.assertEqual(
                        allocation["service_port"], allocation["first_port"] + 100
                    )
                    self.assertEqual(allocation["result_port"], allocation["last_port"])
                    with (
                        root
                        / "test_port_locks"
                        / f"port_{allocation['first_port']}.lock"
                    ).open("r+") as handle:
                        with self.assertRaises(BlockingIOError):
                            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                a, b = allocations
                self.assertTrue(
                    a["last_port"] < b["first_port"] or b["last_port"] < a["first_port"]
                )
                for process, directory in processes:
                    (directory / "release").touch()
                    self.assertEqual(process.wait(timeout=5), 0)
                    self.assertEqual((directory / "status").read_text(), "released")
                self.assertFalse(list((root / "test_port_locks").glob("port_*.lock")))
            finally:
                for process, directory in processes:
                    (directory / "release").touch()
                    if process.poll() is None:
                        process.terminate()
                    process.wait(timeout=5)

    def test_port_holder_expiration_releases_locks(self):
        repo = pathlib.Path(driver.__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as tmp:
            process = subprocess.run(
                [sys.executable, driver.__file__, "--hold-ports", str(repo), tmp, "1"],
                env=dict(os.environ, TMPDIR=tmp),
                capture_output=True,
                timeout=10,
            )
            self.assertEqual(process.returncode, 1, process.stderr)
            self.assertEqual((pathlib.Path(tmp) / "status").read_text(), "expired")
            self.assertFalse(
                list((pathlib.Path(tmp) / "test_port_locks").glob("port_*.lock"))
            )

    def test_main_resolves_endpoints_before_roles_and_cleans_partial_failure(self):
        for fail_decode in (False, True):
            with (
                self.subTest(fail_decode=fail_decode),
                tempfile.TemporaryDirectory() as tmp,
            ):
                args = SimpleNamespace(
                    artifact_root=pathlib.Path(tmp),
                    run_id="ports",
                    dry_run=False,
                    suite="all",
                    prefill_endpoint="p",
                    decode_endpoint="d:0",
                    result_endpoint=None,
                )

                def allocate(_args, role, leases):
                    leases.append((role, f"/ports/{role}"))
                    if fail_decode and role == "decode":
                        raise RuntimeError("allocation failed")
                    port = 15000 if role == "prefill" else 18000
                    return {"service_port": port, "result_port": port + 100}

                with (
                    mock.patch.object(driver, "parse_args", return_value=args),
                    mock.patch.object(driver.signal, "signal"),
                    mock.patch.object(
                        driver, "allocate_remote_ports", side_effect=allocate
                    ),
                    mock.patch.object(
                        driver, "release_port_leases", return_value=True
                    ) as release,
                    mock.patch.object(driver, "run_smoke", return_value=0) as smoke,
                ):
                    self.assertEqual(driver.main(), int(fail_decode))
                self.assertEqual(len(release.call_args.args[1]), 2)
                if fail_decode:
                    smoke.assert_not_called()
                else:
                    self.assertEqual(
                        (
                            args.prefill_endpoint,
                            args.decode_endpoint,
                            args.result_endpoint,
                        ),
                        ("p:15000", "d:18000", "d:18100"),
                    )
                    smoke.assert_called_once()


class KimiK3FullModelTwoHostPdSmokeDriverTest(unittest.TestCase):

    def test_role_cleanup_reaps_a_term_ignoring_group_after_leader_exits(self):
        import ctypes
        import signal
        libc = ctypes.CDLL(None, use_errno=True)
        previous = ctypes.c_int()
        self.assertEqual(libc.prctl(37, ctypes.byref(previous), 0, 0, 0), 0)
        self.assertEqual(libc.prctl(36, 1, 0, 0, 0), 0)
        script = pathlib.Path(driver.__file__).with_name("kimi_k3_full_model_two_host_pd_smoke.sh").read_text()
        start = script.index("stop_owned_process() {")
        cleanup = script[start:script.index("\n}", start) + 2]
        helper = "import os,signal,time\nsignal.signal(signal.SIGTERM,signal.SIG_IGN)\npid=os.fork()\nif pid: os._exit(0)\nprint(os.getpid(),flush=True)\ntime.sleep(60)\n"
        process = None
        child = None
        reaped = False
        try:
            process = subprocess.Popen([sys.executable, "-u", "-c", helper], stdout=subprocess.PIPE,
                                       text=True, start_new_session=True)
            child = int(process.stdout.readline())
            self.assertEqual(process.wait(timeout=5), 0)
            cleanup_start = script.index("cleanup() {")
            cleanup_body = script[cleanup_start:script.index("\n}", cleanup_start) + 2]
            with tempfile.TemporaryDirectory() as directory:
                marker = pathlib.Path(directory) / "cleaning"
                summary = pathlib.Path(directory) / "summary"
                body = (cleanup.replace("stop_owned_process()", "native_stop_owned_process()", 1)
                        + "\nstop_owned_process() { : > " + shlex.quote(str(marker)) + "; native_stop_owned_process \"$@\"; }\n"
                        + cleanup_body + "\nrole=decode; notified=1; listener_pid=; checkpoint_real=/model; role_dir=/role\n"
                        + f"service_pid={process.pid}; summary_file={shlex.quote(str(summary))}\ntrap cleanup EXIT\nexit 2\n")
                role = subprocess.Popen(["bash", "-c", body], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        text=True, start_new_session=True)
                try:
                    for _ in range(50):
                        if marker.exists():
                            break
                        time.sleep(0.02)
                    self.assertTrue(marker.exists())
                    os.killpg(role.pid, signal.SIGTERM)
                    stdout, stderr = role.communicate(timeout=20)
                    self.assertEqual(role.returncode, 2, stdout + stderr)
                    self.assertIn("status=2", summary.read_text())
                finally:
                    if role.poll() is None:
                        os.killpg(role.pid, signal.SIGKILL)
                        role.wait()
            status = None
            for _ in range(50):
                pid, state = os.waitpid(child, os.WNOHANG)
                if pid:
                    reaped, status = True, state
                    break
                time.sleep(0.1)
            self.assertIsNotNone(status, "orphan worker survived native role cleanup")
            self.assertTrue(os.WIFSIGNALED(status))
            self.assertEqual(os.WTERMSIG(status), signal.SIGKILL)
        finally:
            if process is not None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.stdout.close()
            if child is not None and not reaped:
                os.waitpid(child, 0)
            libc.prctl(36, previous.value, 0, 0, 0)


    def test_detached_start_gate_preserves_role_lifecycle(self):
        for deferred in (None, "prefill", "decode"):
            for outcome in ("success", "peer_failure", "launch_failure"):
                with self.subTest(deferred=deferred, outcome=outcome), tempfile.TemporaryDirectory() as tmp:
                    start_file = pathlib.Path(tmp) / "release"
                    args = SimpleNamespace(defer_role=deferred, start_file=start_file,
                                           prefill_start_delay_s=0, overall_timeout=60)
                    started, polled = [], []

                    def launch(args, role):
                        if role == deferred:
                            self.assertTrue(start_file.is_file())
                        started.append(role)
                        if outcome == "launch_failure" and role == "prefill":
                            raise RuntimeError("SSH lost after remote startup")

                    def poll(args, role, command):
                        self.assertIn(role, started)
                        polled.append(role)
                        if outcome == "peer_failure":
                            # The marker may appear just as the first role exits.
                            start_file.touch()
                            return SimpleNamespace(returncode=0, stdout="17", stderr="")
                        if len(started) == 1:
                            start_file.touch()
                            return SimpleNamespace(returncode=3, stdout="", stderr="")
                        return SimpleNamespace(returncode=0, stdout="0", stderr="")

                    with (mock.patch.object(driver, "launch_detached_role", side_effect=launch),
                          mock.patch.object(driver, "run_short_ssh", side_effect=poll),
                          mock.patch.object(driver, "detached_control_paths", return_value={"status": "/status"}),
                          mock.patch.object(driver, "build_detached_control_command", return_value="poll"),
                          mock.patch.object(driver, "stop_detached_role") as stop,
                          mock.patch.object(driver, "fetch_detached_log"),
                          mock.patch.object(driver, "show_tail"),
                          mock.patch.object(driver.time, "sleep")):
                        result = driver.run_detached(args, pathlib.Path(tmp))
                    self.assertEqual(result, 0 if outcome == "success" else 1)
                    self.assertEqual(len(started), len(set(started)))
                    if outcome == "success":
                        self.assertEqual(set(started), {"prefill", "decode"})
                        self.assertEqual(set(polled), set(started))
                        stop.assert_not_called()
                    elif outcome == "peer_failure" and deferred:
                        self.assertNotIn(deferred, started)
                    elif outcome == "launch_failure":
                        self.assertIn(mock.call(args, "prefill"), stop.call_args_list)
                    self.assertTrue(all(call.args[1] in started for call in stop.call_args_list))

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
            "/prefill/eagle3",
            "--decode-sp-checkpoint-path",
            "/decode/eagle3",
            "--prefill-endpoint",
            "10.0.0.1:27188",
            "--decode-endpoint",
            "10.0.0.2:28188",
            "--run-id",
            "projection-ktp",
        ]
        with mock.patch.object(sys, "argv", argv), mock.patch.dict(os.environ, {}, clear=True):
            args = driver.parse_args()
        self.assertEqual(args.prefill_sp_checkpoint_path, "/prefill/eagle3")
        self.assertEqual(args.decode_sp_checkpoint_path, "/decode/eagle3")
        self.assertEqual(args.prefill_start_delay_s, 15)
        for options in (
            ["--defer-role", "prefill"],
            ["--start-file", "/release"],
            ["--defer-role", "prefill", "--start-file", "/release"],
            ["--remote-detached", "--defer-role", "prefill", "--start-file", "relative"],
        ):
            with mock.patch.object(sys, "argv", argv + options), mock.patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(SystemExit):
                    driver.parse_args()
        with mock.patch.object(sys, "argv", argv + ["--remote-detached", "--defer-role", "prefill", "--start-file", "/release"]), mock.patch.dict(os.environ, {}, clear=True):
            parsed = driver.parse_args()
            self.assertEqual(parsed.defer_role, "prefill")
            self.assertEqual(parsed.start_file, pathlib.Path("/release"))
        for options in (["--parallel-start"], ["--parallel-start", "--remote-detached"]):
            with self.subTest(options=options), mock.patch.object(sys, "argv", argv + options), mock.patch.dict(os.environ, {}, clear=True):
                args = driver.parse_args()
            roles = {name: mock.Mock() for name in ("prefill", "decode")}
            with mock.patch.object(driver.time, "sleep") as sleep:
                driver.start_remote_roles(args, roles)
            self.assertEqual(args.prefill_start_delay_s, 0)
            sleep.assert_not_called()
            for role in roles.values():
                role.start.assert_called_once_with()

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
        with mock.patch.object(sys, "argv", argv), mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(SystemExit):
                driver.parse_args()

    def test_role_command_exports_role_local_sp_checkpoint(self):
        args = argparse.Namespace(
            prefill_repo_root="/prefill/repo",
            decode_repo_root="/decode/repo",
            prefill_checkpoint_path="/prefill/checkpoint",
            decode_checkpoint_path="/decode/checkpoint",
            prefill_sp_checkpoint_path="/prefill/eagle3",
            decode_sp_checkpoint_path="/decode/eagle3",
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
        self.assertIn("SP_CHECKPOINT_PATH=/decode/eagle3", command)

    def test_role_command_forwards_core_dump_diagnostic_override(self):
        args = argparse.Namespace(
            prefill_repo_root="/prefill/repo",
            decode_repo_root="/decode/repo",
            prefill_checkpoint_path="/prefill/checkpoint",
            decode_checkpoint_path="/decode/checkpoint",
            prefill_sp_checkpoint_path="/prefill/eagle3",
            decode_sp_checkpoint_path="/decode/eagle3",
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
            prefill_sp_checkpoint_path="/prefill/eagle3",
            decode_sp_checkpoint_path="/decode/eagle3",
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
                "SMOKE_DECODE_KV_CACHE_MEM_MB": "26000",
                "SMOKE_DECODE_KDA_POOL_BLOCKS": "40",
                "SMOKE_DECODE_ROLE_ADDRS": (
                    "10.0.0.2:28188:28189,10.0.0.2:28197:28198"
                ),
                "SMOKE_RDMA_PREWARM_TIMEOUT_S": "300",
            },
            clear=True,
        ):
            _, _, _, command = driver.role_launch_parts(args, "decode")
        self.assertIn("SMOKE_DECODE_KV_CACHE_MEM_MB=26000", command)
        self.assertIn("SMOKE_DECODE_KDA_POOL_BLOCKS=40", command)
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
            driver.time, "sleep", side_effect=lambda seconds: events.append(f"sleep:{seconds}")
        ):
            driver.start_remote_roles(args, roles)

        self.assertEqual(
            events,
            ["start:decode", "sleep:3.5", "start:prefill"],
        )

    def test_detached_control_operations_run_inside_role_container(self):
        args = SimpleNamespace(
            prefill_repo_root="/prefill/repo",
            decode_repo_root="/decode/repo",
            prefill_checkpoint_path="/prefill/checkpoint",
            decode_checkpoint_path="/decode/checkpoint",
            prefill_sp_checkpoint_path="/prefill/eagle3",
            decode_sp_checkpoint_path="/decode/eagle3",
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
            prefill_sp_checkpoint_path="/prefill/eagle3",
            decode_sp_checkpoint_path="/decode/eagle3",
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

        self.assertEqual(driver.role_launch_parts(args, "prefill")[2], "lhc_GPU_prefill")
        self.assertEqual(driver.role_launch_parts(args, "decode")[2], "lhc_GPU_decode")


if __name__ == "__main__":
    unittest.main()
