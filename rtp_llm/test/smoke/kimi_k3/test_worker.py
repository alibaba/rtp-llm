"""Offline regression gates for native Bazel PD startup and scoped cleanup."""

import contextlib
import hashlib
import io
import json
import os
import socket
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pd_config
import server_launcher as launcher
import worker


def config():
    host = {
        "ip": "192.0.2.1",
        "artifact_root": "/data/results",
        "checkpoint": "/data/target",
        "draft_checkpoint": "/data/draft",
        "allow_checkpoint_root": "/data",
        "port": 27188,
    }
    return {
        "external_revision": "a" * 40,
        "internal_revision": "b" * 40,
        "prefill": host,
        "decode": {**host, "ip": "192.0.2.2", "port": 28188},
        "decode_dp": 2,
    }


class NativePdSmokeTest(unittest.TestCase):
    def test_native_config_rejects_transport_fields_and_same_host(self):
        self.assertEqual(
            pd_config.validate_config(config(), local_role="prefill"), config()
        )
        for key in ("host", "runtime", "container", "user", "repo", "worker", "guard"):
            with self.subTest(key=key):
                item = config()
                item["prefill"][key] = "unused"
                with self.assertRaisesRegex(ValueError, "must not specify"):
                    pd_config.validate_config(item, local_role="prefill")
        item = config()
        item["decode"]["ip"] = item["prefill"]["ip"]
        with self.assertRaisesRegex(ValueError, "distinct machines"):
            pd_config.validate_config(item, local_role="prefill")

    def test_config_rejects_invalid_pins_topology_and_resource_values(self):
        for key, value in (
            ("external_revision", "branch"),
            ("internal_revision", "b" * 39),
            ("decode_dp", True),
            ("decode_dp", 3),
            ("fp8_gemm", 1),
            ("model_warm_up", "false"),
            ("moe_strategy", "other"),
            ("startup_timeout_s", 0),
        ):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                item = config()
                item[key] = value
                pd_config.validate_config(item, local_role="prefill")
        for key, value in (
            ("checkpoint", "relative"),
            ("ip", "127.0.0.1"),
            ("port", 65500),
            ("expected_draft_shards", True),
            ("allow_shared_accuracy", "false"),
            ("min_free_gib", float("nan")),
            ("kv_cache_mem_mb", 1.5),
        ):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                item = config()
                item["prefill"][key] = value
                pd_config.validate_config(item, local_role="prefill")

    def test_gpu_context_parser_rejects_malformed_output(self):
        self.assertEqual(pd_config.gpu_contexts("GPU-0, 42\n"), {("GPU-0", "42")})
        self.assertEqual(pd_config.gpu_contexts(""), set())
        for value in ("GPU-0", "GPU-0, error", ", 42"):
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                pd_config.gpu_contexts(value)

    def test_native_output_directory_must_be_owned_bazel_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "outputs"
            output.mkdir()
            outside = root / "unrelated"
            outside.mkdir()
            with patch.dict(os.environ, {"TEST_UNDECLARED_OUTPUTS_DIR": str(output)}):
                worker.Worker._require_native_artifact_root(output)
                with self.assertRaisesRegex(ValueError, "inside Bazel test outputs"):
                    worker.Worker._require_native_artifact_root(outside)
            with patch.dict(os.environ, {"TEST_UNDECLARED_OUTPUTS_DIR": ""}):
                with self.assertRaisesRegex(
                    ValueError, "requires TEST_UNDECLARED_OUTPUTS_DIR"
                ):
                    worker.Worker._require_native_artifact_root(output)

    def test_native_role_does_not_need_a_peer_artifact_directory(self):
        for local_role, peer in (("prefill", "decode"), ("decode", "prefill")):
            with self.subTest(local_role=local_role):
                item = config()
                del item[peer]["artifact_root"]
                pd_config.validate_config(item, local_role=local_role)
                del item[local_role]["artifact_root"]
                with self.assertRaises(KeyError):
                    pd_config.validate_config(item, local_role=local_role)

    def test_native_prepare_records_declared_pins_and_actual_runtime_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "worker.runfiles" / "rtp_llm"
            runtime = workspace / "rtp_llm/test/smoke/kimi_k3"
            runtime.mkdir(parents=True)
            guard = runtime / "checkpoint_guard.py"
            guard.write_text("packaged guard\n")
            alog = workspace / "rtp_llm/config/alog.conf"
            alog.parent.mkdir()
            alog.write_text("test logging config\n")
            server = workspace / "rtp_llm/rtp_llm_server"
            server.write_text("declared server executable\n")
            output = root / "outputs"
            output.mkdir()
            item = config()
            for name in ("checkpoint", "draft_checkpoint"):
                checkpoint = root / name
                checkpoint.mkdir()
                (checkpoint / "config.json").write_text(
                    json.dumps({"num_hidden_layers": 93})
                )
                (checkpoint / "model.safetensors").write_bytes(b"fixture")
                for role in ("prefill", "decode"):
                    item[role][name] = str(checkpoint)
                    item[role]["artifact_root"] = str(output)
            del item["decode"]["artifact_root"]
            role = worker.Worker(threading.Event())
            role.gpu_uuids = [f"GPU-{i}" for i in range(8)]
            with patch.object(
                worker, "__file__", str(runtime / "worker.py")
            ), patch.object(
                worker.subprocess,
                "check_output",
                side_effect=AssertionError("No Git or SSH required"),
            ), patch.object(
                role, "_create_socket_directory"
            ), patch.object(
                role, "_create_runtime_tmp_directory"
            ), patch.object(
                role, "_lock_gpus"
            ) as gpu_lock, patch.dict(
                os.environ, {"TEST_UNDECLARED_OUTPUTS_DIR": str(output)}
            ):
                invalid = dict(item, external_revision="branch")
                with self.assertRaisesRegex(ValueError, "complete Git revision"):
                    role.prepare_native(invalid, "prefill", "k3-pd-test123")
                gpu_lock.assert_not_called()
                self.assertIsNone(role.task)
                result = role.prepare_native(item, "prefill", "k3-pd-test123")
                gpu_lock.assert_called_once()
            self.assertEqual(result["execution_mode"], "bazel-remote")
            self.assertEqual(result["external_revision"], item["external_revision"])
            self.assertEqual(result["internal_revision"], item["internal_revision"])
            self.assertEqual(result["source_identity_kind"], "ci-declared-revisions")
            self.assertNotIn("source_identity", result)
            self.assertNotIn("external_tree", result)
            self.assertEqual(role.server, server)
            self.assertEqual(
                result["guard_sha256"], hashlib.sha256(guard.read_bytes()).hexdigest()
            )
            self.assertEqual(result["gpu_uuids"], role.gpu_uuids)
            self.assertEqual(
                result["checkpoint_identity"]["shards"], [("model.safetensors", 7)]
            )

    def test_gpu_locks_are_all_or_nothing_and_never_kill_occupants(self):
        rows = "\n".join(f"{i}, GPU-{i}, NVIDIA B300" for i in range(8))
        with tempfile.TemporaryDirectory() as tmp:
            first = worker.Worker(threading.Event())
            second = worker.Worker(threading.Event())
            with patch.object(
                worker.subprocess, "check_output", side_effect=[rows, "", rows]
            ):
                first._lock_gpus(Path(tmp))
                with self.assertRaises(BlockingIOError):
                    second._lock_gpus(Path(tmp))
                second.close()
                first.close()
            with patch.object(
                worker.subprocess, "check_output", side_effect=[rows, "GPU-0, 123"]
            ), patch.object(worker.os, "killpg") as kill:
                occupied = worker.Worker(threading.Event())
                with self.assertRaisesRegex(ValueError, "already present"):
                    occupied._lock_gpus(Path(tmp))
                occupied.close()
                kill.assert_not_called()

    def test_shared_worker_keeps_gpu_locks_and_records_existing_contexts(self):
        rows = "\n".join(f"{i}, GPU-{i}, NVIDIA B300" for i in range(8))
        with tempfile.TemporaryDirectory() as tmp:
            first = worker.Worker(threading.Event())
            first.host = {"allow_shared_accuracy": True}
            first.task = Path(tmp)
            second = worker.Worker(threading.Event())
            second.host = {"allow_shared_accuracy": True}
            with patch.object(
                worker.subprocess,
                "check_output",
                side_effect=[rows, "GPU-0, 123\n", rows],
            ), patch.object(worker.os, "killpg") as kill:
                first._lock_gpus(Path(tmp))
                self.assertEqual(first.baseline_gpu_contexts, {("GPU-0", "123")})
                with self.assertRaises(BlockingIOError):
                    second._lock_gpus(Path(tmp))
                second.close()
                first.close()
                kill.assert_not_called()
            evidence = json.loads((Path(tmp) / "gpu-contexts-prepare.json").read_text())
            self.assertTrue(evidence["allow_shared_accuracy"])
            self.assertFalse(evidence["performance_validated"])

    def test_role_launch_preserves_optimized_profile_and_eight_gpu_environment(self):
        item = config()
        item.update(fp8_gemm=True, fp8_kv_cache=True, moe_strategy="mega_moe_se")
        item["decode"].update(
            moe_strategy="mega_moe_k3_se",
            rdma_hcas="mlx5_bond_0",
            kv_cache_mem_mb=34000,
            allow_shared_accuracy=True,
        )
        role = worker.Worker(threading.Event())
        role.config, role.role, role.host = item, "decode", item["decode"]
        role.server = Path("/bazel/runfiles/rtp_llm/rtp_llm_server")
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            worker.subprocess, "Popen", return_value=SimpleNamespace(pid=42)
        ) as launch, patch.object(
            worker.subprocess, "check_output", return_value="GPU-0, 123\n"
        ):
            role.task = Path(tmp)
            role._create_socket_directory()
            role._create_runtime_tmp_directory()
            try:
                with patch.dict(
                    worker.os.environ,
                    {
                        "GPU_COUNT": "0",
                        "FT_SERVER_TEST": "1",
                        "TZ": "Asia/Shanghai",
                        "LD_LIBRARY_PATH": "/remote/native/lib",
                        "LD_PRELOAD": "/remote/native/preload.so",
                    },
                ):
                    role.start()
                argv = launch.call_args.args[0]
                self.assertEqual(
                    Path(launch.call_args.kwargs["stdout"].name),
                    role.task / "service.log",
                )
                self.assertEqual(
                    argv[argv.index("--moe-strategy") + 1], "mega_moe_k3_se"
                )
                self.assertIn("--fp8-gemm", argv)
                self.assertIn("--fp8-kv-cache", argv)
                self.assertIn("--rdma-hcas", argv)
                self.assertIn("--allow-shared-accuracy", argv)
                self.assertEqual(
                    argv[argv.index("--cpu-tp-socket-dir") + 1], str(role.socket_dir)
                )
                self.assertEqual(
                    argv[argv.index("--runtime-tmp-dir") + 1], str(role.runtime_tmp_dir)
                )
                self.assertEqual(role.baseline_gpu_contexts, {("GPU-0", "123")})
                self.assertEqual(launch.call_args.kwargs["env"]["GPU_COUNT"], "8")
                self.assertEqual(launch.call_args.kwargs["env"]["TZ"], "Asia/Shanghai")
                self.assertEqual(
                    launch.call_args.kwargs["env"]["LD_LIBRARY_PATH"],
                    "/remote/native/lib",
                )
                self.assertEqual(
                    launch.call_args.kwargs["env"]["LD_PRELOAD"],
                    "/remote/native/preload.so",
                )
            finally:
                role.console.close()
                role._cleanup_socket_directory()
                role._cleanup_runtime_tmp_directory()

    def test_requests_use_prefill_cache_span_independent_of_decode_topology(self):
        for decode_dp, log_mode in ((1, "0"), (1, "1"), (2, "0"), (2, "1")):
            with self.subTest(
                decode_dp=decode_dp, log_mode=log_mode
            ), tempfile.TemporaryDirectory() as tmp, patch.dict(
                worker.os.environ, {"FT_SERVER_TEST": log_mode}
            ):
                item = config()
                item["decode_dp"] = decode_dp
                role = worker.Worker(threading.Event())
                role.config, role.role, role.host = item, "prefill", item["prefill"]
                role.task = Path(tmp)
                role.workspace = Path("/native.runfiles/rtp_llm")
                process = SimpleNamespace(poll=lambda: 0, returncode=0)
                with patch.object(role, "status"), patch.object(
                    worker.subprocess, "Popen", return_value=process
                ) as spawn:
                    self.assertEqual(
                        role.requests({"timeout_s": 60}), {"requests_passed": True}
                    )
                command = spawn.call_args.args[0]
                recorded = json.loads((role.task / "requests-command.json").read_text())
                self.assertEqual(recorded["argv"], command)
                self.assertEqual(
                    command[command.index("--prefill-engine-log") + 1],
                    str(
                        role.task
                        / ("service.log" if log_mode == "1" else "logs/engine.log")
                    ),
                )
                self.assertEqual(command[command.index("--block-size") + 1], "4096")
                self.assertEqual(
                    command[command.index("--reuse-unit-tokens") + 1], "32768"
                )
                self.assertEqual(command.count("--decode-role-addr"), decode_dp)

                # Check the request span against the real server profile, so
                # changing its page size or CP topology cannot silently drift.
                checkpoint = role.task / "checkpoint"
                checkpoint.mkdir()
                (checkpoint / "config.json").write_text('{"num_hidden_layers":93}')
                args = SimpleNamespace(
                    role="PREFILL",
                    checkpoint=str(checkpoint),
                    draft_checkpoint=str(checkpoint),
                    server=worker.sys.executable,
                    peer_ip="127.0.0.1",
                    start_port=27188,
                    peer_port=28188,
                    orthogonal_smoke=True,
                    tp_size=worker.PREFILL_TP_SIZE,
                )
                with patch.object(launcher.socket, "socket") as sockets:
                    sockets.return_value.getsockname.return_value = ("127.0.0.1", 0)
                    _, server = launcher.launch_config(args)
                options = dict(zip(server[1::2], server[2::2]))
                self.assertEqual(options["--prefill_cp_kv_cache_sharded"], "1")
                self.assertEqual(options["--dp_size"], "1")
                self.assertEqual(
                    int(options["--seq_size_per_block"]), worker.CACHE_PAGE_TOKENS
                )
                self.assertEqual(
                    int(command[command.index("--reuse-unit-tokens") + 1]),
                    int(options["--seq_size_per_block"]) * int(options["--tp_size"]),
                )

    def test_native_roles_allocate_short_unique_private_ipc_for_long_artifacts(self):
        root = "/data/" + "x" * 66
        roles = []
        try:
            for name, original_bytes in (("prefill", 136), ("decode", 135)):
                role = worker.Worker(threading.Event())
                roles.append(role)
                role.task = Path(root) / "k3-pd-0123456789ab" / name
                legacy_socket = role.task / "service/uds/rtp_llm_tp_k3_dp0_0.sock"
                self.assertEqual(len(worker.os.fsencode(legacy_socket)), original_bytes)
                with patch.dict(worker.os.environ, {"TMPDIR": str(role.task)}):
                    role._create_socket_directory()
                    role._create_runtime_tmp_directory()
                info = role.socket_dir.lstat()
                self.assertEqual(role.socket_dir.parent, Path("/tmp"))
                self.assertEqual(info.st_mode & 0o777, 0o700)
                self.assertEqual(info.st_uid, worker.os.geteuid())
                self.assertEqual(
                    role.socket_identity, (info.st_dev, info.st_ino, info.st_uid)
                )
                self.assertLess(
                    len(worker.os.fsencode(role.socket_dir / legacy_socket.name)), 108
                )
                runtime_info = role.runtime_tmp_dir.lstat()
                self.assertEqual(role.runtime_tmp_dir.parent, Path("/tmp"))
                self.assertEqual(runtime_info.st_mode & 0o777, 0o700)
                self.assertEqual(
                    role.runtime_tmp_identity,
                    (runtime_info.st_dev, runtime_info.st_ino, worker.os.geteuid()),
                )
                self.assertNotEqual(role.runtime_tmp_dir, role.socket_dir)
                self.assertLess(
                    len(
                        worker.os.fsencode(role.runtime_tmp_dir / "symm_mem-2147483647")
                    ),
                    108,
                )
            self.assertNotEqual(roles[0].socket_dir, roles[1].socket_dir)
            self.assertNotEqual(roles[0].runtime_tmp_dir, roles[1].runtime_tmp_dir)
        finally:
            for role in roles:
                role._cleanup_socket_directory()
                role._cleanup_runtime_tmp_directory()

    def test_early_start_failure_releases_private_ipc_and_locks(self):
        with tempfile.TemporaryDirectory() as tmp:
            role = worker.Worker(threading.Event())
            role.config, role.role = config(), "prefill"
            role.host = role.config[role.role]
            role.task, role.server = Path(tmp), Path("/unused/server")
            role._create_socket_directory()
            role._create_runtime_tmp_directory()
            directory = role.socket_dir
            runtime_tmp = role.runtime_tmp_dir
            events = []
            role.locks.callback(lambda: events.append("unlock"))
            with patch.object(
                worker.subprocess, "check_output", return_value=""
            ), patch.object(
                worker.subprocess, "Popen", side_effect=OSError("launcher unavailable")
            ), patch.object(
                worker.os, "killpg"
            ) as kill:
                with self.assertRaisesRegex(OSError, "launcher unavailable"):
                    role.start()
                role.close()
                kill.assert_not_called()
            self.assertFalse(directory.exists())
            self.assertFalse(runtime_tmp.exists())
            self.assertTrue(role.console.closed)
            self.assertEqual(events, ["unlock"])

    def test_runtime_tmp_cleanup_removes_nested_files_and_sockets_without_following_links(
        self,
    ):
        role = worker.Worker(threading.Event())
        role.task = Path("/unused")
        role._create_runtime_tmp_directory()
        with tempfile.TemporaryDirectory(dir="/tmp") as outside:
            marker = Path(outside) / "preserve"
            marker.write_text("foreign")
            nested = role.runtime_tmp_dir / "nested" / "subdirectory"
            nested.mkdir(parents=True)
            (nested / "data").write_text("temporary")
            (nested / "external-directory").symlink_to(outside)
            (nested / "external-file").symlink_to(marker)
            with worker.socket.socket(
                worker.socket.AF_UNIX, worker.socket.SOCK_DGRAM
            ) as endpoint:
                endpoint.bind(str(role.runtime_tmp_dir / "symm_mem-123"))
            role._cleanup_runtime_tmp_directory()
            self.assertFalse(role.runtime_tmp_dir.exists())
            self.assertEqual(marker.read_text(), "foreign")

    def test_runtime_tmp_replacement_is_preserved_and_other_resources_are_released(
        self,
    ):
        for replacement in ("directory", "symlink"):
            with self.subTest(replacement=replacement), tempfile.TemporaryDirectory(
                dir="/tmp"
            ) as tmp:
                role = worker.Worker(threading.Event())
                role.task = Path(tmp)
                role._create_socket_directory()
                role._create_runtime_tmp_directory()
                directory = role.runtime_tmp_dir
                moved = directory.with_name(directory.name + "-original")
                directory.rename(moved)
                if replacement == "directory":
                    directory.mkdir(mode=0o700)
                    marker = directory / "preserve"
                else:
                    directory.symlink_to(role.task)
                    marker = role.task / "preserve"
                marker.write_text("foreign")
                role.service = SimpleNamespace(poll=lambda: 1)
                events = []
                role.locks.callback(lambda: events.append("unlock"))
                try:
                    with patch.object(worker, "stop_process"), patch.object(
                        worker.subprocess, "check_output", return_value=""
                    ):
                        with self.assertRaises(RuntimeError) as error:
                            role.close()
                    self.assertIn(
                        "PD service exited unexpectedly: 1", str(error.exception)
                    )
                    self.assertIn("runtime tmp cleanup", str(error.exception))
                    self.assertEqual(marker.read_text(), "foreign")
                    self.assertTrue(moved.is_dir())
                    self.assertFalse(role.socket_dir.exists())
                    self.assertEqual(events, ["unlock"])
                finally:
                    if replacement == "directory":
                        marker.unlink()
                        directory.rmdir()
                    else:
                        directory.unlink()
                    moved.rename(directory)
                    role._cleanup_runtime_tmp_directory()

    def test_runtime_tmp_rejects_foreign_root_or_nested_owner_and_still_unlocks(self):
        for foreign_entry in ("root", "nested"):
            with self.subTest(foreign_entry=foreign_entry), tempfile.TemporaryDirectory(
                dir="/tmp"
            ) as tmp:
                role = worker.Worker(threading.Event())
                role.task = Path(tmp)
                role._create_socket_directory()
                role._create_runtime_tmp_directory()
                nested = role.runtime_tmp_dir / "nested"
                nested.mkdir()
                marker = nested / "preserve"
                marker.write_text("foreign")
                expected_name = (
                    role.runtime_tmp_dir.name if foreign_entry == "root" else "nested"
                )
                original_stat = worker.os.stat

                def foreign_stat(path, *args, **kwargs):
                    info = original_stat(path, *args, **kwargs)
                    if path == expected_name and kwargs.get("dir_fd") is not None:
                        return SimpleNamespace(
                            st_dev=info.st_dev,
                            st_ino=info.st_ino,
                            st_uid=info.st_uid + 1,
                            st_mode=info.st_mode,
                        )
                    return info

                events = []
                role.locks.callback(lambda: events.append("unlock"))
                try:
                    with patch.object(worker.os, "stat", side_effect=foreign_stat):
                        with self.assertRaisesRegex(
                            RuntimeError, "foreign runtime tmp entry"
                        ):
                            role.close()
                    self.assertEqual(marker.read_text(), "foreign")
                    self.assertFalse(role.socket_dir.exists())
                    self.assertEqual(events, ["unlock"])
                finally:
                    role._cleanup_runtime_tmp_directory()

    def test_ipc_cleanup_removes_only_owned_unix_sockets(self):
        role = worker.Worker(threading.Event())
        role.task = Path("/unused")
        role._create_socket_directory()
        directory = role.socket_dir
        with worker.socket.socket(worker.socket.AF_UNIX) as endpoint:
            endpoint.bind(str(directory / "rtp_llm_tp_k3_dp1_0.sock"))
        role._cleanup_socket_directory()
        self.assertFalse(directory.exists())

    def test_ipc_cleanup_rejects_replacement_and_preserves_other_cleanup_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            role = worker.Worker(threading.Event())
            role.task = Path(tmp)
            role._create_socket_directory()
            directory = role.socket_dir
            moved = directory.with_name(directory.name + "-original")
            directory.rename(moved)
            directory.mkdir(mode=0o700)
            foreign = directory / "do-not-delete"
            foreign.write_text("foreign")
            role.service = SimpleNamespace(poll=lambda: 1)
            role.console = SimpleNamespace(
                close=lambda: (_ for _ in ()).throw(OSError("log close failed"))
            )
            events = []
            role.locks.callback(lambda: events.append("unlock"))
            try:
                with patch.object(worker, "stop_process"), patch.object(
                    worker.subprocess, "check_output", return_value=""
                ):
                    with self.assertRaises(RuntimeError) as error:
                        role.close()
                self.assertIn("PD service exited unexpectedly: 1", str(error.exception))
                self.assertIn("log close failed", str(error.exception))
                self.assertIn(
                    "replaced or foreign CPU TP directory", str(error.exception)
                )
                self.assertEqual(foreign.read_text(), "foreign")
                self.assertTrue(moved.is_dir())
                self.assertEqual(events, ["unlock"])
                self.assertEqual(
                    json.loads((role.task / "cleanup.json").read_text())["error"],
                    str(error.exception),
                )
            finally:
                foreign.unlink()
                directory.rmdir()
                moved.rename(directory)
                role._cleanup_socket_directory()

    def test_ipc_cleanup_rejects_foreign_files_inside_owned_directory(self):
        role = worker.Worker(threading.Event())
        role.task = Path("/unused")
        role._create_socket_directory()
        foreign = role.socket_dir / "foreign-file"
        foreign.write_text("preserve")
        try:
            with self.assertRaisesRegex(ValueError, "Unexpected file"):
                role._cleanup_socket_directory()
            self.assertEqual(foreign.read_text(), "preserve")
        finally:
            foreign.unlink()
            role._cleanup_socket_directory()

    def test_cleanup_kills_only_the_owned_process_group(self):
        process = SimpleNamespace(pid=123456, wait=lambda timeout: 0)
        with patch.object(worker.os, "killpg") as kill:
            worker.stop_process(process)
        self.assertEqual({call.args[0] for call in kill.call_args_list}, {123456})

    def test_cleanup_attempts_both_processes_and_preserves_each_failure(self):
        for failures in (
            {"request"},
            {"request", "service"},
            {"request", "service", "verification"},
        ):
            with self.subTest(failures=failures), tempfile.TemporaryDirectory() as tmp:
                role = worker.Worker(threading.Event())
                role.task = Path(tmp)
                role.request = SimpleNamespace(name="request")
                role.service = SimpleNamespace(name="service", poll=lambda: None)
                role.console = io.StringIO()
                events = []
                role.locks.callback(lambda: events.append("unlock"))

                def stop(process):
                    events.append(process.name)
                    if process.name in failures:
                        raise TimeoutError(f"{process.name} did not exit")

                def verify(*args, **kwargs):
                    events.append("verification")
                    if "verification" in failures:
                        raise OSError("GPU query failed")
                    return ""

                with patch.object(
                    worker, "stop_process", side_effect=stop
                ), patch.object(
                    worker.subprocess, "check_output", side_effect=verify
                ), self.assertRaises(
                    RuntimeError
                ) as raised:
                    role.close()

                self.assertEqual(
                    events, ["request", "service", "verification", "unlock"]
                )
                self.assertTrue(role.console.closed)
                cleanup = json.loads((role.task / "cleanup.json").read_text())
                self.assertFalse(cleanup["stopped"])
                self.assertEqual(len(cleanup["errors"]), len(failures))
                self.assertEqual(cleanup["error"], str(raised.exception))
                for name in failures & {"request", "service"}:
                    self.assertIn(f"{name} cleanup: TimeoutError", cleanup["error"])
                if "verification" in failures:
                    self.assertIn("GPU cleanup verification: OSError", cleanup["error"])

    def test_successful_cleanup_records_success_after_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            role = worker.Worker(threading.Event())
            role.task = Path(tmp)
            role.request = SimpleNamespace(name="request")
            role.service = SimpleNamespace(name="service", poll=lambda: None)
            events = []
            role.locks.callback(lambda: events.append("unlock"))
            with patch.object(worker, "stop_process") as stop, patch.object(
                worker.subprocess, "check_output", return_value=""
            ) as verify:
                role.close()
            self.assertEqual(
                [call.args[0] for call in stop.call_args_list],
                [role.request, role.service],
            )
            verify.assert_called_once()
            self.assertEqual(events, ["unlock"])
            self.assertEqual(
                json.loads((role.task / "cleanup.json").read_text()),
                {
                    "stopped": True,
                    "error": None,
                    "errors": [],
                    "baseline_gpu_contexts": [],
                    "remaining_gpu_contexts": [],
                    "unexpected_gpu_contexts": [],
                    "gpu_contexts_verified": True,
                    "gpu_query_errors": [],
                    "allow_shared_accuracy": False,
                    "performance_validated": False,
                },
            )

    def test_worker_cleanup_preserves_baseline_and_reports_new_contexts(self):
        baseline = {("GPU-0", "123")}
        for contexts in (baseline, set(), baseline | {("GPU-1", "456")}):
            with self.subTest(contexts=contexts), tempfile.TemporaryDirectory() as tmp:
                role = worker.Worker(threading.Event())
                role.task = Path(tmp)
                role.host = {"allow_shared_accuracy": True}
                role.baseline_gpu_contexts = baseline
                role.service = SimpleNamespace(pid=123456, poll=lambda: None)
                remaining = "\n".join(f"{uuid}, {pid}" for uuid, pid in contexts)
                with patch.object(worker, "stop_process") as stop, patch.object(
                    worker.subprocess, "check_output", return_value=remaining
                ), patch.object(worker.time, "monotonic", side_effect=[0, 0, 60, 60]):
                    if contexts - baseline:
                        with self.assertRaisesRegex(RuntimeError, "New GPU contexts"):
                            role.close()
                    else:
                        role.close()
                    self.assertEqual(
                        [call.args[0] for call in stop.call_args_list],
                        [None, role.service],
                    )
                cleanup = json.loads((role.task / "cleanup.json").read_text())
                self.assertEqual(cleanup["stopped"], not bool(contexts - baseline))
                self.assertEqual(
                    cleanup["unexpected_gpu_contexts"],
                    [list(item) for item in sorted(contexts - baseline)],
                )
                self.assertFalse(cleanup["performance_validated"])

    def _cleanup_with_fake_gpu_clock(self, tmp, response):
        role = worker.Worker(threading.Event())
        role.task = Path(tmp)
        role.service = SimpleNamespace(poll=lambda: None)
        clock = SimpleNamespace(now=0)
        timeouts = []

        def query(*args, timeout, **kwargs):
            self.assertGreater(timeout, 0)
            self.assertLessEqual(timeout, 15)
            self.assertLessEqual(clock.now + timeout, 60)
            timeouts.append(timeout)
            return response(clock, timeout, len(timeouts))

        def sleep(delay):
            clock.now += delay

        error = None
        with patch.object(worker, "stop_process") as stop, patch.object(
            worker.subprocess, "check_output", side_effect=query
        ), patch.object(
            worker.time, "monotonic", side_effect=lambda: clock.now
        ), patch.object(
            worker.time, "sleep", side_effect=sleep
        ):
            try:
                role.close()
            except RuntimeError as exc:
                error = str(exc)
        self.assertEqual(
            [call.args[0] for call in stop.call_args_list], [None, role.service]
        )
        cleanup = json.loads((role.task / "cleanup.json").read_text())
        return cleanup, error, clock.now, timeouts

    def test_cleanup_accepts_empty_query_after_transient_nvml_timeout(self):
        def response(clock, timeout, attempt):
            if attempt == 1:
                clock.now += timeout
                raise subprocess.TimeoutExpired("nvidia-smi", timeout)
            return ""

        with tempfile.TemporaryDirectory() as tmp:
            cleanup, error, elapsed, timeouts = self._cleanup_with_fake_gpu_clock(
                tmp, response
            )
        self.assertIsNone(error)
        self.assertTrue(cleanup["stopped"])
        self.assertTrue(cleanup["gpu_contexts_verified"])
        self.assertEqual(cleanup["unexpected_gpu_contexts"], [])
        self.assertEqual(len(cleanup["gpu_query_errors"]), 1)
        self.assertIn("TimeoutExpired", cleanup["gpu_query_errors"][0])
        self.assertEqual(elapsed, 16)
        self.assertEqual(timeouts, [15, 15])

    def test_cleanup_timeout_never_proves_release_and_preserves_last_contexts(self):
        for prior_snapshot in (False, True):
            with self.subTest(prior_snapshot=prior_snapshot):

                def response(clock, timeout, attempt):
                    if prior_snapshot and attempt == 1:
                        return "GPU-0, 456\n"
                    clock.now += timeout
                    raise subprocess.TimeoutExpired("nvidia-smi", timeout)

                with tempfile.TemporaryDirectory() as tmp:
                    cleanup, error, elapsed, timeouts = (
                        self._cleanup_with_fake_gpu_clock(tmp, response)
                    )
                self.assertIsNotNone(error)
                self.assertIn("TimeoutExpired", error)
                self.assertFalse(cleanup["stopped"])
                self.assertFalse(cleanup["gpu_contexts_verified"])
                self.assertTrue(cleanup["gpu_query_errors"])
                expected = [["GPU-0", "456"]] if prior_snapshot else []
                self.assertEqual(cleanup["remaining_gpu_contexts"], expected)
                self.assertEqual(cleanup["unexpected_gpu_contexts"], expected)
                self.assertEqual(elapsed, 60)
                self.assertLess(timeouts[-1], 15)

    def test_cleanup_fails_when_new_contexts_persist_until_deadline(self):
        def response(clock, timeout, attempt):
            clock.now += min(10, timeout)
            return "GPU-0, 456\n"

        with tempfile.TemporaryDirectory() as tmp:
            cleanup, error, elapsed, timeouts = self._cleanup_with_fake_gpu_clock(
                tmp, response
            )
        self.assertIn("New GPU contexts remain", error)
        self.assertFalse(cleanup["stopped"])
        self.assertFalse(cleanup["gpu_contexts_verified"])
        self.assertEqual(cleanup["remaining_gpu_contexts"], [["GPU-0", "456"]])
        self.assertEqual(cleanup["unexpected_gpu_contexts"], [["GPU-0", "456"]])
        self.assertEqual(cleanup["gpu_query_errors"], [])
        self.assertEqual(elapsed, 60)
        self.assertLess(timeouts[-1], 15)


class PdReadinessTest(unittest.TestCase):
    @staticmethod
    def write_listener_table(root, ports):
        (root / "tcp").write_text(
            "sl local_address rem_address st\n"
            + "".join(
                f"{index}: 00000000:{port:04X} 00000000:0000 0A\n"
                for index, port in enumerate(sorted(ports))
            )
        )

    @staticmethod
    def role(root, name="prefill"):
        role = worker.Worker(threading.Event())
        role.config, role.role = config(), name
        role.host, role.task = role.config[name], root
        role.service = SimpleNamespace(poll=lambda: None)
        return role

    def test_expected_ports_cover_every_rank_and_both_decode_owners(self):
        prefill = worker.pd_endpoints(27188, 1)
        decode = worker.pd_endpoints(28188, 2)
        self.assertEqual(len(prefill), 25)
        self.assertEqual(len(decode), 26)
        self.assertEqual(
            [item for item in decode if item["kind"] == "http"],
            [
                {"rank": 0, "kind": "http", "port": 28188},
                {"rank": 4, "kind": "http", "port": 28224},
            ],
        )
        self.assertEqual(
            [item for item in prefill if item["rank"] == 7],
            [
                {"rank": 7, "kind": "rpc", "port": 27252},
                {"rank": 7, "kind": "cache_tcp", "port": 27253},
                {"rank": 7, "kind": "cache_rdma", "port": 27255},
            ],
        )

    def test_proc_tables_require_listen_state_and_matching_address(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tcp").write_text(
                "sl local_address rem_address st\n"
                "0: 00000000:6A38 00000000:0000 0A\n"
                "1: 00000000:6A39 00000000:0000 01\n"
                "2: 0100007F:6A3A 00000000:0000 0A\n"
            )
            (root / "tcp6").write_text(
                "sl local_address remote_address st\n"
                "0: 00000000000000000000000000000000:6A3C "
                "00000000000000000000000000000000:0000 0A\n"
            )
            # Wildcard listeners count; an established connection or a listener
            # bound only to localhost cannot serve the advertised peer address.
            self.assertEqual(
                worker.listening_tcp_ports("192.0.2.1", root), {27192, 27196}
            )

    def test_listener_table_errors_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                worker.listening_tcp_ports("127.0.0.1", root)
            for invalid in (
                "",
                "invalid header\n",
                "sl local_address rem_address st\nbad\n",
            ):
                (root / "tcp").write_text(invalid)
                with self.assertRaises(ValueError):
                    worker.listening_tcp_ports("127.0.0.1", root)
            with patch.object(Path, "read_text", side_effect=PermissionError("denied")):
                with self.assertRaises(PermissionError):
                    worker.listening_tcp_ports("127.0.0.1", root)

    def test_real_namespace_observes_listener_open_and_close(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
            self.assertNotIn(port, worker.listening_tcp_ports("127.0.0.1"))
            listener.listen()
            self.assertIn(port, worker.listening_tcp_ports("127.0.0.1"))
        self.assertNotIn(port, worker.listening_tcp_ports("127.0.0.1"))

    def test_master_health_does_not_hide_late_rank_then_all_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            role = self.role(root)
            ports = {item["port"] for item in worker.pd_endpoints(27188, 1)}
            parse = worker.listening_tcp_ports
            self.write_listener_table(root, ports - {27255})
            with patch.object(
                worker, "listening_tcp_ports", side_effect=lambda ip: parse(ip, root)
            ), patch.object(worker.urllib.request, "build_opener") as opener:
                opener.return_value.open.side_effect = (
                    lambda *a, **k: contextlib.nullcontext(SimpleNamespace(status=200))
                )
                self.assertEqual(role.status(), {"healthy": False})
                state = json.loads((root / "readiness.json").read_text())
                self.assertEqual(
                    state["missing_listeners"],
                    [{"rank": 7, "kind": "cache_rdma", "port": 27255}],
                )
                self.write_listener_table(root, ports)
                self.assertEqual(role.status(), {"healthy": True})
                state = json.loads((root / "readiness.json").read_text())
                self.assertEqual(state["missing_listeners"], [])
                self.assertFalse(state["rdma_transfer_verified"])

    def test_any_missing_rank_endpoint_blocks_decode_readiness(self):
        for missing in (28252, 28253, 28255, 28224):
            with self.subTest(port=missing), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                role = self.role(root, "decode")
                ports = {item["port"] for item in worker.pd_endpoints(28188, 2)}
                with patch.object(
                    worker, "listening_tcp_ports", return_value=ports - {missing}
                ), patch.object(worker.urllib.request, "build_opener") as opener:
                    opener.return_value.open.side_effect = (
                        lambda *a, **k: contextlib.nullcontext(
                            SimpleNamespace(status=200)
                        )
                    )
                    self.assertEqual(role.status(), {"healthy": False})
                state = json.loads((root / "readiness.json").read_text())
                self.assertEqual(
                    [item["port"] for item in state["missing_listeners"]], [missing]
                )

    def test_second_decode_owner_must_answer_health(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            role = self.role(root, "decode")
            ports = {item["port"] for item in worker.pd_endpoints(28188, 2)}
            with patch.object(
                worker, "listening_tcp_ports", return_value=ports
            ), patch.object(worker.urllib.request, "build_opener") as opener:
                opener.return_value.open.side_effect = (
                    lambda url, **k: contextlib.nullcontext(
                        SimpleNamespace(status=503 if ":28224/" in url else 200)
                    )
                )
                self.assertEqual(role.status(), {"healthy": False})
                self.assertEqual(
                    [call.args[0] for call in opener.return_value.open.call_args_list],
                    ["http://127.0.0.1:28188/health", "http://127.0.0.1:28224/health"],
                )
            state = json.loads((root / "readiness.json").read_text())
            self.assertEqual(
                state["unhealthy_http"], [{"rank": 4, "kind": "http", "port": 28224}]
            )

    def test_exited_service_fails_before_listener_or_health_queries(self):
        with tempfile.TemporaryDirectory() as tmp:
            role = self.role(Path(tmp))
            role.service = SimpleNamespace(poll=lambda: 1)
            with patch.object(worker, "listening_tcp_ports") as listeners:
                with self.assertRaisesRegex(RuntimeError, "PD service exited"):
                    role.status()
                listeners.assert_not_called()


if __name__ == "__main__":
    unittest.main()
