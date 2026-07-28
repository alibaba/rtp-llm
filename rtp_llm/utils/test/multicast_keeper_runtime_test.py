import os
import platform
import signal
import struct
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from rtp_llm.utils.multicast_keeper import (
    BIN_DIR_ENV,
    CREATOR_ENV,
    ENABLE_ENV,
    HOLDER_ENV,
    SHIM_ENV,
    KeeperArtifacts,
    MulticastKeeperConfigError,
    MulticastKeeperError,
    MulticastKeeperHealthError,
    MulticastKeeperMode,
    MulticastKeeperRuntime,
    discover_artifacts,
    is_enabled,
)

_FAKE_HOLDER = r"""
#!/usr/bin/env python3
import argparse
import os
import signal
import socket
import struct

MAGIC = 0x3250434D505452
VERSION = 3
PING = 1
REQUEST = struct.Struct("<QHHIQQQQIIQ")
RESPONSE = struct.Struct("<QHHIiIQQQQQIIQ")

parser = argparse.ArgumentParser()
parser.add_argument("--socket", required=True)
parser.add_argument("--ready-file", required=True)
parser.add_argument("--parent-pid", required=True)
parser.add_argument("--creator", required=True)
parser.add_argument("--client-timeout-ms", required=True)
parser.add_argument("--creator-timeout-ms", required=True)
parser.add_argument("--gpus", required=True)
parser.add_argument("--fabric-team-size")
args = parser.parse_args()

if os.environ.get("FAKE_HOLDER_EXIT") == "1":
    print("intentional fake holder startup failure", flush=True)
    raise SystemExit(23)

running = True
instance_hi = 0x1111222233334444
instance_lo = 0xAAAABBBBCCCCDDDD
listener = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)

def stop(_signum, _frame):
    global running
    running = False
    listener.close()

def change_identity(_signum, _frame):
    global instance_lo
    instance_lo += 1

if os.environ.get("FAKE_HOLDER_IGNORE_TERM") == "1":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
else:
    signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGUSR1, change_identity)

listener.bind(args.socket)
os.chmod(args.socket, 0o600)
listener.listen(8)
Path = __import__("pathlib").Path
Path(args.ready_file).write_text(
    "state=ready\n"
    + f"pid={os.getpid()}\n"
    + f"instance={instance_hi:016x}{instance_lo:016x}\n"
)
print(
    "holder fake ready "
    + f"parent={args.parent_pid} "
    + f"ld_preload={os.environ.get('LD_PRELOAD')} "
    + f"cuda_visible={os.environ.get('CUDA_VISIBLE_DEVICES')}",
    flush=True,
)

while running:
    try:
        connection, _ = listener.accept()
    except OSError:
        break
    with connection:
        request = connection.recv(REQUEST.size + 1)
        if len(request) != REQUEST.size:
            continue
        values = REQUEST.unpack(request)
        if values[0] != MAGIC or values[1] != VERSION or values[2] != PING:
            continue
        response = RESPONSE.pack(
            MAGIC,
            VERSION,
            PING,
            RESPONSE.size,
            0,
            len(args.gpus.split(",")),
            instance_hi,
            instance_lo,
            0,
            0,
            0,
            0,
            0,
            0,
        )
        connection.send(response)
"""

_FAKE_CREATOR = "#!/bin/sh\nexit 0\n"


def _append_preload_for_test(existing: str, shim: str) -> str:
    entries = [
        entry
        for entry in existing.replace(":", " ").split()
        if entry and entry != shim
    ]
    entries.append(shim)
    return ":".join(entries)


class MulticastKeeperRuntimeTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.artifacts = self._make_artifacts(self.root / "artifacts")

    def tearDown(self):
        self.tempdir.cleanup()

    def _make_artifacts(
        self, directory: Path, holder_source: str = _FAKE_HOLDER
    ) -> KeeperArtifacts:
        directory.mkdir(parents=True, exist_ok=True)
        holder = directory / "keeper_lite_holder"
        creator = directory / "keeper_lite_creator"
        shim = directory / "mc_shim_unified.so"
        rendered_holder = (
            textwrap.dedent(holder_source)
            .lstrip()
            .replace("#!/usr/bin/env python3", f"#!{sys.executable}", 1)
        )
        holder.write_text(rendered_holder)
        creator.write_text(_FAKE_CREATOR)
        shim.write_bytes(b"fake shim")
        holder.chmod(0o755)
        creator.chmod(0o755)
        return KeeperArtifacts(holder=holder, creator=creator, shim=shim)

    def _env(self, **updates):
        env = dict(os.environ)
        env.update(
            {
                ENABLE_ENV: "1",
                "CUDA_VISIBLE_DEVICES": "0,2",
                "LD_PRELOAD": "/opt/tms.so:/opt/other.so",
                "RTP_LLM_MC_KEEPER_START_TIMEOUT_MS": "1000",
                "RTP_LLM_MC_KEEPER_STOP_TIMEOUT_MS": "100",
            }
        )
        env.update(updates)
        return env

    def _runtime(self, **kwargs) -> MulticastKeeperRuntime:
        return MulticastKeeperRuntime(
            kwargs.pop("world_size", 2),
            kwargs.pop("local_world_size", 2),
            kwargs.pop("role", "decode"),
            env=kwargs.pop("env", self._env()),
            artifacts=kwargs.pop("artifacts", self.artifacts),
            state_root=self.root,
            **kwargs,
        )

    @staticmethod
    def _native_artifact(name: str) -> Path:
        relative = Path(
            "rtp_llm/cpp/cuda_checkpoint/multicast_keeper"
        ) / name
        candidates = []
        if os.environ.get("TEST_SRCDIR"):
            candidates.append(
                Path(os.environ["TEST_SRCDIR"])
                / os.environ.get("TEST_WORKSPACE", "__main__")
                / relative
            )
        candidates.append(
            Path(__file__).resolve().parents[3] / "bazel-bin" / relative
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        raise unittest.SkipTest(f"native keeper artifact is unavailable: {name}")

    def test_opt_in_gate(self):
        self.assertFalse(is_enabled({ENABLE_ENV: "true"}))
        self.assertTrue(is_enabled({ENABLE_ENV: "1"}))

        disabled = SimpleNamespace(
            parallelism_config=SimpleNamespace(world_size=2),
            role_config=SimpleNamespace(role_type="DECODE"),
        )
        self.assertIsNone(
            MulticastKeeperRuntime.from_config(disabled, env={ENABLE_ENV: "0"})
        )

    def test_config_rejects_invalid_world_topology(self):
        with self.assertRaisesRegex(MulticastKeeperConfigError, "greater than"):
            MulticastKeeperRuntime(
                1,
                2,
                "decode",
                env=self._env(),
                artifacts=self.artifacts,
            )
        uneven = MulticastKeeperRuntime(
            3,
            2,
            "decode",
            env=self._env(),
            artifacts=self.artifacts,
        )
        self.assertEqual(MulticastKeeperMode.CROSS_NODE_FABRIC, uneven.mode)
        self.assertEqual(3, uneven.fabric_team_size)

    def test_config_always_uses_dense_container_local_gpu_ordinals(self):
        for visible_value in (None, "", "7,9,11,13"):
            with self.subTest(visible_value=visible_value):
                env = self._env()
                if visible_value is None:
                    env.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    env["CUDA_VISIBLE_DEVICES"] = visible_value

                runtime = MulticastKeeperRuntime(
                    4,
                    4,
                    "decode",
                    env=env,
                    artifacts=self.artifacts,
                    state_root=self.root,
                )

                self.assertEqual((0, 1, 2, 3), runtime.gpus)
                self.assertEqual(
                    MulticastKeeperMode.SINGLE_NODE, runtime.mode
                )
                self.assertEqual(4, runtime.fabric_team_size)

    def test_from_config_prefers_configured_local_world_size(self):
        config = SimpleNamespace(
            parallelism_config=SimpleNamespace(world_size=4, local_world_size=2),
            role_config=SimpleNamespace(role_type="RoleType.DECODE"),
        )
        runtime = MulticastKeeperRuntime.from_config(
            config,
            env=self._env(LOCAL_WORLD_SIZE="1"),
            artifacts=self.artifacts,
            state_root=self.root,
        )
        self.assertIsNotNone(runtime)
        self.assertEqual(MulticastKeeperMode.CROSS_NODE_FABRIC, runtime.mode)
        self.assertEqual(2, runtime.local_world_size)
        self.assertEqual(4, runtime.fabric_team_size)

        fallback_config = SimpleNamespace(
            parallelism_config=SimpleNamespace(world_size=4),
            role_config=SimpleNamespace(role_type="DECODE"),
        )
        runtime = MulticastKeeperRuntime.from_config(
            fallback_config,
            env=self._env(LOCAL_WORLD_SIZE="2"),
            artifacts=self.artifacts,
            state_root=self.root,
        )
        self.assertEqual(2, runtime.local_world_size)

    def test_start_uses_direct_elf_private_state_and_ping_identity(self):
        runtime = self._runtime(world_size=4)
        try:
            self.assertIs(runtime, runtime.start())
            self.assertTrue(runtime.is_alive())
            health = runtime.health()
            self.assertEqual((0x1111222233334444, 0xAAAABBBBCCCCDDDD), health.instance)
            self.assertEqual(2, health.local_device_count)
            self.assertEqual(0o700, runtime.state_dir.stat().st_mode & 0o777)
            self.assertEqual(0o600, runtime.log_path.stat().st_mode & 0o777)

            command = list(runtime.process.args)
            self.assertEqual(str(self.artifacts.holder), command[0])
            self.assertEqual(
                str(os.getpid()), command[command.index("--parent-pid") + 1]
            )
            self.assertEqual("4", command[command.index("--fabric-team-size") + 1])
            log = runtime.log_path.read_text()
            self.assertIn("ld_preload=None", log)
            self.assertIn("cuda_visible=None", log)
        finally:
            state_dir = runtime.state_dir
            runtime.stop()
        self.assertFalse(state_dir.exists())
        self.assertIsNotNone(runtime.process.returncode)

    def test_subprocess_env_preserves_flags_and_appends_deduplicated_shim(self):
        runtime = self._runtime(world_size=8)
        try:
            runtime.start()
            shim = str(self.artifacts.shim)
            base = {
                "LD_PRELOAD": f"/opt/tms.so:{shim} /opt/a.so:{shim}",
                "NCCL_NVLS_ENABLE": "custom",
                "TORCH_SYMM_MEM_DISABLE_MULTICAST": "custom",
                "RTP_LLM_MC_REQUEST_TIMEOUT_MS": "77",
            }
            child = runtime.subprocess_env(base)
            self.assertEqual(f"/opt/tms.so:/opt/a.so:{shim}", child["LD_PRELOAD"])
            self.assertEqual("custom", child["NCCL_NVLS_ENABLE"])
            self.assertEqual("custom", child["TORCH_SYMM_MEM_DISABLE_MULTICAST"])
            self.assertEqual("77", child["RTP_LLM_MC_REQUEST_TIMEOUT_MS"])
            self.assertEqual("125000", child["RTP_LLM_MC_CREATE_TIMEOUT_MS"])
            self.assertEqual("0,1", child["RTP_LLM_MC_LOCAL_GPUS"])
            self.assertEqual("8", child["RTP_LLM_MC_FABRIC_TEAM_SIZE"])
            self.assertEqual(
                str(runtime.socket_path), child["RTP_LLM_CUDA_CKPT_MULTICAST_SOCKET"]
            )
            self.assertEqual(str(runtime.state_dir), child["NEKYIA_KEEPER_DIR"])
        finally:
            runtime.stop()

    def test_single_node_overrides_stale_fabric_team_and_context_restores_env(self):
        runtime = self._runtime()
        self.assertEqual(MulticastKeeperMode.SINGLE_NODE, runtime.mode)
        self.assertEqual(2, runtime.fabric_team_size)
        original = dict(os.environ)
        os.environ["RTP_LLM_MC_FABRIC_TEAM_SIZE"] = "99"
        expected = dict(os.environ)
        try:
            runtime.start()
            with runtime.configure_subprocess() as configured:
                self.assertEqual(
                    "2", configured["RTP_LLM_MC_FABRIC_TEAM_SIZE"]
                )
                self.assertEqual(
                    "2", os.environ["RTP_LLM_MC_FABRIC_TEAM_SIZE"]
                )
                self.assertEqual("1", os.environ[ENABLE_ENV])
                self.assertTrue(
                    os.environ["LD_PRELOAD"].endswith(str(self.artifacts.shim))
                )
                os.environ["KEEPER_TEST_CONCURRENT_ENV"] = "preserved"
            self.assertEqual("preserved", os.environ["KEEPER_TEST_CONCURRENT_ENV"])
            expected["KEEPER_TEST_CONCURRENT_ENV"] = "preserved"
            self.assertEqual(expected, dict(os.environ))
        finally:
            runtime.stop()
            os.environ.clear()
            os.environ.update(original)

    def test_native_artifacts_match_host_architecture_and_shim_preloads(self):
        machine = platform.machine().lower()
        expected_elf_machine = {
            "x86_64": 62,
            "aarch64": 183,
        }
        self.assertIn(machine, expected_elf_machine)

        artifacts = {
            name: self._native_artifact(name)
            for name in (
                "keeper_lite_holder",
                "keeper_lite_creator",
                "mc_shim_unified.so",
            )
        }
        for name, path in artifacts.items():
            with path.open("rb") as artifact_file:
                header = artifact_file.read(20)
            self.assertEqual(b"\x7fELF", header[:4], name)
            self.assertEqual(2, header[4], name)  # ELFCLASS64
            self.assertEqual(1, header[5], name)  # little-endian
            self.assertEqual(
                expected_elf_machine[machine],
                struct.unpack_from("<H", header, 18)[0],
                name,
            )

        env = dict(os.environ)
        shim = str(artifacts["mc_shim_unified.so"])
        env["LD_PRELOAD"] = _append_preload_for_test(
            env.get("LD_PRELOAD", ""), shim
        )
        env[ENABLE_ENV] = "1"
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "import ctypes, struct; "
                "assert struct.calcsize('P') == ctypes.sizeof(ctypes.c_void_p); "
                "print('preload-smoke-ok')",
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(
            0,
            completed.returncode,
            f"stdout={completed.stdout}\nstderr={completed.stderr}",
        )
        self.assertIn("preload-smoke-ok", completed.stdout)

    def test_native_holder_starts_in_single_and_cross_node_modes(self):
        artifacts = KeeperArtifacts(
            holder=self._native_artifact("keeper_lite_holder"),
            creator=self._native_artifact("keeper_lite_creator"),
            shim=self._native_artifact("mc_shim_unified.so"),
        )
        cases = (
            (2, MulticastKeeperMode.SINGLE_NODE, 2),
            (4, MulticastKeeperMode.CROSS_NODE_FABRIC, 4),
        )
        for world_size, expected_mode, expected_team_size in cases:
            with self.subTest(mode=expected_mode):
                runtime = self._runtime(
                    world_size=world_size,
                    artifacts=artifacts,
                )
                try:
                    runtime.start()
                    health = runtime.health()
                    self.assertEqual(expected_mode, runtime.mode)
                    self.assertEqual(expected_team_size, runtime.fabric_team_size)
                    self.assertEqual(2, health.local_device_count)
                    diagnostics = runtime.diagnostics()
                    self.assertEqual(expected_mode.value, diagnostics["mode"])
                    self.assertEqual(runtime.process.pid, diagnostics["pid"])
                    self.assertEqual(runtime.instance, diagnostics["instance"])
                    command = list(runtime.process.args)
                    if expected_team_size is None:
                        self.assertNotIn("--fabric-team-size", command)
                    else:
                        self.assertEqual(
                            str(expected_team_size),
                            command[command.index("--fabric-team-size") + 1],
                        )
                finally:
                    runtime.stop()

    def test_startup_failure_is_terminal_and_includes_log(self):
        runtime = self._runtime(env=self._env(FAKE_HOLDER_EXIT="1"))
        with self.assertRaisesRegex(
            MulticastKeeperError, "intentional fake holder startup failure"
        ):
            runtime.start()
        self.assertFalse(runtime.state_dir.exists())
        with self.assertRaisesRegex(
            MulticastKeeperError, "cannot be started more than once"
        ):
            runtime.start()
        runtime.stop()

    def test_health_rejects_live_process_with_changed_identity(self):
        runtime = self._runtime()
        try:
            runtime.start()
            os.kill(runtime.process.pid, signal.SIGUSR1)
            deadline = time.monotonic() + 1
            while runtime.is_alive() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertFalse(runtime.is_alive())
            self.assertIsNone(runtime.process.poll())
            with self.assertRaisesRegex(MulticastKeeperHealthError, "identity changed"):
                runtime.health()
        finally:
            runtime.stop()

    def test_stop_escalates_to_kill_and_is_idempotent(self):
        runtime = self._runtime(env=self._env(FAKE_HOLDER_IGNORE_TERM="1"))
        runtime.start()
        state_dir = runtime.state_dir
        runtime.stop()
        runtime.stop()
        self.assertEqual(-signal.SIGKILL, runtime.process.returncode)
        self.assertFalse(state_dir.exists())

    def test_artifact_discovery_priority_and_fail_closed_overrides(self):
        package_root = self.root / "repo" / "rtp_llm"
        repo_root = package_root.parent
        installed = self._make_artifacts(
            package_root / "cpp/cuda_checkpoint/multicast_keeper"
        )
        bazel = self._make_artifacts(
            repo_root / "bazel-bin/rtp_llm/cpp/cuda_checkpoint/multicast_keeper"
        )
        found = discover_artifacts({}, package_root=package_root, repo_root=repo_root)
        self.assertEqual(installed.holder.resolve(), found.holder)

        installed.holder.unlink()
        found = discover_artifacts({}, package_root=package_root, repo_root=repo_root)
        self.assertEqual(bazel.holder.resolve(), found.holder)

        explicit = self._make_artifacts(self.root / "explicit")
        found = discover_artifacts({BIN_DIR_ENV: str(explicit.holder.parent)})
        self.assertEqual(explicit, found)
        found = discover_artifacts(
            {
                HOLDER_ENV: str(explicit.holder),
                CREATOR_ENV: str(explicit.creator),
                SHIM_ENV: str(explicit.shim),
            }
        )
        self.assertEqual(explicit, found)
        with self.assertRaisesRegex(MulticastKeeperConfigError, "invalid"):
            discover_artifacts({BIN_DIR_ENV: str(self.root / "missing")})


if __name__ == "__main__":
    unittest.main()
