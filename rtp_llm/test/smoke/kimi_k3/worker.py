"""Role-side Bazel runtime owned by one of the paired remote test actions."""

import contextlib
import fcntl
import hashlib
import ipaddress
import json
import os
import re
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from pd_config import (
    CACHE_PAGE_TOKENS,
    COMPACT_CACHE_PAGE_TOKENS,
    PREFILL_TP_SIZE,
    gpu_contexts,
    validate_config,
)
from result_check import failure_details
from server_launcher import cpu_tp_socket_environment, runtime_tmp_environment

WORKER_PORT_STRIDE = 9
PROGRESS_LOG_BYTES = 256 * 1024
REQUEST_PROGRESS_PREFIX = b"KIMI_PD_PROGRESS "
REQUEST_PROGRESS_COUNTS = {
    "elapsed_ms",
    "case_count",
    "tokenize_calls",
    "tokenize_ms",
    "fit_calls",
    "fit_ms",
    "token_cache_hits",
}
REQUEST_STAGE_NAMES = frozenset(
    """
    historical_four_squares rolling_refill prefix_A_seed prefix_B_partial
    prefix_A_return prefix_AB_mixed batch_all_miss batch_all_hit mixed_seed_hits
    batch_mixed_hit_miss batch_mixed_then_all_hit cuda_graph_bucket_8
    tier-device-probe-0 tier-witness-seed tier-memory-probe
    orthogonal_decode_graph_long_output_batch_16 orthogonal_kv_final
    orthogonal_cancel_recovery decode_dp_cross_owner_cached_64k page_before
    checkpoint_after chunked_cold cache_device cross_owner_reuse partial_prefix
    cache_pressure cache_host mixed_cache decode_refill cancel_recovery
    """.split()
)
REQUEST_STAGE_PATTERN = re.compile(
    r"batch_shape_[0-9]+_[0-9]+|cache_block_[0-9]+_(?:cold|repeat)"
    r"|decode_page_(?:short|cross)_[0-9]+_(?:cold|repeat)"
    r"|prefill_64k_(?:single|batch)_(?:cold|reuse)"
    r"|tier-(?:memory|device|partial)-seed-[0-9]+|orthogonal_cache_mixed_[0-9]+"
    r"|tier-pressure-[0-9]+|orthogonal_decode_[0-9]+_batch_[0-9]+"
    r"|orthogonal_page_[0-9]+_[+-][0-9]+|orthogonal_kv_seed_[0-9]+"
    r"|cancel-memory-seed-[0-9]+"
)
STARTUP_MARKERS = (
    (
        b"[PROCESS_START]Start backend server process",
        "backend_started",
        "backend_start",
    ),
    (b"JIT scope ", "jit_scope", "jit_cache"),
    (b"JIT_CACHE_RESTORED:", "jit_restored", "jit_cache"),
    (b"JIT_CACHE_FAIL_OPEN:", "jit_fail_open", "jit_cache"),
    (b"[PROCESS_START]Start local rank process", "rank_started", "backend_start"),
    (b"load weight by device:", "weight_load_started", "weight_loading"),
    (b"fastsafetensors materialization:", "weights_materialized", "weight_loading"),
    (b"create sp model type is kimi_k3_mtp,", "draft_load_started", "weight_loading"),
    (b"create sp model type is kimi_k31_mtp,", "draft_load_started", "weight_loading"),
    (b"load model took:", "model_load_finished", "model_init"),
    (b"cache layout created:", "cache_layout_planned", "cache_init"),
    (b"create cache manager done", "cache_manager_ready", "cache_init"),
    (b"CUDA graph capture is enabled", "cuda_graph_started", "cuda_graph"),
    (b"capture success for batch size:", "cuda_graph_batch_done", "cuda_graph"),
    (b"create normal executor done", "executor_ready", "executor_init"),
    (
        b"Backend server initialized successfully, sending ready status",
        "backend_ready",
        "waiting_for_listeners",
    ),
    (b"start server took ", "server_start_finished", "waiting_for_listeners"),
)
STARTUP_TIMING = re.compile(
    rb"load (weights|model) took: ([0-9]{1,6}(?:\.[0-9]{1,6})?)s"
)


def _new_log_lines(path, offset):
    """Read complete lines from the bounded tail, at most once per reader."""
    try:
        with path.open("rb") as stream:
            end = stream.seek(0, os.SEEK_END)
            if end < offset:
                offset = 0
            start = max(offset, end - PROGRESS_LOG_BYTES)
            stream.seek(start)
            data = stream.read(end - start)
    except OSError:
        return offset, ()
    last_newline = data.rfind(b"\n")
    if last_newline == -1:
        return offset, ()
    # A bounded tail can start in the middle of a line. Do not parse that line.
    first = data.find(b"\n") + 1 if start > offset else 0
    return start + last_newline + 1, data[first : last_newline + 1].split(b"\n")[:-1]


def _request_progress_event(line):
    if len(line) > 1024 or not line.startswith(REQUEST_PROGRESS_PREFIX):
        return None
    try:
        event = json.loads(line[len(REQUEST_PROGRESS_PREFIX) :])
    except (ValueError, RecursionError):
        return None
    if not isinstance(event, dict):
        return None
    name = event.get("event")
    keys = REQUEST_PROGRESS_COUNTS | {"event", "passed"}
    if name in ("request_stage_start", "request_stage"):
        keys = keys | {"stage"}
        stage = event.get("stage")
        if (
            not isinstance(stage, str)
            or len(stage) > 96
            or not (
                stage in REQUEST_STAGE_NAMES or REQUEST_STAGE_PATTERN.fullmatch(stage)
            )
        ):
            return None
    elif name != "request_summary":
        return None
    if keys != event.keys() or type(event["passed"]) is not bool:
        return None
    if any(
        type(event[key]) is not int or not 0 <= event[key] < 2**53
        for key in REQUEST_PROGRESS_COUNTS
    ):
        return None
    return event


def pd_endpoints(start_port, dp_size):
    """Endpoints used by the fixed eight-rank PD profile in this launcher."""
    if dp_size not in (1, 2):
        raise ValueError("PD readiness requires DP1 or DP2")
    endpoints = []
    for rank in range(PREFILL_TP_SIZE):
        base = start_port + rank * WORKER_PORT_STRIDE
        for kind, offset in (("rpc", 1), ("cache_tcp", 2), ("cache_rdma", 4)):
            endpoints.append({"rank": rank, "kind": kind, "port": base + offset})
        if rank % (PREFILL_TP_SIZE // dp_size) == 0:
            endpoints.append({"rank": rank, "kind": "http", "port": base})
    return endpoints


def listening_tcp_ports(bind_ip, proc_net=Path("/proc/net")):
    """Read this network namespace without connecting to the RDMA protocol.

    Only LISTEN sockets bound to the advertised address or a wildcard count.
    /proc encodes each address word in host byte order, including TCP6 words.
    """
    expected = ipaddress.ip_address(bind_ip)
    ports = set()
    for name in ("tcp", "tcp6"):
        try:
            lines = (proc_net / name).read_text().splitlines()
        except FileNotFoundError:
            if name == "tcp6":  # IPv6 may be disabled in the worker kernel.
                continue
            raise
        if not lines or "local_address" not in lines[0]:
            raise ValueError(f"Invalid listener table: {proc_net / name}")
        for line in lines[1:]:
            columns = line.split()
            if len(columns) < 4:
                raise ValueError(f"Invalid listener row in {proc_net / name}")
            if columns[3] != "0A":
                continue
            address_hex, port_hex = columns[1].split(":")
            if len(address_hex) != (8 if name == "tcp" else 32):
                raise ValueError(f"Invalid listener address in {proc_net / name}")
            packed = b"".join(
                int(address_hex[index : index + 8], 16).to_bytes(4, sys.byteorder)
                for index in range(0, len(address_hex), 8)
            )
            address = ipaddress.ip_address(packed)
            if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
                address = address.ipv4_mapped
            if address.is_unspecified or address == expected:
                ports.add(int(port_hex, 16))
    return ports


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checkpoint_identity(path):
    """Metadata identity; actual tensor loading remains the loader guard's gate."""
    path = Path(path)
    index = path / "model.safetensors.index.json"
    return {
        "config_sha256": sha256(path / "config.json"),
        "index_sha256": sha256(index) if index.exists() else None,
        "shards": sorted(
            (file.name, file.stat().st_size) for file in path.glob("*.safetensors")
        ),
    }


def stop_process(process, timeout=60):
    if process is None:
        return
    # Let the supervisor drain frontends before terminating ranked backends.
    # Group-wide TERM bypasses that ordering and breaks live TP connections.
    with contextlib.suppress(ProcessLookupError):
        process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            subprocess.run(
                [
                    "ps",
                    "-p",
                    str(process.pid),
                    "-o",
                    "pid,ppid,pgid,sid,stat,wchan:32,comm",
                ],
                timeout=3,
                check=False,
            )
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)
    # The launcher may exit before its ranked children; close that exact group.
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)


class Worker:
    def __init__(self, disconnected):
        self.disconnected = disconnected
        self.locks = contextlib.ExitStack()
        self.service = None
        self.request = None
        self.task = None
        self.console = None
        self.config = None
        self.baseline_gpu_contexts = set()
        self.socket_dir = None
        self.socket_identity = None
        self.runtime_tmp_dir = None
        self.runtime_tmp_identity = None
        self._startup_started_at = None
        self._startup_log_offset = 0
        self._startup_stage = "waiting_for_logs"
        self._startup_markers = {}
        self._readiness = {}
        self.request_progress_callback = None

    def prepare_native(self, config, role, run_id):
        """Prepare a Bazel remote role without a checkout, SSH or a container."""
        if self.config is not None or os.geteuid() == 0:
            raise ValueError("Worker must be new and run as a non-root user")
        if role not in ("prefill", "decode"):
            raise ValueError("Invalid worker role")
        config = validate_config(config, local_role=role)
        return self._prepare_runtime(config, role, run_id)

    def _prepare_runtime(self, config, role, run_id):
        host = config[role]
        if (
            not isinstance(run_id, str)
            or not run_id.startswith("k3-pd-")
            or not run_id[6:].isalnum()
        ):
            raise ValueError("Invalid run id")
        root = Path(host["artifact_root"])
        self._require_native_artifact_root(root)
        if root.stat().st_uid != os.geteuid():
            raise ValueError("Artifact root must belong to the runtime user")
        self.task = root / run_id / role
        self.task.mkdir(parents=True, exist_ok=False)
        self.role, self.config, self.host = role, config, host
        # The native binary's own runfiles contain the server's complete closure.
        self.workspace = Path(__file__).absolute().parents[4]
        self.server = self.workspace / "rtp_llm" / "rtp_llm_server"
        if not self.server.is_file():
            raise ValueError("rtp_llm_server is missing from Bazel runfiles")
        # Logger.cc resolves this file against cwd; give the service a private
        # writable cwd without losing the Bazel-declared logging configuration.
        alog = self.task / "rtp_llm" / "config" / "alog.conf"
        alog.parent.mkdir(parents=True)
        shutil.copyfile(self.workspace / "rtp_llm" / "config" / "alog.conf", alog)
        target = checkpoint_identity(host["checkpoint"])
        draft = checkpoint_identity(host["draft_checkpoint"])
        layers = json.loads((Path(host["checkpoint"]) / "config.json").read_text())
        layers = layers.get("text_config", layers)
        if layers.get("num_hidden_layers") != 93:
            raise ValueError("The native full-model smoke requires all 93 layers")
        if not target["shards"] or not draft["shards"]:
            raise ValueError("Target and MTP checkpoints must contain weight shards")
        self._create_socket_directory()
        self._create_runtime_tmp_directory()
        self._lock_gpus()
        manifest = {
            "hostname": socket.gethostname(),
            "gpu_uuids": self.gpu_uuids,
            "checkpoint_identity": target,
            "draft_identity": draft,
            "guard_sha256": sha256(self._guard_path()),
            "execution_mode": "bazel-remote",
            "allow_shared_accuracy": host.get("allow_shared_accuracy", False),
            "performance_validated": False,
            "artifact_dir": str(self.task),
            "cpu_tp_socket_dir": str(self.socket_dir),
            "cpu_tp_socket_identity": self.socket_identity,
            "runtime_tmp_dir": str(self.runtime_tmp_dir),
            "runtime_tmp_identity": self.runtime_tmp_identity,
            "time_ns": time.time_ns(),
        }
        (self.task / "identity.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return manifest

    @staticmethod
    def _require_native_artifact_root(root):
        declared = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
        if not declared or not Path(declared).is_absolute():
            raise ValueError("Native worker requires TEST_UNDECLARED_OUTPUTS_DIR")
        declared = Path(declared).resolve(strict=True)
        resolved = root.resolve(strict=True)
        if not resolved.is_relative_to(declared) or not resolved.is_dir():
            raise ValueError("Native artifact root must be inside Bazel test outputs")
        for path in (declared, resolved):
            if path.stat().st_uid != os.geteuid():
                raise ValueError(
                    "Bazel artifact directories must belong to the runtime user"
                )

    def _guard_path(self):
        return str(Path(__file__).with_name("checkpoint_guard.py"))

    def _create_socket_directory(self):
        if self.socket_dir is not None:
            raise ValueError("CPU TP socket directory is already allocated")
        # Do not inherit Bazel's long TMPDIR, or the artifact/run directory.
        self.socket_dir = Path(
            tempfile.mkdtemp(prefix=f"k3pd-{os.geteuid()}-", dir="/tmp")
        )
        info = self.socket_dir.lstat()
        self.socket_identity = (info.st_dev, info.st_ino, info.st_uid)
        cpu_tp_socket_environment(self.task / "service", self.socket_dir)

    def _cleanup_socket_directory(self):
        if self.socket_dir is None:
            return

        def verify(info):
            if (
                not stat.S_ISDIR(info.st_mode)
                or (info.st_dev, info.st_ino, info.st_uid) != self.socket_identity
                or info.st_uid != os.geteuid()
            ):
                raise ValueError(
                    "Refusing to clean a replaced or foreign CPU TP directory"
                )

        verify(self.socket_dir.lstat())
        descriptor = os.open(
            self.socket_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        )
        try:
            verify(os.fstat(descriptor))
            # Operate through the verified descriptor; never traverse a symlink
            # or recursively remove unrelated files under a shared /tmp path.
            names = os.listdir(descriptor)
            for name in names:
                info = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                if not re.fullmatch(
                    r"rtp_llm_tp_k3_(?:pp[0-9]+_)?dp[0-9]+_[0-9]+\.sock", name
                ) or not stat.S_ISSOCK(info.st_mode):
                    raise ValueError("Unexpected file in owned CPU TP socket directory")
            for name in names:
                os.unlink(name, dir_fd=descriptor)
            verify(self.socket_dir.lstat())
            self.socket_dir.rmdir()
        finally:
            os.close(descriptor)

    def _create_runtime_tmp_directory(self):
        if self.runtime_tmp_dir is not None:
            raise ValueError("Runtime tmp directory is already allocated")
        self.runtime_tmp_dir = Path(
            tempfile.mkdtemp(prefix=f"k3tmp-{os.geteuid()}-", dir="/tmp")
        )
        info = self.runtime_tmp_dir.lstat()
        self.runtime_tmp_identity = (info.st_dev, info.st_ino, info.st_uid)
        runtime_tmp_environment(self.task / "service", self.runtime_tmp_dir)

    def _cleanup_runtime_tmp_directory(self):
        if self.runtime_tmp_dir is None:
            return
        if self.runtime_tmp_dir.parent != Path("/tmp"):
            raise ValueError("Refusing to clean an unexpected runtime tmp root")

        def identity(info):
            return info.st_dev, info.st_ino, info.st_uid

        def verify(info, expected):
            if (
                identity(info) != expected
                or info.st_uid != os.geteuid()
                or info.st_dev != self.runtime_tmp_identity[0]
            ):
                raise ValueError(
                    "Refusing to clean a replaced or foreign runtime tmp entry"
                )

        def remove_entry(parent, name, expected=None):
            info = os.stat(name, dir_fd=parent, follow_symlinks=False)
            expected = identity(info) if expected is None else expected
            verify(info, expected)
            if stat.S_ISDIR(info.st_mode):
                descriptor = os.open(
                    name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent
                )
                try:
                    verify(os.fstat(descriptor), expected)
                    for child in os.listdir(descriptor):
                        remove_entry(descriptor, child)
                    verify(
                        os.stat(name, dir_fd=parent, follow_symlinks=False), expected
                    )
                    os.rmdir(name, dir_fd=parent)
                finally:
                    os.close(descriptor)
            else:
                # Includes symlinks: unlink only this entry, never its target.
                verify(os.stat(name, dir_fd=parent, follow_symlinks=False), expected)
                os.unlink(name, dir_fd=parent)

        descriptor = os.open("/tmp", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            root = os.stat(
                self.runtime_tmp_dir.name, dir_fd=descriptor, follow_symlinks=False
            )
            verify(root, self.runtime_tmp_identity)
            if not stat.S_ISDIR(root.st_mode):
                raise ValueError("Refusing to clean a replaced runtime tmp directory")
            remove_entry(
                descriptor, self.runtime_tmp_dir.name, self.runtime_tmp_identity
            )
        finally:
            os.close(descriptor)

    def _lock_gpus(self, lock_root=Path("/tmp/rtp_llm/smoke/test/gpu_status")):
        rows = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=30,
        ).splitlines()
        rows = [tuple(value.strip() for value in row.split(",")) for row in rows]
        if len(rows) != 8 or {row[0] for row in rows} != {str(i) for i in range(8)}:
            raise ValueError("Each worker must expose exactly eight B300 GPUs")
        if any(not ("L20D" in row[2] or "B300" in row[2]) for row in rows):
            raise ValueError("This native K3 profile requires the B300 machine class")
        self.gpu_uuids = [row[1] for row in rows]
        lock_root.mkdir(parents=True, exist_ok=True)
        # Same flock files as //rtp_llm/test/utils:gpu_lock; fail rather than
        # partially reserving one node while waiting indefinitely for the other.
        for index in range(8):
            handle = self.locks.enter_context((lock_root / str(index)).open("a"))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self._capture_gpu_baseline("prepare")

    def _capture_gpu_baseline(self, stage):
        occupied = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=30,
        )
        contexts = gpu_contexts(occupied)
        allow_shared = getattr(self, "host", {}).get("allow_shared_accuracy", False)
        if self.task is not None:
            (self.task / f"gpu-contexts-{stage}.json").write_text(
                json.dumps(
                    {
                        "contexts": sorted(contexts),
                        "compute_processes": occupied,
                        "allow_shared_accuracy": allow_shared,
                        "performance_validated": False,
                    },
                    indent=2,
                )
                + "\n"
            )
        if contexts and not allow_shared:
            raise ValueError(
                "GPU compute contexts are already present; no process was stopped"
            )
        self.baseline_gpu_contexts = contexts

    def start(self):
        if self.config is None or self.service is not None:
            raise ValueError("Prepare exactly once before starting a role")
        if self.socket_dir is None:
            raise ValueError(
                "Prepare the private CPU TP socket directory before starting"
            )
        cpu_tp_socket_environment(self.task / "service", self.socket_dir)
        if self.runtime_tmp_dir is None:
            raise ValueError(
                "Prepare the private runtime tmp directory before starting"
            )
        runtime_tmp_environment(self.task / "service", self.runtime_tmp_dir)
        self._capture_gpu_baseline("before-start")
        peer = self.config["decode" if self.role == "prefill" else "prefill"]
        dp = self.config.get("decode_dp", 1) if self.role == "decode" else 1
        command = [
            sys.executable,
            str(Path(__file__).with_name("server_launcher.py")),
            "--role",
            self.role.upper(),
            "--tp-size",
            str(PREFILL_TP_SIZE // dp),
            "--dp-size",
            str(dp),
            "--ep-size",
            "8",
            "--prefill-source-tp-size",
            str(PREFILL_TP_SIZE),
            "--checkpoint",
            self.host["checkpoint"],
            "--draft-checkpoint",
            self.host["draft_checkpoint"],
            "--peer-ip",
            peer["ip"],
            "--peer-port",
            str(peer["port"]),
            "--start-port",
            str(self.host["port"]),
            "--server",
            str(self.server),
            "--allow-checkpoint-root",
            self.host["allow_checkpoint_root"],
            "--run-dir",
            str(self.task / "service"),
            "--cpu-tp-socket-dir",
            str(self.socket_dir),
            "--runtime-tmp-dir",
            str(self.runtime_tmp_dir),
            "--orthogonal-smoke",
            "--model-type",
            self.config.get("model_type", "kimi_k3"),
            "--moe-strategy",
            self.host.get("moe_strategy")
            or self.config.get("moe_strategy")
            or ("mega_moe" if self.role == "prefill" else "mega_moe_se"),
        ]
        if self.config.get("compact_smoke", False):
            command.append("--compact-smoke")
        if "quantization" in self.config:
            command += ["--quantization", self.config["quantization"]]
        for key in ("fp8_gemm", "fp8_kv_cache", "warm_up", "model_warm_up"):
            if self.config.get(key, False):
                command.append("--" + key.replace("_", "-"))
        if "expected_draft_shards" in self.host:
            command += [
                "--expected-draft-shards",
                str(self.host["expected_draft_shards"]),
            ]
        if self.host.get("rdma_hcas"):
            command += ["--rdma-hcas", self.host["rdma_hcas"]]
        if self.host.get("allow_shared_accuracy", False):
            command.append("--allow-shared-accuracy")
        for option in (
            "reserve_runtime_mem_mb",
            "memory_cache_size_mb",
            "kv_cache_mem_mb",
            "min_free_gib",
        ):
            if option in self.host:
                command += ["--" + option.replace("_", "-"), str(self.host[option])]
        environment = os.environ.copy()
        # The controller is GPU_COUNT=0; the remote server must own all eight.
        environment.update(
            GPU_COUNT="8",
            WORLD_SIZE="8",
            CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7",
        )
        self.console = (self.task / "service.log").open("w")
        self._startup_started_at = time.monotonic()
        self.service = subprocess.Popen(
            command,
            cwd=self.task,
            env=environment,
            stdout=self.console,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        return {"started": True, "pid": self.service.pid}

    def status(self):
        if self.service is None or self.service.poll() is not None:
            service_log = self.task / "service.log" if self.task else "<not prepared>"
            raise RuntimeError(
                f"PD service exited: role={getattr(self, 'role', 'unassigned')}; "
                f"service_log={service_log}"
            )
        dp_size = self.config.get("decode_dp", 1) if self.role == "decode" else 1
        endpoints = pd_endpoints(self.host["port"], dp_size)
        listeners = listening_tcp_ports(self.host["ip"])
        missing = [item for item in endpoints if item["port"] not in listeners]
        unhealthy_http = []
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        for item in endpoints:
            if item["kind"] != "http" or item["port"] not in listeners:
                continue
            try:
                with opener.open(
                    f"http://127.0.0.1:{item['port']}/health", timeout=5
                ) as response:
                    if response.status != 200:
                        unhealthy_http.append(item)
            except OSError:
                unhealthy_http.append(item)
        healthy = not missing and not unhealthy_http
        # Keep the worker wire protocol unchanged; persist the exact missing
        # rank/port diagnosis for a bounded startup failure in either runner.
        self._readiness = {
            "role": self.role,
            "time_ns": time.time_ns(),
            "healthy": healthy,
            "required_endpoints": endpoints,
            "missing_listeners": missing,
            "unhealthy_http": unhealthy_http,
            "rdma_transfer_verified": False,
        }
        (self.task / "readiness.json").write_text(
            json.dumps(self._readiness, indent=2) + "\n"
        )
        return {"healthy": healthy}

    def startup_diagnostics(self) -> dict:
        """Return progress markers from the service log and last readiness probe."""
        if self.task is not None:
            host = getattr(self, "host", {})
            model_markers = tuple(
                (
                    b"Creating python model for " + os.fsencode(host[key]) + b" on ",
                    marker,
                    "model_init",
                )
                for key, marker in (
                    ("checkpoint", "target_weights_ready"),
                    ("draft_checkpoint", "draft_weights_ready"),
                )
                if key in host
            )
            self._startup_log_offset, lines = _new_log_lines(
                self.task / "service.log", self._startup_log_offset
            )
            markers = STARTUP_MARKERS + model_markers
            for line in lines:
                if timing := STARTUP_TIMING.search(line):
                    key = (
                        "max_weight_load_ms"
                        if timing[1] == b"weights"
                        else "max_model_load_ms"
                    )
                    self._startup_markers[key] = max(
                        self._startup_markers.get(key, 0),
                        round(float(timing[2]) * 1000),
                    )
                for message, marker, stage in markers:
                    if message in line:
                        self._startup_markers[marker] = (
                            self._startup_markers.get(marker, 0) + 1
                        )
                        self._startup_stage = stage
                        break
        return {
            "elapsed_ms": (
                0
                if self._startup_started_at is None
                else int((time.monotonic() - self._startup_started_at) * 1000)
            ),
            "stage": "ready" if self._readiness.get("healthy") else self._startup_stage,
            "markers": dict(self._startup_markers),
            "missing_listeners": [
                dict(item) for item in self._readiness.get("missing_listeners", ())
            ],
            "unhealthy_http": [
                dict(item) for item in self._readiness.get("unhealthy_http", ())
            ],
        }

    def _publish_request_progress(self, path, offset):
        if self.request_progress_callback is None:
            return offset
        offset, lines = _new_log_lines(path, offset)
        for line in lines:
            if (event := _request_progress_event(line)) is not None:
                # Telemetry never changes the verdict of the request subprocess.
                with contextlib.suppress(Exception):
                    self.request_progress_callback(event)
        return offset

    def requests(self, message):
        if self.role != "prefill":
            raise ValueError("The request runner must execute on the Prefill host")
        self.status()
        decode = self.config["decode"]
        dp = self.config.get("decode_dp", 1)
        page_tokens = (
            COMPACT_CACHE_PAGE_TOKENS
            if self.config.get("compact_smoke", False)
            else CACHE_PAGE_TOKENS
        )
        command = [
            sys.executable,
            str(Path(__file__).with_name("text_smoke.py")),
            "--base-url",
            f"http://{self.host['ip']}:{self.host['port']}",
            "--decode-health-url",
            f"http://{decode['ip']}:{decode['port']}/health",
            "--output",
            str(self.task / "requests" / "accuracy.json"),
            "--namespace",
            self.task.parent.name,
            "--model-type",
            self.config.get("model_type", "kimi_k3"),
            "--block-size",
            str(page_tokens),
            "--reuse-unit-tokens",
            str(page_tokens * PREFILL_TP_SIZE),
            "--rdma-prewarm-attempts",
            "3",
            "--checkpoint",
            self.host["checkpoint"],
            "--prefill-engine-log",
            # Match Logger.cc: Bazel tests send C++ logs to captured stdout;
            # non-console test environments retain the normal file appender.
            str(
                self.task
                / (
                    "service.log"
                    if os.environ.get("FT_SERVER_TEST") == "1"
                    else "logs/engine.log"
                )
            ),
            "--prefill-rpc-runfiles",
            str(self.workspace.parent),
            "--prefill-grpc-port",
            str(self.host["port"] + 1),
        ]
        if self.config.get("compact_smoke", False):
            command.append("--compact-smoke")
        for owner in range(dp):
            port = decode["port"] + owner * (8 // dp) * 9
            command += ["--decode-role-addr", f"{decode['ip']}:{port}:{port + 1}"]
        (self.task / "requests-command.json").write_text(
            json.dumps({"argv": command}, indent=2) + "\n"
        )
        progress_log = self.task / "requests.log"
        with progress_log.open("w") as log:
            self.request = subprocess.Popen(
                command,
                cwd=self.task,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            deadline = time.monotonic() + message["timeout_s"]
            progress_offset = 0
            try:
                while self.request.poll() is None:
                    if self.disconnected.wait(1):
                        raise RuntimeError("Controller disconnected during requests")
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            f"Full-model requests timed out after {message['timeout_s']}s; "
                            f"requests_log={self.task / 'requests.log'}; "
                            f"service_log={self.task / 'service.log'}"
                        )
                    progress_offset = self._publish_request_progress(
                        progress_log, progress_offset
                    )
                    self.status()
            finally:
                self._publish_request_progress(progress_log, progress_offset)
            if self.request.returncode:
                raise RuntimeError(
                    f"Full-model smoke exited {self.request.returncode}; "
                    f"requests_log={self.task / 'requests.log'}; "
                    f"service_log={self.task / 'service.log'}; "
                    f"{failure_details(self.task / 'requests' / 'accuracy.json')}"
                )
        return {"requests_passed": True}

    def close(self):
        errors = []
        remaining_contexts = set()
        unexpected_contexts = set()
        gpu_contexts_verified = False
        gpu_query_errors = []
        try:
            unexpected_exit = None
            if self.service is not None:
                try:
                    unexpected_exit = self.service.poll()
                except Exception as exc:
                    errors.append(f"service status: {type(exc).__name__}: {exc}")
            # A stuck request runner must not prevent stopping the GPU service.
            # Keep both failures rather than allowing the second to hide the first.
            for name, process in (("request", self.request), ("service", self.service)):
                try:
                    # 120s frontend + 180s deferred backend + 30s supervisor reap.
                    stop_process(process, timeout=330 if name == "service" else 60)
                except Exception as exc:
                    errors.append(f"{name} cleanup: {type(exc).__name__}: {exc}")
            if unexpected_exit is not None:
                errors.append(f"PD service exited unexpectedly: {unexpected_exit}")
            # A supervisor may exit zero after reaping a crashed rank. Keep the
            # role's fresh core dumps in the verdict as well as the artifacts.
            if self.task is not None:
                cores = sorted(
                    path.name for path in self.task.glob("core-*") if path.is_file()
                )
                if cores:
                    errors.append(f"Service generated core dumps: {cores}")
                fatal = re.compile(
                    r"Fatal Python error:|Signal: (?:Aborted|Segmentation fault) \("
                    r"|DeepGEMM grid sync timeout:|CUDA error: unspecified launch failure"
                )
                for pattern in ("service.log", "service/logs/*.log", "logs/engine.log"):
                    for path in self.task.glob(pattern):
                        with path.open(errors="replace") as stream:
                            if any(fatal.search(line) for line in stream):
                                errors.append(f"Service logged a fatal error: {path}")
            if self.service is not None:
                deadline = time.monotonic() + 60
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        last_query_error = gpu_query_errors[-1:] or None
                        if unexpected_contexts:
                            raise RuntimeError(
                                "New GPU contexts remain after scoped service cleanup: "
                                f"{sorted(unexpected_contexts)}; "
                                f"last GPU query error: {last_query_error}; "
                                "no foreign process was stopped"
                            )
                        raise TimeoutError(
                            "GPU cleanup was not verified within 60 seconds; "
                            f"last known contexts: {sorted(remaining_contexts)}; "
                            f"last GPU query error: {last_query_error}"
                        )
                    try:
                        occupied = subprocess.check_output(
                            [
                                "nvidia-smi",
                                "--query-compute-apps=gpu_uuid,pid",
                                "--format=csv,noheader,nounits",
                            ],
                            text=True,
                            timeout=min(15, remaining),
                        )
                    except subprocess.TimeoutExpired as exc:
                        gpu_query_errors.append(f"{type(exc).__name__}: {exc}")
                    else:
                        remaining_contexts = gpu_contexts(occupied)
                        unexpected_contexts = (
                            remaining_contexts - self.baseline_gpu_contexts
                        )
                        if not unexpected_contexts:
                            gpu_contexts_verified = True
                            break
                    remaining = deadline - time.monotonic()
                    if remaining > 0:
                        time.sleep(min(1, remaining))
        except Exception as exc:
            errors.append(f"GPU cleanup verification: {type(exc).__name__}: {exc}")
        finally:
            # Every release is independent; an IPC or log error must not retain
            # GPU locks or hide the original service failure.
            for name, release in (
                (
                    "console",
                    self.console.close if self.console is not None else lambda: None,
                ),
                ("CPU TP IPC", self._cleanup_socket_directory),
                ("runtime tmp", self._cleanup_runtime_tmp_directory),
                ("GPU locks", self.locks.close),
            ):
                try:
                    release()
                except Exception as exc:
                    errors.append(f"{name} cleanup: {type(exc).__name__}: {exc}")
            if self.task is not None:
                (self.task / "cleanup.json").write_text(
                    json.dumps(
                        {
                            "stopped": not errors,
                            "error": "; ".join(errors) if errors else None,
                            "errors": errors,
                            "baseline_gpu_contexts": sorted(self.baseline_gpu_contexts),
                            "remaining_gpu_contexts": sorted(remaining_contexts),
                            "unexpected_gpu_contexts": sorted(unexpected_contexts),
                            "gpu_contexts_verified": gpu_contexts_verified,
                            "gpu_query_errors": gpu_query_errors,
                            "allow_shared_accuracy": getattr(self, "host", {}).get(
                                "allow_shared_accuracy", False
                            ),
                            "performance_validated": False,
                        }
                    )
                    + "\n"
                )
        if errors:
            raise RuntimeError("; ".join(errors))
