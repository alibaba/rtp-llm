#!/usr/bin/env python3
"""Start, inspect, and stop a FlexLB-routed RTP-LLM PD cluster over SSH."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import shlex
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any

HOSTS = [
    {
        "name": "rym-dsv4",
        "ip": "11.86.13.78",
        "gpus": 8,
        "rdma_nic": "mlx5_0",
    },
    {
        "name": "rym-dpsk-v4-pro-1",
        "ip": "11.18.249.63",
        "gpus": 8,
        "rdma_nic": "mlx5_2",
    },
    {
        "name": "rym-dpsk-v4-pro-2",
        "ip": "11.18.49.152",
        "gpus": 8,
        "rdma_nic": "mlx5_0",
    },
    {
        "name": "rym-dpsk-v4-pro-3",
        "ip": "11.21.242.20",
        "gpus": 8,
        "rdma_nic": "mlx5_7",
    },
]

HOST_BY_NAME = {host["name"]: host for host in HOSTS}
DECODE_HOST_ORDER = [
    "rym-dsv4",
    "rym-dpsk-v4-pro-2",
    "rym-dpsk-v4-pro-3",
    "rym-dpsk-v4-pro-1",
]
PREFILL_HOST_ORDER = [
    "rym-dpsk-v4-pro-1",
    "rym-dpsk-v4-pro-2",
    "rym-dpsk-v4-pro-3",
    "rym-dsv4",
]

# Optional single/multi-host restriction for control experiments.
# RTP_HOST_FILTER="rym-dsv4" limits the plan to the named hosts;
# RTP_HOST_GPU_LIMIT=3 caps usable GPUs per remaining host. Defaults: no-op.
_HOST_FILTER = os.environ.get("RTP_HOST_FILTER", "").strip()
if _HOST_FILTER:
    _keep = {name.strip() for name in _HOST_FILTER.split(",") if name.strip()}
    _unknown = _keep - {host["name"] for host in HOSTS}
    if _unknown:
        raise SystemExit(f"RTP_HOST_FILTER has unknown hosts: {sorted(_unknown)}")
    HOSTS = [host for host in HOSTS if host["name"] in _keep]
    # NOTE: HOST_BY_NAME intentionally keeps ALL hosts so that `stop` can
    # still resolve instances from a pre-filter state.json.
    DECODE_HOST_ORDER = [n for n in DECODE_HOST_ORDER if n in _keep]
    PREFILL_HOST_ORDER = [n for n in PREFILL_HOST_ORDER if n in _keep]
_HOST_GPU_LIMIT = os.environ.get("RTP_HOST_GPU_LIMIT", "").strip()
if _HOST_GPU_LIMIT:
    for _host in HOSTS:
        _host["gpus"] = min(int(_host["gpus"]), int(_HOST_GPU_LIMIT))

ROOT = Path(__file__).resolve().parent
STATE_FILE = ROOT / "state.json"

SOURCE_PATH = os.environ.get("RTP_SOURCE_PATH", "/home/admin/project/rtp-llm")
RDMA_SOURCE_PATH = os.environ.get(
    "RTP_RDMA_SOURCE_PATH", "/home/admin/project/rtp-llm-rdma"
)
MODEL_PATH = os.environ.get(
    "RTP_MODEL_PATH", "/home/admin/models/DeepSeek-V2-Lite-Chat"
)
PYTHON = os.environ.get("RTP_PYTHON", "/opt/conda310/bin/python")
RDMA_RUNTIME_PATH = os.environ.get(
    "RTP_RDMA_RUNTIME_PATH",
    "/home/admin/rtp-hol/runtime/rtp-rdma-pdprofile-v2",
)
RDMA_PYTHONPATH = f"{RDMA_RUNTIME_PATH}/site-packages"
REMOTE_ROOT = os.environ.get(
    "RTP_CLUSTER_REMOTE_ROOT", "/home/admin/rtp-hol/cluster/instances"
)

PREFILL_TP = int(os.environ.get("RTP_PREFILL_TP", "2"))
DECODE_TP = int(os.environ.get("RTP_DECODE_TP", "1"))
# DP>1 enables whole-node instances (e.g. V3.2: prefill TP2xDP4, decode TP1xDP8).
# GPUs per instance = TP*DP; ep_size = TP*DP when DP>1 (MoE EP across the node).
PREFILL_DP = int(os.environ.get("RTP_PREFILL_DP", "1"))
DECODE_DP = int(os.environ.get("RTP_DECODE_DP", "1"))
PREFILL_GPUS_PER_INST = PREFILL_TP * PREFILL_DP
DECODE_GPUS_PER_INST = DECODE_TP * DECODE_DP
MODEL_TYPE = os.environ.get("RTP_MODEL_TYPE", "deepseek2")
MAX_SEQ_LEN = int(os.environ.get("RTP_MAX_SEQ_LEN", "65536"))
REUSE_CACHE = int(os.environ.get("RTP_REUSE_CACHE", "1"))
PREFILL_REUSE_CACHE = int(os.environ.get("RTP_PREFILL_REUSE_CACHE", str(REUSE_CACHE)))
DECODE_REUSE_CACHE = int(os.environ.get("RTP_DECODE_REUSE_CACHE", str(REUSE_CACHE)))
PREFILL_CONCURRENCY = int(os.environ.get("RTP_PREFILL_CONCURRENCY", "8"))
DECODE_CONCURRENCY = int(os.environ.get("RTP_DECODE_CONCURRENCY", "32"))
FRONTEND_CONCURRENCY = int(os.environ.get("RTP_FRONTEND_CONCURRENCY", "1024"))
PREFILL_KV_MB = int(os.environ.get("RTP_PREFILL_KV_MB", "60000"))
DECODE_KV_MB = int(os.environ.get("RTP_DECODE_KV_MB", "80000"))
RUNTIME_RESERVE_MB = int(os.environ.get("RTP_RUNTIME_RESERVE_MB", "8192"))
INDUCTOR_COMPILE_THREADS = int(os.environ.get("RTP_INDUCTOR_COMPILE_THREADS", "8"))
PREFILL_BASE_PORT = int(os.environ.get("RTP_PREFILL_BASE_PORT", "8090"))
DECODE_BASE_PORT = int(os.environ.get("RTP_DECODE_BASE_PORT", "27001"))
PORT_STRIDE = int(os.environ.get("RTP_PORT_STRIDE", "100"))
# Ctx-length bucketing: "shortCount:shortConc,longCount:longConc"; empty disables.
# Short-bucket instances come first in decode index order, long bucket last.
DECODE_BUCKET_SPEC = os.environ.get("RTP_DECODE_BUCKET_SPEC", "")
DECODE_BUCKET_SEQLEN = int(os.environ.get("RTP_DECODE_BUCKET_SEQLEN", "8192"))


def parse_decode_bucket_spec(value: str) -> list[tuple[int, int, int]]:
    """Return [(instance_count, concurrency, tp_size), ...]; short bucket first.

    Spec grammar: "count:conc[:tp],count:conc[:tp]"; tp defaults to 1.
    """
    if not value.strip():
        return []
    buckets = []
    for part in value.split(","):
        fields = part.strip().split(":")
        count, conc = int(fields[0]), int(fields[1])
        tp = int(fields[2]) if len(fields) > 2 else 1
        buckets.append((count, conc, tp))
    return buckets


DECODE_BUCKETS = parse_decode_bucket_spec(DECODE_BUCKET_SPEC)
# Output-length slot quota: threshold in tokens (0 disables) + per-instance quota.
DECODE_LONG_OUT_THRESHOLD = int(os.environ.get("RTP_DECODE_LONG_OUT_THRESHOLD", "0"))
DECODE_LONG_SLOT_QUOTA = int(os.environ.get("RTP_DECODE_LONG_SLOT_QUOTA", "8"))
STATE_SCHEMA_VERSION = 3


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be one of 0/1, true/false, yes/no, on/off")


CACHE_STORE_RDMA_MODE = env_flag("RTP_CACHE_STORE_RDMA_MODE", False)

TOTAL_GPUS = sum(int(host["gpus"]) for host in HOSTS)
SERVICE_ID = "aigc.text-generation.generation.engine_service"
FLEXLB_HOST_NAME = os.environ.get("RTP_FLEXLB_HOST", "rym-dpsk-v4-pro-3")
FLEXLB_BASE_PORT = int(os.environ.get("RTP_FLEXLB_PORT", "7001"))
FRONTEND_BASE_PORT = int(os.environ.get("RTP_FRONTEND_PORT", "16000"))
FLEXLB_PREFILL_QUEUE_THRESHOLD = int(
    os.environ.get("RTP_FLEXLB_PREFILL_QUEUE_THRESHOLD", "1000000")
)
DECODE_ROUTING_SCHEME = (
    os.environ.get("RTP_DECODE_ROUTING_SCHEME", "LOAD_ONLY")
    .strip()
    .upper()
    .replace("-", "_")
)
if DECODE_ROUTING_SCHEME not in {
    "LOAD_ONLY",
    "CACHE_ONLY",
    "CACHE_LOAD",
    "TIME_BALANCE",
}:
    raise ValueError(
        "RTP_DECODE_ROUTING_SCHEME must be LOAD_ONLY, CACHE_ONLY, CACHE_LOAD, or TIME_BALANCE"
    )
DECODE_CACHE_LOAD_BETA = float(os.environ.get("RTP_DECODE_CACHE_LOAD_BETA", "1.0"))
FLEXLB_JAVA = os.environ.get(
    "RTP_FLEXLB_JAVA", "/home/admin/rtp-hol/runtime/jdk-21/bin/java"
)
FLEXLB_JAR = os.environ.get(
    "RTP_FLEXLB_JAR", "/home/admin/rtp-hol/flexlb/flexlb-api-pdprofile-v1.jar"
)
SSH_KNOWN_HOSTS = os.environ.get(
    "RTP_SSH_KNOWN_HOSTS", str(Path.home() / ".ssh" / "known_hosts")
)

SSH_OPTIONS = [
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=10",
    "-o",
    "ServerAliveInterval=20",
    "-o",
    "ServerAliveCountMax=15",
    "-o",
    "StrictHostKeyChecking=yes",
    "-o",
    f"UserKnownHostsFile={SSH_KNOWN_HOSTS}",
]


class ClusterError(RuntimeError):
    pass


def shell_join(values: list[str]) -> str:
    return " ".join(shlex.quote(value) for value in values)


REMOTE_RECORD_LAUNCH_IDENTITY = r"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

identity_path = Path(sys.argv[1])
pid = int(sys.argv[2])
token = sys.argv[3]
role = sys.argv[4]
kind = sys.argv[5]
start_port = int(sys.argv[6])
expected_executable = os.path.realpath(sys.argv[7])
required_environment = {
    b"RTP_CLUSTER_MANAGED=1",
    f"RTP_CLUSTER_TOKEN={token}".encode(),
}


def process_identity(target_pid):
    proc = Path("/proc") / str(target_pid)
    raw_stat = (proc / "stat").read_text(encoding="ascii")
    tail = raw_stat.rpartition(")")[2].split()
    if len(tail) < 20:
        raise RuntimeError(f"malformed /proc/{target_pid}/stat")
    raw_cmdline = (proc / "cmdline").read_bytes()
    environment = set((proc / "environ").read_bytes().split(b"\0"))
    executable = os.path.realpath(os.readlink(proc / "exe"))
    return {
        "state": tail[0],
        "process_group": int(tail[2]),
        "session": int(tail[3]),
        "start_time_ticks": int(tail[19]),
        "cmdline_sha256": hashlib.sha256(raw_cmdline).hexdigest(),
        "executable": executable,
        "managed_environment_matches": required_environment.issubset(environment),
    }


def visible_pid(target_pid):
    try:
        status = (Path("/proc") / str(target_pid) / "status").read_text(
            encoding="ascii"
        )
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
        return None
    for line in status.splitlines():
        if line.startswith("NSpid:"):
            values = line.split()[1:]
            return int(values[-1]) if values else None
    return None


def same_pid_namespace(target_pid):
    try:
        return (
            os.stat(Path("/proc") / str(target_pid) / "ns" / "pid").st_ino
            == os.stat("/proc/self/ns/pid").st_ino
        )
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
        return False


def find_outer_pid(namespace_pid):
    matches = []
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        target_pid = int(entry.name)
        if same_pid_namespace(target_pid) and visible_pid(target_pid) == namespace_pid:
            matches.append(target_pid)
    if not matches:
        return None
    if len(matches) != 1:
        raise RuntimeError(
            f"launcher namespace pid {namespace_pid} maps to {matches}"
        )
    return matches[0]


if pid <= 1:
    raise SystemExit(f"refusing unsafe launcher pid {pid}")

deadline = time.monotonic() + 10.0
observed = None
outer_pid = None
while time.monotonic() < deadline:
    try:
        candidate_outer_pid = find_outer_pid(pid)
        candidate = (
            process_identity(candidate_outer_pid)
            if candidate_outer_pid is not None
            else None
        )
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
        candidate = None
    if candidate is not None and candidate["state"] != "Z":
        is_expected_process = (
            candidate["process_group"] == candidate_outer_pid
            and candidate["session"] == candidate_outer_pid
            and candidate["executable"] == expected_executable
        )
        if is_expected_process and not candidate["managed_environment_matches"]:
            raise SystemExit(
                f"launcher {pid} has the expected executable but not the exact "
                "managed token environment"
            )
        if is_expected_process:
            observed = candidate
            outer_pid = candidate_outer_pid
            break
    time.sleep(0.02)

if observed is None:
    raise SystemExit(
        f"launcher {pid} did not become the expected setsid process: "
        f"executable={expected_executable}"
    )

identity = {
    "schema_version": 1,
    "pid": pid,
    "outer_pid": outer_pid,
    "uid": os.geteuid(),
    "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text(
        encoding="ascii"
    ).strip(),
    "process_group": observed["process_group"],
    "session": observed["session"],
    "start_time_ticks": observed["start_time_ticks"],
    "cmdline_sha256": observed["cmdline_sha256"],
    "executable": observed["executable"],
    "managed_token": token,
    "role": role,
    "kind": kind,
    "start_port": start_port,
}
identity_path.parent.mkdir(parents=True, exist_ok=True)
temporary = identity_path.with_name(f".{identity_path.name}.{os.getpid()}.tmp")
payload = (json.dumps(identity, sort_keys=True, separators=(",", ":")) + "\n").encode()
descriptor = os.open(
    temporary,
    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
    0o600,
)
try:
    with os.fdopen(descriptor, "wb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, identity_path)
finally:
    try:
        temporary.unlink()
    except FileNotFoundError:
        pass
"""


REMOTE_MANAGED_PROCESS_HELPER = r"""
import http.client
import errno
import hashlib
import json
import os
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path

config = json.loads(sys.argv[1])
required = {item.encode() for item in config.get("required_env", [])}
try:
    recorded_launcher_pid = int(Path(config.get("pid_file", "")).read_text().strip())
except (FileNotFoundError, PermissionError, ValueError, OSError):
    recorded_launcher_pid = None


class ScanError(RuntimeError):
    pass


def transient(error):
    return getattr(error, "errno", None) in (errno.ENOENT, errno.ESRCH)


def stat_identity(pid):
    proc = Path("/proc") / str(pid)
    try:
        stat_tail = (proc / "stat").read_text(encoding="ascii").rpartition(")")[2]
    except (FileNotFoundError, ProcessLookupError):
        return None
    except OSError as error:
        if transient(error):
            return None
        raise ScanError(f"cannot read stat for pid {pid}: {error}") from error
    fields = stat_tail.split()
    if len(fields) < 20:
        raise ScanError(f"malformed /proc/{pid}/stat")
    return fields[0], int(fields[19])


def stat_details(pid):
    proc = Path("/proc") / str(pid)
    try:
        tail = (proc / "stat").read_text(encoding="ascii").rpartition(")")[2]
    except (FileNotFoundError, ProcessLookupError):
        return None
    except OSError as error:
        if transient(error):
            return None
        raise ScanError(f"cannot read stat for pid {pid}: {error}") from error
    fields = tail.split()
    if len(fields) < 20:
        raise ScanError(f"malformed /proc/{pid}/stat")
    return {
        "state": fields[0],
        "parent": int(fields[1]),
        "process_group": int(fields[2]),
        "session": int(fields[3]),
        "start_time_ticks": int(fields[19]),
    }


def nspid_values(status_path):
    try:
        status = Path(status_path).read_text(encoding="ascii")
    except (FileNotFoundError, ProcessLookupError):
        return None
    except OSError as error:
        if transient(error):
            return None
        raise ScanError(f"cannot read {status_path}: {error}") from error
    for line in status.splitlines():
        if line.startswith("NSpid:"):
            return [int(value) for value in line.split()[1:]]
    raise ScanError(f"NSpid is missing from {status_path}")


def pid_namespace_relation(pid):
    override = config.get("test_pid_namespace_relation", {})
    if str(pid) in override:
        return override[str(pid)]
    target = Path("/proc") / str(pid) / "ns" / "pid"
    try:
        return os.stat(target).st_ino == os.stat("/proc/self/ns/pid").st_ino
    except (FileNotFoundError, ProcessLookupError):
        return False
    except PermissionError as error:
        target_values = nspid_values(Path("/proc") / str(pid) / "status")
        self_values = nspid_values("/proc/self/status")
        if target_values is None:
            return False
        if self_values is not None and len(target_values) != len(self_values):
            return False
        return None
    except OSError as error:
        if transient(error):
            return False
        raise ScanError(f"cannot verify PID namespace for pid {pid}: {error}") from error


def visible_pid_from_status(pid):
    values = nspid_values(Path("/proc") / str(pid) / "status")
    if not values:
        return None
    return values[-1]


def effective_uid(pid):
    try:
        status = (Path("/proc") / str(pid) / "status").read_text(encoding="ascii")
    except (FileNotFoundError, ProcessLookupError):
        return None
    except OSError as error:
        if transient(error):
            return None
        raise ScanError(f"cannot read uid for pid {pid}: {error}") from error
    for line in status.splitlines():
        if line.startswith("Uid:"):
            values = line.split()
            if len(values) >= 3:
                return int(values[2])
    raise ScanError(f"cannot parse uid for pid {pid}")


def looks_like_managed_process(pid):
    try:
        cmdline = (Path("/proc") / str(pid) / "cmdline").read_bytes().replace(
            b"\0", b" "
        ).lower()
    except (FileNotFoundError, ProcessLookupError):
        return False
    except OSError:
        return False
    return any(
        marker in cmdline
        for marker in (
            b"rtp_llm",
            b"rtp-llm",
            b"flexlb-api",
            b"rtp-hol-flexlb",
        )
    )


def unreadable_target(pid):
    return (
        recorded_launcher_pid is not None
        and visible_pid_from_status(pid) == recorded_launcher_pid
    ) or looks_like_managed_process(pid)


def matches(pid):
    if pid <= 1:
        return False
    proc = Path("/proc") / str(pid)
    try:
        if proc.stat().st_uid != os.geteuid():
            return False
        namespace_relation = pid_namespace_relation(pid)
        if namespace_relation is False:
            return False
        identity = stat_identity(pid)
        if identity is None or identity[0] == "Z":
            return False
        entries = set((proc / "environ").read_bytes().split(b"\0"))
    except (FileNotFoundError, ProcessLookupError):
        return False
    except OSError as error:
        if transient(error):
            return False
        if unreadable_target(pid):
            raise ScanError(
                f"cannot inspect managed candidate environment for pid {pid}: {error}"
            ) from error
        return False
    matched = required.issubset(entries)
    if matched and namespace_relation is not True:
        raise ScanError(f"managed token matched pid {pid} in an unverifiable namespace")
    return matched


def matching_pids():
    found = []
    try:
        entries = list(os.scandir("/proc"))
    except OSError as error:
        raise ScanError(f"cannot scan /proc: {error}") from error
    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if matches(pid):
            found.append(pid)
    return sorted(found)


def load_launch_identity():
    identity_name = config.get("identity_file")
    state_identity = config.get("launch_identity")
    file_identity = None
    if identity_name:
        path = Path(identity_name)
        try:
            metadata = path.stat()
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            pass
        except OSError as error:
            if state_identity is not None:
                raise ScanError(
                    f"cannot read launch identity {path}: {error}"
                ) from error
            print(
                f"ignoring untrusted legacy launch identity {path}: {error}",
                file=sys.stderr,
            )
        else:
            try:
                if not stat.S_ISREG(metadata.st_mode):
                    raise ScanError(
                        f"launch identity is not a regular file: {path}"
                    )
                if metadata.st_uid != os.geteuid() or metadata.st_mode & 0o022:
                    raise ScanError(
                        f"launch identity has unsafe ownership or mode: {path}"
                    )
                parsed_identity = json.loads(raw)
                if not isinstance(parsed_identity, dict):
                    raise ScanError("launch identity JSON is not an object")
                file_identity = parsed_identity
            except (ScanError, TypeError, ValueError) as error:
                if state_identity is not None:
                    raise ScanError(
                        f"invalid launch identity {path}: {error}"
                    ) from error
                print(
                    f"ignoring untrusted legacy launch identity {path}: {error}",
                    file=sys.stderr,
                )
                file_identity = None
    if file_identity is not None and state_identity is not None:
        if file_identity != state_identity:
            raise ScanError("remote and state launch identities disagree")
    identity = file_identity if file_identity is not None else state_identity
    if identity is None:
        return None
    try:
        required_keys = {
            "schema_version", "pid", "outer_pid", "uid", "boot_id",
            "process_group", "session", "start_time_ticks", "cmdline_sha256",
            "executable", "managed_token", "role", "kind", "start_port",
        }
        if not isinstance(identity, dict) or not required_keys.issubset(identity):
            raise ScanError(
                f"incomplete launch identity: {identity_name or '<state>'}"
            )
        if identity["schema_version"] != 1:
            raise ScanError(
                f"unsupported launch identity schema: {identity['schema_version']}"
            )
        expected = {
            "managed_token": config.get("managed_token"),
            "role": config.get("role"),
            "kind": config.get("kind"),
            "start_port": config.get("start_port"),
        }
        for key, value in expected.items():
            if value is not None and identity.get(key) != value:
                raise ScanError(
                    f"launch identity {key} mismatch: "
                    f"recorded={identity.get(key)!r} expected={value!r}"
                )
        try:
            boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
                encoding="ascii"
            ).strip()
        except OSError as error:
            raise ScanError(f"cannot read kernel boot id: {error}") from error
        if identity["boot_id"] != boot_id:
            raise ScanError("launch identity belongs to a different kernel boot")
        return identity
    except ScanError as error:
        if state_identity is None and file_identity is not None:
            print(
                "ignoring untrusted legacy launch identity "
                f"{identity_name}: {error}",
                file=sys.stderr,
            )
            return None
        raise


def find_outer_pid(namespace_pid, expected_outer_pid=None):
    if not isinstance(namespace_pid, int) or namespace_pid <= 1:
        raise ScanError(f"unsafe recorded launcher pid: {namespace_pid!r}")
    if expected_outer_pid is not None:
        if not isinstance(expected_outer_pid, int) or expected_outer_pid <= 1:
            raise ScanError(f"unsafe recorded outer pid: {expected_outer_pid!r}")
        try:
            observed_namespace_pid = visible_pid_from_status(expected_outer_pid)
        except (FileNotFoundError, ProcessLookupError):
            return None
        if observed_namespace_pid != namespace_pid:
            return None
        if pid_namespace_relation(expected_outer_pid) is not True:
            raise ScanError(
                f"cannot prove launcher pid {namespace_pid} is in this PID namespace"
            )
        return expected_outer_pid
    found = []
    try:
        entries = list(os.scandir("/proc"))
    except OSError as error:
        raise ScanError(f"cannot scan /proc: {error}") from error
    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            if visible_pid_from_status(pid) != namespace_pid:
                continue
            if pid_namespace_relation(pid) is not True:
                raise ScanError(
                    f"recorded launcher pid {namespace_pid} is in an unverifiable namespace"
                )
            found.append(pid)
        except (FileNotFoundError, ProcessLookupError):
            continue
    if len(found) > 1:
        raise ScanError(
            f"recorded launcher pid {namespace_pid} maps to multiple processes: {found}"
        )
    return found[0] if found else None


def validate_launcher(identity):
    outer_pid = find_outer_pid(identity["pid"], identity["outer_pid"])
    if outer_pid is None:
        return None
    if outer_pid != identity["outer_pid"]:
        raise ScanError(
            f"launcher outer pid mismatch: recorded={identity['outer_pid']} "
            f"observed={outer_pid}"
        )
    proc = Path("/proc") / str(outer_pid)
    try:
        if proc.stat().st_uid != os.geteuid():
            raise ScanError(f"launcher pid {outer_pid} has a different owner")
        details = stat_details(outer_pid)
        raw_cmdline = (proc / "cmdline").read_bytes()
        executable = os.path.realpath(os.readlink(proc / "exe"))
    except (FileNotFoundError, ProcessLookupError):
        return None
    except OSError as error:
        if transient(error):
            return None
        raise ScanError(f"cannot validate launcher pid {outer_pid}: {error}") from error
    if details is None or details["state"] == "Z":
        return None
    checks = {
        "start_time_ticks": details["start_time_ticks"],
        "process_group": details["process_group"],
        "session": details["session"],
        "cmdline_sha256": hashlib.sha256(raw_cmdline).hexdigest(),
        "executable": executable,
        "uid": effective_uid(outer_pid),
    }
    for key, observed in checks.items():
        if observed != identity[key]:
            raise ScanError(
                f"launcher identity mismatch for pid {outer_pid}: "
                f"{key} recorded={identity[key]!r} observed={observed!r}"
            )
    if (
        details["process_group"] != outer_pid
        or details["session"] != outer_pid
    ):
        raise ScanError(f"launcher pid {outer_pid} is not its setsid process-group leader")
    return outer_pid


def process_group_members(identity):
    members = []
    try:
        entries = list(os.scandir("/proc"))
    except OSError as error:
        raise ScanError(f"cannot scan /proc: {error}") from error
    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        details = stat_details(pid)
        if (
            details is None
            or details["state"] == "Z"
            or details["process_group"] != identity["process_group"]
        ):
            continue
        if details["session"] != identity["session"]:
            raise ScanError(
                f"pid {pid} shares managed process group but has a different session"
            )
        # A nondumpable child can hide its namespace symlink even from the same
        # effective UID. Group signalling is permitted only after the leader's
        # namespace was proven `True`; here `False` still rejects a foreign
        # namespace, while `None` is retained for conservative observation.
        if pid_namespace_relation(pid) is False or effective_uid(pid) != os.geteuid():
            raise ScanError(
                f"pid {pid} in managed process group has unverifiable ownership/namespace"
            )
        members.append(pid)
    return sorted(members)


def visible_pid(pid):
    proc = Path("/proc") / str(pid)
    try:
        if pid_namespace_relation(pid) is not True:
            raise ScanError(f"pid {pid} is in a different PID namespace")
        status = (proc / "status").read_text(encoding="ascii")
    except (FileNotFoundError, ProcessLookupError):
        return None
    except OSError as error:
        if transient(error):
            return None
        raise ScanError(f"cannot inspect namespace for pid {pid}: {error}") from error
    visible_pid = pid
    for line in status.splitlines():
        if line.startswith("NSpid:"):
            values = line.split()[1:]
            if values:
                visible_pid = int(values[-1])
            break
    if visible_pid <= 1:
        raise ScanError(f"refusing unsafe visible pid {visible_pid} for outer pid {pid}")
    return visible_pid


def signal_pid(pid, signum):
    identity = stat_identity(pid)
    if identity is None or not matches(pid):
        return False
    namespace_pid = visible_pid(pid)
    if namespace_pid is None:
        return False
    if stat_identity(pid) != identity or not matches(pid):
        return False
    if hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"):
        descriptor = os.pidfd_open(namespace_pid)
        try:
            if stat_identity(pid) != identity or not matches(pid):
                return False
            signal.pidfd_send_signal(descriptor, signum, None, 0)
        finally:
            os.close(descriptor)
    else:
        os.kill(namespace_pid, signum)
    return True


def listening_ports():
    test_owners = config.get("test_listening_ports_by_pid")
    if test_owners is not None:
        found = set()
        for pid, ports in test_owners.items():
            details = stat_details(int(pid))
            if details is not None and details["state"] != "Z":
                found.update(int(port) for port in ports)
        return sorted(found)
    wanted = {int(port) for port in config.get("ports", [])}
    if not wanted:
        return []
    found = set()
    tables = config.get("tcp_tables", ["/proc/net/tcp", "/proc/net/tcp6"])
    tables_read = 0
    for table in tables:
        try:
            lines = Path(table).read_text(encoding="ascii").splitlines()[1:]
            tables_read += 1
        except FileNotFoundError:
            continue
        except OSError as error:
            raise ScanError(f"cannot read TCP table {table}: {error}") from error
        for line in lines:
            fields = line.split()
            if len(fields) < 4 or fields[3] != "0A":
                continue
            try:
                port = int(fields[1].rsplit(":", 1)[1], 16)
            except (IndexError, ValueError):
                continue
            if port in wanted:
                found.add(port)
    if tables_read == 0:
        raise ScanError("no readable TCP socket table")
    return sorted(found)


def health_ok():
    port = config.get("health_port")
    if not port:
        return False
    connection = http.client.HTTPConnection("127.0.0.1", int(port), timeout=2)
    try:
        connection.request("GET", "/health")
        response = connection.getresponse()
        response.read(256)
        return 200 <= response.status < 300
    except OSError:
        return False
    finally:
        connection.close()


def signal_matching(signum, pids):
    signalled = []
    for pid in sorted(set(pids), reverse=True):
        if not matches(pid):
            continue
        try:
            if signal_pid(pid, signum):
                signalled.append(pid)
        except ProcessLookupError:
            pass
        except PermissionError as error:
            print(f"cannot signal managed pid {pid}: {error}", file=sys.stderr)
            raise SystemExit(51)
    return signalled


def gpu_processes():
    override = config.get("test_gpu_apps")
    if override is not None:
        return [
            f"{int(pid)}, {name}, test"
            for pid, name in sorted(override.items(), key=lambda item: int(item[0]))
        ]
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ScanError(f"nvidia-smi failed: {error}") from error
    if result.returncode != 0:
        raise ScanError(
            f"nvidia-smi rc={result.returncode}: {result.stdout.strip()}"
        )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def stop_validated_group(identity, timeout):
    leader = validate_launcher(identity)
    if leader is None:
        return False
    members = process_group_members(identity)
    if leader not in members:
        raise ScanError("validated launcher is missing from its process group")
    # Revalidate immediately before the only signal that relies on the live leader.
    if validate_launcher(identity) != leader:
        raise ScanError("launcher changed identity before process-group signal")
    try:
        os.killpg(identity["pid"], signal.SIGTERM)
    except ProcessLookupError:
        return True
    except PermissionError as error:
        raise ScanError(f"cannot SIGTERM managed process group: {error}") from error

    deadline = time.monotonic() + max(0.2, timeout)
    while time.monotonic() < deadline:
        if not process_group_members(identity) and not listening_ports():
            return True
        time.sleep(0.2)
    if process_group_members(identity):
        try:
            os.killpg(identity["pid"], signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError as error:
            raise ScanError(f"cannot SIGKILL managed process group: {error}") from error
    kill_deadline = time.monotonic() + max(0.2, min(10.0, timeout))
    while time.monotonic() < kill_deadline:
        if not process_group_members(identity) and not listening_ports():
            break
        time.sleep(0.1)
    return True


def persisted_identity_group_members(entry):
    if not isinstance(entry, dict):
        raise ScanError("persisted identity entry is not an object")
    name = entry.get("name")
    identity = entry.get("identity")
    if not isinstance(name, str) or not name:
        raise ScanError("persisted identity entry has no resource name")
    required_keys = {
        "schema_version", "pid", "outer_pid", "uid", "boot_id",
        "process_group", "session", "start_time_ticks", "cmdline_sha256",
        "executable", "managed_token", "role", "kind", "start_port",
    }
    if not isinstance(identity, dict) or not required_keys.issubset(identity):
        raise ScanError(f"persisted launch identity for {name} is incomplete")
    if identity["schema_version"] != 1:
        raise ScanError(
            f"persisted launch identity for {name} has unsupported schema"
        )
    numeric_fields = (
        "pid", "outer_pid", "uid", "process_group", "session",
        "start_time_ticks", "start_port",
    )
    if any(type(identity[field]) is not int for field in numeric_fields):
        raise ScanError(f"persisted launch identity for {name} has invalid types")
    if (
        identity["pid"] <= 1
        or identity["outer_pid"] <= 1
        or identity["process_group"] != identity["outer_pid"]
        or identity["session"] != identity["outer_pid"]
        or identity["uid"] != os.geteuid()
    ):
        raise ScanError(f"persisted launch identity for {name} is unsafe")
    try:
        current_boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
            encoding="ascii"
        ).strip()
    except OSError as error:
        raise ScanError(f"cannot read kernel boot id: {error}") from error
    if identity["boot_id"] != current_boot_id:
        # A process cannot survive a kernel boot, so this identity is expired.
        return name, []
    return name, process_group_members(identity)


def unproven_group_members(entry):
    if not isinstance(entry, dict):
        raise ScanError("unproven resource entry is not an object")
    name = entry.get("name")
    if not isinstance(name, str) or not name:
        raise ScanError("unproven resource entry has no resource name")
    namespace_pgid = entry.get("pid")
    if namespace_pgid is None:
        pid_file = entry.get("pid_file")
        if pid_file:
            try:
                namespace_pgid = int(Path(pid_file).read_text().strip())
            except FileNotFoundError:
                return name, []
            except (OSError, TypeError, ValueError) as error:
                raise ScanError(
                    f"cannot read legacy pid for {name}: {error}"
                ) from error
    if namespace_pgid is None:
        return name, []
    if type(namespace_pgid) is not int or namespace_pgid <= 1:
        raise ScanError(f"legacy pid for {name} is unsafe: {namespace_pgid!r}")

    members = []
    try:
        entries = list(os.scandir("/proc"))
    except OSError as error:
        raise ScanError(f"cannot scan /proc: {error}") from error
    for process in entries:
        if not process.name.isdigit():
            continue
        pid = int(process.name)
        details = stat_details(pid)
        if details is None or details["state"] == "Z":
            continue
        try:
            status = (Path(process.path) / "status").read_text(encoding="ascii")
        except (FileNotFoundError, ProcessLookupError):
            continue
        except OSError as error:
            if transient(error):
                continue
            raise ScanError(f"cannot inspect group for pid {pid}: {error}") from error
        namespace_groups = None
        for line in status.splitlines():
            if line.startswith("NSpgid:"):
                namespace_groups = [int(value) for value in line.split()[1:]]
                break
        if namespace_groups is None:
            raise ScanError(f"NSpgid is missing for pid {pid}")
        if namespace_groups and namespace_groups[-1] == namespace_pgid:
            members.append(pid)
    return name, sorted(members)


action = config["action"]
if action == "status":
    identity = load_launch_identity()
    if identity is not None:
        validate_launcher(identity)
        pids = process_group_members(identity)
    else:
        pids = matching_pids()
    alive = bool(pids)
    healthy = alive and health_ok()
    print(f"pids={','.join(map(str, pids)) or '-'} "
          f"{'alive' if alive else 'dead'} "
          f"{'healthy' if healthy else 'unhealthy'}")
    raise SystemExit(0)

if action == "assert_no_managed":
    pids = matching_pids()
    if pids:
        print(json.dumps({"managed_pids": pids}, separators=(",", ":")))
        raise SystemExit(27)
    print("no token-managed processes")
    raise SystemExit(0)

if action == "stop":
    timeout = max(1.0, float(config.get("timeout", 10)))
    identity = load_launch_identity()
    had_process = False
    used_identity = identity is not None
    legacy_scan_error = None
    if identity is not None:
        had_process = stop_validated_group(identity, timeout)
    else:
        # Backward compatibility for launches created before identity files existed.
        try:
            legacy_pids = matching_pids()
        except ScanError as error:
            legacy_scan_error = error
            legacy_pids = []
        had_process = bool(legacy_pids)
        if legacy_pids:
            signal_matching(signal.SIGTERM, legacy_pids)
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                try:
                    remaining = matching_pids()
                except ScanError as error:
                    legacy_scan_error = error
                    break
                if not remaining and not listening_ports():
                    break
                signal_matching(signal.SIGTERM, remaining)
                time.sleep(0.2)

    if listening_ports() and config.get("backend_instance"):
        detail = (
            f"; token scan failed: {legacy_scan_error}"
            if legacy_scan_error is not None
            else ""
        )
        raise ScanError(
            "managed ports remain, but state has no persisted causal backend "
            "PID/start-time identity; refusing detached-backend kill. Use a "
            "pinned PID/start-time/port cleanup"
            + detail
        )

    if used_identity:
        survivors = process_group_members(identity)
    else:
        # Re-scan after legacy fallback. An inaccessible managed-looking process
        # still fails closed unless it disappeared as part of backend shutdown.
        survivors = matching_pids()
    ports = listening_ports()
    if survivors or ports:
        print(
            json.dumps(
                {"error": "managed resource did not stop", "pids": survivors,
                 "listening_ports": ports},
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        raise SystemExit(50)
    pid_file = Path(config["pid_file"])
    try:
        pid_file.unlink()
    except FileNotFoundError:
        pass
    identity_name = config.get("identity_file")
    if identity_name:
        try:
            Path(identity_name).unlink()
        except FileNotFoundError:
            pass
    state = "stopped" if had_process else "not running"
    print(f"{state}; token processes and ports are clear")
    raise SystemExit(0)

if action == "verify_host":
    timeout = max(1.0, float(config.get("timeout", 60)))
    deadline = time.monotonic() + timeout
    diagnostics = {}
    while time.monotonic() < deadline:
        try:
            managed_pids = matching_pids()
            scan_error = None
        except ScanError as error:
            managed_pids = []
            scan_error = str(error)
        identity_group_survivors = {}
        identity_errors = {}
        for entry in config.get("launch_identities", []):
            entry_name = (
                entry.get("name", "<unknown>")
                if isinstance(entry, dict)
                else "<invalid>"
            )
            try:
                name, members = persisted_identity_group_members(entry)
                if members:
                    identity_group_survivors[name] = members
            except ScanError as error:
                identity_errors[str(entry_name)] = str(error)
        unproven_group_survivors = {}
        unproven_group_errors = {}
        for entry in config.get("unproven_resources", []):
            entry_name = (
                entry.get("name", "<unknown>")
                if isinstance(entry, dict)
                else "<invalid>"
            )
            try:
                name, members = unproven_group_members(entry)
                if members:
                    unproven_group_survivors[name] = members
            except ScanError as error:
                unproven_group_errors[str(entry_name)] = str(error)
        diagnostics = {
            "managed_pids": managed_pids,
            "listening_ports": listening_ports(),
            "gpu_processes": gpu_processes() if config.get("require_gpu_clear") else [],
            "identity_group_survivors": identity_group_survivors,
            "identity_errors": identity_errors,
            "unproven_group_survivors": unproven_group_survivors,
            "unproven_group_errors": unproven_group_errors,
            "scan_error": scan_error,
        }
        if not any(diagnostics.values()):
            print("host stop postcondition passed")
            raise SystemExit(0)
        time.sleep(1.0)
    print(json.dumps(diagnostics, separators=(",", ":")), file=sys.stderr)
    raise SystemExit(52)

print(f"unsupported managed process action: {action}", file=sys.stderr)
raise SystemExit(53)
"""


def management_environment(resource: dict[str, Any]) -> dict[str, str]:
    return {
        "RTP_CLUSTER_MANAGED": "1",
        "RTP_CLUSTER_RUN_ID": str(resource["run_id"]),
        "RTP_CLUSTER_RESOURCE_ID": str(resource["resource_id"]),
        "RTP_CLUSTER_TOKEN": str(resource["managed_token"]),
        "RTP_CLUSTER_START_PORT": str(resource["start_port"]),
        "RTP_CLUSTER_ROLE": str(resource["role"]),
        "RTP_CLUSTER_KIND": str(resource.get("kind", "RTP")),
        "RTP_CLUSTER_GPU_INDICES": ",".join(
            str(gpu) for gpu in resource.get("gpus", [])
        ),
    }


def managed_helper_command(payload: dict[str, Any]) -> str:
    return shell_join(
        [PYTHON, "-c", REMOTE_MANAGED_PROCESS_HELPER, json.dumps(payload)]
    )


def managed_payload(
    resource: dict[str, Any], action: str, *, timeout: int = 0
) -> dict[str, Any]:
    kind = str(resource.get("kind", "RTP"))
    role = str(resource["role"])
    start_port = int(resource["start_port"])
    is_backend_instance = (
        kind == "RTP"
        and role in {"PREFILL", "DECODE", "PDFUSION"}
        and bool(resource.get("gpus"))
    )
    payload: dict[str, Any] = {
        "action": action,
        "required_env": [
            f"{key}={value}" for key, value in management_environment(resource).items()
        ],
        "ports": sorted(resource_port_range(resource, start_port)),
        "health_port": int(resource.get("health_port", start_port)),
        "pid_file": f"{resource['remote_dir']}/server.pid",
        "identity_file": f"{resource['remote_dir']}/server.identity.json",
        "launch_identity": resource.get("launch_identity"),
        "managed_token": str(resource["managed_token"]),
        "role": role,
        "kind": kind,
        "start_port": start_port,
        "backend_instance": is_backend_instance,
    }
    if timeout:
        payload["timeout"] = timeout
    return payload


def ssh_run(
    host: dict[str, Any],
    script: str,
    *,
    timeout: int = 60,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    command = ["ssh", *SSH_OPTIONS, host["ip"], "bash --noprofile --norc -s"]
    try:
        result = subprocess.run(
            command,
            input=script,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ClusterError(f"SSH timeout on {host['name']}: {exc}") from exc
    if check and result.returncode != 0:
        raise ClusterError(
            f"Remote command failed on {host['name']} (rc={result.returncode}):\n"
            f"{result.stdout.rstrip()}"
        )
    return result


def parse_ratio(value: str) -> tuple[int, int]:
    try:
        prefill_text, decode_text = value.split(":", 1)
        prefill_count = int(prefill_text)
        decode_count = int(decode_text)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("ratio must use P:D integer syntax") from exc
    if prefill_count <= 0 or decode_count <= 0:
        raise argparse.ArgumentTypeError("P and D must both be positive")
    return prefill_count, decode_count


def gpu_split(ratio: tuple[int, int]) -> tuple[int, int]:
    """Resolve a requested P:D card ratio onto the fixed 32-GPU inventory."""
    if DECODE_TP != 1 and DECODE_DP == 1:
        raise ClusterError("Decode TP must remain 1 so every Decode has its own queue")
    prefill_weight, decode_weight = ratio
    target_prefill = TOTAL_GPUS * prefill_weight / (prefill_weight + decode_weight)
    feasible = [
        prefill_gpus
        for prefill_gpus in range(PREFILL_GPUS_PER_INST, TOTAL_GPUS)
        if prefill_gpus % PREFILL_GPUS_PER_INST == 0
        and (TOTAL_GPUS - prefill_gpus) % DECODE_GPUS_PER_INST == 0
        and TOTAL_GPUS - prefill_gpus >= DECODE_GPUS_PER_INST
    ]
    if not feasible:
        raise ClusterError(
            f"no feasible split for {TOTAL_GPUS} GPUs with "
            f"Prefill TP={PREFILL_TP}, Decode TP={DECODE_TP}"
        )
    prefill_gpus = min(
        feasible,
        key=lambda value: (abs(value - target_prefill), -value),
    )
    return prefill_gpus, TOTAL_GPUS - prefill_gpus


def host_role_gpu_counts(prefill_gpu_count: int) -> dict[str, dict[str, int]]:
    """Minimize mixed-role hosts while respecting both roles' TP sizes."""
    options: list[list[int]] = []
    for host in HOSTS:
        capacity = int(host["gpus"])
        options.append(
            [
                prefill_count
                for prefill_count in range(0, capacity + 1)
                if prefill_count % PREFILL_GPUS_PER_INST == 0
                and (capacity - prefill_count) % DECODE_GPUS_PER_INST == 0
            ]
        )

    candidates: list[tuple[tuple[Any, ...], tuple[int, ...]]] = []
    for allocation in product(*options):
        if sum(allocation) != prefill_gpu_count:
            continue
        by_host = {host["name"]: allocation[index] for index, host in enumerate(HOSTS)}
        mixed_host_count = sum(
            0 < by_host[host["name"]] < int(host["gpus"]) for host in HOSTS
        )
        # For equally isolated plans, fill preferred Prefill hosts first.
        preference = tuple(-by_host[name] for name in PREFILL_HOST_ORDER)
        candidates.append(((mixed_host_count, preference), allocation))

    if not candidates:
        raise ClusterError(
            f"cannot place {prefill_gpu_count} Prefill GPUs with "
            f"Prefill TP={PREFILL_TP}, Decode TP={DECODE_TP}"
        )

    _, selected = min(candidates, key=lambda item: item[0])
    return {
        host["name"]: {
            "PREFILL": selected[index],
            "DECODE": int(host["gpus"]) - selected[index],
        }
        for index, host in enumerate(HOSTS)
    }


def contiguous_gpus(available: list[int], count: int) -> list[int] | None:
    available_set = set(available)
    for start in available:
        candidate = list(range(start, start + count))
        if all(gpu in available_set for gpu in candidate):
            return candidate
    return None


def allocate_instance(
    available: dict[str, list[int]],
    order: list[str],
    cursor: int,
    gpu_count: int,
) -> tuple[dict[str, Any], list[int], int]:
    for offset in range(len(order)):
        index = (cursor + offset) % len(order)
        host = HOST_BY_NAME[order[index]]
        selected = contiguous_gpus(available[host["name"]], gpu_count)
        if selected is None:
            continue
        for gpu in selected:
            available[host["name"]].remove(gpu)
        return host, selected, (index + 1) % len(order)
    raise ClusterError(
        f"not enough contiguous GPU capacity for an instance needing {gpu_count} GPUs"
    )


def make_plan(
    ratio: tuple[int, int], cache_store_rdma_mode: bool = CACHE_STORE_RDMA_MODE
) -> dict[str, Any]:
    for name, value in (
        ("RTP_REUSE_CACHE", REUSE_CACHE),
        ("RTP_PREFILL_REUSE_CACHE", PREFILL_REUSE_CACHE),
        ("RTP_DECODE_REUSE_CACHE", DECODE_REUSE_CACHE),
    ):
        if value not in (0, 1):
            raise ClusterError(f"{name} must be 0 or 1")
    if INDUCTOR_COMPILE_THREADS < 1:
        raise ClusterError("RTP_INDUCTOR_COMPILE_THREADS must be positive")
    prefill_gpus, decode_gpus = gpu_split(ratio)
    prefill_count = prefill_gpus // PREFILL_TP
    decode_count = decode_gpus // DECODE_TP
    host_role_counts = host_role_gpu_counts(prefill_gpus)
    available = {host["name"]: list(range(int(host["gpus"]))) for host in HOSTS}
    instances: list[dict[str, Any]] = []

    decodes: list[dict[str, Any]] = []
    # Per-instance plan: (concurrency, tp_size, bucket_label)
    if DECODE_BUCKETS:
        bucket_gpu_total = sum(count * tp for count, _, tp in DECODE_BUCKETS)
        if bucket_gpu_total != decode_gpus:
            raise ClusterError(
                f"RTP_DECODE_BUCKET_SPEC covers {bucket_gpu_total} GPUs, "
                f"plan has {decode_gpus}"
            )
        decode_plan = []
        for bucket_idx, (count, conc, tp) in enumerate(DECODE_BUCKETS):
            label = "short" if bucket_idx == 0 else "long"
            decode_plan.extend((conc, tp, label) for _ in range(count))
    else:
        decode_plan = [(DECODE_CONCURRENCY, DECODE_GPUS_PER_INST, None)] * (
            decode_gpus // DECODE_GPUS_PER_INST
        )
    decode_hosts = [
        host_name
        for host_name in DECODE_HOST_ORDER
        for _ in range(host_role_counts[host_name]["DECODE"])
    ]
    host_cursor = 0
    for index, (concurrency, tp, bucket) in enumerate(decode_plan):
        host_name = decode_hosts[host_cursor]
        host_cursor += tp
        host, gpus, _ = allocate_instance(available, [host_name], 0, tp)
        port = DECODE_BASE_PORT + index * PORT_STRIDE
        if port + 20 >= 65536:
            raise ClusterError("Decode port range exceeds 65535")
        instance = {
            "name": f"decode-{index:02d}",
            "role": "DECODE",
            "host": host["name"],
            "ip": host["ip"],
            "gpus": gpus,
            "start_port": port,
            "tp_size": tp,
            "ep_size": tp,
            "real_tp": DECODE_TP,
            "dp_size": DECODE_DP,
            "concurrency": concurrency,
            "kv_cache_mem_mb": DECODE_KV_MB,
            "reuse_cache": DECODE_REUSE_CACHE,
            "rdma_nic": host["rdma_nic"],
        }
        if bucket is not None:
            instance["bucket"] = bucket
        decodes.append(instance)
        instances.append(instance)

    prefills: list[dict[str, Any]] = []
    prefill_hosts = [
        host_name
        for host_name in PREFILL_HOST_ORDER
        for _ in range(host_role_counts[host_name]["PREFILL"] // PREFILL_GPUS_PER_INST)
    ]
    for index, host_name in enumerate(prefill_hosts):
        host, gpus, _ = allocate_instance(
            available, [host_name], 0, PREFILL_GPUS_PER_INST
        )
        port = PREFILL_BASE_PORT + index * PORT_STRIDE
        if port + 20 >= 65536:
            raise ClusterError("Prefill port range exceeds 65535")
        decode = decodes[index % len(decodes)]
        instance = {
            "name": f"prefill-{index:02d}",
            "role": "PREFILL",
            "host": host["name"],
            "ip": host["ip"],
            "gpus": gpus,
            "start_port": port,
            "tp_size": PREFILL_GPUS_PER_INST,
            "ep_size": PREFILL_GPUS_PER_INST,
            "real_tp": PREFILL_TP,
            "dp_size": PREFILL_DP,
            "concurrency": PREFILL_CONCURRENCY,
            "kv_cache_mem_mb": PREFILL_KV_MB,
            "reuse_cache": PREFILL_REUSE_CACHE,
            "peer_name": decode["name"],
            "rdma_nic": host["rdma_nic"],
        }
        prefills.append(instance)
        instances.append(instance)

    by_name = {instance["name"]: instance for instance in instances}
    for index, decode in enumerate(decodes):
        decode["peer_name"] = prefills[index % len(prefills)]["name"]

    for instance in instances:
        peer = by_name[instance["peer_name"]]
        instance["peer_ip"] = peer["ip"]
        instance["peer_port"] = peer["start_port"]
        instance["remote_dir"] = f"{REMOTE_ROOT}/{instance['name']}"
        instance["cache_store_rdma_mode"] = (
            bool(cache_store_rdma_mode)
            and os.environ.get("RTP_CACHE_TRANSPORT_TCP") != "1"
        )

    flexlb_host = HOST_BY_NAME.get(FLEXLB_HOST_NAME)
    if flexlb_host is None:
        raise ClusterError(f"unknown FlexLB host: {FLEXLB_HOST_NAME}")
    services = [
        {
            "name": "flexlb",
            "kind": "FLEXLB",
            "role": "CONTROL",
            "host": flexlb_host["name"],
            "ip": flexlb_host["ip"],
            "start_port": FLEXLB_BASE_PORT,
            "health_port": FLEXLB_BASE_PORT + 1,
            "remote_dir": f"{REMOTE_ROOT}/flexlb",
        },
        {
            "name": "frontend",
            "kind": "RTP",
            "role": "FRONTEND",
            "host": flexlb_host["name"],
            "ip": flexlb_host["ip"],
            "start_port": FRONTEND_BASE_PORT,
            "health_port": FRONTEND_BASE_PORT,
            "remote_dir": f"{REMOTE_ROOT}/frontend",
        },
    ]

    run_id = secrets.token_hex(16)
    for resource in [*instances, *services]:
        resource["run_id"] = run_id
        resource["resource_id"] = resource["name"]
        resource["managed_token"] = secrets.token_hex(32)

    plan = {
        "state_schema_version": STATE_SCHEMA_VERSION,
        "run_id": run_id,
        "ratio": f"{ratio[0]}:{ratio[1]}",
        "gpu_total": TOTAL_GPUS,
        "prefill_gpu_count": prefill_gpus,
        "decode_gpu_count": decode_gpus,
        "prefill_instance_count": prefill_count,
        "decode_instance_count": decode_count,
        "max_seq_len": MAX_SEQ_LEN,
        "frontend_concurrency": FRONTEND_CONCURRENCY,
        "inductor_compile_threads": INDUCTOR_COMPILE_THREADS,
        "cache_store_rdma_mode": bool(cache_store_rdma_mode),
        "cache_transport": "rdma" if cache_store_rdma_mode else "tcp",
        "source_path": RDMA_SOURCE_PATH if cache_store_rdma_mode else SOURCE_PATH,
        "python": PYTHON,
        "pythonpath": RDMA_PYTHONPATH if cache_store_rdma_mode else "",
        "runtime_path": RDMA_RUNTIME_PATH if cache_store_rdma_mode else "",
        "flexlb_runtime_config": flexlb_runtime_config(),
        "decode_routing_scheme": DECODE_ROUTING_SCHEME,
        "decode_cache_load_beta": DECODE_CACHE_LOAD_BETA,
        # Keep the legacy aggregate field for readers that only understand
        # all-role reuse. Role-specific fields are authoritative.
        "reuse_cache": bool(PREFILL_REUSE_CACHE and DECODE_REUSE_CACHE),
        "prefill_reuse_cache": bool(PREFILL_REUSE_CACHE),
        "decode_reuse_cache": bool(DECODE_REUSE_CACHE),
        "host_role_gpu_counts": host_role_counts,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "planned",
        "instances": instances,
        "services": services,
    }
    validate_plan(plan)
    return plan


def validate_plan(plan: dict[str, Any]) -> None:
    used: dict[str, set[int]] = {host["name"]: set() for host in HOSTS}
    for instance in plan["instances"]:
        if instance["role"] == "DECODE" and (
            int(instance["tp_size"]) != len(instance["gpus"])
            or (not DECODE_BUCKETS and DECODE_DP == 1 and int(instance["tp_size"]) != 1)
        ):
            raise ClusterError(f"{instance['name']} decode tp/gpu mismatch")
        if plan["cache_store_rdma_mode"]:
            expected_nic = HOST_BY_NAME[instance["host"]]["rdma_nic"]
            if instance.get("rdma_nic") != expected_nic:
                raise ClusterError(
                    f"{instance['name']} RDMA NIC is {instance.get('rdma_nic')}, "
                    f"expected {expected_nic}"
                )
        for gpu in instance["gpus"]:
            if gpu in used[instance["host"]]:
                raise ClusterError(f"GPU {gpu} allocated twice on {instance['host']}")
            used[instance["host"]].add(gpu)
    for host in HOSTS:
        expected = set(range(int(host["gpus"])))
        if used[host["name"]] != expected:
            raise ClusterError(
                f"plan does not use every GPU on {host['name']}: "
                f"used={sorted(used[host['name']])}, expected={sorted(expected)}"
            )


def instance_port_range(instance: dict[str, Any], start_port: int) -> set[int]:
    # Every rank uses +2 for CacheStore control/data-over-TCP and +4 for RDMA.
    last_port = start_port + (int(instance["tp_size"]) - 1) * 8 + 7
    return set(range(start_port - 11, last_port + 1))


def instance_rank_ports(instance: dict[str, Any], offset: int) -> list[int]:
    return [
        int(instance["start_port"]) + rank * 8 + offset
        for rank in range(int(instance["tp_size"]))
    ]


def resource_port_range(resource: dict[str, Any], start_port: int) -> set[int]:
    if resource.get("kind") == "FLEXLB":
        return {start_port, start_port + 1}
    if resource.get("role") == "FRONTEND":
        return set(range(start_port - 11, start_port + 8))
    return instance_port_range(resource, start_port)


def listening_ports(host: dict[str, Any]) -> set[int]:
    result = ssh_run(
        host,
        'ss -ltnH | awk \'{addr=$4; sub(/^.*:/, "", addr); '
        "if (addr ~ /^[0-9]+$/) print addr}'",
        timeout=30,
    )
    return {
        int(line.strip())
        for line in result.stdout.splitlines()
        if line.strip().isdigit()
    }


def resolve_ports(plan: dict[str, Any]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for resource in [*plan["instances"], *plan.get("services", [])]:
        grouped[resource["host"]].append(resource)

    for host_name, instances in grouped.items():
        host = HOST_BY_NAME[host_name]
        occupied = listening_ports(host)
        reserved: set[int] = set()
        for resource in instances:
            preferred = int(resource["start_port"])
            candidate = preferred
            while candidate < 65536:
                ports = resource_port_range(resource, candidate)
                if min(ports) > 1024 and max(ports) < 65536:
                    if ports.isdisjoint(occupied) and ports.isdisjoint(reserved):
                        break
                candidate += PORT_STRIDE
            else:
                raise ClusterError(
                    f"no free port range for {resource['name']} on {host_name}"
                )
            if candidate != preferred:
                print(
                    f"port adjust {resource['name']} on {host_name}: "
                    f"{preferred} -> {candidate}",
                    flush=True,
                )
            resource["start_port"] = candidate
            if resource.get("kind") == "FLEXLB":
                resource["health_port"] = candidate + 1
            elif resource["role"] == "FRONTEND":
                resource["health_port"] = candidate
            reserved.update(ports)

    by_name = {instance["name"]: instance for instance in plan["instances"]}
    for instance in plan["instances"]:
        peer = by_name[instance["peer_name"]]
        instance["peer_port"] = peer["start_port"]


def endpoint(instance: dict[str, Any]) -> dict[str, str]:
    return {
        "type": "Vipserver",
        "address": f"{instance['ip']}:{instance['start_port']}",
        "protocol": "http",
        "path": "/",
    }


def role_address(plan: dict[str, Any], role: str) -> str:
    return ",".join(
        f"{item['ip']}:{item['start_port']}"
        for item in plan["instances"]
        if item["role"] == role
    )


def decode_worker_address(plan: dict[str, Any]) -> str:
    """Expand DECODE instances into one endpoint per DP group leader.

    Each local rank listens at base = start_port + local_rank * 8 (http=base,
    grpc=base+1) and serves GetWorkerStatus/RemoteGenerate, so FlexLB can treat
    every DP group as an independent decode worker. Without this expansion all
    requests land on dp_rank 0 and the other ranks' KV pools idle (observed:
    decode effective capacity = 1/dp_size).
    """
    if os.environ.get("RTP_DECODE_RANK_FANOUT", "1") == "0":
        return role_address(plan, "DECODE")
    entries = []
    for item in plan["instances"]:
        if item["role"] != "DECODE":
            continue
        dp = int(item.get("dp_size") or 1)
        tp = int(item.get("real_tp") or 1)
        for group in range(max(dp, 1)):
            entries.append(f"{item['ip']}:{item['start_port'] + group * tp * 8}")
    return ",".join(entries)


def direct_endpoint(plan: dict[str, Any], role: str) -> dict[str, str]:
    return {
        "type": "Vipserver",
        "address": role_address(plan, role),
        "protocol": "http",
        "path": "/",
    }


def worker_service_config(plan: dict[str, Any]) -> str:
    config = {
        "service_id": SERVICE_ID,
        "role_endpoints": [
            {
                "group": "default",
                "prefill_endpoint": direct_endpoint(plan, "PREFILL"),
                "decode_endpoint": direct_endpoint(plan, "DECODE"),
            }
        ],
        "use_local": True,
    }
    return json.dumps(config, separators=(",", ":"))


def flexlb_service_config() -> str:
    config = {
        "service_id": SERVICE_ID,
        "load_balance": True,
        "role_endpoints": [
            {
                "group": "default",
                "prefill_endpoint": {
                    "address": "rtp-hol-prefill",
                    "protocol": "http",
                    "path": "/",
                },
                "decode_endpoint": {
                    "address": "rtp-hol-decode",
                    "protocol": "http",
                    "path": "/",
                },
            }
        ],
    }
    return json.dumps(config, separators=(",", ":"))


def flexlb_runtime_config() -> dict[str, Any]:
    config = {
        "enableQueueing": env_flag("RTP_FLEXLB_ENABLE_QUEUEING", False),
        "prefillQueueSizeThreshold": FLEXLB_PREFILL_QUEUE_THRESHOLD,
        "maxPrefillQueueSize": FLEXLB_PREFILL_QUEUE_THRESHOLD,
        "decodeRoutingScheme": DECODE_ROUTING_SCHEME,
        "decodeCacheLoadBeta": DECODE_CACHE_LOAD_BETA,
    }
    if DECODE_BUCKETS:
        short_count = DECODE_BUCKETS[0][0]
        long_count = sum(count for count, _, _ in DECODE_BUCKETS[1:])
        long_ports = [
            DECODE_BASE_PORT + index * PORT_STRIDE
            for index in range(short_count, short_count + long_count)
        ]
        config["decodeBucketSeqLenThreshold"] = DECODE_BUCKET_SEQLEN
        config["decodeLongBucketPorts"] = ",".join(str(p) for p in long_ports)
    if env_flag("RTP_FLEXLB_QUEUE_SRPT", False):
        config["queueSrptEnabled"] = True
        config["srptAgingWindowS"] = int(
            os.environ.get("RTP_SRPT_AGING_WINDOW_S", "30")
        )
    if env_flag("RTP_FLEXLB_SPILLOVER", False):
        config["decodeSpilloverEnabled"] = True
        config["decodeSpilloverBatchThreshold"] = int(
            os.environ.get("RTP_SPILLOVER_BATCH_THRESHOLD", "76")
        )
        config["decodeSpilloverMaxPerWorker"] = int(
            os.environ.get("RTP_SPILLOVER_MAX_PER_WORKER", "8")
        )
        config["decodeSpilloverMaxExpectedOutput"] = int(
            os.environ.get("RTP_SPILLOVER_MAX_EXPECTED_OUTPUT", "0")
        )
    if DECODE_LONG_OUT_THRESHOLD > 0:
        config["decodeLongOutputThreshold"] = DECODE_LONG_OUT_THRESHOLD
        config["decodeLongSlotQuota"] = DECODE_LONG_SLOT_QUOTA
    return config


def frontend_service_config(plan: dict[str, Any]) -> str:
    flexlb = next(item for item in plan["services"] if item["name"] == "flexlb")
    config = json.loads(worker_service_config(plan))
    config["master_endpoint"] = endpoint(flexlb)
    return json.dumps(config, separators=(",", ":"))


def server_args(instance: dict[str, Any], plan: dict[str, Any]) -> list[str]:
    args = [
        plan["python"],
        "-m",
        "rtp_llm.start_server",
        "--checkpoint_path",
        MODEL_PATH,
        "--tokenizer_path",
        MODEL_PATH,
        "--model_type",
        MODEL_TYPE,
        "--act_type",
        "bf16",
        "--role_type",
        instance["role"],
        "--start_port",
        str(instance["start_port"]),
        "--use_local",
        "1",
        "--remote_rpc_server_ip",
        instance["peer_ip"],
        "--remote_server_port",
        str(instance["peer_port"]),
        "--cache_store_rdma_mode",
        "1" if instance["cache_store_rdma_mode"] else "0",
        "--load_cache_timeout_ms",
        "120000",
        "--max_seq_len",
        str(MAX_SEQ_LEN),
        "--reuse_cache",
        str(instance["reuse_cache"]),
        "--warm_up",
        "0",
        "--enable_cuda_graph",
        os.environ.get("RTP_ENABLE_CUDA_GRAPH", "0"),
        "--seq_size_per_block",
        "64",
        "--concurrency_limit",
        str(instance["concurrency"]),
        "--kv_cache_mem_mb",
        str(instance["kv_cache_mem_mb"]),
        "--reserver_runtime_mem_mb",
        str(RUNTIME_RESERVE_MB),
        "--torch_cuda_profiler_dir",
        instance["remote_dir"],
        "--tp_size",
        str(instance.get("real_tp") or instance["tp_size"]),
        "--ep_size",
        str(instance["ep_size"]),
        "--dp_size",
        str(instance.get("dp_size") or 1),
        "--world_size",
        str(instance["tp_size"]),
        "--world_rank",
        "0",
        "--local_world_size",
        str(instance["tp_size"]),
    ]
    if int(instance.get("dp_size") or 1) > 1:
        args.extend(["--max_context_batch_size", "1"])
    if (
        instance["role"] == "PREFILL"
        and instance["tp_size"] > 1
        and int(instance.get("dp_size") or 1) == 1
    ):
        args.extend(
            [
                "--use_all_gather",
                "1",
                "--use_deepep_moe",
                "0",
                "--use_deepep_low_latency",
                "0",
            ]
        )
    return args


def managed_launch_script(
    resource: dict[str, Any], command: list[str], environment: dict[str, str]
) -> str:
    launch_environment = dict(environment)
    for key, value in management_environment(resource).items():
        if key in launch_environment and launch_environment[key] != value:
            raise ClusterError(f"reserved managed environment key overridden: {key}")
        launch_environment[key] = value
    env_args = [f"{key}={value}" for key, value in launch_environment.items()]
    full_command = ["env", *env_args, *command]
    remote_dir = shlex.quote(resource["remote_dir"])
    display_command = shell_join(full_command)
    identity_command = shell_join(
        [
            PYTHON,
            "-c",
            REMOTE_RECORD_LAUNCH_IDENTITY,
            f"{resource['remote_dir']}/server.identity.json",
        ]
    )
    identity_args = shell_join(
        [
            str(resource["managed_token"]),
            str(resource["role"]),
            str(resource.get("kind", "RTP")),
            str(resource["start_port"]),
            command[0],
        ]
    )
    failure_cleanup_command = managed_helper_command(
        managed_payload(resource, "stop", timeout=5)
    )
    return f"""set -e
mkdir -p {remote_dir}
pid_file={remote_dir}/server.pid
identity_file={remote_dir}/server.identity.json
log_file={remote_dir}/server.log
if [ -s \"$pid_file\" ] && kill -0 \"$(cat \"$pid_file\")\" 2>/dev/null; then
  echo \"resource already running: $(cat \"$pid_file\")\"
  exit 42
fi
rm -f \"$identity_file\"
if [ -f \"$log_file\" ]; then
  mv \"$log_file\" \"$log_file.$(date +%Y%m%d-%H%M%S)\"
fi
printf '%s\\n' {shlex.quote(display_command)} > {remote_dir}/command.txt
nohup setsid {display_command} >\"$log_file\" 2>&1 < /dev/null &
pid=$!
printf '%s\\n' \"$pid\" >\"$pid_file\"
if ! {identity_command} \"$pid\" {identity_args}; then
  cleanup_rc=0
  {failure_cleanup_command} || cleanup_rc=$?
  echo \"failed to persist launcher identity for pid $pid; fail-closed cleanup rc=$cleanup_rc\" >&2
  exit 43
fi
printf 'RTP_LAUNCH_IDENTITY=%s\\n' \"$(cat \"$identity_file\")\"
sleep 2
kill -0 \"$pid\"
echo \"$pid\"
"""


def record_managed_launch_result(
    resource: dict[str, Any], result: subprocess.CompletedProcess[str]
) -> None:
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise ClusterError(f"{resource['name']} launch returned no process identity")
    try:
        pid = int(lines[-1])
    except ValueError as error:
        raise ClusterError(
            f"{resource['name']} launch returned an invalid pid: {lines[-1]!r}"
        ) from error
    prefix = "RTP_LAUNCH_IDENTITY="
    identity_lines = [line[len(prefix) :] for line in lines if line.startswith(prefix)]
    if len(identity_lines) != 1:
        raise ClusterError(
            f"{resource['name']} launch returned {len(identity_lines)} identities"
        )
    try:
        identity = json.loads(identity_lines[0])
    except (TypeError, ValueError) as error:
        raise ClusterError(
            f"{resource['name']} launch returned malformed identity JSON"
        ) from error
    expected = {
        "schema_version": 1,
        "pid": pid,
        "managed_token": str(resource["managed_token"]),
        "role": str(resource["role"]),
        "kind": str(resource.get("kind", "RTP")),
        "start_port": int(resource["start_port"]),
    }
    if not isinstance(identity, dict):
        raise ClusterError(f"{resource['name']} launch identity is not an object")
    mismatches = {
        key: {"expected": value, "observed": identity.get(key)}
        for key, value in expected.items()
        if identity.get(key) != value
    }
    required_fields = {
        "outer_pid",
        "uid",
        "boot_id",
        "process_group",
        "session",
        "start_time_ticks",
        "cmdline_sha256",
        "executable",
    }
    missing = sorted(required_fields - identity.keys())
    if mismatches or missing:
        raise ClusterError(
            f"{resource['name']} launch identity validation failed: "
            f"mismatches={mismatches}, missing={missing}"
        )
    resource["pid"] = pid
    resource["launch_identity"] = identity


def launch_script(instance: dict[str, Any], plan: dict[str, Any]) -> str:
    rdma_mode = bool(instance["cache_store_rdma_mode"])
    environment = {
        "CUDA_VISIBLE_DEVICES": ",".join(str(gpu) for gpu in instance["gpus"]),
        "LD_LIBRARY_PATH": "/opt/conda310/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib64:/usr/lib64",
        "MODEL_SERVICE_CONFIG": worker_service_config(plan),
        "REMOTE_RPC_SERVER_IP": instance["peer_ip"],
        "REMOTE_SERVER_PORT": str(instance["peer_port"]),
        "ROLE_TYPE": instance["role"],
        "USE_LOCAL": "1",
        "CACHE_STORE_RDMA_MODE": "1" if rdma_mode else "0",
        "DECODE_ENTRANCE": "0",
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "NCCL_DEBUG": "WARN",
        "NCCL_SOCKET_IFNAME": "eth0",
        "LOG_LEVEL": "INFO",
        "OMP_NUM_THREADS": "8",
        "TORCHINDUCTOR_COMPILE_THREADS": str(INDUCTOR_COMPILE_THREADS),
    }
    if rdma_mode:
        environment["ACCL_USE_NICS"] = instance["rdma_nic"]
    if plan["pythonpath"]:
        environment["PYTHONPATH"] = plan["pythonpath"]
    for _pair in os.environ.get("RTP_WORKER_EXTRA_ENV", "").split(","):
        if "=" in _pair:
            _k, _v = _pair.split("=", 1)
            environment[_k.strip()] = _v.strip()
    return managed_launch_script(instance, server_args(instance, plan), environment)


def frontend_args(service: dict[str, Any], plan: dict[str, Any]) -> list[str]:
    return [
        plan["python"],
        "-m",
        "rtp_llm.start_server",
        "--checkpoint_path",
        MODEL_PATH,
        "--tokenizer_path",
        MODEL_PATH,
        "--model_type",
        MODEL_TYPE,
        "--role_type",
        "FRONTEND",
        "--start_port",
        str(service["start_port"]),
        "--use_local",
        "1",
        "--frontend_server_count",
        "1",
        "--concurrency_limit",
        str(FRONTEND_CONCURRENCY),
        "--max_seq_len",
        str(MAX_SEQ_LEN),
        "--seq_size_per_block",
        "64",
        "--tp_size",
        "1",
        "--ep_size",
        "1",
        "--dp_size",
        "1",
        "--world_size",
        "1",
        "--world_rank",
        "0",
        "--local_world_size",
        "1",
    ]


def service_launch_script(service: dict[str, Any], plan: dict[str, Any]) -> str:
    if service["name"] == "frontend":
        environment = {
            "CUDA_VISIBLE_DEVICES": "",
            "LD_LIBRARY_PATH": "/opt/conda310/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib64:/usr/lib64",
            "MODEL_SERVICE_CONFIG": frontend_service_config(plan),
            "ROLE_TYPE": "FRONTEND",
            "USE_LOCAL": "1",
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "LOG_LEVEL": "INFO",
            "OMP_NUM_THREADS": "4",
        }
        if plan["cache_store_rdma_mode"]:
            environment["PYTHONPATH"] = plan["pythonpath"]
        for _pair in os.environ.get("RTP_WORKER_EXTRA_ENV", "").split(","):
            if "=" in _pair:
                _k, _v = _pair.split("=", 1)
                environment[_k.strip()] = _v.strip()
        return managed_launch_script(service, frontend_args(service, plan), environment)

    environment = {
        "MODEL_SERVICE_CONFIG": flexlb_service_config(),
        "DOMAIN_ADDRESS:rtp-hol-prefill": role_address(plan, "PREFILL"),
        "DOMAIN_ADDRESS:rtp-hol-decode": decode_worker_address(plan),
        "FLEXLB_CONFIG": json.dumps(flexlb_runtime_config(), separators=(",", ":")),
        "FLEXLB_SYNC_CONSISTENCY_CONFIG": json.dumps(
            {"needConsistency": False}, separators=(",", ":")
        ),
        "HIPPO_ROLE": "rtp-hol-flexlb",
        "OTEL_TRACE_SKIP_PATTERN": ".*",
        "OTEL_EXPORTER_OTLP_ENDPOINT": "none",
        "SYNC_STATUS_INTERVAL": "500",
        "SYNC_REQUEST_TIMEOUT_MS": "400",
        "JAVA_TOOL_OPTIONS": "-Xms1g -Xmx4g",
    }
    # v32 block-aware decode scheduling knobs ride through to the FlexLB JVM.
    for _key in (
        "FLEXLB_DECODE_OFFLOAD_RESIDENT_TOKENS",
        "FLEXLB_DECODE_OFFLOAD_MIN_SEQ",
    ):
        if os.environ.get(_key):
            environment[_key] = os.environ[_key]
    command = [
        FLEXLB_JAVA,
        f"-Dserver.port={service['start_port']}",
        "-jar",
        FLEXLB_JAR,
        f"--server.port={service['start_port']}",
        f"--management.server.port={service['health_port']}",
        "--spring.profiles.active=test",
    ]
    return managed_launch_script(service, command, environment)


def preflight(plan: dict[str, Any]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for resource in [*plan["instances"], *plan.get("services", [])]:
        grouped[resource["host"]].append(resource)

    for host_name, resources in grouped.items():
        host = HOST_BY_NAME[host_name]
        managed_process_check = managed_helper_command(
            {
                "action": "assert_no_managed",
                "required_env": ["RTP_CLUSTER_MANAGED=1"],
            }
        )
        port_checks = "\n".join(
            f"for port in $(seq {min(resource_port_range(item, item['start_port']))} "
            f"{max(resource_port_range(item, item['start_port']))}); do "
            'listeners=$(ss -ltnH "sport = :$port") || '
            "{ echo 'ss port query failed:' $port; exit 56; }; "
            'if [ -n "$listeners" ]; then '
            f"echo 'port busy for {item['name']}:' $port; exit 24; fi; done"
            for item in resources
        )
        pid_checks = "\n".join(
            f"pid_file={shlex.quote(item['remote_dir'])}/server.pid; "
            'if [ -s "$pid_file" ] && kill -0 "$(cat "$pid_file")" 2>/dev/null; '
            f"then echo 'live pid file for {item['name']}'; exit 25; fi"
            for item in resources
        )
        flexlb_checks = ""
        if any(item["name"] == "flexlb" for item in resources):
            flexlb_checks = (
                f"test -x {shlex.quote(FLEXLB_JAVA)}\n"
                f"test -f {shlex.quote(FLEXLB_JAR)}\n"
                f"{shlex.quote(FLEXLB_JAVA)} -version >/dev/null 2>&1"
            )
        rdma_checks = ""
        if plan["cache_store_rdma_mode"]:
            rdma_nic = shlex.quote(host["rdma_nic"])
            runtime_path = shlex.quote(plan["runtime_path"])
            pythonpath = shlex.quote(plan["pythonpath"])
            verify_runtime_code = (
                "import hashlib, pathlib, rtp_llm; "
                f"root=pathlib.Path({plan['pythonpath']!r}).resolve(); "
                "module=pathlib.Path(rtp_llm.__file__).resolve(); "
                "assert root in module.parents, (root, module); "
                'so=root / "rtp_llm/libs/libth_transformer.so"; '
                f"manifest=pathlib.Path({plan['runtime_path']!r}, "
                '"libth_transformer.sha256"); '
                "expected=manifest.read_text().split()[0]; "
                "actual=hashlib.sha256(so.read_bytes()).hexdigest(); "
                "assert actual == expected, (actual, expected); "
                'print(f"verified RDMA runtime: {module} {actual}")'
            )
            rdma_checks = f"""test -d /sys/class/infiniband/{rdma_nic}
test -e /dev/infiniband/rdma_cm
compgen -G '/dev/infiniband/uverbs*' >/dev/null
grep -q 'ACTIVE' /sys/class/infiniband/{rdma_nic}/ports/1/state
grep -qx 'RoCE v2' /sys/class/infiniband/{rdma_nic}/ports/1/gid_attrs/types/3
grep -qx 'eth1' /sys/class/infiniband/{rdma_nic}/ports/1/gid_attrs/ndevs/3
command -v ibv_devinfo >/dev/null
ibv_devinfo -d {rdma_nic} -i 1 >/dev/null
command -v gdrcopy_sanity >/dev/null
timeout 60 gdrcopy_sanity 2>&1 | grep -q 'Failed: 0'
test -d {pythonpath}/rtp_llm
test -s {runtime_path}/libth_transformer.sha256
env PYTHONPATH={pythonpath} {shlex.quote(plan["python"])} -c {shlex.quote(verify_runtime_code)}
"""
        source_path = shlex.quote(plan["source_path"])
        wheel_check = ""
        if not plan["cache_store_rdma_mode"]:
            wheel_check = (
                f"test -f {source_path}/bazel-bin/rtp_llm/"
                "rtp_llm-0.2.0-cp310-cp310-manylinux1_x86_64.whl"
            )
        script = f"""set -e
export LD_LIBRARY_PATH=/opt/conda310/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib64:/usr/lib64
command -v ss >/dev/null 2>&1 || {{ echo 'ss is required for preflight' >&2; exit 56; }}
test -d {shlex.quote(MODEL_PATH)}
{wheel_check}
env {('PYTHONPATH=' + shlex.quote(plan['pythonpath'])) if plan['pythonpath'] else ''} {shlex.quote(plan['python'])} -c 'import torch, rtp_llm; from accelerate.utils.memory import clear_device_cache; from sentence_transformers.models import Normalize, Transformer; from rtp_llm.models import BaseModel; assert torch.cuda.is_available()'
{flexlb_checks}
{rdma_checks}
{managed_process_check}
gpu_processes=$(nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader) || {{
  echo 'nvidia-smi compute-process query failed' >&2
  exit 55
}}
if [ -n "$gpu_processes" ]; then
  echo 'GPU compute processes already exist on this host'
  printf '%s\n' "$gpu_processes"
  exit 26
fi
{port_checks}
{pid_checks}
"""
        print(f"preflight {host_name} ...", flush=True)
        ssh_run(host, script, timeout=90)


def process_status(resource: dict[str, Any]) -> tuple[bool, bool, str]:
    host = HOST_BY_NAME[resource["host"]]
    if resource.get("managed_token"):
        result = ssh_run(
            host,
            managed_helper_command(managed_payload(resource, "status")),
            timeout=15,
            check=False,
        )
        output = result.stdout.strip()
        tokens = output.split()
        process_state, health_state = tokens[-2:] if len(tokens) >= 2 else ("", "")
        return (
            result.returncode == 0 and process_state == "alive",
            result.returncode == 0 and health_state == "healthy",
            output,
        )

    remote_dir = shlex.quote(resource["remote_dir"])
    port = int(resource.get("health_port", resource["start_port"]))
    script = f"""pid_file={remote_dir}/server.pid
if [ -s \"$pid_file\" ] && kill -0 \"$(cat \"$pid_file\")\" 2>/dev/null; then
  printf 'alive '
else
  printf 'dead '
fi
if curl -fsS --max-time 3 http://127.0.0.1:{port}/health >/dev/null 2>&1; then
  printf 'healthy'
else
  printf 'unhealthy'
fi
"""
    result = ssh_run(host, script, timeout=15, check=False)
    output = result.stdout.strip()
    tokens = output.split()
    process_state, health_state = tokens[-2:] if len(tokens) >= 2 else ("", "")
    return process_state == "alive", health_state == "healthy", output


def tail_log(resource: dict[str, Any], lines: int = 50) -> str:
    host = HOST_BY_NAME[resource["host"]]
    log_path = shlex.quote(f"{resource['remote_dir']}/server.log")
    result = ssh_run(
        host,
        f"tail -n {int(lines)} {log_path} 2>&1",
        timeout=30,
        check=False,
    )
    return result.stdout.rstrip()


def stop_instance(resource: dict[str, Any], timeout: int) -> None:
    host = HOST_BY_NAME[resource["host"]]
    if resource.get("managed_token"):
        result = ssh_run(
            host,
            managed_helper_command(
                managed_payload(resource, "stop", timeout=max(1, timeout))
            ),
            timeout=max(1, timeout) + 20,
            check=False,
        )
        print(f"{resource['name']}: {result.stdout.strip()}")
        if result.returncode != 0:
            raise ClusterError(
                f"failed to stop token-managed {resource['name']} on "
                f"{resource['host']} (rc={result.returncode})"
            )
        return

    remote_dir = shlex.quote(resource["remote_dir"])
    start_port = int(resource["start_port"])
    legacy_ports = " ".join(
        str(port) for port in sorted(resource_port_range(resource, start_port))
    )
    script = f"""set -euo pipefail
command -v ss >/dev/null 2>&1 || {{ echo 'ss is required for legacy stop diagnostics' >&2; exit 54; }}
pid_file={remote_dir}/server.pid
managed_ports={shlex.quote(legacy_ports)}
open_ports=''
for port in $managed_ports; do
  listeners=$(ss -ltnH "sport = :$port") || {{
    echo "failed to inspect managed port: $port" >&2
    exit 54
  }}
  if [ -n "$listeners" ]; then
    open_ports="$open_ports $port"
  fi
done
pid='-'
group_pids=''
if [ -s \"$pid_file\" ]; then
  pid=$(cat \"$pid_file\")
  case \"$pid\" in
    ''|*[!0-9]*)
      echo 'legacy state has an invalid pid file; refusing automatic stop' >&2
      exit 57
      ;;
  esac
  group_pids=$(ps -eo pid=,pgid= | awk -v target=\"$pid\" '$2 == target {{print $1}}')
fi
if [ -z \"$group_pids\" ] && [ -z \"$open_ports\" ]; then
  echo 'legacy resource is not observably running'
  exit 0
fi
echo \"legacy state has no immutable process identity; refusing automatic signal; pid=$pid group_pids=$group_pids open_ports=$open_ports. Use a pinned PID/start-time/port cleanup\" >&2
exit 57
"""
    result = ssh_run(host, script, timeout=30, check=False)
    print(f"{resource['name']}: {result.stdout.strip()}")
    if result.returncode != 0:
        raise ClusterError(
            f"refused or failed to stop {resource['name']} on {resource['host']} "
            f"(rc={result.returncode})"
        )


def verify_cluster_stopped(plan: dict[str, Any], timeout: int) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for resource in [*plan.get("instances", []), *plan.get("services", [])]:
        grouped[resource["host"]].append(resource)

    def verify_host(host_name: str, resources: list[dict[str, Any]]) -> str:
        ports = sorted(
            {
                port
                for resource in resources
                for port in resource_port_range(resource, int(resource["start_port"]))
            }
        )
        payload = {
            "action": "verify_host",
            "required_env": ["RTP_CLUSTER_MANAGED=1"],
            "ports": ports,
            "require_gpu_clear": any(resource.get("gpus") for resource in resources),
            "launch_identities": [
                {
                    "name": resource.get("name", "<unknown>"),
                    "identity": resource.get("launch_identity"),
                }
                for resource in resources
                if resource.get("launch_identity") is not None
            ],
            "unproven_resources": [
                {
                    "name": resource.get("name", "<unknown>"),
                    "pid": resource.get("pid"),
                    "pid_file": f"{resource['remote_dir']}/server.pid",
                }
                for resource in resources
                if resource.get("launch_identity") is None
            ],
            "timeout": max(1, timeout),
        }
        result = ssh_run(
            HOST_BY_NAME[host_name],
            managed_helper_command(payload),
            timeout=max(1, timeout) + 20,
            check=False,
        )
        if result.returncode != 0:
            raise ClusterError(
                f"stop postcondition failed on {host_name} "
                f"(rc={result.returncode}):\n{result.stdout.rstrip()}"
            )
        return result.stdout.strip()

    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=min(len(grouped), 4)) as executor:
        futures = {
            executor.submit(verify_host, host_name, resources): host_name
            for host_name, resources in grouped.items()
        }
        for future in as_completed(futures):
            host_name = futures[future]
            try:
                detail = future.result()
                print(f"{host_name}: {detail}")
            except BaseException as error:
                errors.append(str(error))
    if errors:
        raise ClusterError("; ".join(errors))


def wait_healthy(resources: list[dict[str, Any]], deadline: float) -> None:
    previous: dict[str, str] = {}
    while time.monotonic() < deadline:
        statuses: dict[str, tuple[bool, bool, str]] = {}
        with ThreadPoolExecutor(max_workers=min(16, len(resources))) as executor:
            futures = {
                executor.submit(process_status, resource): resource
                for resource in resources
            }
            for future in as_completed(futures):
                resource = futures[future]
                statuses[resource["name"]] = future.result()

        all_healthy = True
        for resource in resources:
            alive, healthy, _ = statuses[resource["name"]]
            state = "healthy" if healthy else ("loading" if alive else "dead")
            if previous.get(resource["name"]) != state:
                print(f"{resource['name']}: {state}", flush=True)
                previous[resource["name"]] = state
            if not alive:
                raise ClusterError(
                    f"{resource['name']} exited during startup:\n{tail_log(resource)}"
                )
            all_healthy = all_healthy and healthy
        if all_healthy:
            return
        time.sleep(5)
    names = ", ".join(resource["name"] for resource in resources)
    raise ClusterError(f"resources did not become healthy before timeout: {names}")


def flexlb_schedule_probe(plan: dict[str, Any]) -> tuple[bool, str]:
    flexlb = next(item for item in plan["services"] if item["name"] == "flexlb")
    host = HOST_BY_NAME[flexlb["host"]]
    payload = json.dumps(
        {
            "model": "engine_service",
            "block_cache_keys": [],
            "seq_len": 1,
            "request_id": int(time.time_ns() % 9_000_000_000) + 1,
            "generate_timeout": 10000,
            "request_time_ms": int(time.time() * 1000),
        },
        separators=(",", ":"),
    )
    command = (
        f"curl -fsS --max-time 3 -X POST "
        f"http://127.0.0.1:{flexlb['start_port']}/rtp_llm/schedule "
        "-H 'Content-Type: application/json' "
        f"--data {shlex.quote(payload)}"
    )
    result = ssh_run(host, command, timeout=10, check=False)
    if result.returncode != 0:
        return False, result.stdout.strip()
    json_lines = [
        line.strip()
        for line in result.stdout.splitlines()
        if line.lstrip().startswith("{")
    ]
    if not json_lines:
        return False, result.stdout.strip()
    try:
        response = json.loads(json_lines[-1])
    except json.JSONDecodeError:
        return False, result.stdout.strip()
    statuses = response.get("server_status") or []
    roles = {status.get("role") for status in statuses}
    selected = {
        f"{status.get('server_ip')}:{status.get('http_port')}" for status in statuses
    }
    expected_prefills = set(role_address(plan, "PREFILL").split(","))
    expected_decodes = set(role_address(plan, "DECODE").split(","))
    valid = (
        response.get("code", 200) == 200
        and {"PREFILL", "DECODE"}.issubset(roles)
        and bool(selected & expected_prefills)
        and bool(selected & expected_decodes)
    )
    return valid, json.dumps(response, separators=(",", ":"))


def wait_flexlb_routes(plan: dict[str, Any], deadline: float) -> str:
    last_detail = ""
    while time.monotonic() < deadline:
        alive, healthy, _ = process_status(
            next(item for item in plan["services"] if item["name"] == "flexlb")
        )
        if not alive:
            flexlb = next(item for item in plan["services"] if item["name"] == "flexlb")
            raise ClusterError(f"FlexLB exited:\n{tail_log(flexlb)}")
        if healthy:
            ready, last_detail = flexlb_schedule_probe(plan)
            if ready:
                return last_detail
        time.sleep(2)
    raise ClusterError(f"FlexLB did not discover P/D workers: {last_detail}")


def write_state(plan: dict[str, Any]) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    temporary = STATE_FILE.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    temporary.replace(STATE_FILE)


def read_state() -> dict[str, Any]:
    if not STATE_FILE.exists():
        raise ClusterError(f"state file does not exist: {STATE_FILE}")
    try:
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as error:
        raise ClusterError(
            f"cannot read cluster state {STATE_FILE}: {error}"
        ) from error
    if not isinstance(state, dict):
        raise ClusterError("cluster state is not a JSON object")
    version = state.get("state_schema_version", 1)
    if type(version) is not int or not 1 <= version <= STATE_SCHEMA_VERSION:
        raise ClusterError(
            f"unsupported cluster state schema {version!r}; "
            f"this binary supports versions 1..{STATE_SCHEMA_VERSION}"
        )
    return state


def print_plan(plan: dict[str, Any]) -> None:
    print(
        f"P:D GPU ratio requested: {plan['ratio']} | "
        f"cards={plan['prefill_gpu_count']}:{plan['decode_gpu_count']} | "
        f"instances={plan['prefill_instance_count']}:{plan['decode_instance_count']} | "
        f"max_seq_len={plan['max_seq_len']} | "
        f"frontend_concurrency={plan['frontend_concurrency']} | "
        f"inductor_compile_threads={plan['inductor_compile_threads']} | "
        f"reuse_cache=P:{'on' if plan['prefill_reuse_cache'] else 'off'}"
        f"/D:{'on' if plan['decode_reuse_cache'] else 'off'} | "
        f"cache_transport={plan['cache_transport'].upper()} | "
        f"decode_routing={plan['decode_routing_scheme']}"
        f"(beta={plan['decode_cache_load_beta']:g})"
    )
    print(f"source: {plan['source_path']}")
    print(
        f"{'INSTANCE':<12} {'ROLE':<8} {'HOST':<22} {'GPUS':<8} "
        f"{'PORT':<6} {'CACHE':<23} {'RDMA_NIC':<10} {'PEER_HINT':<12}"
    )
    for instance in plan["instances"]:
        gpus = ",".join(str(gpu) for gpu in instance["gpus"])
        tcp_ports = ",".join(str(port) for port in instance_rank_ports(instance, 2))
        rdma_ports = (
            ",".join(str(port) for port in instance_rank_ports(instance, 4))
            if plan["cache_store_rdma_mode"]
            else "-"
        )
        cache_ports = f"{tcp_ports}/{rdma_ports}"
        rdma_nic = instance["rdma_nic"] if plan["cache_store_rdma_mode"] else "-"
        print(
            f"{instance['name']:<12} {instance['role']:<8} "
            f"{instance['host']:<22} {gpus:<8} {instance['start_port']:<6} "
            f"{cache_ports:<23} {rdma_nic:<10} {instance['peer_name']:<12}"
        )
    print("cache ports: TCP control/data (+2) / RDMA data (+4), listed per TP rank")
    mixed_hosts = [
        host_name
        for host_name, counts in plan["host_role_gpu_counts"].items()
        if counts["PREFILL"] and counts["DECODE"]
    ]
    print(
        "placement: dedicated-role hosts first; mixed hosts: "
        + (", ".join(mixed_hosts) if mixed_hosts else "none")
    )
    print("routing: global Frontend -> FlexLB -> one fixed P/D pair per request")
    print("CONTROL      ROLE     HOST                   PORT   HEALTH_PORT")
    for service in plan.get("services", []):
        print(
            f"{service['name']:<12} {service['role']:<8} "
            f"{service['host']:<22} {service['start_port']:<6} "
            f"{service['health_port']}"
        )


def command_start(args: argparse.Namespace) -> int:
    if STATE_FILE.exists():
        old_state = read_state()
        if old_state.get("status") in {"starting", "running"}:
            raise ClusterError(
                f"cluster state is {old_state['status']}; stop it before starting again"
            )
        try:
            verify_cluster_stopped(old_state, timeout=5)
        except ClusterError as error:
            raise ClusterError(
                f"previous cluster is not verifiably stopped: {error}"
            ) from error

    plan = make_plan(args.ratio, args.cache_store_rdma_mode)
    resolve_ports(plan)
    print_plan(plan)
    if args.dry_run:
        return 0

    preflight(plan)
    plan["status"] = "starting"
    write_state(plan)

    try:
        for role in ("DECODE", "PREFILL"):
            by_host: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for instance in plan["instances"]:
                if instance["role"] == role:
                    by_host[instance["host"]].append(instance)

            wave_count = max(len(items) for items in by_host.values())
            for wave_index in range(wave_count):
                wave = [
                    items[wave_index]
                    for items in by_host.values()
                    if wave_index < len(items)
                ]
                for instance in wave:
                    host = HOST_BY_NAME[instance["host"]]
                    print(
                        f"launch {instance['name']} on {instance['host']} "
                        f"GPU {instance['gpus']} ...",
                        flush=True,
                    )
                    result = ssh_run(host, launch_script(instance, plan), timeout=60)
                    record_managed_launch_result(instance, result)
                    write_state(plan)

                names = ", ".join(item["name"] for item in wave)
                print(f"waiting for startup wave: {names} ...", flush=True)
                wait_healthy(wave, time.monotonic() + args.start_timeout)
                time.sleep(2)

        deadline = time.monotonic() + args.start_timeout

        flexlb = next(item for item in plan["services"] if item["name"] == "flexlb")
        flexlb_host = HOST_BY_NAME[flexlb["host"]]
        print(f"launch FlexLB on {flexlb['host']} ...", flush=True)
        result = ssh_run(flexlb_host, service_launch_script(flexlb, plan), timeout=60)
        record_managed_launch_result(flexlb, result)
        write_state(plan)
        wait_healthy([flexlb], deadline)
        route_detail = wait_flexlb_routes(plan, deadline)
        print(f"FlexLB route probe: {route_detail}", flush=True)

        frontend = next(item for item in plan["services"] if item["name"] == "frontend")
        frontend_host = HOST_BY_NAME[frontend["host"]]
        print(f"launch global Frontend on {frontend['host']} ...", flush=True)
        result = ssh_run(
            frontend_host, service_launch_script(frontend, plan), timeout=60
        )
        record_managed_launch_result(frontend, result)
        write_state(plan)
        wait_healthy([frontend], deadline)

        plan["status"] = "running"
        plan["started_at"] = datetime.now(timezone.utc).isoformat()
        write_state(plan)
        print("cluster is healthy")
        print("request endpoint: " f"http://{frontend['ip']}:{frontend['start_port']}")
        return 0
    except BaseException:
        print("startup failed; stopping launched instances", file=sys.stderr)
        cleanup_errors: list[str] = []
        cleanup_resources = [*plan.get("instances", []), *plan.get("services", [])]
        for instance in reversed(cleanup_resources):
            try:
                stop_instance(instance, timeout=10)
            except BaseException as error:
                cleanup_errors.append(f"{instance['name']}: {error}")
                print(
                    f"startup cleanup failed for {instance['name']}: {error}",
                    file=sys.stderr,
                )
        try:
            verify_cluster_stopped(plan, timeout=60)
        except BaseException as error:
            cleanup_errors.append(f"postcondition: {error}")
            print(f"startup cleanup postcondition failed: {error}", file=sys.stderr)
        plan["status"] = (
            "failed_clean" if not cleanup_errors else "startup_cleanup_partial"
        )
        plan["startup_cleanup_errors"] = cleanup_errors
        write_state(plan)
        raise


def command_status(_: argparse.Namespace) -> int:
    plan = read_state()
    print(
        f"cluster state: {plan.get('status', 'unknown')} "
        f"GPU ratio={plan.get('ratio')} "
        f"cards={plan.get('prefill_gpu_count', '?')}:"
        f"{plan.get('decode_gpu_count', '?')}"
    )
    print(
        f"{'INSTANCE':<12} {'ROLE':<8} {'HOST':<22} "
        f"{'PORT':<6} {'PID':<8} {'HEALTH':<10}"
    )
    all_healthy = True
    resources = [*plan.get("services", []), *plan["instances"]]
    for resource in resources:
        alive, healthy, _ = process_status(resource)
        health = "healthy" if healthy else ("loading" if alive else "dead")
        print(
            f"{resource['name']:<12} {resource['role']:<8} "
            f"{resource['host']:<22} {resource['start_port']:<6} "
            f"{str(resource.get('pid', '-')):<8} {health:<10}"
        )
        all_healthy = all_healthy and healthy
    return 0 if all_healthy else 1


def command_stop(args: argparse.Namespace) -> int:
    plan = read_state()
    services = plan.get("services", [])
    stages = [
        [item for item in services if item["name"] == "frontend"],
        [item for item in services if item["name"] == "flexlb"],
        [item for item in plan["instances"] if item["role"] == "PREFILL"],
        [item for item in plan["instances"] if item["role"] == "DECODE"],
    ]
    resource_stop_errors: list[str] = []
    unverifiable_resource_errors: list[str] = []
    for resources in stages:
        if not resources:
            continue
        with ThreadPoolExecutor(max_workers=min(16, len(resources))) as executor:
            futures = {
                executor.submit(stop_instance, resource, args.stop_timeout): resource
                for resource in resources
            }
            for future, resource in futures.items():
                try:
                    future.result()
                except BaseException as error:
                    message = f"{resource['name']}: {error}"
                    resource_stop_errors.append(message)
                    if not resource.get("managed_token"):
                        unverifiable_resource_errors.append(message)
                    print(f"stop failed for {message}", file=sys.stderr)
    postcondition_error: str | None = None
    try:
        verify_cluster_stopped(plan, timeout=max(60, args.stop_timeout))
    except BaseException as error:
        postcondition_error = f"postcondition: {error}"
        print(f"stop failed for {postcondition_error}", file=sys.stderr)
    if postcondition_error is None and unverifiable_resource_errors:
        postcondition_error = (
            "postcondition: legacy resources without immutable identity reported "
            "stop errors; host-wide checks cannot prove those PID groups belong "
            "to, or are absent from, this cluster"
        )
        print(f"stop failed for {postcondition_error}", file=sys.stderr)

    stop_errors = [*resource_stop_errors]
    if postcondition_error is not None:
        stop_errors.append(postcondition_error)
    plan["status"] = "stopped" if postcondition_error is None else "stop_partial"
    plan["stopped_at"] = datetime.now(timezone.utc).isoformat()
    plan["stop_errors"] = stop_errors if postcondition_error is not None else []
    plan["stop_warnings"] = resource_stop_errors if postcondition_error is None else []
    write_state(plan)
    if postcondition_error is not None:
        raise ClusterError(
            f"cluster stop was partial; {len(stop_errors)} resource(s) failed"
        )
    if resource_stop_errors:
        print(
            "warning: "
            f"{len(resource_stop_errors)} resource stop operation(s) reported errors, "
            "but all host stop postconditions passed",
            file=sys.stderr,
        )
    return 0


def command_restart_frontend(args: argparse.Namespace) -> int:
    plan = read_state()
    if plan.get("status") != "running":
        raise ClusterError(f"cluster state is {plan.get('status')}; expected running")
    frontend = next(
        item for item in plan.get("services", []) if item["name"] == "frontend"
    )
    plan["frontend_concurrency"] = FRONTEND_CONCURRENCY
    host = HOST_BY_NAME[frontend["host"]]
    plan["status"] = "restarting_frontend"
    write_state(plan)
    try:
        stop_instance(frontend, timeout=args.stop_timeout)
        frontend.pop("pid", None)
        frontend.pop("launch_identity", None)
        write_state(plan)
        result = ssh_run(host, service_launch_script(frontend, plan), timeout=60)
        record_managed_launch_result(frontend, result)
        write_state(plan)
        wait_healthy([frontend], time.monotonic() + args.start_timeout)
    except BaseException:
        cleanup_errors: list[str] = []
        try:
            stop_instance(frontend, timeout=max(10, args.stop_timeout))
        except BaseException as error:
            cleanup_errors.append(str(error))
        plan["status"] = (
            "restart_failed_clean" if not cleanup_errors else "restart_cleanup_partial"
        )
        plan["restart_cleanup_errors"] = cleanup_errors
        write_state(plan)
        raise
    plan["status"] = "running"
    plan["restart_cleanup_errors"] = []
    write_state(plan)
    print(f"frontend is healthy: http://{frontend['ip']}:{frontend['start_port']}")
    return 0


def command_restart_flexlb(args: argparse.Namespace) -> int:
    plan = read_state()
    if plan.get("status") != "running":
        raise ClusterError(f"cluster state is {plan.get('status')}; expected running")
    flexlb = next(item for item in plan.get("services", []) if item["name"] == "flexlb")
    plan["flexlb_runtime_config"] = flexlb_runtime_config()
    host = HOST_BY_NAME[flexlb["host"]]
    plan["status"] = "restarting_flexlb"
    write_state(plan)
    try:
        stop_instance(flexlb, timeout=args.stop_timeout)
        flexlb.pop("pid", None)
        flexlb.pop("launch_identity", None)
        write_state(plan)
        result = ssh_run(host, service_launch_script(flexlb, plan), timeout=60)
        record_managed_launch_result(flexlb, result)
        write_state(plan)
        deadline = time.monotonic() + args.start_timeout
        wait_healthy([flexlb], deadline)
        route_detail = wait_flexlb_routes(plan, deadline)
    except BaseException:
        cleanup_errors: list[str] = []
        try:
            stop_instance(flexlb, timeout=max(10, args.stop_timeout))
        except BaseException as error:
            cleanup_errors.append(str(error))
        plan["status"] = (
            "restart_failed_clean" if not cleanup_errors else "restart_cleanup_partial"
        )
        plan["restart_cleanup_errors"] = cleanup_errors
        write_state(plan)
        raise
    plan["status"] = "running"
    plan["restart_cleanup_errors"] = []
    write_state(plan)
    print(f"flexlb is healthy; route probe: {route_detail}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("plan", "start"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument(
            "--ratio",
            type=parse_ratio,
            default=parse_ratio(os.environ.get("PD_RATIO", "1:1")),
            metavar="P:D",
            help="Prefill:Decode GPU-card ratio over all 32 GPUs",
        )
        transport = subparser.add_mutually_exclusive_group()
        transport.add_argument(
            "--rdma",
            dest="cache_store_rdma_mode",
            action="store_true",
            help="use the Barex RDMA CacheStore build and per-host RDMA NIC",
        )
        transport.add_argument(
            "--tcp",
            dest="cache_store_rdma_mode",
            action="store_false",
            help="use TCP CacheStore (the default)",
        )
        subparser.set_defaults(cache_store_rdma_mode=CACHE_STORE_RDMA_MODE)
        if name == "start":
            subparser.add_argument(
                "--start-timeout",
                type=int,
                default=int(os.environ.get("RTP_START_TIMEOUT", "1200")),
            )
            subparser.add_argument("--dry-run", action="store_true")

    stop_parser = subparsers.add_parser("stop")
    stop_parser.add_argument("--stop-timeout", type=int, default=30)
    restart_frontend_parser = subparsers.add_parser("restart-frontend")
    restart_frontend_parser.add_argument("--stop-timeout", type=int, default=5)
    restart_frontend_parser.add_argument("--start-timeout", type=int, default=600)
    restart_flexlb_parser = subparsers.add_parser("restart-flexlb")
    restart_flexlb_parser.add_argument("--stop-timeout", type=int, default=5)
    restart_flexlb_parser.add_argument("--start-timeout", type=int, default=600)
    subparsers.add_parser("status")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.command == "plan":
            print_plan(make_plan(args.ratio, args.cache_store_rdma_mode))
            return 0
        if args.command == "start":
            return command_start(args)
        if args.command == "status":
            return command_status(args)
        if args.command == "stop":
            return command_stop(args)
        if args.command == "restart-frontend":
            return command_restart_frontend(args)
        if args.command == "restart-flexlb":
            return command_restart_flexlb(args)
        parser.error(f"unsupported command: {args.command}")
    except (ClusterError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
