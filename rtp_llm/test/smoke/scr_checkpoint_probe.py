import ctypes
import ipaddress
import json
import os
import re
import select
import socket
import struct
import tempfile
import time
from pathlib import Path

import psutil


class PinnedProbe:
    def __init__(self):
        library = Path(__file__).absolute().with_name("libscr_pinned_probe.so")
        self.library = ctypes.CDLL(str(library))
        self.library.scrPinnedProbeInit.argtypes = [ctypes.c_char_p]
        self.library.scrPinnedProbeInit.restype = ctypes.c_int
        self.library.scrPinnedProbeSnapshot.restype = ctypes.c_char_p
        cupti_dir = (
            Path(__file__).absolute().parents[4] / "local_config_cuda/cuda/cuda/lib"
        )
        candidates = sorted(cupti_dir.glob("libcupti.so*"))
        if not candidates:
            raise RuntimeError(f"CUPTI runfile missing: {cupti_dir}")
        result = self.library.scrPinnedProbeInit(os.fsencode(candidates[0]))
        if result != 0:
            raise RuntimeError(f"CUPTI probe initialization failed: {result}")

    def snapshot(self):
        result = json.loads(self.library.scrPinnedProbeSnapshot())
        if not result["initialized"] or result["errors"]:
            raise RuntimeError(f"Pinned observation failed: {result}")
        return result


def write_message(stream, message):
    stream.write(json.dumps(message).encode() + b"\n")
    stream.flush()


def read_message(stream):
    line = stream.readline(1024 * 1024)
    if not line.endswith(b"\n"):
        raise RuntimeError("Checkpoint participant disconnected or message oversized")
    return json.loads(line)


class BlockingTestEpsilon:
    API_VERSION = "smoke-test-barrier-not-native-scr"
    __file__ = __file__

    def __init__(self, probe):
        self.probe = probe
        self.tensors = []
        self.callbacks = []

    def is_snapstart_enable(self):
        return True

    def register_kv_caches(self, tensors):
        self.tensors = list(tensors)
        return 0

    def register_before_checkpoint_func(self, callback):
        self.callbacks.append(callback)
        return 0

    def snapshot(self):
        return {
            "pid": os.getpid(),
            "pinned": self.probe.snapshot(),
            "gpu_kv_bytes": sum(int(tensor.nbytes) for tensor in self.tensors),
        }

    def snapstart_checkpoint(
        self, *, wait_mode, worker_id, worker_num, timeout, inactivity_timeout
    ):
        if wait_mode != 1:
            raise RuntimeError("Smoke requires a blocking checkpoint arrival")
        for callback in self.callbacks:
            callback()
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(timeout)
            connection.connect(os.environ["SCR_TEST_BARRIER_SOCKET"])
            with connection.makefile("rwb") as stream:
                write_message(
                    stream,
                    {
                        **self.snapshot(),
                        "worker_id": worker_id,
                        "worker_num": worker_num,
                        "generation": os.environ["RTP_LLM_SCR_GENERATION"],
                    },
                )
                while True:
                    command = read_message(stream)
                    if command == "snapshot":
                        write_message(stream, self.snapshot())
                    elif command == "release":
                        return 0
                    else:
                        raise RuntimeError(f"Invalid checkpoint command: {command}")


def read_socket_message(connection, deadline):
    data = bytearray()
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Checkpoint message deadline expired")
        readable, _, _ = select.select([connection], [], [], remaining)
        if not readable:
            raise TimeoutError("Checkpoint message deadline expired")
        chunk = connection.recv(4096)
        if not chunk:
            raise RuntimeError("Checkpoint participant disconnected")
        data.extend(chunk)
        if len(data) > 1024 * 1024:
            raise RuntimeError("Checkpoint message oversized")
        if time.monotonic() >= deadline:
            raise TimeoutError("Checkpoint message deadline expired")
        if b"\n" in data:
            if not data.endswith(b"\n") or data.count(b"\n") != 1:
                raise RuntimeError("Invalid checkpoint message framing")
            return json.loads(data)


def write_socket_message(connection, message, deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Checkpoint message deadline expired")
    connection.settimeout(remaining)
    connection.sendall(json.dumps(message).encode() + b"\n")
    if time.monotonic() >= deadline:
        raise TimeoutError("Checkpoint message deadline expired")


class CheckpointBarrier:
    def __init__(self, generation):
        self.generation = generation
        self.directory = tempfile.TemporaryDirectory(prefix="scr-smoke-", dir="/tmp")
        self.path = str(Path(self.directory.name) / "barrier.sock")
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.listener.bind(self.path)
        self.listener.listen(128)
        self.participants = {}
        self.worker_num = None

    def wait_ready(self, root_pid, timeout):
        deadline = time.monotonic() + timeout
        while self.worker_num is None or len(self.participants) < self.worker_num:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Missing checkpoint participants: {self.participants.keys()}"
                )
            readable, _, _ = select.select([self.listener], [], [], min(remaining, 1))
            root = psutil.Process(root_pid)
            if not root.is_running() or root.status() == psutil.STATUS_ZOMBIE:
                raise RuntimeError("Server exited before checkpoint quorum")
            if not readable:
                continue
            connection, _ = self.listener.accept()
            try:
                arrival = read_socket_message(connection, deadline)
                peer_pid, _, _ = struct.unpack(
                    "3i",
                    connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12),
                )
                worker_id = arrival["worker_id"]
                worker_num = arrival["worker_num"]
                tree = {root_pid, *(p.pid for p in root.children(recursive=True))}
                if (
                    arrival["generation"] != self.generation
                    or arrival["pid"] != peer_pid
                    or peer_pid not in tree
                    or type(worker_id) is not int
                    or type(worker_num) is not int
                    or not 0 <= worker_id < worker_num
                    or worker_id in self.participants
                    or (self.worker_num is not None and self.worker_num != worker_num)
                    or any(p[1]["pid"] == peer_pid for p in self.participants.values())
                ):
                    raise RuntimeError(f"Invalid checkpoint membership: {arrival}")
                self.worker_num = worker_num
                self.participants[worker_id] = (connection, arrival)
            except BaseException:
                connection.close()
                raise
        if root_pid not in {p[1]["pid"] for p in self.participants.values()}:
            raise RuntimeError("Parent process did not join checkpoint quorum")
        return [p[1] for p in self.participants.values()]

    def snapshots(self, timeout=5):
        deadline = time.monotonic() + timeout
        for connection, _ in self.participants.values():
            write_socket_message(connection, "snapshot", deadline)
        snapshots = []
        for connection, arrival in self.participants.values():
            snapshot = read_socket_message(connection, deadline)
            if snapshot["pid"] != arrival["pid"]:
                raise RuntimeError("Checkpoint participant PID changed")
            snapshots.append(snapshot)
        return snapshots

    def observe(self, root_pid, seconds=5):
        expected = None
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            snapshots = self.snapshots(timeout=min(2, deadline - time.monotonic()))
            network = tcp_snapshot(root_pid)
            pids = network["pids"]
            if expected is not None and pids != expected:
                raise RuntimeError(
                    f"Process tree changed at checkpoint: {expected} -> {pids}"
                )
            expected = pids
            instrumented = {s["pid"] for s in snapshots}
            if not instrumented.issubset(pids):
                raise RuntimeError(
                    "Checkpoint participant left the service process tree"
                )
            for pid in set(pids) - instrumented:
                maps = Path(f"/proc/{pid}/maps").read_text()
                if "libcuda" in maps or "libtorch_cuda" in maps:
                    raise RuntimeError(
                        f"CUDA process {pid} has no pinned-memory observation"
                    )
            yield {"network": network, "participants": snapshots}
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            connections = [p[0] for p in self.participants.values()]
            readable, _, _ = select.select(connections, [], [], min(remaining, 1))
            if readable:
                raise RuntimeError(
                    "Checkpoint participant returned or exited before release"
                )

    def release(self):
        deadline = time.monotonic() + 5
        for connection, _ in self.participants.values():
            write_socket_message(connection, "release", deadline)

    def close(self):
        for connection, _ in self.participants.values():
            connection.close()
        self.listener.close()
        self.directory.cleanup()


def normalized_ip(value):
    address = ipaddress.ip_address(value.split("%", 1)[0])
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
        address = address.ipv4_mapped
    return address


def tcp_snapshot(root_pid):
    local_ips = {
        normalized_ip(address.address)
        for addresses in psutil.net_if_addrs().values()
        for address in addresses
        if address.family in (socket.AF_INET, socket.AF_INET6)
    }
    root = psutil.Process(root_pid)
    processes = [root, *root.children(recursive=True)]
    internal = []
    external = []
    listeners = []
    for process in processes:
        if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
            raise RuntimeError(f"Checkpoint process exited: {process.pid}")
        for connection in process.net_connections(kind="tcp"):
            entry = {
                "pid": process.pid,
                "fd": connection.fd,
                "local": list(connection.laddr),
                "remote": list(connection.raddr),
                "status": connection.status,
            }
            if connection.status == psutil.CONN_LISTEN:
                listeners.append(entry)
            elif connection.raddr:
                remote_ip = normalized_ip(connection.raddr[0])
                if remote_ip.is_loopback or remote_ip in local_ips:
                    internal.append(entry)
                else:
                    external.append(entry)
    return {
        "pids": sorted(process.pid for process in processes),
        "internal_tcp": internal,
        "external_tcp": external,
        "tcp_listeners": listeners,
    }


def kv_backing_snapshot(participants, process_log):
    pools = []
    for line in process_log.splitlines():
        if "backing selected:" not in line:
            continue
        match = re.search(
            r"\[(\d+):\d+\].*\[/?.*/(DeviceBlockPool|HostBlockPool)\.cc:\d+\]",
            line,
        )
        if match is None:
            if "DeviceBlockPool.cc:" in line or "HostBlockPool.cc:" in line:
                raise RuntimeError(f"Cannot parse KV backing log: {line}")
            continue
        fields = dict(re.findall(r"(\w+)=(\S+)", line.split("backing selected:", 1)[1]))
        pools.append(
            {
                "pid": int(match[1]),
                "pool_name": fields["pool_name"],
                "ptr": int(fields["ptr"], 16),
                "bytes": int(fields["total_size"]),
                "is_cuda": match[2] == "DeviceBlockPool" and fields["is_cuda"] == "1",
                "is_pinned": match[2] == "HostBlockPool" or fields["is_pinned"] == "1",
            }
        )
    result = {}
    for participant in participants:
        pid = participant["pid"]
        owned = [pool for pool in pools if pool["pid"] == pid]
        gpu_pools = [pool for pool in owned if pool["is_cuda"]]
        if participant["gpu_kv_bytes"] > 0 and not gpu_pools:
            raise RuntimeError(f"GPU KV backing log missing for backend pid={pid}")
        host_pools = []
        for pool in owned:
            if not pool["is_pinned"]:
                continue
            start, end = pool["ptr"], pool["ptr"] + pool["bytes"]
            overlaps = []
            for allocation in participant["pinned"]["allocations"]:
                left = max(start, allocation["ptr"])
                right = min(end, allocation["ptr"] + allocation["bytes"])
                if left < right:
                    overlaps.append((left, right))
            cursor = start
            pinned_bytes = 0
            for left, right in sorted(overlaps):
                left = max(left, cursor)
                if left < right:
                    pinned_bytes += right - left
                    cursor = right
            # Startup pool backings stay owned until the pre-service barrier returns.
            if pinned_bytes != pool["bytes"]:
                raise RuntimeError(f"Host KV backing not fully observed: {pool}")
            host_pools.append({**pool, "active_pinned_bytes": pinned_bytes})
        result[pid] = {
            "gpu_pools": gpu_pools,
            "host_pools": host_pools,
            "active_pinned_bytes": sum(p["active_pinned_bytes"] for p in host_pools),
        }
    if any(pool["pid"] not in result for pool in pools):
        raise RuntimeError("KV backing belongs to an unobserved checkpoint process")
    return result


def assert_clean_checkpoint(snapshot, expected_tp_size=1):
    network = snapshot["network"]
    if network["external_tcp"]:
        raise AssertionError(f"External TCP connections exist at checkpoint: {network}")
    if not snapshot["participants"]:
        raise AssertionError("No checkpoint participant observations")
    for participant in snapshot["participants"]:
        pinned = participant["pinned"]
        if (
            pinned["pid"] != participant["pid"]
            or not pinned["initialized"]
            or pinned["errors"] != 0
        ):
            raise AssertionError(f"Pinned observation failed: {participant}")
    gpu_participants = [p for p in snapshot["participants"] if p["gpu_kv_bytes"] > 0]
    if len(gpu_participants) != expected_tp_size:
        raise AssertionError(
            f"Expected {expected_tp_size} initialized TP backends, got {len(gpu_participants)}"
        )
    if any(p["pinned"]["driver_calls"] == 0 for p in gpu_participants):
        raise AssertionError("CUDA allocation observation is inactive")
    for participant in snapshot["participants"]:
        if participant["host_kv"]["active_pinned_bytes"] != 0:
            raise AssertionError(f"Host KV pinned resources exist: {participant}")
