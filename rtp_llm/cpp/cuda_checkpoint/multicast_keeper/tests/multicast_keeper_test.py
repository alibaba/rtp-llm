#!/usr/bin/env python3

import array
import os
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

PACKAGE = Path("rtp_llm/cpp/cuda_checkpoint/multicast_keeper")
PROTOCOL_MAGIC = 0x3250434D505452
CREATOR_MAGIC = 0x3243464D505452
PROTOCOL_VERSION = 3
OP_PING = 1
OP_CREATE = 2
OP_FETCH = 3
OP_RELEASE = 4
OP_IMPORT_ADD = 5
OP_FETCH_FABRIC = 6
STATUS_OK = 0
STATUS_INVALID_REQUEST = 1
STATUS_UNSUPPORTED_PROPERTIES = 2
STATUS_STALE_INSTANCE = 3
STATUS_UNKNOWN_OBJECT = 4
STATUS_PROPERTY_MISMATCH = 5
STATUS_CREATOR_FAILED = 6
STATUS_CAPACITY_EXCEEDED = 7
STATUS_OWNER_MISMATCH = 9
HANDLE_POSIX = 1
HANDLE_FABRIC = 8
HANDLE_FABRIC_POSIX = HANDLE_FABRIC | HANDLE_POSIX
CREATOR_FLAG_FABRIC_VALID = 1
FABRIC_BYTES = 64
HANG_SIZE = 0xBAD000
UNKNOWN_SIZE = (1 << 64) - 1
MAX_ENTRIES = 256
REQUEST = struct.Struct("<QHHIQQQQIIQ")
REQUEST_EXT = struct.Struct("<QHHIQQQQIIQQQ")
RESPONSE = struct.Struct("<QHHIiIQQQQQIIQ")
# rtp_mc_creator_result (96B) and rtp_mc_import_add_request (rtp_mc_request_ext +
# 64-byte fabric handle = 144B). Kept byte-locked to keeper_protocol.h.
CREATOR_RESULT = struct.Struct("<QQQiI64s")
IMPORT_ADD_REQUEST = struct.Struct("<QHHIQQQQIIQQQ64s")


def canned_fabric_handle(size: int) -> bytes:
    """Deterministic 64-byte fabric handle the fake creator emits for a size."""
    return (b"FAKEFAB:" + str(size).encode()).ljust(FABRIC_BYTES, b"\0")[:FABRIC_BYTES]


def find_runfile(name: str) -> Path:
    candidates = []
    test_srcdir = os.environ.get("TEST_SRCDIR")
    workspace = os.environ.get("TEST_WORKSPACE")
    if test_srcdir and workspace:
        candidates.append(Path(test_srcdir) / workspace / PACKAGE / name)
    candidates.append(Path(__file__).resolve().parents[1] / name)
    repo = Path(__file__).resolve().parents[5]
    candidates.append(repo / "bazel-bin" / PACKAGE / name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"cannot locate {name}: {candidates}")


def wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


class MulticastKeeperTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.holder = find_runfile("keeper_lite_holder")
        cls.creator = find_runfile("keeper_lite_creator")
        cls.launcher = find_runfile("multicast_keeper")
        cls.shim = find_runfile("mc_shim_unified.so")
        cls.fake_cuda = find_runfile("libcuda.so.1")

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="rtp_mc_keeper_test_")
        self.root = Path(self.temporary.name)
        self.socket_path = self.root / "mcsk.sock"
        self.ready_path = self.root / "holder.ready"
        self.counter_path = self.root / "creator.count"
        self.holder_log_path = self.root / "holder.log"
        self.fake_creator = self.root / "fake_creator.py"
        self.fake_creator.write_text(
            f"""#!{sys.executable}
import argparse, array, os, socket, struct, tempfile, time
p=argparse.ArgumentParser()
p.add_argument('--gpus', required=True)
p.add_argument('--size', required=True, type=int)
p.add_argument('--num-devices', required=True, type=int)
p.add_argument('--handle-types', required=True, type=int)
p.add_argument('--flags', required=True, type=int)
p.add_argument('--deposit-fd', required=True, type=int)
p.add_argument('--import-fabric-fd', type=int, default=-1)
a=p.parse_args()
with open(os.environ['FAKE_CREATOR_COUNT'], 'a') as f: f.write(f'{{a.size}}\\n')
if a.size == {HANG_SIZE}: time.sleep(60)
result_flags = 0
fabric = b'\\0' * {FABRIC_BYTES}
if a.import_fabric_fd >= 0:
    # IMPORT_ADD mode: drain the 64-byte fabric handle from the holder's pipe.
    buf = b''
    while len(buf) < {FABRIC_BYTES}:
        chunk = os.read(a.import_fabric_fd, {FABRIC_BYTES} - len(buf))
        if not chunk: break
        buf += chunk
    assert len(buf) == {FABRIC_BYTES}, 'short fabric handle'
    fabric = buf
elif a.handle_types & {HANDLE_FABRIC}:
    # CREATE(FABRIC): emit a deterministic canned 64-byte handle inline.
    fabric = (b'FAKEFAB:' + str(a.size).encode()).ljust({FABRIC_BYTES}, b'\\0')[:{FABRIC_BYTES}]
    result_flags = {CREATOR_FLAG_FABRIC_VALID}
backing=tempfile.TemporaryFile()
fd=backing.fileno()
os.write(fd, f'multicast:{{a.gpus}}:{{a.size}}:{{os.getpid()}}'.encode())
s=socket.socket(fileno=a.deposit_fd)
served = a.size if a.size == {UNKNOWN_SIZE} else a.size + 4096
payload=struct.pack('<QQQiI64s', {CREATOR_MAGIC}, a.size, served, 0, result_flags, fabric)
s.sendmsg([payload], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array('i',[fd]))])
s.close(); backing.close()
""",
            encoding="utf-8",
        )
        self.fake_creator.chmod(0o755)
        self.environment = os.environ.copy()
        self.environment["FAKE_CREATOR_COUNT"] = str(self.counter_path)
        self.holder_log = self.holder_log_path.open("w", encoding="utf-8")
        self.process = self.start_holder()

    def start_holder(
        self,
        client_timeout_ms: int = 200,
        creator_timeout_ms: int = 300,
        fabric_team_size: int | None = 8,
    ) -> subprocess.Popen:
        command = [
            str(self.holder),
            "--socket",
            str(self.socket_path),
            "--ready-file",
            str(self.ready_path),
            "--creator",
            str(self.fake_creator),
            "--gpus",
            "0,2,7",
            "--client-timeout-ms",
            str(client_timeout_ms),
            "--creator-timeout-ms",
            str(creator_timeout_ms),
        ]
        if fabric_team_size is not None:
            command.extend(["--fabric-team-size", str(fabric_team_size)])
        process = subprocess.Popen(
            command,
            env=self.environment,
            stdout=self.holder_log,
            stderr=subprocess.STDOUT,
        )
        self.assertTrue(
            wait_until(lambda: self.ready_path.exists() and self.socket_path.exists()),
            "holder did not become ready",
        )
        return process

    def stop_holder(self) -> None:
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGTERM)
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)

    def tearDown(self) -> None:
        self.stop_holder()
        self.holder_log.close()
        self.temporary.cleanup()

    def exchange(
        self,
        opcode: int,
        *,
        holder_instance=(0, 0),
        object_id: int = 0,
        size: int = 0,
        num_devices: int = 3,
        handle_types: int = HANDLE_POSIX,
        flags: int = 0,
        owner_id=None,
        owner_generation: int = 0,
        fabric_handle=None,
        struct_size=None,
        timeout: float = 2.0,
    ):
        if fabric_handle is not None:
            # 144-byte IMPORT_ADD form: extended request + 64-byte fabric handle.
            # struct_size defaults to 144 but is overridable for negative tests.
            assert len(fabric_handle) == FABRIC_BYTES
            request = IMPORT_ADD_REQUEST.pack(
                PROTOCOL_MAGIC,
                PROTOCOL_VERSION,
                opcode,
                IMPORT_ADD_REQUEST.size if struct_size is None else struct_size,
                holder_instance[0],
                holder_instance[1],
                object_id,
                size,
                num_devices,
                handle_types,
                flags,
                owner_id or 0,
                owner_generation,
                fabric_handle,
            )
        elif owner_id is None:
            # Base 64-byte request (anonymous owner); struct_size must equal 64.
            request = REQUEST.pack(
                PROTOCOL_MAGIC,
                PROTOCOL_VERSION,
                opcode,
                REQUEST.size if struct_size is None else struct_size,
                holder_instance[0],
                holder_instance[1],
                object_id,
                size,
                num_devices,
                handle_types,
                flags,
            )
        else:
            # Extended 80-byte request carrying owner attribution.
            request = REQUEST_EXT.pack(
                PROTOCOL_MAGIC,
                PROTOCOL_VERSION,
                opcode,
                REQUEST_EXT.size if struct_size is None else struct_size,
                holder_instance[0],
                holder_instance[1],
                object_id,
                size,
                num_devices,
                handle_types,
                flags,
                owner_id,
                owner_generation,
            )
        client = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        client.settimeout(timeout)
        client.connect(str(self.socket_path))
        client.sendall(request)
        message, ancillary, message_flags, _ = client.recvmsg(
            RESPONSE.size, socket.CMSG_SPACE(array.array("i").itemsize)
        )
        client.close()
        self.assertEqual(message_flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC), 0)
        self.assertEqual(len(message), RESPONSE.size)
        values = RESPONSE.unpack(message)
        self.assertEqual(values[0], PROTOCOL_MAGIC)
        self.assertEqual(values[1], PROTOCOL_VERSION)
        self.assertEqual(values[2], opcode)
        self.assertEqual(values[3], RESPONSE.size)
        received_fds = array.array("i")
        for level, kind, data in ancillary:
            if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                received_fds.frombytes(
                    data[: len(data) - len(data) % received_fds.itemsize]
                )
        response = {
            "status": values[4],
            "local_device_count": values[5],
            "instance": (values[6], values[7]),
            "object_id": values[8],
            "requested_size": values[9],
            "served_size": values[10],
            "num_devices": values[11],
            "handle_types": values[12],
            "flags": values[13],
        }
        return response, list(received_fds)

    def create(self, size: int, **kwargs):
        return self.exchange(OP_CREATE, size=size, **kwargs)

    def fetch(self, created, **overrides):
        parameters = {
            "holder_instance": created["instance"],
            "object_id": created["object_id"],
            "size": created["requested_size"],
            "num_devices": created["num_devices"],
            "handle_types": created["handle_types"],
            "flags": created["flags"],
        }
        parameters.update(overrides)
        return self.exchange(OP_FETCH, **parameters)

    def release(self, created, *, owner_id, owner_generation=0, **overrides):
        parameters = {
            "holder_instance": created["instance"],
            "object_id": created["object_id"],
            "size": created["requested_size"],
            "num_devices": created["num_devices"],
            "handle_types": created["handle_types"],
            "flags": created["flags"],
            "owner_id": owner_id,
            "owner_generation": owner_generation,
        }
        parameters.update(overrides)
        return self.exchange(OP_RELEASE, **parameters)

    def import_add(
        self,
        fabric_handle: bytes,
        *,
        size: int = 4096,
        num_devices: int = 8,
        handle_types: int = HANDLE_FABRIC_POSIX,
        owner_id: int = 21,
        owner_generation: int = 1,
        **overrides,
    ):
        parameters = {
            "size": size,
            "num_devices": num_devices,
            "handle_types": handle_types,
            "owner_id": owner_id,
            "owner_generation": owner_generation,
            "fabric_handle": fabric_handle,
        }
        parameters.update(overrides)
        return self.exchange(OP_IMPORT_ADD, **parameters)

    def fetch_fabric(self, created, **overrides):
        parameters = {
            "holder_instance": created["instance"],
            "object_id": created["object_id"],
            "size": created["requested_size"],
            "num_devices": created["num_devices"],
            "handle_types": created["handle_types"],
            "flags": created["flags"],
        }
        parameters.update(overrides)
        return self.exchange(OP_FETCH_FABRIC, **parameters)

    def entry_count(self) -> int:
        # The holder reports its live entry count in the PING response object_id.
        ping, _ = self.exchange(OP_PING)
        self.assertEqual(ping["status"], STATUS_OK)
        return ping["object_id"]

    def assert_success_fd(self, response, fds) -> int:
        if response["status"] != STATUS_OK:
            self.holder_log.flush()
            self.fail(
                f"holder status={response['status']}; log:\n"
                f"{self.holder_log_path.read_text(encoding='utf-8')}"
            )
        self.assertEqual(len(fds), 1)
        return fds[0]

    def test_protocol_layout_matches_collective_torch_ping(self) -> None:
        self.assertEqual(PROTOCOL_MAGIC, 0x3250434D505452)
        self.assertEqual(PROTOCOL_VERSION, 3)
        self.assertEqual(OP_PING, 1)
        self.assertEqual(REQUEST.format, "<QHHIQQQQIIQ")
        self.assertEqual(REQUEST.size, 64)
        self.assertEqual(REQUEST_EXT.format, "<QHHIQQQQIIQQQ")
        self.assertEqual(REQUEST_EXT.size, 80)
        self.assertEqual(RESPONSE.format, "<QHHIiIQQQQQIIQ")
        self.assertEqual(RESPONSE.size, 80)
        self.assertEqual(OP_RELEASE, 4)
        # Cross-machine (MNNVL) fabric wire additions must stay byte-locked to
        # keeper_protocol.h (creator_result==96, import_add_request==144).
        self.assertEqual(OP_IMPORT_ADD, 5)
        self.assertEqual(OP_FETCH_FABRIC, 6)
        self.assertEqual(HANDLE_FABRIC, 8)
        self.assertEqual(FABRIC_BYTES, 64)
        self.assertEqual(CREATOR_RESULT.size, 96)
        self.assertEqual(IMPORT_ADD_REQUEST.size, 144)
        self.assertEqual(IMPORT_ADD_REQUEST.size, REQUEST_EXT.size + FABRIC_BYTES)

    def test_same_size_create_has_independent_object_identity(self) -> None:
        first, first_fds = self.create(2 * 1024 * 1024)
        second, second_fds = self.create(2 * 1024 * 1024)
        first_fd = self.assert_success_fd(first, first_fds)
        second_fd = self.assert_success_fd(second, second_fds)
        self.assertNotEqual(first["object_id"], second["object_id"])
        self.assertEqual(first["instance"], second["instance"])
        os.close(first_fd)
        os.close(second_fd)
        self.assertEqual(
            self.counter_path.read_text(encoding="utf-8").splitlines(),
            ["2097152", "2097152"],
        )

    def test_fetch_rebuild_reuses_exact_object_without_creator(self) -> None:
        created, create_fds = self.create(2 * 1024 * 1024)
        os.close(self.assert_success_fd(created, create_fds))
        rebuilt, rebuild_fds = self.fetch(created)
        os.close(self.assert_success_fd(rebuilt, rebuild_fds))
        self.assertEqual(rebuilt["object_id"], created["object_id"])
        self.assertEqual(rebuilt["instance"], created["instance"])
        self.assertEqual(
            self.counter_path.read_text(encoding="utf-8").splitlines(), ["2097152"]
        )

    def test_shim_release_rebuild_fetches_saved_identity(self) -> None:
        script = textwrap.dedent(
            """
            import ctypes, os

            class Properties(ctypes.Structure):
                _fields_ = [
                    ("numDevices", ctypes.c_uint),
                    ("size", ctypes.c_size_t),
                    ("handleTypes", ctypes.c_ulonglong),
                    ("flags", ctypes.c_ulonglong),
                ]

            shim = ctypes.CDLL(os.environ["TEST_MC_SHIM"], mode=ctypes.RTLD_GLOBAL)
            create = shim.cuMulticastCreate
            create.argtypes = [ctypes.POINTER(ctypes.c_ulonglong), ctypes.POINTER(Properties)]
            create.restype = ctypes.c_int
            release = shim.cuMemRelease
            release.argtypes = [ctypes.c_ulonglong]
            release.restype = ctypes.c_int
            add_device = shim.cuMulticastAddDevice
            add_device.argtypes = [ctypes.c_ulonglong, ctypes.c_int]
            add_device.restype = ctypes.c_int

            properties = Properties(3, 2 * 1024 * 1024, 1, 0)
            first = ctypes.c_ulonglong()
            assert create(ctypes.byref(first), ctypes.byref(properties)) == 0
            assert add_device(first.value, 1) == 0
            assert add_device(first.value, 7) == 1
            duplicate = ctypes.c_ulonglong()
            assert create(ctypes.byref(duplicate), ctypes.byref(properties)) == 1
            assert release(first.value) == 0
            rebuilt = ctypes.c_ulonglong()
            assert create(ctypes.byref(rebuilt), ctypes.byref(properties)) == 0
            assert rebuilt.value != first.value
            assert release(rebuilt.value) == 0
            """
        )
        environment = self.environment.copy()
        environment.update(
            {
                "LD_LIBRARY_PATH": f"{self.fake_cuda.parent}:"
                f"{environment.get('LD_LIBRARY_PATH', '')}",
                "TEST_MC_SHIM": str(self.shim),
                "RTP_LLM_CUDA_CKPT_MULTICAST_KEEPER": "1",
                "RTP_LLM_CUDA_CKPT_MULTICAST_SOCKET": str(self.socket_path),
                "RTP_LLM_MC_CREATE_TIMEOUT_MS": "2000",
                "RTP_LLM_MC_REQUEST_TIMEOUT_MS": "1000",
            }
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            env=environment,
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(
            self.counter_path.read_text(encoding="utf-8").splitlines(), ["2097152"]
        )

    def test_shim_raw_fabric_is_promoted_only_by_add_device(self) -> None:
        script = textwrap.dedent(
            """
            import ctypes, os

            shim = ctypes.CDLL(os.environ["TEST_MC_SHIM"], mode=ctypes.RTLD_GLOBAL)
            import_handle = shim.cuMemImportFromShareableHandle
            import_handle.argtypes = [ctypes.POINTER(ctypes.c_ulonglong), ctypes.c_void_p, ctypes.c_int]
            import_handle.restype = ctypes.c_int
            add_device = shim.cuMulticastAddDevice
            add_device.argtypes = [ctypes.c_ulonglong, ctypes.c_int]
            add_device.restype = ctypes.c_int
            release = shim.cuMemRelease
            release.argtypes = [ctypes.c_ulonglong]
            release.restype = ctypes.c_int

            ordinary_raw = (ctypes.c_ubyte * 64).from_buffer_copy(b"ORDINARY".ljust(64, b"\\0"))
            ordinary = ctypes.c_ulonglong()
            assert import_handle(ctypes.byref(ordinary), ordinary_raw, 8) == 0
            assert release(ordinary.value) == 0

            multicast_raw = (ctypes.c_ubyte * 64).from_buffer_copy(b"RAW-FABRIC".ljust(64, b"\\0"))
            first = ctypes.c_ulonglong()
            assert import_handle(ctypes.byref(first), multicast_raw, 8) == 0
            assert add_device(first.value, 0) == 0
            assert add_device(first.value, 2) == 0
            assert release(first.value) == 0

            # A checkpoint-style rebuild imports a new CUDA handle but retains
            # one idempotent holder owner reference for this process generation.
            rebuilt = ctypes.c_ulonglong()
            assert import_handle(ctypes.byref(rebuilt), multicast_raw, 8) == 0
            assert add_device(rebuilt.value, 2) == 0
            assert release(rebuilt.value) == 0
            """
        )
        environment = self.environment.copy()
        environment.update(
            {
                "LD_LIBRARY_PATH": f"{self.fake_cuda.parent}:"
                f"{environment.get('LD_LIBRARY_PATH', '')}",
                "TEST_MC_SHIM": str(self.shim),
                "RTP_LLM_CUDA_CKPT_MULTICAST_KEEPER": "1",
                "RTP_LLM_CUDA_CKPT_MULTICAST_SOCKET": str(self.socket_path),
                "RTP_LLM_MC_FABRIC_TEAM_SIZE": "8",
                "RTP_LLM_MC_LOCAL_GPUS": "0,2,7",
                "CUDA_VISIBLE_DEVICES": "0,2,7",
                "LOCAL_RANK": "4",
            }
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            env=environment,
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        # The promoted holder ref is released by the shim destructor. The
        # ordinary allocation never entered the keeper at all.
        self.assertEqual(self.entry_count(), 0)
        self.assertEqual(
            self.counter_path.read_text(encoding="utf-8").splitlines(),
            [str(UNKNOWN_SIZE)],
        )

    def test_shim_raw_fabric_tracking_grows_without_losing_multicast(self) -> None:
        script = textwrap.dedent(
            """
            import ctypes, os

            shim = ctypes.CDLL(os.environ["TEST_MC_SHIM"], mode=ctypes.RTLD_GLOBAL)
            import_handle = shim.cuMemImportFromShareableHandle
            import_handle.argtypes = [ctypes.POINTER(ctypes.c_ulonglong), ctypes.c_void_p, ctypes.c_int]
            import_handle.restype = ctypes.c_int
            add_device = shim.cuMulticastAddDevice
            add_device.argtypes = [ctypes.c_ulonglong, ctypes.c_int]
            add_device.restype = ctypes.c_int
            release = shim.cuMemRelease
            release.argtypes = [ctypes.c_ulonglong]
            release.restype = ctypes.c_int

            raw = (ctypes.c_ubyte * 64).from_buffer_copy(b"ORDINARY".ljust(64, b"\\0"))
            handles = []
            for _ in range(1025):
                handle = ctypes.c_ulonglong()
                assert import_handle(ctypes.byref(handle), raw, 8) == 0
                handles.append(handle.value)

            multicast_raw = (ctypes.c_ubyte * 64).from_buffer_copy(b"AFTER-GROWTH".ljust(64, b"\\0"))
            multicast = ctypes.c_ulonglong()
            assert import_handle(ctypes.byref(multicast), multicast_raw, 8) == 0
            assert add_device(multicast.value, 2) == 0
            assert release(multicast.value) == 0
            for handle in handles:
                assert release(handle) == 0
            """
        )
        environment = self.environment.copy()
        environment.update(
            {
                "LD_LIBRARY_PATH": f"{self.fake_cuda.parent}:"
                f"{environment.get('LD_LIBRARY_PATH', '')}",
                "TEST_MC_SHIM": str(self.shim),
                "RTP_LLM_CUDA_CKPT_MULTICAST_KEEPER": "1",
                "RTP_LLM_CUDA_CKPT_MULTICAST_SOCKET": str(self.socket_path),
                "RTP_LLM_MC_FABRIC_TEAM_SIZE": "8",
                "RTP_LLM_MC_LOCAL_GPUS": "0,2,7",
                "CUDA_VISIBLE_DEVICES": "0,2,7",
                "LOCAL_RANK": "4",
            }
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            env=environment,
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(self.entry_count(), 0)
        self.assertEqual(
            self.counter_path.read_text(encoding="utf-8").splitlines(),
            [str(UNKNOWN_SIZE)],
        )

    def test_release_frees_slot_and_allows_reregistration(self) -> None:
        created, fds = self.create(4096, owner_id=11, owner_generation=1)
        os.close(self.assert_success_fd(created, fds))
        self.assertEqual(self.entry_count(), 1)
        released, released_fds = self.release(created, owner_id=11, owner_generation=1)
        self.assertEqual(released["status"], STATUS_OK)
        self.assertEqual(released_fds, [])
        self.assertEqual(self.entry_count(), 0)
        # The released object id is monotonic and never reused; fetching it fails.
        stale, stale_fds = self.fetch(created)
        self.assertEqual(stale["status"], STATUS_UNKNOWN_OBJECT)
        self.assertEqual(stale_fds, [])
        # Re-registration after RELEASE succeeds and takes the freed slot.
        again, again_fds = self.create(4096, owner_id=11, owner_generation=1)
        os.close(self.assert_success_fd(again, again_fds))
        self.assertNotEqual(again["object_id"], created["object_id"])
        self.assertEqual(self.entry_count(), 1)

    def test_release_of_unknown_or_foreign_object_fails_closed(self) -> None:
        created, fds = self.create(4096, owner_id=11, owner_generation=1)
        os.close(self.assert_success_fd(created, fds))
        # Unknown object id -> UNKNOWN_OBJECT, object untouched.
        unknown, unknown_fds = self.release(
            created,
            owner_id=11,
            owner_generation=1,
            object_id=created["object_id"] + 4321,
        )
        self.assertEqual(unknown["status"], STATUS_UNKNOWN_OBJECT)
        self.assertEqual(unknown_fds, [])
        # Wrong owner -> OWNER_MISMATCH, object untouched.
        foreign, foreign_fds = self.release(created, owner_id=999, owner_generation=1)
        self.assertEqual(foreign["status"], STATUS_OWNER_MISMATCH)
        self.assertEqual(foreign_fds, [])
        # Stale holder instance -> STALE_INSTANCE, object untouched.
        stale, stale_fds = self.release(
            created, owner_id=11, owner_generation=1, holder_instance=(1, 2)
        )
        self.assertEqual(stale["status"], STATUS_STALE_INSTANCE)
        self.assertEqual(stale_fds, [])
        # None of the failed releases freed the object.
        self.assertEqual(self.entry_count(), 1)
        rebuilt, rebuilt_fds = self.fetch(created)
        os.close(self.assert_success_fd(rebuilt, rebuilt_fds))

    def test_new_owner_generation_reclaims_orphan_entries(self) -> None:
        first, first_fds = self.create(4096, owner_id=7, owner_generation=100)
        os.close(self.assert_success_fd(first, first_fds))
        self.assertEqual(self.entry_count(), 1)
        # Same owner, new generation: the stale entry is reclaimed, netting one.
        second, second_fds = self.create(4096, owner_id=7, owner_generation=200)
        os.close(self.assert_success_fd(second, second_fds))
        self.assertNotEqual(second["object_id"], first["object_id"])
        self.assertEqual(self.entry_count(), 1)
        # The orphan from the previous generation is gone.
        gone, gone_fds = self.fetch(first)
        self.assertEqual(gone["status"], STATUS_UNKNOWN_OBJECT)
        self.assertEqual(gone_fds, [])
        # A different owner is never reclaimed by another owner's CREATE.
        other, other_fds = self.create(4096, owner_id=8, owner_generation=100)
        os.close(self.assert_success_fd(other, other_fds))
        self.assertEqual(self.entry_count(), 2)
        # Same owner AND same generation is NOT reclaimed (dual-object owners).
        dual, dual_fds = self.create(8192, owner_id=7, owner_generation=200)
        os.close(self.assert_success_fd(dual, dual_fds))
        self.assertEqual(self.entry_count(), 3)

    def test_capacity_exhaustion_fails_closed(self) -> None:
        # Use a holder with generous creator/client timeouts so filling every
        # slot with the (interpreter-startup-heavy) fake creator is not flaky.
        self.stop_holder()
        self.assertTrue(wait_until(lambda: not self.socket_path.exists()))
        self.ready_path.unlink(missing_ok=True)
        self.process = self.start_holder(
            client_timeout_ms=2000, creator_timeout_ms=10000
        )
        open_fds = []
        try:
            for _ in range(MAX_ENTRIES):
                response, fds = self.create(4096, timeout=15.0)
                open_fds.append(self.assert_success_fd(response, fds))
            self.assertEqual(self.entry_count(), MAX_ENTRIES)
            full, full_fds = self.create(4096, timeout=15.0)
            self.assertEqual(full["status"], STATUS_CAPACITY_EXCEEDED)
            self.assertEqual(full_fds, [])
            # The holder stays healthy after refusing an over-capacity create.
            self.assertEqual(self.entry_count(), MAX_ENTRIES)
        finally:
            for fd in open_fds:
                os.close(fd)

    def test_stale_holder_instance_is_rejected(self) -> None:
        created, fds = self.create(4096)
        os.close(self.assert_success_fd(created, fds))
        old_instance = created["instance"]
        self.stop_holder()
        self.assertTrue(wait_until(lambda: not self.socket_path.exists()))
        self.process = self.start_holder()
        response, stale_fds = self.fetch(created)
        self.assertEqual(response["status"], STATUS_STALE_INSTANCE)
        self.assertEqual(stale_fds, [])
        self.assertNotEqual(response["instance"], old_instance)

    def test_partial_and_silent_client_cannot_block_holder(self) -> None:
        partial = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        partial.connect(str(self.socket_path))
        partial.sendall(b"partial")
        partial.close()

        silent = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        silent.connect(str(self.socket_path))
        start = time.monotonic()
        ping, ping_fds = self.exchange(OP_PING, timeout=2.0)
        elapsed = time.monotonic() - start
        silent.close()
        self.assertEqual(ping["status"], STATUS_OK)
        self.assertEqual(ping_fds, [])
        self.assertLess(elapsed, 1.0)

    def test_creator_hang_times_out_and_holder_stays_healthy(self) -> None:
        start = time.monotonic()
        response, fds = self.create(HANG_SIZE, timeout=2.0)
        self.assertEqual(response["status"], STATUS_CREATOR_FAILED)
        self.assertEqual(fds, [])
        self.assertLess(time.monotonic() - start, 1.5)
        children = Path(
            f"/proc/{self.process.pid}/task/{self.process.pid}/children"
        ).read_text(encoding="utf-8")
        self.assertEqual(children.strip(), "")
        ping, _ = self.exchange(OP_PING)
        self.assertEqual(ping["status"], STATUS_OK)

    def test_properties_fail_closed_for_subgroup_unknown_handle_and_flags(self) -> None:
        for overrides in (
            {"num_devices": 2},
            {"handle_types": 0x10},
            {"flags": 1},
        ):
            response, fds = self.create(4096, **overrides)
            self.assertEqual(response["status"], STATUS_UNSUPPORTED_PROPERTIES)
            self.assertEqual(fds, [])
        self.assertFalse(self.counter_path.exists())

    def test_holder_is_ready_without_cuda_and_check_reports_entries(self) -> None:
        ready = self.ready_path.read_text(encoding="utf-8")
        self.assertIn("protocol=3", ready)
        self.assertIn("fabric_team_size=8", ready)
        maps = Path(f"/proc/{self.process.pid}/maps").read_text(encoding="utf-8")
        self.assertNotIn("libcuda.so", maps)
        self.assertNotIn("libcudart.so", maps)
        created, fds = self.create(4096)
        os.close(self.assert_success_fd(created, fds))
        checked = subprocess.run(
            [str(self.holder), "--check", "--socket", str(self.socket_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("protocol=3", checked.stdout)
        self.assertIn("entries=1", checked.stdout)

    def test_second_holder_does_not_replace_live_socket(self) -> None:
        duplicate = subprocess.run(
            [
                str(self.holder),
                "--socket",
                str(self.socket_path),
                "--creator",
                str(self.fake_creator),
                "--gpus",
                "0,1",
            ],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(duplicate.returncode, 0)
        self.assertIn("Address already in use", duplicate.stderr)
        self.assertIsNone(self.process.poll())

    def test_sigterm_removes_socket_and_ready_file(self) -> None:
        self.process.send_signal(signal.SIGTERM)
        self.assertEqual(self.process.wait(timeout=5), 0)
        self.assertTrue(wait_until(lambda: not self.socket_path.exists()))
        self.assertFalse(self.ready_path.exists())

    def test_creator_argument_validation_and_dry_run_do_not_initialize_cuda(
        self,
    ) -> None:
        base = [
            str(self.creator),
            "--dry-run",
            "--gpus",
            "0,2,7",
            "--size",
            "64MiB",
            "--num-devices",
            "3",
            "--handle-types",
            "9",
            "--flags",
            "0",
        ]
        valid = subprocess.run(base, check=True, capture_output=True, text=True)
        self.assertIn("num_devices=3", valid.stdout)
        self.assertIn("requested_size=67108864", valid.stdout)
        self.assertIn("handle_types=0x9", valid.stdout)
        self.assertIn("no_cuda=1", valid.stdout)
        invalid = subprocess.run(
            [
                str(self.creator),
                "--dry-run",
                "--gpus",
                "0,0",
                "--size",
                "64MiB",
                "--num-devices",
                "2",
                "--handle-types",
                "1",
                "--flags",
                "0",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(invalid.returncode, 2)

    def test_launcher_requires_gpu_list_and_documents_preload_merge(self) -> None:
        missing = subprocess.run(
            [str(self.launcher), "start", "--keeper-dir", str(self.root / "launch")],
            capture_output=True,
            text=True,
        )
        self.assertEqual(missing.returncode, 2)
        self.assertIn("requires --gpus", missing.stderr)
        launcher_text = Path(self.launcher).read_text(encoding="utf-8")
        self.assertIn("RTP_LLM_CUDA_CKPT_MULTICAST_KEEPER=1", launcher_text)
        self.assertIn("RTP_LLM_MC_LOCAL_GPUS", launcher_text)
        self.assertIn("RTP_LLM_MC_FABRIC_TEAM_SIZE", launcher_text)
        self.assertIn("export NCCL_NVLS_ENABLE=1", launcher_text)
        self.assertIn("export TORCH_SYMM_MEM_DISABLE_MULTICAST=0", launcher_text)
        self.assertIn("env -u LD_PRELOAD -u CUDA_VISIBLE_DEVICES setsid", launcher_text)
        self.assertIn("LD_PRELOAD", launcher_text)
        self.assertIn("${LD_PRELOAD:+${LD_PRELOAD}:}", launcher_text)

    # --- Cross-machine (GB300 NVL72 MNNVL) fabric path ------------------------

    def test_fabric_create_exports_handle_and_fetch_fabric_returns_it(self) -> None:
        # A FABRIC team spans the whole super-node: num_devices (8) may exceed the
        # holder's local device count (3). CREATE stores the exported 64-byte
        # handle; FETCH_FABRIC returns it in a sealed memfd over SCM_RIGHTS.
        created, create_fds = self.create(
            2 * 1024 * 1024,
            num_devices=8,
            handle_types=HANDLE_FABRIC_POSIX,
            owner_id=5,
            owner_generation=1,
        )
        os.close(self.assert_success_fd(created, create_fds))
        self.assertEqual(created["num_devices"], 8)
        self.assertEqual(created["handle_types"], HANDLE_FABRIC_POSIX)
        self.assertEqual(created["local_device_count"], 3)

        fetched, fetch_fds = self.fetch_fabric(created)
        memfd = self.assert_success_fd(fetched, fetch_fds)
        try:
            blob = os.pread(memfd, FABRIC_BYTES, 0)
        finally:
            os.close(memfd)
        self.assertEqual(blob, canned_fabric_handle(2 * 1024 * 1024))
        # FETCH_FABRIC resolves the existing object; it never forks a creator.
        self.assertEqual(
            self.counter_path.read_text(encoding="utf-8").splitlines(), ["2097152"]
        )

    def test_import_add_dedups_by_fabric_handle(self) -> None:
        handle = b"PEER-NODE-HANDLE".ljust(FABRIC_BYTES, b"\0")
        first, first_fds = self.import_add(handle, size=2 * 1024 * 1024)
        os.close(self.assert_success_fd(first, first_fds))
        self.assertEqual(self.entry_count(), 1)
        creator_calls = self.counter_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(creator_calls, ["2097152"])

        # Same fabric handle from a co-located rank dedups onto the one entry:
        # same object id, no new creator fork.
        again, again_fds = self.import_add(handle, size=2 * 1024 * 1024)
        os.close(self.assert_success_fd(again, again_fds))
        self.assertEqual(again["object_id"], first["object_id"])
        self.assertEqual(self.entry_count(), 1)
        self.assertEqual(
            self.counter_path.read_text(encoding="utf-8").splitlines(),
            ["2097152"],
        )

        # A different fabric handle is a distinct object (new importer fork).
        other_handle = b"OTHER-NODE-HANDLE".ljust(FABRIC_BYTES, b"\0")
        other, other_fds = self.import_add(other_handle, size=2 * 1024 * 1024)
        os.close(self.assert_success_fd(other, other_fds))
        self.assertNotEqual(other["object_id"], first["object_id"])
        self.assertEqual(self.entry_count(), 2)

    def test_import_add_same_handle_mismatched_properties_fails_closed(self) -> None:
        handle = b"CONFLICT-HANDLE".ljust(FABRIC_BYTES, b"\0")
        first, first_fds = self.import_add(handle, size=2 * 1024 * 1024)
        os.close(self.assert_success_fd(first, first_fds))
        # Same handle, different size -> PROPERTY_MISMATCH, object untouched.
        mismatch, mismatch_fds = self.import_add(handle, size=4 * 1024 * 1024)
        self.assertEqual(mismatch["status"], STATUS_PROPERTY_MISMATCH)
        self.assertEqual(mismatch_fds, [])
        self.assertEqual(self.entry_count(), 1)

    def test_unknown_size_is_restricted_to_peer_import_entries(self) -> None:
        rejected, rejected_fds = self.create(
            UNKNOWN_SIZE,
            num_devices=8,
            handle_types=HANDLE_FABRIC_POSIX,
        )
        self.assertEqual(rejected["status"], STATUS_INVALID_REQUEST)
        self.assertEqual(rejected_fds, [])
        self.assertFalse(self.counter_path.exists())

        handle = b"UNKNOWN-SIZE-PEER".ljust(FABRIC_BYTES, b"\0")
        imported, imported_fds = self.import_add(handle, size=UNKNOWN_SIZE)
        os.close(self.assert_success_fd(imported, imported_fds))
        self.assertEqual(imported["requested_size"], UNKNOWN_SIZE)

        fetched, fetched_fds = self.fetch(imported)
        os.close(self.assert_success_fd(fetched, fetched_fds))
        released, released_fds = self.release(imported, owner_id=21, owner_generation=1)
        self.assertEqual(released["status"], STATUS_OK)
        self.assertEqual(released_fds, [])
        self.assertEqual(self.entry_count(), 0)

    def test_unknown_size_import_adopts_existing_entry_properties(self) -> None:
        created, created_fds = self.create(
            2 * 1024 * 1024,
            num_devices=8,
            handle_types=HANDLE_FABRIC_POSIX,
            owner_id=11,
            owner_generation=1,
        )
        os.close(self.assert_success_fd(created, created_fds))

        imported, imported_fds = self.import_add(
            canned_fabric_handle(2 * 1024 * 1024),
            size=UNKNOWN_SIZE,
            handle_types=HANDLE_FABRIC,
            owner_id=21,
            owner_generation=1,
        )
        os.close(self.assert_success_fd(imported, imported_fds))
        self.assertEqual(imported["object_id"], created["object_id"])
        self.assertEqual(imported["requested_size"], created["requested_size"])
        self.assertEqual(imported["served_size"], created["served_size"])
        self.assertEqual(imported["handle_types"], HANDLE_FABRIC_POSIX)
        self.assertEqual(
            self.counter_path.read_text(encoding="utf-8").splitlines(), ["2097152"]
        )

    def test_fabric_requires_exact_configured_team_size(self) -> None:
        for num_devices in (2, 7, 9):
            response, fds = self.create(
                4096,
                num_devices=num_devices,
                handle_types=HANDLE_FABRIC_POSIX,
            )
            self.assertEqual(response["status"], STATUS_UNSUPPORTED_PROPERTIES)
            self.assertEqual(fds, [])
        self.assertFalse(self.counter_path.exists())

    def test_fabric_without_explicit_team_contract_fails_closed(self) -> None:
        self.stop_holder()
        self.assertTrue(wait_until(lambda: not self.socket_path.exists()))
        self.ready_path.unlink(missing_ok=True)
        self.process = self.start_holder(fabric_team_size=None)

        rejected, rejected_fds = self.create(
            4096, num_devices=8, handle_types=HANDLE_FABRIC_POSIX
        )
        self.assertEqual(rejected["status"], STATUS_UNSUPPORTED_PROPERTIES)
        self.assertEqual(rejected_fds, [])
        # The existing single-node POSIX contract remains compatible.
        created, created_fds = self.create(
            4096, num_devices=3, handle_types=HANDLE_POSIX
        )
        os.close(self.assert_success_fd(created, created_fds))

    def test_import_add_owner_refs_survive_restart_in_any_arrival_order(self) -> None:
        old_handle = b"OLD-TEAM".ljust(FABRIC_BYTES, b"\0")
        old_first, fds = self.import_add(old_handle, owner_id=11, owner_generation=1)
        os.close(self.assert_success_fd(old_first, fds))
        old_second, fds = self.import_add(old_handle, owner_id=22, owner_generation=1)
        os.close(self.assert_success_fd(old_second, fds))
        self.assertEqual(old_second["object_id"], old_first["object_id"])

        # The next backend generation arrives in the opposite order. Each
        # IMPORT_ADD reclaims only that owner's stale ref; the old entry closes
        # after both owners have moved, independent of the first importer.
        new_handle = b"NEW-TEAM".ljust(FABRIC_BYTES, b"\0")
        new_first, fds = self.import_add(new_handle, owner_id=22, owner_generation=2)
        os.close(self.assert_success_fd(new_first, fds))
        self.assertEqual(self.entry_count(), 2)
        new_second, fds = self.import_add(new_handle, owner_id=11, owner_generation=2)
        os.close(self.assert_success_fd(new_second, fds))
        self.assertEqual(new_second["object_id"], new_first["object_id"])
        self.assertEqual(self.entry_count(), 1)

        released, release_fds = self.release(new_first, owner_id=22, owner_generation=2)
        self.assertEqual(released["status"], STATUS_OK)
        self.assertEqual(release_fds, [])
        self.assertEqual(self.entry_count(), 1)
        released, release_fds = self.release(
            new_second, owner_id=11, owner_generation=2
        )
        self.assertEqual(released["status"], STATUS_OK)
        self.assertEqual(release_fds, [])
        self.assertEqual(self.entry_count(), 0)

    def test_import_add_registration_is_idempotent_per_owner_generation(self) -> None:
        handle = b"IDEMPOTENT".ljust(FABRIC_BYTES, b"\0")
        created, fds = self.import_add(handle, owner_id=31, owner_generation=4)
        os.close(self.assert_success_fd(created, fds))
        again, fds = self.import_add(handle, owner_id=31, owner_generation=4)
        os.close(self.assert_success_fd(again, fds))
        self.assertEqual(again["object_id"], created["object_id"])
        released, release_fds = self.release(created, owner_id=31, owner_generation=4)
        self.assertEqual(released["status"], STATUS_OK)
        self.assertEqual(release_fds, [])
        self.assertEqual(self.entry_count(), 0)

    def test_import_add_without_fabric_trailer_is_rejected(self) -> None:
        # IMPORT_ADD requires the 144-byte form; the 80-byte extended form (no
        # trailer) must fail closed and never fork a creator.
        response, fds = self.exchange(
            OP_IMPORT_ADD,
            size=4096,
            num_devices=8,
            handle_types=HANDLE_FABRIC_POSIX,
            owner_id=21,
            owner_generation=1,
        )
        self.assertEqual(response["status"], STATUS_INVALID_REQUEST)
        self.assertEqual(fds, [])
        self.assertFalse(self.counter_path.exists())

    def test_non_import_add_opcode_with_fabric_trailer_is_rejected(self) -> None:
        # Only IMPORT_ADD may carry the 64-byte trailer. A CREATE in the 144-byte
        # form must be rejected as an invalid request.
        response, fds = self.exchange(
            OP_CREATE,
            size=4096,
            num_devices=8,
            handle_types=HANDLE_FABRIC_POSIX,
            owner_id=21,
            owner_generation=1,
            fabric_handle=b"X".ljust(FABRIC_BYTES, b"\0"),
        )
        self.assertEqual(response["status"], STATUS_INVALID_REQUEST)
        self.assertEqual(fds, [])
        self.assertFalse(self.counter_path.exists())

    def test_fetch_fabric_on_posix_object_fails_closed(self) -> None:
        # A POSIX (single-node) object has no fabric handle; FETCH_FABRIC must not
        # fabricate one.
        created, fds = self.create(4096, handle_types=HANDLE_POSIX, num_devices=3)
        os.close(self.assert_success_fd(created, fds))
        response, fabric_fds = self.fetch_fabric(created)
        self.assertEqual(response["status"], STATUS_PROPERTY_MISMATCH)
        self.assertEqual(fabric_fds, [])


class SymmMemTransportSplitTest(unittest.TestCase):
    """Pure routing unit test for the Level3 multicast-vs-RDMA transport split."""

    def test_classify_symm_mem_transport(self) -> None:
        try:
            from rtp_llm.models_py.distributed.symm_mem import (
                SYMM_MEM_TRANSPORT_MULTICAST,
                SYMM_MEM_TRANSPORT_RDMA,
                classify_symm_mem_transport,
            )
        except Exception as error:  # noqa: BLE001 - torch may be unavailable on CPU
            self.skipTest(f"symm_mem import unavailable: {error}")
        # multicast_ptr == 0 -> RDMA-backed symm_mem: rebuilt on wake, no keeper.
        self.assertEqual(classify_symm_mem_transport(0), SYMM_MEM_TRANSPORT_RDMA)
        # multicast_ptr != 0 -> NVLS multicast: preserved by the keeper.
        self.assertEqual(
            classify_symm_mem_transport(0x7F00_0000), SYMM_MEM_TRANSPORT_MULTICAST
        )
        self.assertEqual(classify_symm_mem_transport(1), SYMM_MEM_TRANSPORT_MULTICAST)


if __name__ == "__main__":
    unittest.main()
