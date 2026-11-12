import fcntl
import json
import os
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import uuid
from unittest.mock import patch

from rtp_llm.resident_weights import store


def _memfd(content: bytes) -> int:
    # Stands in for an exported VMM FD; os.memfd_create is missing from this Python build.
    fd, path = tempfile.mkstemp(dir="/dev/shm")
    os.unlink(path)
    os.write(fd, content)
    return fd


def _read(fd: int) -> bytes:
    return os.pread(fd, 64, 0)


def _manifest(n: int) -> store.Manifest:
    return store.Manifest(allocation_sizes=[1] * n, storages=[], tensors=[])


def _record(fingerprint: str, n: int) -> store.SlotRecord:
    return store.SlotRecord(
        fingerprint=fingerprint,
        fingerprint_text=f"fingerprint: {fingerprint}",
        summary="test",
        total_bytes=n,
        publisher_pid=os.getpid(),
        manifest=_manifest(n),
    )


class TypedProtocolTest(unittest.TestCase):
    """The ledger crosses a JSON boundary; every field must survive it unchanged."""

    def test_manifest_and_record_round_trip_through_json(self):
        manifest = store.Manifest(
            allocation_sizes=[2 << 20, 4 << 20],
            storages=[
                store.StorageSpec(0, 0, 512, "ab12"),
                store.StorageSpec(1, 512, 64, "cd34"),
            ],
            tensors=[
                store.TensorSpec(
                    0,
                    "w",
                    store.WeightOrigin.LOADED,
                    0,
                    "float8_e4m3fn",
                    [128, 4],
                    [4, 1],
                    0,
                    False,
                ),
                store.TensorSpec(
                    "global",
                    "embed",
                    store.WeightOrigin.INIT,
                    1,
                    "bfloat16",
                    [8, 8],
                    [8, 1],
                    16,
                    True,
                ),
                store.TensorSpec(
                    3,
                    "empty",
                    store.WeightOrigin.LOADED,
                    -1,
                    "float32",
                    [0, 8],
                    [8, 1],
                    0,
                    False,
                ),
            ],
            transform_returned_tensors={0: {"moe.routed": True}, 3: {"qkv": False}},
            init_scalars={"global": {"block": 32}, 0: {"ratio": 1.5, "name": "x"}},
        )
        record = store.SlotRecord(
            fingerprint="sha",
            fingerprint_text="a: 1\nb: 2",
            summary="model",
            total_bytes=6 << 20,
            publisher_pid=os.getpid(),
            manifest=manifest,
        )
        back = store.SlotRecord.from_dict(json.loads(json.dumps(record.to_dict())))
        self.assertEqual(back, record)
        with self.assertRaises(store.StoreError):
            store.SlotRecord.from_dict(dict(record.to_dict(), source_digests={}))
        # Scope keys keep their type: a layer stays an int, the global scope stays a string.
        self.assertEqual([t.scope for t in back.manifest.tensors], [0, "global", 3])
        self.assertEqual(sorted(map(str, back.manifest.init_scalars)), ["0", "global"])
        # Origins come back as the enum, not as bare strings.
        self.assertIs(back.manifest.tensors[1].origin, store.WeightOrigin.INIT)

    def test_launcher_timeout_is_a_store_availability_error(self):
        with patch.object(
            store, "list_store", side_effect=ConnectionRefusedError
        ), patch.object(store, "open", create=True), patch.object(
            store.subprocess, "Popen"
        ) as launch:
            launch.return_value.wait.side_effect = subprocess.TimeoutExpired("store", 1)
            with self.assertRaisesRegex(store.StoreError, "launcher timed out"):
                store.ensure_running("timeout-test", timeout=1)

    def test_an_illegal_scope_encoding_raises_store_error(self):
        # A scope that is not "global" or "layer:<non-negative integer>" is a damaged record,
        # not an IndexError or a silently accepted prefix.
        for text in ("0", "layer:", "layer:abc", "layer:-1", "foo:3", "", 3, None):
            with self.subTest(scope=text):
                with self.assertRaises(store.StoreError):
                    store.decode_scope(text)
        self.assertEqual(store.decode_scope("layer:7"), 7)
        self.assertEqual(store.decode_scope("global"), "global")

    def test_a_damaged_record_raises_store_error(self):
        # The engine degrades to a cold load on StoreError; anything else crashes the start.
        good = _record("a", 1).to_dict()
        for mangle in (
            lambda data: data.pop("fingerprint"),
            lambda data: data.pop("manifest"),
            lambda data: data["manifest"].pop("storages"),
            lambda data: data["manifest"]["tensors"].append({"scope": "global"}),
            lambda data: data["manifest"]["transform_returned_tensors"].update(
                {"nonsense": {}}
            ),
            lambda data: data.update(manifest={"allocation_sizes": "not a list"}),
            lambda data: data.update(unexpected_field=1),
        ):
            with self.subTest(mangle=mangle):
                data = json.loads(json.dumps(good))
                mangle(data)
                with self.assertRaises(store.StoreError):
                    store.SlotRecord.from_dict(data)

    def test_an_unknown_op_raises_store_error(self):
        with self.assertRaises(store.StoreError):
            store.StoreRequest.from_message({"op": "nonsense_op"})


class StoreTest(unittest.TestCase):
    def setUp(self):
        self.name = f"rtp_llm_resident_test.{uuid.uuid4().hex}"
        store.ensure_running(self.name)

    def tearDown(self):
        try:
            store.stop_store(self.name)
        except ConnectionRefusedError:
            pass

    def _publish(self, slot_key, fingerprint, n):
        fds = [_memfd(f"allocation {i}".encode()) for i in range(n)]
        store.publish_slot(self.name, slot_key, _record(fingerprint, n), fds)
        for fd in fds:
            os.close(fd)

    def test_publish_and_attach_more_fds_than_one_message_carries(self):
        self._publish("gpu0/target", "a", 300)
        result = store.attach_slot(self.name, "gpu0/target", "a")
        self.assertIs(result.status, store.AttachStatus.HIT)
        self.assertEqual(len(result.allocation_fds), 300)
        self.assertEqual(
            [_read(fd) for fd in result.allocation_fds],
            [f"allocation {i}".encode() for i in range(300)],
        )
        self.assertEqual(result.record.manifest.allocation_sizes, [1] * 300)
        for fd in result.allocation_fds:
            os.close(fd)

    def test_fingerprint_mismatch_reports_fields_without_sending_the_record(self):
        self._publish("gpu0/target", "a", 1)
        result = store.attach_slot(self.name, "gpu0/target", "b")
        self.assertIs(result.status, store.AttachStatus.MISMATCH)
        self.assertEqual(result.published_fingerprint_text, "fingerprint: a")
        # A mismatched record is never decoded on the path that reports the mismatch.
        self.assertIsNone(result.record)
        self.assertEqual(result.allocation_fds, [])

    def test_publish_replaces_and_drop_removes(self):
        self._publish("gpu0/target", "a", 1)
        self._publish("gpu0/target", "b", 2)
        self._publish("gpu0/draft", "c", 1)
        result = store.attach_slot(self.name, "gpu0/target", "b")
        self.assertEqual(len(result.allocation_fds), 2)
        for fd in result.allocation_fds:
            os.close(fd)
        dropped = store.drop_slot(self.name, "gpu0/target")
        self.assertEqual(dropped.fingerprint, "b")
        self.assertIs(
            store.attach_slot(self.name, "gpu0/target", "b").status,
            store.AttachStatus.ABSENT,
        )
        slots = store.list_store(self.name).slots
        self.assertEqual(list(slots), ["gpu0/draft"])
        self.assertEqual(slots["gpu0/draft"].allocations, 1)

    def test_eplb_rearrangement_must_commit(self):
        self._publish("gpu0/target", "a", 1)
        store.eplb_begin(self.name, "gpu0/target")
        self.assertIs(
            store.attach_slot(self.name, "gpu0/target", "a").status,
            store.AttachStatus.EPLB_DIRTY,
        )
        placement = {"3": [1, 0, 2]}
        store.eplb_commit(self.name, "gpu0/target", placement)
        result = store.attach_slot(self.name, "gpu0/target", "a")
        self.assertEqual(result.record.eplb_placement, placement)
        for fd in result.allocation_fds:
            os.close(fd)

    def test_retained_file_survives_until_drop_all(self):
        with tempfile.NamedTemporaryFile(dir="/dev/shm", delete=False) as table:
            table.write(b"table")
        lock = table.name + ".lock"
        open(lock, "w").close()
        holder = os.open(table.name, os.O_RDONLY)
        fcntl.flock(holder, fcntl.LOCK_SH)
        self.assertTrue(store.retain_host_file(self.name, table.name, lock))
        # The last engine leaves: it would delete the table if nobody else held it.
        os.close(holder)
        probe = os.open(table.name, os.O_RDONLY)
        with self.assertRaises(BlockingIOError):
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.close(probe)
        store.drop_slot(self.name)
        self.assertFalse(os.path.exists(table.name))
        self.assertFalse(os.path.exists(lock))

    def test_drop_all_leaves_a_table_whose_creation_lock_is_held(self):
        # An engine holding the creation lock is building or releasing the table itself.
        with tempfile.NamedTemporaryFile(dir="/dev/shm", delete=False) as table:
            table.write(b"table")
        lock = table.name + ".lock"
        creator = os.open(lock, os.O_RDWR | os.O_CREAT, 0o600)
        fcntl.flock(creator, fcntl.LOCK_EX)
        store.retain_host_file(self.name, table.name, lock)
        store.drop_slot(self.name)
        self.assertTrue(os.path.exists(table.name))
        os.close(creator)
        os.unlink(table.name)
        os.unlink(lock)

    def test_publisher_killed_between_frames_leaves_no_fds_in_the_store(self):
        self.assertEqual(self._store_payload_fd_count(), 0)
        fds = [_memfd(b"x") for _ in range(store.MAX_FDS_PER_FRAME)]
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.connect(store._address(self.name))
            # The first of three frames, then the connection drops.
            header = {
                "proto": store.PROTOCOL,
                "op": store.StoreOp.PUBLISH.value,
                "slot_key": "gpu0/target",
            }
            store._send_frame(sock, dict(header, nfds=550), fds)
        for fd in fds:
            os.close(fd)
        # Served after the broken publish, so the store has handled it by now.
        store.list_store(self.name)
        self.assertEqual(self._store_payload_fd_count(), 0)

    def test_payload_fd_detector_counts_retained_payloads(self):
        self._publish("gpu0/target", "a", 1)
        self.assertEqual(self._store_payload_fd_count(), 1)

    def test_eplb_notifications_for_a_dropped_record_are_ignored(self):
        self._publish("gpu0/target", "a", 1)
        store.drop_slot(self.name)
        store.eplb_begin(self.name, "gpu0/target")
        store.eplb_commit(self.name, "gpu0/target", {})

    def _store_payload_fd_count(self) -> int:
        # _memfd uses deleted regular files in /dev/shm, not memfd_create. Exclude sockets
        # and other transient descriptors by checking both the path and descriptor type.
        for pid in os.listdir("/proc"):
            try:
                with open(f"/proc/{pid}/cmdline", "rb") as f:
                    cmdline = f.read().split(b"\0")
            except OSError:
                continue
            if b"serve" in cmdline and self.name.encode() in cmdline:
                fd_dir = f"/proc/{pid}/fd"
                count = 0
                for entry in os.listdir(fd_dir):
                    try:
                        fd_path = f"{fd_dir}/{entry}"
                        target = os.readlink(fd_path)
                        if target.startswith("/dev/shm/") and target.endswith(
                            " (deleted)"
                        ):
                            if stat.S_ISREG(os.stat(fd_path).st_mode):
                                count += 1
                    except OSError:
                        pass
                return count
        raise AssertionError("store process not found")

    def test_protocol_mismatch_and_unknown_op_are_reported_as_errors(self):
        # request() always sends this build's PROTOCOL and a known op; an older engine or a
        # corrupted client is simulated by sending the message directly.
        for message, expected in (
            ({"proto": store.PROTOCOL + 1, "op": "list"}, "protocol"),
            ({"proto": store.PROTOCOL, "op": "nonsense_op"}, "nonsense_op"),
        ):
            with self.subTest(op=message["op"]):
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                    sock.connect(store._address(self.name))
                    store.send_message(sock, message)
                    reply, fds = store.recv_message(sock)
                self.assertEqual(fds, [])
                self.assertRegex(reply["error"], expected)
        # A rejected request reaches a typed caller as StoreError, never as a bare RuntimeError.
        with self.assertRaises(store.StoreError):
            store.request(
                self.name, store.StoreRequest(store.StoreOp.PUBLISH, slot_key="gpu0/x")
            )

    def test_damaged_frame_raises_store_error(self):
        # Client side: a frame whose payload is not JSON must surface as StoreError, not as a
        # bare ValueError the caller would have to guess at. A socketpair stands in for a peer.
        near, far = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        with near, far:
            payload = b"{not json"
            far.sendall(store._HEADER.pack(len(payload)) + payload)
            with self.assertRaises(store.StoreError):
                store.recv_message(near)

    def test_store_survives_a_damaged_frame_and_keeps_serving(self):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.connect(store._address(self.name))
            payload = b"{not json"
            sock.sendall(store._HEADER.pack(len(payload)) + payload)
        self.assertEqual(store.list_store(self.name).slots, {})

    def test_stop_ends_the_store(self):
        store.stop_store(self.name)
        with self.assertRaises(ConnectionRefusedError):
            store.list_store(self.name)

    def test_store_is_not_a_process_descendant(self):
        store_pid = self._store_pid()
        pid = store_pid
        while pid > 1:
            with open(f"/proc/{pid}/stat") as process:
                pid = int(process.read().split()[3])
            self.assertNotEqual(pid, os.getpid())

    def test_owner_death_stops_test_store(self):
        owner = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        name = f"rtp_llm_resident_owner.{uuid.uuid4().hex}"
        try:
            store.ensure_running(name, owner_pid=owner.pid)
        finally:
            owner.terminate()
            owner.wait(timeout=5)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                store.list_store(name)
            except ConnectionRefusedError:
                break
            time.sleep(0.1)
        else:
            store.stop_store(name)
            self.fail("test store remained alive after its owner exited")

    def _store_pid(self) -> int:
        for pid in os.listdir("/proc"):
            try:
                with open(f"/proc/{pid}/cmdline", "rb") as process:
                    cmdline = process.read().split(b"\0")
            except OSError:
                continue
            if b"serve" in cmdline and self.name.encode() in cmdline:
                return int(pid)
        raise AssertionError("store process not found")


class ServerTimeoutTest(unittest.TestCase):
    """A client that connects and then hangs must not block the store for everyone else."""

    def test_hung_client_times_out_and_the_store_keeps_serving(self):
        name = f"rtp_llm_resident_test.{uuid.uuid4().hex}"
        server = threading.Thread(target=store.serve, args=(name, 0.2), daemon=True)
        server.start()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                store.list_store(name)
                break
            except ConnectionRefusedError:
                time.sleep(0.05)
        else:
            self.fail("store did not start")
        hung = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        hung.connect(store._address(name))
        # Announce a payload and never send it: without a timeout the store would block here
        # forever, and every other GPU on this host would fail to acquire its weights.
        hung.sendall(store._HEADER.pack(4096))
        try:
            self.assertEqual(store.list_store(name).slots, {})
        finally:
            hung.close()
            store.stop_store(name)
            server.join(timeout=10)


if __name__ == "__main__":
    unittest.main()
