import fcntl
import json
import os
import resource
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import uuid
from unittest.mock import MagicMock, patch

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
    return store.Manifest(
        allocations=[store.AllocationSpec(store.MemoryKind.CUDA, 1)] * n,
        storages=[],
        tensors=[],
    )


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
            allocations=[
                store.AllocationSpec(store.MemoryKind.CUDA, 2 << 20),
                store.AllocationSpec(store.MemoryKind.CPU, 4 << 20, True),
            ],
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
                    store.MemoryKind.CUDA,
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
                    store.MemoryKind.CPU,
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
                    store.MemoryKind.CPU,
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
        self.assertIs(back.manifest.allocations[1].kind, store.MemoryKind.CPU)
        self.assertTrue(back.manifest.allocations[1].cuda_access_required)
        self.assertIs(back.manifest.tensors[2].device, store.MemoryKind.CPU)
        with self.assertRaises(store.StoreError):
            store.Manifest.from_dict(
                {"allocation_sizes": [], "storages": [], "tensors": []}
            )
        # Scope keys keep their type: a layer stays an int, the global scope stays a string.
        self.assertEqual([t.scope for t in back.manifest.tensors], [0, "global", 3])
        self.assertEqual(sorted(map(str, back.manifest.init_scalars)), ["0", "global"])
        # Origins come back as the enum, not as bare strings.
        self.assertIs(back.manifest.tensors[1].origin, store.WeightOrigin.INIT)

    def test_launcher_timeout_is_a_store_availability_error(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            store, "list_store", side_effect=ConnectionRefusedError
        ), patch.object(store.tempfile, "tempdir", directory), patch.object(
            store.subprocess, "Popen"
        ) as launch:
            launch.return_value.wait.side_effect = subprocess.TimeoutExpired("store", 1)
            with self.assertRaisesRegex(store.StoreError, "launcher timed out"):
                store.ensure_running("timeout-test", timeout=1)
            logs = os.listdir(directory)
            self.assertEqual(len(logs), 1)
            log = os.path.join(directory, logs[0])
            self.assertTrue(logs[0].startswith("rtp-resident-store."))
            self.assertFalse(os.path.islink(log))
            self.assertEqual(stat.S_IMODE(os.stat(log).st_mode), 0o600)

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
            lambda data: data.update(manifest={"allocations": "not a list"}),
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

    def test_client_rejects_a_foreign_uid_before_sending_any_message_or_fd(self):
        fd = _memfd(b"private weights")
        try:
            with patch.object(
                store, "_peer_uid", return_value=os.getuid() + 1
            ), patch.object(store, "send_message") as send:
                with self.assertRaisesRegex(store.StoreError, "peer UID"):
                    store.publish_slot(self.name, "gpu0/target", _record("a", 1), [fd])
                send.assert_not_called()
            self.assertEqual(_read(fd), b"private weights")
            self.assertEqual(store.list_store(self.name).slots, {})
        finally:
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
        self.assertEqual(
            [a.size_bytes for a in result.record.manifest.allocations], [1] * 300
        )
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


class IdleMoveTest(unittest.TestCase):
    """Idle records move to host memory and back, or are dropped; a stand-in mover copies bytes."""

    def _start(
        self,
        offload_after_s=1.0,
        idle_ttl_s=store.DISABLED_TIMER,
        move_timeout_s=store.DEFAULT_MOVE_TIMEOUT_S,
        host_reserve_ratio=0.0,
        host_reserve_min_gb=0.0,
        **env,
    ):
        self.name = f"rtp_llm_resident_test.{uuid.uuid4().hex}"
        mover = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "fake_mover.py"
        )
        with patch.dict(os.environ, {store.MOVER_ENV: mover, **env}):
            store.ensure_running(
                self.name,
                settings=store.StoreSettings(
                    offload_after_s=offload_after_s,
                    idle_ttl_s=idle_ttl_s,
                    move_timeout_s=move_timeout_s,
                    host_reserve_ratio=host_reserve_ratio,
                    host_reserve_min_gb=host_reserve_min_gb,
                ),
            )
        self.addCleanup(store.stop_store, self.name)

    def _publish(self, n=3, slot="gpu0/target", fp="a", hold=False):
        fds = [_memfd(f"allocation {i}".encode()) for i in range(n)]
        sock = store.publish_slot(self.name, slot, _record(fp, n), fds, hold=hold)
        for fd in fds:
            os.close(fd)
        return sock

    def _slot(self, slot="gpu0/target"):
        return store.list_store(self.name).slots.get(slot)

    def _wait(self, condition, what, timeout=15.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            value = condition()
            if value:
                return value
            time.sleep(0.1)
        self.fail(f"timed out waiting for {what}: {self._slot()}")

    def _attach_until(self, status, slot="gpu0/target", fp="a", hold=False):
        return self._wait(
            lambda: (lambda r: r if r.status is status else _drop_fds(r))(
                store.attach_slot(self.name, slot, fp, hold=hold)
            ),
            status.value,
        )

    def test_a_held_record_stays_and_an_idle_one_round_trips_through_host_memory(self):
        self._start()
        hold = self._publish(hold=True)
        time.sleep(2.5)
        self.assertFalse(self._slot().on_host)
        self.assertEqual(self._slot().holders, 1)
        hold.close()
        self._wait(lambda: self._slot().on_host, "offload")
        first = store.attach_slot(self.name, "gpu0/target", "a", hold=True)
        self.assertIs(first.status, store.AttachStatus.MOVING)
        self.assertIsNone(first.hold_sock)
        hit = self._attach_until(store.AttachStatus.HIT, hold=True)
        self.assertIsNotNone(hit.hold_sock)
        self.assertEqual(self._slot().holders, 1)
        self.assertEqual(
            [_read(fd) for fd in hit.allocation_fds],
            [f"allocation {i}".encode() for i in range(3)],
        )
        hit.hold_sock.close()
        _drop_fds(hit)
        self.assertFalse(self._slot().on_host)

    def test_replacing_a_slot_closes_the_previous_generations_hold(self):
        self._start(offload_after_s=-1.0)
        old_hold = self._publish(fp="gen1", hold=True)
        self.addCleanup(old_hold.close)
        new_hold = self._publish(fp="gen2", hold=True)
        self.addCleanup(new_hold.close)
        self.assertEqual(self._slot().holders, 1)
        # The previous generation's hold socket received EOF from the store.
        old_hold.settimeout(2.0)
        self.assertEqual(old_hold.recv(1), b"")
        new_hold.close()
        self._wait(lambda: self._slot().holders == 0, "new hold released")

    def test_an_attach_during_offload_cancels_it_and_gets_the_gpu_copy(self):
        self._start(FAKE_MOVER_DELAY_S="30")
        self._publish()
        self._wait(lambda: self._slot().moving == "to_host", "offload start")
        hit = store.attach_slot(self.name, "gpu0/target", "a")
        self.assertIs(hit.status, store.AttachStatus.HIT)
        _drop_fds(hit)
        slot = self._slot()
        self.assertFalse(slot.on_host)
        self.assertIsNone(slot.moving)

    def test_an_idle_record_is_dropped_after_the_ttl(self):
        self._start(offload_after_s=-1.0, idle_ttl_s=1.5)
        self._publish()
        self._wait(lambda: self._slot() is None, "idle drop")

    def test_a_failed_offload_drops_the_record_so_the_gpu_is_released(self):
        self._start(FAKE_MOVER_FAIL="to_host")
        self._publish()
        self._wait(lambda: self._slot() is None, "drop after failed offload")

    def test_a_failed_restore_is_reported_once_and_the_record_stays_in_host_memory(
        self,
    ):
        self._start(FAKE_MOVER_FAIL="to_gpu")
        self._publish()
        self._wait(lambda: self._slot().on_host, "offload")
        failed = self._attach_until(store.AttachStatus.RESTORE_FAILED)
        self.assertIn("injected to_gpu failure", failed.restore_error)
        self.assertTrue(self._slot().on_host)
        # Reported once: the next start tries again.
        again = store.attach_slot(self.name, "gpu0/target", "a")
        self.assertIs(again.status, store.AttachStatus.MOVING)

    def test_a_killed_holder_releases_its_hold(self):
        self._start(offload_after_s=-1.0)
        self._publish()
        holder = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import sys, time; from rtp_llm.resident_weights import store; "
                "res = store.attach_slot(sys.argv[1], 'gpu0/target', 'a', hold=True); "
                "print('held', flush=True); time.sleep(600)",
                self.name,
            ],
            stdout=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(holder.stdout.readline().strip(), "held")
        self.assertEqual(self._slot().holders, 1)
        holder.kill()
        holder.wait()
        self._wait(lambda: self._slot().holders == 0, "hold released")
        self.assertIsNotNone(self._slot().idle_s)

    def test_store_serves_above_1024_fds(self):
        # Two DSV4.1 targets on one host hold >1078 allocation FDs, pushing holder/mover
        # socket FDs past Linux FD_SETSIZE (1024) where select() raises ValueError.
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if hard < 2048:
            self.skipTest(f"RLIMIT_NOFILE hard limit {hard} < 2048")
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
        self._start(offload_after_s=-1.0)
        # Publish 1100 allocations across two slots (550 each, mirroring dual-GPU DSV4.1).
        h0 = self._publish(n=550, slot="gpu0/target", fp="a", hold=True)
        self.addCleanup(h0.close)
        h1 = self._publish(n=550, slot="gpu1/target", fp="b", hold=True)
        self.addCleanup(h1.close)
        slots = store.list_store(self.name).slots
        self.assertEqual(slots["gpu0/target"].allocations, 550)
        self.assertEqual(slots["gpu1/target"].allocations, 550)
        self.assertEqual(slots["gpu1/target"].holders, 1)
        h1.close()
        self._wait(lambda: self._slot("gpu1/target").holders == 0, "high-fd hold EOF")
        hit = store.attach_slot(self.name, "gpu1/target", "b", hold=True)
        self.assertIs(hit.status, store.AttachStatus.HIT)
        self.assertEqual(len(hit.allocation_fds), 550)
        hit.hold_sock.close()
        _drop_fds(hit)
        store.drop_slot(self.name, "gpu1/target")
        self.assertIsNone(self._slot("gpu1/target"))

    def test_mover_exiting_before_reading_job_does_not_crash_store(self):
        # Early exit triggers BrokenPipeError in _start_mover: the store must drop the
        # un-offloadable slot and keep serving other slots.
        self._start(offload_after_s=1.0, FAKE_MOVER_EXIT_EARLY="1")
        keep = self._publish(slot="gpu1/target", fp="keep", hold=True)
        self.addCleanup(keep.close)
        self._publish(slot="gpu0/target", fp="drop")
        self._wait(lambda: self._slot("gpu0/target") is None, "early-exit drop")
        self.assertIsNotNone(self._slot("gpu1/target"))

    def test_half_frame_mover_does_not_hang_store(self):
        # Run serve() in-process with a short mover_io_timeout_s so a half-frame reply times
        # out quickly instead of wedging the store.
        name = f"rtp_llm_resident_test.{uuid.uuid4().hex}"
        mover = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "fake_mover.py"
        )
        with patch.dict(
            os.environ, {store.MOVER_ENV: mover, "FAKE_MOVER_HALF_FRAME": "1"}
        ):
            server = threading.Thread(
                target=store.serve,
                args=(name,),
                kwargs={
                    "settings": store.StoreSettings(
                        offload_after_s=0.5, host_reserve_ratio=0.0
                    ),
                    "mover_io_timeout_s": 0.5,
                },
                daemon=True,
            )
            server.start()
            self.addCleanup(lambda: (store.stop_store(name), server.join(timeout=10)))
            self._wait(lambda: _try_list(name) is not None, "in-process store start")
            fds = [_memfd(b"x")]
            store.publish_slot(name, "gpu0/target", _record("a", 1), fds)
            os.close(fds[0])
            self._wait(
                lambda: store.list_store(name).slots.get("gpu0/target") is None,
                "half-frame mover timeout drop",
            )
            self.assertEqual(store.list_store(name).slots, {})

    def test_stalled_restore_times_out_in_store_and_keeps_host_copy(self):
        self._start(
            offload_after_s=0.5,
            move_timeout_s=2.0,
            FAKE_MOVER_DELAY_DIR="to_gpu",
            FAKE_MOVER_DELAY_S="30",
        )
        self._publish()
        self._wait(lambda: self._slot().on_host, "offload")
        first = store.attach_slot(self.name, "gpu0/target", "a")
        self.assertIs(first.status, store.AttachStatus.MOVING)
        failed = self._attach_until(store.AttachStatus.RESTORE_FAILED)
        self.assertIn("timed out", failed.restore_error)
        self.assertTrue(self._slot().on_host)
        self.assertIsNone(self._slot().moving)

    def test_stalled_offload_times_out_in_store_and_drops_slot(self):
        self._start(
            offload_after_s=0.5,
            move_timeout_s=2.0,
            FAKE_MOVER_DELAY_DIR="to_host",
            FAKE_MOVER_DELAY_S="30",
        )
        self._publish()
        self._wait(
            lambda: self._slot() is None, "stalled offload timed out and dropped"
        )

    def test_concurrent_offloads_serialize_one_at_a_time(self):
        self._start(offload_after_s=0.5, FAKE_MOVER_DELAY_S="1.5")
        self._publish(slot="gpu0/target", fp="a")
        self._publish(slot="gpu1/target", fp="b")
        self._wait(
            lambda: sum(
                1
                for s in store.list_store(self.name).slots.values()
                if s.moving == "to_host"
            )
            >= 1,
            "first offload starts",
        )
        # While one offload is in flight, the second slot must wait rather than double-commit RAM.
        slots = store.list_store(self.name).slots
        self.assertEqual(sum(1 for s in slots.values() if s.moving == "to_host"), 1)
        self._wait(
            lambda: all(s.on_host for s in store.list_store(self.name).slots.values()),
            "both offloads complete serially",
        )

    def test_config_updates_settings_live_and_rejects_invalid_values(self):
        self._start(offload_after_s=-1.0, idle_ttl_s=-1.0)
        hold = self._publish(slot="gpu0/target", fp="held", hold=True)
        self.addCleanup(hold.close)
        self._publish(slot="gpu1/target", fp="idle")
        # Updating live while another slot is held must work and leave the held slot alone.
        updated = store.configure_store(
            self.name,
            offload_after_s=0.5,
            idle_ttl_s=3600.0,
            move_timeout_s=120.0,
            host_reserve_ratio=0.0,
            host_reserve_min_gb=0.0,
        )
        self.assertEqual(updated.offload_after_s, 0.5)
        self.assertEqual(updated.idle_ttl_s, 3600.0)
        self.assertEqual(updated.move_timeout_s, 120.0)
        self._wait(lambda: self._slot("gpu1/target").on_host, "live config offload")
        self.assertFalse(self._slot("gpu0/target").on_host)
        # 0 is rejected (not a disable sentinel); settings stay untouched.
        with self.assertRaises(store.StoreError):
            store.configure_store(self.name, offload_after_s=0.0)
        with self.assertRaises(store.StoreError):
            store.configure_store(self.name, host_reserve_ratio=1.0)
        self.assertEqual(store.list_store(self.name).settings.offload_after_s, 0.5)

    def test_manual_offload_moves_idle_slots_serially_and_skips_held_slots(self):
        self._start(offload_after_s=-1.0, FAKE_MOVER_DELAY_S="0.5")
        held = self._publish(slot="gpu0/target", fp="held", hold=True)
        self.addCleanup(held.close)
        self._publish(slot="gpu1/target", fp="a")
        self._publish(slot="gpu2/target", fp="b")
        reply = store.offload_slots(self.name)
        self.assertIn("in use", reply.affected["gpu0/target"])
        self.assertEqual(reply.affected["gpu1/target"], "moving to host memory")
        self.assertEqual(reply.affected["gpu2/target"], "queued for host offload")
        self._wait(
            lambda: self._slot("gpu1/target").on_host
            and self._slot("gpu2/target").on_host,
            "manual serial offload",
        )
        self.assertFalse(self._slot("gpu0/target").on_host)

    def test_manual_offload_refuses_and_keeps_gpu_copy_on_low_memory_or_failure(self):
        # Impossibly high host_reserve_min_gb guarantees refusal without dropping the GPU record.
        self._start(offload_after_s=-1.0, host_reserve_min_gb=1e9)
        self._publish(slot="gpu0/target", fp="a")
        reply = store.offload_slots(self.name, "gpu0/target")
        self.assertTrue(reply.affected["gpu0/target"].startswith("refused:"))
        slot = self._slot("gpu0/target")
        self.assertIsNotNone(slot)
        self.assertFalse(slot.on_host)

    def test_manual_offload_mover_failure_keeps_gpu_copy(self):
        self._start(offload_after_s=-1.0, FAKE_MOVER_FAIL="to_host")
        self._publish(slot="gpu0/target", fp="a")
        reply = store.offload_slots(self.name, "gpu0/target")
        self.assertEqual(reply.affected["gpu0/target"], "moving to host memory")
        self._wait(lambda: self._slot("gpu0/target").moving is None, "mover finished")
        slot = self._slot("gpu0/target")
        self.assertIsNotNone(slot)
        self.assertFalse(slot.on_host)

    def test_keepalive_resets_current_stage_timer_without_accumulating(self):
        # GPU stage (offload_after_s=2.0) and host stage (idle_ttl_s=2.0) are decoupled:
        # keepalive resets whichever stage's clock is active, never stacking unbounded time.
        self._start(offload_after_s=2.0, idle_ttl_s=2.0)
        self._publish(slot="gpu0/target", fp="a")
        time.sleep(1.2)
        # Still on GPU: keepalive resets the GPU offload countdown back to 2.0 s.
        r1 = store.keepalive_slots(self.name, "gpu0/target")
        self.assertIn(
            "reset GPU idle timer (offloads in 2 s)", r1.affected["gpu0/target"]
        )
        # Calling keepalive twice right away does NOT stack to 4.0 s.
        r2 = store.keepalive_slots(self.name, "gpu0/target")
        self.assertIn(
            "reset GPU idle timer (offloads in 2 s)", r2.affected["gpu0/target"]
        )
        time.sleep(1.1)
        # Past 2.3 s total since publish, still on GPU thanks to the reset at 1.2 s.
        self.assertFalse(self._slot("gpu0/target").on_host)
        # Once 2.0 s passes without another keepalive, it offloads to host and starts TTL fresh.
        self._wait(lambda: self._slot("gpu0/target").on_host, "offload after reset")
        time.sleep(1.1)
        # Now in host memory: keepalive resets the host TTL countdown back to 2.0 s.
        r3 = store.keepalive_slots(self.name, "gpu0/target")
        self.assertIn(
            "reset host idle timer (expires in 2 s)", r3.affected["gpu0/target"]
        )
        time.sleep(1.2)
        # Still alive in host memory (>2.3 s after landing on host).
        self.assertIsNotNone(self._slot("gpu0/target"))
        # Without further keepalive, it expires and drops after 2.0 s.
        self._wait(lambda: self._slot("gpu0/target") is None, "host TTL drop")

    def test_decoupled_ttl_does_not_preempt_gpu_offload_and_direct_gpu_ttl_works(self):
        # Even when idle_ttl_s < offload_after_s, GPU stage runs offload_after_s first and
        # only starts idle_ttl_s once the record arrives in host memory.
        s = store._Store(
            store.StoreSettings(
                offload_after_s=10.0, idle_ttl_s=3.0, host_reserve_ratio=0.0
            )
        )
        s._publish("gpu0/target", _record("a", 1), [_memfd(b"x")])
        s.slots["gpu0/target"].idle_since = time.monotonic() - 5.0
        s.tick(time.monotonic())
        # 5 s > idle_ttl_s (3 s), but < offload_after_s (10 s) -> stays on GPU, NOT dropped.
        self.assertIn("gpu0/target", s.slots)
        self.assertFalse(s.slots["gpu0/target"].on_host)

        # When offload is disabled (-1), idle_ttl_s drops directly from GPU and keepalive resets it.
        s2 = store._Store(store.StoreSettings(offload_after_s=-1.0, idle_ttl_s=5.0))
        s2._publish("gpu0/target", _record("a", 1), [_memfd(b"y")])
        s2.slots["gpu0/target"].idle_since = time.monotonic() - 4.0
        r, _ = s2._keepalive(
            store.StoreRequest(store.StoreOp.KEEPALIVE, slot_key="gpu0/target")
        )
        self.assertIn(
            "reset GPU idle timer (expires in 5 s)", r.affected["gpu0/target"]
        )
        s2.tick(time.monotonic())
        self.assertIn("gpu0/target", s2.slots)
        s2.slots["gpu0/target"].idle_since = time.monotonic() - 6.0
        s2.tick(time.monotonic())
        self.assertNotIn("gpu0/target", s2.slots)

        # Keepalive resets the automatic timer without cancelling an explicit queued offload.
        s.slots["gpu0/target"].offload_requested = True
        s._keepalive(
            store.StoreRequest(store.StoreOp.KEEPALIVE, slot_key="gpu0/target")
        )
        self.assertTrue(s.slots["gpu0/target"].offload_requested)

    def test_conditional_drop_does_not_delete_newer_generation(self):
        self._start(offload_after_s=-1.0)
        self._publish(slot="gpu0/target", fp="gen1")
        mismatch = store.attach_slot(self.name, "gpu0/target", "other")
        self.assertIs(mismatch.status, store.AttachStatus.MISMATCH)
        self.assertTrue(mismatch.generation)
        # Another process replaces the slot before the first caller issues its stale drop.
        self._publish(slot="gpu0/target", fp="gen2")
        dropped = store.drop_slot(
            self.name, "gpu0/target", expected_generation=mismatch.generation
        )
        self.assertIsNone(dropped)
        # Newer generation is still intact.
        hit = store.attach_slot(self.name, "gpu0/target", "gen2")
        self.assertIs(hit.status, store.AttachStatus.HIT)
        _drop_fds(hit)


class StoreStateMachineEdgeTest(unittest.TestCase):
    """Isolated _Store state-machine tests for failure, deadline, and reaping edge cases."""

    def _make_store(self, **kwargs):
        return store._Store(store.StoreSettings(**kwargs))

    def _publish(self, s, slot="gpu0/target", fp="a"):
        fd = _memfd(b"payload")
        s._publish(slot, _record(fp, 1), [fd])

    def test_queued_manual_offload_crossing_auto_timer_keeps_gpu_on_refusal(self):
        gib = 1 << 30
        s = self._make_store(
            offload_after_s=1.0, host_reserve_ratio=0.2, host_reserve_min_gb=0.0
        )
        self._publish(s, "gpu0/target", "a")
        self._publish(s, "gpu1/target", "b")
        # Pretend gpu0 has an active to_host mover; manual offload on gpu1 queues behind it.
        dummy_proc = MagicMock(poll=MagicMock(return_value=None))
        dummy_sock = MagicMock()
        s.slots["gpu0/target"].mover = store._Mover(
            to_host=True,
            process=dummy_proc,
            sock=dummy_sock,
            deadline=time.monotonic() + 60.0,
        )
        reply, _ = s._offload(
            store.StoreRequest(store.StoreOp.OFFLOAD, slot_key="gpu1/target")
        )
        self.assertEqual(reply.affected["gpu1/target"], "queued for host offload")
        # Finish gpu0 mover and advance gpu1 past offload_after_s while host RAM is too low.
        s.slots["gpu0/target"].mover = None
        s.slots["gpu1/target"].idle_since = time.monotonic() - 10.0
        with patch.object(store, "_host_memory", return_value=(100 * gib, 10 * gib)):
            s.tick(time.monotonic())
        # Queued manual offload must refuse and preserve gpu1 (and defer immediate auto drop).
        self.assertIn("gpu1/target", s.slots)
        self.assertFalse(s.slots["gpu1/target"].on_host)
        self.assertIsNotNone(s.slots["gpu1/target"].last_offload_error)

    def test_drop_during_restore_holds_attach_in_moving_until_mover_exits(self):
        s = self._make_store()
        self._publish(s, "gpu0/target", "a")
        s.slots["gpu0/target"].on_host = True
        exiting = MagicMock()
        exiting.poll.return_value = None
        s.slots["gpu0/target"].mover = store._Mover(
            to_host=False,
            process=exiting,
            sock=MagicMock(),
            deadline=time.monotonic() + 60.0,
        )
        s._release("gpu0/target", "dropped while restoring")
        # Slot is gone, but killed to_gpu mover is still alive -> attach must wait (MOVING).
        reply, _ = s._attach(
            store.StoreRequest(
                store.StoreOp.ATTACH, slot_key="gpu0/target", fingerprint="a"
            )
        )
        self.assertIs(reply.status, store.AttachStatus.MOVING)
        # Once the killed mover exits, attach cleanly reports ABSENT.
        exiting.poll.return_value = -9
        reply, _ = s._attach(
            store.StoreRequest(
                store.StoreOp.ATTACH, slot_key="gpu0/target", fingerprint="a"
            )
        )
        self.assertIs(reply.status, store.AttachStatus.ABSENT)

    def test_reaping_offload_mover_holds_serial_offload_slot(self):
        s = self._make_store(offload_after_s=1.0, host_reserve_ratio=0.0)
        exiting = MagicMock()
        exiting.poll.return_value = None
        s._track_reaping(exiting, to_host=True, slot_key="gpu0/target", grace_s=10.0)
        self.assertTrue(s._has_active_offload())
        exiting.poll.return_value = 0
        self.assertFalse(s._has_active_offload())

    def test_socketpair_emfile_and_popen_enomem_are_contained_per_slot(self):
        s = self._make_store(offload_after_s=-1.0, host_reserve_ratio=0.0)
        self._publish(s, "gpu0/target", "a")
        self._publish(s, "gpu1/target", "b")
        with patch.object(
            socket, "socketpair", side_effect=OSError(24, "Too many open files")
        ):
            reply, _ = s._offload(
                store.StoreRequest(store.StoreOp.OFFLOAD, slot_key="gpu0/target")
            )
        self.assertTrue(reply.affected["gpu0/target"].startswith("refused:"))
        self.assertIn("gpu0/target", s.slots)
        self.assertIn("gpu1/target", s.slots)
        with patch.object(
            subprocess, "Popen", side_effect=OSError(12, "Cannot allocate memory")
        ):
            reply, _ = s._offload(
                store.StoreRequest(store.StoreOp.OFFLOAD, slot_key="gpu0/target")
            )
        self.assertTrue(reply.affected["gpu0/target"].startswith("refused:"))
        listed, _ = s._list(store.StoreRequest(store.StoreOp.LIST))
        self.assertIn(
            "Cannot allocate memory", listed.slots["gpu0/target"].last_offload_error
        )

    def test_extreme_host_reserve_min_gb_rejected_before_overflow(self):
        with self.assertRaises(store.StoreError):
            store.validate_settings(store.StoreSettings(host_reserve_min_gb=1e308))

    def test_mover_reply_arriving_past_deadline_is_rejected(self):
        s = self._make_store()
        self._publish(s, "gpu0/target", "a")
        ours, theirs = socket.socketpair()
        self.addCleanup(theirs.close)
        proc = MagicMock(poll=MagicMock(return_value=0))
        s.slots["gpu0/target"].mover = store._Mover(
            to_host=True,
            process=proc,
            sock=ours,
            deadline=time.monotonic() - 1.0,
            timeout_s=5.0,
            drop_on_failure=False,
        )
        s.mover_ready(ours)
        self.assertFalse(s.slots["gpu0/target"].on_host)
        self.assertIn("timed out after 5 s", s.slots["gpu0/target"].last_offload_error)

    def test_mismatch_during_active_restore_aborts_mover_and_waits_for_exit(self):
        s = self._make_store()
        self._publish(s, "gpu0/target", "old_fp")
        s.slots["gpu0/target"].on_host = True
        exiting = MagicMock()
        exiting.poll.return_value = None
        s.slots["gpu0/target"].mover = store._Mover(
            to_host=False,
            process=exiting,
            sock=MagicMock(),
            deadline=time.monotonic() + 60.0,
        )
        # A mismatched attach must abort the live to_gpu mover and return MOVING until exit.
        reply, _ = s._attach(
            store.StoreRequest(
                store.StoreOp.ATTACH, slot_key="gpu0/target", fingerprint="new_fp"
            )
        )
        self.assertIs(reply.status, store.AttachStatus.MOVING)
        exiting.kill.assert_called_once()
        exiting.poll.return_value = -9
        reply, _ = s._attach(
            store.StoreRequest(
                store.StoreOp.ATTACH, slot_key="gpu0/target", fingerprint="new_fp"
            )
        )
        self.assertIs(reply.status, store.AttachStatus.MISMATCH)

    def test_restarted_store_has_distinct_generation_tokens(self):
        s1 = self._make_store()
        s2 = self._make_store()
        g1 = s1._publish("gpu0/target", _record("a", 1), [_memfd(b"1")])
        g2 = s2._publish("gpu0/target", _record("a", 1), [_memfd(b"2")])
        self.assertNotEqual(g1, g2)
        reply, _ = s2._drop(
            store.StoreRequest(
                store.StoreOp.DROP, slot_key="gpu0/target", expected_generation=g1
            )
        )
        self.assertIsNone(reply.record)
        self.assertIn("gpu0/target", s2.slots)

    def test_recv_message_enforces_total_deadline_across_partial_chunks(self):
        left, right = socket.socketpair()
        self.addCleanup(left.close)
        self.addCleanup(right.close)
        payload = b'{"ok":true}'
        header = store._HEADER.pack(len(payload))

        def slow_sender():
            try:
                right.sendall(header)
                for b in payload:
                    time.sleep(0.03)
                    right.sendall(bytes([b]))
            except OSError:
                pass

        t = threading.Thread(target=slow_sender, daemon=True)
        t.start()
        with self.assertRaises((TimeoutError, socket.timeout)):
            store.recv_message(left, deadline=time.monotonic() + 0.08)
        t.join(timeout=2)

    def test_drop_during_to_host_waits_for_mover_exit_before_absent(self):
        s = self._make_store()
        self._publish(s, "gpu0/target", "a")
        exiting = MagicMock()
        exiting.poll.return_value = None
        s.slots["gpu0/target"].mover = store._Mover(
            to_host=True,
            process=exiting,
            sock=MagicMock(),
            deadline=time.monotonic() + 60.0,
        )
        s._release("gpu0/target", "dropped during offload")
        # Source VMM FDs stay pinned until the killed to_host mover exits.
        reply, _ = s._attach(
            store.StoreRequest(
                store.StoreOp.ATTACH, slot_key="gpu0/target", fingerprint="a"
            )
        )
        self.assertIs(reply.status, store.AttachStatus.MOVING)
        exiting.poll.return_value = -9
        reply, _ = s._attach(
            store.StoreRequest(
                store.StoreOp.ATTACH, slot_key="gpu0/target", fingerprint="a"
            )
        )
        self.assertIs(reply.status, store.AttachStatus.ABSENT)


class HostMemoryTest(unittest.TestCase):
    def _cgroup(self, root, rel, limit, current=0, file=0, shmem=0):
        path = os.path.join(root, rel.lstrip("/")) if rel else root
        os.makedirs(path, exist_ok=True)
        for name, text in (
            ("memory.max", limit),
            ("memory.current", str(current)),
            ("memory.stat", f"anon 1\nfile {file}\nshmem {shmem}\n"),
        ):
            with open(os.path.join(path, name), "w") as f:
                f.write(text)

    def test_a_container_limit_below_the_host_is_the_limit(self):
        gib = 1 << 30
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root)
        meminfo = self._meminfo(root, 64 * gib, 48 * gib)
        proc_cg = os.path.join(root, "proc_cgroup")
        with open(proc_cg, "w") as f:
            f.write("0::/kubepods/pod1\n")
        self._cgroup(root, "", str(16 * gib), current=2 * gib)
        self._cgroup(
            root,
            "/kubepods",
            str(8 * gib),
            current=6 * gib,
            file=3 * gib,
            shmem=2 * gib,
        )
        self._cgroup(root, "/kubepods/pod1", "max", current=1 * gib)
        with patch.object(store, "_PROC_MEMINFO", meminfo), patch.object(
            store, "_CGROUP", root
        ), patch.object(store, "_PROC_CGROUP", proc_cg):
            # Tightest ancestor is /kubepods: 8 GiB limit, 6 GiB charged, 1 GiB reclaimable.
            self.assertEqual(store._host_memory(), (8 * gib, 3 * gib))
        self._cgroup(root, "", "max")
        self._cgroup(root, "/kubepods", "max")
        with patch.object(store, "_PROC_MEMINFO", meminfo), patch.object(
            store, "_CGROUP", root
        ), patch.object(store, "_PROC_CGROUP", proc_cg):
            self.assertEqual(store._host_memory(), (64 * gib, 48 * gib))

    def test_host_reserve_uses_max_of_ratio_and_min_gb(self):
        gib = 1 << 30
        s = store._Store(
            store.StoreSettings(host_reserve_ratio=0.2, host_reserve_min_gb=0.0)
        )
        # 20% of 480 GiB = 96 GiB (no upper cap).
        self.assertEqual(s._host_reserve_bytes(480 * gib), int(480 * gib * 0.2))
        self.assertEqual(s._host_reserve_bytes(4000 * gib), int(4000 * gib * 0.2))
        s2 = store._Store(
            store.StoreSettings(host_reserve_ratio=0.2, host_reserve_min_gb=64.0)
        )
        # 20% of 100 GiB = 20 GiB < 64 GiB -> min_gb wins.
        self.assertEqual(s2._host_reserve_bytes(100 * gib), 64 * gib)
        # 20% of 480 GiB = 96 GiB > 64 GiB -> ratio wins.
        self.assertEqual(s2._host_reserve_bytes(480 * gib), int(480 * gib * 0.2))

    def test_group_oom_cgroup_uses_raw_headroom_without_page_cache(self):
        gib = 1 << 30
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root)
        meminfo = self._meminfo(root, 64 * gib, 48 * gib)
        proc_cg = os.path.join(root, "proc_cgroup")
        with open(proc_cg, "w") as f:
            f.write("0::/kubepods/pod1\n")
        self._cgroup(
            root,
            "/kubepods/pod1",
            str(8 * gib),
            current=6 * gib,
            file=3 * gib,
            shmem=1 * gib,
        )
        with open(os.path.join(root, "kubepods/pod1/memory.oom.group"), "w") as f:
            f.write("1\n")
        with patch.object(store, "_PROC_MEMINFO", meminfo), patch.object(
            store, "_CGROUP", root
        ), patch.object(store, "_PROC_CGROUP", proc_cg):
            # Group OOM ignores the 2 GiB file-shmem cache and uses raw 8 - 6 = 2 GiB headroom.
            self.assertEqual(store._host_memory(), (8 * gib, 2 * gib))

    def test_cgroup_subtree_mount_resolves_leaf_limit(self):
        gib = 1 << 30
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root)
        meminfo = self._meminfo(root, 64 * gib, 48 * gib)
        cg_mount = os.path.join(root, "cg")
        proc_cg = os.path.join(root, "proc_cgroup")
        mountinfo = os.path.join(root, "mountinfo")
        with open(proc_cg, "w") as f:
            f.write("0::/tenant/pod\n")
        # Subtree /tenant mounted at cg_mount -> /tenant/pod lives at cg_mount/pod!
        with open(mountinfo, "w") as f:
            f.write(f"42 1 0:28 /tenant {cg_mount} rw - cgroup2 cgroup2 rw\n")
        self._cgroup(cg_mount, "", str(16 * gib), current=2 * gib)
        self._cgroup(cg_mount, "/pod", str(4 * gib), current=3 * gib)
        with patch.object(store, "_PROC_MEMINFO", meminfo), patch.object(
            store, "_PROC_CGROUP", proc_cg
        ), patch.object(store, "_PROC_MOUNTINFO", mountinfo):
            self.assertEqual(store._host_memory(), (4 * gib, 1 * gib))

    def test_multiple_cgroup_mounts_aggregate_visible_ancestors(self):
        gib = 1 << 30
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root)
        meminfo = self._meminfo(root, 64 * gib, 48 * gib)
        full_mount = os.path.join(root, "full")
        narrow_mount = os.path.join(root, "scope")
        proc_cg = os.path.join(root, "proc_cgroup")
        mountinfo = os.path.join(root, "mountinfo")
        with open(proc_cg, "w") as f:
            f.write("0::/tenant/pod\n")
        # Both full-tree mount and narrow bind mount exist: narrow has memory.max=max,
        # while /tenant on the full mount enforces 6 GiB + group OOM!
        with open(mountinfo, "w") as f:
            f.write(f"41 1 0:28 / {full_mount} rw - cgroup2 cgroup2 rw\n")
            f.write(f"42 1 0:28 /tenant/pod {narrow_mount} rw - cgroup2 cgroup2 rw\n")
        self._cgroup(full_mount, "", "max")
        self._cgroup(
            full_mount, "/tenant", str(6 * gib), current=4 * gib, file=2 * gib, shmem=0
        )
        with open(os.path.join(full_mount, "tenant/memory.oom.group"), "w") as f:
            f.write("1\n")
        self._cgroup(full_mount, "/tenant/pod", "max")
        self._cgroup(narrow_mount, "", "max")
        with patch.object(store, "_PROC_MEMINFO", meminfo), patch.object(
            store, "_PROC_CGROUP", proc_cg
        ), patch.object(store, "_PROC_MOUNTINFO", mountinfo):
            # Must see /tenant's 6 GiB limit and group OOM (raw 6 - 4 = 2 GiB, no file credit).
            self.assertEqual(store._host_memory(), (6 * gib, 2 * gib))

    def _meminfo(self, root, total_bytes, avail_bytes):
        path = os.path.join(root, "meminfo")
        with open(path, "w") as f:
            f.write(
                f"MemTotal:       {total_bytes // 1024} kB\n"
                f"MemAvailable:   {avail_bytes // 1024} kB\n"
            )
        return path

    def _cgroup_v1(
        self,
        root,
        rel,
        limit,
        usage=0,
        *,
        total_cache=0,
        total_shmem=0,
        hierarchical_limit=None,
        hierarchical_memsw_limit=None,
        memsw_limit=None,
        memsw_usage=0,
        use_hierarchy=1,
        write_usage=True,
        write_stat=True,
    ):
        path = os.path.join(root, rel.lstrip("/")) if rel else root
        os.makedirs(path, exist_ok=True)
        stat_lines = [
            f"cache {total_cache}",
            f"shmem {total_shmem}",
            f"total_cache {total_cache}",
            f"total_shmem {total_shmem}",
        ]
        if hierarchical_limit is not None:
            stat_lines.append(f"hierarchical_memory_limit {hierarchical_limit}")
        if hierarchical_memsw_limit is not None:
            stat_lines.append(f"hierarchical_memsw_limit {hierarchical_memsw_limit}")
        files = [
            ("memory.limit_in_bytes", str(limit)),
            ("memory.use_hierarchy", str(use_hierarchy)),
        ]
        if write_usage:
            files.append(("memory.usage_in_bytes", str(usage)))
        if write_stat:
            files.append(("memory.stat", "\n".join(stat_lines) + "\n"))
        if memsw_limit is not None:
            files.append(("memory.memsw.limit_in_bytes", str(memsw_limit)))
            files.append(("memory.memsw.usage_in_bytes", str(memsw_usage)))
        for name, text in files:
            with open(os.path.join(path, name), "w") as f:
                f.write(text)

    def test_hybrid_cgroup_v1_memory_ignores_unmatched_cgroup2_and_honors_memsw(self):
        gib = 1 << 30
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root)
        meminfo = self._meminfo(root, 64 * gib, 48 * gib)
        v1_mount = os.path.join(root, "memory")
        v2_mount = os.path.join(root, "unified")
        proc_cg = os.path.join(root, "proc_cgroup")
        mountinfo = os.path.join(root, "mountinfo")
        # Hybrid Linux 5.10 container: v1 owns memory; cgroup2 is mounted elsewhere for
        # systemd/pids and does not cover /user.slice, which must not cause a failure.
        with open(proc_cg, "w") as f:
            f.write("4:memory:/kubepods/pod1\n0::/user.slice\n")
        with open(mountinfo, "w") as f:
            f.write(
                f"35 1 0:30 / {v1_mount} rw - cgroup cgroup rw,memory\n"
                f"36 1 0:31 /other {v2_mount} rw - cgroup2 cgroup2 rw\n"
            )
        self._cgroup_v1(v1_mount, "", limit=9223372036854771712, usage=1 * gib)
        self._cgroup_v1(
            v1_mount,
            "/kubepods",
            limit=12 * gib,
            usage=10 * gib,
            total_cache=2 * gib,
            total_shmem=1 * gib,
        )
        self._cgroup_v1(
            v1_mount,
            "/kubepods/pod1",
            limit=10 * gib,
            usage=4 * gib,
            total_cache=1 * gib,
            total_shmem=0,
            hierarchical_limit=10 * gib,
            memsw_limit=12 * gib,
            memsw_usage=11 * gib,
        )
        with patch.object(store, "_PROC_MEMINFO", meminfo), patch.object(
            store, "_PROC_CGROUP", proc_cg
        ), patch.object(store, "_PROC_MOUNTINFO", mountinfo):
            # Parent /kubepods RAM headroom: 12 - 10 + (2 - 1) = 3 GiB;
            # Leaf RAM headroom: 10 - 4 + 1 = 7 GiB;
            # Leaf memsw headroom (with 7 GiB swapped): 12 - 11 + 1 = 2 GiB -> total=10, avail=2.
            self.assertEqual(store._host_memory(), (10 * gib, 2 * gib))

    def test_cgroup_v1_use_hierarchy_zero_stops_ancestor_walk_and_unlimited_falls_back(
        self,
    ):
        gib = 1 << 30
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root)
        meminfo = self._meminfo(root, 64 * gib, 40 * gib)
        v1_mount = os.path.join(root, "memory")
        proc_cg = os.path.join(root, "proc_cgroup")
        mountinfo = os.path.join(root, "mountinfo")
        with open(proc_cg, "w") as f:
            f.write("4:memory:/parent/leaf\n0::/\n")
        with open(mountinfo, "w") as f:
            f.write(f"35 1 0:30 / {v1_mount} rw - cgroup cgroup rw,memory\n")
        # Legal Linux v1 configuration: parent has use_hierarchy=0, leaf has use_hierarchy=1.
        self._cgroup_v1(
            v1_mount, "/parent", limit=4 * gib, usage=3 * gib, use_hierarchy=0
        )
        self._cgroup_v1(
            v1_mount,
            "/parent/leaf",
            limit=9223372036854771712,
            usage=1 * gib,
            use_hierarchy=1,
        )
        with patch.object(store, "_PROC_MEMINFO", meminfo), patch.object(
            store, "_PROC_CGROUP", proc_cg
        ), patch.object(store, "_PROC_MOUNTINFO", mountinfo):
            # Parent use_hierarchy=0 isolates leaf from /parent's 4 GiB limit; unlimited falls
            # back to /proc/meminfo (64 GiB / 40 GiB).
            self.assertEqual(store._host_memory(), (64 * gib, 40 * gib))

    def test_invisible_cgroup_v1_ancestor_limit_fails_closed(self):
        gib = 1 << 30
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root)
        meminfo = self._meminfo(root, 64 * gib, 48 * gib)
        v1_mount = os.path.join(root, "memory")
        proc_cg = os.path.join(root, "proc_cgroup")
        mountinfo = os.path.join(root, "mountinfo")
        with open(proc_cg, "w") as f:
            f.write("4:memory:/kubepods/pod1\n0::/\n")
        with open(mountinfo, "w") as f:
            f.write(
                f"35 1 0:30 /kubepods/pod1 {v1_mount} rw - cgroup cgroup rw,memory\n"
            )
        # Leaf limit is 16 GiB, but an invisible parent enforces 8 GiB whose usage is unknown.
        self._cgroup_v1(
            v1_mount,
            "",
            limit=16 * gib,
            usage=2 * gib,
            hierarchical_limit=8 * gib,
        )
        with patch.object(store, "_PROC_MEMINFO", meminfo), patch.object(
            store, "_PROC_CGROUP", proc_cg
        ), patch.object(store, "_PROC_MOUNTINFO", mountinfo):
            with self.assertRaisesRegex(OSError, "invisible cgroup v1 ancestor"):
                store._host_memory()

    def test_sibling_bind_mount_and_missing_usage_fail_closed(self):
        gib = 1 << 30
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root)
        meminfo = self._meminfo(root, 64 * gib, 48 * gib)
        v1_mount = os.path.join(root, "memory")
        proc_cg = os.path.join(root, "proc_cgroup")
        mountinfo = os.path.join(root, "mountinfo")
        self._cgroup_v1(v1_mount, "", limit=8 * gib, usage=1 * gib)
        # Sibling bind mounts (including /../sibling) must not be accepted as covering "/".
        with open(proc_cg, "w") as f:
            f.write("4:memory:/\n0::/\n")
        for bad_root in ("/docker/sibling", "/../sibling"):
            with open(mountinfo, "w") as f:
                f.write(
                    f"35 1 0:30 {bad_root} {v1_mount} rw - cgroup cgroup rw,memory\n"
                )
            with patch.object(store, "_PROC_MEMINFO", meminfo), patch.object(
                store, "_PROC_CGROUP", proc_cg
            ), patch.object(store, "_PROC_MOUNTINFO", mountinfo):
                with self.assertRaises(OSError):
                    store._host_memory()
        # Missing memory.usage_in_bytes with a finite limit fails closed; missing memory.stat
        # degrades conservatively to zero reclaimable cache.
        with open(mountinfo, "w") as f:
            f.write(f"35 1 0:30 / {v1_mount} rw - cgroup cgroup rw,memory\n")
        no_stat_dir = os.path.join(root, "no_stat")
        self._cgroup_v1(no_stat_dir, "", limit=8 * gib, usage=5 * gib, write_stat=False)
        with open(mountinfo, "w") as f:
            f.write(f"35 1 0:30 / {no_stat_dir} rw - cgroup cgroup rw,memory\n")
        with patch.object(store, "_PROC_MEMINFO", meminfo), patch.object(
            store, "_PROC_CGROUP", proc_cg
        ), patch.object(store, "_PROC_MOUNTINFO", mountinfo):
            self.assertEqual(store._host_memory(), (8 * gib, 3 * gib))
        no_usage_dir = os.path.join(root, "no_usage")
        self._cgroup_v1(no_usage_dir, "", limit=8 * gib, write_usage=False)
        with open(mountinfo, "w") as f:
            f.write(f"35 1 0:30 / {no_usage_dir} rw - cgroup cgroup rw,memory\n")
        with patch.object(store, "_PROC_MEMINFO", meminfo), patch.object(
            store, "_PROC_CGROUP", proc_cg
        ), patch.object(store, "_PROC_MOUNTINFO", mountinfo):
            with self.assertRaises(OSError):
                store._host_memory()
        # Missing memory.limit_in_bytes on a resolved v1 memory directory fails closed.
        empty_v1_dir = os.path.join(root, "empty_v1")
        os.makedirs(empty_v1_dir)
        with open(mountinfo, "w") as f:
            f.write(f"35 1 0:30 / {empty_v1_dir} rw - cgroup cgroup rw,memory\n")
        with patch.object(store, "_PROC_MEMINFO", meminfo), patch.object(
            store, "_PROC_CGROUP", proc_cg
        ), patch.object(store, "_PROC_MOUNTINFO", mountinfo):
            with self.assertRaises(OSError):
                store._host_memory()
        # Process cgroup path containing ".." must be rejected, not collapsed to "/x"
        # even when valid mounts and "/x" accounting files exist.
        v2_escape = os.path.join(root, "v2_escape")
        self._cgroup(v2_escape, "/x", str(8 * gib), current=1 * gib)
        v1_escape = os.path.join(root, "v1_escape")
        self._cgroup_v1(v1_escape, "/x", limit=8 * gib, usage=1 * gib)
        for bad_proc, mline in (
            ("0::/../x\n", f"41 1 0:28 / {v2_escape} rw - cgroup2 cgroup2 rw\n"),
            (
                "4:memory:/../x\n",
                f"35 1 0:30 / {v1_escape} rw - cgroup cgroup rw,memory\n",
            ),
        ):
            with open(proc_cg, "w") as f:
                f.write(bad_proc)
            with open(mountinfo, "w") as f:
                f.write(mline)
            with patch.object(store, "_PROC_MEMINFO", meminfo), patch.object(
                store, "_PROC_CGROUP", proc_cg
            ), patch.object(store, "_PROC_MOUNTINFO", mountinfo):
                with self.assertRaisesRegex(OSError, "lies outside cgroup namespace"):
                    store._host_memory()

    def test_missing_cgroup_mounts_fail_closed(self):
        gib = 1 << 30
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root)
        meminfo = self._meminfo(root, 64 * gib, 48 * gib)
        proc_cg = os.path.join(root, "proc_cgroup")
        mountinfo = os.path.join(root, "mountinfo")
        with open(mountinfo, "w") as f:
            f.write("21 1 0:20 / / rw - ext4 /dev/sda1 rw\n")
        for content in ("0::/\n", "4:memory:/kubepods/pod1\n0::/\n"):
            with open(proc_cg, "w") as f:
                f.write(content)
            with patch.object(store, "_PROC_MEMINFO", meminfo), patch.object(
                store, "_PROC_CGROUP", proc_cg
            ), patch.object(store, "_PROC_MOUNTINFO", mountinfo):
                with self.assertRaises(OSError):
                    store._host_memory()


def _try_list(name):
    try:
        return store.list_store(name)
    except OSError:
        return None


def _drop_fds(result):
    for fd in result.allocation_fds:
        os.close(fd)
    if result.hold_sock is not None:
        result.hold_sock.close()
    return None


if __name__ == "__main__":
    unittest.main()
