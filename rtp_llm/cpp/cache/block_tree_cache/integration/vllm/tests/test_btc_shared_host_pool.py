# SPDX-License-Identifier: Apache-2.0
"""Exercise the native Host pool over vLLM's actual cross-process mapping."""

import json
import mmap
import os
from pathlib import Path
import select
import subprocess
import sys
from time import monotonic
from types import SimpleNamespace
from unittest.mock import patch
import uuid

from block_tree_cache import Binding, Group, Page, Position, Receipt, Result, Rule, Status, WorkKind
from btc_vllm.connector import BtcHostHandshakeMetadata
from btc_vllm.scheduler import BtcScheduler
from btc_vllm.torchtpu.worker import BtcHostGeometry, open_tpu_host_region


PAYLOAD = 257
SLOTS = 3
STRIDE = mmap.PAGESIZE
GUARD = b"\xe3"
REPLY = b"BTC_SHARED_HOST_PEER "


def geometry(identity):
    return BtcHostGeometry(identity, 1, PAYLOAD, STRIDE, SLOTS)


def identity(region):
    stat = os.fstat(region.fd)
    return stat.st_dev, stat.st_ino, stat.st_size


def scheduler_with_host(region_geometry, remote=None):
    extra = {} if remote is None else {"btc_pace": remote}
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config=extra),
        model_config=SimpleNamespace(model="test"),
        cache_config=SimpleNamespace(cache_dtype="test"),
    )
    scheduler = BtcScheduler(config, region_geometry)
    scheduler.group_sizes = (8, 8)
    scheduler.last_checkpoint = (False, False)
    scheduler._groups = (Group(0, Rule.FULL, 8), Group(1, Rule.FULL, 8))
    scheduler.step = 8
    scheduler._bind_local_host(region_geometry)
    return scheduler


def worker_process(engine_id):
    region = open_tpu_host_region(geometry(engine_id))
    try:
        sys.stdout.buffer.write(REPLY + json.dumps(region.backing_identity()).encode() + b"\n")
        sys.stdout.buffer.flush()
        assert sys.stdin.buffer.read(1) == b"1"
        with region.create_kv_memoryview().cast("B") as view:
            assert bytes(view[:PAYLOAD]) == b"I" * PAYLOAD
            assert bytes(view[2 * STRIDE:2 * STRIDE + PAYLOAD]) == b"L" * PAYLOAD
            view[:PAYLOAD] = b"W" * PAYLOAD
            view[2 * STRIDE:2 * STRIDE + PAYLOAD] = b"Z" * PAYLOAD
    finally:
        region.cleanup()


def test_shared_host_mapping_across_processes_and_reset():
    item = geometry("btc_test_" + uuid.uuid4().hex)
    with subprocess.Popen(
        [sys.executable, __file__, "--peer", item.engine_id],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    ) as peer:
        try:
            worker_identity = wait_for_worker_host(peer)
            exercise_attached_host(item, worker_identity, peer)
        finally:
            if peer.poll() is None:
                peer.kill()


def wait_for_worker_host(peer):
    output = b""
    deadline = monotonic() + 40
    while select.select([peer.stdout], [], [], max(0, deadline - monotonic()))[0]:
        chunk = os.read(peer.stdout.fileno(), 4096)
        assert chunk, "worker exited before attaching"
        output += chunk
        lines = output.split(b"\n")
        for line in lines[:-1]:
            if line.startswith(REPLY):
                return tuple(json.loads(line.removeprefix(REPLY)))
        output = lines[-1]
    raise AssertionError("worker did not attach")


def exercise_attached_host(item, worker_identity, peer):
    scheduler = scheduler_with_host(item)
    path = Path(scheduler._local_host.mmap_path)
    file_identity = identity(scheduler._local_host)
    assert scheduler._remote is None
    try:
        assert path.exists() and not scheduler._local_host._creator
        assert worker_identity == file_identity
        for incomplete in ({}, {(0, 0): BtcHostHandshakeMetadata(0, (0, 0, 0))}):
            try:
                scheduler.finish_host_attachment(incomplete)
            except ValueError:
                pass
            else:
                raise AssertionError("invalid attachment was accepted")
            assert path.exists()
        attached = {(0, 0): BtcHostHandshakeMetadata(0, worker_identity)}
        scheduler.finish_host_attachment(attached)
        assert not path.exists() and os.fstat(scheduler._local_host.fd).st_nlink == 0
        # Reuse the exact pathname before the old creator tears down. The old
        # worker and scheduler still exchange bytes through their original inode.
        replacement = open_tpu_host_region(item)
        try:
            assert replacement.backing_identity() != file_identity
            scheduler.finish_host_attachment(attached)
            exchange_host_rows(scheduler, peer)
            assert identity(replacement)[1] == path.stat().st_ino
        finally:
            replacement.cleanup()
        initial = scheduler.native
        assert scheduler.reset() and scheduler.native is not initial
        assert identity(scheduler._local_host) == file_identity
        with scheduler.native.host_slot_buffer(1, 2) as after_reset:
            assert bytes(after_reset) == b"Z" * PAYLOAD
    finally:
        scheduler.close()
    assert not path.exists()


def exchange_host_rows(scheduler, peer):
    backing = scheduler._host_region.buffer
    backing[:] = GUARD * (SLOTS * STRIDE)
    with scheduler.native.host_slot_buffer(0, 0) as first:
        first[:] = b"I" * PAYLOAD
    with scheduler.native.host_slot_buffer(1, 2) as last:
        last[:] = b"L" * PAYLOAD
    peer.communicate("1", timeout=40)
    assert peer.returncode == 0
    with scheduler.native.host_slot_buffer(0, 0) as first:
        assert bytes(first) == b"W" * PAYLOAD
    with scheduler.native.host_slot_buffer(1, 2) as last:
        assert bytes(last) == b"Z" * PAYLOAD
    assert bytes(backing[PAYLOAD:2 * STRIDE]) == GUARD * (2 * STRIDE - PAYLOAD)
    assert bytes(backing[2 * STRIDE + PAYLOAD:]) == GUARD * (STRIDE - PAYLOAD)


def test_pace_slot_lookup_follows_reopened_native_cache():
    class PayloadStorage:
        def __init__(self, _config, _payload, _ranks, *, memory, memory_fd):
            self.size = memory.nbytes
            self.fd = memory_fd

        def read_described(self, items):
            for group, _key, view, _proof in items:
                view[:] = bytes([group + 9]) * PAYLOAD
            return True

        def write(self, _group, _key, source):
            return bytes(source) == b"\x0a" * PAYLOAD

        def close(self):
            pass

    item = geometry("btc_test_" + uuid.uuid4().hex)
    with patch("btc_vllm.torchtpu.remote.PaceStorageBackend", PayloadStorage):
        scheduler = scheduler_with_host(item, {})
    path = Path(scheduler._local_host.mmap_path)
    try:
        scheduler.finish_host_attachment({
            (0, 0): BtcHostHandshakeMetadata(0, identity(scheduler._local_host)),
        })
        assert not path.exists()
        backend = scheduler._remote.storage
        assert backend.storage.fd == scheduler._local_host.fd
        assert backend.storage.size == STRIDE * SLOTS
        assert scheduler.reset()
        native = scheduler.native
        with patch.object(native, "host_slot_buffer", wraps=native.host_slot_buffer) as resolve:
            assert backend.read_described([(0, b"first", 0, b"x"), (1, b"end", 2, b"y")])
            assert backend.write(1, b"end", 2)
            assert [call.args for call in resolve.call_args_list] == [(0, 0), (1, 2), (1, 2)]
        backing = scheduler._host_region.buffer
        assert bytes(backing[:PAYLOAD]) == b"\x09" * PAYLOAD
        assert bytes(backing[2 * STRIDE:2 * STRIDE + PAYLOAD]) == b"\x0a" * PAYLOAD
    finally:
        scheduler.close()
    assert not path.exists()


def test_host_attachment_waits_for_every_rank():
    item = BtcHostGeometry("btc_test_" + uuid.uuid4().hex, 2, PAYLOAD, STRIDE, SLOTS)
    with_attached = [open_tpu_host_region(item), open_tpu_host_region(item)]
    try:
        scheduler = scheduler_with_host(item)
        try:
            path = Path(scheduler._local_host.mmap_path)
            ranks = {
                (0, rank): BtcHostHandshakeMetadata(rank, region.backing_identity())
                for rank, region in enumerate(with_attached)
            }
            try:
                scheduler.finish_host_attachment({(0, 0): ranks[(0, 0)]})
            except ValueError:
                pass
            else:
                raise AssertionError("missing worker rank was accepted")
            assert path.exists()
            scheduler.finish_host_attachment(ranks)
            assert not path.exists()
            assert all(os.fstat(region.fd).st_nlink == 0 for region in with_attached)
        finally:
            scheduler.close()
    finally:
        for region in with_attached:
            region.cleanup()


def test_unsafe_host_copy_keeps_shared_mapping_until_late_receipt():
    scheduler = scheduler_with_host(geometry("btc_test_" + uuid.uuid4().hex))
    cache = scheduler.native
    path = Path(scheduler._local_host.mmap_path)
    page = Page(0, 1, 1, 1)
    assert cache.publish((b"dirty-host",), 8, [Binding(Position(0, 8), (page,))]) is Status.OK
    assert not cache.poll().work
    cache.balance_device(0, 2, 0, [page])
    work, = cache.poll().work
    assert work.kind is WorkKind.DEVICE_TO_HOST
    cache.accept(work.id, 0)
    cache.complete([Receipt(work.id, 0, Result.TIMEOUT, False)])
    assert cache.close() is Status.PENDING
    assert path.exists()
    slot = work.pages[0].target[0].slot
    with cache.host_slot_buffer(0, slot) as view:
        assert len(view) == PAYLOAD
    cache.complete([Receipt(work.id, 0, Result.TIMEOUT, True)])
    cache.drain()
    for release in cache.poll().work:
        assert release.kind is WorkKind.RELEASE
        cache.complete([Receipt(release.id, 0, Result.SUCCESS, True)])
    scheduler.close()
    assert not path.exists()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--peer":
        worker_process(sys.argv[2])
    else:
        for check in (
            test_shared_host_mapping_across_processes_and_reset,
            test_pace_slot_lookup_follows_reopened_native_cache,
            test_host_attachment_waits_for_every_rank,
            test_unsafe_host_copy_keeps_shared_mapping_until_late_receipt,
        ):
            check()
            print("BTC_SHARED_HOST_TEST_OK", check.__name__, flush=True)
