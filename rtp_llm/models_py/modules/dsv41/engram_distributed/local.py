# SPDX-License-Identifier: Apache-2.0
"""Single-host GPU startup transport; memory and runtime lifetimes stay shared.

The Store carries metadata and barriers. Only SCM_RIGHTS transports POSIX FDs;
no process-local descriptor number is published as a portable memory handle.
The private attempt directory and peer credentials identify each sender; the
socket carries only fixed-size rank/count headers alongside the descriptors.
"""

from __future__ import annotations

import array
import json
import logging
import os
import shutil
import socket
import struct
import tempfile
import threading
import time
from dataclasses import asdict, replace
from datetime import timedelta
from pathlib import Path

from .checkpoint import (
    _expected_layout,
    _same_snapshot,
    checkpoint_samples,
    hash_state_digest,
    inspect_checkpoint,
    load_rows,
)
from .config import (
    EngramOptions,
    StorageMode,
    StoragePolicy,
    _checkpoint_revision,
    capacity_from_environ,
)
from .manifest import AllocationSlice, Manifest, TensorSpec, ValidatedManifest
from .placement import check_gpu_capacity, plan_shards
from .storage import CleanupPendingError, FabricEngramStorage, _load_native

_PHASES = ("metadata", "directory", "loaded", "listeners", "verified")


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


class _Agreement:
    def __init__(self, store, rank, world, deadline):
        self.store, self.rank, self.world, self.deadline = store, rank, world, deadline
        attempt = store.add(f"engram/cuda_local/start_count/{rank}", 1)
        self.prefix = f"engram/cuda_local/attempt/{attempt}/"
        self.completed = set()

    def remaining(self):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("LOCAL_CUDA_ENGRAM_STARTUP_TIMEOUT")
        return remaining

    def exchange(self, phase, payload):
        self.store.set(
            self.prefix + f"{phase}/{self.rank}", _json({"ok": True, "value": payload})
        )
        self.completed.add(phase)
        keys = [self.prefix + f"{phase}/{rank}" for rank in range(self.world)]
        self.store.wait(keys, timedelta(seconds=self.remaining()))
        values = [json.loads(self.store.get(key)) for key in keys]
        if any(value.get("ok") is not True for value in values):
            raise RuntimeError("LOCAL_CUDA_ENGRAM_PEER_STARTUP_FAILED")
        return [value["value"] for value in values]

    def fail(self, error):
        # A broken Store must not replace the original error or skip cleanup.
        try:
            for phase in _PHASES:
                if phase not in self.completed:
                    self.store.set(
                        self.prefix + f"{phase}/{self.rank}",
                        _json({"ok": False, "error": type(error).__name__}),
                    )
        except BaseException:
            pass


def _peer(conn, pid):
    actual_pid, uid, _ = struct.unpack(
        "3i", conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
    )
    if actual_pid != pid or uid != os.getuid():
        raise RuntimeError("LOCAL_CUDA_ENGRAM_UNIX_PEER_MISMATCH")


class _FdServer:
    def __init__(self, path, rank, peers, fds, agreement):
        self.path, self.rank = path, rank
        self.peers, self.fds, self.agreement = peers, fds, agreement
        self.stop = threading.Event()
        self.error = None
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        try:
            self.listener.bind(path)
            self.listener.listen(len(peers))
            self.listener.settimeout(0.25)
            self.thread = threading.Thread(
                target=self._serve, name="engram-local-fds", daemon=True
            )
            self.thread.start()
        except BaseException:
            self.listener.close()
            Path(path).unlink(missing_ok=True)
            raise

    def _serve(self):
        served = {self.rank}
        try:
            while len(served) < len(self.peers) and not self.stop.is_set():
                self.agreement.remaining()
                try:
                    conn, _ = self.listener.accept()
                except socket.timeout:
                    continue
                with conn:
                    conn.settimeout(min(2.0, self.agreement.remaining()))
                    (rank,) = struct.unpack("!I", conn.recv(4096))
                    if rank in served or not 0 <= rank < len(self.peers):
                        raise RuntimeError("LOCAL_CUDA_ENGRAM_INVALID_FD_REQUEST")
                    _peer(conn, self.peers[rank]["pid"])
                    payload = struct.pack("!II", self.rank, len(self.fds))
                    sent = conn.sendmsg(
                        [payload],
                        [
                            (
                                socket.SOL_SOCKET,
                                socket.SCM_RIGHTS,
                                array.array("i", self.fds),
                            )
                        ],
                    )
                    if sent != len(payload):
                        raise RuntimeError("LOCAL_CUDA_ENGRAM_SHORT_FD_SEND")
                    served.add(rank)
        except BaseException as error:
            if not self.stop.is_set():
                self.error = error
                self.agreement.fail(error)

    def close(self):
        self.stop.set()
        self.thread.join(timeout=3.0)
        self.listener.close()
        Path(self.path).unlink(missing_ok=True)
        if self.thread.is_alive():
            raise RuntimeError("LOCAL_CUDA_ENGRAM_FD_SERVER_DID_NOT_STOP")


def _receive_fds(path, owner, rank, peer_pid, count, timeout):
    fds = []
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as conn:
            conn.settimeout(timeout)
            conn.connect(path)
            _peer(conn, peer_pid)
            conn.sendall(struct.pack("!I", rank))
            payload, controls, flags, _ = conn.recvmsg(
                4096,
                socket.CMSG_SPACE(count * array.array("i").itemsize),
                socket.MSG_CMSG_CLOEXEC,
            )
            for level, kind, data in controls:
                if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                    received = array.array("i")
                    received.frombytes(
                        data[: len(data) - len(data) % received.itemsize]
                    )
                    fds.extend(received)
            for fd in fds:
                os.set_inheritable(fd, False)
            if (
                flags & (socket.MSG_CTRUNC | socket.MSG_TRUNC)
                or payload != struct.pack("!II", owner, count)
                or len(fds) != count
            ):
                raise RuntimeError("LOCAL_CUDA_ENGRAM_INVALID_FD_REPLY")
            return fds
    except BaseException:
        for fd in fds:
            os.close(fd)
        raise


class LocalCudaStartup:
    """Load each rank's GPU shard, exchange FDs, and stage the common storage."""

    def __init__(
        self,
        store,
        *,
        rank,
        world_size,
        device,
        checkpoint,
        gang_hosts=None,
        environ=None,
        install=None,
    ):
        if (
            type(rank) is not int
            or type(world_size) is not int
            or not 2 <= world_size <= 64
            or not 0 <= rank < world_size
        ):
            raise ValueError(
                "cuda_local requires at least two and at most 64 GPU ranks"
            )
        if gang_hosts is not None and (
            len(gang_hosts) != world_size or len(set(gang_hosts)) != 1
        ):
            raise ValueError("cuda_local does not support cross-host groups")
        if store is None:
            raise ValueError("cuda_local requires the existing startup Store")
        if type(device) is not int or device < 0:
            raise ValueError("cuda_local requires an explicit local CUDA ordinal")
        self.env = os.environ if environ is None else environ
        if any(key in self.env for key in ("RTP_ENGRAM_COHORT", "ENGRAM_SHARD_COUNT")):
            raise ValueError("cuda_local does not use a cross-host cohort")
        self.timeout = float(self.env.get("RTP_ENGRAM_COHORT_TIMEOUT", "1800"))
        if not 0 < self.timeout <= 86400:
            raise ValueError("cuda_local requires a bounded startup timeout")
        self.store, self.rank, self.world, self.device = store, rank, world_size, device
        self.checkpoint = checkpoint
        self.install = install

    def run(self):
        from ..engram import EngramLayout, NgramHashState

        if self.install is None:
            raise RuntimeError("cuda_local requires a runtime installation callback")
        root = Path(self.checkpoint).resolve(strict=True)
        config = json.loads((root / "config.json").read_text())
        layout = EngramLayout({**config, **config.get("text_config", {})})
        if not layout.layer_ids:
            return None
        expected = replace(
            _expected_layout(
                layout,
                _checkpoint_revision(root),
                hash_state_digest(NgramHashState(layout, str(root))),
            ),
            backend="cuda_ipc",
        )
        storage = self.prepare(expected)
        try:
            self.install(
                storage,
                EngramOptions(
                    StoragePolicy(StorageMode.CUDA_LOCAL), None, "process_lifetime"
                ),
            )
        except BaseException as original:
            try:
                storage.close()
            except BaseException:
                from .runtime import process_runtime

                process_runtime.retain(storage)
                raise CleanupPendingError(
                    "local GPU runtime installation cleanup incomplete", storage
                ) from original
            raise
        logging.getLogger(__name__).info(
            "engram event=local_gpu_ready rank=%d world=%d device=%d before_kv_sizing=true",
            self.rank,
            self.world,
            self.device,
        )
        return storage

    def prepare(self, expected):
        """Build verified storage. Used directly by the real multi-process GPU test."""
        import torch

        from .runtime import _verify_samples, process_runtime

        agreement = _Agreement(
            self.store, self.rank, self.world, time.monotonic() + self.timeout
        )
        server, directory, all_fds, owners, storage = None, None, [], [], None
        try:
            native = _load_native()
            granularity = native.device_granularity(self.device, handle_type="posix_fd")
            peers = agreement.exchange(
                "metadata",
                dict(
                    hostname=socket.gethostname(),
                    boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                    device=self.device,
                    pid=os.getpid(),
                    granularity=granularity,
                    layout=asdict(expected),
                ),
            )
            if (
                len({(p["hostname"], p["boot_id"]) for p in peers}) != 1
                or len({p["device"] for p in peers}) != self.world
                or any(
                    p["granularity"] != granularity or p["layout"] != peers[0]["layout"]
                    for p in peers
                )
            ):
                raise ValueError(
                    "cuda_local requires distinct local GPUs and matching layout/granularity"
                )
            root = Path(self.checkpoint).resolve(strict=True)
            plan = inspect_checkpoint(root, expected)
            plans = {
                layer: plan_shards(
                    rows, expected.head_dim, self.world, granularity=granularity
                )
                for layer, rows in expected.layers
            }
            sizes = [
                size
                for layer in plan.layers
                for size in (
                    plans[layer.layer_id].weight_bytes,
                    plans[layer.layer_id].scale_bytes,
                )
            ]
            if len(sizes) > 128:
                raise ValueError("cuda_local FD bundle exceeds the SCM_RIGHTS limit")
            max_allocation, max_total = capacity_from_environ(self.env)
            if (
                any(
                    size > min(max_allocation, native.max_allocation_bytes)
                    for size in sizes
                )
                or sum(sizes) > max_total
            ):
                raise ValueError("GPU Engram shard exceeds allocation budget")
            reserve = int(self.env.get("RTP_ENGRAM_GPU_RESERVE_BYTES", str(32 << 30)))
            if not 0 < reserve < 2**63:
                raise ValueError("INVALID_ENGRAM_GPU_RESERVE")
            check_gpu_capacity({self.device: sum(sizes)}, reserve)
            directory = (
                tempfile.mkdtemp(prefix="rtp-engram-", dir="/tmp")
                if self.rank == 0
                else None
            )
            descriptions = agreement.exchange("directory", directory)
            directory = descriptions[0]
            stat = os.stat(directory, follow_symlinks=False)
            if stat.st_uid != os.getuid() or stat.st_mode & 0o777 != 0o700:
                raise RuntimeError(
                    "cuda_local directory must be owned by this UID with mode0700"
                )
            snapshots = {file.relative_path: file for file in plan.files}
            for layer in plan.layers:
                shard = plans[layer.layer_id]
                start, end = shard.row_range(self.rank)
                for tensor, size in zip(
                    layer.tensors, (shard.weight_bytes, shard.scale_bytes)
                ):
                    agreement.remaining()
                    allocation = native.FabricAllocation(
                        size, -1, self.device, memory_kind="gpu", handle_type="posix_fd"
                    )
                    owners.append(allocation)
                    load_rows(
                        allocation,
                        root,
                        tensor,
                        snapshots[tensor.relative_path],
                        start,
                        end,
                        tensor.shape[1],
                        chunk_bytes=min(64 << 20, native.max_load_chunk_bytes),
                        deadline=agreement.deadline,
                        discard_file_cache=self.env.get(
                            "RTP_ENGRAM_RELEASE_INPUT_PAGE_CACHE"
                        )
                        == "1",
                    )
                    all_fds.append(allocation.export_fd())
            _same_snapshot(root, plan.index, index=True)
            agreement.exchange("loaded", True)
            own_fds = tuple(all_fds)
            server = _FdServer(
                str(Path(directory) / f"{self.rank}.sock"),
                self.rank,
                peers,
                own_fds,
                agreement,
            )
            agreement.exchange("listeners", True)
            bundles = []
            for owner_rank, peer in enumerate(peers):
                received = (
                    list(own_fds)
                    if owner_rank == self.rank
                    else _receive_fds(
                        str(Path(directory) / f"{owner_rank}.sock"),
                        owner_rank,
                        self.rank,
                        peer["pid"],
                        len(sizes),
                        agreement.remaining(),
                    )
                )
                if owner_rank != self.rank:
                    all_fds.extend(received)
                bundles.append(received)
            tensors, handles = [], {}
            for index, layer in enumerate(plan.layers):
                shard = plans[layer.layer_id]
                for role, tensor in enumerate(layer.tensors):
                    columns = tensor.shape[1]
                    parts = []
                    for rank in range(self.world):
                        begin, end = shard.row_range(rank)
                        aid = f"{tensor.name}/{rank}"
                        parts.append(
                            AllocationSlice(
                                begin,
                                end,
                                (end - begin) * columns,
                                shard.shard_rows * columns,
                                granularity,
                                peers[0]["hostname"],
                                aid,
                            )
                        )
                        handles[aid] = bundles[rank][2 * index + role]
                    tensors.append(TensorSpec(tensor.name, tensor.shape, tuple(parts)))
            manifest = Manifest(
                expected.model_revision,
                agreement.prefix,
                expected.layout_abi,
                expected.hash_abi,
                expected.backend,
                tuple(tensors),
            )
            validated = ValidatedManifest(
                manifest, sum(sizes) * self.world, len(sizes) * self.world, expected
            )
            storage = FabricEngramStorage._attach_validated(
                validated,
                handles=handles,
                device=self.device,
                native=native,
                handle_type="posix_fd",
                backings=owners,
            )
            # Ownership moved to the storage that owns the imported mappings.
            owners = []
            samples = checkpoint_samples(root, manifest)
            storage.verify(
                lambda views, contract: _verify_samples(
                    views, contract, samples, torch.device("cuda", self.device)
                )
            )
            if server.error is not None:
                raise server.error
            agreement.exchange("verified", True)
            return storage
        except BaseException as original:
            agreement.fail(original)
            pending = (
                original.resource
                if isinstance(original, CleanupPendingError)
                else storage
            )
            if pending is not None:
                try:
                    pending.close()
                except BaseException:
                    process_runtime.retain(pending)
                    process_runtime.retain(owners)
                    raise CleanupPendingError(
                        "local GPU storage cleanup incomplete", pending
                    ) from original
            while owners:
                try:
                    owners[-1].close()
                    owners.pop()
                except BaseException:
                    process_runtime.retain(owners)
                    raise CleanupPendingError(
                        "local GPU backing cleanup incomplete", owners
                    ) from original
            raise
        finally:
            if server is not None:
                try:
                    server.close()
                except BaseException as error:
                    # An unfinished sender may still reference these FD numbers.
                    resources = (server, all_fds, directory, storage, owners)
                    process_runtime.retain(resources)
                    raise CleanupPendingError(
                        "local GPU FD sender cleanup incomplete", resources
                    ) from error
            for fd in all_fds:
                os.close(fd)
            if self.rank == 0 and directory is not None:
                shutil.rmtree(directory, ignore_errors=True)
