# SPDX-License-Identifier: Apache-2.0
"""Startup-only, same-host DEVICE VMM shards; online lookup uses existing VA views."""
from __future__ import annotations

import array
import hashlib
import json
import os
import shutil
import socket
import struct
import tempfile
import threading
import time
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

from .engram_shard import plan_shards
from .engram_shard_storage import CudaDriverOps, ShardedLayerMapping
from .engram_storage import CleanupPendingError, FabricEngramStorage

_STAGED = None
_STAGED_LOCK = threading.Lock()
_PHASES = ("metadata", "directory", "loaded", "listeners", "verified")


def stage_local_cuda_storage(storage):
    global _STAGED
    with _STAGED_LOCK:
        if _STAGED is not None and _STAGED is not storage:
            raise RuntimeError("LOCAL_CUDA_ENGRAM_ALREADY_STAGED")
        if storage.info()["state"] != "READY":
            raise RuntimeError("LOCAL_CUDA_ENGRAM_NOT_READY")
        _STAGED = storage


def staged_local_cuda_storage():
    with _STAGED_LOCK:
        return _STAGED


def prepare_local_cuda_engram(
    store, *, rank, world_size, device, checkpoint, gang_hosts, timeout_s
):
    """Reserve real VRAM before the engine computes its KV-cache budget."""
    from .engram import EngramLayout, NgramHashState
    from .engram_bootstrap import hash_state_digest
    from .engram_manifest import ExpectedLayout
    from .engram_startup import _checkpoint_revision

    if device is None or checkpoint is None or staged_local_cuda_storage() is not None:
        raise RuntimeError("cuda_local requires one explicit pre-model startup")
    root = Path(checkpoint).resolve(strict=True)
    config = json.loads((root / "config.json").read_text())
    config = {**config, **config.get("text_config", {})}
    layout = EngramLayout(config)
    if not layout.layer_ids:
        return None
    expected = ExpectedLayout(
        layers=tuple(zip(layout.layer_ids, layout.num_embeddings)),
        head_dim=layout.head_dim,
        model_revision=_checkpoint_revision(root),
        hash_abi=hash_state_digest(NgramHashState(layout, str(root))),
    )
    storage = LocalCudaEngramStorage.from_checkpoint(
        root,
        expected,
        device=device,
        timeout_s=timeout_s,
        store=store,
        rank=rank,
        world_size=world_size,
        gang_hosts=gang_hosts,
    )
    try:
        stage_local_cuda_storage(storage)
    except BaseException:
        storage.close()
        raise
    import logging

    logging.getLogger(__name__).info(
        "Engram cuda_local staged before KV sizing: rank=%d shards=%d bytes=%d",
        rank,
        world_size,
        storage.local_physical_bytes,
    )
    return storage.info()


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
        for phase in _PHASES:
            if phase not in self.completed:
                self.store.set(
                    self.prefix + f"{phase}/{self.rank}",
                    _json({"ok": False, "error": type(error).__name__}),
                )


def _peer(conn, pid):
    actual_pid, uid, _ = struct.unpack(
        "3i", conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
    )
    if actual_pid != pid or uid != os.getuid():
        raise RuntimeError("LOCAL_CUDA_ENGRAM_UNIX_PEER_MISMATCH")


class _FdServer:
    def __init__(self, path, nonce, rank, peers, fds, agreement):
        self.path, self.nonce, self.rank = path, nonce, rank
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
                    request = json.loads(conn.recv(4096))
                    rank = request.get("rank")
                    if (
                        type(rank) is not int
                        or rank in served
                        or not 0 <= rank < len(self.peers)
                        or request.get("nonce") != self.nonce
                    ):
                        raise RuntimeError("LOCAL_CUDA_ENGRAM_INVALID_FD_REQUEST")
                    _peer(conn, self.peers[rank]["pid"])
                    payload = _json(
                        {"nonce": self.nonce, "rank": self.rank, "count": len(self.fds)}
                    )
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


def _receive_fds(path, nonce, owner, rank, peer_pid, count, timeout):
    fds = []
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as conn:
            conn.settimeout(timeout)
            conn.connect(path)
            _peer(conn, peer_pid)
            conn.sendall(_json({"nonce": nonce, "rank": rank}))
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
                or json.loads(payload)
                != {"nonce": nonce, "rank": owner, "count": count}
                or len(fds) != count
            ):
                raise RuntimeError("LOCAL_CUDA_ENGRAM_INVALID_FD_REPLY")
            return fds
    except BaseException:
        for fd in fds:
            os.close(fd)
        raise


class _LocalFdOps(CudaDriverOps):
    def __init__(self, device, native, sizes, granularity):
        super().__init__(device)
        self.native, self.sizes, self.granularity = native, iter(sizes), granularity
        self.imports = {}

    def import_handle(self, raw):
        imported = self.native.DeviceImport(
            struct.unpack("i", raw)[0], next(self.sizes), self.granularity, self.device
        )
        handle = imported.handle()
        self.imports[handle] = imported
        return handle

    def release(self, handle):
        self.imports[handle].close()
        del self.imports[handle]


class LocalCudaEngramStorage(FabricEngramStorage):
    """Reuse reader/graph-pin/drain contracts, with POSIX_FD instead of Fabric imports."""

    direct_device_scales = True

    def __init__(self):
        super().__init__()
        self._owners = []

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint,
        expected_layout,
        *,
        device,
        group=None,
        timeout_s=180,
        store=None,
        rank=None,
        world_size=None,
        gang_hosts=None,
    ):
        import torch
        import torch.distributed as dist

        from .engram_bootstrap import _load_native, retain_for_process
        from .engram_capacity import capacity_from_environ
        from .engram_checkpoint import _same_snapshot, inspect_checkpoint
        from .engram_publish import load_rows

        if store is None:
            if not dist.is_initialized():
                raise RuntimeError("LOCAL_CUDA_ENGRAM_REQUIRES_STARTUP_STORE")
            store = dist.distributed_c10d._get_default_store()
            rank, world_size = dist.get_rank(group), dist.get_world_size(group)
        if (
            type(rank) is not int
            or type(world_size) is not int
            or not 2 <= world_size <= 64
            or not 0 <= rank < world_size
        ):
            raise ValueError("cuda_local requires at least two real local GPU ranks")
        if gang_hosts is not None and (
            len(gang_hosts) != world_size or len(set(gang_hosts)) != 1
        ):
            raise ValueError("cuda_local does not support cross-host groups")
        target = (
            torch.device("cuda", device)
            if isinstance(device, int)
            else torch.device(device)
        )
        if target.type != "cuda" or target.index is None or not 0 < timeout_s <= 86400:
            raise ValueError(
                "cuda_local requires an explicit CUDA ordinal and bounded timeout"
            )
        result = cls()
        result.device, result.shard_index, result.shard_count = (
            target.index,
            rank,
            world_size,
        )
        result.expected_layout = expected_layout
        agreement = _Agreement(store, rank, world_size, time.monotonic() + timeout_s)
        server, directory, all_fds = None, None, []
        try:
            native = _load_native()
            if any(
                not hasattr(native, name)
                for name in ("DeviceOwner", "DeviceImport", "device_granularity")
            ):
                raise RuntimeError(
                    "cuda_local requires the prebuilt DEVICE/POSIX_FD native API"
                )
            plan = inspect_checkpoint(checkpoint, expected_layout)
            result.checkpoint_path = plan.root
            properties = torch.cuda.get_device_properties(target)
            identity = hashlib.sha256(
                _json(
                    {
                        "layout": asdict(expected_layout),
                        "index": asdict(plan.index),
                        "files": [asdict(file) for file in plan.files],
                    }
                )
            ).hexdigest()
            peers = agreement.exchange(
                "metadata",
                dict(
                    hostname=socket.gethostname(),
                    boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                    uuid=str(properties.uuid),
                    identity=identity,
                    pid=os.getpid(),
                    granularity=native.device_granularity(target.index),
                ),
            )
            if (
                len({(peer["hostname"], peer["boot_id"]) for peer in peers}) != 1
                or len({peer["uuid"] for peer in peers}) != world_size
                or len({peer["identity"] for peer in peers}) != 1
                or len({peer["granularity"] for peer in peers}) != 1
            ):
                raise ValueError(
                    "cuda_local requires identical checkpoint/layout and distinct same-host GPU ranks"
                )
            granularity = peers[0]["granularity"]
            plans = {
                layer: plan_shards(
                    rows, expected_layout.head_dim, world_size, granularity=granularity
                )
                for layer, rows in expected_layout.layers
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
                raise ValueError(
                    "cuda_local FD bundle exceeds the bounded SCM_RIGHTS contract"
                )
            max_allocation, max_total = capacity_from_environ()
            if (
                any(
                    size > min(max_allocation, native.max_allocation_bytes)
                    for size in sizes
                )
                or sum(sizes) > max_total
            ):
                raise ValueError("cuda_local shard exceeds allocation budget")
            result.local_physical_bytes = sum(sizes)
            with torch.cuda.device(target):
                if result.local_physical_bytes > torch.cuda.mem_get_info(target)[0]:
                    raise RuntimeError(
                        "cuda_local has insufficient free VRAM for this rank's shard"
                    )
            directory = tempfile.mkdtemp(prefix="rtp-engram-") if rank == 0 else None
            descriptions = agreement.exchange(
                "directory",
                dict(path=directory, nonce=uuid4().hex) if rank == 0 else None,
            )
            directory, nonce = descriptions[0]["path"], descriptions[0]["nonce"]
            stat = os.stat(directory, follow_symlinks=False)
            if stat.st_uid != os.getuid() or stat.st_mode & 0o777 != 0o700:
                raise RuntimeError(
                    "cuda_local Unix directory must be owned by this UID with mode0700"
                )
            snapshots = {file.relative_path: file for file in plan.files}
            for layer in plan.layers:
                shard = plans[layer.layer_id]
                start, end = shard.row_range(rank)
                for tensor, size in zip(
                    layer.tensors, (shard.weight_bytes, shard.scale_bytes)
                ):
                    agreement.remaining()
                    owner = native.DeviceOwner(size, target.index)
                    result._owners.append(owner)
                    load_rows(
                        owner,
                        Path(checkpoint),
                        tensor,
                        snapshots[tensor.relative_path],
                        start,
                        end,
                        tensor.shape[1],
                        chunk_bytes=min(64 << 20, native.max_load_chunk_bytes),
                        deadline=agreement.deadline,
                    )
                    fd = owner.export_fd()
                    os.set_inheritable(fd, False)
                    all_fds.append(fd)
            _same_snapshot(Path(checkpoint), plan.index, index=True)
            agreement.exchange("loaded", True)
            own_fds = tuple(all_fds)
            server = _FdServer(
                str(Path(directory) / f"{rank}.sock"),
                nonce,
                rank,
                peers,
                own_fds,
                agreement,
            )
            agreement.exchange("listeners", True)
            handles = []
            for owner_rank, peer in enumerate(peers):
                received = (
                    list(own_fds)
                    if owner_rank == rank
                    else _receive_fds(
                        str(Path(directory) / f"{owner_rank}.sock"),
                        nonce,
                        owner_rank,
                        rank,
                        peer["pid"],
                        len(sizes),
                        agreement.remaining(),
                    )
                )
                if owner_rank != rank:
                    all_fds.extend(received)
                handles.append(received)
            for index, layer in enumerate(plan.layers):
                shard = plans[layer.layer_id]
                ops = _LocalFdOps(
                    target.index,
                    native,
                    [
                        size
                        for _ in peers
                        for size in (shard.weight_bytes, shard.scale_bytes)
                    ],
                    granularity,
                )
                mapping = ShardedLayerMapping(
                    shard,
                    [struct.pack("i", fds[2 * index]) for fds in handles],
                    [struct.pack("i", fds[2 * index + 1]) for fds in handles],
                    target.index,
                    ops=ops,
                )
                result._mappings[layer.layer_id] = mapping
                mapping.attach()
                result._views[f"layers.{layer.layer_id}.engram.embed.weight"] = (
                    mapping.weight_view()[: shard.total_rows]
                )
                result._views[f"layers.{layer.layer_id}.engram.embed.scale"] = (
                    mapping.scale_view()[: shard.total_rows]
                )
            result._state = "VERIFYING"
            result._verify_samples(plan, plans, snapshots, target)
            if server.error is not None:
                raise server.error
            agreement.exchange("verified", True)
            server.close()
            server = None
            result.identity_key = f"cuda_local|{identity}|{nonce}"
            result._state = "READY"
            return result
        except BaseException as original:
            agreement.fail(original)
            result._state = "FAILED"
            try:
                result.close()
            except BaseException:
                retain_for_process(result)
                raise CleanupPendingError(
                    "cuda_local attach cleanup incomplete", result
                ) from original
            raise
        finally:
            try:
                if server is not None:
                    server.close()
            except BaseException as cleanup_error:
                # An unfinished sender may still reference these FD numbers.
                # Retain them instead of recycling descriptors underneath it.
                result._failed_fd_resources = (server, all_fds, directory)
                retain_for_process(result)
                raise CleanupPendingError(
                    "cuda_local FD cleanup incomplete", result
                ) from cleanup_error
            for fd in all_fds:
                os.close(fd)
            if rank == 0 and directory is not None:
                shutil.rmtree(directory, ignore_errors=True)

    def _verify_samples(self, plan, plans, snapshots, target):
        import torch

        from .engram_checkpoint import _open_tensor_source, _read_exact

        for layer in plan.layers:
            shard = plans[layer.layer_id]
            rows = sorted(
                {0, shard.total_rows // 2, shard.total_rows - 1}
                | {
                    row
                    for index in range(shard.shard_count)
                    for row in (
                        shard.row_range(index)[0],
                        shard.row_range(index)[1] - 1,
                    )
                }
            )
            for tensor in layer.tensors:
                columns = tensor.shape[1]
                with _open_tensor_source(
                    Path(plan.root), snapshots[tensor.relative_path]
                ) as (source, data_start):
                    expected = bytearray()
                    for row in rows:
                        source.seek(data_start + tensor.data_range[0] + row * columns)
                        expected.extend(_read_exact(source, columns))
                actual = (
                    self._views[tensor.name]
                    .index_select(0, torch.tensor(rows, device=target))
                    .cpu()
                )
                if actual.contiguous().view(torch.uint8).numpy().tobytes() != expected:
                    raise ValueError(
                        "cuda_local mapped GPU bytes differ from checkpoint samples"
                    )

    def require_model_layout(self, layer_ids, num_embeddings, head_dim):
        if (
            tuple(sorted(zip(layer_ids, num_embeddings))) != self.expected_layout.layers
            or head_dim != self.expected_layout.head_dim
        ):
            raise ValueError("cuda_local Engram layout differs from the model")

    def info(self):
        with self._lock:
            return dict(
                state=self._state,
                device=self.device,
                shard_count=self.shard_count,
                shard_index=self.shard_index,
                local_physical_bytes=getattr(self, "local_physical_bytes", 0),
                graph_pins=self._graph_pins,
                owners=[owner.info() for owner in self._owners],
                mappings={
                    layer: mapping.info() for layer, mapping in self._mappings.items()
                },
                imports=[
                    imported.info()
                    for mapping in self._mappings.values()
                    for imported in mapping.ops.imports.values()
                ],
            )

    def close(self):
        with self._lock:
            super().close()
            while self._owners:
                self._owners[-1].close()
                self._owners.pop()
