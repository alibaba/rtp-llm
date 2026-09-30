# SPDX-License-Identifier: Apache-2.0
# Adapted from RTP-LLM develop/dsv41_ced_release engram.py shared host tables.
"""Bounded Engram table staging and explicit mapped-host ownership.

Tables are ordinary Python owners, never nn.Module buffers. Workers must use
spawn, discard graphs referencing a table before close(), and keep the checkpoint
immutable. A failed CUDA drain/unregister retains the mapping and lease.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import mmap
import os
import shutil
import threading
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from rtp_llm.utils.safetensors_header import (
    SafetensorsSpan,
    named_safetensors_spans,
    safetensors_file_identity as _identity,
    unique_json_object,
)

_COPY_BYTES = 8 * 1024 * 1024
_FAILED_CLOSES = []


@dataclass(frozen=True)
class V41EngramTableSource:
    checkpoint_sha256: str
    layer_id: int
    weight: SafetensorsSpan
    scales: SafetensorsSpan

    @classmethod
    def from_checkpoint(cls, checkpoint, layer_id, rows, dim, checkpoint_sha256):
        if (
            layer_id not in (1, 14)
            or type(rows) is not int
            or rows <= 0
            or type(dim) is not int
            or dim <= 0
            or dim % 32
        ):
            raise ValueError("V4.1 Engram table requires layer 1/14 and group-32 rows")
        if (
            not isinstance(checkpoint_sha256, str)
            or len(checkpoint_sha256) != 64
            or any(c not in "0123456789abcdef" for c in checkpoint_sha256)
        ):
            raise ValueError("Engram requires a canonical deployment checkpoint SHA256")
        prefix = f"layers.{layer_id}.engram.embed."
        spans = named_safetensors_spans(
            checkpoint,
            {
                prefix + "weight": ((rows, dim), "F8_E4M3"),
                prefix + "scale": ((rows, dim // 32), "F8_E8M0"),
            },
        )
        return cls(
            checkpoint_sha256,
            layer_id,
            spans[prefix + "weight"],
            spans[prefix + "scale"],
        )

    @property
    def nbytes(self):
        return self.weight.size + self.scales.size

    @property
    def manifest(self):
        return json.dumps(
            dict(schema=1, **asdict(self)), sort_keys=True, separators=(",", ":")
        ).encode("ascii")

    @property
    def key(self):
        return hashlib.sha256(self.manifest).hexdigest()

    def check_sources(self):
        self.weight.check_source()
        self.scales.check_source()


def plan_stage_engram_tables(config, checkpoint, owned_layer_ids, checkpoint_sha256):
    """Header-only plan; a stage without Engram never opens checkpoint files."""
    owned = tuple(owned_layer_ids)
    if len(set(owned)) != len(owned) or any(
        type(i) is not int or not 0 <= i < config.text["num_hidden_layers"]
        for i in owned
    ):
        raise ValueError("Engram stage has invalid or duplicate layer ownership")
    return {
        layer: V41EngramTableSource.from_checkpoint(
            checkpoint, layer, rows, config.text["engram_head_dim"], checkpoint_sha256
        )
        for layer, rows in zip(
            config.text["engram_layer_ids"], config.text["engram_num_embeddings"]
        )
        if layer in owned
    }


@contextmanager
def _locked(path):
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def _live_leases(path):
    # The creation lock is held by callers. Kernel locks, not PID liveness,
    # identify active users and recover automatically after SIGKILL.
    active = False
    for lease in path.parent.glob(path.name + ".lease.*"):
        fd = os.open(lease, os.O_RDWR | os.O_NOFOLLOW)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                active = True
            else:
                lease.unlink()
        finally:
            os.close(fd)
    return active


def _copy_span(span, destination):
    with open(span.path, "rb", buffering=0) as source:
        if _identity(os.fstat(source.fileno())) != span.file_identity:
            raise RuntimeError("Engram checkpoint changed before table copy")
        source.seek(span.offset)
        remaining = span.size
        while remaining:
            chunk = source.read(min(remaining, _COPY_BYTES))
            if not chunk:
                raise ValueError("Engram checkpoint ended during table copy")
            destination.write(chunk)
            remaining -= len(chunk)
        if _identity(os.fstat(source.fileno())) != span.file_identity:
            raise RuntimeError("Engram checkpoint changed during table copy")


class _CudaArrayView:
    def __init__(self, owner, shape, pointer):
        self.owner = owner
        self.__cuda_array_interface__ = {
            "shape": shape,
            "strides": None,
            "typestr": "|u1",
            "data": (pointer, False),
            "version": 3,
        }


class _PinnedSharedTable:
    """One shared table allocation; CUDA registration is an explicit second step."""

    def __init__(
        self, source: V41EngramTableSource, *, max_bytes: int, directory="/dev/shm"
    ):
        self._mutex = threading.RLock()
        self._pid = os.getpid()
        self._closed = True
        self._closing = False
        self._mapping = None
        self._storage = self._weight = self._scales = None
        self._cuda_weight = self._cuda_scales = None
        self._runtime = self._device = None
        self._lease_fd = None
        self.source = source
        if type(max_bytes) is not int or not 0 < source.nbytes <= max_bytes:
            raise ValueError("Engram table exceeds the admitted host byte budget")
        source.check_sources()
        root = Path(directory).resolve()
        self._path = root / f"rtp_llm_engram_{os.getuid()}_{source.key}"
        self._lock_path = Path(str(self._path) + ".lock")
        self._ready_path = Path(str(self._path) + ".json")
        self._lease_path = Path(str(self._path) + f".lease.{uuid.uuid4().hex}")
        with _locked(self._lock_path):
            source.check_sources()
            live = _live_leases(self._path)
            ready = self._path.exists() and self._ready_path.exists()
            if ready:
                ready = (
                    self._path.stat().st_size == source.nbytes
                    and self._ready_path.read_bytes() == source.manifest
                )
            if not ready:
                if live:
                    raise RuntimeError(
                        "Engram shared table metadata is invalid while readers are active"
                    )
                self._create()
            try:
                fd = os.open(self._path, os.O_RDWR | os.O_NOFOLLOW)
                try:
                    self._mapping = mmap.mmap(
                        fd, source.nbytes, access=mmap.ACCESS_WRITE
                    )
                finally:
                    os.close(fd)
                self._lease_fd = os.open(
                    self._lease_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600
                )
                fcntl.flock(self._lease_fd, fcntl.LOCK_EX)
                self._storage = torch.frombuffer(self._mapping, dtype=torch.uint8)
                self._weight = self._storage[: source.weight.size].view(
                    source.weight.shape
                )
                self._scales = self._storage[source.weight.size :].view(
                    source.scales.shape
                )
                self._closed = False
            except BaseException:
                self._storage = self._weight = self._scales = None
                if self._mapping is not None:
                    self._mapping.close()
                    self._mapping = None
                if self._lease_fd is not None:
                    os.close(self._lease_fd)
                    self._lease_fd = None
                self._lease_path.unlink(missing_ok=True)
                if not _live_leases(self._path):
                    self._ready_path.unlink(missing_ok=True)
                    self._path.unlink(missing_ok=True)
                raise

    def _create(self):
        partial = Path(str(self._path) + ".partial")
        ready_partial = Path(str(self._ready_path) + ".partial")
        partial.unlink(missing_ok=True)
        ready_partial.unlink(missing_ok=True)
        self._path.unlink(missing_ok=True)
        self._ready_path.unlink(missing_ok=True)
        if shutil.disk_usage(self._path.parent).free < self.source.nbytes:
            raise RuntimeError("Engram shared storage has insufficient free bytes")
        try:
            fd = os.open(partial, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            with os.fdopen(fd, "wb") as destination:
                # Reserve before writes; ENOSPC must be an exception, not mmap SIGBUS.
                os.posix_fallocate(destination.fileno(), 0, self.source.nbytes)
                _copy_span(self.source.weight, destination)
                _copy_span(self.source.scales, destination)
                destination.flush()
                os.fsync(destination.fileno())
            self.source.check_sources()
            ready_partial.write_bytes(self.source.manifest)
            os.chmod(ready_partial, 0o600)
            os.replace(partial, self._path)
            os.replace(ready_partial, self._ready_path)
        except BaseException:
            partial.unlink(missing_ok=True)
            ready_partial.unlink(missing_ok=True)
            raise

    def _check_open(self):
        if os.getpid() != self._pid:
            raise RuntimeError(
                "Engram table handles cannot be used after fork; use spawn"
            )
        if self._closed or self._closing:
            raise RuntimeError(
                "Engram table is closed or waiting for failed CUDA teardown"
            )

    def register_cuda(self, device):
        with self._mutex:
            self._check_open()
            device = torch.device(device)
            if device.type != "cuda":
                raise ValueError("Engram registration requires a CUDA device")
            if device.index is None:
                device = torch.device("cuda", torch.cuda.current_device())
            if self._runtime is not None:
                if self._device != device:
                    raise ValueError(
                        "Engram handle is already registered on another device"
                    )
                return
            from cuda.bindings import runtime

            with torch.cuda.device(device):
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "Engram host registration must precede graph capture"
                    )
                (error,) = runtime.cudaHostRegister(
                    self._storage.data_ptr(), self.source.nbytes, 3
                )
                if int(error):
                    raise RuntimeError(f"Engram cudaHostRegister failed: {error}")
                self._runtime, self._device = runtime, device
                try:
                    views = []
                    for tensor in (self._weight, self._scales):
                        error, pointer = runtime.cudaHostGetDevicePointer(
                            tensor.data_ptr(), 0
                        )
                        if int(error):
                            raise RuntimeError(
                                f"Engram cudaHostGetDevicePointer failed: {error}"
                            )
                        view = torch.as_tensor(
                            _CudaArrayView(self, tuple(tensor.shape), pointer),
                            device=device,
                        )
                        if view.data_ptr() != pointer:
                            raise RuntimeError(
                                "Engram UVA view unexpectedly copied the host table"
                            )
                        views.append(view)
                    self._cuda_weight, self._cuda_scales = views
                except BaseException:
                    self.close()
                    raise

    def close(self):
        with self._mutex:
            if os.getpid() != self._pid:
                raise RuntimeError("Engram table handles cannot be closed after fork")
            if self._closed:
                return
            self._closing = True
            if self._runtime is not None:
                with torch.cuda.device(self._device):
                    if torch.cuda.is_current_stream_capturing():
                        raise RuntimeError(
                            "Engram teardown cannot run during graph capture"
                        )
                    torch.cuda.synchronize(self._device)
                    (error,) = self._runtime.cudaHostUnregister(
                        self._storage.data_ptr()
                    )
                    if int(error):
                        raise RuntimeError(f"Engram cudaHostUnregister failed: {error}")
                self._runtime = None
            self._cuda_weight = self._cuda_scales = None
            self._storage = self._weight = self._scales = None
            if self._mapping is not None:
                self._mapping.close()
                self._mapping = None
            with _locked(self._lock_path):
                if self._lease_fd is not None:
                    os.close(self._lease_fd)
                    self._lease_fd = None
                self._lease_path.unlink(missing_ok=True)
                if not _live_leases(self._path):
                    self._ready_path.unlink(missing_ok=True)
                    self._path.unlink(missing_ok=True)
            self._closed = True

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, *args):
        self.close()

    def __getstate__(self):
        raise TypeError("Engram table handles cannot be pickled; attach in the worker")

    def __del__(self):
        if getattr(self, "_closed", True) or os.getpid() != self._pid:
            return
        try:
            self.close()
        except BaseException:
            # Do not let GC unmap pages still registered or used by CUDA.
            _FAILED_CLOSES.append(self)
            logging.getLogger(__name__).exception(
                "Engram retained storage after failed teardown"
            )


class HostEngramEmbedding:
    """A table owner reference, deliberately outside nn.Module buffer traversal."""

    def __init__(self, table: _PinnedSharedTable):
        self._table = table

    def __call__(self, indices, valid_rows, *, backend="triton"):
        table = self._table
        with table._mutex:
            table._check_open()
            if (
                indices.ndim != 2
                or indices.shape[1] <= 0
                or indices.dtype not in (torch.int32, torch.int64)
                or valid_rows.shape != (indices.shape[0],)
                or valid_rows.dtype != torch.bool
                or valid_rows.device != indices.device
                or not valid_rows.is_contiguous()
            ):
                raise ValueError(
                    "Engram lookup requires integer [tokens, heads] and a device bool [tokens] mask"
                )
            if backend == "eager":
                if indices.device.type != "cpu":
                    raise ValueError("Engram eager lookup is a CPU diagnostic path")
                dim = table.source.weight.shape[1]
                output = torch.zeros((*indices.shape, dim), dtype=torch.bfloat16)
                live = (
                    (indices >= 0)
                    & (indices < table.source.weight.shape[0])
                    & valid_rows[:, None]
                )
                selected = indices[live].long()
                raw = (
                    table._weight.index_select(0, selected)
                    .view(torch.float8_e4m3fn)
                    .float()
                )
                encoded = table._scales.index_select(0, selected)
                scales = torch.exp2(encoded.float() - 127)
                scales = torch.where(encoded == 255, float("nan"), scales)
                output[live] = (raw * scales.repeat_interleave(32, dim=1)).to(
                    torch.bfloat16
                )
                return output
            if backend != "triton":
                raise ValueError("Unknown Engram lookup backend")
            if table._runtime is None or indices.device != table._device:
                raise ValueError(
                    "Engram GPU lookup requires registration on the query device"
                )
            from rtp_llm.models_py.modules.dsv4._v41_engram_lookup_triton import (
                lookup_rows,
            )

            return lookup_rows(
                table._cuda_weight, table._cuda_scales, indices, valid_rows
            )
