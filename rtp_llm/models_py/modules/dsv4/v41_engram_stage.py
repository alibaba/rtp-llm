"""Explicit worker lifetime and per-forward state for stage-local Engram layers."""

import os
import threading
from dataclasses import dataclass

import torch

from rtp_llm.models_py.modules.dsv4.v41_engram_hash import (
    V41EngramHasher,
    V41EngramHashes,
)
from rtp_llm.models_py.modules.dsv4.v41_engram_host import (
    HostEngramEmbedding,
    _PinnedSharedTable,
)
from rtp_llm.models_py.modules.dsv4.v41_engram_weights import load_engram_module
from rtp_llm.models_py.modules.dsv4.v41_mxfp8 import _require_sm120


@dataclass(frozen=True)
class V41EngramForward:
    owner: object
    hashes: V41EngramHashes


class V41EngramStage:
    """Own one worker's table leases/modules, never module-registered host storage.

    Construct in a spawned worker, then initialize before publishing the model.
    Quotas cover combined host tables and resident projection/prepared scales.
    The launcher must separately admit physical host/cgroup/NUMA capacity, startup
    staging, provider workspace and other model allocations. Graph capture remains
    rejected until the engine coordinates graph lifetimes with this owner.
    """

    def __init__(
        self,
        sources,
        hasher,
        *,
        device,
        max_host_bytes,
        max_weight_bytes,
        directory="/dev/shm",
        chunk_tokens=128,
        backend="flashinfer",
    ):
        self._pid = os.getpid()
        self._mutex = threading.RLock()
        self._state = "new"
        self._owner = object()
        self._sources = dict(sources)
        self._tables, self._modules = {}, {}
        self._hasher = hasher
        self.device = torch.device(device)
        self.directory, self.chunk_tokens, self.backend = (
            directory,
            chunk_tokens,
            backend,
        )
        if backend not in ("flashinfer", "eager"):
            raise ValueError("Unknown Engram stage backend")
        if type(chunk_tokens) is not int or chunk_tokens <= 0:
            raise ValueError("Engram chunk_tokens must be a positive integer")
        for budget in (max_host_bytes, max_weight_bytes):
            if type(budget) is not int or budget < 0:
                raise ValueError(
                    "Engram aggregate budgets must be nonnegative integers"
                )
        self.host_bytes = sum(source.table.nbytes for source in self._sources.values())
        self.weight_bytes = sum(
            self._module_bytes(source) for source in self._sources.values()
        )
        if self.host_bytes > max_host_bytes or self.weight_bytes > max_weight_bytes:
            raise ValueError("Engram stage exceeds its aggregate host or weight budget")
        if not self._sources:
            if hasher is not None:
                raise ValueError("A stage without Engram must not own hash constants")
            self._hash_indices = {}
            return
        if not isinstance(hasher, V41EngramHasher):
            raise ValueError("An Engram stage requires its explicit-window hasher")
        if (backend == "eager") != (self.device.type == "cpu"):
            raise ValueError(
                "Engram eager backend requires CPU; FlashInfer requires CUDA"
            )
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        if any(buffer.device != self.device for buffer in hasher.buffers()):
            raise ValueError("Engram hash constants must be on the final worker device")
        self._hash_indices = {}
        for layer_id, source in self._sources.items():
            if (
                type(layer_id) is not int
                or layer_id != source.layer_id
                or layer_id not in hasher.layout.layer_ids
            ):
                raise ValueError(
                    "Engram source ownership disagrees with the hash layout"
                )
            index = hasher.layout.layer_ids.index(layer_id)
            if (
                source.table.weight.shape
                != (hasher.layout.num_embeddings[index], hasher.layout.head_dim)
                or source.weight.shape[1]
                != hasher.layout.n_hash_cols * hasher.layout.head_dim
            ):
                raise ValueError(
                    "Engram table/projection geometry disagrees with the hash layout"
                )
            self._hash_indices[layer_id] = index
        if (
            len({source.table.checkpoint_sha256 for source in self._sources.values()})
            != 1
        ):
            raise ValueError("Engram stage cannot mix deployment checkpoint identities")

    @property
    def layer_ids(self):
        return tuple(sorted(self._sources))

    @property
    def state(self):
        return self._state

    def _module_bytes(self, source):
        return source.weight_bytes + (
            source.packed_scale_bytes if self.backend == "flashinfer" else 0
        )

    def _check_worker(self):
        if os.getpid() != self._pid:
            raise RuntimeError(
                "Engram stage cannot be used after fork; initialize in a spawned worker"
            )

    def _check_ready(self):
        self._check_worker()
        if self._state != "ready":
            raise RuntimeError("Engram stage is not ready or is closing")

    def _check_capture(self):
        if self._sources and self.device.type == "cuda":
            with torch.cuda.device(self.device):
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "Engram stage graph capture requires integrated worker lifecycle"
                    )

    def initialize(self):
        with self._mutex:
            self._check_worker()
            if self._state == "ready":
                return self
            if self._state != "new":
                raise RuntimeError("Engram stage initialization requires a new owner")
            if self._sources and self.backend == "flashinfer":
                _require_sm120(self.device)
                self._check_capture()
            for source in self._sources.values():
                source.check_sources()
            self._state = "initializing"
            try:
                for layer_id in self.layer_ids:
                    source = self._sources[layer_id]
                    table = _PinnedSharedTable(
                        source.table,
                        max_bytes=source.table.nbytes,
                        directory=self.directory,
                    )
                    self._tables[layer_id] = table
                    if self.backend == "flashinfer":
                        table.register_cuda(self.device)
                    self._modules[layer_id] = load_engram_module(
                        source,
                        HostEngramEmbedding(table),
                        device=self.device,
                        max_bytes=self._module_bytes(source),
                        chunk_tokens=self.chunk_tokens,
                        backend=self.backend,
                    )
            except BaseException as failure:
                try:
                    self.close()
                except BaseException as cleanup:
                    raise RuntimeError(
                        "Engram startup failed and cleanup must be retried on this owner: "
                        + str(cleanup)
                    ) from failure
                raise
            self._state = "ready"
            return self

    def prepare(self, windows):
        """Return request-local hashes once; nothing is retained on the stage."""
        with self._mutex:
            self._check_ready()
            if not self._sources:
                return None
            self._check_capture()
            backend = "eager" if self.backend == "eager" else "triton"
            return V41EngramForward(self._owner, self._hasher(windows, backend=backend))

    def apply(self, layer_id, hidden, frame, *, out=None):
        with self._mutex:
            self._check_ready()
            if type(layer_id) is not int or layer_id not in self._modules:
                raise ValueError("Engram layer is not owned by this stage")
            if (
                not isinstance(frame, V41EngramForward)
                or frame.owner is not self._owner
            ):
                raise ValueError(
                    "Engram forward hashes belong to a different stage owner"
                )
            hashes = frame.hashes
            if (
                hidden.ndim != 3
                or hashes.ids.ndim != 3
                or hashes.ids.shape[1:]
                != (len(self._hasher.layout.layer_ids), self._hasher.layout.n_hash_cols)
                or hashes.ids.shape[0] != hidden.shape[0]
            ):
                raise ValueError(
                    "Engram forward hashes do not match the residual rows/layout"
                )
            return self._modules[layer_id](
                hidden,
                hashes.ids[:, self._hash_indices[layer_id]],
                hashes.valid_rows,
                out=out,
            )

    def validate_forward(self, frame, valid_rows, *, dim):
        """Preflight a model forward before any earlier layer can publish caches."""
        with self._mutex:
            self._check_ready()
            self._check_capture()
            if not self._sources:
                if frame is not None:
                    raise ValueError("A stage without Engram cannot receive hashes")
                return
            if (
                not isinstance(frame, V41EngramForward)
                or frame.owner is not self._owner
            ):
                raise ValueError(
                    "Engram forward hashes belong to a different stage owner"
                )
            hashes = frame.hashes
            if (
                valid_rows.ndim != 1
                or valid_rows.dtype != torch.bool
                or valid_rows.device != self.device
                or hashes.ids.shape
                != (
                    valid_rows.numel(),
                    len(self._hasher.layout.layer_ids),
                    self._hasher.layout.n_hash_cols,
                )
                or hashes.ids.dtype != torch.int64
                or hashes.ids.device != self.device
                or hashes.valid_rows.device != self.device
                or not torch.equal(hashes.valid_rows, valid_rows)
                or any(
                    module.dim != dim or module.lanes != 4 or module.eps != 1e-20
                    for module in self._modules.values()
                )
            ):
                raise ValueError(
                    "Engram forward mask or module geometry disagrees with the stage"
                )

    def close(self):
        with self._mutex:
            self._check_worker()
            if self._state == "closed":
                return
            self._check_capture()
            self._state = "closing"
            failures = []
            for layer_id in reversed(tuple(self._tables)):
                try:
                    self._tables[layer_id].close()
                except BaseException as failure:
                    failures.append(failure)
                else:
                    del self._tables[layer_id]
            if failures:
                # Keep projections/hash constants until every queued GPU use drains.
                raise RuntimeError(
                    "Engram stage teardown failed; retain owner and retry close"
                ) from failures[0]
            self._modules.clear()
            self._hasher = None
            self._state = "closed"

    def __enter__(self):
        return self.initialize()

    def __exit__(self, *args):
        self.close()

    def __getstate__(self):
        raise TypeError(
            "Engram stage owners cannot be pickled; initialize in the worker"
        )
