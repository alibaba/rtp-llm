# SPDX-License-Identifier: Apache-2.0
"""Explicit EGM attach and graph lifetimes; no RPC or checkpoint loading.

Only operations submitted under ``use()`` participate in the drain contract.
Raw tensor views must not be captured in an unmanaged CUDA graph. This first
slice uses an injected native backend; it never compiles an extension in forward.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Mapping

from .engram_manifest import ExpectedLayout, Limits, parse_manifest, validate_for_attach


class StorageBusyError(RuntimeError):
    pass


class CleanupPendingError(RuntimeError):
    """Caller must retain/retry this resource or terminate the worker."""

    def __init__(self, message, resource):
        super().__init__(message)
        self.resource = resource


class FabricEngramStorage:
    """One worker's mappings of one immutable, complete Engram generation."""

    def __init__(self):
        self._lock = threading.RLock()
        self._mappings = {}
        self._shard_mappings = []
        self._allocation_count = 0
        self._views = {}
        self._graph_pins = 0
        self._state = "ATTACHING"

    @classmethod
    def attach(cls, manifest_bytes: bytes, *, expected_layout: ExpectedLayout,
               dataset_id: str, generation: str, handles: Mapping[str, bytes],
               device: int, native, limits: Limits = Limits(), shard_ops=None):
        if type(device) is not int or device < 0:
            raise ValueError("device must be a local CUDA ordinal")
        verified = validate_for_attach(parse_manifest(manifest_bytes, limits),
                                       expected_layout, dataset_id, generation, limits)
        if verified.manifest.backend != "fabric_egm":
            raise ValueError("FabricEngramStorage requires fabric_egm")
        slices = {s.allocation_id: s
                  for t in verified.manifest.tensors for s in t.shards}
        if not isinstance(handles, Mapping) or set(handles) != set(slices):
            raise ValueError("handle allocation set does not match the manifest")
        if any(not isinstance(h, bytes) for h in handles.values()):
            raise ValueError("Fabric capabilities must be opaque bytes")
        result = cls()
        result.validated = verified
        result.device = device
        result._allocation_count = len(slices)
        try:
            if any(len(t.shards) > 1 for t in verified.manifest.tensors):
                result._attach_sharded(verified.manifest, handles, device, shard_ops)
            else:
                for aid, part in sorted(slices.items()):
                    result._mappings[aid] = native.Consumer(
                        handles[aid], part.allocation_bytes,
                        part.allocation_granularity, device)
                for tensor in verified.manifest.tensors:
                    part = tensor.shards[0]
                    result._views[tensor.name] = result._mappings[part.allocation_id].tensor_view(
                        list(tensor.shape), part.byte_offset)
            result._state = "MAPPED"
            return result
        except BaseException as original:
            # No GPU work can have been submitted through this unpublished object.
            result._views.clear()
            failures = []
            for mapping in reversed(result._shard_mappings):
                try:
                    mapping.close()
                except Exception as error:
                    failures.append(type(error).__name__)
            for mapping in reversed(list(result._mappings.values())):
                try:
                    mapping.close()
                except Exception as error:
                    failures.append(type(error).__name__)
            result._state = "FAILED"
            if failures:
                raise CleanupPendingError("attach failed and native cleanup was incomplete", result) from original
            raise

    def _attach_sharded(self, manifest, handles, device, shard_ops=None):
        """Stitch every tensor's shards into two contiguous VA spans."""
        from .engram_shard import plan_shards
        from .engram_shard_config import ShardConfig

        ShardConfig(len(manifest.tensors[0].shards)).validate_manifest(manifest)
        from .engram_shard_storage import ShardedLayerMapping

        by_name = {t.name: t for t in manifest.tensors}
        paired = set()
        for tensor in manifest.tensors:
            if tensor.name in paired:
                continue
            prefix, role = tensor.name.rsplit(".", 1)
            if role == "scale":
                continue
            if role != "weight":
                raise ValueError("sharded manifest requires weight/scale pairs")
            weight = tensor
            scale = by_name.get(prefix + ".scale")
            if scale is None:
                raise ValueError("sharded manifest is missing a scale tensor")
            plan = plan_shards(
                total_rows=weight.shape[0], shard_count=len(weight.shards),
                dim=weight.shape[1],
                block=weight.quant_block,
                granularity=weight.shards[0].allocation_granularity)
            if plan.weight_bytes != weight.shards[0].allocation_bytes:
                raise ValueError("weight shard geometry mismatch")
            if plan.scale_bytes != scale.shards[0].allocation_bytes:
                raise ValueError("scale shard geometry mismatch")
            if len(scale.shards) != plan.shard_count:
                raise ValueError("scale shard count mismatch")
            for index, (weight_shard, scale_shard) in enumerate(
                    zip(weight.shards, scale.shards)):
                if any(part.allocation_granularity != plan.granularity for part in (weight_shard, scale_shard)):
                    raise ValueError("shard allocation granularity mismatch")
                if weight_shard.allocation_bytes != plan.weight_bytes or scale_shard.allocation_bytes != plan.scale_bytes:
                    raise ValueError("nonuniform shard allocations")
                expected = plan.row_range(index)
                if (weight_shard.row_begin, weight_shard.row_end) != expected:
                    raise ValueError("weight shard row range mismatch")
                if (scale_shard.row_begin, scale_shard.row_end) != expected:
                    raise ValueError("scale shard row range mismatch")
            mapping = ShardedLayerMapping(
                plan,
                [handles[s.allocation_id] for s in weight.shards],
                [handles[s.allocation_id] for s in scale.shards],
                device, ops=shard_ops)
            self._shard_mappings.append(mapping)
            mapping.attach()
            self._views[weight.name] = mapping.weight_view()
            self._views[scale.name] = mapping.scale_view()
            paired.update((weight.name, scale.name))
        if paired != set(by_name):
            raise ValueError("sharded manifest has an unpaired tensor")

    def verify(self, verify_views):
        """Verify GPU reads before READY; callback must drain and return True.

        The authenticated control channel binds capabilities to allocations.
        Metadata hashes alone cannot prove the physical bytes. The caller's
        verifier compares device-side samples to trusted expected data; full
        content hashing belongs to the owner loading path.
        """
        from types import MappingProxyType
        with self._lock:
            if self._state != "MAPPED":
                raise RuntimeError("verification requires freshly MAPPED storage")
            self._state = "VERIFYING"
            try:
                if verify_views(MappingProxyType(self._views), self.validated) is not True:
                    raise ValueError("GPU content verifier must explicitly return True")
            except BaseException:
                self._state = "FAILED"
                raise
            if self._state != "VERIFYING":
                raise RuntimeError("verification state was changed unexpectedly")
            self._state = "READY"

    def _require_ready(self):
        if self._state != "READY":
            raise RuntimeError("Engram storage is not READY")

    @contextmanager
    def use(self):
        """Serialize enqueue vs drain. Hold until GPU work has been submitted."""
        with self._lock:
            self._require_ready()
            yield

    def table_views(self, layer_id: int):
        with self.use():
            prefix = f"layers.{layer_id}.engram.embed."
            try:
                return self._views[prefix + "weight"], self._views[prefix + "scale"]
            except KeyError:
                raise KeyError("requested layer is absent from this generation") from None

    def _pin_graph(self):
        with self.use():
            self._graph_pins += 1

    def _unpin_graph(self):
        with self._lock:
            if self._graph_pins <= 0:
                raise RuntimeError("unbalanced graph lifetime")
            self._graph_pins -= 1

    def info(self):
        with self._lock:
            return dict(state=self._state, generation=self.validated.manifest.generation,
                        dataset_id=self.validated.manifest.dataset_id,
                        physical_bytes=self.validated.physical_bytes,
                        allocation_count=self._allocation_count, graph_pins=self._graph_pins,
                        allocations={aid: mapping.info() for aid, mapping in self._mappings.items()})

    def close(self):
        """Stop enqueue, drop owned views, then drain and release imports.

        Live graph pins reject close without changing READY. Other externally
        held views leave the object CLOSING: release them and retry close.
        """
        with self._lock:
            if self._state == "CLOSED":
                return
            if self._state == "VERIFYING":
                raise StorageBusyError("content verification is still in progress")
            if self._graph_pins:
                raise StorageBusyError("captured graphs still hold this storage")
            self._state = "CLOSING"
            self._views.clear()
            if any(m.info()["active_views"] for m in [*self._mappings.values(), *self._shard_mappings]):
                raise StorageBusyError("external tensor views must be released before close")
            try:
                for mapping in reversed(self._shard_mappings):
                    mapping.close()
                for mapping in reversed(list(self._mappings.values())):
                    mapping.close()
            except Exception:
                self._state = "FAILED"
                raise
            self._shard_mappings.clear()
            self._mappings.clear()
            self._state = "CLOSED"


class CapturedEngramGraph:
    """A CUDA graph that pins every storage it can read until reset + drain.

    ``forward`` must use only the supplied storages. Keep graph ownership here:
    do not expose/reuse the private graph outside this wrapper. The wrapper has
    no unsafe GC-driven unmap; explicit close is required to release its pins.
    """

    def __init__(self):
        self._closed = False
        self._replay_allowed = False
        self._graph = None
        self._storages = ()
        self.output = None
        self._lock = threading.RLock()

    @classmethod
    def capture(cls, forward, storages, *, warmup=3):
        import torch
        if type(warmup) is not int or warmup < 1:
            raise ValueError("warmup must be positive")
        unique = tuple(dict.fromkeys(storages))
        if not unique or len({s.device for s in unique}) != 1:
            raise ValueError("capture requires storages on one CUDA device")
        result = cls()
        result.device = unique[0].device
        pinned = []
        try:
            for storage in unique:
                storage._pin_graph()
                pinned.append(storage)
            result._storages = tuple(pinned)
            with torch.cuda.device(result.device):
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(warmup):
                        forward()
                stream.synchronize()
                result._graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(result._graph, stream=stream):
                    result.output = forward()
            result._replay_allowed = True
            return result
        except BaseException as original:
            # Never release pins unless all work and partial capture are drained.
            result._storages = tuple(pinned)
            try:
                result.close()
            except BaseException:
                raise CleanupPendingError("capture failed and cleanup remains pending", result) from original
            raise

    def replay(self):
        from contextlib import ExitStack
        with self._lock:
            if not self._replay_allowed:
                raise RuntimeError("graph is not ready for replay")
            # Fixed order avoids deadlocks when a graph reads multiple storages.
            with ExitStack() as scope:
                for storage in sorted(self._storages, key=id):
                    scope.enter_context(storage.use())
                import torch
                with torch.cuda.device(self.device):
                    self._graph.replay()
            return self.output

    def close(self):
        import torch
        with self._lock:
            if self._closed:
                return
            self._replay_allowed = False
            with torch.cuda.device(self.device):
                torch.cuda.synchronize()
                if self._graph is not None:
                    self._graph.reset()
                    self._graph = None
            self.output = None
            for storage in self._storages:
                storage._unpin_graph()
            self._storages = ()
            self._closed = True
