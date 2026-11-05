# SPDX-License-Identifier: Apache-2.0
"""Explicit EGM attach and graph lifetimes; no RPC or checkpoint loading.

Only operations submitted under ``use()`` participate in the drain contract.
Raw tensor views must not be captured in an unmanaged CUDA graph. This first
slice uses an injected native backend; it never compiles an extension in forward.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Mapping, Protocol

from .engram_manifest import (AllocationSlice, ExpectedLayout, Limits, TensorSpec,
                              ValidatedManifest, parse_for_attach)
from .engram_shard import ShardPlan, plan_shards
from .engram_shard_config import ShardConfig


class _ManagedMapping(Protocol):
    """Both native allocation imports and stitched layer spans own leases."""

    def info(self) -> dict: ...

    def close(self): ...


@dataclass(frozen=True)
class _LayerPlan:
    weight: TensorSpec
    scale: TensorSpec
    shards: ShardPlan


@dataclass(frozen=True)
class _AttachPlan:
    validated: ValidatedManifest
    allocations: tuple[AllocationSlice, ...]
    layers: tuple[_LayerPlan, ...]

    @classmethod
    def build(cls, validated):
        """Finish all geometry checks before any native resource is acquired."""
        if not isinstance(validated, ValidatedManifest):
            raise TypeError("attach execution requires a validated manifest")
        manifest = validated.manifest
        if manifest.backend != "fabric_egm":
            raise ValueError("FabricEngramStorage requires fabric_egm")
        allocations = {part.allocation_id: part
                       for tensor in manifest.tensors for part in tensor.shards}
        layers = []
        if any(len(t.shards) > 1 for t in manifest.tensors):
            ShardConfig(len(manifest.tensors[0].shards)).validate_manifest(manifest)
            by_name = {t.name: t for t in manifest.tensors}
            for weight in manifest.tensors:
                if not weight.name.endswith(".weight"):
                    continue
                scale = by_name[weight.name.removesuffix(".weight") + ".scale"]
                shards = plan_shards(
                    total_rows=weight.shape[0], shard_count=len(weight.shards),
                    dim=weight.shape[1], block=weight.quant_block,
                    granularity=weight.shards[0].allocation_granularity)
                for tensor, size in ((weight, shards.weight_bytes), (scale, shards.scale_bytes)):
                    for index, part in enumerate(tensor.shards):
                        if (part.allocation_granularity != shards.granularity
                                or part.allocation_bytes != size or part.byte_offset != 0
                                or (part.row_begin, part.row_end) != shards.row_range(index)):
                            raise ValueError("shard geometry does not match the mapping plan")
                layers.append(_LayerPlan(weight, scale, shards))
        return cls(validated, tuple(allocations[aid] for aid in sorted(allocations)), tuple(layers))


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
        self._mappings: dict[str, _ManagedMapping] = {}
        self._allocation_count = 0
        self._views = {}
        self._graph_pins = 0
        self._lookup_events = []
        self._state = "ATTACHING"

    @classmethod
    def attach(cls, manifest_bytes: bytes, *, expected_layout: ExpectedLayout,
               dataset_id: str, generation: str, handles: Mapping[str, bytes],
               device: int, native, limits: Limits = Limits(), shard_ops=None):
        validated = parse_for_attach(manifest_bytes, expected_layout, dataset_id, generation, limits)
        return cls._attach_validated(validated, handles=handles, device=device,
                                     native=native, shard_ops=shard_ops)

    @classmethod
    def _attach_validated(cls, validated: ValidatedManifest, *, handles: Mapping[str, bytes],
                          device: int, native, shard_ops=None):
        """Internal execution boundary; raw callers enter through attach()."""
        if type(device) is not int or device < 0:
            raise ValueError("device must be a local CUDA ordinal")
        plan = _AttachPlan.build(validated)
        if not isinstance(handles, Mapping) or set(handles) != {s.allocation_id for s in plan.allocations}:
            raise ValueError("handle allocation set does not match the manifest")
        if any(not isinstance(h, bytes) or not 0 < len(h) <= 1024 for h in handles.values()):
            raise ValueError("Fabric capabilities must be bounded opaque bytes")
        result = cls()
        result.validated = validated
        result.device = device
        result._allocation_count = validated.allocation_count
        try:
            result._execute_plan(plan, handles, native, shard_ops)
            result._state = "MAPPED"
            return result
        except BaseException as original:
            # No GPU work can have been submitted through this unpublished object.
            result._views.clear()
            failures = []
            for mapping in reversed(list(result._mappings.values())):
                try:
                    mapping.close()
                except Exception as error:
                    failures.append(type(error).__name__)
            result._state = "FAILED"
            if failures:
                raise CleanupPendingError("attach failed and native cleanup was incomplete", result) from original
            raise

    def _execute_plan(self, plan, handles, native, shard_ops):
        if plan.layers:
            from .engram_shard_storage import ShardedLayerMapping

            for layer in plan.layers:
                mapping = ShardedLayerMapping(
                    layer.shards,
                    [handles[s.allocation_id] for s in layer.weight.shards],
                    [handles[s.allocation_id] for s in layer.scale.shards],
                    self.device, ops=shard_ops)
                # Register ownership before attach: partial failures must be retryable.
                key = layer.weight.name.removesuffix(".weight")
                self._mappings[key] = mapping
                mapping.attach()
                self._views[layer.weight.name] = mapping.weight_view()
                self._views[layer.scale.name] = mapping.scale_view()
        else:
            # Preserve the standalone native Consumer primitive and its slice offsets.
            for part in plan.allocations:
                self._mappings[part.allocation_id] = native.Consumer(
                    handles[part.allocation_id], part.allocation_bytes,
                    part.allocation_granularity, self.device)
            for tensor in plan.validated.manifest.tensors:
                part = tensor.shards[0]
                self._views[tensor.name] = self._mappings[part.allocation_id].tensor_view(
                    list(tensor.shape), part.byte_offset)

    def require_model_layout(self, layer_ids, num_embeddings, head_dim):
        """Check the complete model against the contract validated at attach."""
        expected = self.validated.layout
        layers, rows = tuple(layer_ids), tuple(num_embeddings)
        if (len(layers) != len(rows) or tuple(sorted(zip(layers, rows))) != expected.layers
                or head_dim != expected.head_dim or expected.backend != "fabric_egm"):
            raise ValueError("external generation does not match the complete model Engram layout")

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

    def register_lookup_event(self, event):
        """Retain a side-stream completion fence until imports are drained."""
        with self.use():
            self._lookup_events.append(event)

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
                        allocation_count=self._allocation_count, mapping_count=len(self._mappings),
                        graph_pins=self._graph_pins,
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
            for event in self._lookup_events:
                event.synchronize()
            self._lookup_events.clear()
            self._views.clear()
            if any(m.info()["active_views"] for m in self._mappings.values()):
                raise StorageBusyError("external tensor views must be released before close")
            try:
                for mapping in reversed(list(self._mappings.values())):
                    mapping.close()
            except Exception:
                self._state = "FAILED"
                raise
            self._mappings.clear()
            self._state = "CLOSED"
