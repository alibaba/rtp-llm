# SPDX-License-Identifier: Apache-2.0
"""Fabric storage attachment, tensor views, and CUDA graph lifetimes.

Only operations submitted under ``use()`` participate in the drain contract.
Raw tensor views must not be captured in an unmanaged CUDA graph. The native
backend owns allocations and mappings; this module coordinates their lifetimes.
"""
from __future__ import annotations

import hashlib
import logging
import threading
import time
from contextlib import contextmanager
from typing import Mapping

from .config import storage_host_count
from .manifest import ValidatedManifest

_LOGGER = logging.getLogger(__name__)


class BootstrapError(ValueError):
    """Bounded error category; never formats capability bytes or payloads."""

    def __init__(self, code):
        self.code = code
        super().__init__(code)


def _load_native():
    module_name = "libfabric_memory"
    try:
        from rtp_llm.ops import load_engram_ops

        return load_engram_ops()
    except (ImportError, OSError) as error:
        if isinstance(error, ModuleNotFoundError) and error.name == module_name:
            code, category = "PREBUILT_ENGRAM_NATIVE_MODULE_REQUIRED", "module_missing"
        elif isinstance(error, ModuleNotFoundError) and error.name:
            code, category = "NATIVE_MODULE_DEPENDENCY_MISSING", "dependency_missing"
        elif isinstance(error, OSError):
            code, category = "NATIVE_MODULE_LOAD_FAILED", "shared_library_failure"
        else:
            code, category = "NATIVE_MODULE_LOAD_FAILED", "import_failure"
        # Keep the local loader cause (missing dependency / ABI symbol). Only
        # cross-rank diagnostics are restricted to bounded protocol fields.
        try:
            logging.getLogger(__name__).error(
                "engram event=native_load_failed phase=native_load module=%s "
                "error_code=%s error_class=%s",
                module_name,
                code,
                category,
            )
        except Exception:
            pass  # An unhealthy logging handler must not hide the loader category.
        raise BootstrapError(code) from error


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
        self._allocation_count = 0
        self._shard_count = 0
        self._views = {}
        self._graph_pins = 0
        self._lookup_events = []
        self._state = "ATTACHING"

    @property
    def device_resident_tables(self):
        """Both weight and scales already reside in HBM; never cache a copy."""
        return self.memory_kind == "gpu"

    @classmethod
    def _attach_validated(
        cls,
        validated: ValidatedManifest,
        *,
        handles: Mapping[str, bytes],
        device: int,
        native,
    ):
        """Map allocations after the layout has been bound to the model."""
        if type(device) is not int or device < 0:
            raise ValueError("device must be a local CUDA ordinal")
        if not isinstance(validated, ValidatedManifest):
            raise TypeError("attach requires a validated manifest")
        tensors = validated.manifest.tensors
        if not isinstance(handles, Mapping) or set(handles) != {
            part.allocation_id for tensor in tensors for part in tensor.shards
        }:
            raise ValueError("handle allocation set does not match the manifest")
        if any(
            not isinstance(h, bytes) or not 0 < len(h) <= 1024 for h in handles.values()
        ):
            raise ValueError("Fabric capabilities must be bounded opaque bytes")
        result = cls()
        result.validated = validated
        result.device = device
        result.memory_kind = (
            "gpu" if validated.manifest.backend == "fabric_gpu" else "host"
        )
        result._allocation_count = validated.allocation_count
        result._shard_count = storage_host_count(validated.manifest)
        started = time.monotonic()
        result._log_state("storage_attach_started", "attach", started)
        try:
            result._map_tensors(tensors, handles, native)
            result._state = "MAPPED"
            result._log_state("storage_attached", "attach", started)
            return result
        except BaseException as original:
            # No GPU work can have been submitted through this unpublished object.
            result._views.clear()
            failures = []
            for mapping in reversed(list(result._mappings.values())):
                try:
                    mapping.close()
                except BaseException as error:
                    failures.append(type(error).__name__)
            result._state = "FAILED"
            result._log_state(
                "storage_attach_failed",
                "attach",
                started,
                original,
                cleanup_pending=bool(failures),
            )
            if failures:
                raise CleanupPendingError(
                    "attach failed and native cleanup was incomplete", result
                ) from original
            raise

    def _map_tensors(self, tensors, handles, native):
        for tensor in tensors:
            with self._mapping_operation(tensor.name, tensor.shards, "attach"):
                mapping = native.FabricMapping(
                    device=self.device, memory_kind=self.memory_kind
                )
                # A failed attach can leave resources for close() to retry.
                self._mappings[tensor.name] = mapping
                mapping.attach(
                    [
                        (
                            handles[part.allocation_id],
                            part.allocation_bytes,
                            part.allocation_granularity,
                        )
                        for part in tensor.shards
                    ]
                )
        for tensor in tensors:
            with self._mapping_operation(tensor.name, tensor.shards, "tensor_view"):
                # The validated geometry pads only the last segment's logical
                # end. Expose the physical row extent used by lookup/prefetch.
                padded_bytes = sum(part.allocation_bytes for part in tensor.shards)
                shape = (padded_bytes // tensor.row_bytes, tensor.row_bytes)
                self._views[tensor.name] = self._mappings[tensor.name].tensor_view(
                    shape, 0
                )

    @contextmanager
    def _mapping_operation(self, name, parts, operation):
        try:
            yield
        except BaseException as error:
            try:
                segment = getattr(error, "segment", 0 if len(parts) == 1 else -1)
                owner = (
                    hashlib.sha256(parts[segment].host_id.encode()).hexdigest()[:16]
                    if 0 <= segment < len(parts)
                    else "none"
                )
                _LOGGER.error(
                    "engram event=mapping_failed phase=attach generation=%s layer=%s "
                    "tensor_kind=%s segment=%s owner_sha256=%s device=%s operation=%s "
                    "error_class=%s cuda_error=%s cuda_error_name=%s",
                    self.validated.manifest.generation,
                    int(name.split(".")[1]),
                    name.rsplit(".", 1)[1],
                    segment,
                    owner,
                    self.device,
                    getattr(error, "operation", operation),
                    type(error).__name__,
                    getattr(error, "cuda_error", "none"),
                    getattr(error, "cuda_error_name", "none"),
                )
            except Exception:
                pass
            raise

    def _log_state(self, event, phase, started, error=None, *, cleanup_pending=False):
        # Called only at initialization transitions. Do not walk native mappings
        # or perform CUDA work to collect diagnostics; failures are best effort.
        try:
            _LOGGER.log(
                logging.INFO if error is None else logging.ERROR,
                "engram event=%s phase=%s state=%s generation=%s device=%s shard_count=%s memory=%s "
                "physical_bytes=%s allocation_count=%s mapping_count=%s graph_pins=%s "
                "duration_ms=%.3f error_class=%s cleanup_pending=%s",
                event,
                phase,
                self._state,
                self.validated.manifest.generation,
                self.device,
                self._shard_count,
                self.memory_kind,
                self.validated.physical_bytes,
                self._allocation_count,
                len(self._mappings),
                self._graph_pins,
                (time.monotonic() - started) * 1000,
                type(error).__name__[:64] if error is not None else "none",
                cleanup_pending,
            )
        except Exception:
            pass

    def require_model_layout(self, layer_ids, num_embeddings, head_dim):
        """Check the complete model against the contract validated at attach."""
        expected = self.validated.layout
        layers, rows = tuple(layer_ids), tuple(num_embeddings)
        if (
            len(layers) != len(rows)
            or tuple(sorted(zip(layers, rows))) != expected.layers
            or head_dim != expected.head_dim
            or expected.backend not in ("fabric_egm", "fabric_gpu")
        ):
            raise ValueError(
                "external generation does not match the complete model Engram layout"
            )

    def verify(self, verify_views):
        """Verify GPU reads before READY; callback must drain and return True.

        The caller compares device-side samples against checkpoint rows before
        publishing views to the model. Metadata alone cannot verify mapped bytes.
        """
        from types import MappingProxyType

        with self._lock:
            if self._state != "MAPPED":
                raise RuntimeError("verification requires freshly MAPPED storage")
            self._state = "VERIFYING"
            started = time.monotonic()
            try:
                if (
                    verify_views(MappingProxyType(self._views), self.validated)
                    is not True
                ):
                    raise ValueError("GPU content verifier must explicitly return True")
                if self._state != "VERIFYING":
                    raise RuntimeError("verification state was changed unexpectedly")
            except BaseException as error:
                self._state = "FAILED"
                self._log_state("storage_verification_failed", "verify", started, error)
                raise
            self._state = "READY"
            self._log_state("storage_verified", "verify", started)

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
                raise KeyError(
                    "requested layer is absent from this generation"
                ) from None

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
            return dict(
                state=self._state,
                generation=self.validated.manifest.generation,
                device=self.device,
                shard_count=self._shard_count,
                physical_bytes=self.validated.physical_bytes,
                allocation_count=self._allocation_count,
                mapping_count=len(self._mappings),
                graph_pins=self._graph_pins,
                allocations={
                    aid: mapping.info() for aid, mapping in self._mappings.items()
                },
            )

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
                raise StorageBusyError(
                    "external tensor views must be released before close"
                )
            try:
                for mapping in reversed(list(self._mappings.values())):
                    mapping.close()
            except Exception:
                self._state = "FAILED"
                raise
            self._mappings.clear()
            self._state = "CLOSED"
