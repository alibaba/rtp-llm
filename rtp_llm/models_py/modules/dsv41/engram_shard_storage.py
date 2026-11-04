# SPDX-License-Identifier: Apache-2.0
"""Map N Engram shards into two contiguous VA spans.

The lookup kernel is unchanged: it still indexes ``weight + row * dim`` and
``scales + row * (dim // block)``.  This module only arranges the address space
so that global row ``r`` lands in the shard that owns it.

Driver calls are behind :class:`DriverOps` so the layout and rollback logic can
be tested on a CPU-only machine with a fake recorder.
"""
from __future__ import annotations

from typing import Protocol, Sequence

from .engram_shard import MappingRequest, ShardPlan, mapping_requests


class DriverOps(Protocol):
    def reserve(self, size: int, alignment: int) -> int: ...

    def import_handle(self, raw: bytes) -> int: ...

    def map(self, va: int, size: int, offset: int, handle: int) -> None: ...

    def set_access(self, va: int, size: int) -> None: ...

    def release(self, handle: int) -> None: ...

    def synchronize(self) -> None: ...

    def unmap(self, va: int, size: int) -> None: ...

    def free(self, va: int, size: int) -> None: ...

    def view(self, va: int, shape: tuple[int, ...], owner): ...


class CudaDriverOps:
    """Thin wrapper over ``cuda.bindings.driver``; every call is checked."""

    def __init__(self, device: int):
        from cuda.bindings import driver

        self._driver = driver
        self.device = device

    def _check(self, result, what: str):
        error = result[0] if isinstance(result, tuple) else result
        if int(error) != 0:
            raise RuntimeError(f"engram shard {what} failed: {int(error)}")
        return result[1] if isinstance(result, tuple) and len(result) > 1 else None

    def _call(self, function, *args):
        # Select/initialize the consumer's primary context on the calling thread.
        import torch
        with torch.cuda.device(self.device):
            torch.cuda.init()
            return function(*args)

    def synchronize(self):
        import torch
        torch.cuda.synchronize(self.device)

    def release(self, handle):
        self._check(self._call(self._driver.cuMemRelease, handle), "release")

    def reserve(self, size: int, alignment: int) -> int:
        pointer = self._check(
            self._call(self._driver.cuMemAddressReserve, size, alignment, 0, 0), "address_reserve"
        )
        return int(pointer)

    def import_handle(self, raw: bytes) -> int:
        handle_type = self._driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
        handle = self._check(
            self._call(self._driver.cuMemImportFromShareableHandle, raw, handle_type),
            "import_handle",
        )
        return int(handle)

    def map(self, va: int, size: int, offset: int, handle: int) -> None:
        self._check(self._call(self._driver.cuMemMap, va, size, offset, handle, 0), "map")

    def set_access(self, va: int, size: int) -> None:
        # Mapping alone leaves the range inaccessible to the device; grant the
        # consumer GPU read access exactly like the native Consumer does.
        access = self._driver.CUmemAccessDesc()
        access.location.type = self._driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        access.location.id = self.device
        access.flags = self._driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READ
        self._check(self._call(self._driver.cuMemSetAccess, va, size, [access], 1), "set_access")

    def unmap(self, va: int, size: int) -> None:
        self._check(self._call(self._driver.cuMemUnmap, va, size), "unmap")

    def free(self, va: int, size: int) -> None:
        self._check(self._call(self._driver.cuMemAddressFree, va, size), "address_free")

    def view(self, va: int, shape: tuple[int, ...], owner):
        import torch

        # Torch retains the CUDA array interface exporter as its storage owner;
        # derived tensor aliases therefore retain this lease too. No weakref to
        # the top-level Tensor: that would miss slices surviving its deletion.
        return torch.as_tensor(ArrayViewLease(va, shape, owner),
                               device=f"cuda:{self.device}")


class ArrayViewLease:
    """Storage-level ownership for the CUDA array interface exporter."""

    def __init__(self, pointer, shape, owner):
        owner._require_mapped()
        self.owner = owner
        owner._active_views += 1
        self.__cuda_array_interface__ = {
            "shape": shape, "strides": None, "typestr": "|u1",
            # Torch requires a writable exporter. CUDA VMM still enforces READ;
            # callers must never use in-place tensor operations on these views.
            "data": (pointer, False), "version": 3,
        }

    def __del__(self):
        owner = getattr(self, "owner", None)
        if owner is not None:
            owner._active_views -= 1


class ShardedLayerMapping:
    """One tensor (layer) mapped from ``shard_count`` fabric handles."""

    def __init__(self, plan: ShardPlan, weight_handles: Sequence[bytes],
                 scale_handles: Sequence[bytes], device: int, ops=None):
        for name, handles in (("weight", weight_handles), ("scale", scale_handles)):
            if len(handles) != plan.shard_count:
                raise ValueError(f"{name} handle count must equal shard_count")
            if any(not isinstance(handle, bytes) or not handle for handle in handles):
                raise ValueError(f"{name} fabric handles must be non-empty bytes")
        self.plan = plan
        self.weight_handles = tuple(weight_handles)
        self.scale_handles = tuple(scale_handles)
        self.device = device
        self.ops = ops if ops is not None else CudaDriverOps(device)
        self._weight_ptr = 0
        self._scale_ptr = 0
        self._mapped: list[MappingRequest] = []
        self._imported: list[int] = []
        self._active_views = 0
        self._state = "NEW"

    @property
    def state(self) -> str:
        return self._state

    def attach(self) -> "ShardedLayerMapping":
        if self._state != "NEW":
            raise RuntimeError("shard mapping can only be attached once")
        try:
            self._weight_ptr = self.ops.reserve(self.plan.weight_span, self.plan.granularity)
            self._scale_ptr = self.ops.reserve(self.plan.scale_span, self.plan.granularity)
            for request in mapping_requests(self.plan, self._weight_ptr, self._scale_ptr):
                source = self.weight_handles if request.kind == "weight" else self.scale_handles
                handle = self.ops.import_handle(source[request.shard])
                self._imported.append(handle)
                self.ops.map(request.va, request.size, request.handle_offset, handle)
                self._mapped.append(request)
                self.ops.set_access(request.va, request.size)
            self._state = "MAPPED"
            return self
        except BaseException as original:
            try:
                self.close()
            except BaseException:
                from .engram_storage import CleanupPendingError
                raise CleanupPendingError("shard attach cleanup incomplete", self) from original
            raise

    def _require_mapped(self):
        if self._state != "MAPPED":
            raise RuntimeError("shard mapping is not MAPPED")

    def weight_view(self):
        self._require_mapped()
        return self.ops.view(self._weight_ptr, (self.plan.padded_rows, self.plan.dim), self)

    def scale_view(self):
        self._require_mapped()
        return self.ops.view(
            self._scale_ptr, (self.plan.padded_rows, self.plan.dim // self.plan.block), self
        )

    def info(self):
        return dict(state=self._state, active_views=self._active_views,
                    imported_handles=len(self._imported), mapped_ranges=len(self._mapped))

    def close(self) -> None:
        """Drain before unmap; retain exact remaining resources on any failure.

        Caller must serialize view creation/enqueue vs close (FabricEngramStorage
        does so). Unmanaged CUDA Graphs require a separate process-lifetime pin.
        """
        if self._state == "CLOSED":
            return
        if self._active_views:
            from .engram_storage import StorageBusyError
            raise StorageBusyError("external tensor storage/view leases remain")
        self._state = "CLOSING"
        if self._mapped:
            self.ops.synchronize()
        while self._mapped:
            request = self._mapped[-1]
            self.ops.unmap(request.va, request.size)
            self._mapped.pop()
        while self._imported:
            self.ops.release(self._imported[-1])
            self._imported.pop()
        if self._weight_ptr:
            self.ops.free(self._weight_ptr, self.plan.weight_span)
            self._weight_ptr = 0
        if self._scale_ptr:
            self.ops.free(self._scale_ptr, self.plan.scale_span)
            self._scale_ptr = 0
        self._state = "CLOSED"
