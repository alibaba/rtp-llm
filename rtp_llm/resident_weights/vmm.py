"""CUDA VMM allocations exported as POSIX file descriptors and wrapped as torch storages."""

import functools
import sys
from typing import List, Optional, Tuple

import torch


def _driver():
    # cuda.bindings is absent from ROCm and CPU builds, where this module must still import.
    from cuda.bindings import driver

    return driver


def is_available(device: int) -> bool:
    """Whether this runtime/device supports the VMM allocation and FD export contract."""
    if torch.version.hip is not None or not torch.cuda.is_available():
        return False
    try:
        cu = _driver()
    except ImportError:
        return False
    for attr in (
        cu.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED,
        cu.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED,
    ):
        error, supported = cu.cuDeviceGetAttribute(attr, device)
        if error != cu.CUresult.CUDA_SUCCESS or not supported:
            return False
    return True


def _check(result, what: str):
    """Return the output of a cuda.bindings driver call, raising on a CUresult error."""
    error, *outputs = result
    if error != _driver().CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{what} failed: {error!r}")
    return outputs[0] if outputs else None


def _posix_fd():
    return _driver().CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR


def _prop(device: int):
    cu = _driver()
    prop = cu.CUmemAllocationProp()
    prop.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.requestedHandleTypes = _posix_fd()
    prop.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = device
    return prop


@functools.lru_cache(maxsize=None)
def allocation_granularity(device: int) -> int:
    """The hardware's recommended VMM allocation granularity (2 MiB on B300/H20/L20).

    Constant per device for the life of the process, so each device asks the driver once:
    a cold start rounds up several hundred allocations and the query is not free.
    """
    cu = _driver()
    return _check(
        cu.cuMemGetAllocationGranularity(
            _prop(device),
            cu.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
        ),
        "cuMemGetAllocationGranularity",
    )


def round_up_to_granularity(nbytes: int, device: int) -> int:
    """``nbytes`` rounded up to a whole number of VMM allocation granules."""
    step = allocation_granularity(device)
    return (nbytes + step - 1) // step * step


# Releases requested while a CUDA graph is being captured; synchronizing then is illegal.
_deferred_releases: List["VmmAllocation"] = []


class VmmAllocation:
    """One cuMemCreate allocation mapped read-write or read-only at a private address range.

    The physical memory lives as long as any process holds the allocation handle, an
    exported FD or a mapping, so the store keeps it alive across engine restarts by
    holding only the FD.
    """

    def __init__(self, handle, size_bytes: int, device: int, writable: bool):
        self.handle = handle
        self.size_bytes = size_bytes
        self.device = device
        self._writable_alias_ptr: Optional[int] = None
        # Unset until mapped; __del__ then has nothing to release.
        self.ptr: Optional[int] = None
        try:
            self.ptr = self._map_new_range(writable)
        except BaseException:
            # The caller may retry (``_create`` does after empty_cache); do not keep the memory.
            _driver().cuMemRelease(handle)
            raise

    @classmethod
    def create(cls, size_bytes: int, device: int) -> "VmmAllocation":
        handle = _check(
            _driver().cuMemCreate(size_bytes, _prop(device), 0),
            f"cuMemCreate({size_bytes})",
        )
        return cls(handle, size_bytes, device, writable=True)

    @classmethod
    def from_fd(
        cls, fd: int, size_bytes: int, device: int, writable: bool
    ) -> "VmmAllocation":
        handle = _check(
            _driver().cuMemImportFromShareableHandle(fd, _posix_fd()),
            "cuMemImportFromShareableHandle",
        )
        return cls(handle, size_bytes, device, writable)

    def _map_new_range(self, writable: bool) -> int:
        cu = _driver()
        ptr = _check(
            cu.cuMemAddressReserve(self.size_bytes, 0, 0, 0), "cuMemAddressReserve"
        )
        try:
            _check(cu.cuMemMap(ptr, self.size_bytes, 0, self.handle, 0), "cuMemMap")
            try:
                self._set_access(ptr, writable)
            except BaseException:
                cu.cuMemUnmap(ptr, self.size_bytes)
                raise
        except BaseException:
            cu.cuMemAddressFree(ptr, self.size_bytes)
            raise
        return int(ptr)

    def _set_access(self, ptr: int, writable: bool) -> None:
        cu = _driver()
        flags = cu.CUmemAccess_flags
        desc = cu.CUmemAccessDesc()
        desc.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        desc.location.id = self.device
        desc.flags = (
            flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
            if writable
            else flags.CU_MEM_ACCESS_FLAGS_PROT_READ
        )
        _check(cu.cuMemSetAccess(ptr, self.size_bytes, [desc], 1), "cuMemSetAccess")

    def set_writable(self, writable: bool) -> None:
        """Re-protect the primary mapping; relocation writes, then publishes read-only.

        The caller must have no accesses to this mapping in flight: re-protecting pages a
        running kernel is reading faults the context with an illegal access.
        """
        self._set_access(self.ptr, writable)

    def writable_alias_ptr(self) -> int:
        """Address of a second, read-write mapping of the same physical memory (mapped once)."""
        if self._writable_alias_ptr is None:
            self._writable_alias_ptr = self._map_new_range(writable=True)
        return self._writable_alias_ptr

    def export_fd(self) -> int:
        return _check(
            _driver().cuMemExportToShareableHandle(self.handle, _posix_fd(), 0),
            "cuMemExportToShareableHandle",
        )

    def as_torch_storage(
        self, offset_bytes: int, nbytes: int, base_ptr: Optional[int] = None
    ) -> Tuple[torch.UntypedStorage, "_CudaArrayBridge"]:
        """A torch storage over [offset_bytes, offset_bytes + nbytes) of the mapping at
        ``base_ptr`` (default: the primary one). The returned bridge lives exactly as long as
        the storage and keeps this allocation mapped."""
        bridge = _CudaArrayBridge(
            self, (self.ptr if base_ptr is None else base_ptr) + offset_bytes, nbytes
        )
        data = torch.as_tensor(bridge, device=torch.device("cuda", self.device))
        return data.untyped_storage(), bridge

    def _release(self) -> None:
        cu = _driver()
        alias = self._writable_alias_ptr
        for ptr in [self.ptr] + ([] if alias is None else [alias]):
            _check(cu.cuMemUnmap(ptr, self.size_bytes), "cuMemUnmap")
            _check(cu.cuMemAddressFree(ptr, self.size_bytes), "cuMemAddressFree")
        _check(cu.cuMemRelease(self.handle), "cuMemRelease")

    def __del__(self):
        if sys.is_finalizing() or self.ptr is None:
            return  # teardown frees every mapping and handle; a failed __init__ mapped nothing
        if torch.cuda.is_current_stream_capturing():
            _deferred_releases.append(self)
            return
        # Kernels already queued may still read this memory; unmapping does not wait for them.
        torch.cuda.synchronize(self.device)
        self._release()
        while _deferred_releases:
            _deferred_releases.pop()._release()


class _CudaArrayBridge:
    """Keeps a VMM allocation mapped for exactly as long as one torch UntypedStorage.

    torch holds this object for the lifetime of the storage it exports through
    ``__cuda_array_interface__``; the strong reference to the allocation is what guarantees
    the physical memory and its address range outlive every tensor viewing them.
    """

    def __init__(self, allocation: VmmAllocation, ptr: int, nbytes: int):
        self.allocation = allocation
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 3,
        }
