"""Direct CUDA VMM storage for the Qwen4 RL weight set.

The manager deliberately owns only allocations created for RL weights. It does
not export handles to a resident store: sleep releases the local physical
backing while retaining each virtual address, and the next refit fills those
same addresses before the native engine resumes.
"""

from __future__ import annotations

import functools
import gc
import hashlib
import json
import logging
import sys
import threading
import weakref
from dataclasses import dataclass
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_DEDICATED_MIN_BYTES = 1 << 20
_PACK_CHUNK_BYTES = 16 << 20
_STORAGE_ALIGNMENT = 512
_DEFERRED_CLOSES: list["VmmAllocation"] = []
_MANAGERS: "weakref.WeakSet[RlVmmWeightManager]" = weakref.WeakSet()


def _driver():
    try:
        from cuda.bindings import driver
    except ImportError as error:
        raise RuntimeError(
            "RL VMM sleep requires the CUDA Python driver bindings in the inference runtime"
        ) from error
    return driver


def _check(result, operation: str):
    error, *outputs = result
    if error != _driver().CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{operation} failed: {error!r}")
    return outputs[0] if outputs else None


def _allocation_prop(device: int):
    cu = _driver()
    prop = cu.CUmemAllocationProp()
    prop.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = device
    # Co-located RL sleep has no out-of-process physical-memory owner.
    prop.requestedHandleTypes = cu.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_NONE
    return prop


@functools.lru_cache(maxsize=None)
def allocation_granularity(device: int) -> int:
    cu = _driver()
    _check(cu.cuInit(0), "cuInit")
    size = _check(
        cu.cuMemGetAllocationGranularity(
            _allocation_prop(device),
            cu.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
        ),
        "cuMemGetAllocationGranularity",
    )
    if size <= 0:
        raise RuntimeError(f"CUDA returned invalid VMM granularity {size}")
    return int(size)


def round_up_to_granularity(size_bytes: int, device: int) -> int:
    if size_bytes <= 0:
        raise ValueError(f"VMM allocation size must be positive, got {size_bytes}")
    granularity = allocation_granularity(device)
    return (size_bytes + granularity - 1) // granularity * granularity


def is_available(device: int) -> bool:
    if torch.version.hip is not None or not torch.cuda.is_available():
        return False
    try:
        cu = _driver()
        _check(cu.cuInit(0), "cuInit")
        error, cu_device = cu.cuDeviceGet(device)
        if error != cu.CUresult.CUDA_SUCCESS:
            return False
        for attribute in (
            cu.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
        ):
            error, supported = cu.cuDeviceGetAttribute(attribute, cu_device)
            if error != cu.CUresult.CUDA_SUCCESS or not supported:
                return False
        allocation_granularity(device)
        return True
    except (ImportError, RuntimeError):
        return False


class VmmAllocation:
    """One local VMM allocation with a single stable, read/write virtual address."""

    def __init__(self, size_bytes: int, device: int, tag: str):
        self.device = int(device)
        self.tag = tag
        self.size_bytes = round_up_to_granularity(int(size_bytes), self.device)
        self.handle = None
        self.ptr: Optional[int] = None
        self._mapped = False
        self._closed = False
        self._lock = threading.RLock()
        cu = _driver()
        _check(cu.cuInit(0), "cuInit")
        self.handle = _check(
            cu.cuMemCreate(self.size_bytes, _allocation_prop(self.device), 0),
            f"cuMemCreate({self.size_bytes})",
        )
        try:
            self.ptr = int(
                _check(
                    cu.cuMemAddressReserve(self.size_bytes, 0, 0, 0),
                    "cuMemAddressReserve",
                )
            )
            self._map_existing_handle()
        except BaseException:
            self._cleanup_new_mapping()
            raise

    @property
    def mapped(self) -> bool:
        return self._mapped

    def _set_read_write_access(self, ptr: int) -> None:
        cu = _driver()
        desc = cu.CUmemAccessDesc()
        desc.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        desc.location.id = self.device
        desc.flags = cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        _check(cu.cuMemSetAccess(ptr, self.size_bytes, [desc], 1), "cuMemSetAccess")

    def _map_existing_handle(self) -> None:
        _check(
            _driver().cuMemMap(self.ptr, self.size_bytes, 0, self.handle, 0),
            "cuMemMap",
        )
        try:
            self._set_read_write_access(self.ptr)
        except BaseException:
            try:
                _check(
                    _driver().cuMemUnmap(self.ptr, self.size_bytes),
                    "cuMemUnmap mapping rollback",
                )
            except BaseException:
                logger.exception("failed to undo a VMM mapping after cuMemSetAccess failed")
            raise
        self._mapped = True

    def _cleanup_new_mapping(self) -> None:
        cu = _driver()
        if self._mapped and self.ptr is not None:
            try:
                _check(cu.cuMemUnmap(self.ptr, self.size_bytes), "cuMemUnmap rollback")
            except BaseException:
                logger.exception("failed to undo initial VMM mapping")
            self._mapped = False
        if self.ptr is not None:
            try:
                _check(cu.cuMemAddressFree(self.ptr, self.size_bytes), "cuMemAddressFree rollback")
            except BaseException:
                logger.exception("failed to free initial VMM address reservation")
            self.ptr = None
        if self.handle is not None:
            try:
                _check(cu.cuMemRelease(self.handle), "cuMemRelease rollback")
            except BaseException:
                logger.exception("failed to release initial VMM handle")
            self.handle = None

    def as_torch_storage(
        self, offset_bytes: int, nbytes: int
    ) -> tuple[torch.UntypedStorage, "_CudaArrayBridge"]:
        if not self._mapped or self.ptr is None:
            raise RuntimeError("cannot create torch storage from an unmapped VMM allocation")
        if offset_bytes < 0 or nbytes < 0 or offset_bytes + nbytes > self.size_bytes:
            raise ValueError(
                f"VMM storage range [{offset_bytes}, {offset_bytes + nbytes}) "
                f"exceeds allocation size {self.size_bytes}"
            )
        bridge = _CudaArrayBridge(self, self.ptr + offset_bytes, nbytes)
        with torch.cuda.device(self.device):
            data = torch.as_tensor(bridge, device=torch.device("cuda", self.device))
        return data.untyped_storage(), bridge

    def unmap_for_sleep(self) -> int:
        """Release physical backing but retain the original virtual address reservation."""
        with self._lock:
            if self.handle is None:
                return 0
            if self._mapped:
                _check(_driver().cuMemUnmap(self.ptr, self.size_bytes), "cuMemUnmap sleep")
                self._mapped = False
            _check(_driver().cuMemRelease(self.handle), "cuMemRelease sleep")
            self.handle = None
            return self.size_bytes

    def remap(self) -> bool:
        """Map fresh physical backing at the same VA; repeated calls are no-ops."""
        with self._lock:
            if self._mapped:
                return False
            cu = _driver()
            created_handle = self.handle is None
            if created_handle:
                self.handle = _check(
                    cu.cuMemCreate(self.size_bytes, _allocation_prop(self.device), 0),
                    f"cuMemCreate({self.size_bytes}) on wake",
                )
            try:
                self._map_existing_handle()
            except BaseException:
                if created_handle and self.handle is not None:
                    try:
                        _check(cu.cuMemRelease(self.handle), "cuMemRelease wake rollback")
                    finally:
                        self.handle = None
                raise
            return True

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            cu = _driver()
            if self._mapped:
                _check(cu.cuMemUnmap(self.ptr, self.size_bytes), "cuMemUnmap close")
                self._mapped = False
            if self.handle is not None:
                _check(cu.cuMemRelease(self.handle), "cuMemRelease close")
                self.handle = None
            if self.ptr is not None:
                _check(cu.cuMemAddressFree(self.ptr, self.size_bytes), "cuMemAddressFree close")
                self.ptr = None
            self._closed = True

    def __del__(self):
        if sys.is_finalizing() or self._closed:
            return
        try:
            if torch.cuda.is_current_stream_capturing():
                _DEFERRED_CLOSES.append(self)
                return
            torch.cuda.synchronize(self.device)
            self.close()
            while _DEFERRED_CLOSES:
                _DEFERRED_CLOSES.pop().close()
        except BaseException:
            logger.exception("failed to close RL VMM allocation during teardown")


class _CudaArrayBridge:
    """Retain a VMM allocation while torch storage views its CUDA array interface."""

    def __init__(self, allocation: VmmAllocation, ptr: int, nbytes: int):
        self.allocation = allocation
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 3,
        }


@dataclass(frozen=True)
class _StorageRange:
    ptr: int
    nbytes: int
    device: torch.device

    @classmethod
    def of(cls, storage: torch.UntypedStorage) -> "_StorageRange":
        return cls(int(storage.data_ptr()), int(storage.nbytes()), storage.device)


@dataclass(frozen=True)
class _StorageBinding:
    allocation: VmmAllocation
    offset_bytes: int
    nbytes: int
    tag: str


def _bytes_of(storage: torch.UntypedStorage) -> torch.Tensor:
    return torch.empty(0, dtype=torch.uint8, device=storage.device).set_(
        storage, 0, (storage.nbytes(),), (1,)
    )


def _view(
    storage: torch.UntypedStorage,
    dtype: torch.dtype,
    storage_offset: int,
    shape,
    strides,
) -> torch.Tensor:
    return torch.empty(0, dtype=dtype, device=storage.device).set_(
        storage, storage_offset, tuple(shape), tuple(strides)
    )


class _VmmPacker:
    def __init__(self, manager: "RlVmmWeightManager", tag: str):
        self.manager = manager
        self.tag = tag
        self.chunk: Optional[VmmAllocation] = None
        self.used = 0

    def place(self, nbytes: int) -> tuple[VmmAllocation, int]:
        if nbytes <= 0:
            raise ValueError(f"cannot pack empty VMM storage of {nbytes} bytes")
        if nbytes >= _DEDICATED_MIN_BYTES:
            allocation = self.manager._new_allocation(self.tag, nbytes)
            return allocation, 0
        offset = (self.used + _STORAGE_ALIGNMENT - 1) // _STORAGE_ALIGNMENT * _STORAGE_ALIGNMENT
        if self.chunk is None or offset + nbytes > self.chunk.size_bytes:
            self.chunk = self.manager._new_allocation(self.tag, _PACK_CHUNK_BYTES)
            offset = 0
        self.used = offset + nbytes
        return self.chunk, offset


class RlVmmWeightManager:
    """Own VMM-backed Qwen4 weights for one inference GPU process."""

    def __init__(self, device: str | torch.device):
        self.device = torch.device(device)
        if self.device.type != "cuda" or self.device.index is None:
            raise ValueError(f"RL VMM weights require an indexed CUDA device, got {device}")
        if not is_available(int(self.device.index)):
            raise RuntimeError(f"CUDA VMM is unavailable on {self.device}")
        self.device_index = int(self.device.index)
        self._allocations: list[VmmAllocation] = []
        self._packer = _VmmPacker(self, "weights")
        self._source_storage_map: dict[_StorageRange, tuple[torch.UntypedStorage, _StorageBinding]] = {}
        self._storage_bindings: dict[_StorageRange, _StorageBinding] = {}
        self._allocation_owner_refs: dict[VmmAllocation, list[weakref.ReferenceType]] = {}
        self._zero_on_remap: list[torch.Tensor] = []
        self._startup_tensor_addresses: Optional[dict[str, dict]] = None
        self._loader = None
        self._py_model_ref: Optional[weakref.ReferenceType] = None
        self._sealed = False
        _MANAGERS.add(self)

    def _new_allocation(self, tag: str, nbytes: int) -> VmmAllocation:
        if tag != "weights":
            raise ValueError(f"unsupported RL VMM allocation tag {tag!r}")
        allocation = VmmAllocation(nbytes, self.device_index, tag)
        self._allocations.append(allocation)
        self._allocation_owner_refs[allocation] = []
        return allocation

    def _storage_in_vmm(
        self, source: torch.UntypedStorage, tag: str
    ) -> tuple[torch.UntypedStorage, _StorageBinding]:
        source_range = _StorageRange.of(source)
        previous = self._source_storage_map.get(source_range)
        if previous is not None:
            storage, binding = previous
            if binding.tag != tag:
                raise RuntimeError(
                    f"one source storage is assigned to both {binding.tag!r} and {tag!r} VMM tags"
                )
            return storage, binding

        allocation, offset = self._packer.place(int(source.nbytes()))
        storage, bridge = allocation.as_torch_storage(offset, int(source.nbytes()))
        _bytes_of(storage).copy_(_bytes_of(source))
        binding = _StorageBinding(allocation, offset, int(source.nbytes()), tag)
        target_range = _StorageRange.of(storage)
        self._storage_bindings[target_range] = binding
        self._source_storage_map[source_range] = (storage, binding)
        self._allocation_owner_refs[allocation].append(weakref.ref(bridge))
        return storage, binding

    def pack_model_weights(self, weights, loader) -> dict:
        """Rehome loader outputs before any Python model consumer is constructed."""
        if self._sealed:
            raise RuntimeError("cannot pack model weights after VMM addresses were sealed")
        self._loader = loader
        static_table_keys, static_meta_keys = loader.get_rl_static_target_keys(weights)
        static_keys = static_table_keys | static_meta_keys
        qsa_target_keys = loader.get_rl_qsa_target_keys()
        entries = []
        tag_by_source_storage: dict[_StorageRange, str] = {}
        for layer_id, mapping in [(None, weights.global_weights)] + list(enumerate(weights.weights)):
            for name, tensor in mapping.items():
                key = (layer_id, name)
                if key in static_keys or tensor.device.type != "cuda":
                    continue
                if tensor.device.index != self.device_index:
                    raise ValueError(
                        f"RL VMM weight {key!r} is on {tensor.device}, expected {self.device}"
                    )
                # QSA is small and remains in ordinary device storage. This keeps
                # its last committed bytes resident even when a frozen QSA triplet
                # is omitted from a later refit.
                if key in qsa_target_keys:
                    continue
                tag = "weights"
                source_range = _StorageRange.of(tensor.untyped_storage())
                previous_tag = tag_by_source_storage.get(source_range)
                if previous_tag is not None and previous_tag != tag:
                    raise ValueError(
                        f"aliased model weights cross VMM residency classes: {previous_tag} and {tag}"
                    )
                tag_by_source_storage[source_range] = tag
                entries.append((layer_id, name, tensor, tag, source_range))

        for layer_id, name, tensor, tag, _ in entries:
            mapping = weights.global_weights if layer_id is None else weights.weights[layer_id]
            source_storage = tensor.untyped_storage()
            vmm_storage, _ = self._storage_in_vmm(source_storage, tag)
            view = _view(
                vmm_storage,
                tensor.dtype,
                tensor.storage_offset(),
                tensor.shape,
                tensor.stride(),
            )
            with torch.inference_mode(tensor.is_inference()):
                mapping[name] = view
            del view, tensor, source_storage

        # Source-side storage references are no longer needed; ModelWeights is now the owner.
        self._source_storage_map.clear()
        return self.inventory_summary()

    def bind_model(self, py_model) -> None:
        self._py_model_ref = weakref.ref(py_model)

    def pack_tensor(self, tensor: torch.Tensor, *, tag: str = "weights") -> torch.Tensor:
        """Pack one final model-local compute tensor before graph capture."""
        if tag != "weights":
            raise ValueError(f"unsupported RL VMM weight tag {tag!r}")
        if tensor.device.type != "cuda" or tensor.device.index != self.device_index:
            raise ValueError(f"cannot pack non-local CUDA tensor {tensor.device}")
        if self.allocation_of(tensor) is not None:
            return tensor
        storage, _ = self._storage_in_vmm(tensor.untyped_storage(), tag)
        view = _view(storage, tensor.dtype, tensor.storage_offset(), tensor.shape, tensor.stride())
        self._source_storage_map.clear()
        with torch.inference_mode(tensor.is_inference()):
            return view

    def register_zero_on_remap(self, tensor: torch.Tensor) -> None:
        binding = self.allocation_of(tensor)
        if binding is None or binding.tag != "weights":
            raise ValueError("remap zero ranges must be views of sleepable VMM weights")
        if tensor.numel() > 0:
            self._zero_on_remap.append(tensor)

    def allocation_of(self, tensor: torch.Tensor) -> Optional[_StorageBinding]:
        if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cuda":
            return None
        try:
            return self._storage_bindings.get(_StorageRange.of(tensor.untyped_storage()))
        except Exception:
            return None

    def owns_tensor(self, tensor: torch.Tensor) -> bool:
        return self.allocation_of(tensor) is not None

    def finish_after_initialize(self, weights, py_model) -> dict:
        """Reap cold-load temporaries and verify live owners before graph capture."""
        if self._sealed:
            return self.verify_live_owners(weights, py_model)
        torch.cuda.synchronize(self.device)
        gc.collect()
        self._reap_unowned_allocations()
        self._sealed = True
        self.bind_model(py_model)
        return self.verify_live_owners(weights, py_model)

    def verify_live_owners(self, weights, py_model) -> dict:
        """Validate all CUDA loader targets and Qwen4 merged HC outputs before graph capture."""
        if not self._sealed:
            raise RuntimeError("RL VMM addresses must be sealed before owner verification")
        if self._loader is None:
            raise RuntimeError("RL VMM manager was not initialized from the model loader")
        static_table_keys, static_meta_keys = self._loader.get_rl_static_target_keys(weights)
        static_keys = static_table_keys | static_meta_keys
        records = {}
        qsa_target_keys = self._loader.get_rl_qsa_target_keys()
        for layer_id, mapping in [(None, weights.global_weights)] + list(enumerate(weights.weights)):
            for name, tensor in mapping.items():
                binding = self.allocation_of(tensor)
                if (layer_id, name) in static_keys:
                    if binding is not None:
                        raise RuntimeError(
                            f"static PLE target {(layer_id, name)!r} must remain outside sleepable VMM storage"
                        )
                    records[f"{layer_id}:{name}"] = self._tensor_address_record(
                        tensor,
                        None,
                        tag="static_cpu" if tensor.device.type == "cpu" else "static_metadata",
                    )
                    continue
                if (layer_id, name) in qsa_target_keys:
                    if binding is not None:
                        raise RuntimeError(
                            f"QSA indexer target {(layer_id, name)!r} must remain in its original Torch storage"
                        )
                    records[f"{layer_id}:{name}"] = self._tensor_address_record(
                        tensor, None, tag="resident_qsa"
                    )
                    continue
                if tensor.device.type != "cuda":
                    raise RuntimeError(
                        f"mutable RL model weight {(layer_id, name)!r} is not on CUDA: {tensor.device}"
                    )
                if binding is None:
                    raise RuntimeError(f"live CUDA model weight {(layer_id, name)!r} is not VMM-backed")
                if not binding.allocation.mapped:
                    raise RuntimeError(f"live CUDA model weight {(layer_id, name)!r} is unmapped")
                records[f"{layer_id}:{name}"] = self._tensor_address_record(tensor, binding)

        for layer_idx, layer in enumerate(getattr(py_model, "layers", ())):
            for unit_name in ("attn_hyper_connection", "mlp_hyper_connection"):
                unit = getattr(layer, unit_name, None)
                merged = getattr(unit, "merged_down_inject", None)
                if merged is None:
                    continue
                binding = self.allocation_of(merged)
                if binding is None or binding.tag != "weights":
                    raise RuntimeError(f"live HC compute merge layer={layer_idx} {unit_name} is not VMM-backed")
                records[
                    f"layer.{layer_idx}.{unit_name}.merged_down_inject"
                ] = self._tensor_address_record(merged, binding)

        records.update(self._gdn_projection_records(py_model, require_mapped=True))

        orphaned = []
        for index, allocation in enumerate(self._allocations):
            if not any(ref() is not None for ref in self._allocation_owner_refs[allocation]):
                orphaned.append(f"weights[{index}]@{allocation.ptr:#x}")
        if orphaned:
            raise RuntimeError(f"RL VMM allocations have no live torch storage owners: {orphaned[:8]}")
        self._startup_tensor_addresses = records
        return self.inventory_summary(records)

    def _reap_unowned_allocations(self) -> None:
        """Release pack chunks whose last Torch storage owner died during model construction."""
        keep = []
        for allocation in self._allocations:
            if any(ref() is not None for ref in self._allocation_owner_refs[allocation]):
                keep.append(allocation)
                continue
            allocation.close()
            self._allocation_owner_refs.pop(allocation, None)
            for key, binding in list(self._storage_bindings.items()):
                if binding.allocation is allocation:
                    del self._storage_bindings[key]
            if self._packer.chunk is allocation:
                self._packer.chunk = None
                self._packer.used = 0
        self._allocations = keep

    def _gdn_projection_records(self, py_model, *, require_mapped: bool) -> dict[str, dict]:
        records = {}
        for layer_idx, layer in enumerate(getattr(py_model, "layers", ())):
            attention = getattr(layer, "self_attn", None)
            if attention is None or not hasattr(attention, "in_proj_fused"):
                continue
            projection = attention.in_proj_fused
            weight = getattr(projection, "weight", None)
            if projection is None or not isinstance(weight, torch.Tensor):
                raise RuntimeError(
                    f"RL VMM Qwen4 GDN layer={layer_idx} lacks its fused projection weight"
                )
            binding = self.allocation_of(weight)
            if binding is None or binding.tag != "weights":
                raise RuntimeError(
                    f"RL VMM Qwen4 GDN layer={layer_idx} fused projection is not VMM-backed"
                )
            if require_mapped and not binding.allocation.mapped:
                raise RuntimeError(
                    f"RL VMM Qwen4 GDN layer={layer_idx} fused projection is unmapped"
                )
            records[f"layer.{layer_idx}.self_attn.in_proj_fused.weight"] = (
                self._tensor_address_record(weight, binding)
            )
        return records

    @staticmethod
    def _tensor_address_record(
        tensor: torch.Tensor, binding: Optional[_StorageBinding], *, tag: Optional[str] = None
    ) -> dict:
        storage = tensor.untyped_storage()
        return {
            "data_ptr": int(tensor.data_ptr()),
            "storage_ptr": int(storage.data_ptr()),
            "storage_nbytes": int(storage.nbytes()),
            "storage_offset": int(tensor.storage_offset()),
            "shape": list(tensor.shape),
            "stride": list(tensor.stride()),
            "dtype": str(tensor.dtype),
            "tag": tag or (binding.tag if binding is not None else "ordinary"),
            "allocation_ptr": int(binding.allocation.ptr) if binding is not None else None,
            "allocation_bytes": binding.allocation.size_bytes if binding is not None else None,
        }

    def unmap_sleepable(self) -> int:
        released = 0
        with torch.cuda.device(self.device):
            for allocation in self._allocations:
                released += allocation.unmap_for_sleep()
        return released

    def remap_sleepable(self) -> None:
        with torch.cuda.device(self.device):
            for allocation in self._allocations:
                allocation.remap()
            for tensor in self._zero_on_remap:
                tensor.zero_()
            # The native control path publishes residency immediately after this
            # call, and the zeroed HC padding is read by captured GEMMs.
            torch.cuda.synchronize(self.device)

    def verify_remapped_weights(self, weights) -> dict:
        if self._startup_tensor_addresses is None:
            raise RuntimeError("RL VMM startup tensor address manifest was not recorded")
        if self._loader is None:
            raise RuntimeError("RL VMM manager has no model loader")
        py_model = self._py_model_ref() if self._py_model_ref is not None else None
        if py_model is None:
            raise RuntimeError("RL VMM Qwen4 Python model is no longer alive")
        current = {}
        static_table_keys, static_meta_keys = self._loader.get_rl_static_target_keys(weights)
        static_keys = static_table_keys | static_meta_keys
        qsa_target_keys = self._loader.get_rl_qsa_target_keys()
        for layer_id, mapping in [(None, weights.global_weights)] + list(enumerate(weights.weights)):
            for name, tensor in mapping.items():
                binding = self.allocation_of(tensor)
                if (layer_id, name) in static_keys:
                    if binding is not None:
                        raise RuntimeError(f"static target {(layer_id, name)!r} entered the sleepable VMM tag")
                    current[f"{layer_id}:{name}"] = self._tensor_address_record(
                        tensor,
                        None,
                        tag="static_cpu" if tensor.device.type == "cpu" else "static_metadata",
                    )
                elif (layer_id, name) in qsa_target_keys:
                    if binding is not None:
                        raise RuntimeError(f"QSA target {(layer_id, name)!r} entered the sleepable VMM tag")
                    current[f"{layer_id}:{name}"] = self._tensor_address_record(
                        tensor, None, tag="resident_qsa"
                    )
                else:
                    if tensor.device.type != "cuda":
                        raise RuntimeError(
                            f"mutable RL model weight {(layer_id, name)!r} moved off CUDA: {tensor.device}"
                        )
                    if binding is None or not binding.allocation.mapped:
                        raise RuntimeError(f"RL VMM target {(layer_id, name)!r} did not remap")
                    current[f"{layer_id}:{name}"] = self._tensor_address_record(tensor, binding)
        for layer_idx, layer in enumerate(getattr(py_model, "layers", ())):
            for unit_name in ("attn_hyper_connection", "mlp_hyper_connection"):
                unit = getattr(layer, unit_name, None)
                merged = getattr(unit, "merged_down_inject", None)
                if merged is None:
                    continue
                binding = self.allocation_of(merged)
                if binding is None or not binding.allocation.mapped:
                    raise RuntimeError(f"RL VMM HC merge layer={layer_idx} {unit_name} did not remap")
                current[f"layer.{layer_idx}.{unit_name}.merged_down_inject"] = self._tensor_address_record(
                    merged, binding
                )
        current.update(self._gdn_projection_records(py_model, require_mapped=True))
        expected = {
            key: value
            for key, value in self._startup_tensor_addresses.items()
        }
        if current != expected:
            changed = [key for key in set(current) | set(expected) if current.get(key) != expected.get(key)]
            raise RuntimeError(f"RL VMM live target addresses/layout changed during remap: {changed[:8]}")
        return self.inventory_summary(current)

    def inventory_summary(self, tensors: Optional[dict] = None) -> dict:
        allocations = []
        tag_bytes = {"weights": 0}
        mapped_tag_bytes = {"weights": 0}
        for allocation in self._allocations:
            tag_bytes[allocation.tag] += allocation.size_bytes
            if allocation.mapped:
                mapped_tag_bytes[allocation.tag] += allocation.size_bytes
            allocations.append(
                {
                    "tag": allocation.tag,
                    "va": allocation.ptr,
                    "size_bytes": allocation.size_bytes,
                    "mapped": allocation.mapped,
                }
            )
        allocations.sort(key=lambda item: (item["tag"], item["va"]))
        payload = {
            "device": str(self.device),
            "tag_bytes": tag_bytes,
            "mapped_tag_bytes": mapped_tag_bytes,
            "allocations": allocations,
            "tensors": tensors or {},
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        payload["inventory_sha256"] = hashlib.sha256(encoded).hexdigest()
        return payload

    def close(self) -> None:
        for allocation in self._allocations:
            allocation.close()


def manager_for_tensor(tensor: torch.Tensor) -> Optional[RlVmmWeightManager]:
    for manager in list(_MANAGERS):
        if manager.owns_tensor(tensor):
            return manager
    return None


def pack_if_rl_vmm(tensor: torch.Tensor, reference: torch.Tensor) -> tuple[torch.Tensor, Optional[RlVmmWeightManager]]:
    manager = manager_for_tensor(reference)
    return (tensor, None) if manager is None else (manager.pack_tensor(tensor), manager)
