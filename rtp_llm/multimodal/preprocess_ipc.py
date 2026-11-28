"""Transport image sources and CPU worker results between preprocess processes."""

import os
from dataclasses import dataclass
from multiprocessing.reduction import ForkingPickler
from threading import Lock

import torch
from torch.multiprocessing import reductions

_pack_cpu_tensors = None
_MAX_INLINE_TABLE_BYTES = 128 * 1024


def shared_preprocess_ipc_enabled():
    return os.environ.get("MM_VIT_SHARED_PREPROCESS_IPC") == "1"


@dataclass(frozen=True)
class _SourceSlice:
    offset: int
    size: int


class SharedPreprocessSources:
    """Keep large sources off the Pool's single input pipe with bounded storage."""

    _SLOT_BYTES = 8 * 1024**2
    _MIN_BYTES = 512 * 1024
    _SLOTS = 16

    def __init__(self, context):
        self.buffer = context.RawArray("B", self._SLOTS * self._SLOT_BYTES)
        self._view = memoryview(self.buffer).cast("B")
        self._lock = Lock()
        self._owners = [None] * self._SLOTS
        self._free = list(range(len(self._owners)))

    def prepare(self, sources):
        if sources is None:
            return None, None
        owner = None
        transport = None
        slots = []
        try:
            for index, body in enumerate(sources):
                if (
                    not isinstance(body, bytes)
                    or not self._MIN_BYTES <= len(body) <= self._SLOT_BYTES
                ):
                    continue
                if transport is None:
                    owner = object()
                    transport = list(sources)
                with self._lock:
                    if not self._free:
                        break
                    slot = self._free[-1]
                    slots.append(slot)
                    self._owners[slot] = owner
                    self._free.pop()
                offset = slot * self._SLOT_BYTES
                self._view[offset : offset + len(body)] = body
                transport[index] = _SourceSlice(offset, len(body))
            return (tuple(transport), (owner, slots)) if slots else (sources, None)
        except BaseException:
            self.release((owner, slots))
            raise

    def release(self, lease):
        if lease is None:
            return
        owner, slots = lease
        with self._lock:
            for slot in slots:
                if self._owners[slot] is owner:
                    self._owners[slot] = None
                    self._free.append(slot)

    def reset(self):
        """Return abandoned slots after all workers using this buffer have exited."""
        with self._lock:
            self._owners = [None] * len(self._owners)
            self._free = list(range(len(self._owners)))


def _read_source_slice(source, view):
    if not isinstance(source, _SourceSlice):
        return source
    end = source.offset + source.size
    if view is None or not 0 <= source.offset < end <= len(view):
        raise RuntimeError("Invalid shared preprocessing source")
    # The worker and asynchronous inspection retain independent owned bytes.
    return bytes(view[source.offset:end])


def read_preprocess_sources(sources, view):
    if sources is None or view is None:
        return sources
    return tuple(_read_source_slice(source, view) for source in sources)


def _inline_tensor(tensor):
    data = tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()
    return tensor.dtype, tuple(tensor.shape), data


def _restore_inline_tensor(data):
    dtype, shape, contents = data
    if not contents:
        return torch.empty(shape, dtype=dtype, device="cpu")
    return torch.frombuffer(bytearray(contents), dtype=dtype).reshape(shape)


def _rebuild_resize_tables(source_hw, target_hw, patch_size, horizontal, vertical, lut):
    from rtp_llm.models_py.modules.dsv41.vision_preprocess_tables import (
        ResizeTables,
        _normalization_lut,
    )

    return ResizeTables(
        source_hw,
        target_hw,
        patch_size,
        tuple(map(_restore_inline_tensor, horizontal)) if horizontal else None,
        tuple(map(_restore_inline_tensor, vertical)) if vertical else None,
        _normalization_lut() if lut is None else _restore_inline_tensor(lut),
    )


def _reduce_resize_tables(tables):
    from rtp_llm.models_py.modules.dsv41.vision_preprocess_tables import _normalization_lut

    lut = None if tables.normalization is _normalization_lut() else tables.normalization
    tensors = (tables.horizontal or ()) + (tables.vertical or ())
    if lut is not None:
        tensors += (lut,)
    if sum(tensor.nbytes for tensor in tensors) > _MAX_INLINE_TABLE_BYTES:
        return tables.__reduce_ex__(4)
    return _rebuild_resize_tables, (
        tables.source_hw,
        tables.target_hw,
        tables.patch_size,
        tuple(map(_inline_tensor, tables.horizontal)) if tables.horizontal else None,
        tuple(map(_inline_tensor, tables.vertical)) if tables.vertical else None,
        _inline_tensor(lut) if lut is not None else None,
    )


def packed_preprocess_enabled():
    return _pack_cpu_tensors is not None


def pack_preprocess_tensors(tensors):
    return _pack_cpu_tensors(tensors)


def _rebuild_storage_fd(cls, handle, size):
    from libth_transformer_config import map_preprocess_storage

    fd = handle.detach()
    try:
        key = reductions.fd_id(fd)
        storage = reductions.storage_from_cache(cls, key)
        if storage is None:
            storage = map_preprocess_storage(fd, size)
            reductions.shared_cache[key] = reductions.StorageWeakRef(storage)
        return storage
    finally:
        os.close(fd)


def _reduce_storage(storage):
    rebuild, args = reductions.reduce_storage(storage)
    if rebuild is reductions.rebuild_storage_fd:
        rebuild = _rebuild_storage_fd
    return rebuild, args


def register_preprocess_storage_reducer():
    global _pack_cpu_tensors

    # Only the sending worker registers this reducer. Empty tensors and other
    # sharing strategies retain their original reconstruction functions.
    ForkingPickler.register(torch.UntypedStorage, _reduce_storage)
    if (
        os.environ.get("MM_VIT_PACKED_PREPROCESS_IPC") == "1"
        and torch.multiprocessing.get_sharing_strategy() == "file_descriptor"
    ):
        from libth_transformer_config import pack_preprocess_cpu_tensors

        _pack_cpu_tensors = pack_preprocess_cpu_tensors
    else:
        _pack_cpu_tensors = None
        if shared_preprocess_ipc_enabled():
            from rtp_llm.models_py.modules.dsv41.vision_preprocess_tables import ResizeTables

            # RGB keeps the existing FD reducer. Resize metadata is small and
            # does not need additional resource-sharer round trips.
            ForkingPickler.register(ResizeTables, _reduce_resize_tables)
