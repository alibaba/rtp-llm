"""Receive PyTorch shared-fd preprocessing results without holding the GIL during mmap."""

import os
from multiprocessing.reduction import ForkingPickler

import torch
from torch.multiprocessing import reductions


def _rebuild_storage_fd(cls, handle, size):
    try:
        from libth_transformer_config import map_preprocess_storage
    except ImportError:
        return reductions.rebuild_storage_fd(cls, handle, size)

    fd = handle.detach()
    try:
        key = reductions.fd_id(fd)
        storage = reductions.storage_from_cache(cls, key)
        if storage is not None:
            return storage

        mapped_storage = map_preprocess_storage(fd, size)
        # Another receiver may have populated the cache while mmap released the GIL.
        storage = reductions.storage_from_cache(cls, key)
        if storage is not None:
            return storage
        reductions.shared_cache[key] = reductions.StorageWeakRef(mapped_storage)
        return mapped_storage
    finally:
        os.close(fd)


def _reduce_storage(storage):
    rebuild, args = reductions.reduce_storage(storage)
    if rebuild is reductions.rebuild_storage_fd:
        rebuild = _rebuild_storage_fd
    return rebuild, args


def register_preprocess_storage_reducer():
    if torch.multiprocessing.get_sharing_strategy() == "file_descriptor":
        ForkingPickler.register(torch.UntypedStorage, _reduce_storage)
