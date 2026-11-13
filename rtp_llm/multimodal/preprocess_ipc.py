"""Share CPU worker results using PyTorch's fd protocol without holding the GIL in mmap."""

import os
from multiprocessing.reduction import ForkingPickler

import torch
from torch.multiprocessing import reductions


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
    # Only the sending worker registers this reducer. Empty tensors and other
    # sharing strategies retain their original reconstruction functions.
    ForkingPickler.register(torch.UntypedStorage, _reduce_storage)
