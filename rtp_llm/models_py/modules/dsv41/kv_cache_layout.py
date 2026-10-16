"""Architecture-specific V4.1 cache ABI, shared by allocator and consumers."""


def use_v4_kv_cache(device=None):
    """Hopper uses FlashMLA's V4 planar FP8 cache, not SM100's V4.1 records."""
    import torch

    if device is not None and torch.device(device).type != "cuda":
        return False
    return torch.cuda.is_available() and torch.cuda.get_device_capability(device) == (9, 0)
