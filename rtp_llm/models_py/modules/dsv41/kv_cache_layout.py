"""Architecture-specific V4.1 cache ABI, shared by allocator and consumers."""


def use_v4_kv_cache(device=None):
    """Hopper uses FlashMLA's V4 planar FP8 cache, not SM100's V4.1 records."""
    import torch

    from rtp_llm.device.device_type import is_cuda

    if device is not None and torch.device(device).type != "cuda":
        return False
    return is_cuda() and torch.cuda.get_device_capability(device) == (9, 0)
