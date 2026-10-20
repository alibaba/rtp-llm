"""Host-only architecture queries shared with the frozen FlashInfer runtime."""

import functools

import torch
from flashinfer.utils import _get_cache_buf, get_device_sm_count


@functools.cache
def get_compute_capability(device):
    return torch.cuda.get_device_capability(device)


@functools.cache
def get_device_name(device):
    return torch.cuda.get_device_name(device)
