# SPDX-License-Identifier: Apache-2.0
"""CUDA helpers for the pinned K3 kernels, independent of the vLLM runtime."""

import contextlib

import torch
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as tldevice

# Preserve the upstream FP32 approximation, including constant folding.
RCP_LN2 = 1.4426950216
cdiv = triton.cdiv
next_power_of_2 = triton.next_power_of_2


class CudaPlatform:
    def is_cuda(self):
        return torch.version.cuda is not None

    def is_cuda_alike(self):
        return self.is_cuda()

    def has_device_capability(self, capability, device_id=0):
        requested = (
            capability if isinstance(capability, tuple)
            else (capability // 10, capability % 10)
        )
        return torch.cuda.get_device_capability(device_id) >= requested


current_platform = CudaPlatform()
# RTP prefill metadata planning already runs outside Graph capture.
gpu_sync_allowed = contextlib.nullcontext


def async_tensor_h2d(data, device=None, dtype=None, out=None):
    tensor = torch.as_tensor(data, device="cpu", dtype=dtype)
    if not tensor.is_pinned():
        tensor = torch.empty(
            tensor.shape, dtype=tensor.dtype, device="cpu", pin_memory=True
        ).copy_(tensor)
    if out is not None:
        return out.copy_(tensor, non_blocking=True)
    return tensor.to(device=device, dtype=dtype, non_blocking=True)
