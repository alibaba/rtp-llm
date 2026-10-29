"""Exact MTP embedding, position mask and paired RMSNorm fusion."""

import logging
from functools import lru_cache
from pathlib import Path

import torch
from torch.nn.modules import module as torch_module
from rtp_llm.models_py import mtp_optimization


def no_module_hooks(module):
    return not (
        module._forward_hooks
        or module._forward_pre_hooks
        or module._backward_hooks
        or module._backward_pre_hooks
        or torch_module._global_forward_hooks
        or torch_module._global_forward_pre_hooks
        or torch_module._global_backward_hooks
        or torch_module._global_backward_pre_hooks
        or module._compiled_call_impl is not None
    )


@lru_cache(maxsize=1)
def native():
    if not mtp_optimization.ENABLED:
        return None
    try:
        from torch.utils.cpp_extension import load

        return load(
            name="rtp_mtp_frontend",
            sources=[str(Path(__file__).with_suffix(".cu"))],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
    except Exception:
        logging.exception("MTP frontend fusion unavailable; using existing path")
        return None


def try_mtp_frontend(native_op, embedding, hidden, e_norm, h_norm, ids, positions):
    if native_op is None or embedding.tp_size != 1:
        return None
    if not all(no_module_hooks(module) for module in (embedding, e_norm, h_norm)):
        return None
    weight = embedding.weight
    if (
        not torch.is_tensor(positions)
        or hidden.ndim != 2
        or hidden.shape[1] != 6144
        or hidden.shape[0] == 0
        or weight.ndim != 2
        or weight.shape[1] != 6144
        or ids.ndim != 1
        or ids.numel() != hidden.shape[0]
        or positions.numel() != hidden.shape[0]
        or ids.dtype != torch.int32
        or positions.dtype != torch.int32
    ):
        return None
    tensors = (weight, hidden, e_norm.weight, h_norm.weight, ids, positions)
    if any(t.device != hidden.device or not t.is_contiguous() for t in tensors):
        return None
    if not hidden.is_cuda or any(t.dtype != torch.bfloat16 for t in tensors[:4]):
        return None
    if torch.cuda.current_device() != hidden.device.index:
        return None
    if e_norm.weight.numel() != 6144 or h_norm.weight.numel() != 6144:
        return None
    if e_norm.variance_epsilon != h_norm.variance_epsilon:
        return None
    pointers = tuple(t.data_ptr() for t in tensors)
    if (pointers[0] | pointers[1] | pointers[2] | pointers[3]) & 15:
        return None
    output = torch.empty(
        (hidden.shape[0], 12288), dtype=hidden.dtype, device=hidden.device
    )
    residual = torch.empty_like(hidden)
    native_op.launch(
        *pointers,
        output.data_ptr(),
        residual.data_ptr(),
        hidden.shape[0],
        6144,
        e_norm.variance_epsilon,
        torch.cuda.current_stream(hidden.device).cuda_stream,
    )
    return output, residual
