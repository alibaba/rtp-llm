"""Ordinary-router/GroupTopK/group32 pack fusion; no dispatch hook here.

Inputs are the existing ordinary BF16 norm and BF16-gate logits cast to FP32.
Only fixed GLM5 n_group=topk_group=1, experts256/topk8, width6144 is supported.
Outputs are caller-owned contiguous Mega views; no inputs are overwritten.
"""

import hashlib
from functools import lru_cache
from pathlib import Path

import torch

_BINDING = """
#include <torch/extension.h>
void initialize_router_pack();
void launch_router_pack(const at::Tensor&, const at::Tensor&, const at::Tensor&,
                        const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("initialize", &initialize_router_pack);
  m.def("launch", &launch_router_pack);
}
"""
_CXX_FLAGS = ("-O2", "-std=c++17")
_CUDA_FLAGS = ("-O3", "-std=c++17", "-gencode=arch=compute_100f,code=sm_100f")


@lru_cache(maxsize=1)
def _load_extension():
    if torch.cuda.is_initialized() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Warm up GLM5 router/pack before capture")
    from torch.utils.cpp_extension import load_inline

    source = (
        Path(__file__).with_name("glm5_moe_router_pack_src") / "kernel.cu"
    ).read_text()
    identity = (
        source,
        _BINDING,
        torch.__version__,
        torch.version.cuda,
        torch.compiled_with_cxx11_abi(),
        _CXX_FLAGS,
        _CUDA_FLAGS,
    )
    digest = hashlib.sha256(repr(identity).encode())
    return load_inline(
        name="glm5_moe_router_pack_" + digest.hexdigest()[:16],
        cpp_sources=_BINDING,
        cuda_sources=source,
        extra_cflags=list(_CXX_FLAGS),
        extra_cuda_cflags=list(_CUDA_FLAGS),
        with_cuda=True,
    )


@lru_cache(maxsize=None)
def _prepare_device(index):
    with torch.cuda.device(index):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Warm up GLM5 router/pack on this device before capture")
        if torch.cuda.get_device_capability(index) not in ((10, 0), (10, 3)):
            raise RuntimeError("GLM5 router/pack requires SM100 or SM103")
        extension = _load_extension()
        extension.initialize()
        return extension


def _check_output_ranges(tensors):
    # Mega views may be disjoint slices of one symmetric-memory allocation.
    ranges = [
        (x.data_ptr(), x.data_ptr() + x.numel() * x.element_size()) for x in tensors
    ]
    for index in range(3, 7):
        begin, end = ranges[index]
        if any(
            begin < other_end and other_begin < end
            for other_begin, other_end in ranges[:index]
        ):
            raise ValueError("outputs must not overlap inputs or other outputs")


def fused_router_pack(
    hidden, logits, bias, *, activation_out, scales_out, topk_ids_out, topk_weights_out
):
    tensors = (
        hidden,
        logits,
        bias,
        activation_out,
        scales_out,
        topk_ids_out,
        topk_weights_out,
    )
    if not all(isinstance(x, torch.Tensor) for x in tensors):
        raise ValueError("all operands must be tensors")
    if hidden.ndim != 2 or hidden.shape[1] != 6144 or not 1 <= hidden.shape[0] <= 256:
        raise ValueError("hidden must be [M,6144], 1<=M<=256")
    rows = hidden.shape[0]
    expected = (
        (torch.bfloat16, (rows, 6144)),
        (torch.float32, (rows, 256)),
        (torch.float32, (256,)),
        (torch.float8_e4m3fn, (rows, 6144)),
        (torch.int32, (rows, 48)),
        (torch.int64, (rows, 8)),
        (torch.float32, (rows, 8)),
    )
    for tensor, (dtype, shape) in zip(tensors, expected):
        if (
            tensor.dtype != dtype
            or tuple(tensor.shape) != shape
            or not tensor.is_contiguous()
        ):
            raise ValueError("unsupported router/pack tensor dtype, shape or stride")
        if not tensor.is_cuda or tensor.device != hidden.device:
            raise ValueError("operands must share one CUDA device")
    _check_output_ranges(tensors)
    with torch.cuda.device(hidden.device):
        _prepare_device(hidden.device.index).launch(*tensors)
    return activation_out, scales_out, topk_ids_out, topk_weights_out
