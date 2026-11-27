"""Run fixed-row GLM5 routing in one host call on supported Triton CUDA builds."""

import logging
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

import torch
import triton

from rtp_llm.models_py.kernels.cuda.mtp_frontend import no_module_hooks
from rtp_llm.models_py.modules.factory.linear.fixed_m_pad_triton import _pad_rows_kernel
from rtp_llm.models_py.modules.factory.linear.impl.cuda.f16_linear import CudaF16Linear
from rtp_llm.models_py.triton_kernels.moe.router_single_group import _route

try:
    from triton import knobs
except ImportError:
    knobs = None

_executors = OrderedDict()
_raw_stream = getattr(torch._C, "_cuda_getCurrentRawStream", None)
_hook_chain_type = getattr(knobs, "HookChain", None)


@lru_cache(maxsize=1)
def native():
    try:
        from torch.utils.cpp_extension import load

        return load(
            name="rtp_glm5_router",
            sources=[str(Path(__file__).with_suffix(".cc"))],
            extra_cflags=["-O3", "-g0"],
            extra_ldflags=["-ldl"],
            with_cuda=False,
        )
    except (ImportError, OSError, RuntimeError):
        logging.exception("Native GLM5 router unavailable")
        return None


def hooks_idle():
    if knobs is None:
        return False
    enter, leave = knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook
    return (
        enter is None or (type(enter) is _hook_chain_type and not enter.calls)
    ) and (leave is None or (type(leave) is _hook_chain_type and not leave.calls))


def native_launcher(kernel):
    """Prepare only the standard CUDA ABI; retain unsupported launch modes."""
    runner = kernel.run
    cached = False
    if (
        triton.__version__.split(".")[:2] == ["3", "6"]
        and torch.version.hip is None
        and getattr(runner, "global_scratch_size", -1) == 0
        and getattr(runner, "profile_scratch_size", -1) == 0
        and not getattr(kernel.metadata, "tensordesc_meta", None)
        and not getattr(runner, "launch_cooperative_grid", True)
        and not getattr(runner, "launch_pdl", True)
        and getattr(kernel.metadata, "num_ctas", 1) == 1
    ):
        names = kernel.src.fn.arg_names

        def position(key):
            key = key[0] if isinstance(key, tuple) else key
            return names.index(key) if isinstance(key, str) else key

        constants = {position(k) for k in kernel.src.constants}
        types = {"i32": 1, "u32": 2, "i64": 3, "u64": 4, "fp32": 5, "fp64": 6}
        slots, codes = [], []
        for key, dtype in sorted(
            kernel.src.signature.items(), key=lambda item: position(item[0])
        ):
            index = position(key)
            if index in constants or dtype == "constexpr":
                continue
            if not isinstance(dtype, str):
                break
            code = (
                0
                if isinstance(dtype, str) and dtype.startswith("*")
                else types.get(dtype)
            )
            if code is None:
                break
            slots.append(index)
            codes.append(code)
        else:
            library = native()
            if library is not None and len(slots) <= 64:
                cached = library.NativeLauncher(
                    kernel.function,
                    kernel.metadata.shared,
                    kernel.metadata.num_warps,
                    slots,
                    codes,
                    kernel,
                )
    return cached


def try_route(layer, hidden):
    if (
        triton.__version__.split(".")[:2] != ["3", "6"]
        or _raw_stream is None
        or torch.compiler.is_compiling()
    ):
        return None
    gate, bias, config = layer.gate, layer.correction_bias, layer.config
    if not (
        type(gate) in (CudaF16Linear, torch.nn.Linear)
        and torch.version.hip is None
        and gate.bias is None
        and bias is not None
        and config.moe_n_group == config.moe_topk_group == 1
        and layer.top_k == 8
        and hidden.is_cuda
        and hidden.dtype == torch.bfloat16
        and hidden.ndim == 2
        and hidden.shape[0] > 0
        and hidden.is_contiguous()
        and hidden.shape[1] % 16 == 0
        and not hidden.requires_grad
        and gate.weight.dtype == torch.bfloat16
        and gate.weight.shape == (256, hidden.shape[1])
        and gate.weight.device == hidden.device
        and not gate.weight.requires_grad
        and bias.dtype == torch.float32
        and bias.shape == (256,)
        and bias.device == hidden.device
        and bias.is_contiguous()
        and not bias.requires_grad
        and bias.data_ptr() % 16 == 0
        and hidden.data_ptr() % 16 == 0
        and knobs is not None
        and hooks_idle()
        and layer.fused_moe.topk_ids_dtype in (torch.int32, torch.int64)
        and no_module_hooks(gate)
        and no_module_hooks(layer.group_topk)
    ):
        return None
    if torch.cuda.current_device() != hidden.device.index:
        with torch.cuda.device(hidden.device):
            return try_route(layer, hidden)
    rows, width = hidden.shape
    step = layer.gate_chunk_rows or rows
    if step * width >= 2**31:
        return None
    ids64 = layer.fused_moe.topk_ids_dtype == torch.int64
    key = (
        hidden.device.index,
        ids64,
        config.has_moe_norm,
        config.routed_scaling_factor,
    )
    executor = _executors.get(key)
    if executor is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        library = native()
        if library is None:
            return None
        weights = torch.empty((1, 8), device=hidden.device, dtype=torch.float32)
        ids = torch.empty(
            (1, 8), device=hidden.device, dtype=layer.fused_moe.topk_ids_dtype
        )
        route = _route.warmup(
            hidden,
            bias,
            weights,
            ids,
            256,
            8,
            config.has_moe_norm,
            config.routed_scaling_factor,
            grid=(1,),
            num_warps=1,
            enable_fp_fusion=False,
        )
        pad = _pad_rows_kernel.warmup(
            hidden,
            hidden,
            width * min(rows, step),
            width * step,
            1024,
            grid=(1,),
            num_warps=4,
        )
        route, pad = native_launcher(route), native_launcher(pad)
        if not route or not pad:
            return None
        executor = library.NativeRouter(route, pad)
        if len(_executors) >= 128:
            _executors.popitem(last=False)
        _executors[key] = executor
    return executor.run(
        hidden,
        gate.weight,
        bias,
        layer.gate_chunk_rows,
        ids64,
        _raw_stream(hidden.device.index),
    )
