"""Single CPU entry for fixed-M BF16 gate, output allocation and fused routing."""

import os
from collections import OrderedDict

import torch
from rtp_llm.models_py import prefill_optimization as _prefill_optimization
from rtp_llm.models_py.kernels.cuda import prefill_cpu_ops
from rtp_llm.models_py.modules.factory.linear.fixed_m_pad_triton import _pad_rows_kernel
from rtp_llm.models_py.modules.factory.linear.impl.cuda.f16_linear import CudaF16Linear
from rtp_llm.models_py.triton_kernels.common import fast_launch
from rtp_llm.models_py.triton_kernels.common.fast_launch import native_launcher
from rtp_llm.models_py.triton_kernels.moe.router_single_group import _route

ENABLED = _prefill_optimization.ENABLED
_executors = OrderedDict()


def try_route(layer, hidden):
    if not ENABLED or fast_launch._raw_stream is None or torch.compiler.is_compiling():
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
        and fast_launch.knobs is not None
        and fast_launch.hooks_idle()
        and layer.fused_moe.topk_ids_dtype in (torch.int32, torch.int64)
        and prefill_cpu_ops.no_module_hooks(gate)
        and prefill_cpu_ops.no_module_hooks(layer.group_topk)
    ):
        return None
    rows, width = hidden.shape
    step = layer.gate_chunk_rows or rows
    if rows % step and os.environ.get("RTP_LLM_FUSED_ROUTER_PAD", "1") != "1":
        return None
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
        library = prefill_cpu_ops.native()
        if library is None:
            return None
        weights, ids = library.router_outputs(1, 8, hidden.device.index, ids64)
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
    if torch.cuda.current_device() != hidden.device.index:
        with torch.cuda.device(hidden.device):
            return try_route(layer, hidden)
    return executor.run(
        hidden,
        gate.weight,
        bias,
        layer.gate_chunk_rows,
        ids64,
        fast_launch._raw_stream(hidden.device.index),
    )
