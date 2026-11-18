"""Forward-owned scratch for plain TP1 Qwen3 eager prefill.

No graph capture, fused residual arithmetic, shared request buffers or model
monkeypatching. Unknown AITER descriptor ABIs retain the regular module path.
"""

import logging
import os
from collections.abc import Mapping
from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def _native_ops():
    try:
        from aiter.jit.core import _pybind_develop_hooks, get_module

        norm = get_module("module_rmsnorm_quant")
        activation = get_module("module_activation")
        convert, tensor_type, raw_stream, current_device = _pybind_develop_hooks()
        # Older AITER releases accept torch.Tensor instead. Never pass their
        # entrypoints borrowed descriptors using an unrecognized signature.
        for fn in (norm.rmsnorm, activation.silu_and_mul):
            signature = (fn.__doc__ or "").split("\n", 1)[0]
            if "aiter_tensor_t" not in signature or "out:" not in signature:
                return None, "native_descriptor_signature"
        if tensor_type is not torch.Tensor:
            return None, "native_tensor_type"
        if not all(
            callable(fn)
            for fn in (
                convert,
                raw_stream,
                current_device,
                norm._set_current_hip_stream,
                activation._set_current_hip_stream,
            )
        ):
            return None, "native_stream_hooks"
        return (norm, activation, convert, raw_stream), None
    except (ImportError, AttributeError, RuntimeError, TypeError) as error:
        return None, "native_unavailable:%s" % type(error).__name__


def _fallback(model, reason):
    # Fixed reason codes (not request IDs or token counts): at most one message
    # per reason and model initialization, including repeated unsupported calls.
    reasons = model._eager_prefill_fallback_reasons
    if reason not in reasons:
        reasons.add(reason)
        logging.info(
            "Qwen3 eager prefill fallback: reason=%s; retaining module dispatch",
            reason,
        )
    return None


def _projection_layout_supported(weight):
    # The existing GEMM handles row- and column-major matrices. In particular,
    # swizzle_tensor(weight.t(), False).t() is a column-major *view*, not a
    # contiguous tensor. Preserve it: materializing contiguous() changes the
    # physical swizzle layout expected by bpreshuffle=True.
    return (
        weight.layout == torch.strided
        and weight.dim() == 2
        and min(weight.shape) > 0
        and (
            (weight.stride(1) == 1 and weight.stride(0) >= weight.shape[1])
            or (weight.stride(0) == 1 and weight.stride(1) >= weight.shape[0])
        )
    )


def _weight_metadata(weight):
    strided = weight.layout == torch.strided
    return (
        tuple(weight.shape),
        tuple(weight.stride()) if strided else None,
        weight.data_ptr() if strided and weight.device.type != "meta" else None,
        weight.dtype,
        weight.device,
        weight.requires_grad,
    )


_CONFIG_ATTRIBUTES = (
    "tp_size",
    "ffn_tp_size",
    "is_gated",
    "layer_idx",
    "variance_epsilon",
    "_direct_rmsnorm",
)


def _configuration_signature(modules):
    return tuple(
        tuple(attributes.get(name) for name in _CONFIG_ATTRIBUTES)
        for attributes in (vars(module) for module in modules)
    )


class _FailedPlan:
    """Negative cache, invalidated by module/weight/layout/config changes."""

    def __init__(self, modules, reason):
        self.modules = modules
        self.reason = reason
        self.config = _configuration_signature(modules)
        # Retain owners, not just id(tensor), so replacing a weight cannot
        # accidentally hit the negative cache through Python object-ID reuse.
        self.weights = tuple(
            (module, weight, _weight_metadata(weight))
            for module in modules
            if isinstance(weight := getattr(module, "weight", None), torch.Tensor)
        )
        self.non_tensor_weights = tuple(
            (module, getattr(module, "weight", None))
            for module in modules
            if not isinstance(getattr(module, "weight", None), torch.Tensor)
        )

    def matches(self, modules):
        return (
            modules == self.modules
            and self.config == _configuration_signature(modules)
            and all(
                getattr(module, "weight", None) is weight
                and _weight_metadata(weight) == metadata
                for module, weight, metadata in self.weights
            )
            and all(
                getattr(module, "weight", None) is weight
                for module, weight in self.non_tensor_weights
            )
        )


def _has_hooks(module):
    return any(
        getattr(module, name, None)
        for name in (
            "_forward_hooks",
            "_forward_pre_hooks",
            "_backward_hooks",
            "_backward_pre_hooks",
        )
    )


class _Plan:
    def __init__(self, model, native):
        from rtp_llm.models_py.model_desc.qwen3 import Qwen3DecoderLayer
        from rtp_llm.models_py.modules.base.rocm.activation import FusedSiluAndMul
        from rtp_llm.models_py.modules.base.rocm.norm import RMSNorm
        from rtp_llm.models_py.modules.factory.linear.impl.rocm.f16_linear import (
            RocmF16LinearNoSwizzle,
            RocmF16LinearWithSwizzle,
        )
        from rtp_llm.models_py.modules.hybrid.causal_attention import CausalAttention
        from rtp_llm.models_py.modules.hybrid.dense_mlp import DenseMLP

        self.native = native
        self.modules = tuple(model.modules())
        self.config = _configuration_signature(self.modules)
        self.weights = []
        self.ops = []
        convert = native[2]
        shapes = None
        for layer in model.layers:
            a, m = layer.self_attn, layer.mlp
            if (
                type(layer) is not Qwen3DecoderLayer
                or type(a) is not CausalAttention
                or type(m) is not DenseMLP
                or a.tp_size != 1
                or m.ffn_tp_size != 1
                or not m.is_gated
                or type(m.act_fn) is not FusedSiluAndMul
            ):
                raise ValueError("unsupported_decoder")
            projections = (a.qkv_proj, a.o_proj, m.up_proj, m.down_proj)
            norms = (layer.input_layernorm, layer.post_attention_layernorm)
            if any(
                type(p) not in (RocmF16LinearWithSwizzle, RocmF16LinearNoSwizzle)
                for p in projections
            ) or any(type(n) is not RMSNorm or not n._direct_rmsnorm for n in norms):
                raise ValueError("unsupported_projection_or_norm")
            current_shapes = tuple(tuple(p.weight.shape) for p in projections)
            if shapes is not None and shapes != current_shapes:
                raise ValueError("nonuniform_layers")
            shapes = current_shapes
            for module in (*projections, *norms):
                weight = module.weight
                if (
                    weight.requires_grad
                    or weight.dtype not in (torch.bfloat16, torch.float16)
                    or not weight.is_cuda
                ):
                    raise ValueError("weight_device_dtype_or_grad")
                if module in norms:
                    if weight.dim() != 1 or not weight.is_contiguous():
                        raise ValueError("norm_weight_layout")
                elif not _projection_layout_supported(weight):
                    raise ValueError("projection_weight_layout")
                if self.weights and (
                    weight.dtype != self.weights[0][1].dtype
                    or weight.device != self.weights[0][1].device
                ):
                    raise ValueError("inconsistent_weight_device_dtype")
                self.weights.append((module, weight, _weight_metadata(weight)))
            hidden = norms[0].weight.numel()
            if (
                not 0 < hidden <= 8192
                or hidden % 2
                or any(n.weight.dim() != 1 or n.weight.numel() != hidden for n in norms)
                or m.up_proj.weight.dim() != 2
                or m.up_proj.weight.shape[1] % 2
                or any(p.weight.dim() != 2 for p in projections)
                or a.qkv_proj.weight.shape[0] != hidden
                or a.o_proj.weight.shape[1] != hidden
                or m.up_proj.weight.shape[0] != hidden
                or m.down_proj.weight.shape != (m.up_proj.weight.shape[1] // 2, hidden)
            ):
                raise ValueError("unsupported_width")
            self.ops.append(
                (
                    convert(norms[0].weight),
                    norms[0].variance_epsilon,
                    a.qkv_proj.forward,
                    None if a.qk_fuse_norm is None else a.qk_fuse_norm.forward,
                    a.layer_idx,
                    a.o_proj.forward,
                    convert(norms[1].weight),
                    norms[1].variance_epsilon,
                    m.up_proj.forward,
                    m.down_proj.forward,
                )
            )
        if not self.ops:
            raise ValueError("empty_stage")
        self.hidden = hidden
        self.width = model.layers[0].mlp.up_proj.weight.shape[1] // 2

    def matches(self, modules):
        return (
            modules == self.modules
            and self.config == _configuration_signature(modules)
            and all(
                m.weight is w and _weight_metadata(w) == metadata
                for m, w, metadata in self.weights
            )
        )

    def accepts_input(self, x):
        return (
            x.is_cuda
            and x.dim() == 2
            and x.shape[0] > 0
            and x.shape[1] == self.hidden
            and x.is_contiguous()
            and x.dtype == self.weights[0][1].dtype
            and x.device == self.weights[0][1].device
        )

    def workspace(self, x):
        return _Workspace(self, x)


class _Workspace:
    def __init__(self, plan, x):
        self.plan = plan
        norm, activation, self.convert, raw_stream = plan.native
        # Bind the input's device/stream, not an unrelated current device.
        stream = raw_stream(x.device.index)
        norm._set_current_hip_stream(stream)
        activation._set_current_hip_stream(stream)
        self.norm = norm.rmsnorm
        self.activation = activation.silu_and_mul
        self.normalized = torch.empty_like(x)
        self.residual_a = torch.empty_like(x)
        self.residual_b = torch.empty_like(x)
        self.activated = torch.empty(
            (x.shape[0], plan.width), dtype=x.dtype, device=x.device
        )
        # Keep every owning tensor alive as long as its borrowed descriptor.
        self.nr, self.ar, self.br, self.actr, self.xr = (
            self.convert(t)
            for t in (
                self.normalized,
                self.residual_a,
                self.residual_b,
                self.activated,
                x,
            )
        )
        self.input_owner = x

    def layer(self, index, x, fmha, cache):
        w0, e0, qkv_fn, qk_fn, idx, out_fn, w1, e1, up_fn, down_fn = self.plan.ops[
            index
        ]
        self.norm(self.nr, self.br if index else self.xr, w0, e0)
        qkv = qkv_fn(self.normalized)
        if qk_fn is not None:
            qkv = qk_fn(qkv)
        attended = fmha.forward(qkv, cache, idx)
        attended = attended.reshape(x.shape[0], -1).contiguous()
        projected = out_fn(attended)
        # Separate adds intentionally preserve the original BF16 rounding.
        torch.add(x, projected, out=self.residual_a)
        self.norm(self.nr, self.ar, w1, e1)
        up = up_fn(self.normalized)
        self.activation(self.actr, self.convert(up))
        down = down_fn(self.activated)
        torch.add(self.residual_a, down, out=self.residual_b)
        return self.residual_b


def prepare_workspace(model, inputs, x, capture_enabled):
    """Select before any layer/KV writes; never retry a partially run forward."""
    from rtp_llm.models_py.model_desc.qwen3 import Qwen3Model

    if torch.version.hip is None or type(model) is not Qwen3Model:
        return None
    attention = inputs.attention_inputs
    if isinstance(attention, Mapping):
        if len(attention) != 1:
            return _fallback(model, "attention_groups")
        attention = next(iter(attention.values()))
    # Decode uses its existing path by design, not a failed prefill plan.
    if not attention.is_prefill:
        return None
    if os.environ.get("RTP_LLM_ROCM_EAGER_PREFILL_SCRATCH", "1") != "1":
        return _fallback(model, "disabled")
    if capture_enabled or inputs.capture_hidden_states:
        return _fallback(model, "hidden_state_capture")
    if torch.is_grad_enabled() or torch.is_autocast_enabled() or torch.jit.is_tracing():
        return _fallback(model, "grad_autocast_or_tracing")
    if attention.is_cuda_graph or torch.cuda.is_current_stream_capturing():
        return _fallback(model, "graph_capture")
    # Recheck hooks each forward, including hooks installed after initialization.
    from torch.nn.modules import module as torch_module

    if any(
        value
        for key, value in vars(torch_module).items()
        if key.startswith("_global_") and key.endswith("hooks")
    ):
        return _fallback(model, "global_hooks")
    modules = tuple(model.modules())
    if any(_has_hooks(m) or "forward" in m.__dict__ for m in modules):
        return _fallback(model, "module_hooks_or_forward_override")
    plan = model._eager_prefill_plan
    if plan is not None and not plan.matches(modules):
        model._eager_prefill_plan = plan = None
        _fallback(model, "model_or_weight_changed")
    if plan is None:
        failure = model._eager_prefill_failure
        if failure is not None and failure.matches(modules):
            return _fallback(model, failure.reason)
        model._eager_prefill_failure = None
        native, reason = _native_ops()
        if native is None:
            model._eager_prefill_failure = _FailedPlan(modules, reason)
            return _fallback(model, reason)
        try:
            plan = _Plan(model, native)
        except (ValueError, AttributeError, TypeError) as error:
            reason = (
                "plan:%s" % error
                if isinstance(error, ValueError)
                else "plan:%s" % type(error).__name__
            )
            model._eager_prefill_failure = _FailedPlan(modules, reason)
            return _fallback(model, reason)
        model._eager_prefill_plan = plan
        logging.info(
            "Qwen3 eager prefill: bound %d TP1 layers with forward-owned scratch; "
            "projection layouts preserve original GEMM/swizzle storage",
            len(plan.ops),
        )
    if not plan.accepts_input(x):
        # Input eligibility is per-request: do not evict a valid model plan.
        return _fallback(model, "input_layout_dtype_or_device")
    return plan.workspace(x)
