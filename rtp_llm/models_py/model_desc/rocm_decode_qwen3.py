"""Opt-in, graph-only residual fusion for the validated ROCm Qwen3 shape.

This is not a Prefill graph. Unsupported models, hooks, dtypes, attention
layouts and graph batch sizes retain Qwen3Model.forward's regular path.
"""

import logging
from collections.abc import Mapping

import torch


def _fallback(model, reason):
    reasons = model._rocm_decode_fallback_reasons
    if reason not in reasons:
        reasons.add(reason)
        logging.info("Qwen3 ROCm decode residual fusion fallback: reason=%s", reason)
    return None


def _norm_supported(norm, device):
    from rtp_llm.models_py.modules.base.rocm.norm import RMSNorm

    weight = getattr(norm, "weight", None)
    return (
        type(norm) is RMSNorm
        and norm._direct_rmsnorm
        and isinstance(weight, torch.Tensor)
        and weight.dtype == torch.bfloat16
        and weight.device == device
        and weight.shape == (2560,)
        and weight.is_contiguous()
        and not weight.requires_grad
    )


def _has_hooks(model):
    from torch.nn.modules import module as module_runtime

    if any(
        getattr(module_runtime, name, None)
        for name in (
            "_global_forward_hooks",
            "_global_forward_pre_hooks",
            "_global_backward_hooks",
            "_global_backward_pre_hooks",
        )
    ):
        return True
    return any(
        getattr(module, name, None)
        for module in model.modules()
        for name in (
            "_forward_hooks",
            "_forward_pre_hooks",
            "_backward_hooks",
            "_backward_pre_hooks",
        )
    )


def _rounding_supported(model, norm, device):
    """Probe outside capture; future AITER round-to-nearest must fall back.

    The inspected native module_rmsnorm_quant stores BF16 with RTZ. Reduction
    order can still change a value by one BF16 ULP: this is not a bitwise
    equivalence guarantee or a replacement for model-level precision checks.
    """
    state = model._rocm_decode_rounding_state
    key = (device, type(norm), norm.variance_epsilon)
    if state is not None and state[0] == key:
        return state[1]
    if torch.cuda.is_current_stream_capturing():
        return False
    from rtp_llm.models_py.model_desc.rocm_rounded_add_norm import rounded_add_norm

    values = torch.arange(2560, device=device, dtype=torch.float32)
    # Do not calibrate against checkpoint values (e.g. an all-zero gamma).
    weight = torch.ones(2560, device=device, dtype=torch.bfloat16)
    reference_norm = type(norm)(weight, eps=norm.variance_epsilon)
    x = ((values.remainder(31) - 15) / 17).to(torch.bfloat16).view(1, -1)
    residual = ((values.remainder(17) - 8) / 19).to(torch.bfloat16).view(1, -1)
    expected_residual = x + residual
    expected = reference_norm(expected_residual)
    actual, actual_residual = rounded_add_norm(
        x, residual, weight, norm.variance_epsilon
    )
    compatible = (
        torch.equal(expected_residual, actual_residual)
        and torch.allclose(actual, expected, rtol=0.008, atol=0)
        and int(torch.count_nonzero(actual != expected).item()) <= 2
    )
    model._rocm_decode_rounding_state = (key, compatible)
    return compatible


def forward_decode(model, inputs, fmha_impl=None):
    # Examine the role before importing Triton or traversing modules. Inference
    # Decode is replayed by the native graph runner, not by this Python loop.
    ai = inputs.attention_inputs
    if (
        torch.version.hip is None
        or torch.is_grad_enabled()
        or torch.compiler.is_compiling()
        or isinstance(ai, Mapping)
        or ai.is_prefill
        or not ai.is_cuda_graph
        or inputs.capture_hidden_states
    ):
        return None

    from rtp_llm.models_py.model_desc.qwen3 import Qwen3DecoderLayer, Qwen3Model

    if (
        type(model) is not Qwen3Model
        or model.embed_tokens is None
        or model.norm is None
        or model.layer_num != 36
        or len(model.layers) != 36
        or model._local_global_layer_ids != tuple(range(36))
        or model.parallelism_config.pp_size != 1
        or model.parallelism_config.world_size != 1
    ):
        return _fallback(model, "model_or_pipeline")
    if inputs.input_ids.dim() != 1 or not 0 < inputs.input_ids.numel() <= 8:
        return _fallback(model, "decode_shape_or_embedding")
    if _has_hooks(model):
        return _fallback(model, "module_hooks")
    if fmha_impl is None:
        fmha_impl = model.prepare_fmha_impl(inputs)
    if isinstance(fmha_impl, Mapping):
        return _fallback(model, "tagged_attention")

    device = inputs.input_ids.device
    if device.type != "cuda":
        return _fallback(model, "input_device")
    arch = getattr(torch.cuda.get_device_properties(device), "gcnArchName", "").split(
        ":", 1
    )[0]
    if arch != "gfx942":
        return _fallback(model, "gpu_architecture")
    from rtp_llm.models_py.modules.base.common.embedding import Embedding

    embedding_weight = getattr(model.embed_tokens, "weight", None)
    if (
        type(model.embed_tokens) is not Embedding
        or not isinstance(embedding_weight, torch.Tensor)
        or embedding_weight.dtype != torch.bfloat16
        or embedding_weight.device != device
        or embedding_weight.dim() != 2
        or embedding_weight.shape[1] != 2560
    ):
        return _fallback(model, "decode_shape_or_embedding")
    norms = [model.norm]
    for layer in model.layers:
        if (
            type(layer) is not Qwen3DecoderLayer
            or layer.self_attn.tp_size != 1
            or layer.mlp.ffn_tp_size != 1
        ):
            return _fallback(model, "decoder_or_parallelism")
        for projection in (
            layer.self_attn.qkv_proj,
            layer.self_attn.o_proj,
            layer.mlp.up_proj,
            layer.mlp.down_proj,
        ):
            if type(projection).__name__ not in (
                "RocmF16LinearWithSwizzle",
                "RocmF16LinearNoSwizzle",
            ):
                return _fallback(model, "projection_type")
        norms.extend((layer.input_layernorm, layer.post_attention_layernorm))
    if not all(_norm_supported(norm, device) for norm in norms):
        return _fallback(model, "normalization_contract")
    if not _rounding_supported(model, model.norm, device):
        return _fallback(model, "normalization_rounding_or_unwarmed_capture")
    caches = model._get_cached_layer_kv_caches()
    if caches is None:
        return _fallback(model, "native_kv_cache")

    h = model.embed_tokens(inputs.input_ids)
    if (
        h.dtype != torch.bfloat16
        or h.dim() != 2
        or not 0 < h.shape[0] <= 8
        or h.shape[1] != 2560
        or not h.is_contiguous()
    ):
        return _fallback(model, "decode_shape")
    from rtp_llm.models_py.model_desc.rocm_rounded_add_norm import rounded_add_norm
    from rtp_llm.ops.compute_ops import PyModelOutputs

    residual = None
    for index, layer in enumerate(model.layers):
        if residual is None:
            residual = h
            h = layer.input_layernorm(h)
        else:
            h, residual = rounded_add_norm(
                h,
                residual,
                layer.input_layernorm.weight,
                layer.input_layernorm.variance_epsilon,
            )
        h = layer.self_attn(
            hidden_states=h, fmha_impl=fmha_impl, kv_cache=caches[index]
        )
        h, residual = rounded_add_norm(
            h,
            residual,
            layer.post_attention_layernorm.weight,
            layer.post_attention_layernorm.variance_epsilon,
        )
        h = layer.mlp(h)
    h, residual = rounded_add_norm(
        h, residual, model.norm.weight, model.norm.variance_epsilon
    )
    if not model._rocm_decode_active_logged:
        logging.info(
            "Qwen3 ROCm decode residual fusion activated: layers=36 hidden=2560 batch=%s; "
            "72 standalone add kernels removed; Prefill unchanged",
            h.shape[0],
        )
        model._rocm_decode_active_logged = True
    return PyModelOutputs(h)
