"""Checkpoint-owned quantized updates staged before any live parameter is written.

The existing linear checkpoint loader owns TP/fusion and the quantization
method owns runtime layout conversion. A complete quantization unit is needed:
prequantized weights and scales travel together; online quantization accepts
the complete floating-point weight and computes its scales. Partial fused
QKV/gate-up updates cannot reuse an already quantized base without loss.
"""

import copy

import torch

from rtp_llm.models_py.quant_methods.unquantized import UnquantizedLinearMethod
from rtp_llm.models_py.quant_methods.w8a8_int8 import W8A8Int8LinearMethod
from rtp_llm.models_py.quant_methods.weight_only_int8 import WeightOnlyInt8LinearMethod
from rtp_llm.models_py.runtime_weight_update import (
    TensorUpdate,
    prepare_leaf_weight_updates,
)

_LOAD_STATE = (
    "_rtp_loaded_weight_names",
    "_loaded_parameter_names",
    "_loaded_weight_shards",
    "_loaded_parameter_shards",
    "_fused_loaded_parameters",
    "_merged_per_tensor_scales",
    "_qkv_per_tensor_scales",
)


def _is_modelopt_fp4(module) -> bool:
    method_type = type(module.quant_method)
    if method_type.__module__ != "rtp_llm.models_py.quant_methods.fp4":
        return False
    from rtp_llm.models_py.quant_methods.fp4 import ModelOptFp4LinearMethod

    return method_type is ModelOptFp4LinearMethod


def validate_quantized_weight_update(module, weights=None) -> None:
    # Exact types are deliberate: a new subclass may add derived caches which
    # cannot be refreshed merely by copying registered parameters (e.g. FP4).
    from rtp_llm.models_py.quant_methods.fp8 import (
        Fp8BlockLinearMethod,
        Fp8BlockOnlineLinearMethod,
        Fp8LinearMethod,
        Fp8OnlineLinearMethod,
        Fp8PerChannelLinearMethod,
        Fp8PerChannelOnlineLinearMethod,
    )

    supported = (
        UnquantizedLinearMethod,
        WeightOnlyInt8LinearMethod,
        W8A8Int8LinearMethod,
        Fp8LinearMethod,
        Fp8OnlineLinearMethod,
        Fp8PerChannelLinearMethod,
        Fp8PerChannelOnlineLinearMethod,
        Fp8BlockLinearMethod,
        Fp8BlockOnlineLinearMethod,
    )
    # In main, ModelOpt quantization wraps weight descriptors, not their bias
    # descriptors. Its executor receives bias=None and apply reads layer.bias,
    # so a bias-only update does not invalidate packed weights or cached scales.
    fp4_bias_only = (
        _is_modelopt_fp4(module)
        and bool(weights)
        and all(name.rsplit(".", 1)[-1] == "bias" for name in weights)
    )
    if type(module.quant_method) not in supported and not fp4_bias_only:
        raise NotImplementedError(
            f"{type(module.quant_method).__name__} has no atomic weight/scale update contract"
        )
    if type(module.quant_method) is W8A8Int8LinearMethod:
        # The PPU implementation uses reshape(...).contiguous(). If a future
        # backend copies or repacks those inputs, an in-place install would
        # silently leave its executable state stale. Check actual live views.
        method = module.quant_method
        exe = method._executor
        if exe is not None and (
            method._storage != method._storage_signature(module)
            or exe.weight.data_ptr() != module.weight.data_ptr()
            or exe.weight_scale.data_ptr() != module.weight_scale.data_ptr()
        ):
            raise RuntimeError("PPU W8A8 executor does not alias the live weight/scale")
    if list(module.children()):
        raise NotImplementedError("Quantized updates require leaf-owned conversion")


def _prepare_quantized_bias_updates(module, weights) -> list[TensorUpdate]:
    if module.bias is None:
        raise ValueError("Cannot update bias on a bias-free quantized projection")
    # Avoid copying the potentially huge quantized weight/scale payload just to
    # replace a bias. The owner's loader needs only bias and layout metadata.
    staged = copy.copy(module)
    staged._parameters = {
        "bias": torch.nn.Parameter(
            module.bias.detach().to(device="cpu", copy=True), requires_grad=False
        )
    }
    staged._buffers = {}
    staged.quant_method = copy.copy(module.quant_method)
    staged.quant_method._pieces = {}
    staged.quant_method._scalar_pieces = {}
    for name in _LOAD_STATE:
        value = getattr(module, name, None)
        if isinstance(value, (set, dict)):
            setattr(staged, name, type(value)())
    staged.load_weights(weights)
    # Partial q/k/v bias preserves the other slices in the private bias copy;
    # it need not mark the complete fused bias loaded to be a valid update.
    return [TensorUpdate("bias", module.bias, staged.bias)]


@torch.inference_mode()
def prepare_quantized_weight_updates(module, weights) -> list[TensorUpdate]:
    validate_quantized_weight_update(module, weights)
    if not weights:
        raise ValueError("Quantized update requires checkpoint tensors")
    if _is_modelopt_fp4(module) or (
        type(module.quant_method) in (WeightOnlyInt8LinearMethod, W8A8Int8LinearMethod)
        and all(name.rsplit(".", 1)[-1] == "bias" for name in weights)
    ):
        return _prepare_quantized_bias_updates(module, weights)
    if type(module.quant_method) is UnquantizedLinearMethod:
        prepare = getattr(module.quant_method, "prepare_weight_updates", None)
        if callable(prepare):
            return prepare(module, weights)
        return prepare_leaf_weight_updates(module, weights, reset_fields=_LOAD_STATE)
    # Bias is not quantized and can be changed independently, using the same
    # fused/sharded loader as initial loading without touching weight/scales.
    if all(name.rsplit(".", 1)[-1] == "bias" for name in weights):
        return prepare_leaf_weight_updates(module, weights, reset_fields=_LOAD_STATE)

    staged = copy.copy(module)
    staged._parameters = {}
    staged._buffers = {
        name: None if value is None else value.detach().cpu().clone()
        for name, value in module._buffers.items()
    }
    staged.quant_method = copy.copy(module.quant_method)
    compressed_int8 = type(module.quant_method) is W8A8Int8LinearMethod
    if compressed_int8:
        staged.quant_method._pieces = {}
        staged.quant_method._executor = None
        staged.quant_method._storage = None
    # Cached ROCm executors on the LIVE method retain references to the live
    # Parameters. Conversion creates a separate executor on this private copy.
    staged.quant_method.__dict__.pop("_rocm_executor", None)
    staged._post_load_done = False
    staged._main_weight_loaded = False
    for name in _LOAD_STATE:
        value = getattr(module, name, None)
        if isinstance(value, (set, dict)):
            setattr(staged, name, type(value)())
    staged.quant_method.create_weights(
        staged,
        input_size=module.input_size,
        output_size=module.output_size,
        params_dtype=module.params_dtype,
    )
    bias = module.bias
    if bias is None:
        staged.bias = None
    else:
        staged.register_parameter(
            "bias", torch.nn.Parameter(bias.detach().cpu().clone(), requires_grad=False)
        )
    staged.load_weights(weights)
    # An unchanged bias is a valid independent unit; weight and all required
    # scales must nevertheless be complete in THIS update, not a prior load.
    if bias is not None:
        staged._loaded_parameter_names.add("bias")
    try:
        staged.validate_weights_loaded()
    except RuntimeError as error:
        raise ValueError(
            "Quantized update requires a complete weight/scale unit, including "
            "every fused QKV or gate/up shard"
        ) from error
    # Compressed W8A8 is already in the PPU executor's canonical [N,K]/[N,1]
    # storage. Validate on CPU, then copy into the live views in the transaction.
    # Never build a temporary executor or swap a cached PPU storage pointer.
    if not compressed_int8:
        staged.to(device=module.weight.device)
    staged.process_weights_after_loading()
    if getattr(staged.quant_method, "_runtime_backend", None) != getattr(
        module.quant_method, "_runtime_backend", None
    ):
        raise RuntimeError("Quantized update changed the runtime backend")
    for name in ("_fp8_logical_input_size", "_fp8_logical_output_size"):
        if getattr(staged, name, None) != getattr(module, name, None):
            raise RuntimeError(f"Quantized update changed {name}")
    if staged._parameters.keys() != module._parameters.keys():
        raise RuntimeError("Quantized update changed parameter registrations")
    updates = []
    for name, target in module._parameters.items():
        source = staged._parameters[name]
        if target is None and source is None:
            continue
        if (
            target is None
            or source is None
            or source.shape != target.shape
            or source.dtype != target.dtype
        ):
            raise ValueError(f"Quantized update changed runtime shape/dtype for {name}")
        updates.append(TensorUpdate(name, target, source.detach().cpu().clone()))
    return updates
