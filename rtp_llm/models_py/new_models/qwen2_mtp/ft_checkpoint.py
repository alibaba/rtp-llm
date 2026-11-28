"""Qwen2 MTP's rank-local FT boundary (all fusion projections are replicated)."""

import torch

from rtp_llm.models_py.ft_checkpoint import RuntimeTensorBinding, ft_layer_bindings
from rtp_llm.models_py.new_models.qwen2.ft_checkpoint import load_gqa_ffn_ft_checkpoint


def load_mtp_ft_checkpoint(model, checkpoint):
    h = model.mtp_block.e_norm.weight.numel()
    dtype = model.mtp_block.e_norm.weight.dtype

    def eh_kernel(value):
        # The historical writer retained the source orientation. Accept both
        # supported HF layouts, returning the actual linear's [out, in] shape.
        if value.shape == (2 * h, h):
            value = value.T
        if value.shape != (h, 2 * h) or value.dtype != dtype:
            raise ValueError("Invalid Qwen2 MTP FT fusion projection")
        return value

    def norm(value):
        if value.shape != (h,) or value.dtype != dtype:
            raise ValueError("Invalid Qwen2 MTP FT norm")
        return value

    def zero_beta(value):
        if torch.count_nonzero(norm(value)).item():
            raise ValueError("Qwen2 MTP FT final norm beta must be zero")

    bindings, validators = [], {}
    for layer, _, _ in ft_layer_bindings(model):
        prefix = f"layers.{layer}.multi_tokens_predict_"
        for key, target, convert in (
            ("enorm.weight", "mtp_block.e_norm.weight", norm),
            ("hnorm.weight", "mtp_block.h_norm.weight", norm),
            ("eh_proj.weight", "mtp_block.fc.weight", eh_kernel),
        ):
            if layer == 0:
                bindings.append(RuntimeTensorBinding(
                    (prefix + key,), target, lambda values, convert=convert: convert(values[0])
                ))
            else:
                validators[prefix + key] = convert
        if layer:
            validators[prefix + "final_layernorm.gamma"] = norm
            validators[prefix + "final_layernorm.beta"] = zero_beta

    map_keys = {name: f"global.multi_tokens_predict_{name}_map" for name in ("d2t", "t2d")}
    present = [name for name, key in map_keys.items() if key in checkpoint.entries]
    if present and len(present) != len(map_keys):
        raise ValueError("Qwen2 MTP FT vocabulary maps require both d2t and t2d")
    for name in present:
        key = map_keys[name]
        shape = checkpoint.read_rank_slice(key, checkpoint.topology.tp_rank)
        if len(shape) != 1:
            raise ValueError("Expected a vector for Qwen2 MTP FT vocabulary map " + name)
        model._buffers[name] = model.norm.weight.new_empty(shape, dtype=torch.int64)
        bindings.append(RuntimeTensorBinding((key,), name, lambda values: values[0]))

    load_gqa_ffn_ft_checkpoint(
        model, checkpoint, extra_bindings=bindings, source_validators=validators,
        final_norm_keys=(
            "layers.0.multi_tokens_predict_final_layernorm.gamma",
            "layers.0.multi_tokens_predict_final_layernorm.beta",
        ),
    )
    model._extra_draft_seen.update(model._extra_draft_shapes)
