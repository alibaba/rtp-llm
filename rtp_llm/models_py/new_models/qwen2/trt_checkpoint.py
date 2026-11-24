"""Qwen2's TRT-style checkpoint boundary, independent of legacy W.* names.

This format is a tensor checkpoint, not a serialized TensorRT engine. The
historical Qwen2 loader leaves fused QKV in [input, Q+K+V] runtime orientation,
but transposes the other checkpoint projections. Normalize only fused QKV to
the module's [output, input] checkpoint contract before normal TP dispatch.
"""

import re

import torch

from rtp_llm.models_py.read_spec import unwrap_checkpoint_tensor
from rtp_llm.models_py.weight_mapper import WeightsMapper


class Qwen2CheckpointMapper(WeightsMapper):
    _GLOBALS = {
        "transformer.vocab_embedding.weight": "embed_tokens.weight",
        "transformer.ln_f.weight": "norm.weight",
    }
    _LAYER = re.compile(r"^transformer\.layers\.(\d+)\.(.+)$")
    _PROJECTIONS = {
        "attention.qkv": "self_attn.qkv_proj",
        "attention.dense": "self_attn.o_proj",
        # These names intentionally follow Qwen2's historical gated TRT map:
        # fc is the nonlinear gate; gate is the multiplicative up projection.
        "mlp.fc": "mlp.gate_proj",
        "mlp.gate": "mlp.up_proj",
        "mlp.proj": "mlp.down_proj",
    }

    def __init__(self, *, exact_mapping=None):
        super().__init__(exact_mapping=exact_mapping, prefix_mapping={"model.": ""})

    def map_name(self, name):
        if name in self._GLOBALS:
            return self._GLOBALS[name]
        match = self._LAYER.fullmatch(name)
        if match is None:
            return super().map_name(name)
        layer, suffix = match.groups()
        if suffix == "input_layernorm.weight":
            return f"layers.{layer}.input_layernorm.weight"
        if suffix == "post_layernorm.weight":
            return f"layers.{layer}.post_attention_layernorm.weight"
        projection, separator, parameter = suffix.rpartition(".")
        if (
            separator
            and projection in self._PROJECTIONS
            and parameter
            in (
                "weight",
                "bias",
            )
        ):
            return f"layers.{layer}.{self._PROJECTIONS[projection]}.{parameter}"
        # Do not silently discard unadapted quant scales or unknown payloads.
        return name

    def apply(self, weights):
        for name, checkpoint_value in weights:
            match = self._LAYER.fullmatch(name)
            if match is not None and match[2] == "attention.qkv.weight":
                tensor, spec = unwrap_checkpoint_tensor(checkpoint_value)
                if spec is not None:
                    raise ValueError(
                        "TRT fused QKV must be normalized before TP sharding"
                    )
                if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
                    raise ValueError("TRT fused QKV weight must be a rank-2 tensor")
                checkpoint_value = tensor.t().contiguous()
            yield self.map_name(name), checkpoint_value
