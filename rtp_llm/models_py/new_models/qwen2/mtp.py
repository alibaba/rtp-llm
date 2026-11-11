"""Qwen2 EAGLE draft runtime (the legacy ``qwen_2-mtp`` checkpoint format).

This is not EAGLE-3: the draft consumes one target hidden state per token,
concatenates hidden then embedding, and runs the ordinary Qwen2 decoder.
"""

import torch
from torch import nn

from rtp_llm.models_py.layers.linear import ColumnParallelLinear
from rtp_llm.models_py.layers.norm import RMSNorm
from rtp_llm.models_py.new_models.qwen2.language import (
    Qwen2ForCausalLM,
    _extract_config_values,
)


class Qwen2MTPForCausalLM(Qwen2ForCausalLM):
    def __init__(self, model_config, load_config):
        is_mtp = (
            model_config.get("is_mtp", False)
            if isinstance(model_config, dict)
            else model_config.is_mtp
        )
        if is_mtp is not True:
            raise ValueError("Qwen2 EAGLE draft requires is_mtp=True")
        super().__init__(model_config, load_config)
        cfg = _extract_config_values(model_config, load_config)
        hidden = cfg["hidden_size"]
        for layer in self.layers:
            layer.e_norm = RMSNorm(
                hidden, eps=cfg["rms_norm_eps"], params_dtype=cfg["params_dtype"]
            )
            layer.h_norm = RMSNorm(
                hidden, eps=cfg["rms_norm_eps"], params_dtype=cfg["params_dtype"]
            )
            layer.eh_proj = ColumnParallelLinear(
                input_size=2 * hidden,
                output_size=hidden,
                tp_size=1,
                tp_rank=0,
                bias=False,
                quant_config=None,
                params_dtype=cfg["params_dtype"],
            )
            layer.final_head = nn.Module()
            layer.final_head.norm = RMSNorm(
                hidden, eps=cfg["rms_norm_eps"], params_dtype=cfg["params_dtype"]
            )
        # Legacy Qwen2MtpModel uses layer zero's auxiliary weights, even for a
        # multi-layer draft. Retain/validate all checkpoint tensors, but preserve
        # that execution contract and alias the actual final norm.
        self.norm = self.layers[0].final_head.norm
        self.register_buffer("d2t", None)
        self.register_buffer("t2d", None)

    def load_weights(self, weights):
        entries = weights.items() if isinstance(weights, dict) else weights
        seen_maps = set()

        def converted():
            for name, tensor in entries:
                if name == "model.norm.weight":
                    # Exported drafts can retain the base model's final norm.
                    # Legacy Qwen2MtpModel uses layers.0.final_head.norm instead;
                    # this unused base-model tensor must not overwrite it.
                    continue
                if name in ("model.d2t", "model.t2d"):
                    field = name.removeprefix("model.")
                    if field in seen_maps:
                        raise RuntimeError(f"Duplicate EAGLE vocabulary map: {name}")
                    seen_maps.add(field)
                    if tensor.ndim != 1 or tensor.dtype not in (
                        torch.bool,
                        torch.int32,
                        torch.int64,
                    ):
                        raise ValueError(
                            f"{name} must be a one-dimensional integer map"
                        )
                    tensor = tensor.to(dtype=torch.int64)
                    # Allocate the optional persistent buffer before the common
                    # dispatcher so completeness tracking and device migration
                    # apply to vocabulary maps just like other weights.
                    self.register_buffer(field, torch.empty_like(tensor))
                elif name.endswith(".eh_proj.weight"):
                    hidden = self.embed_tokens.weight.shape[1]
                    if tensor.shape != (2 * hidden, hidden):
                        raise ValueError(
                            f"{name} must have legacy [2H,H] layout, got {tuple(tensor.shape)}"
                        )
                    tensor = tensor.t().contiguous()
                yield name, tensor

        super().load_weights(converted())
        if seen_maps and seen_maps != {"d2t", "t2d"}:
            raise ValueError("Qwen2 EAGLE requires both model.d2t and model.t2d")

    def checkpoint_weight_name_filter(self):
        return lambda name: name != "model.norm.weight"

    def word_embedding(self, inputs):
        embedding = self.embed_tokens(inputs.input_ids)
        hidden = inputs.input_hiddens
        if hidden is None or hidden.shape != embedding.shape:
            raise ValueError("Qwen2 EAGLE input_hiddens must match token embeddings")
        auxiliary = self.layers[0]
        return auxiliary.eh_proj(
            torch.cat((auxiliary.h_norm(hidden), auxiliary.e_norm(embedding)), dim=-1)
        )

    def runtime_weight_view(self):
        view = super().runtime_weight_view()
        if self.d2t is not None:
            view["multi_tokens_predict_d2t_map"] = self.d2t
            view["multi_tokens_predict_t2d_map"] = self.t2d
        return view
