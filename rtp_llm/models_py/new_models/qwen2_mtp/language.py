"""Qwen2 MTP (Multi-Token Prediction) draft head for new-loader.

Single-layer model that combines the main model's last hidden states with new
token embeddings via an MTPBlock projection, then runs one decoder layer and a
final norm to produce the draft logits.

HF ckpt structure (after WEIGHTS_MAPPER strips "model."):
  embed_tokens.weight                  -> embed_tokens.weight
  lm_head.weight                       -> lm_head.weight
  layers.0.e_norm.weight               -> mtp_block.e_norm.weight
  layers.0.h_norm.weight               -> mtp_block.h_norm.weight
  layers.0.eh_proj.weight              -> mtp_block.fc.weight  (transposed)
  layers.0.final_head.norm.weight      -> norm.weight
  layers.0.self_attn.*                 -> layers.0.self_attn.*
  layers.0.mlp.*                       -> layers.0.mlp.*
  layers.0.input_layernorm.*           -> layers.0.input_layernorm.*
  layers.0.post_attention_layernorm.*  -> layers.0.post_attention_layernorm.*
"""

from typing import Any

import torch.nn as nn

from rtp_llm.models_py.layers.embedding import ParallelLMHead, VocabParallelEmbedding
from rtp_llm.models_py.layers.norm import RMSNorm
from rtp_llm.models_py.model_desc.block_map import select_block_map_for_layer
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.new_models.mtp import MTPBlock
from rtp_llm.models_py.new_models.qwen2_vl.language import (
    Qwen2DecoderLayer,
    _extract_config_values,
)
from rtp_llm.models_py.weight_mapper import WeightsMapper
from rtp_llm.ops.compute_ops import PyModelInputs, PyModelOutputs


class Qwen2MTPForCausalLM(GptModelBase):
    """MTP draft head for Qwen2 (prefix model.).

    Single-layer decoder with MTPBlock projection.  Uses reverse_concat=True
    (cat [h_norm, e_norm], dim=-1) matching the legacy QwenV2MTP loader.
    """

    WEIGHTS_MAPPER = WeightsMapper(prefix_mapping={"model.": ""})

    def load_weights(self, weights):
        import logging

        if isinstance(weights, dict):
            weights_iter = iter(weights.items())
        else:
            weights_iter = weights
        has_lm_head = False

        def _transform(it):
            nonlocal has_lm_head
            for name, tensor in it:
                if name == "lm_head.weight" or name.startswith("lm_head."):
                    has_lm_head = True
                # Remap MTP-specific weights from layers.{i}.* to target modules.
                # The HF ckpt stores MTP weights under model.layers.{i}.* ; we
                # intercept them here (before WEIGHTS_MAPPER strips "model.")
                # and rewrite the key so it lands on the correct submodule.
                if ".e_norm." in name:
                    yield "mtp_block.e_norm." + name.split(".e_norm.", 1)[1], tensor
                    continue
                if ".h_norm." in name:
                    yield "mtp_block.h_norm." + name.split(".h_norm.", 1)[1], tensor
                    continue
                if ".eh_proj." in name:
                    mapped_name = "mtp_block.fc." + name.split(".eh_proj.", 1)[1]
                    if mapped_name == "mtp_block.fc.weight":
                        tensor = tensor.t().contiguous()
                    yield mapped_name, tensor
                    continue
                if ".final_head.norm." in name:
                    yield "norm." + name.split(".final_head.norm.", 1)[1], tensor
                    continue
                yield name, tensor

        mapped_iter = self.WEIGHTS_MAPPER.apply(_transform(weights_iter))
        super().load_weights(mapped_iter)

        # Handle tied embeddings: when ckpt has no lm_head.weight (e.g. Qwen2.5
        # with tie_word_embeddings=true), reuse embed_tokens.weight for lm_head.
        if not has_lm_head:
            logging.info(
                "[Qwen2MTPForCausalLM] lm_head.weight not found in ckpt; "
                "tying lm_head to embed_tokens (tie_word_embeddings)"
            )
            self.lm_head.weight.data.copy_(self.embed_tokens.weight.data)

    def __init__(self, model_config: Any, load_config: Any):
        parallelism_config = getattr(load_config, "parallelism_config", None)
        fmha_config = getattr(load_config, "fmha_config", None)
        device_resource_config = getattr(load_config, "device_resource_config", None)

        super().__init__(
            config=model_config,
            parallelism_config=parallelism_config,
            weight=None,
            max_generate_batch_size=0,
            fmha_config=fmha_config,
            device_resource_config=device_resource_config,
        )

        cfg = _extract_config_values(model_config, load_config)

        # MTP: single decoder layer
        cfg["num_layers"] = 1
        if hasattr(model_config, "is_mtp"):
            model_config.is_mtp = True

        self.embed_tokens = VocabParallelEmbedding(
            vocab_size=cfg["vocab_size"],
            embedding_dim=cfg["hidden_size"],
            tp_size=cfg["tp_size"],
            tp_rank=cfg["tp_rank"],
            params_dtype=cfg["params_dtype"],
        )

        # MTP projection block: reverse_concat=True matches legacy QwenV2MTP
        # (cat [h_norm, e_norm], dim=-1)
        self.mtp_block = MTPBlock(
            hidden_size=cfg["hidden_size"],
            rms_norm_eps=1e-6,
            reverse_concat=True,
            bias=False,
            params_dtype=cfg["params_dtype"],
        )

        self.layers = nn.ModuleList(
            [
                Qwen2DecoderLayer(
                    hidden_size=cfg["hidden_size"],
                    num_heads=cfg["num_heads"],
                    num_kv_heads=cfg["num_kv_heads"],
                    intermediate_size=cfg["intermediate_size"],
                    head_dim=cfg["head_dim"],
                    layer_idx=0,
                    tp_size=cfg["tp_size"],
                    tp_rank=cfg["tp_rank"],
                    quant_config=cfg["quant_config"],
                    params_dtype=cfg["params_dtype"],
                )
            ]
        )

        # Final norm (from layers.0.final_head.norm.weight)
        self.norm = RMSNorm(cfg["hidden_size"], params_dtype=cfg["params_dtype"])

        self.lm_head = ParallelLMHead(
            vocab_size=cfg["vocab_size"],
            hidden_size=cfg["hidden_size"],
            tp_size=cfg["tp_size"],
            tp_rank=cfg["tp_rank"],
            params_dtype=cfg["params_dtype"],
        )

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        input_ids = inputs.input_ids
        inputs_embeds = self.embed_tokens(input_ids)
        # MTP block: combine embed with last hidden states from the main model
        hidden_states = self.mtp_block(inputs_embeds, inputs.input_hiddens)

        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)
        for i, layer in enumerate(self.layers):
            select_block_map_for_layer(inputs.attention_inputs, i)
            hidden_states = layer(
                hidden_states,
                fmha_impl,
                kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
            )
        hidden_states = self.norm(hidden_states)
        return PyModelOutputs(hidden_states, fmha_impl.fmha_params)
