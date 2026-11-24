"""Main runtime entry for GLM4 non-Lite (GQA), not the MLA model."""

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models_py.model_desc.block_map import get_primary_attention_inputs
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.new_models.model_base import select_fmha_impl_for_layer
from rtp_llm.models_py.new_models.qwen3.language import (
    Qwen3ForCausalLM,
    _extract_config_values,
    _validate_supported_parallelism,
)

from .parameters import install_glm4_parameters


class GqaMoeForCausalLM(Qwen3ForCausalLM):
    """Direct GQA MoE owner; model-specific installers describe checkpoint layout."""

    # Expert/projection updates need explicit owner preflight, not the dense
    # model's blanket checkpoint-update capability.
    supports_checkpoint_weight_updates = False
    supports_ft_handler_heads = True

    def load_ft_checkpoint(self, checkpoint):
        from rtp_llm.models_py.new_models.qwen2.ft_checkpoint import (
            load_gqa_ffn_ft_checkpoint,
        )

        load_gqa_ffn_ft_checkpoint(self, checkpoint)

    def validate_lora_checkpoint_target(self, name):
        from rtp_llm.models_py.lora_checkpoint import (
            validate_attention_lora_checkpoint_target,
        )

        validate_attention_lora_checkpoint_target(name, type(self).__name__)

    def __init__(self, model_config, load_config):
        if not isinstance(model_config, ModelConfig):
            raise TypeError(f"{type(self).__name__} requires a typed ModelConfig")
        parallel = load_config.parallelism_config
        _validate_supported_parallelism(parallel)
        GptModelBase.__init__(
            self,
            config=model_config,
            parallelism_config=parallel,
            weight=None,
            max_generate_batch_size=load_config.max_generate_batch_size,
            fmha_config=load_config.fmha_config,
            py_hw_kernel_config=load_config.py_hw_kernel_config,
        )
        cfg = _extract_config_values(
            model_config, load_config, validate_intermediate_size=False
        )
        # Inherited checkpoint resolution, validation and runtime exports all
        # consult this owner, including when the service requests no custom head.
        self._init_handler_heads(load_config, cfg)
        cfg.update(
            num_experts=model_config.expert_num,
            top_k=model_config.moe_k,
            moe_intermediate_size=model_config.moe_inter_size,
            model_config=model_config,
            parallelism_config=parallel,
            moe_config=load_config.moe_config,
            ep_size=load_config.ep_size,
            ep_rank=load_config.ep_rank,
        )
        self._install_parameters(
            self,
            cfg,
            layer_ids=self.layer_ids(),
            partitioned=getattr(parallel, "pp_size", 1) > 1,
        )
        self._init_capture_context(
            self._capture_canonical_layer, self._capture_canonical_final
        )

    def checkpoint_weight_name_filter(self):
        stage_owns = super().checkpoint_weight_name_filter()
        if stage_owns is None:
            return self._layout.target_owns
        return lambda name: self._layout.target_owns(name) and stage_owns(name)

    def _capture_canonical_layer(self, hidden_states, residual):
        return hidden_states + residual

    def _capture_canonical_final(self, hidden_states, residual):
        return self.norm(hidden_states, residual)[0]

    def forward(self, inputs, fmha_impl=None):
        hidden = (
            self._input_embeddings(inputs)
            if self.embed_tokens is not None
            else inputs.pp_intermediates["hidden_states"]
        )
        residual = torch.zeros_like(hidden)
        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)
        capture = self.capture_context(inputs.capture_hidden_states)
        projector = getattr(self, "_eagle3_capture_projection", None)
        if projector is not None:
            self.begin_aux_hidden_capture(
                hidden, bool(get_primary_attention_inputs(inputs).is_target_verify)
            )
        layers = self.layers.values() if self._pipeline_partitioned else self.layers
        for local, (global_id, layer) in enumerate(zip(self._owned_layer_ids, layers)):
            hidden, residual = layer(
                hidden,
                residual,
                select_fmha_impl_for_layer(fmha_impl, self.kv_cache, local),
                (
                    self.kv_cache.get_layer_cache(local)
                    if self.kv_cache is not None
                    else None
                ),
            )
            capture.capture_layer(global_id, hidden, residual)
            if projector is not None:
                self.capture_aux_hidden(global_id, hidden, residual)
        if self.norm is None:
            materialized = hidden + residual
            from rtp_llm.ops.compute_ops import PyModelOutputs

            out = PyModelOutputs(materialized)
            out.pp_intermediates = {"hidden_states": materialized}
            return out
        out = capture.finalize(hidden, residual)
        if projector is not None:
            self.finish_aux_hidden_capture()
            # Project the initial three H-wide features *once*, on the target;
            # subsequent Eagle3 recurrent steps already have width H. This
            # keeps graph input shape and PD draft feature storage fixed at H.
            project = projector()
            if project is None:
                raise RuntimeError("EAGLE3 draft was released before its target")
            self._mtp_target_hidden_states = project(self._mtp_target_hidden_states)
            out.mtp_target_hidden_states = self._mtp_target_hidden_states
        return out


class Glm4MoeForCausalLM(GqaMoeForCausalLM):
    _install_parameters = staticmethod(install_glm4_parameters)
