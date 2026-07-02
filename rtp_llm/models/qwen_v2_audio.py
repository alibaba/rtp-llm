from typing import Any, Sequence

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.models.qwen_v2 import QWenV2, QWenV2Weight
from rtp_llm.models_py.model_desc.block_map import select_block_map_for_layer
from rtp_llm.models_py.model_desc.qwen3 import Qwen3Model
from rtp_llm.models_py.modules import MultimodalEmbeddingInjector
from rtp_llm.ops.compute_ops import PyModelInputs, PyModelOutputs
from rtp_llm.utils.util import get_config_from_path


class _Qwen2AudioModel(Qwen3Model):
    """Qwen3Model subclass that injects audio features into text embeddings.

    Avoids monkey-patching ``embed_tokens`` in the hot forward path. Instead,
    it computes ``inputs_embeds`` once, applies the multimodal injector when
    audio features are present, and then runs the standard decoder stack.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._injector = MultimodalEmbeddingInjector()

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        input_ids: torch.Tensor = inputs.input_ids
        inputs_embeds = self.embed_tokens(input_ids)

        mm_features: Sequence[torch.Tensor] = inputs.multimodal_inputs.multimodal_features
        mm_feature_locs = inputs.multimodal_inputs.mm_features_locs
        if mm_features and mm_feature_locs is not None:
            inputs_embeds = self._injector(inputs_embeds, mm_features, mm_feature_locs)

        hidden_states = inputs_embeds
        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)
        for i, decoder_layer in enumerate(self.layers[: self.layer_num]):
            select_block_map_for_layer(inputs.attention_inputs, i)
            hidden_states = decoder_layer(
                hidden_states,
                fmha_impl,
                kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
            )
        hidden_states = self.norm(hidden_states)
        return PyModelOutputs(hidden_states, fmha_impl.fmha_params)


class QWenV2Audio(QWenV2):
    @classmethod
    def _create_config(cls, ckpt_path: str):
        # Build the base ModelConfig with Qwen2 defaults directly, then call
        # QWenV2Audio._from_hf. Do NOT route through QWenV2._create_config,
        # because that calls QWenV2._from_hf on the top-level audio config.json,
        # which has audio_token_index at the root and text_config for the LLM.
        config = ModelConfig()
        config.ckpt_path = ckpt_path
        config.vocab_size = 152064
        config.max_seq_len = 8192
        config.attn_config.rope_config.dim = 128
        config.attn_config.rope_config.style = 1
        config.has_pre_decoder_layernorm = False
        config.special_tokens.bos_token_id = -1
        config.special_tokens.eos_token_id = 151643
        config.special_tokens.stop_words_id_list = [[151645], [151644]]

        QWenV2Audio._from_hf(config, ckpt_path)
        assert config.attn_config.head_num > 0
        config.mm_model_config.is_multimodal = True
        return config

    @staticmethod
    def get_weight_cls():
        return QWenV2Weight

    def _create_python_model(self):
        model_config = self.model_config
        parallelism_config = self.parallelism_config
        fmha_config = self.fmha_config
        py_hw_kernel_config = self.hw_kernel_config
        quant_config = self.model_config.quant_config

        self.py_model = _Qwen2AudioModel(
            model_config,
            parallelism_config,
            self.weight,
            max_generate_batch_size=self.max_generate_batch_size,
            quant_config=quant_config,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=self.device_resource_config,
        )

    @classmethod
    def _from_hf(cls, config: ModelConfig, ckpt_path: str):
        config_json = get_config_from_path(ckpt_path)
        if not config_json:
            raise Exception(f"failed to get config.json from path: {ckpt_path}")
        sep_token = config_json["audio_token_index"]
        config_json = config_json["text_config"]

        config.inter_size = config_json.get("intermediate_size", 11008)
        config.attn_config.head_num = config_json.get("num_attention_heads", 32)
        config.attn_config.kv_head_num = config_json.get(
            "num_key_value_heads", config.attn_config.head_num
        )
        config.attn_config.size_per_head = (
            config_json.get("hidden_size", 4096) // config.attn_config.head_num
        )
        config.num_layers = config_json.get("num_hidden_layers", 32)
        config.attn_config.rope_config.base = int(
            config_json.get("rope_theta", config.attn_config.rope_config.base)
        )
        config.vocab_size = config_json["vocab_size"]
        config.attn_config.rope_config.dim = config.attn_config.size_per_head
        config.layernorm_eps = config_json.get("rms_norm_eps", 1e-06)
        config.tie_word_embeddings = config_json.get("tie_word_embeddings", False)

        config.mm_model_config.mm_sep_tokens = [[sep_token]]
        config.config_dtype = config_json.get("torch_dtype", None)

        config.mm_related_params.config["ckpt_path"] = config.ckpt_path


register_model("qwen_v2_audio", QWenV2Audio)
