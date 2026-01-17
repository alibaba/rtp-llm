import functools
from typing import Any, Dict

import torch

from rtp_llm.config.model_config import ModelConfig, VitParameters
from rtp_llm.model_factory_register import register_model
from rtp_llm.model_loader.attn_weight import AttnAtomicWeight
from rtp_llm.model_loader.ffn_weight import FfnAtomicWeight
from rtp_llm.model_loader.model_weight_info import (
    ModelDeployWeightInfo,
    ModelWeightInfo,
)
from rtp_llm.model_loader.weight_module import AtomicWeight
from rtp_llm.models.base_model import BaseModel
from rtp_llm.utils.model_weight import (
    CkptWeightInfo,
    W,
    identity,
    slopes,
    trans_qkv,
    trans_qkv_b,
    transpose,
)
from rtp_llm.utils.util import get_config_from_path


class BloomWeightInfo(ModelDeployWeightInfo):
    def _process_meta(self, meta_dicts, weight_keys):
        if "lm_head.weight" in weight_keys:
            self._lm_head = True
        else:
            self._lm_head = False

        if "transformer.h.0.input_layernorm.weight" in weight_keys:
            self._transformer_prefix = True
        else:
            self._transformer_prefix = False

    def _get_weight_info(self):
        weights = [
            AtomicWeight(
                W.embedding,
                [CkptWeightInfo("word_embeddings.weight", identity)],
                identity,
            ),
            AtomicWeight(
                W.pre_decoder_ln_gamma,
                [CkptWeightInfo("word_embeddings_layernorm.weight", identity)],
                identity,
            ),
            AtomicWeight(
                W.pre_decoder_ln_beta,
                [CkptWeightInfo("word_embeddings_layernorm.bias", identity)],
                identity,
            ),
            AtomicWeight(
                W.final_ln_gamma, [CkptWeightInfo("ln_f.weight", identity)], identity
            ),
            AtomicWeight(
                W.final_ln_beta, [CkptWeightInfo("ln_f.bias", identity)], identity
            ),
        ]
        if self.model_config.use_attention_linear_bias:
            weights.append(
                AtomicWeight(
                    W.linear_bias_slopes,
                    [],
                    functools.partial(slopes, n=self.model_config.attn_config.head_num),
                    data_type=torch.float,
                )
            )
        attn_config = self.attn_config
        ffn_config = self.ffn_config

        layer_weights = []
        for _ in range(self._num_layers):
            layer_weight = [
                AtomicWeight(
                    W.pre_ln_beta,
                    [CkptWeightInfo("h.{i}.input_layernorm.bias", identity)],
                    identity,
                ),
                AtomicWeight(
                    W.pre_ln_gamma,
                    [CkptWeightInfo("h.{i}.input_layernorm.weight", identity)],
                    identity,
                ),
                AttnAtomicWeight(
                    W.attn_qkv_w,
                    [
                        CkptWeightInfo(
                            "h.{i}.self_attention.query_key_value.weight", identity
                        )
                    ],
                    functools.partial(
                        trans_qkv,
                        hidden_size=self._hidden_size,
                        head_num=self._head_num,
                    ),
                    config=attn_config,
                ),
                AttnAtomicWeight(
                    W.attn_qkv_b,
                    [
                        CkptWeightInfo(
                            "h.{i}.self_attention.query_key_value.bias", identity
                        )
                    ],
                    functools.partial(
                        trans_qkv_b,
                        hidden_size=self._hidden_size,
                        head_num=self._head_num,
                    ),
                    config=attn_config,
                ),
                AttnAtomicWeight(
                    W.attn_o_w,
                    [CkptWeightInfo("h.{i}.self_attention.dense.weight", identity)],
                    transpose,
                    config=attn_config,
                ),
                AttnAtomicWeight(
                    W.attn_o_b,
                    [CkptWeightInfo("h.{i}.self_attention.dense.bias", identity)],
                    identity,
                    config=attn_config,
                ),
                FfnAtomicWeight(
                    W.ffn_w3,
                    [CkptWeightInfo("h.{i}.mlp.dense_h_to_4h.weight", identity)],
                    transpose,
                    config=ffn_config,
                ),
                FfnAtomicWeight(
                    W.ffn_b3,
                    [CkptWeightInfo("h.{i}.mlp.dense_h_to_4h.bias", identity)],
                    identity,
                    config=ffn_config,
                ),
                FfnAtomicWeight(
                    W.ffn_w2,
                    [CkptWeightInfo("h.{i}.mlp.dense_4h_to_h.weight", identity)],
                    transpose,
                    config=ffn_config,
                ),
                FfnAtomicWeight(
                    W.ffn_b2,
                    [CkptWeightInfo("h.{i}.mlp.dense_4h_to_h.bias", identity)],
                    identity,
                    config=ffn_config,
                ),
                AtomicWeight(
                    W.post_ln_beta,
                    [CkptWeightInfo("h.{i}.post_attention_layernorm.bias", identity)],
                    identity,
                ),
                AtomicWeight(
                    W.post_ln_gamma,
                    [CkptWeightInfo("h.{i}.post_attention_layernorm.weight", identity)],
                    identity,
                ),
            ]
            layer_weights.append(layer_weight)

        if self._transformer_prefix:
            for w in layer_weights:
                w.weights[0].name = "transformer." + w.weights[0].name
            for w in weights:
                w.weights[0].name = "transformer." + w.weights[0].name

        if self._lm_head:
            weights.append(
                AtomicWeight(
                    W.lm_head, [CkptWeightInfo("lm_head.weight", identity)], identity
                )
            )

        return ModelWeightInfo(weights, layer_weights)


class Bloom(BaseModel):
    @staticmethod
    def get_weight_cls():
        return BloomWeightInfo

    @staticmethod
    def from_huggingface(config_json: Dict[str, Any]):
        model_type = config_json["model_type"]
        config = ModelConfig()
        config.attn_config.head_num = 32
        config.attn_config.size_per_head = 128
        config.num_layers = 30
        config.max_seq_len = 2048
        config.vocab_size = 250682
        if model_type != "bloom":
            raise BaseException(f"model type is not bloom: {model_type}")
        config.attn_config.head_num = config_json.get(
            "num_attention_heads", config_json.get("n_head")
        )
        config.attn_config.kv_head_num = config.attn_config.head_num
        config.hidden_size = config_json.get("n_embed", config_json.get("hidden_size"))
        config.attn_config.size_per_head = (
            config.hidden_size // config.attn_config.head_num
        )
        config.num_layers = config_json["n_layer"]
        config.max_seq_len = config_json.get("seq_length", 2048)
        config.vocab_size = config_json["vocab_size"]
        config.layernorm_eps = config_json["layer_norm_epsilon"]
        config.inter_size = config.hidden_size * 4
        config.special_tokens.eos_token_id = config_json.get("eos_token_id", 0)
        config.tie_word_embeddings = config_json.get("tie_word_embeddings", False)
        config.config_dtype = config_json.get("torch_dtype", None)
        return config

    @classmethod
    def _create_config(cls, ckpt_path: str):
        from rtp_llm.model_config_creators.bloom import create_bloom_config

        config = create_bloom_config(ckpt_path)
        return config


register_model("bloom", Bloom, ["BloomForCausalLM"])
