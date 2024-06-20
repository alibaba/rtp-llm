

import torch
import unicodedata
import types
import functools
import os
import json
from typing import List, Any

from maga_transformer.utils.model_weight import (W, WeightInfo, ModelWeightInfo, ModelDeployWeightInfo,
                                                 CkptWeightInfo, identity, zeros, transpose, transpose_pad,
                                                 merge_qkv_b, merge_qkv_hf)
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.models.qwen import QWen
from maga_transformer.models.gpt import GPT
from transformers import AutoTokenizer
from maga_transformer.model_factory_register import register_model
from maga_transformer.utils.group_quant_weight_util import get_layer_group_quant_weight_info

class QWenV2Weight(ModelDeployWeightInfo):
    def _get_weight_info(self):
        return self._get_hf_weight_info()

    def _get_hf_ffn_layer_weight_info(self, layer_id: int):
        inter_padding_size = self._layer_inter_padding_size[layer_id] if self._layer_inter_padding_size else self._inter_padding_size
        return [WeightInfo(W.ffn_w1, [CkptWeightInfo('model.layers.{i}.mlp.gate_proj.weight', identity)],
            functools.partial(transpose_pad, inter_padding_size=inter_padding_size, dim=0)),
        WeightInfo(W.ffn_w3, [CkptWeightInfo('model.layers.{i}.mlp.up_proj.weight', identity)],
            functools.partial(transpose_pad, inter_padding_size=inter_padding_size, dim=0)),
        WeightInfo(W.ffn_w2, [CkptWeightInfo('model.layers.{i}.mlp.down_proj.weight', identity)],
            functools.partial(transpose_pad, inter_padding_size=inter_padding_size, dim=1))]

    def _get_hf_layer_weight_info(self, layer_id: int):
        layer_weights = [
            WeightInfo(W.pre_ln_gamma, [CkptWeightInfo('model.layers.{i}.input_layernorm.weight', identity)],
                       identity),
            WeightInfo(W.attn_qkv_b, [
                    CkptWeightInfo('model.layers.{i}.self_attn.q_proj.bias', identity),
                    CkptWeightInfo('model.layers.{i}.self_attn.k_proj.bias', identity),
                    CkptWeightInfo('model.layers.{i}.self_attn.v_proj.bias', identity)
                ],
                functools.partial(merge_qkv_b)),
            WeightInfo(W.attn_qkv_w, [
                    CkptWeightInfo('model.layers.{i}.self_attn.q_proj.weight', identity),
                    CkptWeightInfo('model.layers.{i}.self_attn.k_proj.weight', identity),
                    CkptWeightInfo('model.layers.{i}.self_attn.v_proj.weight', identity)
                ],
                functools.partial(merge_qkv_hf)),
            WeightInfo(W.attn_o_w, [CkptWeightInfo('model.layers.{i}.self_attn.o_proj.weight', identity)],
                       transpose),
            WeightInfo(W.post_ln_gamma, [CkptWeightInfo('model.layers.{i}.post_attention_layernorm.weight', identity)],
                       identity),
        ]
        layer_weights.extend(self._get_hf_ffn_layer_weight_info(layer_id))
        return layer_weights

    def _get_hf_weight_info(self):
        weights = [
            WeightInfo(W.embedding, [CkptWeightInfo('model.embed_tokens.weight', identity)], identity),
            WeightInfo(W.lm_head, [CkptWeightInfo('lm_head.weight', identity)], identity),
            WeightInfo(W.final_ln_gamma, [CkptWeightInfo('model.norm.weight', identity)], identity),
            WeightInfo(W.final_ln_beta, [], functools.partial(zeros, shape=[self._hidden_size])),
        ]

        layer_weights: List[List[WeightInfo]] = []
        for layer in range(self._num_layers):
            if self._quant_algo.isGroupwise():
                inter_padding_size = self._layer_inter_padding_size[layer] if self._layer_inter_padding_size else self._inter_padding_size
                w = self._get_hf_layer_weight_info(layer)
                w = get_layer_group_quant_weight_info(w, self._quant_algo, inter_padding_size)
                layer_weights.append(w)
            else:
                w = self._get_hf_layer_weight_info(layer)
                layer_weights.append(w)

        return ModelWeightInfo(layer_weights=layer_weights, weights=weights, tp_strategy=self._get_gpt_style_tp_strategy())

class QWenV2(QWen):
    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = GptInitModelParameters(
            head_num=0,
            head_num_kv=0,
            size_per_head=0,
            layer_num=0,
            inter_size=0, # 13696
            vocab_size=152064,
            max_seq_len=8192)
        config.rotary_embedding_dim = 128
        config.rotary_embedding_style = 1
        config.activation_type = 'SiGLU'
        config.has_pre_decoder_layernorm = False
        config.has_post_decoder_layernorm = True
        config.norm_type = 'rmsnorm'
        config.special_tokens.bos_token_id = -1
        config.special_tokens.eos_token_id = 151643
        # <|im_start|> and <|im_end|>
        config.special_tokens.stop_words_list = [[151645], [151644]]
        config.special_tokens.system.token_ids = [151644, 8948, 198] # '<|im_start|>system\n'
        config.special_tokens.system.eos_token_ids = [151645, 198] # '<|im_end|>\n'
        config.special_tokens.user.token_ids = [151644, 872, 198] # '<|im_start|>user\n'
        config.special_tokens.user.eos_token_ids = [151645, 198]  # '<|im_end|>\n'
        config.special_tokens.assistant.token_ids = [151644, 77091, 198] # '<|im_start|>assistant\n'
        config.special_tokens.assistant.eos_token_ids = [151645, 198] # '<|im_end|>\n'

        QWenV2._from_hf(config, ckpt_path)
        assert config.head_num > 0 and config.head_num_kv > 0 and config.size_per_head > 0 and config.layer_num > 0 and config.inter_size > 0, "error config"
        return config

    @staticmethod
    def _from_hf(config: GptInitModelParameters, ckpt_path: str):
        config_path = os.path.join(ckpt_path, "config.json")
        if not os.path.exists(config_path):
            return
        with open(config_path) as reader:
            content = reader.read()
            config_json = json.loads(content)

        # config.activation_type = config_json["hidden_act"]
        config.inter_size = config_json["intermediate_size"]
        config.head_num = config_json["num_attention_heads"]
        config.head_num_kv = config_json.get("num_key_value_heads", config.head_num)
        config.size_per_head = config_json["hidden_size"] // config.head_num
        config.layer_num = config_json["num_hidden_layers"]
        config.rotary_embedding_base = config_json.get("rope_theta", config.rotary_embedding_base)
        config.vocab_size = config_json["vocab_size"]
        config.rotary_embedding_dim = config.size_per_head
        config.layernorm_eps = config_json.get("rms_norm_eps", 1e-06)
        config.tie_word_embeddings = config_json.get('tie_word_embeddings', False)

        GPT._load_quant_config(ckpt_path, config_json, config)

    @staticmethod
    def get_weight_cls():
        return QWenV2Weight

    @classmethod
    def get_tokenizer(cls, config: GptInitModelParameters):
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path, verbose=False, trust_remote_code=True, use_fast=False)
        tokenizer.im_start_id = tokenizer.encode('<|im_start|>')[0]
        tokenizer.im_end_id = tokenizer.encode('<|im_end|>')[0]
        return tokenizer

class QWenV2Embedding(QWenV2):
    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = QWenV2._create_config(ckpt_path)
        config.is_causal = False
        return config


register_model('qwen_2', QWenV2, ["Qwen2ForCausalLM"])
register_model('qwen_2_embedding', QWenV2Embedding)
