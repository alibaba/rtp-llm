
import os
import json
import functools

from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.utils.model_weight import W, WeightInfo, ModelWeightInfo, ModelDeployWeightInfo, CkptWeightInfo, identity, zeros, ones, transpose, qkv_gather
from maga_transformer.models.gpt import GPT
from maga_transformer.model_factory_register import register_model

class FalconWeightInfo(ModelDeployWeightInfo):
    def _process_meta(self, meta_dicts, weight_keys):
        if 'transformer.h.0.ln_attn.weight' in weight_keys:
            self.falcon_40b = True
        elif 'transformer.h.0.input_layernorm.weight' in weight_keys:
            self.falcon_40b = False

    def _get_weight_info(self):
        weights = [
            WeightInfo(W.embedding, [CkptWeightInfo('transformer.word_embeddings.weight', identity)], identity),
            WeightInfo(W.lm_head, [CkptWeightInfo('lm_head.weight', identity)], identity),
            WeightInfo(W.final_ln_gamma, [CkptWeightInfo('transformer.ln_f.weight', identity)], identity),
            WeightInfo(W.final_ln_beta, [CkptWeightInfo('transformer.ln_f.bias', identity)], identity),
        ]

        layer_weights = [

            WeightInfo(W.attn_qkv_b, [], functools.partial(zeros, shape=self._hidden_size * 3)),

            WeightInfo(W.attn_o_w, [CkptWeightInfo('transformer.h.{i}.self_attention.dense.weight', identity)], transpose),

            WeightInfo(W.attn_o_b, [], functools.partial(zeros, shape=self._hidden_size)),

            WeightInfo(W.ffn_w1, [CkptWeightInfo('transformer.h.{i}.mlp.dense_h_to_4h.weight', identity)], transpose),

            WeightInfo(W.ffn_w2, [CkptWeightInfo('transformer.h.{i}.mlp.dense_4h_to_h.weight', identity)], transpose),
        ]

        if self.falcon_40b:
            layer_weights.extend([
                WeightInfo(W.attn_qkv_w, [CkptWeightInfo('transformer.h.{i}.self_attention.query_key_value.weight', identity)],
                           functools.partial(qkv_gather, dim0=self._hidden_size, head_num=self._head_num, head_num_kv=self._head_num_kv)),
                WeightInfo(W.pre_ln_beta, [CkptWeightInfo('transformer.h.{i}.ln_mlp.bias', identity)], identity),
                WeightInfo(W.pre_ln_gamma, [CkptWeightInfo('transformer.h.{i}.ln_mlp.weight', identity)], identity),
                WeightInfo(W.pre_attn_ln_beta, [CkptWeightInfo('transformer.h.{i}.ln_attn.bias', identity)], identity),
                WeightInfo(W.pre_attn_ln_gamma, [CkptWeightInfo('transformer.h.{i}.ln_attn.weight', identity)], identity),
            ])

        else:
            layer_weights.extend([
                WeightInfo(W.attn_qkv_w, [CkptWeightInfo('transformer.h.{i}.self_attention.query_key_value.weight', identity)], transpose),
                WeightInfo(W.pre_ln_beta, [CkptWeightInfo('transformer.h.{i}.input_layernorm.bias', identity)], identity),
                WeightInfo(W.pre_ln_gamma, [CkptWeightInfo('transformer.h.{i}.input_layernorm.weight', identity)], identity),
            ])

        return ModelWeightInfo(layer_weights=layer_weights, weights=weights, tp_strategy=W.gpt_style_tp_strategy)

class Falcon(GPT):
    @staticmethod
    def get_weight_cls():
        return FalconWeightInfo

    @classmethod
    def _create_config(cls, ckpt_path: str):
        config_path = os.path.join(ckpt_path, 'config.json')
        with open(config_path) as f:
            config_json = json.load(f)
        head_num = config_json.get('n_head', config_json.get('num_attention_heads'))
        config = GptInitModelParameters(
            head_num=head_num,
            head_num_kv=config_json.get('n_head_kv', config_json.get('num_kv_heads', 1)),
            size_per_head=config_json['hidden_size'] // head_num,
            inter_size=config_json['hidden_size'] * 4,
            layer_num=config_json.get('n_layer', config_json.get('num_hidden_layers')),
            max_seq_len=2048,
            vocab_size=config_json['vocab_size'],
            activation_type='gelu-none-approximate',
            has_post_decoder_layernorm=True,
            rotary_embedding_style=1,
            ckpt_path=ckpt_path)
        config.special_tokens.bos_token_id = config_json['bos_token_id']
        config.special_tokens.eos_token_id = config_json['eos_token_id']
        config.rotary_embedding_dim = config.size_per_head
        return config

register_model('falcon', Falcon, ["FalconForCausalLM"])
