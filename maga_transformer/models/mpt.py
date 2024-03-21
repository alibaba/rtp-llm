
import os
import json
import functools

from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.utils.model_weight import W, WeightInfo, ModelWeightInfo, \
    ModelDeployWeightInfo, CkptWeightInfo, identity, zeros, transpose
from maga_transformer.models.gpt import GPT
from maga_transformer.model_factory_register import register_model

class MptWeightInfo(ModelDeployWeightInfo):
    def _get_weight_info(self):
        weights = [
            WeightInfo(W.embedding, [CkptWeightInfo('transformer.wte.weight', identity)], identity),
            WeightInfo(W.final_ln_gamma, [CkptWeightInfo('transformer.norm_f.weight', identity)], identity),
            WeightInfo(W.final_ln_beta, [], functools.partial(zeros, shape=self._hidden_size)),
        ]
        layer_weights = [
            WeightInfo(W.pre_ln_gamma, [CkptWeightInfo('transformer.blocks.{i}.norm_1.weight', identity)], identity),
            WeightInfo(W.pre_ln_beta, [], functools.partial(zeros, shape=self._hidden_size)),
            WeightInfo(W.attn_qkv_w, [CkptWeightInfo('transformer.blocks.{i}.attn.Wqkv.weight', identity)], transpose),
            WeightInfo(W.attn_qkv_b, [], functools.partial(zeros, shape=self._hidden_size * 3)),
            WeightInfo(W.attn_o_w, [CkptWeightInfo('transformer.blocks.{i}.attn.out_proj.weight', identity)], transpose),
            WeightInfo(W.attn_o_b, [], functools.partial(zeros, shape=self._hidden_size)),
            WeightInfo(W.post_ln_gamma, [CkptWeightInfo('transformer.blocks.{i}.norm_2.weight', identity)], identity),
            WeightInfo(W.post_ln_beta, [], functools.partial(zeros, shape=self._hidden_size)),
            WeightInfo(W.ffn_w1, [CkptWeightInfo('transformer.blocks.{i}.ffn.up_proj.weight', identity)], transpose),
            WeightInfo(W.ffn_w2, [CkptWeightInfo('transformer.blocks.{i}.ffn.down_proj.weight', identity)], transpose),
        ]
        return ModelWeightInfo(layer_weights=layer_weights, weights=weights, tp_strategy=W.gpt_style_tp_strategy)

class Mpt(GPT):
    @staticmethod
    def get_weight_cls():
        return MptWeightInfo

    @classmethod
    def _create_config(cls, ckpt_path: str):
        config_path = os.path.join(ckpt_path, 'config.json')
        with open(config_path) as f:
            config_json = json.load(f)
        config = GptInitModelParameters(
            head_num=config_json['n_heads'],
            size_per_head=config_json['d_model'] // config_json['n_heads'],
            inter_size=config_json['d_model'] * 4,
            layer_num=config_json['n_layers'],
            max_seq_len=8192,
            vocab_size=config_json['vocab_size'],
            activation_type='gelu-none-approximate',
            has_post_decoder_layernorm=True,
            use_attention_linear_bias=True)
        return config

register_model('mpt', Mpt)