
import os
import json
import functools

from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.utils.model_weight import W, CkptWeightInfo, identity, zeros, transpose
from maga_transformer.models.base_model import BaseModel
from maga_transformer.model_factory_register import register_model
from maga_transformer.model_loader.model_weight_info import ModelWeightInfo, ModelDeployWeightInfo
from maga_transformer.model_loader.weight_module import AtomicWeight
from maga_transformer.model_loader.ffn_weight import FfnAtomicWeight, FfnWeight
from maga_transformer.model_loader.attn_weight import AttnAtomicWeight

class MptWeightInfo(ModelDeployWeightInfo):
    def _get_weight_info(self):
        weights = [
            AtomicWeight(W.embedding, [CkptWeightInfo('transformer.wte.weight', identity)], identity),
            AtomicWeight(W.final_ln_gamma, [CkptWeightInfo('transformer.norm_f.weight', identity)], identity),
            AtomicWeight(W.final_ln_beta, [], functools.partial(zeros, shape=self._hidden_size)),
        ]
        attn_config = self.attn_config
        ffn_config = self.ffn_config
        layer_weights = []
        for _ in range(self._num_layers):
            layer_weight = [
                AtomicWeight(W.pre_ln_gamma, [CkptWeightInfo('transformer.blocks.{i}.norm_1.weight', identity)], identity),
                AtomicWeight(W.pre_ln_beta, [], functools.partial(zeros, shape=self._hidden_size)),
                AttnAtomicWeight(W.attn_qkv_w, [CkptWeightInfo('transformer.blocks.{i}.attn.Wqkv.weight', identity)], transpose, config=attn_config),
                AttnAtomicWeight(W.attn_o_w, [CkptWeightInfo('transformer.blocks.{i}.attn.out_proj.weight', identity)], transpose, config=attn_config),
                AtomicWeight(W.post_ln_gamma, [CkptWeightInfo('transformer.blocks.{i}.norm_2.weight', identity)], identity),
                AtomicWeight(W.post_ln_beta, [], functools.partial(zeros, shape=self._hidden_size)),
                FfnWeight(sub_weights=[
                    FfnAtomicWeight(W.ffn_w1, [CkptWeightInfo('transformer.blocks.{i}.ffn.up_proj.weight', identity)], transpose, config=ffn_config),
                    FfnAtomicWeight(W.ffn_w2, [CkptWeightInfo('transformer.blocks.{i}.ffn.down_proj.weight', identity)], transpose, config=ffn_config)
                ], config=ffn_config)
            ]
            layer_weights.append(layer_weight)
        return ModelWeightInfo(layer_weights=layer_weights, weights=weights)

class Mpt(BaseModel):
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