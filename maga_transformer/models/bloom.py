
import functools
from typing import Any, Dict

from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.utils.util import get_config_from_path
from maga_transformer.utils.model_weight import W, WeightInfo, ModelWeightInfo,\
    ModelDeployWeightInfo, CkptWeightInfo, identity, transpose, trans_qkv, trans_qkv_b
from maga_transformer.models.gpt import GPT
from maga_transformer.model_factory_register import register_model

class BloomWeightInfo(ModelDeployWeightInfo):
    def _process_meta(self, meta_dicts, weight_keys):
        if 'lm_head.weight' in weight_keys:
            self._lm_head = True
        else:
            self._lm_head = False

        if 'transformer.h.0.input_layernorm.weight' in weight_keys:
            self._transformer_prefix = True
        else:
            self._transformer_prefix = False

    def _get_weight_info(self):
        weights = [
            WeightInfo(W.embedding, [CkptWeightInfo('word_embeddings.weight', identity)], identity),
            WeightInfo(W.pre_decoder_ln_gamma, [CkptWeightInfo('word_embeddings_layernorm.weight', identity)], identity),
            WeightInfo(W.pre_decoder_ln_beta, [CkptWeightInfo('word_embeddings_layernorm.bias', identity)], identity),
            WeightInfo(W.final_ln_gamma, [CkptWeightInfo('ln_f.weight', identity)], identity),
            WeightInfo(W.final_ln_beta, [CkptWeightInfo('ln_f.bias', identity)], identity),
        ]

        layer_weights = [
            WeightInfo(W.pre_ln_beta, [CkptWeightInfo('h.{i}.input_layernorm.bias', identity)],
                       identity),

            WeightInfo(W.pre_ln_gamma, [CkptWeightInfo('h.{i}.input_layernorm.weight', identity)],
                       identity),

            WeightInfo(W.attn_qkv_w, [CkptWeightInfo('h.{i}.self_attention.query_key_value.weight', identity)],
                       functools.partial(trans_qkv, hidden_size=self._hidden_size, head_num=self._head_num)),

            WeightInfo(W.attn_qkv_b, [CkptWeightInfo('h.{i}.self_attention.query_key_value.bias', identity)],
                       functools.partial(trans_qkv_b, hidden_size=self._hidden_size, head_num=self._head_num)),

            WeightInfo(W.attn_o_w, [CkptWeightInfo('h.{i}.self_attention.dense.weight', identity)],
                       transpose),

            WeightInfo(W.attn_o_b, [CkptWeightInfo('h.{i}.self_attention.dense.bias', identity)],
                       identity),

            WeightInfo(W.ffn_w1, [CkptWeightInfo('h.{i}.mlp.dense_h_to_4h.weight', identity)],
                       transpose),

            WeightInfo(W.ffn_b1, [CkptWeightInfo('h.{i}.mlp.dense_h_to_4h.bias', identity)],
                       identity),

            WeightInfo(W.ffn_w2, [CkptWeightInfo('h.{i}.mlp.dense_4h_to_h.weight', identity)],
                       transpose),

            WeightInfo(W.ffn_b2, [CkptWeightInfo('h.{i}.mlp.dense_4h_to_h.bias', identity)],
                       identity),

            WeightInfo(W.post_ln_beta, [CkptWeightInfo('h.{i}.post_attention_layernorm.bias', identity)],
                       identity),

            WeightInfo(W.post_ln_gamma, [CkptWeightInfo('h.{i}.post_attention_layernorm.weight', identity)],
                       identity),
        ]

        if self._transformer_prefix:
            for w in layer_weights:
                w.weights[0].name = 'transformer.' + w.weights[0].name
            for w in weights:
                w.weights[0].name = 'transformer.' + w.weights[0].name

        if self._lm_head:
            weights.append(WeightInfo(W.lm_head, [CkptWeightInfo('lm_head.weight', identity)], identity))

        return ModelWeightInfo(weights, layer_weights)


class Bloom(GPT):
    @staticmethod
    def get_weight_cls():
        return BloomWeightInfo

    @staticmethod
    def from_huggingface(config_json: Dict[str, Any]):
        model_type = config_json['model_type']
        config = GptInitModelParameters(
            head_num=32,
            size_per_head=128,
            layer_num=30,
            max_seq_len=2048,
            vocab_size=250682,
        )
        if model_type != 'bloom':
            raise BaseException(f'model type is not bloom: {model_type}')
        config.head_num = config_json.get('num_attention_heads', config_json.get('n_head'))
        config.head_num_kv = config.head_num
        hidden_size = config_json.get('n_embed', config_json.get('hidden_size'))
        config.size_per_head = hidden_size // config.head_num
        config.layer_num = config_json['n_layer']
        config.max_seq_len = config_json.get('seq_length', 2048)
        config.vocab_size = config_json['vocab_size']
        config.layernorm_eps = config_json['layer_norm_epsilon']
        config.inter_size = hidden_size * 4
        config.special_tokens.eos_token_id = config_json['eos_token_id']
        return config

    @classmethod
    def _create_config(cls, ckpt_path: str):
        config_dict = get_config_from_path(ckpt_path)
        if config_dict:
            config = Bloom.from_huggingface(config_dict)
        else:
            config = GptInitModelParameters(
                head_num=32,
                head_num_kv=32,
                size_per_head=128,
                inter_size=4 * 32 * 128,
                layer_num=30,
                max_seq_len=2048,
                vocab_size=250880)
        config.layernorm_eps=1e-5
        config.layernorm_type = 'pre_layernorm'
        config.activation_type = 'gelu'
        config.has_positional_encoding=False
        config.has_pre_decoder_layernorm=True
        config.has_post_decoder_layernorm=True
        config.use_attention_linear_bias=True
        return config
    
register_model('bloom', Bloom, ["BloomForCausalLM"])
