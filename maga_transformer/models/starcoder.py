import functools
from typing import Any, Dict, List
from maga_transformer.utils.util import get_config_from_path
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.utils.model_weight import W, WeightInfo, \
    ModelWeightInfo, ModelDeployWeightInfo, CkptWeightInfo, identity, ones, transpose
from maga_transformer.models.gpt import GPT
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
from maga_transformer.model_factory_register import register_model

class StarcoderWeightInfo(ModelDeployWeightInfo):
    def _get_weight_info(self):
        weights = [
            WeightInfo(W.embedding, [CkptWeightInfo('transformer.wte.weight', identity)], identity),
            WeightInfo(W.lm_head, [CkptWeightInfo('lm_head.weight', identity)], identity),
            WeightInfo(W.positional_embedding, [CkptWeightInfo('transformer.wpe.weight', identity)], identity),
            WeightInfo(W.final_ln_gamma, [CkptWeightInfo('transformer.ln_f.weight', identity)], identity),
            WeightInfo(W.final_ln_beta, [CkptWeightInfo('transformer.ln_f.bias', identity)], identity),
        ]
        
        layer_weights: List[List[WeightInfo]] = []
        for layer in range(self._num_layers):
            if (self._quant_algo.isGptq() or
                self._quant_algo.isAwq() or
                self._quant_algo.isSmoothQuant()):
                w=self._get_hf_4bit_quant_weight_info(layer)
            elif self._quant_algo.isOmniQuant():
                w = self._get_omni_quant_weight_info(layer)
            else:
                w = self._get_hf_layer_weight_info(layer)
            layer_weights.append(w)


        return ModelWeightInfo(layer_weights=layer_weights, weights=weights, tp_strategy=W.gpt_style_tp_strategy)
    
    def _get_hf_layer_weight_info(self, layer_id: int) -> List[WeightInfo]:
        layer_weights = [
            WeightInfo(W.pre_ln_beta, [CkptWeightInfo('transformer.h.{i}.ln_1.bias', identity)], identity),

            WeightInfo(W.pre_ln_gamma, [CkptWeightInfo('transformer.h.{i}.ln_1.weight', identity)], identity),

            WeightInfo(W.attn_qkv_w, [CkptWeightInfo('transformer.h.{i}.attn.c_attn.weight', identity)], transpose),

            WeightInfo(W.attn_qkv_b, [CkptWeightInfo('transformer.h.{i}.attn.c_attn.bias', identity)], identity),

            WeightInfo(W.attn_o_w, [CkptWeightInfo('transformer.h.{i}.attn.c_proj.weight', identity)], transpose),

            WeightInfo(W.attn_o_b, [CkptWeightInfo('transformer.h.{i}.attn.c_proj.bias', identity)], identity),

            WeightInfo(W.ffn_w3, [CkptWeightInfo('transformer.h.{i}.mlp.c_fc.weight', identity)], transpose),

            WeightInfo(W.ffn_b3, [CkptWeightInfo('transformer.h.{i}.mlp.c_fc.bias', identity)], identity),

            WeightInfo(W.ffn_w2, [CkptWeightInfo('transformer.h.{i}.mlp.c_proj.weight', identity)], transpose),

            WeightInfo(W.ffn_b2, [CkptWeightInfo('transformer.h.{i}.mlp.c_proj.bias', identity)], identity),

            WeightInfo(W.post_ln_beta, [CkptWeightInfo('transformer.h.{i}.ln_2.bias', identity)], identity),

            WeightInfo(W.post_ln_gamma, [CkptWeightInfo('transformer.h.{i}.ln_2.weight', identity)], identity),
        ]
        return layer_weights

    def _get_hf_4bit_quant_weight_info(self, layer_id: int):
        orig_weights = [W.pre_ln_gamma, W.pre_ln_beta, W.attn_qkv_b, W.attn_o_b, W.ffn_b1, W.ffn_b2, W.post_ln_gamma, W.post_ln_beta]
        layer_quant_weights: List[WeightInfo] = self._get_layer_weight_by_names(layer_id, orig_weights)
        
        layer_quant_weights.extend([
            WeightInfo(W.attn_qkv_w, [CkptWeightInfo('transformer.h.{i}.attn.c_attn.qweight', identity)], identity),
            WeightInfo(W.attn_qkv_z, [CkptWeightInfo('transformer.h.{i}.attn.c_attn.qzeros', identity)], identity),
            WeightInfo(W.attn_qkv_s, [CkptWeightInfo('transformer.h.{i}.attn.c_attn.scales')], identity),
            
            WeightInfo(W.attn_o_w, [CkptWeightInfo('transformer.h.{i}.attn.c_proj.qweight', identity)], identity),
            WeightInfo(W.attn_o_z, [CkptWeightInfo('transformer.h.{i}.attn.c_proj.qzeros', identity)], identity),
            WeightInfo(W.attn_o_s, [CkptWeightInfo('transformer.h.{i}.attn.c_proj.scales')], identity),

            WeightInfo(W.ffn_w3, [CkptWeightInfo('transformer.h.{i}.mlp.c_fc.qweight', identity)], identity),
            WeightInfo(W.ffn_z3, [CkptWeightInfo('transformer.h.{i}.mlp.c_fc.qzeros', identity)], identity),
            WeightInfo(W.ffn_s3, [CkptWeightInfo('transformer.h.{i}.mlp.c_fc.scales', identity)], identity),
            WeightInfo(W.ffn_act_s, [CkptWeightInfo('transformer.h.{i}.mlp.act.scales', identity)], identity),

            WeightInfo(W.ffn_w2, [CkptWeightInfo('transformer.h.{i}.mlp.c_proj.qweight', identity)], identity),
            WeightInfo(W.ffn_z2, [CkptWeightInfo('transformer.h.{i}.mlp.c_proj.qzeros', identity)], identity),
            WeightInfo(W.ffn_s2, [CkptWeightInfo('transformer.h.{i}.mlp.c_proj.scales', identity)], identity),
        ])
        return layer_quant_weights

    def _get_omni_quant_weight_info(self, layer_id: int):
        orig_weights = [W.pre_ln_gamma, W.pre_ln_beta, W.post_ln_gamma, W.post_ln_beta]
        layer_quant_weights = self._get_layer_weight_by_names(layer_id, orig_weights)
        layer_quant_weights.extend([
            WeightInfo(W.attn_qkv_w, [CkptWeightInfo('transformer.h.{i}.attn.c_attn.qweight')], transpose),
            WeightInfo(W.attn_qkv_b, [CkptWeightInfo('transformer.h.{i}.attn.c_attn.bias')], identity),
            WeightInfo(W.attn_qkv_s, [CkptWeightInfo('transformer.h.{i}.attn.c_attn.scales')], identity),

            WeightInfo(W.attn_o_w, [CkptWeightInfo('transformer.h.{i}.attn.c_proj.qweight', identity)], transpose),
            WeightInfo(W.attn_o_b, [CkptWeightInfo('transformer.h.{i}.attn.c_proj.bias')], identity),
            WeightInfo(W.attn_o_s, [CkptWeightInfo('transformer.h.{i}.attn.c_proj.scales')], identity),
            WeightInfo(W.attn_o_smoother, [CkptWeightInfo('transformer.h.{i}.attn.c_proj.smoother')], identity),
            WeightInfo(W.attn_o_shift, [CkptWeightInfo('transformer.h.{i}.attn.c_proj.shift')], identity),

            WeightInfo(W.ffn_w3, [CkptWeightInfo('transformer.h.{i}.mlp.c_fc.qweight', identity)], transpose),
            WeightInfo(W.ffn_b3, [CkptWeightInfo('transformer.h.{i}.mlp.c_fc.bias')], identity),
            WeightInfo(W.ffn_s3, [CkptWeightInfo('transformer.h.{i}.mlp.c_fc.scales', identity)], identity),

            WeightInfo(W.ffn_w2, [CkptWeightInfo('transformer.h.{i}.mlp.c_proj.qweight')], transpose),
            WeightInfo(W.ffn_b2, [CkptWeightInfo('transformer.h.{i}.mlp.c_proj.bias')], identity),
            WeightInfo(W.ffn_s2, [CkptWeightInfo('transformer.h.{i}.mlp.c_proj.scales')], identity),

            WeightInfo(W.ffn_smoother, [], functools.partial(ones, shape=self._inter_padding_size)), 
        ])
        return layer_quant_weights

    def _get_layer_weight_by_names(self, layer_id: int, weight_names: List[str]) -> List[WeightInfo]:
        layer_weights = self._get_hf_layer_weight_info(layer_id)
        res = []
        for layer_weight in layer_weights:
            if layer_weight.name in weight_names:
                res.append(layer_weight)
        return res
    
    def _get_layer_weight_by_exclude_names(self, layer_id: int, weight_names: List[str]) -> List[WeightInfo]:
        layer_weights = self._get_hf_layer_weight_info(layer_id)
        res = []
        for layer_weight in layer_weights:
            if layer_weight.name not in weight_names:
                res.append(layer_weight)
        return res


StarcoderTokenizer = GPT2TokenizerFast

class StarCoder(GPT):
    @classmethod
    def get_tokenizer(cls, config: GptInitModelParameters):
        return StarcoderTokenizer.from_pretrained(config.tokenizer_path)

    @staticmethod
    def get_weight_cls():
        return StarcoderWeightInfo

    @staticmethod
    def from_huggingface(ckpt_path: str, config_json: Dict[str, Any]):
        model_type = config_json['model_type']
        config = GptInitModelParameters(
            head_num=config_json['n_head'],
            size_per_head=config_json['n_embd'] // config_json['n_head'],
            layer_num=config_json['n_layer'],
            max_seq_len=config_json.get('n_positions', 8192),
            vocab_size=config_json['vocab_size'],
        )
        if model_type != 'gpt_bigcode':
            raise BaseException(f'model type is not starcoder: {model_type}')
        config.head_num_kv = 1
        config.layernorm_eps = config_json['layer_norm_epsilon']
        config.inter_size = config_json['n_inner']
        config.special_tokens.eos_token_id = config_json['eos_token_id']
        config.special_tokens.bos_token_id = config_json['bos_token_id']
        # config.activation_type = config_json['activation_function']
        config.has_positional_encoding = True
        config.has_post_decoder_layernorm = True
        config.tie_word_embeddings = config_json.get('tie_word_embeddings', False)
        GPT._load_quant_config(ckpt_path, config_json, config)
        config.need_ffn_act_scale = config.quant_algo.isAwq()
        return config

    @classmethod
    def _create_config(cls, ckpt_path: str):
        config_dict = get_config_from_path(ckpt_path)
        if config_dict:
            config = StarCoder.from_huggingface(ckpt_path, config_dict)
        else:
            config = GptInitModelParameters(
                head_num=48,
                head_num_kv=1,
                size_per_head=128,
                inter_size=4 * 6144,
                layer_num=40,
                max_seq_len=8192,
                vocab_size=49152,
                bos_token_id=0,
                eos_token_id=0,
                has_positional_encoding=True,
                has_post_decoder_layernorm=True)
        return config

register_model('gpt_bigcode', StarCoder)
register_model('wizardcoder', StarCoder)
