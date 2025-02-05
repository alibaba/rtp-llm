import functools
import torch
from typing import Any, Dict, List
from maga_transformer.utils.util import get_config_from_path
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.utils.model_weight import W, WeightInfo, \
    ModelWeightInfo, ModelDeployWeightInfo, CkptWeightInfo, identity, ones, transpose, get_tensor_reciprocal, get_tensor_from_scalar, Fp8WeightStyle
from maga_transformer.models.base_model import BaseModel
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
from maga_transformer.model_factory_register import register_model
from maga_transformer.utils.group_quant_weight_util import get_layer_group_quant_weight_info
from maga_transformer.utils.per_tensor_fp8_weight_util import get_layer_per_tensor_fp8_scale_weight_info, get_trt_engine_layer_weight_info

class StarcoderWeightInfo(ModelDeployWeightInfo):

    def _process_meta(self, meta_dicts, weight_keys):
        for meta_dict in meta_dicts:
            if self._quant_algo.isFp8() and 'transformer.h.0.attn.c_proj.weight' in meta_dict:
               self.fp8_weight_stype = Fp8WeightStyle.TRANSFORMER_ENGINE
            elif self._quant_algo.isFp8() and 'transformer.layers.0.attention.dense.weight' in meta_dict:
               self.fp8_weight_stype = Fp8WeightStyle.TRT_ENGINE        

    def _get_weight_info(self):
        if self.fp8_weight_stype != Fp8WeightStyle.TRT_ENGINE:
            embedding_tensor_name = 'transformer.wte.weight'
            positional_tensor_name = 'transformer.wpe.weight'
        else:
            embedding_tensor_name = 'transformer.vocab_embedding.weight'
            positional_tensor_name = 'transformer.position_embedding.weight'

                
        embedding_tensor_name = 'transformer.wte.weight' if self.fp8_weight_stype != Fp8WeightStyle.TRT_ENGINE \
            else 'transformer.vocab_embedding.weight'
        positional_tensor_name = 'transformer.wpe.weight' if self.fp8_weight_stype != Fp8WeightStyle.TRT_ENGINE else 'transformer.position_embedding.weight'
        weights = [
            WeightInfo(W.embedding, [CkptWeightInfo(embedding_tensor_name, identity)], identity),
            WeightInfo(W.lm_head, [CkptWeightInfo('lm_head.weight', identity)], identity),
            WeightInfo(W.positional_embedding, [CkptWeightInfo(positional_tensor_name, identity)], identity),
            WeightInfo(W.final_ln_gamma, [CkptWeightInfo('transformer.ln_f.weight', identity)], identity),
            WeightInfo(W.final_ln_beta, [CkptWeightInfo('transformer.ln_f.bias', identity)], identity),
        ]
        # TODO(luoli.hn) lm_head gem use fp16, maybe can use fp8 gemm
        layer_weights: List[List[WeightInfo]] = []
        for layer in range(self._num_layers):
            if (self._quant_algo.isGptq() or
                self._quant_algo.isAwq()):
                w = self._get_hf_layer_weight_info(layer)
                inter_padding_size = self._layer_inter_padding_size[layer] if self._layer_inter_padding_size else self._inter_padding_size
                w = get_layer_group_quant_weight_info(w, self._quant_algo, inter_padding_size)
                w.append(WeightInfo(W.ffn_act_s, [CkptWeightInfo('transformer.h.{i}.mlp.act.scales', identity)], identity))
            elif self._quant_algo.isOmniQuant():
                w = self._get_omni_quant_weight_info(layer)
            elif self.fp8_weight_stype == Fp8WeightStyle.TRT_ENGINE:
                hf_w = self._get_hf_layer_weight_info(layer)
                w = get_trt_engine_layer_weight_info(hf_w)
                scale_w = get_layer_per_tensor_fp8_scale_weight_info(w)
                w.extend(scale_w)
            elif self.fp8_weight_stype == Fp8WeightStyle.TRANSFORMER_ENGINE:
                w = self._get_hf_layer_weight_info(layer)
                scale_w = get_layer_per_tensor_fp8_scale_weight_info(w)
                w.extend(scale_w)
            else:
                w = self._get_hf_layer_weight_info(layer)
            layer_weights.append(w)

        return ModelWeightInfo(layer_weights=layer_weights, weights=weights, tp_strategy=self._get_gpt_style_tp_strategy())

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

    def _get_omni_quant_weight_info(self, layer_id: int):
        orig_weights = [W.pre_ln_gamma, W.pre_ln_beta, W.post_ln_gamma, W.post_ln_beta]
        layer_quant_weights = self._get_layer_weight_by_names(layer_id, orig_weights)
        layer_quant_weights.extend([
            WeightInfo(W.pre_ln_beta, [CkptWeightInfo('transformer.h.{i}.ln_1.bias', identity)], identity),
            WeightInfo(W.pre_ln_gamma, [CkptWeightInfo('transformer.h.{i}.ln_1.weight', identity)], identity),

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

    def _get_fp8_weight_info(self, layer_id: int):
        orig_weights = []
        layer_quant_weights = self._get_layer_weight_by_names(layer_id, orig_weights)
        layer_quant_weights.extend([
            WeightInfo(W.pre_ln_beta, [CkptWeightInfo('transformer.layers.{i}.input_layernorm.bias', identity)], identity),
            WeightInfo(W.pre_ln_gamma, [CkptWeightInfo('transformer.layers.{i}.input_layernorm.weight', identity)], identity),

            WeightInfo(W.pre_ln_static_quant, [CkptWeightInfo('transformer.layers.{i}.attention.qkv.activation_scaling_factor', identity)], get_tensor_reciprocal, torch.float32),
            WeightInfo(W.pre_ln_static_quant_reciprocal, [CkptWeightInfo('transformer.layers.{i}.attention.qkv.activation_scaling_factor', identity)], get_tensor_from_scalar, torch.float32),
            WeightInfo(W.attn_qkv_w, [CkptWeightInfo('transformer.layers.{i}.attention.qkv.weight')], identity),
            WeightInfo(W.attn_qkv_b, [CkptWeightInfo('transformer.layers.{i}.attention.qkv.bias')], identity),
            WeightInfo(W.attn_qkv_s, [CkptWeightInfo('transformer.layers.{i}.attention.qkv.weights_scaling_factor')], identity),

            WeightInfo(W.attention_output_static_quant, [CkptWeightInfo('transformer.layers.{i}.attention.dense.activation_scaling_factor', identity)], get_tensor_from_scalar, torch.float32),
            WeightInfo(W.attention_output_static_quant_reciprocal, [CkptWeightInfo('transformer.layers.{i}.attention.dense.activation_scaling_factor', identity)], get_tensor_reciprocal, torch.float32),
            WeightInfo(W.attn_o_w, [CkptWeightInfo('transformer.layers.{i}.attention.dense.weight', identity)], identity),
            WeightInfo(W.attn_o_b, [CkptWeightInfo('transformer.layers.{i}.attention.dense.bias')], identity),
            WeightInfo(W.attn_o_s, [CkptWeightInfo('transformer.layers.{i}.attention.dense.weights_scaling_factor')], identity),

            WeightInfo(W.ffn_intermediate_weight3_static_quant, [CkptWeightInfo('transformer.layers.{i}.mlp.fc.activation_scaling_factor', identity)], get_tensor_reciprocal, torch.float32),
            WeightInfo(W.ffn_intermediate_weight3_static_quant_reciprocal, [CkptWeightInfo('transformer.layers.{i}.mlp.fc.activation_scaling_factor', identity)], get_tensor_from_scalar, torch.float32),
            WeightInfo(W.ffn_w3, [CkptWeightInfo('transformer.layers.{i}.mlp.fc.weight', identity)], identity),
            WeightInfo(W.ffn_b3, [CkptWeightInfo('transformer.layers.{i}.mlp.fc.bias')], identity),
            WeightInfo(W.ffn_s3, [CkptWeightInfo('transformer.layers.{i}.mlp.fc.weights_scaling_factor', identity)], identity),

            WeightInfo(W.ffn_intermediate_weight2_static_quant, [CkptWeightInfo('transformer.layers.{i}.mlp.proj.activation_scaling_factor', identity)], get_tensor_from_scalar, torch.float32), # now use quant so use get_tensor_from_scalar
            WeightInfo(W.ffn_intermediate_weight2_static_quant_reciprocal, [CkptWeightInfo('transformer.layers.{i}.mlp.proj.activation_scaling_factor', identity)], get_tensor_from_scalar, torch.float32),
            WeightInfo(W.ffn_w2, [CkptWeightInfo('transformer.layers.{i}.mlp.proj.weight')], identity),
            WeightInfo(W.ffn_b2, [CkptWeightInfo('transformer.layers.{i}.mlp.proj.bias')], identity),
            WeightInfo(W.ffn_s2, [CkptWeightInfo('transformer.layers.{i}.mlp.proj.weights_scaling_factor')], identity),

            WeightInfo(W.post_ln_gamma, [CkptWeightInfo('transformer.layers.{i}.post_layernorm.weight', identity)], identity),
            WeightInfo(W.post_ln_beta, [CkptWeightInfo('transformer.layers.{i}.post_layernorm.bias', identity)], identity),
            WeightInfo(W.post_ln_static_quant, [CkptWeightInfo('transformer.layers.{i}.mlp.fc.activation_scaling_factor', get_tensor_reciprocal)], get_tensor_from_scalar, torch.float32),
            WeightInfo(W.post_ln_static_quant_reciprocal, [CkptWeightInfo('transformer.layers.{i}.mlp.fc.activation_scaling_factor', identity)], get_tensor_from_scalar, torch.float32),
            
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

class StarCoder(BaseModel):
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
                has_positional_encoding=True,
                has_post_decoder_layernorm=True)
            config.special_tokens.bos_token_id=0
            config.special_tokens.eos_token_id=0
        return config

    @classmethod
    def _load_quant_config(cls, ckpt_path: str,  config: GptInitModelParameters):
        super(StarCoder, cls)._load_quant_config(ckpt_path, config)
        config.need_ffn_act_scale = config.quant_algo.isAwq()

register_model('gpt_bigcode', StarCoder, ['GPTBigCodeForCausalLM'])
register_model('wizardcoder', StarCoder)
