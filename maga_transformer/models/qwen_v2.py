
import torch
import unicodedata
import types
import functools
import os
import json
from typing import List, Any

from maga_transformer.utils.model_weight import W, WeightInfo, ModelWeightInfo,\
    ModelDeployWeightInfo, CkptWeightInfo, \
    concat_0, concat_1, identity, zeros, transpose, trans_qkv, trans_qkv_b, trans_lora_qkv
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.models.qwen import QWen, transpose_pad
from maga_transformer.tokenizer.tokenization_qwen2 import Qwen2Tokenizer as QWen2Tokenizer
from maga_transformer.model_factory_register import register_model

def merge_qkv_b(ts: List[torch.Tensor]):
    q, k, v = ts
    qkv_b = torch.concat([q, k, v], dim=0).contiguous()
    return qkv_b

def merge_qkv_hf(ts: List[torch.Tensor]):
    q, k, v = ts
    qkv_weight = torch.concat([q.T, k.T, v.T], dim=1).contiguous()
    return qkv_weight

class QWenV2Weight(ModelDeployWeightInfo):
    def _get_weight_info(self):
        return self._get_hf_weight_info()
        
    def _get_hf_layer_weight_info(self, layer_id):
        inter_padding_size = self._layer_inter_padding_size[layer_id] if self._layer_inter_padding_size else self._inter_padding_size
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
            WeightInfo(W.attn_o_b, [], functools.partial(zeros, shape=[self._hidden_size])),
            WeightInfo(W.ffn_w1, [CkptWeightInfo('model.layers.{i}.mlp.gate_proj.weight', identity)],
                       functools.partial(transpose_pad, inter_padding_size=inter_padding_size, dim=0)),
            WeightInfo(W.ffn_w3, [CkptWeightInfo('model.layers.{i}.mlp.up_proj.weight', identity)],
                       functools.partial(transpose_pad, inter_padding_size=inter_padding_size, dim=0)),
            WeightInfo(W.ffn_w2, [CkptWeightInfo('model.layers.{i}.mlp.down_proj.weight', identity)],
                       functools.partial(transpose_pad, inter_padding_size=inter_padding_size, dim=1)),
            WeightInfo(W.post_ln_gamma, [CkptWeightInfo('model.layers.{i}.post_attention_layernorm.weight', identity)],
                       identity),
        ]
        return layer_weights
    
    def _get_hf_qptq_weight_info(self, layer_id):
        layer_quant_weights =[
            WeightInfo(W.pre_ln_gamma, [CkptWeightInfo('model.layers.{i}.input_layernorm.weight', identity)],
                       identity),
            # quant_weight
            WeightInfo(W.attn_qkv_w, [
                    CkptWeightInfo('model.layers.{i}.self_attn.q_proj.qweight', transpose),
                    CkptWeightInfo('model.layers.{i}.self_attn.k_proj.qweight', transpose),
                    CkptWeightInfo('model.layers.{i}.self_attn.v_proj.qweight', transpose)
                ],
                functools.partial(merge_qkv_hf)),
            WeightInfo(W.attn_qkv_z, [
                    CkptWeightInfo('model.layers.{i}.self_attn.q_proj.qzeros', transpose),
                    CkptWeightInfo('model.layers.{i}.self_attn.k_proj.qzeros', transpose),
                    CkptWeightInfo('model.layers.{i}.self_attn.v_proj.qzeros', transpose)
                ],
                functools.partial(merge_qkv_hf)),
            WeightInfo(W.attn_qkv_s, [
                    CkptWeightInfo('model.layers.{i}.self_attn.q_proj.scales', transpose),
                    CkptWeightInfo('model.layers.{i}.self_attn.k_proj.scales', transpose),
                    CkptWeightInfo('model.layers.{i}.self_attn.v_proj.scales', transpose)
                ],
                functools.partial(merge_qkv_hf)),
            WeightInfo(W.attn_qkv_b, [
                    CkptWeightInfo('model.layers.{i}.self_attn.q_proj.bias', identity),
                    CkptWeightInfo('model.layers.{i}.self_attn.k_proj.bias', identity),
                    CkptWeightInfo('model.layers.{i}.self_attn.v_proj.bias', identity)
                ], 
                functools.partial(merge_qkv_b)),

            WeightInfo(W.attn_o_w, [CkptWeightInfo('model.layers.{i}.self_attn.o_proj.qweight', identity)],
                       identity),
            WeightInfo(W.attn_o_z, [CkptWeightInfo('model.layers.{i}.self_attn.o_proj.qzeros', identity)],
                       identity),
            WeightInfo(W.attn_o_s, [CkptWeightInfo('model.layers.{i}.self_attn.o_proj.scales', identity)],
                       identity),

            WeightInfo(W.ffn_w1, [CkptWeightInfo('model.layers.{i}.mlp.gate_proj.qweight', identity)],
                       identity),
            WeightInfo(W.ffn_z1, [CkptWeightInfo('model.layers.{i}.mlp.gate_proj.qzeros', identity)],
                       identity),
            WeightInfo(W.ffn_s1, [CkptWeightInfo('model.layers.{i}.mlp.gate_proj.scales', identity)],
                       identity),

            WeightInfo(W.ffn_w3, [CkptWeightInfo('model.layers.{i}.mlp.up_proj.qweight', identity)],
                       identity),
            WeightInfo(W.ffn_z3, [CkptWeightInfo('model.layers.{i}.mlp.up_proj.qzeros', identity)],
                       identity),
            WeightInfo(W.ffn_s3, [CkptWeightInfo('model.layers.{i}.mlp.up_proj.scales', identity)],
                       identity),

            WeightInfo(W.ffn_w2, [CkptWeightInfo('model.layers.{i}.mlp.down_proj.qweight', identity)],
                       identity),
            WeightInfo(W.ffn_z2, [CkptWeightInfo('model.layers.{i}.mlp.down_proj.qzeros', identity)],
                       identity),
            WeightInfo(W.ffn_s2, [CkptWeightInfo('model.layers.{i}.mlp.down_proj.scales', identity)],
                       identity),

            WeightInfo(W.post_ln_gamma, [CkptWeightInfo('model.layers.{i}.post_attention_layernorm.weight', identity)],
                       identity),
        ]
        return layer_quant_weights

    def _get_hf_weight_info(self):
        weights = [
            WeightInfo(W.embedding, [CkptWeightInfo('model.embed_tokens.weight', identity)], identity),
            WeightInfo(W.lm_head, [CkptWeightInfo('lm_head.weight', identity)], identity),
            WeightInfo(W.final_ln_gamma, [CkptWeightInfo('model.norm.weight', identity)], identity),
            WeightInfo(W.final_ln_beta, [], functools.partial(zeros, shape=[self._hidden_size])),
        ]

        layer_weights: List[List[WeightInfo]] = []
        for layer in range(self._num_layers):
            if self._int4_mode:
                w=self._get_hf_qptq_weight_info(layer)
                layer_weights.append(w)
            else:
                w = self._get_hf_layer_weight_info(layer)
                layer_weights.append(w)

        return ModelWeightInfo(layer_weights=layer_weights, weights=weights, tp_strategy=W.gpt_style_tp_strategy)

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
        config.rotary_embedding_base = int(config_json.get("rope_theta", config.rotary_embedding_base))
        config.vocab_size = config_json["vocab_size"]
        config.rotary_embedding_dim = config.size_per_head
        config.layernorm_eps = config_json.get("rms_norm_eps", 1e-06)

        quant_config = config_json.get("quantization_config", None)
        if quant_config is not None:
            quant_bits = quant_config.get("bits", 0)
            if quant_bits != 4:
                raise ValueError("Unsupported quant bits: %s" % (quant_bits))
            config.quant_algo.int4_mode = True
            group_size = quant_config.get("group_size", 0)
            assert group_size == 128 or group_size == 64, "int4 only support group size == 64 or 128"
            config.quant_algo.weight_only_group_size = group_size
            quant_method = quant_config.get("quant_method", None)
            if quant_method == 'awq':
                config.quant_algo.is_awq = True
                config.quant_algo.has_zeros = True
            elif quant_method == 'gptq':
                config.quant_algo.is_gptq = True
                config.quant_algo.has_zeros = True
            else: 
                raise ValueError("Unsupported quant method: %s" % (quant_method))

    
    @staticmethod
    def get_weight_cls():
        return QWenV2Weight

    @classmethod
    def get_tokenizer(cls, config: GptInitModelParameters):
        tokenizer = QWen2Tokenizer.from_pretrained(config.tokenizer_path, verbose = False)
        tokenizer.im_start_id = tokenizer.encode('<|im_start|>')[0]
        tokenizer.im_end_id = tokenizer.encode('<|im_end|>')[0]
        return tokenizer

register_model('qwen_2', QWenV2, ["Qwen2ForCausalLM"])
