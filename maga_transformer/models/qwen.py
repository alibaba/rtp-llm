
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
from maga_transformer.models.gpt import GPT
from maga_transformer.tokenizer.tokenization_qwen import QWenTokenizer as QwenTokenizerOrigin
from maga_transformer.tokenizer.tokenization_qwen2 import Qwen2Tokenizer as QWen2Tokenizer
from maga_transformer.model_factory_register import register_model
from pathlib import Path
import logging

def transpose_pad(ts, inter_padding_size, dim):
    if dim == 0:
        pad_shape = [inter_padding_size - ts[0].shape[0], ts[0].shape[1]]
    elif dim == 1:
        pad_shape = [ts[0].shape[0], inter_padding_size - ts[0].shape[1]]
    else:
        raise Exception('unknown padding dim: ' + str(dim))
    z = torch.zeros(pad_shape, device='cuda:0').half()
    return torch.cat((ts[0].cuda(), z), dim).T.to('cuda:0').contiguous()

def hidden_to_inter(hidden_size):
    ffn_m = 256
    return int((int(4 * 2 / 3 * hidden_size) * 2 + ffn_m - 1) // ffn_m * ffn_m / 2)

class QWenTokenizer(QwenTokenizerOrigin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decoder.update({v: k for k, v in self.special_tokens.items()})

class QWenWeight(ModelDeployWeightInfo):
    def _get_layer_id_info(self, ckpt_meta):
        try:
            layers = [int(name.split('.')[1]) for name in ckpt_meta['model']['language_model']['encoder'].keys() if "layers.0.self_attention.query_key_value.weight" in name]
            return layers[0], layers[-1]
        except Exception as _:
            # 'transformer.h.{i}.attn.c_attn.weight'
            layers = [int(name.split('.')[2]) for name in ckpt_meta.keys() if ".attn.c_attn.weight" in name]
            return layers[0], layers[-1]

    def _fix_megatron_layer_id_by_offset(self, meta, offset):
        try:
            fix_layer_meta = {}
            for k, v in meta['model']['language_model']['encoder'].items():
                if 'layers.' not in k:
                    fix_layer_meta.update({k:v})
                    continue
                layer_id = k.split('.')[1]
                fix_layer_id = int(layer_id) + int(offset)
                fix_k = k.replace(f"layers.{layer_id}", f"layers.{fix_layer_id}")
                fix_layer_meta.update({fix_k:v})
                # meta['model']['language_model']['encoder'][fix_k] = v
            meta['model']['language_model']['encoder'] = fix_layer_meta

        except KeyError as _:
            layer_metas = {}
            to_del_k = []
            for k, v in meta.items():
                if 'transformer.h' not in k:
                    continue
                layer_id = k.split('.')[2]
                fix_layer_id = int(layer_id) + int(offset)
                fix_k = k.replace(f"transformer.h.{layer_id}", f"transformer.h.{fix_layer_id}")
                layer_metas.update({fix_k:v})
                to_del_k.append(k)
            for k in to_del_k:
                del meta[k]
        return meta


    def _process_meta(self, meta_dicts, weight_keys):
        for meta_dict in meta_dicts:
            if 'model' in meta_dict:
                language_model = meta_dict['model']['language_model']
                if 'encoder' in language_model:
                    meta_dict.update(language_model['encoder'])
                if 'embedding' in language_model:
                    meta_dict['emb'] = language_model['embedding']['word_embeddings']['weight']
                if 'output_layer' in language_model.keys():
                    meta_dict['lm_head'] =language_model['output_layer']['weight']
                self._megatron = True
            else:
                self._megatron = False

    def _get_weight_info(self):
        if self._megatron:
            return self._get_megatron_weight_info()
        else:
            return self._get_hf_weight_info()

    def _get_megatron_layer_weight_info(self, layer_id):
        layer_weights = [
            WeightInfo(W.pre_ln_gamma, [CkptWeightInfo('layers.{i}.input_layernorm.weight', identity)],
                       identity),
            WeightInfo(W.attn_qkv_w, [CkptWeightInfo('layers.{i}.self_attention.query_key_value.weight', concat_0)],
                       functools.partial(trans_qkv, hidden_size=self._hidden_size, head_num=self._head_num)),
            WeightInfo(W.attn_qkv_b, [CkptWeightInfo('layers.{i}.self_attention.query_key_value.bias', concat_0)],
                       functools.partial(trans_qkv_b, hidden_size=self._hidden_size, head_num=self._head_num)),
            WeightInfo(W.attn_o_w, [CkptWeightInfo('layers.{i}.self_attention.dense.weight', concat_1)],
                       transpose),
            WeightInfo(W.ffn_w1, [CkptWeightInfo('layers.{i}.mlp.w2.weight', concat_0)],
                       transpose),
            WeightInfo(W.ffn_w3, [CkptWeightInfo('layers.{i}.mlp.w1.weight', concat_0)],
                       transpose),
            WeightInfo(W.ffn_w2, [CkptWeightInfo('layers.{i}.mlp.dense_4h_to_h.weight', concat_1)],
                       transpose),
            WeightInfo(W.post_ln_gamma, [CkptWeightInfo('layers.{i}.post_attention_layernorm.weight', identity)],
                       identity),
        ]

        return layer_weights

    def _get_megatron_weight_info(self):
        weights = [
            WeightInfo(W.embedding, [CkptWeightInfo('emb', concat_0)], identity),
            WeightInfo(W.lm_head, [CkptWeightInfo('lm_head', concat_0)], identity),
            WeightInfo(W.final_ln_gamma, [CkptWeightInfo('final_layernorm.weight', identity)], identity),
            WeightInfo(W.final_ln_beta, [], functools.partial(zeros, shape=[self._hidden_size])),
        ]

        layer_weights: List[List[WeightInfo]] = []
        for layer in range(self._num_layers):
            w = self._get_megatron_layer_weight_info(layer)
            layer_weights.append(w)

        model_weight_info = ModelWeightInfo(layer_weights=layer_weights, weights=weights, tp_strategy=W.gpt_style_tp_strategy)
        model_weight_info.set_lora(qkv_fun=functools.partial(trans_lora_qkv, head_num=self._head_num, head_size=self._size_per_head))
        return model_weight_info

    def _get_hf_layer_weight_info(self, layer_id):
        inter_padding_size = self._layer_inter_padding_size[layer_id] if self._layer_inter_padding_size else self._inter_padding_size
        layer_weights = [
            WeightInfo(W.pre_ln_gamma, [CkptWeightInfo('transformer.h.{i}.ln_1.weight', identity)],
                       identity),
            WeightInfo(W.attn_qkv_w, [CkptWeightInfo('transformer.h.{i}.attn.c_attn.weight', identity)],
                       transpose),
            WeightInfo(W.attn_qkv_b, [CkptWeightInfo('transformer.h.{i}.attn.c_attn.bias', identity)],
                       identity),
            WeightInfo(W.attn_o_w, [CkptWeightInfo('transformer.h.{i}.attn.c_proj.weight', identity)],
                       transpose),
            WeightInfo(W.ffn_w1, [CkptWeightInfo('transformer.h.{i}.mlp.w2.weight', identity)],
                       functools.partial(transpose_pad, inter_padding_size=inter_padding_size, dim=0)),
            WeightInfo(W.ffn_w3, [CkptWeightInfo('transformer.h.{i}.mlp.w1.weight', identity)],
                       functools.partial(transpose_pad, inter_padding_size=inter_padding_size, dim=0)),
            WeightInfo(W.ffn_w2, [CkptWeightInfo('transformer.h.{i}.mlp.c_proj.weight', identity)],
                       functools.partial(transpose_pad, inter_padding_size=inter_padding_size, dim=1)),
            WeightInfo(W.post_ln_gamma, [CkptWeightInfo('transformer.h.{i}.ln_2.weight', identity)],
                       identity),
        ]
        return layer_weights

    def _get_hf_qptq_weight_info(self, layer_id):
        layer_quant_weights =[
            WeightInfo(W.pre_ln_gamma, [CkptWeightInfo('transformer.h.{i}.ln_1.weight', identity)],
                       identity),
            WeightInfo(W.attn_qkv_w, [CkptWeightInfo('transformer.h.{i}.attn.c_attn.qweight', identity)],
                       identity),
            WeightInfo(W.attn_qkv_z, [CkptWeightInfo('transformer.h.{i}.attn.c_attn.qzeros', identity)],
                       identity),
            WeightInfo(W.attn_qkv_s, [CkptWeightInfo('transformer.h.{i}.attn.c_attn.scales', identity)],
                       identity),
            WeightInfo(W.attn_qkv_b, [CkptWeightInfo('transformer.h.{i}.attn.c_attn.bias', identity)],
                       identity),
            WeightInfo(W.attn_o_w, [CkptWeightInfo('transformer.h.{i}.attn.c_proj.qweight', identity)],
                       identity),
            WeightInfo(W.attn_o_z, [CkptWeightInfo('transformer.h.{i}.attn.c_proj.qzeros', identity)],
                       identity),
            WeightInfo(W.attn_o_s, [CkptWeightInfo('transformer.h.{i}.attn.c_proj.scales', identity)],
                       identity),
            WeightInfo(W.ffn_w1, [CkptWeightInfo('transformer.h.{i}.mlp.w2.qweight', identity)],
                       identity),
            WeightInfo(W.ffn_z1, [CkptWeightInfo('transformer.h.{i}.mlp.w2.qzeros', identity)],
                       identity),
            WeightInfo(W.ffn_s1, [CkptWeightInfo('transformer.h.{i}.mlp.w2.scales', identity)],
                       identity),
            WeightInfo(W.ffn_w3, [CkptWeightInfo('transformer.h.{i}.mlp.w1.qweight', identity)],
                       identity),
            WeightInfo(W.ffn_z3, [CkptWeightInfo('transformer.h.{i}.mlp.w1.qzeros', identity)],
                       identity),
            WeightInfo(W.ffn_s3, [CkptWeightInfo('transformer.h.{i}.mlp.w1.scales', identity)],
                       identity),
            WeightInfo(W.ffn_w2, [CkptWeightInfo('transformer.h.{i}.mlp.c_proj.qweight', identity)],
                       identity),
            WeightInfo(W.ffn_z2, [CkptWeightInfo('transformer.h.{i}.mlp.c_proj.qzeros', identity)],
                       identity),
            WeightInfo(W.ffn_s2, [CkptWeightInfo('transformer.h.{i}.mlp.c_proj.scales', identity)],
                       identity),
            WeightInfo(W.post_ln_gamma, [CkptWeightInfo('transformer.h.{i}.ln_2.weight', identity)],
                       identity),
        ]
        return layer_quant_weights
        

    def _get_hf_weight_info(self):
        weights = [
            WeightInfo(W.embedding, [CkptWeightInfo('transformer.wte.weight', identity)], identity),
            WeightInfo(W.lm_head, [CkptWeightInfo('lm_head.weight', identity)], identity),
            WeightInfo(W.final_ln_gamma, [CkptWeightInfo('transformer.ln_f.weight', identity)], identity),
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

class QWenBase(GPT):
    @staticmethod
    def get_weight_cls():
        return QWenWeight

    @staticmethod
    def _common_config(config, ckpt_path: str) -> GptInitModelParameters:
        config.rotary_embedding_dim = 128
        config.rotary_embedding_style = 1
        config.activation_type = 'SiGLU'
        config.has_pre_decoder_layernorm = False
        config.has_post_decoder_layernorm = True
        config.norm_type = 'rmsnorm'
        config.layernorm_eps = 1e-5
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
        QWen._from_hf(config, ckpt_path)
        return config

    @staticmethod
    def _from_hf(config: GptInitModelParameters, ckpt_path: str):
        config_path = os.path.join(ckpt_path, "config.json")
        if not os.path.exists(config_path):
            return
        with open(config_path) as reader:
            content = reader.read()
            config_json = json.loads(content)

        config.head_num = config_json.get("n_head", config_json.get("num_attention_heads", config.head_num))  # 如果2者不一致就是 attention sparse场景,headnum不能用attention的heads
        config.head_num_kv = config.head_num
        config.size_per_head = config_json.get("kv_channels", config.size_per_head)
        config.hidden_size = config_json.get("hidden_size", config.hidden_size)
        config.inter_size = int(
            config_json.get("intermediate_size", config_json.get("ffn_hidden_size", hidden_to_inter(config.head_num * config.size_per_head) * 2)) / 2
        )
        config.layernorm_eps = config_json.get("layer_norm_epsilon", config.layernorm_eps)
        config.layer_num = config_json.get("num_hidden_layers", config_json.get("n_layer", config.layer_num))
        config.vocab_size = config_json.get("vocab_size", config_json.get("padded_vocab_size", config.vocab_size))
        config.rotary_embedding_base = int(config_json.get('rotary_emb_base', 10000))
        config.rotary_embedding_dim = config.size_per_head
        config.special_tokens.eos_token_id = config_json.get("eos_token_id", config.special_tokens.eos_token_id)

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

        use_dynamic_ntk = config_json.get("use_dynamic_ntk")
        use_logn_attn = config_json.get("use_logn_attn")
        if (use_dynamic_ntk):
            config.rotary_embedding_style = 4
        if (use_logn_attn):
            config.use_logn_attn = True
            config.logn_seq_len = config_json.get("seq_length")

    @classmethod
    def get_tokenizer(cls, config: GptInitModelParameters):
        return QWenTokenizer.from_pretrained(config.tokenizer_path)

class QWen(QWenBase):
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
        QWenBase._common_config(config, ckpt_path)
        assert config.head_num > 0 and config.head_num_kv > 0 and config.size_per_head > 0 and config.layer_num > 0 and config.inter_size > 0, "error config"
        return config

class QWen_7B(QWenBase):
    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = GptInitModelParameters(
            head_num=32,
            head_num_kv=32,
            size_per_head=128,
            layer_num=32,
            inter_size=hidden_to_inter(4096), # 11008
            vocab_size=151936,
            max_seq_len=8192)
        QWenBase._common_config(config, ckpt_path)
        return config

class QWen_13B(QWenBase):
    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = GptInitModelParameters(
            head_num=40,
            head_num_kv=40,
            size_per_head=128,
            layer_num=40,
            inter_size=hidden_to_inter(5120), # 13696
            vocab_size=152064,
            max_seq_len=8192)
        QWen._common_config(config, ckpt_path)
        return config

class QWen_1B8(QWenBase):
    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = GptInitModelParameters(
            head_num=16,
            head_num_kv=16,
            size_per_head=128,
            layer_num=24,
            inter_size=hidden_to_inter(2048), # 5504
            vocab_size=151936,
            max_seq_len=2048)
        QWenBase._common_config(config, ckpt_path)
        return config

register_model('qwen', QWen, ["QWenLMHeadModel"])
register_model('qwen_7b', QWen_7B)
register_model('qwen_13b', QWen_13B)
register_model('qwen_1b8', QWen_1B8)
