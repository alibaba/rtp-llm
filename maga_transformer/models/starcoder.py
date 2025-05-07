from typing import Any, Dict, List
from maga_transformer.utils.util import get_config_from_path
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.utils.model_weight import W, \
     CkptWeightInfo, identity, transpose, WeightStyle
from maga_transformer.models.base_model import BaseModel
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
from maga_transformer.model_factory_register import register_model
from maga_transformer.model_loader.weight_module import WeightModule, AtomicWeight
from maga_transformer.model_loader.ffn_weight import FfnAtomicWeight, FfnConfig, FfnWeight
from maga_transformer.model_loader.attn_weight import AttnAtomicWeight
from maga_transformer.model_loader.model_weight_info import ModelWeightInfo, ModelDeployWeightInfo

class StarcoderWeightInfo(ModelDeployWeightInfo):

    def _process_meta(self, meta_dicts, weight_keys):
        for meta_dict in meta_dicts:
            if self._quant_algo.isFp8() and 'transformer.h.0.attn.c_proj.weight' in meta_dict:
               self.weight_style = WeightStyle.TRANSFORMER_ENGINE
            elif self._quant_algo.isFp8() and 'transformer.layers.0.attention.dense.weight' in meta_dict:
               self.weight_style = WeightStyle.TRT_ENGINE

    def _get_weight_info(self):
        if self.weight_style != WeightStyle.TRT_ENGINE:
            embedding_tensor_name = 'transformer.wte.weight'
            positional_tensor_name = 'transformer.wpe.weight'
        else:
            embedding_tensor_name = 'transformer.vocab_embedding.weight'
            positional_tensor_name = 'transformer.position_embedding.weight'


        embedding_tensor_name = 'transformer.wte.weight' if self.weight_style != WeightStyle.TRT_ENGINE \
            else 'transformer.vocab_embedding.weight'
        positional_tensor_name = 'transformer.wpe.weight' if self.weight_style != WeightStyle.TRT_ENGINE else 'transformer.position_embedding.weight'
        weights = [
            AtomicWeight(W.embedding, [CkptWeightInfo(embedding_tensor_name, identity)], identity),
            AtomicWeight(W.lm_head, [CkptWeightInfo('lm_head.weight', identity)], identity),
            AtomicWeight(W.positional_embedding, [CkptWeightInfo(positional_tensor_name, identity)], identity),
            AtomicWeight(W.final_ln_gamma, [CkptWeightInfo('transformer.ln_f.weight', identity)], identity),
            AtomicWeight(W.final_ln_beta, [CkptWeightInfo('transformer.ln_f.bias', identity)], identity),
        ]
        # TODO(luoli.hn) lm_head gem use fp16, maybe can use fp8 gemm
        layer_weights: List[List[WeightModule]] = []
        for layer in range(self._num_layers):
            w = self._get_hf_layer_weight_info(layer)
            layer_weights.append(w)

        return ModelWeightInfo(layer_weights=layer_weights, weights=weights)

    def _get_hf_layer_weight_info(self, layer_id: int) -> List[WeightModule]:
        attn_config=self.attn_config
        ffn_config=self.ffn_config
        ffn_w2_config = FfnConfig(
            is_gated_activation=self._is_gated_activation,
            inter_padding_size=self._inter_padding_size,
            is_moe=False,
            need_ffn_act_scale=self.need_ffn_act_scale
        )
        layer_weights = [
            AtomicWeight(W.pre_ln_beta, [CkptWeightInfo('transformer.h.{i}.ln_1.bias', identity)], identity),

            AtomicWeight(W.pre_ln_gamma, [CkptWeightInfo('transformer.h.{i}.ln_1.weight', identity)], identity),

            AttnAtomicWeight(W.attn_qkv_w, [CkptWeightInfo('transformer.h.{i}.attn.c_attn.weight', identity)], transpose, config=attn_config),

            AttnAtomicWeight(W.attn_qkv_b, [CkptWeightInfo('transformer.h.{i}.attn.c_attn.bias', identity)], identity, config=attn_config),

            AttnAtomicWeight(W.attn_o_w, [CkptWeightInfo('transformer.h.{i}.attn.c_proj.weight', identity)], transpose, config=attn_config),

            AttnAtomicWeight(W.attn_o_b, [CkptWeightInfo('transformer.h.{i}.attn.c_proj.bias', identity)], identity, config=attn_config),
            FfnWeight(sub_weights=[

                FfnAtomicWeight(W.ffn_w3, [CkptWeightInfo('transformer.h.{i}.mlp.c_fc.weight', identity)], transpose, config=ffn_config),

                FfnAtomicWeight(W.ffn_b3, [CkptWeightInfo('transformer.h.{i}.mlp.c_fc.bias', identity)], identity, config=ffn_config),

                FfnAtomicWeight(W.ffn_w2, [CkptWeightInfo('transformer.h.{i}.mlp.c_proj.weight', identity)], transpose, config=ffn_w2_config),

                FfnAtomicWeight(W.ffn_b2, [CkptWeightInfo('transformer.h.{i}.mlp.c_proj.bias', identity)], identity, config=ffn_w2_config)
            ], config=ffn_config),

            AtomicWeight(W.post_ln_beta, [CkptWeightInfo('transformer.h.{i}.ln_2.bias', identity)], identity),

            AtomicWeight(W.post_ln_gamma, [CkptWeightInfo('transformer.h.{i}.ln_2.weight', identity)], identity),
        ]
        return layer_weights


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
