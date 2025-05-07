import logging

import torch
import functools
import math
import os
import json
from typing import List

from maga_transformer.models.base_model import BaseModel
from maga_transformer.utils.model_weight import (
    W,
    CkptWeightInfo,
    concat_0_tranpose,
    identity,
    kv_split1,
    kv_split2,
    mla_pad_t,
    transpose,
    stack_,
    transpose_kv_rope,
    transpose_q_rope,
    transpose_slice_k,
    transpose_slice_v,
    yarn_get_mscale,
    zeros,
    transpose_pad,
    multipy_identity,
    stack_moe_w1
)
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters, MlaOpsType
from maga_transformer.model_factory_register import register_model
from maga_transformer.models.rotary_embedding.deepseek_rotary_embedding import DeepseekV3YarnRotaryEmbedding
from maga_transformer.model_loader.weight_module import WeightModule, AtomicWeight
from maga_transformer.model_loader.ffn_weight import FfnAtomicWeight, FfnWeight, FfnConfig
from maga_transformer.model_loader.model_weight_info import ModelWeightInfo, ModelDeployWeightInfo
from maga_transformer.model_loader.ffn_weight import FfnAtomicWeight, FfnConfig, FfnWeight, MoeAtomicWeight, MoeConfig, MoeWithSharedWeight
from maga_transformer.model_loader.attn_weight import MlaConfig, MlaAttnAtomicWeight


class DeepSeekV2Weight(ModelDeployWeightInfo):
    q_use_lora = False
    has_e_score_correction_bias = False

    def __init__(self, config: GptInitModelParameters, tp_size: int, tp_rank: int):
        super().__init__(config, tp_size, tp_rank)

    def _process_meta(self, meta_dict, weight_keys):
        if "model.layers.0.self_attn.q_a_proj.weight" in weight_keys:
            self.q_use_lora = True
        for layer_id in range(self._num_layers):
            if f"model.layers.{layer_id}.mlp.gate.e_score_correction_bias" in weight_keys:
                self.has_e_score_correction_bias = True
                break

    def _get_hf_layer_weight_info(self, layer_id: int):
        attn_config = MlaConfig(head_num=self._head_num,
                                 nope_head_dim=self.nope_head_dim,
                                 rope_head_dim=self.rope_head_dim,
                                 kv_lora_rank=self.kv_lora_rank,
                                 ope_head_dim=self.nope_head_dim,
                                 v_head_dim=self.v_head_dim,
                                 use_mla=self.config.use_mla and self.config.mla_ops_type != MlaOpsType.MHA,
                                 q_use_lora=self.q_use_lora)
        layer_weights = [
            AtomicWeight(W.pre_ln_gamma, [CkptWeightInfo('model.layers.{i}.input_layernorm.weight', identity)],
                       identity),
            MlaAttnAtomicWeight(W.attn_o_w, [CkptWeightInfo('model.layers.{i}.self_attn.o_proj.weight', identity)],
                       functools.partial(mla_pad_t, head_num=self._head_num, nope_head_dim=self.nope_head_dim, rope_head_dim=0),
                       config=attn_config),
            MlaAttnAtomicWeight(W.post_ln_gamma, [CkptWeightInfo('model.layers.{i}.post_attention_layernorm.weight', identity)],
                       identity),
        ]
        mla_layer_weights = [
            MlaAttnAtomicWeight(W.mla_k_nope_w, [CkptWeightInfo('model.layers.{i}.self_attn.kv_b_proj.weight', identity)],
                       functools.partial(kv_split1, kv_lora_rank=self.kv_lora_rank, nope_head_dim=self.nope_head_dim, v_head_dim=self.v_head_dim),
                       config=attn_config),
            MlaAttnAtomicWeight(W.mla_v_w, [CkptWeightInfo('model.layers.{i}.self_attn.kv_b_proj.weight', identity)],
                       functools.partial(kv_split2, kv_lora_rank=self.kv_lora_rank, nope_head_dim=self.nope_head_dim, v_head_dim=self.v_head_dim),
                       config=attn_config),
            MlaAttnAtomicWeight(W.mla_kv_a_ln_gamma, [CkptWeightInfo('model.layers.{i}.self_attn.kv_a_layernorm.weight', identity)],
                       identity, config=attn_config),
        ]

        if self.q_use_lora:
            mla_layer_weights.extend([
                MlaAttnAtomicWeight(W.mla_q_b_w, [CkptWeightInfo('model.layers.{i}.self_attn.q_b_proj.weight', functools.partial(transpose_q_rope, head_num=self._head_num, nope_head_dim=self.nope_head_dim, rope_size=self.rope_head_dim))],
                        transpose, config=attn_config),
                MlaAttnAtomicWeight(W.mla_q_a_ln_gamma, [CkptWeightInfo('model.layers.{i}.self_attn.q_a_layernorm.weight')],
                        identity, config=attn_config)
            ])
            q_a_weight = CkptWeightInfo('model.layers.{i}.self_attn.q_a_proj.weight')
            mla_layer_weights.append(
                MlaAttnAtomicWeight(W.mla_fusedqkrope_w, [q_a_weight, CkptWeightInfo('model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight', functools.partial(transpose_kv_rope, kv_lora_rank=self.kv_lora_rank, rope_size=self.rope_head_dim))],
                                    concat_0_tranpose, config=attn_config)
            )
        else:
            q_a_weight = CkptWeightInfo('model.layers.{i}.self_attn.q_proj.weight', functools.partial(transpose_q_rope, head_num=self._head_num, nope_head_dim=self.nope_head_dim, rope_size=self.rope_head_dim))
            mla_layer_weights.append(
                AtomicWeight(W.mla_fusedqkrope_no_lora_w, [q_a_weight, CkptWeightInfo('model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight', functools.partial(transpose_kv_rope, kv_lora_rank=self.kv_lora_rank, rope_size=self.rope_head_dim))],
                             concat_0_tranpose, config=attn_config)
            )

        if self.config.use_mla and self.config.mla_ops_type != MlaOpsType.MHA:
            mla_layer_weights.append(
                MlaAttnAtomicWeight(W.mla_kc, [CkptWeightInfo('model.layers.{i}.self_attn.kv_b_proj.weight', identity)],
                           functools.partial(transpose_slice_k, head_num=self._head_num, nope_head_dim=self.nope_head_dim, v_head_dim=self.v_head_dim, lora_rank=self.kv_lora_rank), config=attn_config))
            mla_layer_weights.append(
                MlaAttnAtomicWeight(W.mla_vc, [CkptWeightInfo('model.layers.{i}.self_attn.kv_b_proj.weight', identity)],
                           functools.partial(transpose_slice_v, head_num=self._head_num, nope_head_dim=self.nope_head_dim, v_head_dim=self.v_head_dim, lora_rank=self.kv_lora_rank), config=attn_config))

        layer_weights.extend(mla_layer_weights)
        layer_weights.extend(self._get_hf_ffn_layer_weight_info(layer_id))
        return layer_weights


    def _get_hf_ffn_layer_weight_info(self, layer_id: int):
        inter_padding_size = self._layer_inter_padding_size[layer_id] if self._layer_inter_padding_size else self._inter_padding_size

        ffn_config = FfnConfig(
            inter_padding_size=inter_padding_size,
            is_gated_activation=self._is_gated_activation,
            is_moe=False
        )

        if layer_id in self.moe_layer_index_:
            moe_config = MoeConfig(
                inter_padding_size=inter_padding_size,
                expert_num=self.expert_num_,
                routed_scaling_factor=self.routed_scaling_factor
            )
            layer_weights = [
                MoeWithSharedWeight(sub_weights=[
                    MoeAtomicWeight(W.moe_gate, [CkptWeightInfo('model.layers.{i}.mlp.gate.weight', identity)], transpose, config=moe_config),
                    FfnAtomicWeight(W.ffn_w1, [CkptWeightInfo('model.layers.{i}.mlp.shared_experts.gate_proj.weight', identity)], functools.partial(transpose_pad, inter_padding_size=inter_padding_size, dim=0), config=ffn_config),
                    FfnAtomicWeight(W.ffn_w2, [CkptWeightInfo('model.layers.{i}.mlp.shared_experts.down_proj.weight', identity)], functools.partial(transpose_pad, inter_padding_size=inter_padding_size, dim=1), config=ffn_config),
                    FfnAtomicWeight(W.ffn_w3, [CkptWeightInfo('model.layers.{i}.mlp.shared_experts.up_proj.weight', identity)], functools.partial(transpose_pad, inter_padding_size=inter_padding_size, dim=0), config=ffn_config),
                    MoeAtomicWeight(W.moe_w2, [CkptWeightInfo('model.layers.{i}.mlp.experts.{expert_id}.down_proj.weight',
                                                        functools.partial(multipy_identity, scale=self.routed_scaling_factor))],
                                    stack_, config=moe_config),
                    MoeAtomicWeight(W.moe_w1, [CkptWeightInfo('model.layers.{i}.mlp.experts.{expert_id}.up_proj.weight', identity)] + \
                                        [CkptWeightInfo('model.layers.{i}.mlp.experts.{expert_id}.gate_proj.weight', identity)],
                                        stack_moe_w1, config=moe_config)
                ], config = moe_config
                )
            ]
            if self.has_e_score_correction_bias:
                layer_weights.append(AtomicWeight(W.e_score_correction_b, [CkptWeightInfo('model.layers.{i}.mlp.gate.e_score_correction_bias', identity)], identity, data_type=torch.float32))
            return layer_weights
        else:

            return [
                FfnWeight(sub_weights=[
                    FfnAtomicWeight(W.ffn_w1, [CkptWeightInfo('model.layers.{i}.mlp.gate_proj.weight', identity)],
                                    functools.partial(transpose_pad, inter_padding_size=inter_padding_size, dim=0), config=ffn_config),
                    FfnAtomicWeight(W.ffn_w2, [CkptWeightInfo('model.layers.{i}.mlp.down_proj.weight', identity)],
                                    functools.partial(transpose_pad, inter_padding_size=inter_padding_size, dim=1), config=ffn_config),
                    FfnAtomicWeight(W.ffn_w3, [CkptWeightInfo('model.layers.{i}.mlp.up_proj.weight', identity)],
                                    functools.partial(transpose_pad, inter_padding_size=inter_padding_size, dim=0), config=ffn_config)
                ], config=ffn_config
                ),
            ]

    def _get_weight_info(self):
        layer_weights: List[List[WeightModule]] = []
        weights = [
            AtomicWeight(W.embedding, [CkptWeightInfo('model.embed_tokens.weight', identity)], identity),
            AtomicWeight(W.final_ln_gamma, [CkptWeightInfo('model.norm.weight', identity)], identity),
            AtomicWeight(W.final_ln_beta, [], functools.partial(zeros, shape=[self._hidden_size])),
            AtomicWeight(W.lm_head, [CkptWeightInfo('lm_head.weight', identity)], identity),
        ]
        for layer in range(self._num_layers):
            layer_weights.append(self._get_hf_layer_weight_info(layer))
        return ModelWeightInfo(layer_weights=layer_weights, weights=weights)


class DeepSeekV2(BaseModel):
    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = GptInitModelParameters(
            head_num=0,
            head_num_kv=0,
            size_per_head=0,
            layer_num=0,
            inter_size=0,
            vocab_size=102400,
            max_seq_len=8192,
            norm_type='rmsnorm',
            has_post_decoder_layernorm=True)
        config.activation_type = "gated-silu"
        DeepSeekV2._from_hf(config, ckpt_path)
        return config

    @staticmethod
    def _from_hf(config: GptInitModelParameters, ckpt_path: str):
        config_path = os.path.join(ckpt_path, "config.json")
        if not os.path.exists(config_path):
            return
        with open(config_path) as reader:
            content = reader.read()
            config_json = json.loads(content)
            config.inter_size = config_json["intermediate_size"]
            config.head_num = config_json["num_attention_heads"]
            config.head_num_kv = config_json.get("num_key_value_heads", config.head_num)
            config.layer_num = config_json["num_hidden_layers"]
            config.rotary_embedding_base = config_json.get("rope_theta", config.rotary_embedding_base)
            config.vocab_size = config_json["vocab_size"]
            config.layernorm_eps = config_json.get("rms_norm_eps", 1e-06)
            config.tie_word_embeddings = config_json.get('tie_word_embeddings', False)
            config.hidden_size = config_json["hidden_size"]

            # MLA config
            config.use_mla = True
            config.mla_ops_type = MlaOpsType.__members__[os.environ.get('MLA_OPS_TYPE', 'AUTO')]
            logging.info(f"deepseek2 mla_ops_type: {config.mla_ops_type.name}")
            config.q_lora_rank = config_json['q_lora_rank']
            config.kv_lora_rank = config_json['kv_lora_rank']
            config.nope_head_dim = config_json['qk_nope_head_dim']
            config.rope_head_dim = config_json['qk_rope_head_dim']
            config.v_head_dim = config_json['v_head_dim']
            config.size_per_head = config.nope_head_dim + config.rope_head_dim
            config.rotary_embedding_dim = config.rope_head_dim

            # yarn rotary config
            if config.mla_ops_type != MlaOpsType.MHA:
                config.rotary_embedding_style = 0
            else:
                config.rotary_embedding_style = 5
            rope_scaling = config_json.get('rope_scaling')
            config.rotary_embedding_scale = rope_scaling['factor']
            config.rotary_factor1 = float(rope_scaling.get('beta_slow', 1))
            config.rotary_factor2 = float(rope_scaling.get('beta_fast', 32))
            config.org_embedding_max_pos = rope_scaling['original_max_position_embeddings']

            scaling_factor = rope_scaling['factor']
            mscale = rope_scaling['mscale']
            mscale_all_dim = rope_scaling['mscale_all_dim']
            config.deepseek_rope_mscale = mscale
            config.deepseek_mscale_all_dim = mscale_all_dim
            config.rotary_embedding_mscale = yarn_get_mscale(scaling_factor, mscale) / yarn_get_mscale(scaling_factor, mscale_all_dim)
            config.rotary_embedding_offset = config.nope_head_dim

            # softmax scale config
            softmax_mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
            config.softmax_extra_scale = softmax_mscale * softmax_mscale

            # MOE config
            if "scoring_func" in config_json:
                scoring_func = config_json['scoring_func']
                if scoring_func == "softmax":
                    config.scoring_func = 0
                elif scoring_func == "sigmoid":
                    config.scoring_func = 1
                else:
                    raise ValueError(f"Unknown scoring_func: {scoring_func}")

            config.routed_scaling_factor = config_json['routed_scaling_factor']
            config.moe_k = config_json['num_experts_per_tok']
            config.expert_num = config_json['n_routed_experts']
            config.moe_inter_padding_size=config_json['moe_intermediate_size']
            config.moe_n_group = config_json.get('n_group', 1)
            config.moe_topk_group = config_json.get('topk_group', 1)

            n_shared_experts = config_json['n_shared_experts']
            config.inter_size = n_shared_experts * config.moe_inter_padding_size

            config.layernorm_eps = config_json.get("rms_norm_eps", 1e-06)
            config.has_moe_norm = config_json.get("norm_topk_prob", False)
            config.moe_style = 2 # shared + expert

            moe_step = config_json['moe_layer_freq']
            first_k_dense_replace = config_json['first_k_dense_replace']
            config.moe_layer_index = [i for i in range(config.layer_num) if i >= first_k_dense_replace and i % moe_step == 0]

            ffn_inter_size = config_json.get('intermediate_size', config.inter_size)
            layer_inter_size = []
            for i in range(config.layer_num):
                if i in config.moe_layer_index:
                    layer_inter_size.append(config.inter_size)
                else:
                    layer_inter_size.append(ffn_inter_size)
            config.layer_inter_size = layer_inter_size

    @staticmethod
    def get_weight_cls():
        return DeepSeekV2Weight


    def _initialize_rope(self):
        if self.config.mla_ops_type == MlaOpsType.MHA:
            return
        assert self.weight
        config = self.config
        logging.info(f"initialize rope cos sin cache with seq_len: {config.max_seq_len}")
        rotary_emb = DeepseekV3YarnRotaryEmbedding(config.rotary_embedding_dim,
                                                   config.max_seq_len,
                                                   config.rotary_embedding_base,
                                                   scaling_factor=config.rotary_embedding_scale,
                                                   original_max_position_embeddings=config.org_embedding_max_pos,
                                                   beta_fast=config.rotary_factor2,
                                                   beta_slow=config.rotary_factor1,
                                                   mscale=config.deepseek_rope_mscale,
                                                   mscale_all_dim=config.deepseek_mscale_all_dim)
        half_rope_dim = config.rotary_embedding_dim // 2
        cos_cache = rotary_emb.cos_cached[:, :half_rope_dim]
        sin_cache = rotary_emb.sin_cached[:, :half_rope_dim]
        # cos sin cache must be float32
        cos_sin_cache = torch.cat([cos_cache, sin_cache], dim=-1).contiguous().to(self.device).to(torch.float32)
        self.weight.global_weights[W.rope_cos_sin_cache] = cos_sin_cache

class DeepSeekV3MtpWeight(DeepSeekV2Weight):

    def __init__(self, config: GptInitModelParameters, tp_size: int, tp_rank: int):
        super().__init__(config, tp_size, tp_rank)

    def _get_weight_info(self):
        layer_weights: List[List[WeightModule]] = []
        weights = [
            AtomicWeight(W.embedding, [CkptWeightInfo('model.layers.0.embed_tokens.weight', identity)], identity),
            AtomicWeight(W.lm_head, [CkptWeightInfo('model.layers.0.shared_head.head.weight', identity)], identity)
        ]
        assert self._num_layers == 1
        for layer in range(self._num_layers):
            if self._quant_algo.isFp8():
                layer_weights_tmp = self._get_fp8_layer_weight_info(layer)
            else:
                layer_weights_tmp = self._get_hf_layer_weight_info(layer)
            layer_weights_tmp.extend([
                AtomicWeight(W.multi_tokens_predict_final_ln_gamma, [CkptWeightInfo('model.layers.{i}.shared_head.norm.weight', identity)], identity),
                AtomicWeight(W.multi_tokens_predict_final_ln_beta, [], functools.partial(zeros, shape=[self._hidden_size])),
                AtomicWeight(W.multi_tokens_predict_enorm, [CkptWeightInfo('model.layers.{i}.enorm.weight', identity)], identity),
                AtomicWeight(W.multi_tokens_predict_hnorm, [CkptWeightInfo('model.layers.{i}.hnorm.weight', identity)], identity),
                AtomicWeight(W.multi_tokens_predict_eh_proj, [CkptWeightInfo('model.layers.{i}.eh_proj.weight', identity)], transpose),
            ])
            layer_weights.append(layer_weights_tmp)

        return ModelWeightInfo(layer_weights=layer_weights, weights=weights)

class DeepSeekV3Mtp(DeepSeekV2):

    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = super()._create_config(ckpt_path)
        config.moe_layer_index = [i for i in range(config.layer_num)]
        config.reverse_e_h_norm = True
        config.is_mtp = True
        return config

    @staticmethod
    def get_weight_cls():
        return DeepSeekV3MtpWeight


register_model('deepseek2', DeepSeekV2, ["DeepseekV2ForCausalLM"])
register_model('deepseek3', DeepSeekV2, ["DeepseekV3ForCausalLM"])
register_model("deepseek-v3-mtp", DeepSeekV3Mtp, ["DeepseekV3ForCausalLMNextN"])
