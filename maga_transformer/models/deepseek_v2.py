import logging
import torch
import unicodedata
import types
import functools
import math
import os
import json
from typing import List, Any

from maga_transformer.eplb.ep_balancer import MoeWeightInfo
from maga_transformer.models.base_model import BaseModel
from maga_transformer.utils.model_weight import (
    W,
    ModelDeployWeightInfo,
    ModelWeightInfo,
    WeightInfo,
    CkptWeightInfo,
    identity,
    transpose,
    stack_,
    w_half1,
    concat_0,
    w_half2,
    zeros,
    pad,
    transpose_pad,
    multipy_identity,
    stack_moe_w1
)
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters, MlaOpsType
from maga_transformer.model_factory_register import register_model
from maga_transformer.models.rotary_embedding.deepseek_rotary_embedding import DeepseekV3YarnRotaryEmbedding

def yarn_get_mscale(scale: float=1, mscale: float=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def kv_split(ts: List[torch.Tensor], kv_lora_rank: int, nope_head_dim: int, v_head_dim: int, idx: int):
    res_list = (
        ts[0]
        .reshape(-1, nope_head_dim + v_head_dim, kv_lora_rank)
        .split([nope_head_dim, v_head_dim], dim=1)
    )
    res = res_list[idx]
    res = res.reshape(-1, kv_lora_rank)
    # [head_num*head_dim, lora_rank]
    return res.contiguous()

def k_split_for_group_gemm(ts: List[torch.Tensor], kv_lora_rank: int, nope_head_dim: int, v_head_dim: int, idx: int):
    res_list = (
        ts[0]
        .reshape(-1, nope_head_dim + v_head_dim, kv_lora_rank)
        .split([nope_head_dim, v_head_dim], dim=1)
    )
    res = res_list[idx]
    res = res.permute(0, 2, 1)
    # [head_num, head_dim, lora_rank]
    return res.contiguous()

def v_split_for_group_gemm(ts: List[torch.Tensor], kv_lora_rank: int, nope_head_dim: int, v_head_dim: int, idx: int):
    res_list = (
        ts[0]
        .reshape(-1, nope_head_dim + v_head_dim, kv_lora_rank)
        .split([nope_head_dim, v_head_dim], dim=1)
    )
    res = res_list[idx]
    # res = res.permute(0, 2, 1)
    # [head_num,  lora_rank, head_dim]
    return res.contiguous()


def kv_split1(
    ts: List[torch.Tensor], kv_lora_rank: int, nope_head_dim: int, v_head_dim: int
) -> torch.Tensor:
    k, _ = (
        ts[0]
        .transpose(0, 1)
        .reshape(kv_lora_rank, -1, nope_head_dim + v_head_dim)
        .split([nope_head_dim, v_head_dim], dim=-1)
    )
    k = k.reshape(kv_lora_rank, -1)
    # [lora_rank, head_num * head_dim]
    return k.contiguous()


def kv_split2(
    ts: List[torch.Tensor], kv_lora_rank: int, nope_head_dim: int, v_head_dim: int
) -> torch.Tensor:
    _, v = (
        ts[0]
        .transpose(0, 1)
        .reshape(kv_lora_rank, -1, nope_head_dim + v_head_dim)
        .split([nope_head_dim, v_head_dim], dim=-1)
    )
    v = v.reshape(kv_lora_rank, -1)
    return v.contiguous()

def mla_pad(ts: List[torch.Tensor], head_num: int, nope_head_dim: int, rope_head_dim: int) -> torch.Tensor:
    t = ts[0]
    t = t.reshape(-1, head_num, nope_head_dim)
    z = torch.zeros(t.shape[0], head_num, rope_head_dim, device=t.device, dtype=t.dtype)
    t = torch.cat([t, z], dim=-1)
    t = t.reshape(-1, head_num * (nope_head_dim + rope_head_dim))
    return t.contiguous()

def mla_pad_t(ts: List[torch.Tensor], head_num: int, nope_head_dim: int, rope_head_dim: int) -> torch.Tensor:
    t = ts[0]
    t = t.reshape(-1, head_num, nope_head_dim)
    z = torch.zeros(t.shape[0], head_num, rope_head_dim, device=t.device, dtype=t.dtype)
    t = torch.cat([t, z], dim=-1)
    t = t.reshape(-1, head_num * (nope_head_dim + rope_head_dim))
    return t.T.contiguous()

def transpose_slice_k(ts: List[torch.Tensor], head_num: int, nope_head_dim: int, v_head_dim: int, lora_rank: int) -> torch.Tensor:
    t = ts[0]
    t = t.transpose(0, 1).view(lora_rank, head_num, nope_head_dim + v_head_dim)
    return t[:, :, :nope_head_dim].permute(1, 2, 0).contiguous()

def transpose_slice_v(ts: List[torch.Tensor], head_num: int, nope_head_dim: int, v_head_dim: int, lora_rank: int) -> torch.Tensor:
    t = ts[0]
    t = t.transpose(0, 1).view(lora_rank, head_num, nope_head_dim + v_head_dim)
    return t[:, :, nope_head_dim:].transpose(0, 1).contiguous()

def mla_pad_scale(ts: List[torch.Tensor], head_num: int, nope_head_dim: int, rope_head_dim: int, group_size: int) -> torch.Tensor:
    t = ts[0]
    t = t.reshape(-1, head_num* nope_head_dim // group_size)
    z = torch.zeros(t.shape[0], head_num* rope_head_dim // group_size, device=t.device, dtype=t.dtype)
    t = torch.cat([t, z], dim=-1)
    t = t.reshape(-1, head_num * (nope_head_dim + rope_head_dim) // group_size)
    return  t.contiguous()

def concat_0_tranpose(ts: List[torch.Tensor]):
    return torch.concat(ts, dim=0).transpose(0, 1).contiguous()

def dequant_weight_split_k(ts: List[torch.Tensor], block_size: int, head_num: int, nope_head_dim: int, v_head_dim: int, lora_rank: int) -> torch.Tensor:
    from maga_transformer.models.deepseek_dequant import weight_dequant
    return transpose_slice_k([weight_dequant(ts[0], ts[1], block_size)],
                             head_num, nope_head_dim, v_head_dim, lora_rank)

def dequant_weight_split_v(ts: List[torch.Tensor], block_size: int, head_num: int, nope_head_dim: int, v_head_dim: int, lora_rank: int) -> torch.Tensor:
    from maga_transformer.models.deepseek_dequant import weight_dequant
    return transpose_slice_v([weight_dequant(ts[0], ts[1], block_size)],
                             head_num, nope_head_dim, v_head_dim, lora_rank)

def transpose_kv_rope(ts: List[torch.Tensor], kv_lora_rank: int, rope_size: int):
    rope_size_half = rope_size // 2
    kva = ts[0]
    kva[kv_lora_rank: , :]  = kva[kv_lora_rank: , :].reshape([rope_size_half, 2, -1]).transpose(0, 1).reshape([rope_size, -1])
    return kva.reshape(ts[0].shape).contiguous()

def transpose_q_rope(ts: List[torch.Tensor], head_num: int, nope_head_dim: int, rope_size: int):
    rope_size_half = rope_size // 2
    q = ts[0]
    q = q.reshape([head_num, nope_head_dim + rope_size, -1])
    q[:, nope_head_dim: , :] = q[:, nope_head_dim: , :].reshape([head_num, rope_size_half, 2, -1]).transpose(1, 2).reshape([head_num, rope_size, -1])
    return q.reshape(ts[0].shape).contiguous()

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
        layer_weights = [
            WeightInfo(W.pre_ln_gamma, [CkptWeightInfo('model.layers.{i}.input_layernorm.weight', identity)],
                       identity),
            WeightInfo(W.attn_o_w, [CkptWeightInfo('model.layers.{i}.self_attn.o_proj.weight', identity)],
                       functools.partial(mla_pad_t, head_num=self._head_num, nope_head_dim=self.nope_head_dim, rope_head_dim=0)),
            WeightInfo(W.post_ln_gamma, [CkptWeightInfo('model.layers.{i}.post_attention_layernorm.weight', identity)],
                       identity),
        ]
        mla_layer_weights = [
            WeightInfo(W.mla_k_nope_w, [CkptWeightInfo('model.layers.{i}.self_attn.kv_b_proj.weight', identity)],
                       functools.partial(kv_split1, kv_lora_rank=self.kv_lora_rank, nope_head_dim=self.nope_head_dim, v_head_dim=self.v_head_dim)),
            WeightInfo(W.mla_v_w, [CkptWeightInfo('model.layers.{i}.self_attn.kv_b_proj.weight', identity)],
                       functools.partial(kv_split2, kv_lora_rank=self.kv_lora_rank, nope_head_dim=self.nope_head_dim, v_head_dim=self.v_head_dim)),
            WeightInfo(W.mla_kv_a_ln_gamma, [CkptWeightInfo('model.layers.{i}.self_attn.kv_a_layernorm.weight', identity)],
                       identity),
        ]

        if self.q_use_lora:
            mla_layer_weights.extend([
                WeightInfo(W.mla_q_b_w, [CkptWeightInfo('model.layers.{i}.self_attn.q_b_proj.weight', functools.partial(transpose_q_rope, head_num=self._head_num, nope_head_dim=self.nope_head_dim, rope_size=self.rope_head_dim))],
                        transpose),
                WeightInfo(W.mla_q_a_ln_gamma, [CkptWeightInfo('model.layers.{i}.self_attn.q_a_layernorm.weight')],
                        identity)
            ])
            q_a_weight = CkptWeightInfo('model.layers.{i}.self_attn.q_a_proj.weight')
            mla_layer_weights.append(
                WeightInfo(W.mla_fusedqkrope_w, [q_a_weight, CkptWeightInfo('model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight', functools.partial(transpose_kv_rope, kv_lora_rank=self.kv_lora_rank, rope_size=self.rope_head_dim))], concat_0_tranpose)
            )
        else:
            q_a_weight = CkptWeightInfo('model.layers.{i}.self_attn.q_proj.weight', functools.partial(transpose_q_rope, head_num=self._head_num, nope_head_dim=self.nope_head_dim, rope_size=self.rope_head_dim))
            mla_layer_weights.append(
                WeightInfo(W.mla_fusedqkrope_no_lora_w, [q_a_weight, CkptWeightInfo('model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight', functools.partial(transpose_kv_rope, kv_lora_rank=self.kv_lora_rank, rope_size=self.rope_head_dim))], concat_0_tranpose)
            )

        if self.config.use_mla and self.config.mla_ops_type != MlaOpsType.MHA:
            mla_layer_weights.append(
                WeightInfo(W.mla_kc, [CkptWeightInfo('model.layers.{i}.self_attn.kv_b_proj.weight', identity)],
                           functools.partial(transpose_slice_k, head_num=self._head_num, nope_head_dim=self.nope_head_dim, v_head_dim=self.v_head_dim, lora_rank=self.kv_lora_rank)))
            mla_layer_weights.append(
                WeightInfo(W.mla_vc, [CkptWeightInfo('model.layers.{i}.self_attn.kv_b_proj.weight', identity)],
                           functools.partial(transpose_slice_v, head_num=self._head_num, nope_head_dim=self.nope_head_dim, v_head_dim=self.v_head_dim, lora_rank=self.kv_lora_rank)))

        layer_weights.extend(mla_layer_weights)
        layer_weights.extend(self._get_hf_ffn_layer_weight_info(layer_id))
        return layer_weights


    def _get_fp8_layer_weight_info(self, layer_id: int):
        group_size = self._quant_algo.getGroupSize()
        layer_weights = [
            WeightInfo(W.pre_ln_gamma, [CkptWeightInfo('model.layers.{i}.input_layernorm.weight', identity)],
                       identity),
            WeightInfo(W.attn_o_w, [CkptWeightInfo('model.layers.{i}.self_attn.o_proj.weight', identity)],
                       functools.partial(mla_pad, head_num=self._head_num, nope_head_dim=self.nope_head_dim, rope_head_dim=0)),
            WeightInfo(W.attn_o_s, [CkptWeightInfo('model.layers.{i}.self_attn.o_proj.weight_scale_inv', identity)],
                       functools.partial(mla_pad_scale, head_num=self._head_num, nope_head_dim=self.nope_head_dim, rope_head_dim=0, group_size=group_size)),
            WeightInfo(W.post_ln_gamma, [CkptWeightInfo('model.layers.{i}.post_attention_layernorm.weight', identity)],
                       identity),
        ]
        mla_layer_weights = [
            WeightInfo(W.mla_k_nope_w, [CkptWeightInfo('model.layers.{i}.self_attn.kv_b_proj.weight', identity)],
                       functools.partial(kv_split, kv_lora_rank=self.kv_lora_rank, nope_head_dim=self.nope_head_dim, v_head_dim=self.v_head_dim, idx=0)),
            WeightInfo(W.mla_k_nope_s, [CkptWeightInfo('model.layers.{i}.self_attn.kv_b_proj.weight_scale_inv', identity)],
                       functools.partial(kv_split, kv_lora_rank=self.kv_lora_rank // group_size, nope_head_dim=self.nope_head_dim // group_size, v_head_dim=self.v_head_dim // group_size, idx=0)),

            WeightInfo(W.mla_v_w, [CkptWeightInfo('model.layers.{i}.self_attn.kv_b_proj.weight', identity)],
                       functools.partial(kv_split, kv_lora_rank=self.kv_lora_rank, nope_head_dim=self.nope_head_dim, v_head_dim=self.v_head_dim, idx=1)),
            WeightInfo(W.mla_v_s, [CkptWeightInfo('model.layers.{i}.self_attn.kv_b_proj.weight_scale_inv', identity)],
                       functools.partial(kv_split, kv_lora_rank=self.kv_lora_rank // group_size, nope_head_dim=self.nope_head_dim // group_size, v_head_dim=self.v_head_dim // group_size, idx=1)),

            WeightInfo(W.mla_kv_a_ln_gamma, [CkptWeightInfo('model.layers.{i}.self_attn.kv_a_layernorm.weight', identity)],
                       identity),
        ]

        if self.config.use_mla and self.config.mla_ops_type != MlaOpsType.MHA:
            mla_layer_weights.append(
                WeightInfo(W.mla_kc, [CkptWeightInfo('model.layers.{i}.self_attn.kv_b_proj.weight', identity), CkptWeightInfo('model.layers.{i}.self_attn.kv_b_proj.weight_scale_inv', identity)],
                           functools.partial(dequant_weight_split_k, block_size=128, head_num=self._head_num, nope_head_dim=self.nope_head_dim, v_head_dim=self.v_head_dim, lora_rank=self.kv_lora_rank)))

            mla_layer_weights.append(
                WeightInfo(W.mla_vc, [CkptWeightInfo('model.layers.{i}.self_attn.kv_b_proj.weight', identity), CkptWeightInfo('model.layers.{i}.self_attn.kv_b_proj.weight_scale_inv', identity)],
                           functools.partial(dequant_weight_split_v, block_size=128, head_num=self._head_num, nope_head_dim=self.nope_head_dim, v_head_dim=self.v_head_dim, lora_rank=self.kv_lora_rank)))

        if not self.q_use_lora:
            raise Exception("fp8 only support q_use_lora")
        mla_layer_weights.extend([
            WeightInfo(W.mla_q_b_w, [CkptWeightInfo('model.layers.{i}.self_attn.q_b_proj.weight', functools.partial(transpose_q_rope, head_num=self._head_num, nope_head_dim=self.nope_head_dim, rope_size=self.rope_head_dim))],
                    identity),
            WeightInfo(W.mla_q_b_s, [CkptWeightInfo('model.layers.{i}.self_attn.q_b_proj.weight_scale_inv', identity)],
                    identity),
            WeightInfo(W.mla_q_a_ln_gamma, [CkptWeightInfo('model.layers.{i}.self_attn.q_a_layernorm.weight', identity)],
                    identity),
            WeightInfo(W.mla_fusedqkrope_w, [
                CkptWeightInfo('model.layers.{i}.self_attn.q_a_proj.weight', identity),
                CkptWeightInfo('model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight', functools.partial(transpose_kv_rope, kv_lora_rank=self.kv_lora_rank, rope_size=self.rope_head_dim))], concat_0),
            WeightInfo(W.mla_fusedqkrope_s, [
                CkptWeightInfo('model.layers.{i}.self_attn.q_a_proj.weight_scale_inv', identity),
                CkptWeightInfo('model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight_scale_inv', identity)], concat_0)
            ]
        )

        layer_weights.extend(mla_layer_weights)
        layer_weights.extend(self._get_fp8_ffn_layer_weight_info(layer_id))
        return layer_weights


    def _get_hf_ffn_layer_weight_info(self, layer_id: int):
        inter_padding_size = self._layer_inter_padding_size[layer_id] if self._layer_inter_padding_size else self._inter_padding_size

        selected_experts = self._get_selected_experts(layer_id)

        if layer_id in self.moe_layer_index_:
            layer_weights = [
                WeightInfo(W.moe_gate, [CkptWeightInfo('model.layers.{i}.mlp.gate.weight', identity)], transpose),
                WeightInfo(W.ffn_w1, [CkptWeightInfo('model.layers.{i}.mlp.shared_experts.gate_proj.weight', identity)], functools.partial(transpose_pad, inter_padding_size=inter_padding_size, dim=0)),
                WeightInfo(W.ffn_w2, [CkptWeightInfo('model.layers.{i}.mlp.shared_experts.down_proj.weight', identity)], functools.partial(transpose_pad, inter_padding_size=inter_padding_size, dim=1)),
                WeightInfo(W.ffn_w3, [CkptWeightInfo('model.layers.{i}.mlp.shared_experts.up_proj.weight', identity)], functools.partial(transpose_pad, inter_padding_size=inter_padding_size, dim=0)),
                WeightInfo(W.moe_w2, [CkptWeightInfo('model.layers.{i}.mlp.experts.' + str(expert_id) + '.down_proj.weight',
                                                     functools.partial(multipy_identity, scale=self.routed_scaling_factor)) \
                                        for expert_id in selected_experts], stack_),
                WeightInfo(W.moe_w1, [CkptWeightInfo('model.layers.{i}.mlp.experts.' + str(expert_id) + '.up_proj.weight', identity) for expert_id in selected_experts] + \
                                     [CkptWeightInfo('model.layers.{i}.mlp.experts.' + str(expert_id) + '.gate_proj.weight', identity) for expert_id in selected_experts], stack_moe_w1),
            ]
            if self.has_e_score_correction_bias:
                layer_weights.append(WeightInfo(W.e_score_correction_b, [CkptWeightInfo('model.layers.{i}.mlp.gate.e_score_correction_bias', identity)], identity))
            return layer_weights
        else:
            return [
                WeightInfo(W.ffn_w1, [CkptWeightInfo('model.layers.{i}.mlp.gate_proj.weight', identity)],
                           functools.partial(transpose_pad, inter_padding_size=inter_padding_size, dim=0)),
                WeightInfo(W.ffn_w2, [CkptWeightInfo('model.layers.{i}.mlp.down_proj.weight', identity)],
                           functools.partial(transpose_pad, inter_padding_size=inter_padding_size, dim=1)),
                WeightInfo(W.ffn_w3, [CkptWeightInfo('model.layers.{i}.mlp.up_proj.weight', identity)],
                           functools.partial(transpose_pad, inter_padding_size=inter_padding_size, dim=0)),
            ]

    def _get_fp8_ffn_layer_weight_info(self, layer_id: int):
        inter_padding_size = self._layer_inter_padding_size[layer_id] if self._layer_inter_padding_size else self._inter_padding_size
        group_size = self._quant_algo.getGroupSize()
        selected_experts = self._get_selected_experts(layer_id)

        if layer_id in self.moe_layer_index_:
            layer_weights = [
                WeightInfo(W.moe_gate, [CkptWeightInfo('model.layers.{i}.mlp.gate.weight', identity)], transpose),
                WeightInfo(W.ffn_w1, [CkptWeightInfo('model.layers.{i}.mlp.shared_experts.gate_proj.weight', identity)], functools.partial(pad, inter_padding_size=inter_padding_size, dim=0)),
                WeightInfo(W.ffn_s1, [CkptWeightInfo('model.layers.{i}.mlp.shared_experts.gate_proj.weight_scale_inv', identity)], functools.partial(pad, inter_padding_size=inter_padding_size // group_size, dim=0)),

                WeightInfo(W.ffn_w2, [CkptWeightInfo('model.layers.{i}.mlp.shared_experts.down_proj.weight', identity)], functools.partial(pad, inter_padding_size=inter_padding_size, dim=1)),
                WeightInfo(W.ffn_s2, [CkptWeightInfo('model.layers.{i}.mlp.shared_experts.down_proj.weight_scale_inv', identity)], functools.partial(pad, inter_padding_size=inter_padding_size // group_size, dim=1)),

                WeightInfo(W.ffn_w3, [CkptWeightInfo('model.layers.{i}.mlp.shared_experts.up_proj.weight', identity)], functools.partial(pad, inter_padding_size=inter_padding_size, dim=0)),
                WeightInfo(W.ffn_s3, [CkptWeightInfo('model.layers.{i}.mlp.shared_experts.up_proj.weight_scale_inv', identity)], functools.partial(pad, inter_padding_size=inter_padding_size // group_size, dim=0)),

                WeightInfo(W.moe_w2, [CkptWeightInfo('model.layers.{i}.mlp.experts.' + str(expert_id) + '.down_proj.weight',
                                                     identity) \
                                        for expert_id in selected_experts], stack_),
                WeightInfo(W.moe_s2, [CkptWeightInfo('model.layers.{i}.mlp.experts.' + str(expert_id) + '.down_proj.weight_scale_inv',
                                        functools.partial(multipy_identity, scale=self.routed_scaling_factor)) \
                        for expert_id in selected_experts], stack_),

                WeightInfo(W.moe_w1, [CkptWeightInfo('model.layers.{i}.mlp.experts.' + str(expert_id) + '.up_proj.weight', identity) for expert_id in selected_experts] + \
                                     [CkptWeightInfo('model.layers.{i}.mlp.experts.' + str(expert_id) + '.gate_proj.weight', identity) for expert_id in selected_experts], stack_moe_w1),
                WeightInfo(W.moe_s1, [CkptWeightInfo('model.layers.{i}.mlp.experts.' + str(expert_id) + '.up_proj.weight_scale_inv', identity) for expert_id in selected_experts] + \
                                     [CkptWeightInfo('model.layers.{i}.mlp.experts.' + str(expert_id) + '.gate_proj.weight_scale_inv', identity) for expert_id in selected_experts], stack_moe_w1),
            ]
            if self.has_e_score_correction_bias:
                layer_weights.append(WeightInfo(W.e_score_correction_b, [CkptWeightInfo('model.layers.{i}.mlp.gate.e_score_correction_bias', identity)], identity))
            return layer_weights
        else:
            return [
                WeightInfo(W.ffn_w1, [CkptWeightInfo('model.layers.{i}.mlp.gate_proj.weight', identity)],
                           functools.partial(pad, inter_padding_size=inter_padding_size, dim=0)),
                WeightInfo(W.ffn_s1, [CkptWeightInfo('model.layers.{i}.mlp.gate_proj.weight_scale_inv', identity)],
                           functools.partial(pad, inter_padding_size=inter_padding_size // group_size, dim=0)),

                WeightInfo(W.ffn_w2, [CkptWeightInfo('model.layers.{i}.mlp.down_proj.weight', identity)],
                           functools.partial(pad, inter_padding_size=inter_padding_size, dim=1)),
                WeightInfo(W.ffn_s2, [CkptWeightInfo('model.layers.{i}.mlp.down_proj.weight_scale_inv', identity)],
                           functools.partial(pad, inter_padding_size=inter_padding_size // group_size, dim=1)),

                WeightInfo(W.ffn_w3, [CkptWeightInfo('model.layers.{i}.mlp.up_proj.weight', identity)],
                           functools.partial(pad, inter_padding_size=inter_padding_size, dim=0)),
                WeightInfo(W.ffn_s3, [CkptWeightInfo('model.layers.{i}.mlp.up_proj.weight_scale_inv', identity)],
                           functools.partial(pad, inter_padding_size=inter_padding_size // group_size, dim=0)),
            ]

    def _get_weight_info(self):
        layer_weights: List[List[WeightInfo]] = []
        weights = [
            WeightInfo(W.embedding, [CkptWeightInfo('model.embed_tokens.weight', identity)], identity),
            WeightInfo(W.final_ln_gamma, [CkptWeightInfo('model.norm.weight', identity)], identity),
            WeightInfo(W.final_ln_beta, [], functools.partial(zeros, shape=[self._hidden_size])),
            WeightInfo(W.lm_head, [CkptWeightInfo('lm_head.weight', identity)], identity),
        ]
        for layer in range(self._num_layers):
            if self._quant_algo.isFp8():
                layer_weights.append(self._get_fp8_layer_weight_info(layer))
            else:
                layer_weights.append(self._get_hf_layer_weight_info(layer))
        return ModelWeightInfo(layer_weights=layer_weights, weights=weights, tp_strategy=self._get_gpt_style_tp_strategy())


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

    def create_moe_weight_info(self):
        if self.config.quant_algo.isFp8():
            gate = CkptWeightInfo("model.layers.{}.mlp.experts.{}.gate_proj.weight", identity)
            up = CkptWeightInfo("model.layers.{}.mlp.experts.{}.up_proj.weight", identity)
            down = CkptWeightInfo("model.layers.{}.mlp.experts.{}.down_proj.weight", identity)

            gate_s = CkptWeightInfo('model.layers.{}.mlp.experts.{}.gate_proj.weight_scale_inv', identity)
            up_s = CkptWeightInfo('model.layers.{}.mlp.experts.{}.up_proj.weight_scale_inv', identity)
            down_s = CkptWeightInfo('model.layers.{}.mlp.experts.{}.down_proj.weight_scale_inv', functools.partial(multipy_identity, scale=self.config.routed_scaling_factor))

            return MoeWeightInfo(gate, up, down, True, gate_s, up_s, down_s)
        else:
            gate = CkptWeightInfo("model.layers.{}.mlp.experts.{}.gate_proj.weight", identity)
            up = CkptWeightInfo("model.layers.{}.mlp.experts.{}.up_proj.weight", identity)
            down = CkptWeightInfo("model.layers.{}.mlp.experts.{}.down_proj.weight", functools.partial(multipy_identity, scale=self.config.routed_scaling_factor))
            return MoeWeightInfo(gate, up, down)

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
        layer_weights: List[List[WeightInfo]] = []
        weights = [
            WeightInfo(W.embedding, [CkptWeightInfo('model.layers.0.embed_tokens.weight', identity)], identity),
            WeightInfo(W.lm_head, [CkptWeightInfo('model.layers.0.shared_head.head.weight', identity)], identity)
        ]
        assert self._num_layers == 1
        for layer in range(self._num_layers):
            if self._quant_algo.isFp8():
                layer_weights_tmp = self._get_fp8_layer_weight_info(layer)
            else:
                layer_weights_tmp = self._get_hf_layer_weight_info(layer)
            layer_weights_tmp.extend([
                WeightInfo(W.multi_tokens_predict_final_ln_gamma, [CkptWeightInfo('model.layers.{i}.shared_head.norm.weight', identity)], identity),
                WeightInfo(W.multi_tokens_predict_final_ln_beta, [], functools.partial(zeros, shape=[self._hidden_size])),
                WeightInfo(W.multi_tokens_predict_enorm, [CkptWeightInfo('model.layers.{i}.enorm.weight', identity)], identity),
                WeightInfo(W.multi_tokens_predict_hnorm, [CkptWeightInfo('model.layers.{i}.hnorm.weight', identity)], identity),
                WeightInfo(W.multi_tokens_predict_eh_proj, [CkptWeightInfo('model.layers.{i}.eh_proj.weight', identity)], transpose),
            ])
            layer_weights.append(layer_weights_tmp)

        return ModelWeightInfo(layer_weights=layer_weights, weights=weights, tp_strategy=self._get_gpt_style_tp_strategy())

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
