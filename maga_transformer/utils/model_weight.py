import logging
import math
import os
from functools import reduce
import threading
import torch
import torch.serialization
import functools
import copy
from enum import Enum
from typing import Any, NamedTuple, Callable, List, Dict, Set, Tuple, Optional, Union
from maga_transformer.utils.database import FinetuneType, TrainType, CkptFileInfo
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.distribute.worker_info import g_parallel_info
from maga_transformer.utils.database import BaseDatabase

def w_half1(ts: List[torch.Tensor], inter_size: int):
    return ts[0][:inter_size, ...].T.contiguous()

def w_half2(ts: List[torch.Tensor], inter_size: int):
    return ts[0][inter_size:, ...].T.contiguous()

def concat_0(ts: List[torch.Tensor]) -> torch.Tensor:
    if len(ts) == 1:
        return ts[0]

    return torch.concat(ts, dim=0).contiguous()

def concat_1(ts: List[torch.Tensor]) -> torch.Tensor:
    if len(ts) == 1:
        return ts[0]
    return torch.concat(ts, dim=1).contiguous()

def pad(ts: List[torch.Tensor], inter_padding_size: int, dim: int):
    if dim == 0:
        pad_shape = [inter_padding_size - ts[0].shape[0], ts[0].shape[1]]
    elif dim == 1:
        pad_shape = [ts[0].shape[0], inter_padding_size - ts[0].shape[1]]
    else:
        raise Exception('unknown padding dim: ' + str(dim))
    if pad_shape[0] == 0 or pad_shape[1] == 0:
        return ts[0].contiguous()
    z = torch.zeros(pad_shape, device=ts[0].device).to(ts[0].dtype)
    return torch.cat((ts[0], z), dim).to(ts[0].device).contiguous()

def transpose_pad(ts: List[torch.Tensor], inter_padding_size: int, dim: int):
    if dim == 0:
        pad_shape = [inter_padding_size - ts[0].shape[0], ts[0].shape[1]]
    elif dim == 1:
        pad_shape = [ts[0].shape[0], inter_padding_size - ts[0].shape[1]]
    else:
        raise Exception('unknown padding dim: ' + str(dim))
    z = torch.zeros(pad_shape, device=ts[0].device).half()
    return torch.cat((ts[0], z), dim).T.to(ts[0].device).contiguous()

def b_half_merge(ts: List[torch.Tensor]):
    n_ts_1 = []
    n_ts_2 = []
    for t in ts:
        t_a = t.chunk(2, dim=-1)
        n_ts_1.append(t_a[0].cuda())
        n_ts_2.append(t_a[1].cuda())
    return concat_0([concat_0(n_ts_1), concat_0(n_ts_2)])

def zeros(ts: List[torch.Tensor], shape: List[int]) -> torch.Tensor:
    return torch.zeros(shape, dtype=torch.half).contiguous()

def ones(ts: List[torch.Tensor], shape: List[int] = [1]) -> torch.Tensor:
    return torch.ones(shape, dtype=torch.half).contiguous()

def transpose(ts: List[torch.Tensor]) -> torch.Tensor:
    return ts[0].t().contiguous()

def identity(ts: List[torch.Tensor], allow_empty:bool = False) -> torch.Tensor:
    if len(ts) == 0:
        if allow_empty:
            return None
        else:
            raise Exception("ts is empty")
    return ts[0].contiguous()

def multipy_identity(ts: List[torch.Tensor], scale: float) -> torch.Tensor:
    t = identity(ts)
    return t * scale

def div(ts: List[torch.Tensor], allow_empty:bool = False) -> torch.Tensor:
    if len(ts) == 0:
        if allow_empty:
            return None
        else:
            raise Exception("ts is empty")
    return (1.0/ts[0]).to(torch.float32).contiguous()
    #return (torch.tensor(1.0, dtype=torch.float32, device="cuda:0")).contiguous()

def get_tensor_reciprocal(ts: List[torch.Tensor]) -> torch.Tensor:
    return 1.0 / ts[0].reshape(-1)

def get_tensor_from_scalar(ts: List[torch.Tensor]) -> torch.Tensor:
    return ts[0].reshape(-1)

def tolerate_failed(ts: List[torch.Tensor], origin_func: Callable[[List[torch.Tensor]], torch.Tensor]) -> torch.Tensor:
    try:
        return origin_func(ts)
    except Exception as _:
        return None

def choose_available(ts: List[Optional[torch.Tensor]], origin_func_list: List[Callable[[List[torch.Tensor]], torch.Tensor]]) -> torch.Tensor:
    for t, func in zip(ts, origin_func_list):
        if t is not None and len(ts) > 0:
            return func([t])
    raise ValueError(f"all tensor is empty, but not allow empty")


def shift_one(ts: List[torch.Tensor], allow_empty: bool = False) -> torch.Tensor:
    if len(ts) == 0 and allow_empty:
        return None
    return (ts[0] + 1.0).contiguous()

def sp_0(t: torch.Tensor, tp: int, tp_rank: int, **kwargs: Any) -> torch.Tensor:
    return torch.split(t, t.shape[0] // tp, dim=0)[tp_rank]

def sp_1(t: torch.Tensor, tp: int, tp_rank: int, **kwargs: Any) -> torch.Tensor:
    return torch.split(t, t.shape[1] // tp, dim=1)[tp_rank]

def sp_neg1(t: torch.Tensor, tp: int, tp_rank: int, **kwargs: Any) -> torch.Tensor:
    return torch.split(t, t.shape[-1] // tp, dim=-1)[tp_rank]

def sp_id(t: torch.Tensor, tp: int, tp_rank: int, **kwargs: Any) -> torch.Tensor:
    return t

def sp_moe_neg1(t: torch.Tensor, tp: int, tp_rank: int, ep: int, ep_rank: int, **kwargs: Any) -> torch.Tensor:
    if ep > 1:
        tp = int(tp / ep)
        tp_rank = int(tp_rank / ep)
    t1 = torch.split(t, t.shape[-1] // tp, dim=-1)[tp_rank]
    if ep > 1:
        t1 = torch.split(t1, t1.shape[0]// ep, dim=0)[ep_rank]
    return t1 


def sp_moe_w1(t: torch.Tensor, tp: int, tp_rank: int, ep: int, ep_rank: int,  **kwargs: Any) -> torch.Tensor:
    # [expert_num, 2*n, k]
    if ep > 1:
        tp = int(tp / ep)
        tp_rank = int(tp_rank / ep)
    t1 = t.reshape([t.shape[0], 2, -1, t.shape[-1]])
    t2 = torch.split(t1, t1.shape[2] // tp, dim=2)[tp_rank]
    t2 = t2.reshape([t2.shape[0], -1, t2.shape[-1]]) 
    if ep > 1:
        t2 = torch.split(t2, t2.shape[0] // ep, dim=0)[ep_rank]
    t3 = t2.reshape([t2.shape[0], -1, t2.shape[-1]])
    return t3

def stack_(ts: List[torch.Tensor]):
    return torch.stack(ts, dim=0)

def stack_moe_w1(ts: List[torch.Tensor]):
    gate = ts[:len(ts) // 2]
    up = ts[len(ts) // 2:]
    ws = []
    for w1, w3 in zip(gate, up):
        ws.append(torch.concat([w1, w3], dim=0))
    x =  torch.stack(ws, dim=0)
    return x

def get_sp_tensor(t: torch.Tensor, head_num: int, head_num_kv: int, size_per_head: int,
                  tp: int, tp_rank: int, **kwargs):
    t = t.reshape([-1, (head_num + head_num_kv * 2) * size_per_head])
    q_hidden = head_num * size_per_head
    kv_hidden = head_num_kv * size_per_head
    if len(t.shape) == 1:
        t = t.unsqueeze(0)
    qs = sp_neg1(t[:,:q_hidden], tp, tp_rank)
    if head_num_kv == 1:
        ks = t[:,q_hidden:q_hidden + kv_hidden]
        vs = t[:,q_hidden + kv_hidden:]
    else:
        ks = sp_neg1(t[:,q_hidden:q_hidden + kv_hidden], tp, tp_rank)
        vs = sp_neg1(t[:,q_hidden + kv_hidden:], tp, tp_rank)
    return torch.concat([qs, ks, vs], dim=1).contiguous()

# MHA layout: [D, head*size_per_head, head*size_per_head, head*size_per_head] == [D, 3, D] (sp_neg)
# MQA layout: [D, head*size_per_head, kv_head*size_per_head, kv_head*size_per_head] (sp_head)
def sp_head(t: torch.Tensor, hidden_size: int, head_num: int, head_num_kv: int, size_per_head: int, bits: int,
            **kwargs: Any) -> torch.Tensor:
    # quant
    if len(t.shape) == 2 and t.dtype == torch.int32:
        nums = 32 // bits
        # awq
        if t.shape[0] == hidden_size and t.shape[1] == ((head_num + head_num_kv * 2) * size_per_head) // nums:
            size_per_head = size_per_head // nums
    return get_sp_tensor(t,
                         head_num=head_num,
                         head_num_kv=head_num_kv,
                         size_per_head=size_per_head,
                         **kwargs)

def sp_head_s(t: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    return get_sp_tensor(t, **kwargs)

def sp_head_z(t: torch.Tensor, size_per_head: int, bits: int, **kwargs: Any) -> torch.Tensor:
    size_per_head = size_per_head // (32 // bits)
    z = get_sp_tensor(t, size_per_head=size_per_head, **kwargs)
    return z

def sp_head_b(t: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    return get_sp_tensor(t, **kwargs)

def sp_head_qk_norm(t: torch.Tensor, tp, tp_rank, head_num, head_num_kv, size_per_head, **kwargs: Any) -> torch.Tensor:
    q_hidden = head_num * size_per_head
    t = t.reshape(1, -1)
    qs = sp_neg1(t[:,:q_hidden], tp, tp_rank)
    if head_num_kv == 1:
        ks = t[:,q_hidden:]
    else:
        ks = sp_neg1(t[:,q_hidden:], tp, tp_rank)
    return torch.concat([qs, ks], dim=1).contiguous()

def sp_head_lora(t: torch.Tensor, hidden_size, **kwargs: Any) -> torch.Tensor:
    hidden_size = t.shape[0]
    return get_sp_tensor(t, hidden_size=hidden_size, **kwargs)

def sp_head_gemm_a8(t: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    return get_sp_tensor(t.reshape([t.shape[0], -1]).T, **kwargs).T

def sp_head_s_gemm_a8(t: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    return sp_head_s(t, **kwargs)

def trans_qkv(ts: List[torch.Tensor], hidden_size: int, head_num: int, size_per_head: int = -1) -> torch.Tensor:
    if size_per_head == -1:
        size_per_head = hidden_size // head_num
    return ts[0].T.reshape(hidden_size, head_num, 3, size_per_head)\
        .permute(0, 2, 1, 3)\
        .reshape(hidden_size, 3 * head_num * size_per_head)\
        .contiguous()

def trans_qkv_b(ts: List[torch.Tensor], hidden_size: int, head_num: int) -> torch.Tensor:
    return ts[0].reshape(head_num, 3, hidden_size // head_num)\
        .permute(1, 0, 2)\
        .reshape(3 * hidden_size)\
        .contiguous()

def qkv_transpose(ts, hidden_size):
    return ts[0].reshape(hidden_size, -1)

def qkv_gather(ts: List[torch.Tensor], dim0: int, head_num: int, head_num_kv: int, size_per_head: int = -1) -> torch.Tensor:
    t = ts[0].t().contiguous().reshape(dim0, -1)
    if size_per_head == -1:
        size_per_head = t.shape[1] // (head_num + head_num_kv * 2)
    new_idxs: List[int] = []
    q2kv_ratio = head_num // head_num_kv
    for q2kv_idx in range(head_num_kv):
        base_idx = (q2kv_ratio + 2) * q2kv_idx
        new_idxs.extend(list(range(base_idx, base_idx + q2kv_ratio)))
    for q2kv_idx in range(head_num_kv):
        new_idxs.append((q2kv_ratio + 2) * q2kv_idx + q2kv_ratio)
    for q2kv_idx in range(head_num_kv):
        new_idxs.append((q2kv_ratio + 2) * q2kv_idx + q2kv_ratio + 1)
    return t.reshape(dim0, head_num + head_num_kv * 2, size_per_head)[:,new_idxs,:].reshape(dim0, -1)

def sp_0_pad8(t: torch.Tensor, tp: int, tp_rank: int, **kwargs: Any) -> torch.Tensor:
    align_size = tp * 8
    paded_size = int(math.ceil(t.shape[0] * 1.0 / align_size) * align_size)
    pad_size = int(paded_size - t.shape[0])
    per_slice_size = int(paded_size / tp)
    if pad_size != 0 and tp_rank == tp - 1:
        if len(t.shape) == 2:
            return torch.concat([t[tp_rank * per_slice_size:,:],
                                 torch.zeros([pad_size, t.shape[1]], dtype=t.dtype)], dim=0)
        else:
            return torch.concat([t[tp_rank * per_slice_size:,:],
                                 torch.zeros([pad_size], dtype=t.dtype)], dim=0)
    else:
        if len(t.shape) == 2:
            return t[tp_rank * per_slice_size:(tp_rank + 1) * per_slice_size,:]
        else:
            return t[tp_rank * per_slice_size:(tp_rank + 1) * per_slice_size]

def merge_qkv_hf(ts: List[torch.Tensor]):
    q, k, v = ts
    qkv_weight = torch.concat([q.T, k.T, v.T], dim=1).contiguous()
    return qkv_weight

def merge_qkv_b(ts: List[torch.Tensor]):
    q, k, v = ts
    qkv_b = torch.concat([q, k, v], dim=0).contiguous()
    return qkv_b

def trans_lora_qkv(ts: List[torch.Tensor], head_num: int, head_size: int):
    split = 3
    r = ts[0].shape[1]
    return ts[0].T.reshape(r, head_num, split, head_size).permute(0, 2, 1, 3).reshape(r, split, head_num * head_size).contiguous()

def merge_qkv_lora_A(ts: List[torch.Tensor]):
    assert(len(ts) == 3), f"merge_qkv_lora_A expect 3 tensor list but get {len(ts)}"
    q, k, v = ts
    try:
        qkv_weight = torch.concat([q.T, k.T, v.T], dim=1).contiguous()
        return qkv_weight
    except:
        raise Exception(
            f"merge_qkv_lora_A failed: q shape {q.shape}, k shape {k.shape}, v shape {v.shape}")

def merge_qkv_lora_B(ts: List[torch.Tensor]):
    q, k, v = ts
    t_q = torch.zeros_like(q)
    t_k = torch.zeros_like(k)
    t_v = torch.zeros_like(v)
    return torch.cat((torch.cat((q,   t_q, t_q), dim=1),
                      torch.cat((t_k, k,   t_k), dim=1),
                      torch.cat((t_v, t_v, v  ), dim=1))).T.contiguous()

def merge_te_qkv(ts: List[torch.Tensor]):
    q, k, v = ts
    qkv_weight = torch.concat([q, k, v], dim=0).contiguous()
    return qkv_weight

class W:
    # global
    embedding = 'embedding'
    lm_head = 'lm_head'
    lm_head_b = 'lm_head_b'
    prefix_w = 'transformer.prefix_encoder.embedding.weight'
    pre_decoder_ln_gamma = 'pre_decoder_layernorm.gamma'
    pre_decoder_ln_beta = 'pre_decoder_layernorm.bias'
    positional_embedding = 'position_encoding.weight'
    token_type_embedding = 'token_type_embedding.weight'
    final_ln_gamma = 'final_layernorm.gamma'
    final_ln_beta = 'final_layernorm.beta'

    # attn
    pre_ln_gamma = 'pre_layernorm_weights.gamma'
    pre_ln_beta = 'pre_layernorm_weights.beta'
    pre_attn_ln_gamma = 'pre_attn_layernorm_weights.gamma'
    pre_attn_ln_beta = 'pre_attn_layernorm_weights.beta'
    attn_qkv_w = 'self_attention_weights.query_weight.kernel'
    attn_qkv_b = 'self_attention_weights.query_weight.bias'
    attn_ln_gamma = 'self_attention_weights.attention_layernorm.gamma'
    attn_ln_beta = 'self_attention_weights.attention_layernorm.beta'
    qk_ln_gamma = 'self_attention_weights.qk_layernorm.gamma'
    attn_o_w = 'self_attention_weights.attention_output_weight.kernel'
    attn_o_b = 'self_attention_weights.attention_output_weight.bias'
    post_ln_gamma = 'post_layernorm_weights.gamma'
    post_ln_beta = 'post_layernorm_weights.beta'
    linear_bias_slopes = "linear_bias_slopes"

    # jina_bert
    q_ln_gamma      = "self_attention_weights.q_layernorm.gamma"
    q_ln_beta       = "self_attention_weights.q_layernorm.beta"
    k_ln_gamma      = "self_attention_weights.k_layernorm.gamma"
    k_ln_beta       = "self_attention_weights.k_layernorm.beta"

    post_ln_2_gamma = 'post_layernorm_weights_2.gamma'
    post_ln_2_beta = 'post_layernorm_weights_2.beta'

    # mla
    mla_q_w = "self_attention_weights.mla.query_weight.kernel"
    mla_q_a_w = "self_attention_weights.mla.query_a_weight.kernel"
    mla_q_b_w = "self_attention_weights.mla.query_b_weight.kernel"
    mla_kv_a_w = "self_attention_weights.mla.key_value_a_weight.kernel"
    mla_k_rope_w = "self_attention_weights.mla.key_rope_weight.kernel"
    mla_k_nope_w = "self_attention_weights.mla.key_nope_weight.kernel"
    mla_v_w = "self_attention_weights.mla.value_weight.kernel"
    mla_q_a_ln_gamma = "self_attention_weights.mla.query_a_layernorm_weight.gamma"
    mla_q_a_ln_beta = "self_attention_weights.mla.query_a_layernorm_weight.beta"
    mla_kv_a_ln_gamma = "self_attention_weights.mla.key_value_a_layernorm_weight.gamma"
    mla_kv_a_ln_beta = "self_attention_weights.mla.key_value_a_layernorm_weight.beta"

    mla_q_s = "self_attention_weights.mla.query_weight.weight_only_quant_scale"
    mla_q_a_s = "self_attention_weights.mla.query_a_weight.weight_only_quant_scale"
    mla_q_b_s = "self_attention_weights.mla.query_b_weight.weight_only_quant_scale"
    mla_kv_a_s = "self_attention_weights.mla.key_value_a_weight.weight_only_quant_scale"
    mla_k_rope_s = "self_attention_weights.mla.key_rope_weight.weight_only_quant_scale"
    mla_k_nope_s = "self_attention_weights.mla.key_nope_weight.weight_only_quant_scale"
    mla_v_s = "self_attention_weights.mla.value_weight.weight_only_quant_scale"

    # ffn
    ffn_w1 = 'ffn_weights.intermediate_weight.kernel'
    ffn_b1 = 'ffn_weights.intermediate_weight.bias'
    ffn_w3 = 'ffn_weights.intermediate_weight3.kernel'
    ffn_b3 = 'ffn_weights.intermediate_weight3.bias'
    ffn_ln_gamma = 'ffn_weights.dense_layernorm.gamma'
    ffn_ln_beta = 'ffn_weights.dense_layernorm.beta'
    ffn_w2 = 'ffn_weights.intermediate_weight2.kernel'
    ffn_b2 = 'ffn_weights.intermediate_weight2.bias'
    post_ffn_ln_gamma = "post_ffn_layernorm_weights.gamma"
    post_ffn_ln_beta = "post_ffn_layernorm_weights.beta"

    # cogvlm2-expert-attn
    vision_attn_qkv_w = 'self_attention_weights.vision_query_weight.kernel'
    vision_attn_qkv_b = 'self_attention_weights.vision_query_weight.bias'
    vision_attn_o_w = 'self_attention_weights.vision_attention_output_weight.kernel'

    # cogvlm2-expert-ffn
    vision_ffn_w1 = 'vision_ffn_weights.intermediate_weight.kernel'
    vision_ffn_w3 = 'vision_ffn_weights.intermediate_weight3.kernel'
    vision_ffn_w2 = 'vision_ffn_weights.intermediate_weight2.kernel'

    # partial moe
    shared_expert_gate = 'ffn_weights.shared_expert_gate.kernel'
    moe_w1   = 'partial_moe_weights.intermediate_weight.kernel'
    moe_b1   = 'partial_moe_weights.intermediate_weight.bias'
    moe_w2   = 'partial_moe_weights.intermediate_weight2.kernel'
    moe_b2   = 'partial_moe_weights.intermediate_weight2.bias'
    moe_gate = 'partial_moe_weights.gate.kernel'

    # cross attn
    cross_attn_pre_ln_gamma = 'cross_attention_weights_pre_layernorm.gamma'
    cross_attn_pre_ln_beta = 'cross_attention_weights_pre_layernorm.beta'
    cross_attn_qkv_w = 'cross_attention_weights.query_weight.weight'
    cross_attn_qkv_b = 'cross_attention_weights.query_weight.bias'
    cross_attn_o_w = 'cross_attention_weights.output_weight.weight'
    cross_attn_o_b = 'cross_attention_weights.output_weight.bias'

    # lora
    attn_qkv_w_lora_a = 'self_attention_weights.query_weight.kernel.lora_A'
    attn_qkv_w_lora_b = 'self_attention_weights.query_weight.kernel.lora_B'
    attn_o_w_lora_a = 'self_attention_weights.attention_output_weight.kernel.lora_A'
    attn_o_w_lora_b = 'self_attention_weights.attention_output_weight.kernel.lora_B'
    ffn_w1_lora_a = 'ffn_weights.intermediate_weight.kernel.lora_A'
    ffn_w1_lora_b = 'ffn_weights.intermediate_weight.kernel.lora_B'
    ffn_w3_lora_a = 'ffn_weights.intermediate_weight3.kernel.lora_A'
    ffn_w3_lora_b = 'ffn_weights.intermediate_weight3.kernel.lora_B'
    ffn_w2_lora_a = 'ffn_weights.intermediate_weight2.kernel.lora_A'
    ffn_w2_lora_b = 'ffn_weights.intermediate_weight2.kernel.lora_B'

    # gptq
    attn_qkv_z = 'self_attention_weights.query_weight.zero'
    attn_qkv_s = 'self_attention_weights.query_weight.weight_only_quant_scale'
    vision_attn_qkv_s = 'self_attention_weights.vision_query_weight.weight_only_quant_scale'
    attn_o_z = 'self_attention_weights.attention_output_weight.zero'
    attn_o_s = 'self_attention_weights.attention_output_weight.weight_only_quant_scale'
    vision_attn_o_s = 'self_attention_weights.vision_attention_output_weight.weight_only_quant_scale'
    ffn_z1 = 'ffn_weights.intermediate_weight.zero'
    ffn_s1 = 'ffn_weights.intermediate_weight.weight_only_quant_scale'
    vision_ffn_s1 = 'vision_ffn_weights.intermediate_weight.weight_only_quant_scale'
    ffn_z3 = 'ffn_weights.intermediate_weight3.zero'
    ffn_s3 = 'ffn_weights.intermediate_weight3.weight_only_quant_scale'
    vision_ffn_s3 = 'vision_ffn_weights.intermediate_weight3.weight_only_quant_scale'
    ffn_act_s = 'ffn_weights.intermediate_weight2.act_quant_scale'  # gpt_xx model awq quant act need div scales
    ffn_z2 = 'ffn_weights.intermediate_weight2.zero'
    ffn_s2 = 'ffn_weights.intermediate_weight2.weight_only_quant_scale'
    vision_ffn_s2 = 'vision_ffn_weights.intermediate_weight2.weight_only_quant_scale'
    moe_z1 = 'partial_moe_weights.intermediate_weight.zero'
    moe_s1 = 'partial_moe_weights.intermediate_weight.weight_only_quant_scale'
    moe_z2 = 'partial_moe_weights.intermediate_weight2.zero'
    moe_s2 = 'partial_moe_weights.intermediate_weight2.weight_only_quant_scale'

    # sq
    attn_i_smoother = 'self_attention_weights.query_weight.smoother'
    attn_o_smoother = 'self_attention_weights.attention_output_weight.smoother'
    attn_o_shift = 'self_attention_weights.attention_output_weight.shift'
    ffn_smoother = 'ffn_weights.intermediate_weight2.smoother'
    
    #per tensor quant
    pre_decoder_ln_static_quant = "pre_decoder_layernorm.static_quant"
    pre_decoder_ln_static_quant_reciprocal = 'pre_decoder_layernorm.static_quant_reciprocal'
    pre_ln_static_quant = 'pre_layernorm_weights.static_quant'
    pre_ln_static_quant_reciprocal = 'pre_layernorm_weights.static_quant_reciprocal'
    attention_output_static_quant = 'self_attention_weights.attention_output_weight.static_quant'
    attention_output_static_quant_reciprocal = 'self_attention_weights.attention_output_weight.static_quant_reciprocal'
    post_ln_static_quant = 'post_layernorm_weights.static_quant'
    post_ln_static_quant_reciprocal = 'post_layernorm_weights.static_quant_reciprocal'
    ffn_intermediate_weight2_static_quant = 'ffn_weights.intermediate_weight2.static_quant'
    ffn_intermediate_weight2_static_quant_reciprocal = 'ffn_weights.intermediate_weight2.static_quant_reciprocal'
    ffn_intermediate_weight3_static_quant = "ffn_weights.intermediate_weight3.static_quant"
    ffn_intermediate_weight3_static_quant_reciprocal = "ffn_weights.intermediate_weight3.static_quant_reciprocal"

    post_ffn_ln_static_quant = "post_ffn_layernorm_weights.static_quant"
    post_ffn_ln_static_quant_reciprocal = "post_ffn_layernorm_weights.static_quant_reciprocal"

    # per tensor quant
    pre_decoder_ln_static_quant = "pre_decoder_layernorm.static_quant"
    pre_decoder_ln_static_quant_reciprocal = 'pre_decoder_layernorm.static_quant_reciprocal'
    pre_ln_static_quant = 'pre_layernorm_weights.static_quant'
    pre_ln_static_quant_reciprocal = 'pre_layernorm_weights.static_quant_reciprocal'
    attention_output_static_quant = 'self_attention_weights.attention_output_weight.static_quant'
    attention_output_static_quant_reciprocal = 'self_attention_weights.attention_output_weight.static_quant_reciprocal'
    post_ln_static_quant = 'post_layernorm_weights.static_quant'
    post_ln_static_quant_reciprocal = 'post_layernorm_weights.static_quant_reciprocal'
    ffn_intermediate_weight2_static_quant = 'ffn_weights.intermediate_weight2.static_quant'
    ffn_intermediate_weight2_static_quant_reciprocal = 'ffn_weights.intermediate_weight2.static_quant_reciprocal'
    post_ffn_ln_static_quant = "post_ffn_layernorm_weights.static_quant"
    post_ffn_ln_static_quant_reciprocal = "post_ffn_layernorm_weights.static_quant_reciprocal"

    # medusa lm_head
    medusa_head = 'medusa_head'

    partial_moe_w = set([
        moe_w1,
        moe_b1,
        moe_w2,
        moe_b2,
        moe_gate,
        shared_expert_gate
    ])

    quant_w = set([
        attn_qkv_w,
        vision_attn_qkv_w,
        attn_o_w,
        vision_attn_o_w,
        ffn_w1,
        vision_ffn_w1,
        ffn_w2,
        vision_ffn_w2,
        ffn_w3,
        vision_ffn_w3,
        moe_w1,
        moe_w2,
        attn_qkv_s,
        attn_o_s,
        ffn_s1,
        ffn_s2,
        ffn_act_s,
        ffn_s3,
        attn_i_smoother,
        attn_o_smoother,
        attn_o_shift,
        ffn_smoother,
        mla_q_w,
        mla_q_a_w,
        mla_q_b_w,
        mla_kv_a_w,
        mla_k_rope_w,
        mla_k_nope_w,
        mla_v_w
    ])

    groupwise_quant_params = set([
        attn_qkv_z,
        attn_qkv_s,
        attn_o_z,
        attn_o_s,
        ffn_z1,
        ffn_s1,
        ffn_z2,
        ffn_s2,
        ffn_z3,
        ffn_s3,
        moe_z1,
        moe_s1,
        moe_z2,
        moe_s2,
    ])

    sq_quant_weights = [
        attn_qkv_w,
        attn_o_w,
        ffn_w1,
        ffn_w3,
        ffn_w2
    ]

    sq_quant_scales = [
        attn_qkv_s,
        attn_o_s,
        ffn_s1,
        ffn_s2,
        ffn_s3,
        attn_i_smoother,
        attn_o_smoother,
        ffn_smoother
    ]

    sq_quant_shifts = [
        attn_o_shift
    ]
    
    static_quant_scales = [
        pre_ln_static_quant,
        pre_ln_static_quant_reciprocal,
        attention_output_static_quant,
        attention_output_static_quant_reciprocal,
        post_ln_static_quant,
        post_ln_static_quant_reciprocal,
        ffn_intermediate_weight2_static_quant,
        ffn_intermediate_weight2_static_quant_reciprocal,
        ffn_intermediate_weight3_static_quant,
        ffn_intermediate_weight3_static_quant_reciprocal,
        post_ffn_ln_static_quant,
        post_ffn_ln_static_quant_reciprocal
    ]

    static_quant_scales = [
        pre_ln_static_quant,
        pre_ln_static_quant_reciprocal,
        attention_output_static_quant,
        attention_output_static_quant_reciprocal,
        post_ln_static_quant,
        post_ln_static_quant_reciprocal,
        ffn_intermediate_weight2_static_quant,
        ffn_intermediate_weight2_static_quant_reciprocal,
        post_ffn_ln_static_quant,
        post_ffn_ln_static_quant_reciprocal
    ]

    int8_attn_weights = [
        [attn_qkv_w, attn_qkv_s],
        [attn_o_w, attn_o_s],
        [mla_q_w, mla_q_s],
        [mla_q_a_w, mla_q_a_s],
        [mla_q_b_w, mla_q_b_s],
        [mla_kv_a_w, mla_kv_a_s],
        [mla_k_rope_w, mla_k_rope_s],
        [mla_k_nope_w, mla_k_nope_s],
        [mla_v_w, mla_v_s],
    ]

    int8_attn_vision_weights = [
        [vision_attn_qkv_w, vision_attn_qkv_s],
        [vision_attn_o_w, vision_attn_o_s],
    ]

    int8_ffn_weights = [
        [ffn_w1, ffn_s1],
        [ffn_w3, ffn_s3],
        [ffn_w2, ffn_s2],
    ]

    int8_vision_ffn_weights = [
        [vision_ffn_w1, vision_ffn_s1],
        [vision_ffn_w3, vision_ffn_s3],
        [vision_ffn_w2, vision_ffn_s2],
    ]

    int8_ffn_weights_2 = [
        [ffn_w3, ffn_s3],
        [ffn_w2, ffn_s2],
    ]

    int8_partial_moe_weights = [
        [moe_w1, moe_s1],
        [moe_w2, moe_s2]
    ]

    int8_partial_moe_weights_2 = [
        [moe_w1, moe_s1],
        [moe_w2, moe_s2],
    ]

    groupwise_attn_weights = [
        [attn_qkv_w, attn_qkv_z, attn_qkv_s],
        [attn_o_w, attn_o_z, attn_o_s],
    ]

    groupwise_ffn_weights = [
        [ffn_w1, ffn_z1, ffn_s1],
        [ffn_w3, ffn_z3, ffn_s3],
        [ffn_w2, ffn_z2, ffn_s2],
    ]

    groupwise_ffn_weights_2 = [
        [ffn_w3, ffn_z3, ffn_s3],
        [ffn_w2, ffn_z2, ffn_s2],
    ]

    groupwise_partial_moe_weights = [
        [moe_w1, moe_z1, moe_s1],
        [moe_w2, moe_z2, moe_s2]
    ]

    gpt_style_tp_strategy: Dict[str, Any] = {
        embedding: sp_neg1,
        lm_head: sp_0_pad8,
        lm_head_b: sp_0_pad8,
        pre_decoder_ln_gamma: sp_id,
        pre_decoder_ln_beta: sp_id,
        final_ln_gamma: sp_id,
        final_ln_beta: sp_id,
        pre_ln_gamma: sp_id,
        pre_ln_beta: sp_id,

        pre_attn_ln_gamma: sp_id,
        pre_attn_ln_beta: sp_id,
        qk_ln_gamma: sp_head_qk_norm,
        attn_qkv_w: sp_head,
        vision_attn_qkv_w: sp_head,
        attn_qkv_z: sp_head_z,
        attn_qkv_s: sp_head_s,
        attn_qkv_b: sp_head_b,
        vision_attn_qkv_b: sp_head_b,
        attn_o_w: sp_0,
        vision_attn_o_w: sp_0,
        attn_o_z: sp_0,
        attn_o_s: sp_0,
        attn_o_b: sp_id,
        attn_i_smoother: sp_0,
        attn_o_smoother: sp_0,
        attn_o_shift: sp_0,

        # mla
        mla_q_w: sp_neg1,
        mla_q_a_w: sp_id,
        mla_q_b_w: sp_neg1,
        mla_kv_a_w: sp_id,
        mla_k_rope_w: sp_id,
        mla_k_nope_w: sp_neg1,
        mla_v_w: sp_neg1,
        mla_q_a_ln_gamma: sp_id,
        mla_q_a_ln_beta: sp_id,
        mla_kv_a_ln_gamma: sp_id,
        mla_kv_a_ln_beta: sp_id,
        mla_q_s: sp_neg1,
        mla_q_a_s: sp_id,
        mla_q_b_s: sp_neg1,
        mla_kv_a_s: sp_id,
        mla_k_rope_s: sp_id,
        mla_k_nope_s: sp_neg1,
        mla_v_s: sp_neg1,

        cross_attn_pre_ln_gamma: sp_id,
        cross_attn_pre_ln_beta: sp_id,
        cross_attn_qkv_w: sp_head,
        cross_attn_qkv_b: sp_head_b,
        cross_attn_o_w: sp_0,
        cross_attn_o_b: sp_id,

        ffn_w1: sp_neg1,
        vision_ffn_w1: sp_neg1,
        ffn_z1: sp_neg1,
        ffn_s1: sp_neg1,
        ffn_b1: sp_neg1,
        ffn_w3: sp_neg1,
        vision_ffn_w3: sp_neg1,
        ffn_z3: sp_neg1,
        ffn_s3: sp_neg1,
        ffn_b3: sp_neg1,
        ffn_w2: sp_0,
        vision_ffn_w2: sp_0,
        ffn_z2: sp_0,
        ffn_s2: sp_0,
        ffn_b2: sp_id,
        ffn_act_s: sp_0,
        ffn_smoother: sp_0,

        moe_w1: sp_moe_w1,
        moe_z1: sp_moe_w1,
        moe_s1: sp_moe_w1,
        moe_b1: sp_moe_neg1,
        moe_w2: sp_moe_neg1,
        moe_z2: sp_moe_neg1,
        moe_s2: sp_moe_neg1,
        moe_b2: sp_moe_neg1,

        post_ln_beta: sp_id,
        post_ln_gamma: sp_id,
        positional_embedding: sp_neg1,

        attn_qkv_w_lora_a: sp_id,
        attn_qkv_w_lora_b: sp_head_lora,
        attn_o_w_lora_a: sp_0,
        attn_o_w_lora_b: sp_id,
        ffn_w1_lora_a: sp_id,
        ffn_w1_lora_b: sp_neg1,
        ffn_w3_lora_a: sp_id,
        ffn_w3_lora_b: sp_neg1,
        ffn_w2_lora_a: sp_0,
        ffn_w2_lora_b: sp_id,
        moe_gate: sp_id,
        shared_expert_gate: sp_id,
        post_ffn_ln_beta: sp_id,
        post_ffn_ln_gamma: sp_id,
        token_type_embedding: sp_neg1
    }

    weights_list = [
        embedding,
        lm_head,
        lm_head_b,
        pre_decoder_ln_gamma,
        pre_decoder_ln_beta,
        positional_embedding,
        final_ln_gamma,
        final_ln_beta,
        prefix_w
    ]

    layer_weights_list = [
        pre_ln_gamma,
        pre_ln_beta,
        attn_qkv_w,
        attn_qkv_b,
        attn_ln_gamma,
        attn_ln_beta,
        qk_ln_gamma,
        attn_o_w,
        attn_o_b,
        post_ln_gamma,
        post_ln_beta,
        ffn_w1,
        ffn_b1,
        ffn_w3,
        ffn_b3,
        ffn_ln_gamma,
        ffn_ln_beta,
        ffn_w2,
        ffn_b2,
    ]

    skip_weights_list = [
        attn_qkv_w,
        attn_qkv_b,
        attn_ln_gamma,
        attn_ln_beta,
        qk_ln_gamma,
        attn_o_w,
    ]

    @staticmethod
    def gemm_int8_gpt_style_tp_strategy():
        gemm_a8_weight_tp_strategy: Dict[str, Any] = {
            W.attn_qkv_w: sp_head_gemm_a8,
            W.attn_qkv_s: sp_head_s_gemm_a8,
            W.attn_o_w: sp_neg1,
            W.attn_o_s: sp_id,
            W.attn_o_smoother: sp_0,
            W.attn_o_shift: sp_0,
            W.ffn_w1: sp_0,
            W.ffn_s1: sp_0,
            W.ffn_w3: sp_0,
            W.ffn_s3: sp_0,
            W.ffn_w2: sp_neg1,
            W.ffn_s2: sp_id,
        }
        tp_strategy = copy.deepcopy(W.gpt_style_tp_strategy)
        tp_strategy.update(gemm_a8_weight_tp_strategy)
        return tp_strategy

class CkptWeightInfo:
    name: str
    merge_fun: Callable[[List[torch.Tensor]], torch.Tensor]

    # hf checkpoint没有tensor做拆分的ckpt，所以默认函数可以是identity
    def __init__(self, name: str, merge_fun: Callable[[List[torch.Tensor]], torch.Tensor] = identity) -> None:
        self.name = name
        self.merge_fun = merge_fun

    def tensor_name(self, layer_id: Optional[int]):
        if layer_id is not None:
            return self.name.format(i=str(layer_id), i_1=str(layer_id + 1))
        return self.name
    def __str__(self) -> str:
        return f"CkptWeightInfo[{self.name}]"

    def __repr__(self) -> str:
        return self.__str__()

class WeightInfo:
    name: str
    weights: List[CkptWeightInfo]
    process_fun: Callable[[List[torch.Tensor]], torch.Tensor]
    data_type: Optional[torch.dtype] = None

    def __init__(self, name: str, weights: List[CkptWeightInfo], process_fun: Callable[[List[torch.Tensor]], torch.Tensor] = identity, data_type: Optional[torch.dtype] = None) -> None:
        self.name = name
        self.weights = weights
        self.process_fun = process_fun
        self.data_type = data_type

    def get_ckpt_tensor_names(self) -> List[str]:
        if not bool(self.weights):
            return []
        return [ckpt.name for ckpt in self.weights]

    def __str__(self) -> str:
        return f"WeightInfo[{self.name}]{self.weights}"

    def __repr__(self) -> str:
        return self.__str__()


class Fp8WeightStyle(Enum):
    NONE = 0
    TRT_ENGINE = 1
    TRANSFORMER_ENGINE = 2
    
class ModelWeightInfo:
    layer_weights: Union[List[WeightInfo], List[List[WeightInfo]]]
    weights: List[WeightInfo]
    tp_strategy: Optional[Dict[Any, Any]]
    lora_weights: List[WeightInfo] = []
    medusa_weights: List[WeightInfo] = []

    def __init__(self, weights: List[WeightInfo],
                 layer_weights: Union[List[WeightInfo], List[List[WeightInfo]]],
                 tp_strategy: Optional[Dict[Any, Any]] = W.gpt_style_tp_strategy,
                 lora_weights: Optional[List[WeightInfo]] = None,
                 medusa_weights: List[WeightInfo] = []) -> None:
        self.weights = weights
        self.layer_weights = layer_weights
        self.tp_strategy = tp_strategy
        if lora_weights == None :
            self.lora_weights = self.convert_lora()
        else:
            self.lora_weights = lora_weights
        self.medusa_weights = medusa_weights

    def convert_lora(self):
        if isinstance(self.layer_weights[0], list):
            layer_weights = self.layer_weights[0]
        else:
            layer_weights = self.layer_weights

        lora_layer_weights: List[WeightInfo] = []

        lora_base_name = "base_model.model.{}.{}.weight"

        target_modules = set([W.attn_o_w, W.ffn_w1, W.ffn_w2, W.ffn_w3, W.attn_qkv_w])
        layer_names = set([layer_weight.name for layer_weight in layer_weights])

        lora_names = target_modules & layer_names
        assert(len(lora_names) != 0)

        # logging.info(f"lora_names is {lora_names}")

        for lora_a_b in ['lora_A', 'lora_B']:
            for lora_name in lora_names:
                ckpt_layer_weight = None
                orig_process_fun = None
                for layer_weight in layer_weights:

                    if layer_weight.name == lora_name:
                        ckpt_layer_weight = layer_weight.weights
                        orig_process_fun = layer_weight.process_fun

                assert (ckpt_layer_weight != None and orig_process_fun != None)
                if lora_name == W.attn_qkv_w and len(ckpt_layer_weight) == 3:
                    ckpt_names = [lora_base_name.format(ckpt.name[:-len(".weight")], lora_a_b) for ckpt in ckpt_layer_weight]
                    qkv_ckpt_weights = [CkptWeightInfo(name, identity) for name in ckpt_names]
                    if lora_a_b == 'lora_A':
                        lora_layer_weights.append(WeightInfo(lora_name + "." + lora_a_b, qkv_ckpt_weights, merge_qkv_lora_A))
                    elif lora_a_b == 'lora_B':
                        lora_layer_weights.append(WeightInfo(lora_name + "." + lora_a_b, qkv_ckpt_weights, merge_qkv_lora_B))
                else:
                    ckpt_name = lora_base_name.format(ckpt_layer_weight[0].name[:-len(".weight")], lora_a_b)
                    ckpt_weight_info = CkptWeightInfo(ckpt_name, identity)

                    if lora_a_b == 'lora_A' and isinstance(orig_process_fun, functools.partial) and orig_process_fun.func.__name__ == 'transpose_pad' and orig_process_fun.keywords['dim'] ==1:
                        lora_b_process_fun = orig_process_fun
                    elif lora_a_b == 'lora_B' and isinstance(orig_process_fun, functools.partial) and orig_process_fun.func.__name__ == 'transpose_pad' and orig_process_fun.keywords['dim'] ==0:
                        lora_b_process_fun = orig_process_fun
                    else:
                        lora_b_process_fun = transpose
                    lora_layer_weights.append(WeightInfo(lora_name + "." + lora_a_b, [ckpt_weight_info], lora_b_process_fun))
        return lora_layer_weights

    def set_lora(self, qkv_fun = None, half1 = None , half2 = None):
        for lora_weight in self.lora_weights:
            if lora_weight.name == W.attn_qkv_w + '.lora_B' and qkv_fun != None:
                lora_weight.process_fun = qkv_fun
            if lora_weight.name == W.ffn_w1 + '.lora_B' and half1 != None:
                lora_weight.process_fun = half1
            if lora_weight.name == W.ffn_w3 + '.lora_B' and half2 != None:
                lora_weight.process_fun = half2


    def has_lora_weight(self):
        if len(self.lora_weights) == 0:
            return False
        return True

    def find_lora_a(self, weight: WeightInfo):
        for lora_weight in self.lora_weights:
            if weight.name + "." + 'lora_A' == lora_weight.name:
                return lora_weight

    def find_lora_b(self, weight: WeightInfo):
        for lora_weight in self.lora_weights:
            if weight.name + "." + 'lora_B' == lora_weight.name:
                return lora_weight


class ModelDeployWeightInfo:

    def __init__(self, config: GptInitModelParameters, tp_size: int, tp_rank: int):
        self._hidden_size = config.hidden_size
        self._inter_size = config.inter_size
        self._inter_padding_size = config.inter_padding_size
        self._moe_inter_padding_size = config.moe_inter_padding_size
        self._head_num = config.head_num
        self._head_num_kv = config.head_num_kv
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.ep_size = config.ep_size
        self.ep_rank = config.ep_rank
        self._size_per_head = config.size_per_head
        if self._head_num_kv == -1:
            self._head_num_kv = self._head_num
        self._quant_algo = config.quant_algo
        self._num_layers = config.num_layers
        self._layer_head_num = config.layer_head_num
        self._layer_inter_padding_size = config.layer_inter_padding_size
        self._has_prefix_encoder = False
        self._megatron = False
        self._is_sparse_head = config.is_sparse_head
        self._layer_head_num = config.layer_head_num
        self._src_quantization_bit = config.src_quantization_bit
        self.tp_split_emb_and_lm_head = config.tp_split_emb_and_lm_head

        self._is_medusa_model = config.gpt_init_params.use_medusa
        self._medusa_head_num = 0 if config.medusa_config is None else config.medusa_config.medusa_num_heads
        self._medusa_layer_num = 0 if config.medusa_config is None else config.medusa_config.medusa_num_layers

        self._is_gated_activation = config.gpt_init_params.isGatedActivation()
        self.expert_num_ = config.gpt_init_params.expert_num
        self.moe_k_      = config.gpt_init_params.moe_k
        self.moe_layer_index_ = config.gpt_init_params.moe_layer_index
        self.moe_style_ = config.gpt_init_params.moe_style

        self.tie_word_embeddings = config.tie_word_embeddings
        self.need_ffn_act_scale = config.need_ffn_act_scale
        self.use_expert_attention = config.use_expert_attention
        self.fp8_weight_stype = Fp8WeightStyle.NONE


        # for mla
        self.kv_lora_rank = config.kv_lora_rank
        self.nope_head_dim = config.nope_head_dim
        self.rope_head_dim = config.rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.routed_scaling_factor = config.routed_scaling_factor


    def get_preprocessed_weight_info(self, all_names: Set[str]) -> ModelWeightInfo:
        # auto create weight info based on exist tensor names
        weights: List[WeightInfo] = []
        layer_weights: List[WeightInfo] = []
        for name in W.weights_list:
            if name in all_names:
                weights.append(WeightInfo(name, [CkptWeightInfo(name, identity)], identity))

        for name in W.layer_weights_list:
            check_name = f'layer.0.{name}'
            int8_check_name = f'layer.0.{name}.int8_weight'
            if check_name in all_names or int8_check_name in all_names:
                layer_weights.append(WeightInfo(name, [CkptWeightInfo('layer.{i}.' + name, identity)], identity))

        return ModelWeightInfo(layer_weights=layer_weights,
                               weights=weights,
                               tp_strategy=self._get_gpt_style_tp_strategy())

    def get_weight_info(self) -> ModelWeightInfo:
        weight_info = self._get_weight_info()
        if self.tie_word_embeddings:
            logging.info("fix tie_word_embeddings")
            weight_info = self._fix_tie_lm_head(weight_info)
        if self._is_sparse_head:
            logging.info("Skiping load empty weight for head_num == 0")
            weight_info = self._process_sparse_weight(weight_info)
        if self._is_medusa_model:
            weight_info = self._add_medusa_head_info(weight_info)
        return weight_info

    def _fix_tie_lm_head(self, origin_weight_info: ModelWeightInfo) -> ModelWeightInfo:
        word_emb_idx = -1
        word_emb = None
        lm_head_idx = -1
        lm_head = None
        for idx, weight in enumerate(origin_weight_info.weights):
            if weight.name == W.embedding:
                word_emb_idx = idx
                word_emb = weight
            elif weight.name == W.lm_head:
                lm_head = weight
                lm_head_idx = idx
        if not lm_head or not word_emb:
            return origin_weight_info

        assert len(lm_head.weights) == 1 and len(word_emb.weights) == 1
        lm_head_ckpt_weigth_infos = [CkptWeightInfo(w.name, functools.partial(tolerate_failed, origin_func=w.merge_fun)) for w in lm_head.weights]
        lm_head_ckpt_weigth_infos.extend([CkptWeightInfo(w.name, functools.partial(tolerate_failed, origin_func=w.merge_fun)) for w in word_emb.weights])
        lm_head_merge_funcs = [lm_head.process_fun, word_emb.process_fun]
        lm_head = WeightInfo(W.lm_head, lm_head_ckpt_weigth_infos, functools.partial(choose_available, origin_func_list = lm_head_merge_funcs))
        origin_weight_info.weights[lm_head_idx] = lm_head
        return origin_weight_info

    def _process_sparse_weight(self, origin_weight_info: ModelWeightInfo) -> ModelWeightInfo:
        if not isinstance(origin_weight_info.layer_weights[0], list):
            raise Exception("model weight use sparse config should be list(list())")
        new_layer_weights = []
        for i, layer_weight in enumerate(origin_weight_info.layer_weights):
            if self._layer_head_num[i] == 0:
                new_weights = [weight for weight in layer_weight if weight.name not in W.skip_weights_list]
            else:
                new_weights = layer_weight
            new_layer_weights.append(new_weights)
        return ModelWeightInfo(origin_weight_info.weights, new_layer_weights, origin_weight_info.tp_strategy)

    def _add_medusa_head_info(self, weight_info: ModelWeightInfo) -> ModelWeightInfo:
        '''
        self.medusa_head = nn.ModuleList(
            [
                nn.Sequential(
                    *([ResBlock(self.hidden_size)] * medusa_num_layers),
                    nn.Linear(self.hidden_size, self.vocab_size, bias=False),
                )
                for _ in range(medusa_num_heads)
            ]
        )
        '''
        medusa_weight_names = []
        for head in range(self._medusa_head_num):
            for layer in range(self._medusa_layer_num):
                medusa_weight_names.append(f"medusa_head.{head}.{layer}.linear.weight")
                medusa_weight_names.append(f"medusa_head.{head}.{layer}.linear.bias")
            medusa_weight_names.append(f"medusa_head.{head}.{self._medusa_layer_num}.weight")
        weight_info.medusa_weights.append(WeightInfo(W.medusa_head, [CkptWeightInfo(x, identity) for x in medusa_weight_names], identity))
        return weight_info

    def _get_weight_info(self) -> ModelWeightInfo:
        raise NotImplementedError()

    def process_meta_from_ckpt(self, ckpt_metas: List[CkptFileInfo]):
        if len(ckpt_metas) == 0:
            return
        if 'ft_module' not in ckpt_metas[0].get_tensor_names():
            # call subclass process_meta
            self.fix_megatron_layer_id(ckpt_metas)
            meta_dicts = [ckpt_file.get_metadata() for ckpt_file in ckpt_metas]
            weight_keys = set(reduce(lambda x,y:x+y, [list(meta.keys()) for meta in meta_dicts], []))
            self._process_meta(meta_dicts, weight_keys)

    def process_meta_from_dict(self, ref_dict: Dict[str, torch.Tensor]):
        if len(ref_dict) == 0:
            return
        if 'ft_module' not in ref_dict.keys():
            self._process_meta([ref_dict], ref_dict.keys())

    def _process_meta(self, meta_dict, weight_keys):
        pass

    def fix_megatron_layer_id(self, meta_dict: List[CkptFileInfo]):

        pp_size = 1
        for meta in meta_dict:
            if meta.pretrain_pp_tp != (1, 1):
                pp_size, _ = meta.pretrain_pp_tp

        if pp_size <= 1:
            return

        per_pp_size = ((pp_size + self._num_layers -1 ) // pp_size)
        for ckpt_file in meta_dict:
            if ckpt_file.finetune_type != FinetuneType.pretrain:
                continue
            if ckpt_file.train_type != TrainType.megatron:
                continue

            pp_rank = ckpt_file.pp_rank
            start_id, end_id = self._get_layer_id_info(ckpt_file.get_metadata())
            if start_id == ((pp_size + self._num_layers) // pp_size) * pp_rank:
                assert pp_rank != pp_size -1 or end_id ==  per_pp_size * pp_rank -1
                continue
            if start_id != per_pp_size * pp_rank:
                assert start_id == 0, f"{start_id} != 0"
                offset = per_pp_size * pp_rank
                logging.info(f"fix {ckpt_file.file_name}'s: add offset {offset}")
                self._fix_megatron_layer_id_by_offset(ckpt_file.get_metadata(), offset)

    def _get_layer_start_end_id(self) -> Tuple[int, int]:
        raise NotImplementedError()

    def _fix_megatron_layer_id_by_offset(self, meta, offset):
        raise NotImplementedError()
    def _get_gpt_style_tp_strategy(self):
        if self._quant_algo.isSmoothQuant() or self._quant_algo.isOmniQuant():
            return W.gemm_int8_gpt_style_tp_strategy()
        else:
            return W.gpt_style_tp_strategy

    @staticmethod
    def _contains(keys: List[str], val: str):
        for key in keys:
            if val in key:
                return True
        return False



class ModelWeights:
    def __init__(self, num_layers: int, device: str, dtype: torch.dtype):
        self.device = device
        self.weights: List[Dict[str, torch.Tensor]] = []
        self.global_weights: Dict[str, torch.Tensor] = {}
        self._dtype = dtype

        for _ in range(num_layers):
            self.weights.append({})

    def set_layer_weight(self, layer_id: int, name: str, tensor: torch.Tensor):
        self.weights[layer_id][name] = tensor

    def set_global_weight(self, name: str, tensor: torch.Tensor):
        self.global_weights[name] = tensor

    def steal_global_weight(self, name: str):
        if name not in self.global_weights:
            return None
        tensor = self.global_weights[name]
        del self.global_weights[name]
        return tensor

    @property
    def dtype(self):
        return self._dtype
