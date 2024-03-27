import logging
import math
import os
from functools import reduce
import torch
import torch.serialization
from typing import Any, NamedTuple, Callable, List, Dict, Set, Tuple, Optional, Union
from maga_transformer.utils.database import FinetuneType, TrainType, CkptFileInfo, LoraConfig
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.utils.database import BaseDatabase
from maga_transformer.utils.RWLock import RWlock

def concat_0(ts: List[torch.Tensor]) -> torch.Tensor:
    if len(ts) == 1:
        return ts[0]

    return torch.concat(ts, dim=0).contiguous()

def concat_1(ts: List[torch.Tensor]) -> torch.Tensor:
    if len(ts) == 1:
        return ts[0]
    return torch.concat(ts, dim=1).contiguous()


def b_half_merge(ts: List[torch.Tensor]):
    n_ts_1 = []
    n_ts_2 = []
    for t in ts:
        t_a = t.chunk(2, dim=-1)
        n_ts_1.append(t_a[0].cuda())
        n_ts_2.append(t_a[1].cuda())
    return concat_0([concat_0(n_ts_1), concat_0(n_ts_2)])

def zeros(ts: List[torch.Tensor], shape) -> torch.Tensor:
    return torch.zeros(shape, dtype=torch.half).contiguous()

def ones(ts: List[torch.Tensor], shape) -> torch.Tensor:
    return torch.ones(shape, dtype=torch.half).contiguous()

def transpose(ts: List[torch.Tensor]) -> torch.Tensor:
    return ts[0].t().contiguous()

def identity(ts: List[torch.Tensor], allow_empty=False) -> torch.Tensor:
    if len(ts) == 0 and allow_empty:
        return None
    return ts[0].contiguous()

def shift_one(ts: List[torch.Tensor], allow_empty=False) -> torch.Tensor:
    if len(ts) == 0 and allow_empty:
        return None
    return (ts[0] + 1.0).contiguous()

def sp_0(t: torch.Tensor, tp: int, tp_rank: int, **kwargs: Any) -> List[torch.Tensor]:
    if (t.dim() == 3):
        return torch.split(t, t.shape[1] // tp, dim=1)[tp_rank]
    return torch.split(t, t.shape[0] // tp, dim=0)[tp_rank]

def sp_neg1(t: torch.Tensor, tp: int, tp_rank: int, **kwargs: Any) -> List[torch.Tensor]:
    return torch.split(t, t.shape[-1] // tp, dim=-1)[tp_rank]

def sp_id(t: torch.Tensor, tp: int, tp_rank: int, **kwargs: Any) -> List[torch.Tensor]:
    return t

# MHA layout: [D, head*size_per_head, head*size_per_head, head*size_per_head] == [D, 3, D] (sp_neg)
# MQA layout: [D, head*size_per_head, kv_head*size_per_head, kv_head*size_per_head] (sp_head)
def sp_head(t: torch.Tensor, tp: int, tp_rank: int, kv_broadcast: bool, **kwargs) -> List[torch.Tensor]:
    hidden_size = t.shape[0]
    qkv_hidden_size = t.shape[1]
    if t.dtype == torch.int32:
        return sp_neg1(t.reshape(hidden_size, 3, qkv_hidden_size // 3), tp, tp_rank)
    else:     
        if len(t.shape) == 2 and qkv_hidden_size != hidden_size * 3:
            kv_hidden_size = (qkv_hidden_size - hidden_size) // 2
            qs = sp_neg1(t[:,:hidden_size], tp, tp_rank)
            if kv_broadcast:
                ks = t[:,hidden_size:hidden_size + kv_hidden_size]
                vs = t[:,hidden_size + kv_hidden_size:]
            else:
                ks = sp_neg1(t[:,hidden_size:hidden_size + kv_hidden_size], tp, tp_rank)
                vs = sp_neg1(t[:,hidden_size + kv_hidden_size:], tp, tp_rank)
            return torch.concat([qs, ks, vs], dim=1).contiguous()
        else:
            return sp_neg1(t.reshape(hidden_size, 3, hidden_size), tp, tp_rank)

def sp_head_s(t: torch.Tensor, tp: int, tp_rank: int, hidden_size: int, kv_broadcast: bool, **kwargs) -> List[torch.Tensor]:
    return sp_neg1(t.reshape(t.shape[0], 3, t.shape[1] // 3), tp, tp_rank)

def sp_head_z(t: torch.Tensor, tp: int, tp_rank: int, hidden_size: int, kv_broadcast: bool, **kwargs) -> List[torch.Tensor]:
    return sp_neg1(t.reshape(t.shape[0], 3, t.shape[1] // 3), tp, tp_rank)

def sp_head_b(t: torch.Tensor, tp: int, tp_rank: int, hidden_size: int, kv_broadcast: bool, **kwargs) -> List[torch.Tensor]:
    t = t.reshape(-1)
    qk_hidden_size = (t.shape[0] - hidden_size) // 2
    qs = sp_neg1(t[:hidden_size], tp, tp_rank)
    if kv_broadcast:
        ks = t[hidden_size:hidden_size + qk_hidden_size]
        vs = t[hidden_size + qk_hidden_size:]
    else:
        ks = sp_neg1(t[hidden_size:hidden_size + qk_hidden_size], tp, tp_rank)
        vs = sp_neg1(t[hidden_size + qk_hidden_size:], tp, tp_rank)
    return torch.concat([qs, ks, vs], dim=0).contiguous()

def sp_head_lora(t: torch.Tensor, tp: int, tp_rank: int, hidden_size: int, kv_broadcast: bool, **kwargs) -> List[torch.Tensor]:
    # lora_b[dim0, 3*hidden_size]
    dim0 = t.shape[0]
    if len(t.shape) == 2 and t.shape[1] != hidden_size * 3:
        qk_hidden_size = (t.shape[1] - hidden_size) // 2
        qs = sp_neg1(t[:,:hidden_size], tp, tp_rank)
        if kv_broadcast:
            ks = t[:,hidden_size:hidden_size + qk_hidden_size]
            vs = t[:,hidden_size + qk_hidden_size:]
        else:
            ks = sp_neg1(t[:,hidden_size:hidden_size + qk_hidden_size], tp, tp_rank)
            vs = sp_neg1(t[:,hidden_size + qk_hidden_size:], tp, tp_rank)
        return torch.concat([qs, ks, vs], dim=1).contiguous()
    else:
        return sp_neg1(t.reshape(dim0, 3, hidden_size), tp, tp_rank)

def trans_qkv(ts: List[torch.Tensor], hidden_size: int, head_num: int, size_per_head: int = -1) -> torch.Tensor:
    if size_per_head == -1:
        size_per_head = hidden_size // head_num
    return ts[0].T.reshape(hidden_size, head_num, 3, size_per_head)\
        .permute(0, 2, 1, 3)\
        .reshape(hidden_size, 3, head_num * size_per_head)\
        .contiguous()

def trans_qkv_b(ts: List[torch.Tensor], hidden_size: int, head_num: int) -> torch.Tensor:
    return ts[0].reshape(head_num, 3, hidden_size // head_num)\
        .permute(1, 0, 2)\
        .reshape(3, hidden_size)\
        .contiguous()

def qkv_gather(ts: List[torch.Tensor], dim0, head_num: int, head_num_kv: int, size_per_head: int = -1) -> torch.Tensor:
    t = ts[0].t().contiguous().reshape(dim0, -1)
    if size_per_head == -1:
        size_per_head = t.shape[1] // (head_num + head_num_kv * 2)
    new_idxs = []
    q2kv_ratio = head_num // head_num_kv
    for q2kv_idx in range(head_num_kv):
        base_idx = (q2kv_ratio + 2) * q2kv_idx
        new_idxs.extend(list(range(base_idx, base_idx + q2kv_ratio)))
    for q2kv_idx in range(head_num_kv):
        new_idxs.append((q2kv_ratio + 2) * q2kv_idx + q2kv_ratio)
    for q2kv_idx in range(head_num_kv):
        new_idxs.append((q2kv_ratio + 2) * q2kv_idx + q2kv_ratio + 1)
    return t.reshape(dim0, head_num + head_num_kv * 2, size_per_head)[:,new_idxs,:].reshape(dim0, -1)

def sp_0_pad8(t: torch.Tensor, tp: int, tp_rank: int, **kwargs: Any) -> List[torch.Tensor]:
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

def trans_lora_qkv(ts: List[torch.Tensor], head_num: int, head_size: int):
    split = 3
    r = ts[0].shape[1]
    return ts[0].T.reshape(r, head_num, split, head_size).permute(0, 2, 1, 3).reshape(r, split, head_num * head_size).contiguous()

def merge_qkv_lora_A(ts: torch.Tensor):
    q, k, v = ts
    qkv_weight = torch.concat([q.T, k.T, v.T], dim=1).contiguous()
    return qkv_weight

def merge_qkv_lora_B(ts: List[torch.Tensor]):
    q, k, v = ts
    t_q = torch.zeros_like(q)
    t_k = torch.zeros_like(k)
    t_v = torch.zeros_like(v)
    return torch.cat((torch.cat((q,   t_q, t_q), dim=1),
                      torch.cat((t_k, k,   t_k), dim=1),
                      torch.cat((t_v, t_v, v  ), dim=1))).T.contiguous()


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
    attn_o_w = 'self_attention_weights.attention_output_weight.kernel'
    attn_o_b = 'self_attention_weights.attention_output_weight.bias'
    post_ln_gamma = 'post_layernorm_weights.gamma'
    post_ln_beta = 'post_layernorm_weights.beta'

    # ffn
    ffn_w1 = 'ffn_weights.intermediate_weight.kernel'
    ffn_b1 = 'ffn_weights.intermediate_weight.bias'
    ffn_w3 = 'ffn_weights.intermediate_weight3.kernel'
    ffn_b3 = 'ffn_weights.intermediate_weight3.bias'
    ffn_ln_gamma = 'ffn_weights.dense_layernorm.gamma'
    ffn_ln_beta = 'ffn_weights.dense_layernorm.beta'
    ffn_w2 = 'ffn_weights.intermediate_weight2.kernel'
    ffn_b2 = 'ffn_weights.intermediate_weight2.bias'
    ffn_gate = 'ffn_weights.gate.kernel'
    post_ffn_ln_gamma = "post_ffn_layernorm_weights.gamma"
    post_ffn_ln_beta = "post_ffn_layernorm_weights.beta"

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
    attn_o_z = 'self_attention_weights.attention_output_weight.zero'
    attn_o_s = 'self_attention_weights.attention_output_weight.weight_only_quant_scale'
    ffn_z1 = 'ffn_weights.intermediate_weight.zero'
    ffn_s1 = 'ffn_weights.intermediate_weight.weight_only_quant_scale'
    ffn_z3 = 'ffn_weights.intermediate_weight3.zero'
    ffn_s3 = 'ffn_weights.intermediate_weight3.weight_only_quant_scale'
    ffn_z2 = 'ffn_weights.intermediate_weight2.zero'
    ffn_s2 = 'ffn_weights.intermediate_weight2.weight_only_quant_scale'

    # medusa lm_head
    medusa_head = 'medusa_head'

    quant_w = set([
        attn_qkv_w,
        attn_o_w,
        ffn_w1,
        ffn_w2,
        ffn_w3,
    ])

    int4_quant_params = set([
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
    ])

    int8_attn_weights = [
        [attn_qkv_w, attn_qkv_s],
        [attn_o_w, attn_o_s],
    ]

    int8_ffn_weights = [
        [ffn_w1, ffn_s1],
        [ffn_w3, ffn_s3],
        [ffn_w2, ffn_s2], 
    ]

    int8_ffn_weights_2 = [
        [ffn_w1, ffn_s1],
        [ffn_w2, ffn_s2], 
    ]

    int4_attn_weights = [
        [attn_qkv_w, attn_qkv_z, attn_qkv_s],
        [attn_o_w, attn_o_z, attn_o_s],
    ]

    int4_ffn_weights = [
        [ffn_w1, ffn_z1, ffn_s1],
        [ffn_w3, ffn_z3, ffn_s3],
        [ffn_w2, ffn_z2, ffn_s2], 
    ]

    int4_ffn_weights_2 = [
        [ffn_w1, ffn_z1, ffn_s1],
        [ffn_w2, ffn_z2, ffn_s2], 
    ]

    moe_int8_quant_weights = [
        [ffn_w1, ffn_s1],
        [ffn_w3, ffn_s3],
        [ffn_w2, ffn_s2], 
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
        attn_qkv_w: sp_head,
        attn_qkv_z: sp_head_z,
        attn_qkv_s: sp_head_s,
        attn_qkv_b: sp_head_b,
        attn_o_w: sp_0,
        attn_o_z: sp_0,
        attn_o_s: sp_0,
        attn_o_b: sp_id,
        ffn_w1: sp_neg1,
        ffn_z1: sp_neg1,
        ffn_s1: sp_neg1,
        ffn_b1: sp_neg1,
        ffn_w3: sp_neg1,
        ffn_z3: sp_neg1,
        ffn_s3: sp_neg1,
        ffn_b3: sp_neg1,
        ffn_w2: sp_0,
        ffn_z2: sp_0,
        ffn_s2: sp_0,
        ffn_b2: sp_id,
        post_ln_beta: sp_id,
        post_ln_gamma: sp_id,
        positional_embedding: sp_id,
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
        ffn_gate: sp_id,
        post_ffn_ln_beta: sp_id,
        post_ffn_ln_gamma: sp_id,
        token_type_embedding: sp_id        
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
        ffn_gate
    ]

    skip_weights_list = [
        attn_qkv_w,
        attn_qkv_b,
        attn_ln_gamma,
        attn_ln_beta,
        attn_o_w,
    ]



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

    def __init__(self, name: str, weights: List[CkptWeightInfo], process_fun: Callable[[List[torch.Tensor]], torch.Tensor] = identity) -> None:
        self.name = name
        self.weights = weights
        self.process_fun = process_fun

    def get_ckpt_tensor_names(self) -> List[str]:
        if not bool(self.weights):
            return []
        return [ckpt.name for ckpt in self.weights]

    def __str__(self) -> str:
        return f"WeightInfo[{self.name}]{self.weights}"

    def __repr__(self) -> str:
        return self.__str__()

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
                for layer_weight in layer_weights:

                    if layer_weight.name == lora_name:
                        ckpt_layer_weight = layer_weight.weights

                assert (ckpt_layer_weight != None)
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
                    lora_layer_weights.append(WeightInfo(lora_name + "." + lora_a_b, [ckpt_weight_info], transpose))
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
        self._head_num = config.head_num
        self._head_num_kv = config.head_num_kv
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self._size_per_head = config.size_per_head
        if self._head_num_kv == -1:
            self._head_num_kv = self._head_num
        self._int8_mode = config.quant_algo.int8_mode
        self._int4_mode = config.quant_algo.int4_mode
        self._is_quant_mode = config.is_quant_mode
        self._is_gptq = config.quant_algo.is_gptq
        self._is_awq = config.quant_algo.is_awq
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
                               tp_strategy=W.gpt_style_tp_strategy)

    def get_weight_info(self, preprocessed: bool, all_names: Set[str]) -> ModelWeightInfo:
        if preprocessed:
            logging.info("Using preprocessed weight info")
            return self.get_preprocessed_weight_info(all_names)
        else:
            weight_info = self._get_weight_info()
            if self._is_sparse_head:
                logging.info("Skiping load empty weight for head_num == 0")
                weight_info = self._process_sparse_weight(weight_info)
            if self._is_medusa_model:
                weight_info = self._add_medusa_head_info(weight_info)
            return weight_info

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

    def process_meta(self, ckpt_metas: List[CkptFileInfo]):
        if len(ckpt_metas) == 0:
            return
        if 'ft_module' not in ckpt_metas[0].get_tensor_names():
            # call subclass process_meta
            self.fix_megatron_layer_id(ckpt_metas)
            meta_dicts = [ckpt_file.get_metadata() for ckpt_file in ckpt_metas]
            weight_keys = set(reduce(lambda x,y:x+y, [list(meta.keys()) for meta in meta_dicts], []))
            self._process_meta(meta_dicts, weight_keys)

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

class LoRAWeightInfo(NamedTuple):
    layer_weights: List[WeightInfo]
    weights: List[WeightInfo]
    tp_strategy: Optional[Dict[Any, Any]]

class LoRAModelWeightInfo():

    def __init__(self, config: Any, tp_size: int, tp_rank: int):
        self.config = config
        self._num_layers = config.num_layers
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.tp_split_emb_and_lm_head = config.tp_split_emb_and_lm_head

    def process_meta(self, meta_dict: Any):
        pass

    def get_weight_info(self, *args: Any, **kwargs: Any):
       pass


class LoRAWeights:

    def __init__(self, num_layers: int) -> None:
        self.lora_a_weights: List[Dict[str, torch.Tensor]] = []
        self.lora_b_weights: List[Dict[str, torch.Tensor]] = []
        self.lora_rank = 0
        for _ in range(num_layers):
            self.lora_a_weights.append({})
            self.lora_b_weights.append({})

    def set_lora_rank(self, lora_rank: int):
        self.lora_rank = lora_rank

    def append_layer_weight(self, int8_flag: bool, layer_id: int, name: str, tensor: torch.Tensor):
        assert not int8_flag, "LoRA does not support int8 mode"
        prefix_name = name[:-len(".lora_A")]
        if name.endswith('.lora_A'):
            self.lora_a_weights[layer_id][prefix_name] = tensor
        elif name.endswith('.lora_B'):
            self.lora_b_weights[layer_id][prefix_name] = tensor
        else:
            raise ValueError(f"Invalid lora weight name: {name}")

    def apply_scale(self, scale: float):
        logging.info(f"scale size {scale}")
        for i, layer_weights in enumerate(self.lora_b_weights):
            for name, weight in layer_weights.items():
                self.lora_b_weights[i][name] = weight * scale

class LoRAMap():

    def __init__(self) -> None:
        self.name_id_map: Dict[str, int] = {}
        self.weights_map: Dict[int, LoRAWeights] = {}

        self.lora_cnt = 0
        self.max_rank = 0

    def _create_id(self, name: str):
        if name not in self.name_id_map:
            id = self.lora_cnt
            self.lora_cnt += 1
            self.name_id_map[name] = id

    def get_id(self, name: str) -> int:
        if name == "":
            return -1
        if name not in self.name_id_map:
            raise Exception(f"lora map has no adapter model: {name}")
        return self.name_id_map[name]

    def has_id(self, name: str) -> bool:
        if name not in self.name_id_map:
            return False
        return True

    def add_lora_name(self, name: str, weights: LoRAWeights) -> int:
        self._create_id(name)
        id = self.name_id_map[name]
        self.weights_map[id] = weights
        return id

    def remove_lora_name(self, name: str):
        if name in self.name_id_map:
            id = self.name_id_map[name]
            del self.name_id_map[name]
            del self.weights_map[id]
            return id

    def split(self, lora_names: List[str]):
        result = LoRAMap()
        for lora_name in lora_names:
            lora_id = self.name_id_map.get(lora_name)
            if lora_id != None:
                result.lora_cnt += 1
                result.name_id_map[lora_name] = lora_id
                result.weights_map[lora_id] = self.weights_map[lora_id]
                if self.weights_map[lora_id].lora_rank > result.max_rank:
                    result.max_rank = self.weights_map[lora_id].lora_rank
        return result

class LoraCountException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

class LoraPathException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

class LoraResource():
    def __init__(self, lora_infos: Dict[str, str] = dict(), database: Optional[BaseDatabase] = None,
                 weights_info: Optional[WeightInfo] = None,
                 lora_map: Optional[LoRAMap] = None):
        self.lora_infos = lora_infos
        self.database = database
        self.model_weights_loader = None
        self.weights_info = weights_info
        self.ft_op = []
        self.lora_map = lora_map

        self.max_lora_model_size = int(os.environ.get("MAX_LORA_MODEL_SIZE", "-1"))
        self.rlock_map: Dict[str, RWLock] = {}
        self.to_add_lora_id = list()
        self.to_remove_lora_id = list()
        self.stream = torch.cuda.Stream()

    def clear_for_update(self):
        self.to_add_lora_id.clear()
        self.to_remove_lora_id.clear()

    def delete_rlock(self, name: str) -> None:
        if name in self.rlock_map:
            del self.rlock_map[name]

    def add_lora_name(self, name: str, weights: LoRAWeights) -> int:
        id = self.lora_map.add_lora_name(name, weights)
        self.rlock_map[name] = RWlock()
        self.to_add_lora_id.append(id)
        return id

    def remove_lora_name(self, name: str):
        id = self.lora_map.remove_lora_name(name)
        self.to_remove_lora_id.append(id)

    def check_remaining_lora(self, lora_infos: Dict[str, str]):
        for lora_name, old_lora_path in self.lora_infos.items():
            if lora_name in lora_infos:
                if old_lora_path != lora_infos[lora_name]:
                    raise LoraPathException(f'lora[{lora_name}]\'s path is changing from [{old_lora_path}] to [{lora_infos[lora_name]}]')

    def remove_old_lora(self, lora_infos: Dict[str, str]):
        remove_lora_names = list()
        for lora_name, lora_path in self.lora_infos.items():
            if lora_name not in lora_infos:
                self.database.remove_lora(lora_name)
                remove_lora_names.append(lora_name)

        for lora_name in remove_lora_names:
            self.write_acquire(lora_name)
        try:
            self.clear_for_update()
            for lora_name in remove_lora_names:
                self.remove_lora_name(lora_name)
            for op in self.ft_op:
                op.update_lora()
        finally:
            for lora_name in remove_lora_names:
                self.write_release(lora_name)
        for lora_name in remove_lora_names:
            self.delete_rlock(lora_name)

    def add_new_lora(self, lora_infos: Dict[str, str]):
        for lora_name, lora_path in lora_infos.items():
            if lora_name not in self.lora_infos:
                self.database.load_lora(lora_name, lora_path)
        self.clear_for_update()
        for lora_config in self.database.LoraFileList.keys():
            lora_name = lora_config.name
            if self.lora_map.has_id(lora_name):
                continue
            lora_weights = self.model_weights_loader.load_lora_weights_from_scratch(lora_name,  self.weights_info._int8_mode, 'cuda:0')
            _ = self.add_lora_name(lora_name, lora_weights)
        for op in self.ft_op:
            op.update_lora()

    def update(self, lora_infos: Dict[str, str]):
        if self.max_lora_model_size != -1 and len(lora_infos) > self.max_lora_model_size:
            raise LoraCountException(f'lora_infos[{lora_infos}]\'s size exceed MAX_LORA_MODEL_SIZE[{self.max_lora_model_size}]')
        with torch.cuda.stream(self.stream):
            self.check_remaining_lora(lora_infos)
            self.remove_old_lora(lora_infos)
            self.add_new_lora(lora_infos)
        self.lora_infos = lora_infos

    def get_id(self, name: str) -> int:
        if self.lora_map != None:
            return self.lora_map.get_id(name)

    def read_acquire(self, lora_name: str):
        if lora_name in self.rlock_map:
            self.rlock_map[lora_name].read_acquire()

    def read_release(self, lora_name: str):
        if lora_name in self.rlock_map:
            self.rlock_map[lora_name].read_release()

    def write_acquire(self, lora_name: str):
        if lora_name in self.rlock_map:
            self.rlock_map[lora_name].write_acquire()

    def write_release(self, lora_name: str):
        if lora_name in self.rlock_map:
            self.rlock_map[lora_name].write_release()

class ModelWeights:
    def __init__(self, num_layers: int):
        self.weights: List[Dict[str, torch.Tensor]] = []
        self._pytorch_weights: Dict[str, torch.Tensor] = {}
        self.lora_resource: LoraResource = LoraResource()
        self._dtype = None

        for i in range(num_layers):
            self.weights.append({})

    def append_pytorch_weight(self, name: str, tensor: torch.Tensor):
        self._dtype = tensor.dtype
        self._pytorch_weights[name] = tensor

    def steal_pytorch_weight(self, name: str):
        if name not in self._pytorch_weights:
            return None
        t = self._pytorch_weights[name]
        del self._pytorch_weights[name]
        return t

    def has_pytorch_weight(self, name: str):
        return name in self._pytorch_weights

    def append_layer_weight(self, layer_id: int, name: str, tensor: torch.Tensor):
        self.weights[layer_id][name] = tensor

    @property
    def device(self):
        return 'cuda:0'

    @property
    def dtype(self):
        return self._dtype

class LoraResourceHolder:
    def __init__(self, lora_resource, adapter_name):
        self._lora_resource = lora_resource
        self._adapter_name = adapter_name
        self._lora_resource.read_acquire(self._adapter_name)
        self._lora_id = self._lora_resource.get_id(self._adapter_name)

    @property
    def lora_id(self):
        return self._lora_id

    def release(self):
        self._lora_resource.read_release(self._adapter_name)