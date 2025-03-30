from typing import Any, List
import functools

from maga_transformer.utils.model_weight import (W, WeightInfo, CkptWeightInfo,
                                                 identity, transpose, pad,
                                                 merge_qkv_hf, stack_, stack_moe_w1)

QW_SUFFIX = '.qweight'
QZ_SUFFIX = '.qzeros'
QS_SUFFIX = '.scales'
W_SUFFIX = '.weight'


def get_qkv_quant_weight_info(weights: List[CkptWeightInfo]) -> List[WeightInfo]:
    assert len(weights) == 1 or len(weights) == 3
    if len(weights) == 3:
        q_name = weights[0].name
        k_name = weights[1].name
        v_name = weights[2].name
        assert q_name.endswith(W_SUFFIX) and k_name.endswith(
            W_SUFFIX) and v_name.endswith(W_SUFFIX)
        q_name = q_name[:-len(W_SUFFIX)]
        k_name = k_name[:-len(W_SUFFIX)]
        v_name = v_name[:-len(W_SUFFIX)]
        return [
            WeightInfo(W.attn_qkv_w, [
                CkptWeightInfo(q_name + QW_SUFFIX, transpose),
                CkptWeightInfo(k_name + QW_SUFFIX, transpose),
                CkptWeightInfo(v_name + QW_SUFFIX, transpose)
            ], functools.partial(merge_qkv_hf)),
            WeightInfo(W.attn_qkv_z, [
                CkptWeightInfo(q_name + QZ_SUFFIX, transpose),
                CkptWeightInfo(k_name + QZ_SUFFIX, transpose),
                CkptWeightInfo(v_name + QZ_SUFFIX, transpose)
            ], functools.partial(merge_qkv_hf)),
            WeightInfo(W.attn_qkv_s, [
                CkptWeightInfo(q_name + QS_SUFFIX, transpose),
                CkptWeightInfo(k_name + QS_SUFFIX, transpose),
                CkptWeightInfo(v_name + QS_SUFFIX, transpose)
            ], functools.partial(merge_qkv_hf))
        ]
    else:
        qkv_name = weights[0].name
        assert qkv_name.endswith(W_SUFFIX)
        qkv_name = qkv_name[:-len(W_SUFFIX)]
        return [
            WeightInfo(W.attn_qkv_w,
                       [CkptWeightInfo(qkv_name + QW_SUFFIX, identity)],
                       identity),
            WeightInfo(W.attn_qkv_z,
                       [CkptWeightInfo(qkv_name + QZ_SUFFIX, identity)],
                       identity),
            WeightInfo(W.attn_qkv_s, [CkptWeightInfo(qkv_name + QS_SUFFIX)],
                       identity)
        ]

def get_qkv_quant_weight_info(weights: List[CkptWeightInfo], quant_algo: Any, QW_SUFFIX = '.qweight', QZ_SUFFIX = '.qzeros', QS_SUFFIX = '.scales', W_SUFFIX = '.weight') -> List[WeightInfo]:
    assert weights[0].name.endswith(W_SUFFIX)
    w_name = weights[0].name[:-len(W_SUFFIX)]
    group_size = quant_algo.getGroupSize()
    pad_div = 32 // quant_algo.getWeightBits()
    is_awq = quant_algo.isAwq()
    is_gptq = quant_algo.isGptq()
    is_fp8 = quant_algo.isFp8()
    

def get_ffn_quant_weight_info(weights: List[CkptWeightInfo], quant_algo: Any,
                              ffn_w_name: str, inter_padding_size: int) -> List[WeightInfo]:
    assert weights[0].name.endswith(W_SUFFIX)
    assert ffn_w_name in [W.ffn_w1, W.ffn_w2, W.ffn_w3, W.moe_w1, W.moe_w2]
    if ffn_w_name in [W.ffn_w1, W.ffn_w2, W.ffn_w3]:
        assert len(weights) == 1
    w_name = weights[0].name[:-len(W_SUFFIX)]
    group_size = quant_algo.getGroupSize()
    pad_div = 32 // quant_algo.getWeightBits()
    is_awq = quant_algo.isAwq()
    is_gptq = quant_algo.isGptq()
    if ffn_w_name == W.ffn_w2:
        return [
            WeightInfo(
                W.ffn_w2, [CkptWeightInfo(w_name + QW_SUFFIX, identity)],
                functools.partial(pad,
                                  inter_padding_size=inter_padding_size //
                                  pad_div if is_gptq else inter_padding_size,
                                  dim=0)),
            WeightInfo(
                W.ffn_z2, [CkptWeightInfo(w_name + QZ_SUFFIX, identity)],
                functools.partial(pad,
                                  inter_padding_size=inter_padding_size //
                                  group_size,
                                  dim=0)),
            WeightInfo(
                W.ffn_s2, [CkptWeightInfo(w_name + QS_SUFFIX, identity)],
                functools.partial(pad,
                                  inter_padding_size=inter_padding_size //
                                  group_size,
                                  dim=0))
        ]
    elif ffn_w_name in [W.moe_w2, W.moe_w1]:
        if ffn_w_name == W.moe_w1:
            w, z, s = (W.moe_w1, W.moe_z1, W.moe_s1)
            stack = stack_moe_w1
        elif ffn_w_name == W.moe_w2:
            w, z, s = (W.moe_w2, W.moe_z2, W.moe_s2)
            stack = stack_

        w_name = [weight.name[:-len(W_SUFFIX)] for weight in weights]
        return [
            WeightInfo(
                w, [CkptWeightInfo(name + QW_SUFFIX, transpose) \
                    for name in w_name], stack),
            WeightInfo(
                z, [CkptWeightInfo(name + QZ_SUFFIX, transpose) \
                    for name in w_name], stack),
            WeightInfo(
                s, [CkptWeightInfo(name + QS_SUFFIX, transpose) \
                    for name in w_name], stack),
        ]

    else:
        w, z, s = (W.ffn_w1, W.ffn_z1,
                   W.ffn_s1) if ffn_w_name == W.ffn_w1 else (W.ffn_w3,
                                                             W.ffn_z3,
                                                             W.ffn_s3)
        return [
            WeightInfo(
                w, [CkptWeightInfo(w_name + QW_SUFFIX, identity)],
                functools.partial(pad,
                                  inter_padding_size=inter_padding_size //
                                  pad_div if is_awq else inter_padding_size,
                                  dim=1)),
            WeightInfo(
                z, [CkptWeightInfo(w_name + QZ_SUFFIX, identity)],
                functools.partial(pad,
                                  inter_padding_size=inter_padding_size //
                                  pad_div,
                                  dim=1)),
            WeightInfo(
                s, [CkptWeightInfo(w_name + QS_SUFFIX, identity)],
                functools.partial(pad,
                                  inter_padding_size=inter_padding_size,
                                  dim=1))
        ]


def get_layer_group_quant_weight_info(
        layer_weights: List[WeightInfo], quant_algo: Any,
        inter_padding_size: int) -> List[WeightInfo]:
    quant_weights: List[WeightInfo] = []
    for weight_info in layer_weights:
        if weight_info.name == W.attn_qkv_w:
            quant_weights.extend(get_qkv_quant_weight_info(
                weight_info.weights))
        elif weight_info.name == W.attn_o_w:
            w_name = weight_info.weights[0].name[:-len(W_SUFFIX)]
            quant_weights.extend([
                WeightInfo(W.attn_o_w,
                           [CkptWeightInfo(w_name + QW_SUFFIX, identity)],
                           identity),
                WeightInfo(W.attn_o_z,
                           [CkptWeightInfo(w_name + QZ_SUFFIX, identity)],
                           identity),
                WeightInfo(W.attn_o_s,
                           [CkptWeightInfo(w_name + QS_SUFFIX, identity)],
                           identity)
            ])
        elif weight_info.name in W.mla_quant_weights:
            quant_weights.extend(get_mla_quant_weight_info(
                weight_info.weights))
        elif weight_info.name in [W.ffn_w1, W.ffn_w2, W.ffn_w3, W.moe_w1, W.moe_w2]:
            quant_weights.extend(
                get_ffn_quant_weight_info(weight_info.weights, quant_algo,
                                          weight_info.name,
                                          inter_padding_size))
        else:
            quant_weights.append(weight_info)
    return quant_weights
