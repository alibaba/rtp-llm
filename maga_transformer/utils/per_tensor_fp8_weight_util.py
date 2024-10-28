from typing import Any, List
import functools
import torch

from maga_transformer.utils.model_weight import (W, WeightInfo, CkptWeightInfo,
                                                 identity, transpose, pad,
                                                 merge_qkv_hf, stack_,
                                                 get_tensor_reciprocal, get_tensor_from_scalar)

QW_SUFFIX = '.qweight'
QZ_SUFFIX = '.qzeros'
QS_SUFFIX = '.scales'
W_SUFFIX = '.weight'

FP8_SCALE_MAP = {
    W.attn_qkv_w : W.attn_qkv_s,
    W.attn_o_w : W.attn_o_s,
    W.ffn_w3 : W.ffn_s3,
    W.ffn_w2 : W.ffn_s2,
    W.ffn_w1 : W.ffn_s1,
}

FP8_ACT_SCALE_MAP = {
    W.attn_qkv_w : [
        (W.pre_ln_static_quant, get_tensor_reciprocal),
        (W.pre_ln_static_quant_reciprocal, get_tensor_from_scalar),
    ],
    W.attn_o_w : [
        (W.attention_output_static_quant, get_tensor_from_scalar),
        (W.attention_output_static_quant_reciprocal, get_tensor_reciprocal),
    ],
    W.ffn_w2 : [
        (W.ffn_intermediate_weight2_static_quant, get_tensor_from_scalar),
        (W.ffn_intermediate_weight2_static_quant_reciprocal, get_tensor_from_scalar)
    ],
    W.ffn_w3 : [
        (W.post_ln_static_quant, get_tensor_reciprocal),
        (W.post_ln_static_quant_reciprocal, get_tensor_from_scalar)
    ]
}

TRT_ENGINE_LAYER_WEIGHT_MAP = {
    W.pre_ln_beta : 'transformer.layers.{i}.input_layernorm.bias',
    W.pre_ln_gamma : 'transformer.layers.{i}.input_layernorm.weight',
    W.attn_qkv_w : 'transformer.layers.{i}.attention.qkv.weight',
    W.attn_qkv_b : 'transformer.layers.{i}.attention.qkv.bias',
    W.attn_qkv_s : 'transformer.layers.{i}.attention.qkv.weights_scaling_factor',

    W.attn_o_w : 'transformer.layers.{i}.attention.dense.weight',
    W.attn_o_b : 'transformer.layers.{i}.attention.dense.bias',
    W.attn_o_s : 'transformer.layers.{i}.attention.dense.weights_scaling_factor',

    W.ffn_w3 : 'transformer.layers.{i}.mlp.fc.weight',
    W.ffn_b3 : 'transformer.layers.{i}.mlp.fc.bias',
    W.ffn_s3 : 'transformer.layers.{i}.mlp.fc.weights_scaling_factor',

    W.ffn_w2 : 'transformer.layers.{i}.mlp.proj.weight',
    W.ffn_b2 : 'transformer.layers.{i}.mlp.proj.bias',
    W.ffn_s2 : 'transformer.layers.{i}.mlp.proj.weights_scaling_factor',

    W.post_ln_gamma : 'transformer.layers.{i}.post_layernorm.weight',
    W.post_ln_beta : 'transformer.layers.{i}.post_layernorm.bias',

}
def get_layer_per_tensor_fp8_scale_weight_info(weights: List[CkptWeightInfo]) -> List[WeightInfo]:
    return _get_per_tensor_fp8_scale_weight_info(weights)

def get_trt_engine_layer_weight_info(weights: List[CkptWeightInfo]):
    layer_weights = []
    for weight_info in weights:
        if weight_info.name in TRT_ENGINE_LAYER_WEIGHT_MAP:
            layer_weights.append(WeightInfo(weight_info.name, [CkptWeightInfo(TRT_ENGINE_LAYER_WEIGHT_MAP[weight_info.name], identity)], identity))

    return layer_weights

def _get_per_tensor_fp8_scale_weight_info(weights: List[CkptWeightInfo]):
    quant_weights: List[WeightInfo] = []
    for weight_info in weights:
        w_name = weight_info.name
        tensor_names = [_.name for _ in weight_info.weights]
        if w_name in FP8_SCALE_MAP:
            scale_name = _get_weight_scale_name(w_name, tensor_names, "weights_scaling_factor")
            quant_weights.extend(
                [WeightInfo(FP8_SCALE_MAP[w_name], [CkptWeightInfo(scale_name, identity)], identity)]
            )
        if w_name in FP8_ACT_SCALE_MAP:
            for act_w in FP8_ACT_SCALE_MAP[w_name]:
                act_name = _get_weight_scale_name(w_name, tensor_names, "activation_scaling_factor")
                quant_weights.extend(
                    [WeightInfo(act_w[0], [CkptWeightInfo(act_name, identity)], act_w[1], torch.float32)]
                )
    return quant_weights


def _get_weight_scale_name(w_name, tensor_names: List[str], suffix):
    if w_name == W.attn_qkv_w:
        assert len(tensor_names) == 1 or len(tensor_names) == 3
    else:
        assert len(tensor_names) == 1
    parts = tensor_names[0].rsplit('.', 1)

    # 将最后一部分替换为suffix
    parts[-1] = suffix

    # 重新组合字符串
    scale_name = '.'.join(parts)
    return scale_name

