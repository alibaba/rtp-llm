import functools
import torch
from typing import Any, Dict, List, Optional, Union
from maga_transformer.model_loader.ffn_weight import FfnAtomicWeight
from maga_transformer.model_loader.attn_weight import AttnAtomicWeight, MlaAttnAtomicWeight
from maga_transformer.model_loader.w8a8_weight import create_w8a8_int8_weight
from maga_transformer.model_loader.load_config import LoadConfig
from maga_transformer.model_loader.weight_module import WeightModule, AtomicWeight, CompositeWeight, QuantWeight
from maga_transformer.model_loader.ffn_weight import FfnAtomicWeight
from maga_transformer.model_loader.attn_weight import AttnAtomicWeight
from maga_transformer.utils.model_weight import W, CkptWeightInfo, identity, merge_qkv_hf, ones, stack_, stack_moe_w1, transpose, transpose_w13, concat_w13

QW_SUFFIX = '.qweight'
QS_SUFFIX = '.scales'
W_SUFFIX = '.weight'
B_SUFFIX = ".bias"
SMOOTHER_SUFFIX = '.smoother'
SHIFT_SUFFIX='.shift'
def get_qkv_quant_weight_info(src_weight: Union[AttnAtomicWeight, MlaAttnAtomicWeight]) -> List[AtomicWeight]:
    weights = src_weight.weights
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
            create_w8a8_int8_weight(src_weight, W.attn_qkv_w, [
                CkptWeightInfo(q_name + QW_SUFFIX, identity),
                CkptWeightInfo(k_name + QW_SUFFIX, identity),
                CkptWeightInfo(v_name + QW_SUFFIX, identity)
            ], functools.partial(merge_qkv_hf), data_type=torch.int8, config=src_weight.config),
            create_w8a8_int8_weight(src_weight, W.attn_qkv_b, [
                CkptWeightInfo(q_name + QS_SUFFIX, identity),
                CkptWeightInfo(k_name + QS_SUFFIX, identity),
                CkptWeightInfo(v_name + QS_SUFFIX, identity)
            ], functools.partial(merge_qkv_hf), data_type=torch.float32, config=src_weight.config),
            create_w8a8_int8_weight(src_weight, W.attn_qkv_s, [
                CkptWeightInfo(q_name + QS_SUFFIX, identity),
                CkptWeightInfo(k_name + QS_SUFFIX, identity),
                CkptWeightInfo(v_name + QS_SUFFIX, identity)
            ], functools.partial(merge_qkv_hf), data_type=torch.float32, config=src_weight.config)
        ]
    else:
        qkv_name = weights[0].name
        assert qkv_name.endswith(W_SUFFIX)
        qkv_name = qkv_name[:-len(W_SUFFIX)]
        return [
            create_w8a8_int8_weight(src_weight, W.attn_qkv_w, [CkptWeightInfo(qkv_name + QW_SUFFIX)], transpose, data_type=torch.int8, config=src_weight.config),
            create_w8a8_int8_weight(src_weight, W.attn_qkv_b, [CkptWeightInfo(qkv_name + B_SUFFIX)], identity, data_type=torch.float32, config=src_weight.config),
            create_w8a8_int8_weight(src_weight, W.attn_qkv_s, [CkptWeightInfo(qkv_name + QS_SUFFIX)], identity, data_type=torch.float32, config=src_weight.config)
        ]


def get_ffn_quant_weight_info(src_weight: FfnAtomicWeight, quant_algo: Any) -> List[AtomicWeight]:
    weights = src_weight.weights
    ffn_w_name = src_weight.name
    assert weights[0].name.endswith(W_SUFFIX)
    assert ffn_w_name in [W.ffn_w1, W.ffn_w2, W.ffn_w3, W.moe_w1, W.moe_w2]

    if ffn_w_name in [W.ffn_w1, W.ffn_w2, W.ffn_w3]:
        assert len(weights) == 1
    w_name = weights[0].name[:-len(W_SUFFIX)]

    if ffn_w_name in [W.moe_w2, W.moe_w1]:
        if ffn_w_name == W.moe_w1:
            w, b, s = (W.moe_w1, W.moe_b1, W.moe_s1)
            stack = stack_moe_w1
        elif ffn_w_name == W.moe_w2:
            w, b, s = (W.moe_w2, W.moe_b2, W.moe_s2)
            stack = stack_

        w_name = [weight.name[:-len(W_SUFFIX)] for weight in weights]
        return [
            create_w8a8_int8_weight(src_weight,
                w, [CkptWeightInfo(name + QW_SUFFIX, transpose) \
                    for name in w_name], stack, data_type=torch.int8,
                    config=src_weight.config),
            create_w8a8_int8_weight(src_weight,
                b, [CkptWeightInfo(name + B_SUFFIX, identity) \
                    for name in w_name], stack, data_type=torch.float32,
                    config=src_weight.config),
           create_w8a8_int8_weight(src_weight,
                s, [CkptWeightInfo(name + QS_SUFFIX, identity) \
                    for name in w_name], stack, data_type=torch.float32,
                    config=src_weight.config)
        ]

    elif ffn_w_name == W.ffn_w2:
        w, b, s = (W.ffn_w2, W.ffn_b2, W.ffn_s2)

        return [
            create_w8a8_int8_weight(src_weight,
                w, [CkptWeightInfo(w_name + QW_SUFFIX, identity)],
                transpose, data_type=torch.int8,
                    config=src_weight.config),
            None,
            create_w8a8_int8_weight(src_weight,
                s, [CkptWeightInfo(w_name + QS_SUFFIX, identity)],
                    identity, data_type=torch.float32,
                    config=src_weight.config)
        ]
    elif ffn_w_name == W.ffn_w13:
        w, b, s = (W.ffn_w13, W.ffn_b13, W.ffn_s13)
        w1_name = weights[0].name[:-len(W_SUFFIX)]
        w3_name = weights[1].name[:-len(W_SUFFIX)]
        return [
            create_w8a8_int8_weight(src_weight,
                w, [CkptWeightInfo(w1_name + QW_SUFFIX, identity), CkptWeightInfo(w3_name + QW_SUFFIX, identity)],
                transpose_w13, data_type=torch.int8,
                    config=src_weight.config),
            create_w8a8_int8_weight(src_weight,
                b, [CkptWeightInfo(w1_name + B_SUFFIX, identity), CkptWeightInfo(w3_name + B_SUFFIX, identity)],
                concat_w13, data_type=torch.float32,
                config=src_weight.config),
            create_w8a8_int8_weight(src_weight,
                s, [CkptWeightInfo(w1_name + QS_SUFFIX, identity), CkptWeightInfo(w3_name + QS_SUFFIX, identity)],
                concat_w13, data_type=torch.float32,
                config=src_weight.config)
        ]
    else:
        if ffn_w_name == W.ffn_w1:
            w, b, s = (W.ffn_w1, W.ffn_b1, W.ffn_s1)
        elif ffn_w_name == W.ffn_w3:
            w, b, s = (W.ffn_w3, W.ffn_b3, W.ffn_s3)
        else:
            raise NotImplementedError(
                f"ffn_w_name {ffn_w_name} not supported in omni_quant")
        return [
            create_w8a8_int8_weight(src_weight,
                w, [CkptWeightInfo(w_name + QW_SUFFIX, identity)],
                transpose, data_type=torch.int8,
                    config=src_weight.config),
            create_w8a8_int8_weight(src_weight,
                b, [CkptWeightInfo(w_name + B_SUFFIX, identity)],
                    identity, data_type=torch.float32,
                    config=src_weight.config),
            create_w8a8_int8_weight(src_weight,
                s, [CkptWeightInfo(w_name + QS_SUFFIX, identity)],
                    identity, data_type=torch.float32,
                    config=src_weight.config)
        ]


class OmniQuantWeightInfo(CompositeWeight, QuantWeight):
    w8a8_weight_list: List[str] = [
            W.pre_ln_gamma,
            W.attn_qkv_w,
            W.attn_o_w,
            W.ffn_w1,
            W.ffn_w3,
            W.ffn_w2,
            W.post_ln_gamma
    ]

    @classmethod
    def support(cls, quant_algo: Any, src_weight_info: WeightModule) -> bool:
        name = src_weight_info.name
        return  quant_algo.isOmniQuant() and name in cls.w8a8_weight_list

    def __init__(self, src_weight_info: AtomicWeight, quant_algo: Any, *args, **kwargs):
        self.quant_algo = quant_algo
        kernel: AtomicWeight = None
        scale: Optional[AtomicWeight] = None
        bias: Optional[AtomicWeight] = None  # TODO(luoli.hn) 有些有bias，有些没有bias，最好的方案是在AutoWeight再implement LinearWeight 之类的，改动比较多，先串通
        smoother: Optional[AtomicWeight] = None
        shift: Optional[AtomicWeight] = None
        if src_weight_info.name == W.attn_qkv_w:
            (kernel, bias, scale) = get_qkv_quant_weight_info(src_weight_info)
        elif src_weight_info.name == W.attn_o_w:
            w_name = src_weight_info.weights[0].name[:-len(W_SUFFIX)]
            kernel = create_w8a8_int8_weight(src_weight_info, W.attn_o_w,
                           [CkptWeightInfo(w_name + QW_SUFFIX, identity)],
                           transpose, data_type=torch.int8, config=src_weight_info.config)
            scale = create_w8a8_int8_weight(src_weight_info, W.attn_o_s, [CkptWeightInfo(w_name + QS_SUFFIX)], identity, data_type=torch.float32, config=src_weight_info.config)
            bias = create_w8a8_int8_weight(src_weight_info, W.attn_o_b, [CkptWeightInfo(w_name + B_SUFFIX)], identity, config=src_weight_info.config)
            smoother = create_w8a8_int8_weight(src_weight_info, W.attn_o_smoother, [CkptWeightInfo(w_name + SMOOTHER_SUFFIX)], identity, data_type=torch.float32, config=src_weight_info.config)
            shift = create_w8a8_int8_weight(src_weight_info, W.attn_o_shift, [CkptWeightInfo(w_name + SHIFT_SUFFIX)], identity, data_type=torch.float32, config=src_weight_info.config)
        elif src_weight_info.name in [W.ffn_w1, W.ffn_w2, W.ffn_w3, W.ffn_w13, W.moe_w1, W.moe_w2]:
            (kernel, bias, scale) = get_ffn_quant_weight_info(src_weight_info, quant_algo)
            if src_weight_info.name == W.ffn_w2:
                smoother = create_w8a8_int8_weight(src_weight_info, W.ffn_smoother, [], functools.partial(ones, shape=src_weight_info.config.inter_padding_size), data_type=torch.float32, config=src_weight_info.config)
        elif src_weight_info.name == W.pre_ln_gamma:
            w_name = src_weight_info.weights[0].name[:-len(W_SUFFIX)]
            kernel = src_weight_info
            bias = create_w8a8_int8_weight(src_weight_info, W.pre_ln_beta, [CkptWeightInfo(w_name + B_SUFFIX)], identity)
        elif src_weight_info.name == W.post_ln_gamma:
            w_name = src_weight_info.weights[0].name[:-len(W_SUFFIX)]
            kernel = src_weight_info
            bias = create_w8a8_int8_weight(src_weight_info, W.post_ln_beta, [CkptWeightInfo(w_name + B_SUFFIX)], identity)
        else:
            raise ValueError(f"Unsupported weight name {src_weight_info.name}")

        sub_weights = {
            kernel.name: kernel,
        }
        if bias is not None:
            sub_weights[bias.name] = bias
        if scale is not None:
            sub_weights[scale.name] = scale
        if smoother is not None:
            sub_weights[smoother.name] = smoother
        if shift is not None:
            sub_weights[shift.name] = shift

        super().__init__(sub_weights, quant_algo=quant_algo, *args, **kwargs)
        self.kernel = kernel
        self.bias = bias
        self.scale = scale
        self.smoother = smoother
        self.shift = shift

    def _postprocess(self, tensor: Union[torch.Tensor, Dict[str, torch.Tensor]], device: str, load_config: LoadConfig):
        # need reshape for kernel weight
        processed_res = super()._postprocess(tensor, device, load_config)
        kernel_weight = processed_res[self.kernel.name]
        kernel_weight = kernel_weight.reshape(kernel_weight.shape[-1], -1)
        processed_res[self.kernel.name] = kernel_weight

        return processed_res
