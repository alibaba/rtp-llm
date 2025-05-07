import logging
import torch
from typing import Any, Dict, List, Optional, Union
from maga_transformer.model_loader.ffn_weight import FfnAtomicWeight, MoeAtomicWeight
from maga_transformer.model_loader.w8a8_weight import W8A8Fp8AtomicWeight, create_w8a8_fp8_weight
from maga_transformer.model_loader.load_config import LoadConfig
from maga_transformer.model_loader.weight_module import CompositeWeight, QuantWeight, WeightModule, AtomicWeight
from maga_transformer.utils.util import check_with_info
from maga_transformer.utils.model_weight import W, CkptWeightInfo, WeightStyle, concat_0, get_tensor_from_scalar, \
    get_tensor_reciprocal, identity, merge_te_qkv, stack_, stack_moe_w1, concat_w13

W_SUFFIX = '.weight'
B_SUFFIX = ".bias"
ACT_S_SUFFIX = '.activation_scaling_factor'
W_S_SUFFIX = '.weights_scaling_factor'

def fp8_view(ts: List[torch.Tensor]):
    m, n = ts[0].shape
    return ts[0].reshape(n, m)

def qkv_concat_fp8(ts: List[torch.Tensor]):
    return fp8_view([concat_0(ts)])

def transpose_moe_down(ts: List[torch.Tensor]):
    return stack_(ts).transpose(1, 2)

def merge_qkv_scale(ts: List[torch.Tensor]):
    check_with_info(len(ts) == 3, "qkv scale should have 3 tensors")
    out_scale = torch.concat(ts, dim=0)
    return torch.max(out_scale, dim=0).unsqueeze(0)

def merge_qkv_hf_fp8_with_scale_t(ts: List[torch.Tensor]):
    check_with_info(len(ts) == 6, "qkv weight+scale should have 6 tensors")
    origin_scales = ts[3:]
    max_scale = merge_qkv_scale(origin_scales)
    q, k, v, q_scale_inv, k_scale_inv, v_scale_inv = ts
    q = q * q_scale_inv
    k = k * k_scale_inv
    v = v * v_scale_inv
    quanted_qkv = (torch.cat([q, k, v], dim=0) / max_scale).transpose(0, 1).to(torch.float8_e4m3fn)
    return quanted_qkv

class StaticPerTensorFp8Weight(CompositeWeight, QuantWeight):
    w8a8_weight_list = [
        W.attn_qkv_w,
        W.attn_o_w,
        W.ffn_w1,
        W.ffn_w3,
        W.ffn_w2
    ]
    FP8_SCALE_MAP = {
        W.attn_qkv_w : W.attn_qkv_s,
        W.attn_o_w : W.attn_o_s,
        W.ffn_w3 : W.ffn_s3,
        W.ffn_w2 : W.ffn_s2,
        W.ffn_w1 : W.ffn_s1,
        W.ffn_w13: W.ffn_s13
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
    @classmethod
    def support(cls, quant_algo: Any, src_weight_info: WeightModule) -> bool:
        name = src_weight_info.name
        return quant_algo.isFp8() and not quant_algo.isGroupwise() and name in cls.w8a8_weight_list and (src_weight_info.weight_style not in [WeightStyle.TRT_ENGINE, WeightStyle.RTP_SMOOTH_LLM_STYLE])


    def __init__(self, src_weight_info: AtomicWeight, quant_algo: Any, *args, **kwargs):
        self.quant_algo = quant_algo
        kernel: AtomicWeight = None
        scale: Optional[AtomicWeight] = None
        act_scale: Optional[AtomicWeight] = None
        act_scale_inv: Optional[AtomicWeight] = None
        logging.debug(f"StaticPerTensorFp8Weight : {self.qs_suffix}, {self.qw_suffix}")

        if src_weight_info.name == W.attn_qkv_w:
            (kernel, scale, act_scale, act_scale_inv) = self._get_qkv_quant_weight_info(
                src_weight_info)
        elif src_weight_info.name == W.attn_o_w:
            (kernel, scale, act_scale, act_scale_inv) = self._get_attn_out_quant_weight_info(src_weight_info)
        elif src_weight_info.name in [W.ffn_w1, W.ffn_w2, W.ffn_w3, W.ffn_w13, W.moe_w1, W.moe_w2]:
            (kernel, scale, act_scale, act_scale_inv) = self._get_ffn_quant_weight_info(src_weight_info, quant_algo)
        else:
            raise ValueError(f"Unsupported weight name {src_weight_info.name}")

        sub_weights = {
            kernel.name: kernel,
        }

        if scale is not None:
            sub_weights[scale.name] = scale
        if act_scale is not None:
            sub_weights[act_scale.name] = act_scale
        if act_scale_inv is not None:
            sub_weights[act_scale_inv.name] = act_scale_inv


        super().__init__(sub_weights, quant_algo=quant_algo, *args, **kwargs)
        self.kernel = kernel
        self.scale = scale
        self.act_scale = act_scale
        self.act_scale_inv = act_scale_inv

    @property
    def qw_suffix(self) -> str:
        return W_SUFFIX

    @property
    def qs_suffix(self) -> str:
        return W_S_SUFFIX

    @property
    def act_s_suffix(self) -> str:
        return ACT_S_SUFFIX

    def _get_qkv_quant_weight_info(self, src_weight_info: AtomicWeight) -> List[W8A8Fp8AtomicWeight]:
        weights: List[CkptWeightInfo] = src_weight_info.weights
        assert len(weights) == 1 or len(weights) == 3

        qkv_name = weights[0].name
        assert qkv_name.endswith(W_SUFFIX)
        qkv_name = qkv_name[:-len(W_SUFFIX)]
        act_s = self.FP8_ACT_SCALE_MAP[src_weight_info.name][0]
        act_s_inv = self.FP8_ACT_SCALE_MAP[src_weight_info.name][1]

        return [
            create_w8a8_fp8_weight(src_weight_info, W.attn_qkv_w, weights,
                                   merge_te_qkv, data_type=torch.float8_e4m3fn, config=src_weight_info.config),
            create_w8a8_fp8_weight(src_weight_info, W.attn_qkv_s, [CkptWeightInfo(qkv_name+ self.qs_suffix)],
                                   identity, data_type=torch.float32, config=src_weight_info.config),
            create_w8a8_fp8_weight(src_weight_info, act_s[0],  [CkptWeightInfo(qkv_name+ self.act_s_suffix)],
                                   act_s[1], data_type=torch.float32, config=src_weight_info.config),
            create_w8a8_fp8_weight(src_weight_info, act_s_inv[0],  [CkptWeightInfo(qkv_name+ self.act_s_suffix)],
                                   act_s_inv[1], data_type=torch.float32, config=src_weight_info.config)
        ]

    def _get_attn_out_quant_weight_info(self, src_weight_info: WeightModule) -> List[W8A8Fp8AtomicWeight]:
        w_name = src_weight_info.weights[0].name[:-len(W_SUFFIX)]
        act_s = self.FP8_ACT_SCALE_MAP[src_weight_info.name][0]
        act_s_inv = self.FP8_ACT_SCALE_MAP[src_weight_info.name][1]

        return [create_w8a8_fp8_weight(src_weight_info, W.attn_o_w, [CkptWeightInfo(w_name+ self.qw_suffix, identity)],
                                       identity, data_type=torch.float8_e4m3fn, config=src_weight_info.config),
                create_w8a8_fp8_weight(src_weight_info, W.attn_o_s, [CkptWeightInfo(w_name+ self.qs_suffix)],
                                       identity, data_type=torch.float32, config=src_weight_info.config),
                create_w8a8_fp8_weight(src_weight_info, act_s[0],  [CkptWeightInfo(w_name+ self.act_s_suffix)],
                                       act_s[1], data_type=torch.float32, config=src_weight_info.config),
                create_w8a8_fp8_weight(src_weight_info, act_s_inv[0],  [CkptWeightInfo(w_name+ self.act_s_suffix)],
                                       act_s_inv[1], data_type=torch.float32, config=src_weight_info.config)
            ]

    def _get_ffn_quant_weight_info(self, src_weight: Union[FfnAtomicWeight, MoeAtomicWeight], quant_algo: Any) -> List[Optional[W8A8Fp8AtomicWeight]]:
        weights = src_weight.weights
        ffn_w_name = src_weight.name
        assert weights[0].name.endswith(W_SUFFIX)
        assert ffn_w_name in [W.ffn_w1, W.ffn_w2, W.ffn_w3, W.moe_w1, W.moe_w2]

        if ffn_w_name in [W.ffn_w1, W.ffn_w2, W.ffn_w3]:
            assert len(weights) == 1
        w_name = weights[0].name[:-len(W_SUFFIX)]
        w: str = None
        s: str = None
        if ffn_w_name in [W.moe_w2, W.moe_w1]:
            if ffn_w_name == W.moe_w1:
                w, s = (W.moe_w1, W.moe_s1)
                stack = stack_moe_w1
            elif ffn_w_name == W.moe_w2:
                w, s = (W.moe_w2, W.moe_s2)
                stack = stack_

            w_name = [weight.name[:-len(W_SUFFIX)] for weight in weights]
            # TODO(luoli.hn) 等拿到ckpt 再改
            return [
                create_w8a8_fp8_weight(src_weight,
                    w, [CkptWeightInfo(name+ self.qw_suffix, identity) \
                        for name in w_name], stack, data_type=torch.float8_e4m3fn,
                        config=src_weight.config),
                create_w8a8_fp8_weight(src_weight,
                    s, [CkptWeightInfo(name+ self.qs_suffix, identity) \
                        for name in w_name], stack, data_type=torch.float32,
                        config=src_weight.config),
                None,
                None
            ]
        elif ffn_w_name == W.ffn_w13:
            w, b, s = (W.ffn_w13, W.ffn_b13, W.ffn_s13)
            w1_name = weights[0].name[:-len(W_SUFFIX)]
            w3_name = weights[1].name[:-len(W_SUFFIX)]

            act_s = self.FP8_ACT_SCALE_MAP[W.ffn_w3][0]
            act_s_inv = self.FP8_ACT_SCALE_MAP[W.ffn_w3][1]
            return [
                create_w8a8_fp8_weight(src_weight,
                    w, [CkptWeightInfo(w1_name + self.qw_suffix, identity), CkptWeightInfo(w3_name + self.qw_suffix, identity)],
                    concat_w13, data_type=torch.float8_e4m3fn,
                        config=src_weight.config),
                create_w8a8_fp8_weight(src_weight,
                    b, [CkptWeightInfo(w1_name + self.qs_suffix, identity), CkptWeightInfo(w3_name + self.qs_suffix, identity)],
                    concat_w13, data_type=torch.float8_e4m3fn,
                    config=src_weight.config),
                create_w8a8_fp8_weight(src_weight,
                    act_s[0],  [CkptWeightInfo(w_name+ self.act_s_suffix)], act_s[1], data_type=torch.float32, config=src_weight.config
                ),
                create_w8a8_fp8_weight(src_weight,
                    act_s_inv[0],  [CkptWeightInfo(w_name+ self.act_s_suffix)], act_s_inv[1], data_type=torch.float32, config=src_weight.config
                )            ]

        else:
            w = src_weight.name
            s = self.FP8_SCALE_MAP.get(src_weight.name)

            w_list = [
                create_w8a8_fp8_weight(src_weight,
                    w, [CkptWeightInfo(w_name+ self.qw_suffix, identity)],
                    identity, data_type=torch.float8_e4m3fn,
                        config=src_weight.config),
                create_w8a8_fp8_weight(src_weight,
                    s, [CkptWeightInfo(w_name+ self.qs_suffix, identity)],
                        identity, data_type=torch.float32,
                        config=src_weight.config)
            ]
            if w in self.FP8_ACT_SCALE_MAP:
                act_s = self.FP8_ACT_SCALE_MAP[src_weight.name][0]
                act_s_inv = self.FP8_ACT_SCALE_MAP[src_weight.name][1]
                w_list.extend([
                    create_w8a8_fp8_weight(src_weight,
                        act_s[0],  [CkptWeightInfo(w_name+ self.act_s_suffix)], act_s[1], data_type=torch.float32, config=src_weight.config
                    ),
                    create_w8a8_fp8_weight(src_weight,
                        act_s_inv[0],  [CkptWeightInfo(w_name+ self.act_s_suffix)], act_s_inv[1], data_type=torch.float32, config=src_weight.config
                    )
                ]
                )
            else:
                w_list.extend([None, None])
            return w_list

    def _postprocess(self, tensor: Union[torch.Tensor, Dict[str, torch.Tensor]], device: str, load_config: LoadConfig):
        # need reshape for kernel weight
        processed_res = super()._postprocess(tensor, device, load_config)
        kernel_weight = processed_res[self.kernel.name]
        kernel_weight = kernel_weight.reshape(kernel_weight.shape[-1], -1)
        processed_res[self.kernel.name] = kernel_weight

        return processed_res


class TrtEngineStaticPerTensorFp8Weight(StaticPerTensorFp8Weight):

    def __init__(self, src_weight_info: AtomicWeight, quant_algo: Any, *args, **kwargs):
        super().__init__(src_weight_info, quant_algo, *args, **kwargs)

    @classmethod
    def support(cls, quant_algo: Any, src_weight_info: WeightModule) -> bool:
        name = src_weight_info.name
        return quant_algo.isFp8() and not quant_algo.isGroupwise() and name in cls.w8a8_weight_list and src_weight_info.weight_style == WeightStyle.TRT_ENGINE

    def _get_qkv_quant_weight_info(self, src_weight_info: AtomicWeight) -> List[W8A8Fp8AtomicWeight]:
        weights: List[CkptWeightInfo] = src_weight_info.weights
        assert len(weights) == 1

        qkv_name = weights[0].name
        assert qkv_name.endswith(W_SUFFIX)
        qkv_name = qkv_name[:-len(W_SUFFIX)]
        act_s = self.FP8_ACT_SCALE_MAP[src_weight_info.name][0]
        act_s_inv = self.FP8_ACT_SCALE_MAP[src_weight_info.name][1]

        return [
            create_w8a8_fp8_weight(src_weight_info, W.attn_qkv_w, [CkptWeightInfo(qkv_name+ self.qw_suffix)],
                                   identity, data_type=torch.float8_e4m3fn, config=src_weight_info.config),
            create_w8a8_fp8_weight(src_weight_info, W.attn_qkv_s, [CkptWeightInfo(qkv_name+ self.qs_suffix)],
                                   identity, data_type=torch.float32, config=src_weight_info.config),
            create_w8a8_fp8_weight(src_weight_info, act_s[0],  [CkptWeightInfo(qkv_name+ self.act_s_suffix)],
                                   act_s[1], data_type=torch.float32, config=src_weight_info.config),
            create_w8a8_fp8_weight(src_weight_info, act_s_inv[0],  [CkptWeightInfo(qkv_name+ self.act_s_suffix)],
                                   act_s_inv[1], data_type=torch.float32, config=src_weight_info.config)
        ]
    @property
    def qw_suffix(self) -> str:
        return ".weight"

    @property
    def qs_suffix(self) -> str:
        return W_S_SUFFIX

    @property
    def act_s_suffix(self) -> str:
        return ACT_S_SUFFIX