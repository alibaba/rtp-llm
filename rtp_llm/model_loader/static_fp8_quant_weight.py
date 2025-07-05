import logging
import torch
import copy
from typing import Any, Dict, List, Optional, Union
from rtp_llm.config.quant_config import QuantizationConfig, Fp8PerTensorQuantConfig
from rtp_llm.model_loader.ffn_weight import FfnAtomicWeight, MoeAtomicWeight
from rtp_llm.model_loader.w8a8_weight import W8A8Fp8AtomicWeight, create_w8a8_fp8_weight
from rtp_llm.model_loader.load_config import LoadConfig
from rtp_llm.model_loader.weight_module import CompositeWeight, QuantWeight, WeightModule, AtomicWeight
from rtp_llm.utils.database import BaseDatabase
from rtp_llm.utils.util import check_with_info
from rtp_llm.utils.model_weight import FP8_E4M3_MAX, W, CkptWeightInfo, WeightStyle, concat_0, get_tensor_from_scalar, \
    get_tensor_reciprocal, identity, merge_te_qkv, sp_id, stack_, stack_moe_w1, concat_w13

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


def quantize_weight_to_fp8(ts):
    max_abs_value = ts.abs().max()
    scaling_factor = max_abs_value / FP8_E4M3_MAX
    min_scaling_factor = 1.0 / (FP8_E4M3_MAX * 512.0)
    scaling_factor = max(min_scaling_factor, scaling_factor)

    # 量化操作
    quantized_weight = (ts / scaling_factor).to(torch.float8_e4m3fn)
    return quantized_weight.contiguous(), scaling_factor.to(torch.float32)

def one_scales(ts: List[torch.Tensor]) -> torch.Tensor:
    return torch.ones([], dtype=torch.float32).contiguous()


class StaticPerTensorFp8Weight(CompositeWeight, QuantWeight):
    w8a8_weight_list = [
        W.attn_qkv_w,
        W.attn_o_w,
        W.ffn_w1,
        W.ffn_w3,
        W.ffn_w2,
        W.ffn_w13
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
    def support(cls, quant_config: QuantizationConfig, src_weight_info: WeightModule) -> bool:
        if not quant_config.is_quanted() or not isinstance(quant_config, Fp8PerTensorQuantConfig):
            return False

        name = src_weight_info.name
        return name in cls.w8a8_weight_list and (src_weight_info.weight_style not in [WeightStyle.TRT_ENGINE, WeightStyle.RTP_SMOOTH_LLM_STYLE])


    def __init__(self, src_weight_info: AtomicWeight, quant_config: QuantizationConfig, *args, **kwargs):
        self.quant_config = quant_config
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
            (kernel, scale, act_scale, act_scale_inv) = self._get_ffn_quant_weight_info(src_weight_info, quant_config)
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


        super().__init__(sub_weights, quant_config=quant_config, *args, **kwargs)
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

    def _get_ffn_quant_weight_info(self, src_weight: Union[FfnAtomicWeight, MoeAtomicWeight], quant_config: Any) -> List[Optional[W8A8Fp8AtomicWeight]]:
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
    def __init__(self, src_weight_info: AtomicWeight, quant_config: QuantizationConfig, *args, **kwargs):
        super().__init__(src_weight_info, quant_config, *args, **kwargs)

    @classmethod
    def support(cls, quant_config: QuantizationConfig, src_weight_info: WeightModule) -> bool:
        if not quant_config.is_quanted() or not isinstance(quant_config, Fp8PerTensorQuantConfig):
            return False
        name = src_weight_info.name
        return name in cls.w8a8_weight_list and src_weight_info.weight_style == WeightStyle.TRT_ENGINE

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


class LoadQuantStaticPerTensorFp8Weight(StaticPerTensorFp8Weight):
    fp8_attn_weights_map = {
        W.attn_qkv_w: (W.attn_qkv_s, W.pre_ln_static_quant, W.pre_ln_static_quant_reciprocal),
        W.attn_o_w: (W.attn_o_s, W.attention_output_static_quant, W.attention_output_static_quant_reciprocal),
        W.mla_fusedqkrope_w: (W.mla_fusedqkrope_s, None, None),
        W.mla_fusedqkrope_no_lora_w: (W.mla_fusedqkrope_no_lora_s, None, None),
        W.mla_q_b_w: (W.mla_q_b_s, None, None),
        W.mla_k_nope_w: (W.mla_k_nope_s, None, None),
        W.mla_v_w: (W.mla_v_s, None, None),
    }

    fp8_attn_vision_weights_map = {
        W.vision_attn_qkv_w: (W.vision_attn_qkv_s, None, None),
        W.vision_attn_o_w: (W.vision_attn_o_s, None, None),
    }

    fp8_ffn_weights_maps = {
        W.ffn_w1: (W.ffn_s1, None, None),
        W.ffn_w3: (W.ffn_s3, W.post_ln_static_quant, W.post_ln_static_quant_reciprocal),
        W.ffn_w2: (W.ffn_s2, W.ffn_intermediate_weight2_static_quant, W.ffn_intermediate_weight2_static_quant_reciprocal),
        W.ffn_w13: (W.ffn_s13, W.post_ln_static_quant, W.post_ln_static_quant_reciprocal)
    }

    fp8_vision_ffn_weights_maps = {
        W.vision_ffn_w1: (W.vision_ffn_s1, None, None),
        W.vision_ffn_w3: (W.vision_ffn_s3, None, None),
        W.vision_ffn_w2: (W.vision_ffn_s2, None, None),
    }

    fp8_partial_moe_weights_maps = {
        W.moe_w1: (W.moe_s1, None, None),
        W.moe_w2: (W.moe_s2, None, None),
    }

    weight_scale_map = {
        **fp8_attn_weights_map,
        **fp8_attn_vision_weights_map,
        **fp8_ffn_weights_maps,
        **fp8_vision_ffn_weights_maps,
        **fp8_partial_moe_weights_maps,
    }
    @classmethod
    def support(cls, quant_config: QuantizationConfig, src_weight_info: WeightModule) -> bool:
        if quant_config.is_quanted() or not isinstance(quant_config, Fp8PerTensorQuantConfig):
            return False
        name = src_weight_info.name
        return name in cls.w8a8_weight_list and (src_weight_info.weight_style not in [WeightStyle.TRT_ENGINE, WeightStyle.RTP_SMOOTH_LLM_STYLE])

    def __init__(self, src_weight_info: AtomicWeight, quant_config: QuantizationConfig, *args, **kwargs):
        if src_weight_info.name in [W.moe_w1, W.moe_w2]:
            raise ValueError("now not support moe_w1, moe_w2 fp8_static quant")
        params = src_weight_info.extract_params(src_weight_info.__class__, src_weight_info, quant_config)
        kernel: AtomicWeight = create_w8a8_fp8_weight(src_weight_info, **params)
        sub_weights = {kernel.name: kernel}

        scale_name, act_scale_name, act_scale_inv_name = self.weight_scale_map.get(src_weight_info.name)
        scale_params = copy.deepcopy(params)
        scale_params['name'] = scale_name
        scale: AtomicWeight = create_w8a8_fp8_weight(src_weight_info, **scale_params)
        sub_weights.update({scale.name: scale})

        act_scale = None
        act_scale_inv = None
        if act_scale_name:
            act_scale_params = copy.deepcopy(params)
            act_scale_params['name'] = act_scale_name
            act_scale_params['weights'] = []
            act_scale_params['process_fun'] = one_scales
            act_scale_params['data_type'] = torch.float32
            act_scale_params['split_func'] = sp_id
            act_scale = create_w8a8_fp8_weight(src_weight_info, **act_scale_params)
            sub_weights.update({act_scale.name: act_scale})
        if act_scale_inv_name:
            act_scale_inv_params = copy.deepcopy(params)
            act_scale_inv_params['name'] = act_scale_inv_name
            act_scale_inv_params['weights'] = []
            act_scale_inv_params['process_fun'] = one_scales
            act_scale_inv_params['data_type'] = torch.float32
            act_scale_inv_params['split_func'] = sp_id
            act_scale_inv = create_w8a8_fp8_weight(src_weight_info, **act_scale_inv_params)
            sub_weights.update({act_scale_inv.name: act_scale_inv})
        CompositeWeight.__init__(self, sub_weights, quant_config=quant_config, *args, **kwargs)
        self.kernel = kernel
        self.scale = scale
        self.act_scale = act_scale
        self.act_scale_inv = act_scale_inv

    def _load_raw_tensor(self, database: BaseDatabase, layer_id: Optional[int], device: str, load_config: LoadConfig):
        kernel = self.kernel._load_raw_tensor(database, layer_id, device, load_config)
        res = {}
        quant_kernel, scale = quantize_weight_to_fp8(kernel.get(self.kernel.name))
        quant_kernel = quant_kernel.T
        res = {self.kernel.name: quant_kernel.contiguous().to(device), self.scale.name: scale.contiguous().to(device)}

        if self.act_scale:
            act_scale = self.act_scale._load_raw_tensor(database, layer_id, device, load_config)
            res.update(act_scale)
        if self.act_scale_inv:
            act_scale_inv = self.act_scale_inv._load_raw_tensor(database, layer_id, device, load_config)
            res.update(act_scale_inv)
        return res
