import functools
import logging

import torch
from typing import Any, Dict, List, Optional, Union
from maga_transformer.model_loader.ffn_weight import FfnAtomicWeight, MoeAtomicWeight
from maga_transformer.model_loader.load_config import LoadConfig
from maga_transformer.model_loader.weight_module import CompositeWeight, QuantWeight, WeightModule, AtomicWeight
from maga_transformer.utils.model_weight import W, CkptWeightInfo, WeightStyle, expand_scale, get_tensor_from_scalar, \
    get_tensor_reciprocal, identity, merge_qkv_transpose_concat0, stack_, stack_moe_w1, transpose, \
        transpose_w13, concat_w13
from maga_transformer.model_loader.w8a8_weight import W8A8Int8AtomicWeight, create_w8a8_int8_weight

W_SUFFIX = '.weight'
B_SUFFIX = ".bias"
ACT_S_SUFFIX = '_input_scale'
W_S_SUFFIX = '.scales'


class PerTensorInt8QuantWeight(CompositeWeight, QuantWeight):
    w8a8_weight_list = [
        W.attn_qkv_w,
        W.attn_o_w,
        W.ffn_w1,
        W.ffn_w3,
        W.ffn_w2,
        W.ffn_w13,
        W.pre_decoder_ln_gamma   # TODO(luoli.hn) hack for bert, we need refactor GptModel.cc
    ]
    EMB_NORM_S: str = "encoder.layer.0.attention.self.qkv_input_scale"
    FFN_OUTPUT_LAYERNORM_S_SUFFIX: str = ".qkv_input_scale"
    INT8_SCALE_MAP = {
        W.attn_qkv_w : W.attn_qkv_s,
        W.attn_o_w : W.attn_o_s,
        W.ffn_w3 : W.ffn_s3,
        W.ffn_w2 : W.ffn_s2,
        W.ffn_w1 : W.ffn_s1,
    }

    INT8_ACT_SCALE_MAP = {
        W.attn_qkv_w : [
            (W.pre_ln_static_quant, get_tensor_reciprocal),
            (W.pre_ln_static_quant_reciprocal, get_tensor_from_scalar),
        ],
        W.attn_o_w : [
            (W.attention_output_static_quant, get_tensor_reciprocal),
            (W.attention_output_static_quant_reciprocal, get_tensor_from_scalar),
        ],
        W.ffn_w2 : [
            (W.ffn_intermediate_weight2_static_quant, get_tensor_reciprocal),
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
        return quant_algo.isPerTensorQuant() and name in cls.w8a8_weight_list and (src_weight_info.weight_style not in [WeightStyle.TRT_ENGINE, WeightStyle.RTP_SMOOTH_LLM_STYLE])


    def __init__(self, src_weight_info: AtomicWeight, quant_algo: Any, *args: Any, **kwargs: Any):
        self.quant_algo = quant_algo
        kernel: AtomicWeight = None
        scale: Optional[AtomicWeight] = None
        act_scale: Optional[AtomicWeight] = None
        act_scale_inv: Optional[AtomicWeight] = None
        logging.debug(f"PerTensorInt8QuantWeight : {self.qs_suffix}, {self.qw_suffix}")

        if src_weight_info.name == W.attn_qkv_w:
            (kernel, scale, act_scale, act_scale_inv) = self._get_qkv_quant_weight_info(
                src_weight_info)
        elif src_weight_info.name == W.attn_o_w:
            (kernel, scale, act_scale, act_scale_inv) = self._get_attn_out_quant_weight_info(src_weight_info)
        elif src_weight_info.name in [W.ffn_w1, W.ffn_w2, W.ffn_w3, W.ffn_w13, W.moe_w1, W.moe_w2]:
            (kernel, scale, act_scale, act_scale_inv) = self._get_ffn_quant_weight_info(src_weight_info, quant_algo)
        elif src_weight_info.name == W.pre_decoder_ln_gamma:
            (kernel, scale, act_scale, act_scale_inv) = self.get_pre_decoder_ln_weight(src_weight_info)
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


    def get_pre_decoder_ln_weight(self, src_weight_info: AtomicWeight) -> List[Optional[W8A8Int8AtomicWeight]]:
        w_name = src_weight_info.weights[0].name
        act_s = W.pre_decoder_ln_static_quant
        act_s_inv = W.pre_decoder_ln_static_quant_reciprocal
        prefix = w_name.split('.')[0] + '.' # TODO(luoli.hn) hack for bert, we need refactor GptModel.cc

        return [create_w8a8_int8_weight(src_weight_info, src_weight_info.name, [CkptWeightInfo(w_name, identity)],
                                       identity, data_type=src_weight_info.data_type),
                None,
                create_w8a8_int8_weight(src_weight_info, act_s,  [CkptWeightInfo(prefix + self.EMB_NORM_S)],
                                       get_tensor_reciprocal, data_type=torch.float32),
                create_w8a8_int8_weight(src_weight_info, act_s_inv,  [CkptWeightInfo(prefix + self.EMB_NORM_S)],
                                       get_tensor_from_scalar, data_type=torch.float32)
            ]

    def _get_qkv_quant_weight_info(self, src_weight_info: AtomicWeight) -> List[W8A8Int8AtomicWeight]:
        weights: List[CkptWeightInfo] = src_weight_info.weights
        assert len(weights) == 3

        need_post_ln = src_weight_info.config.need_post_ln

        qkv_w_list: List[CkptWeightInfo] = [CkptWeightInfo(sub_w.name[:-len(W_SUFFIX)] + self.qw_suffix) for sub_w in weights]
        qkv_s_list: List[CkptWeightInfo] = [CkptWeightInfo(sub_w.name[:-len(W_SUFFIX)] + self.qs_suffix) for sub_w in weights]


        qkv_w = create_w8a8_int8_weight(src_weight_info,
                                        W.attn_qkv_w,
                                        qkv_w_list,
                                        merge_qkv_transpose_concat0,
                                        data_type=torch.int8,
                                        config=src_weight_info.config)

        qkv_s = create_w8a8_int8_weight(src_weight_info,
                                        W.attn_qkv_s,
                                        qkv_s_list,
                                        functools.partial(expand_scale, hidden_size=src_weight_info.config.hidden_size),
                                        data_type=torch.float32,
                                        config=src_weight_info.config)
        if need_post_ln:
            post_ln_w = weights[0].name.replace('{i}', '{i_1}').rsplit('.', 2)[0] + self.FFN_OUTPUT_LAYERNORM_S_SUFFIX  # TODO(luoli.hn) hack for bert, we need refactor GptModel.cc
            return [
                qkv_w,
                qkv_s,
                create_w8a8_int8_weight(src_weight_info, W.post_ffn_ln_static_quant, [CkptWeightInfo(post_ln_w)],
                                        get_tensor_reciprocal, data_type=torch.float32, config=src_weight_info.config),
                create_w8a8_int8_weight(src_weight_info, W.post_ffn_ln_static_quant_reciprocal, [CkptWeightInfo(post_ln_w)],
                                        get_tensor_from_scalar, data_type=torch.float32, config=src_weight_info.config)
            ]
        else:
            return [qkv_w, qkv_s, None, None]

    def _get_attn_out_quant_weight_info(self, src_weight_info: WeightModule) -> List[W8A8Int8AtomicWeight]:
        w_name = src_weight_info.weights[0].name[:-len(W_SUFFIX)]
        act_s = self.INT8_ACT_SCALE_MAP[src_weight_info.name][0]
        act_s_inv = self.INT8_ACT_SCALE_MAP[src_weight_info.name][1]

        return [create_w8a8_int8_weight(src_weight_info, W.attn_o_w, [CkptWeightInfo(w_name+ self.qw_suffix, identity)],
                                       transpose, data_type=torch.int8, config=src_weight_info.config),
                create_w8a8_int8_weight(src_weight_info, W.attn_o_s, [CkptWeightInfo(w_name+ self.qs_suffix)],
                                       identity, data_type=torch.float32, config=src_weight_info.config),
                create_w8a8_int8_weight(src_weight_info, act_s[0],  [CkptWeightInfo(w_name+ self.act_s_suffix)],
                                       act_s[1], data_type=torch.float32, config=src_weight_info.config),
                create_w8a8_int8_weight(src_weight_info, act_s_inv[0],  [CkptWeightInfo(w_name+ self.act_s_suffix)],
                                       act_s_inv[1], data_type=torch.float32, config=src_weight_info.config)
            ]

    def _get_ffn_quant_weight_info(self, src_weight: Union[FfnAtomicWeight, MoeAtomicWeight], quant_algo: Any) -> List[Optional[W8A8Int8AtomicWeight]]:
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
            raise ValueError(f"now not support {ffn_w_name}")
        elif ffn_w_name == W.ffn_w13:
            w, s = (W.ffn_w13, W.ffn_s13)
            w1_name = weights[0].name[:-len(W_SUFFIX)]
            w3_name = weights[1].name[:-len(W_SUFFIX)]
            act_s = self.INT8_ACT_SCALE_MAP[W.ffn_w3][0]
            act_s_inv = self.INT8_ACT_SCALE_MAP[W.ffn_w3][1]

            return [
                create_w8a8_int8_weight(src_weight,
                    w, [CkptWeightInfo(w1_name + self.qw_suffix, identity), CkptWeightInfo(w3_name + self.qw_suffix, identity)],
                    transpose_w13, data_type=torch.int8,
                        config=src_weight.config),
                create_w8a8_int8_weight(src_weight,
                    s, [CkptWeightInfo(w1_name + self.qs_suffix, identity), CkptWeightInfo(w3_name + self.qs_suffix, identity)],
                    transpose_w13, data_type=torch.float32,
                    config=src_weight.config),
                create_w8a8_int8_weight(src_weight,
                    act_s[0],  [CkptWeightInfo(w_name+ self.act_s_suffix)], act_s[1], data_type=torch.float32, config=src_weight.config
                ),
                create_w8a8_int8_weight(src_weight,
                    act_s_inv[0],  [CkptWeightInfo(w_name+ self.act_s_suffix)], act_s_inv[1], data_type=torch.float32, config=src_weight.config
                )
            ]
        else:
            w = src_weight.name
            s = self.INT8_SCALE_MAP.get(src_weight.name)

            w_list = [
                create_w8a8_int8_weight(src_weight,
                    w, [CkptWeightInfo(w_name+ self.qw_suffix, identity)],
                    transpose, data_type=torch.int8,
                        config=src_weight.config),
                create_w8a8_int8_weight(src_weight,
                    s, [CkptWeightInfo(w_name+ self.qs_suffix, identity)],
                        transpose, data_type=torch.float32,
                        config=src_weight.config)
            ]
            if w in self.INT8_ACT_SCALE_MAP:
                act_s = self.INT8_ACT_SCALE_MAP[src_weight.name][0]
                act_s_inv = self.INT8_ACT_SCALE_MAP[src_weight.name][1]
                w_list.extend([
                    create_w8a8_int8_weight(src_weight,
                        act_s[0],  [CkptWeightInfo(w_name+ self.act_s_suffix)], act_s[1], data_type=torch.float32, config=src_weight.config
                    ),
                    create_w8a8_int8_weight(src_weight,
                        act_s_inv[0],  [CkptWeightInfo(w_name+ self.act_s_suffix)], act_s_inv[1], data_type=torch.float32, config=src_weight.config
                    )
                ]
                )
            else:
                w_list.extend([None, None])
            return w_list


    @property
    def qw_suffix(self) -> str:
        return ".qweight"

    @property
    def qs_suffix(self) -> str:
        return W_S_SUFFIX

    @property
    def act_s_suffix(self) -> str:
        return ACT_S_SUFFIX

    def _postprocess(self, tensor: Union[torch.Tensor, Dict[str, torch.Tensor]], device: str, load_config: LoadConfig):
        # need reshape for kernel weight
        processed_res = super()._postprocess(tensor, device, load_config)
        kernel_weight = processed_res[self.kernel.name]
        kernel_weight = kernel_weight.reshape(kernel_weight.shape[-1], -1)
        processed_res[self.kernel.name] = kernel_weight

        return processed_res
