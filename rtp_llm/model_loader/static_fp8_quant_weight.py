import copy
import functools
import logging
from typing import Any, Dict, List, Optional, Union

import torch

from rtp_llm.config.quant_config import (
    Fp8PerTensorCompressedQuantConfig,
    Fp8PerTensorQuantConfig,
    QuantizationConfig,
)
from rtp_llm.model_loader.ffn_weight import FfnAtomicWeight, MoeAtomicWeight
from rtp_llm.model_loader.load_config import LoadConfig
from rtp_llm.model_loader.tensor_source import TensorSource
from rtp_llm.model_loader.w8a8_weight import W8A8Fp8AtomicWeight, create_w8a8_fp8_weight
from rtp_llm.model_loader.weight_module import (
    AtomicWeight,
    CompositeWeight,
    QuantWeight,
    WeightModule,
)
from rtp_llm.utils.model_weight import (
    FP8_E4M3_MAX,
    CkptWeightInfo,
    W,
    WeightStyle,
    concat_0,
    concat_w13,
    concat_w13_2,
    get_list_tensor_from_scalar,
    get_list_tensor_reciprocal,
    get_tensor_from_scalar,
    get_tensor_reciprocal,
    identity,
    merge_te_qkv,
    pad_w13,
    sp_id,
    stack_,
    stack_moe_w1,
)
from rtp_llm.utils.util import check_with_info

W_SUFFIX = ".weight"
B_SUFFIX = ".bias"
ACT_S_SUFFIX = ".activation_scaling_factor"
W_S_SUFFIX = ".weights_scaling_factor"


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
    quanted_qkv = (
        (torch.cat([q, k, v], dim=0) / max_scale)
        .transpose(0, 1)
        .to(torch.float8_e4m3fn)
    )
    return quanted_qkv


def merge_qkv_hf_fp8_with_scale(ts: List[torch.Tensor]):
    check_with_info(len(ts) == 6, "qkv weight+scale should have 6 tensors")
    origin_scales = ts[3:]
    max_scale = torch.concat(origin_scales, dim=0).max()
    q, k, v, q_scale_inv, k_scale_inv, v_scale_inv = ts
    q = q.to(torch.float32) * q_scale_inv
    k = k.to(torch.float32) * k_scale_inv
    v = v.to(torch.float32) * v_scale_inv
    quanted_qkv = (torch.cat([q, k, v], dim=0) / max_scale).to(torch.float8_e4m3fn)
    return quanted_qkv, max_scale


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


class Fp8PerTensorCompressedWeight(CompositeWeight, QuantWeight):
    w8a8_weight_list = [
        W.attn_qkv_w,
        W.attn_o_w,
        W.ffn_w1,
        W.ffn_w3,
        W.ffn_w2,
        W.ffn_w13,
        W.moe_w1,
        W.moe_w2,
    ]
    FP8_SCALE_MAP = {
        W.attn_qkv_w: W.attn_qkv_s,
        W.attn_o_w: W.attn_o_s,
        W.ffn_w3: W.ffn_s3,
        W.ffn_w2: W.ffn_s2,
        W.ffn_w1: W.ffn_s1,
        W.ffn_w13: W.ffn_s13,
        W.moe_w1: W.moe_s1,
        W.moe_w2: W.moe_s2,
    }

    FP8_ACT_SCALE_MAP = {
        W.attn_qkv_w: [
            (W.pre_ln_static_quant, get_list_tensor_reciprocal),
            (W.pre_ln_static_quant_reciprocal, get_list_tensor_from_scalar),
        ],
        W.attn_o_w: [
            (W.attention_output_static_quant, get_tensor_from_scalar),
            (W.attention_output_static_quant_reciprocal, get_tensor_reciprocal),
        ],
        W.ffn_w2: [
            (W.ffn_intermediate_weight2_static_quant, get_tensor_from_scalar),
            (
                W.ffn_intermediate_weight2_static_quant_reciprocal,
                get_tensor_from_scalar,
            ),
        ],
        W.ffn_w3: [
            (W.post_ln_static_quant, get_tensor_reciprocal),
            (W.post_ln_static_quant_reciprocal, get_tensor_from_scalar),
        ],
        W.moe_w1: [
            (W.moe_w1_input_s, get_list_tensor_reciprocal),
            (W.moe_w1_input_sr, get_list_tensor_from_scalar),
        ],
        W.moe_w2: [
            (W.moe_w2_input_s, get_list_tensor_reciprocal),
            (W.moe_w2_input_sr, get_list_tensor_from_scalar),
        ],
    }

    @classmethod
    def support(
        cls, quant_config: QuantizationConfig, src_weight_info: WeightModule
    ) -> bool:
        if not quant_config.is_quanted() or not isinstance(
            quant_config, Fp8PerTensorCompressedQuantConfig
        ):
            return False
        name = src_weight_info.name
        return name in cls.w8a8_weight_list

    def __init__(
        self,
        src_weight_info: AtomicWeight,
        quant_config: QuantizationConfig,
        *args,
        **kwargs,
    ):
        self.quant_config = quant_config
        kernel: AtomicWeight = None
        scale: Optional[AtomicWeight] = None
        act_scale: Optional[AtomicWeight] = None
        act_scale_inv: Optional[AtomicWeight] = None
        logging.debug(
            "Fp8PerTensorCompressedWeight : %s, %s", self.qs_suffix, self.qw_suffix
        )

        if src_weight_info.name == W.attn_qkv_w:
            (kernel, scale, act_scale, act_scale_inv) = self._get_qkv_quant_weight_info(
                src_weight_info
            )
        elif src_weight_info.name == W.attn_o_w:
            (kernel, scale, act_scale, act_scale_inv) = (
                self._get_attn_out_quant_weight_info(src_weight_info)
            )
        elif src_weight_info.name in [
            W.ffn_w1,
            W.ffn_w2,
            W.ffn_w3,
            W.ffn_w13,
            W.moe_w1,
            W.moe_w2,
        ]:
            (kernel, scale, act_scale, act_scale_inv) = self._get_ffn_quant_weight_info(
                src_weight_info, quant_config
            )
        else:
            raise ValueError(f"Unsupported weight name {src_weight_info.name}")

        sub_weights = {
            kernel.name: kernel,
        }
        self.dynamic = self.quant_config.is_dynamic()
        if scale is not None:
            sub_weights[scale.name] = scale
        if not self.dynamic and act_scale is not None:
            sub_weights[act_scale.name] = act_scale
        if not self.dynamic and act_scale_inv is not None:
            sub_weights[act_scale_inv.name] = act_scale_inv

        CompositeWeight.__init__(
            self, sub_weights, quant_config=quant_config, *args, **kwargs
        )
        self.kernel = kernel
        self.scale = scale
        if not self.dynamic:
            self.act_scale = act_scale
            self.act_scale_inv = act_scale_inv
        else:
            self.act_scale = None
            self.act_scale_inv = None

    @property
    def qw_suffix(self) -> str:
        return W_SUFFIX

    @property
    def qs_suffix(self) -> str:
        return (
            self.quant_config._weight_s_suffix
            if self.quant_config._weight_s_suffix is not None
            else W_S_SUFFIX
        )

    @property
    def act_s_suffix(self) -> str:
        return (
            self.quant_config._act_s_suffix
            if self.quant_config._act_s_suffix is not None
            else ACT_S_SUFFIX
        )

    def _get_qkv_quant_weight_info(
        self, src_weight_info: AtomicWeight
    ) -> List[W8A8Fp8AtomicWeight]:
        weights: List[CkptWeightInfo] = src_weight_info.weights
        assert len(weights) == 1 or len(weights) == 3

        qkv_name = weights[0].name
        assert qkv_name.endswith(W_SUFFIX)
        qkv_name = qkv_name[: -len(W_SUFFIX)]
        act_s = self.FP8_ACT_SCALE_MAP[src_weight_info.name][0]
        act_s_inv = self.FP8_ACT_SCALE_MAP[src_weight_info.name][1]

        return [
            create_w8a8_fp8_weight(
                src_weight_info,
                W.attn_qkv_w,
                weights,
                merge_te_qkv,
                data_type=torch.float8_e4m3fn,
                config=src_weight_info.config,
            ),
            # qkv weight scale
            create_w8a8_fp8_weight(
                src_weight_info,
                W.attn_qkv_s,
                [
                    CkptWeightInfo(weights[0].name[: -len(W_SUFFIX)] + self.qs_suffix),
                    CkptWeightInfo(weights[1].name[: -len(W_SUFFIX)] + self.qs_suffix),
                    CkptWeightInfo(weights[2].name[: -len(W_SUFFIX)] + self.qs_suffix),
                ],
                stack_,
                data_type=torch.float32,
                config=src_weight_info.config,
            ),
            create_w8a8_fp8_weight(
                src_weight_info,
                act_s[0],
                [
                    CkptWeightInfo(
                        weights[0].name[: -len(W_SUFFIX)] + self.act_s_suffix
                    ),
                    CkptWeightInfo(
                        weights[1].name[: -len(W_SUFFIX)] + self.act_s_suffix
                    ),
                    CkptWeightInfo(
                        weights[2].name[: -len(W_SUFFIX)] + self.act_s_suffix
                    ),
                ],
                act_s[1],
                data_type=torch.float32,
                config=src_weight_info.config,
            ),
            create_w8a8_fp8_weight(
                src_weight_info,
                act_s_inv[0],
                [
                    CkptWeightInfo(
                        weights[0].name[: -len(W_SUFFIX)] + self.act_s_suffix
                    ),
                    CkptWeightInfo(
                        weights[1].name[: -len(W_SUFFIX)] + self.act_s_suffix
                    ),
                    CkptWeightInfo(
                        weights[2].name[: -len(W_SUFFIX)] + self.act_s_suffix
                    ),
                ],
                act_s_inv[1],
                data_type=torch.float32,
                config=src_weight_info.config,
            ),
        ]

    def _get_attn_out_quant_weight_info(
        self, src_weight_info: WeightModule
    ) -> List[W8A8Fp8AtomicWeight]:
        w_name = src_weight_info.weights[0].name[: -len(W_SUFFIX)]
        act_s = self.FP8_ACT_SCALE_MAP[src_weight_info.name][0]
        act_s_inv = self.FP8_ACT_SCALE_MAP[src_weight_info.name][1]

        return [
            create_w8a8_fp8_weight(
                src_weight_info,
                W.attn_o_w,
                [CkptWeightInfo(w_name + self.qw_suffix, identity)],
                identity,
                data_type=torch.float8_e4m3fn,
                config=src_weight_info.config,
            ),
            create_w8a8_fp8_weight(
                src_weight_info,
                W.attn_o_s,
                [CkptWeightInfo(w_name + self.qs_suffix)],
                identity,
                data_type=torch.float32,
                config=src_weight_info.config,
            ),
            create_w8a8_fp8_weight(
                src_weight_info,
                act_s[0],
                [CkptWeightInfo(w_name + self.act_s_suffix)],
                act_s[1],
                data_type=torch.float32,
                config=src_weight_info.config,
            ),
            create_w8a8_fp8_weight(
                src_weight_info,
                act_s_inv[0],
                [CkptWeightInfo(w_name + self.act_s_suffix)],
                act_s_inv[1],
                data_type=torch.float32,
                config=src_weight_info.config,
            ),
        ]

    def _get_ffn_quant_weight_info(
        self, src_weight: Union[FfnAtomicWeight, MoeAtomicWeight], quant_config: Any
    ) -> List[Optional[W8A8Fp8AtomicWeight]]:
        weights = src_weight.weights
        ffn_w_name = src_weight.name
        assert weights[0].name.endswith(W_SUFFIX)
        assert ffn_w_name in [W.ffn_w13, W.ffn_w2, W.moe_w1, W.moe_w2]

        if ffn_w_name in [W.ffn_w2]:
            assert len(weights) == 1
        w_name = weights[0].name[: -len(W_SUFFIX)]
        w: str = None
        s: str = None
        if ffn_w_name in [W.moe_w2, W.moe_w1]:
            if ffn_w_name == W.moe_w1:
                w, s = (W.moe_w1, W.moe_s1)
                stack = stack_moe_w1
            elif ffn_w_name == W.moe_w2:
                w, s = (W.moe_w2, W.moe_s2)
                stack = stack_

            act_s = self.FP8_ACT_SCALE_MAP[ffn_w_name][0]
            act_s_inv = self.FP8_ACT_SCALE_MAP[ffn_w_name][1]
            w_name = [weight.name[: -len(W_SUFFIX)] for weight in weights]
            return [
                create_w8a8_fp8_weight(
                    src_weight,
                    w,
                    [
                        CkptWeightInfo(name + self.qw_suffix, identity)
                        for name in w_name
                    ],
                    stack,
                    data_type=torch.float8_e4m3fn,
                    config=src_weight.config,
                ),
                create_w8a8_fp8_weight(
                    src_weight,
                    s,
                    [
                        CkptWeightInfo(name + self.qs_suffix, identity)
                        for name in w_name
                    ],
                    stack,
                    data_type=torch.float32,
                    config=src_weight.config,
                ),
                create_w8a8_fp8_weight(
                    src_weight,
                    act_s[0],
                    [CkptWeightInfo(name + self.act_s_suffix) for name in w_name],
                    act_s[1],
                    data_type=torch.float32,
                    config=src_weight.config,
                ),
                create_w8a8_fp8_weight(
                    src_weight,
                    act_s_inv[0],
                    [CkptWeightInfo(name + self.act_s_suffix) for name in w_name],
                    act_s_inv[1],
                    data_type=torch.float32,
                    config=src_weight.config,
                ),
            ]
        elif ffn_w_name == W.ffn_w13:
            w, _, s = (W.ffn_w13, W.ffn_b13, W.ffn_s13)
            w1_name = weights[0].name[: -len(W_SUFFIX)]
            w3_name = weights[1].name[: -len(W_SUFFIX)]

            act_s = self.FP8_ACT_SCALE_MAP[W.ffn_w3][0]
            act_s_inv = self.FP8_ACT_SCALE_MAP[W.ffn_w3][1]
            return [
                create_w8a8_fp8_weight(
                    src_weight,
                    w,
                    [
                        CkptWeightInfo(w1_name + self.qw_suffix, identity),
                        CkptWeightInfo(w3_name + self.qw_suffix, identity),
                    ],
                    functools.partial(
                        pad_w13,
                        align_size=src_weight.config.align_size,
                        dim=0,
                    ),
                    data_type=torch.float8_e4m3fn,
                    config=src_weight.config,
                ),
                create_w8a8_fp8_weight(
                    src_weight,
                    s,
                    [
                        CkptWeightInfo(w1_name + self.qs_suffix, identity),
                        CkptWeightInfo(w3_name + self.qs_suffix, identity),
                    ],
                    concat_w13,
                    data_type=torch.float32,
                    config=src_weight.config,
                ),
                create_w8a8_fp8_weight(
                    src_weight,
                    act_s[0],
                    [CkptWeightInfo(w_name + self.act_s_suffix)],
                    act_s[1],
                    data_type=torch.float32,
                    config=src_weight.config,
                ),
                create_w8a8_fp8_weight(
                    src_weight,
                    act_s_inv[0],
                    [CkptWeightInfo(w_name + self.act_s_suffix)],
                    act_s_inv[1],
                    data_type=torch.float32,
                    config=src_weight.config,
                ),
            ]

        else:
            w = src_weight.name
            s = self.FP8_SCALE_MAP.get(src_weight.name)

            w_list = [
                create_w8a8_fp8_weight(
                    src_weight,
                    w,
                    [CkptWeightInfo(w_name + self.qw_suffix, identity)],
                    identity,
                    data_type=torch.float8_e4m3fn,
                    config=src_weight.config,
                ),
                create_w8a8_fp8_weight(
                    src_weight,
                    s,
                    [CkptWeightInfo(w_name + self.qs_suffix, identity)],
                    identity,
                    data_type=torch.float32,
                    config=src_weight.config,
                ),
            ]
            if w in self.FP8_ACT_SCALE_MAP:
                act_s = self.FP8_ACT_SCALE_MAP[src_weight.name][0]
                act_s_inv = self.FP8_ACT_SCALE_MAP[src_weight.name][1]
                w_list.extend(
                    [
                        create_w8a8_fp8_weight(
                            src_weight,
                            act_s[0],
                            [CkptWeightInfo(w_name + self.act_s_suffix)],
                            act_s[1],
                            data_type=torch.float32,
                            config=src_weight.config,
                        ),
                        create_w8a8_fp8_weight(
                            src_weight,
                            act_s_inv[0],
                            [CkptWeightInfo(w_name + self.act_s_suffix)],
                            act_s_inv[1],
                            data_type=torch.float32,
                            config=src_weight.config,
                        ),
                    ]
                )
            else:
                w_list.extend([None, None])
            return w_list

    def _postprocess(
        self,
        tensor: Union[torch.Tensor, Dict[str, torch.Tensor]],
        device: str,
        load_config: LoadConfig,
    ):
        # this func do nothing but is called here
        processed_res = super()._postprocess(tensor, device, load_config)

        kernel_weight = processed_res[self.kernel.name]
        weight_scale_name = self.FP8_SCALE_MAP.get(self.kernel.name)

        input_scale_r_str, _ = self.FP8_ACT_SCALE_MAP.get(self.kernel.name)[1]
        intput_scale_str, _ = self.FP8_ACT_SCALE_MAP.get(self.kernel.name)[0]

        kernel_scale = processed_res.get(weight_scale_name, None)
        input_scale = processed_res.get(input_scale_r_str, None)

        if isinstance(self.kernel, MoeAtomicWeight):
            if self.kernel.name is W.moe_w1:
                # handle moe w13 weight (w13 is concatenated, so split in half)
                num_local_experts, total_padded_size, _ = kernel_weight.shape
                assert kernel_scale is not None

                # Total padded size should be 2x the individual w1/w3 padded size
                half_size = total_padded_size // 2
                max_kernel_scale = kernel_scale.max(dim=1).values

                # Rescale each expert's w1 and w3 shards if needed
                for expert_id in range(num_local_experts):
                    start = 0
                    for shard_id in range(2):
                        if (
                            max_kernel_scale[expert_id]
                            != kernel_scale[expert_id][shard_id]
                        ):
                            # rescale shard
                            dq_weight = (
                                kernel_weight[expert_id][
                                    start : start + half_size, :
                                ].to(torch.float16)
                                * kernel_scale[expert_id][shard_id]
                            )
                            kernel_weight[expert_id][start : start + half_size, :] = (
                                dq_weight / max_kernel_scale[expert_id]
                            ).to(torch.float8_e4m3fn)

                processed_res[self.kernel.name] = kernel_weight
                processed_res[W.moe_s1] = max_kernel_scale

            if input_scale is not None:
                input_scale = input_scale.max()
                processed_res[input_scale_r_str] = input_scale
                processed_res[intput_scale_str] = 1.0 / input_scale
            return processed_res

        # handle qkv_proj quant weight
        if self.kernel.name is W.attn_qkv_w:
            kernel_weight = processed_res[self.kernel.name]
            kernel_scale = processed_res[W.attn_qkv_s]

            head_size = load_config.size_per_head
            head_num_kv = load_config.head_num_kv // load_config.tp_size
            head_num_q = load_config.head_num // load_config.tp_size
            assert (head_num_q + 2 * head_num_kv) == kernel_weight.shape[0] // head_size
            logical_widths = [
                head_num_q * head_size,
                head_num_kv * head_size,
                head_num_kv * head_size,
            ]

            qkv_rescale, max_scale = merge_qkv_hf_fp8_with_scale(
                [
                    kernel_weight[0 : logical_widths[0], :],
                    kernel_weight[
                        logical_widths[0] : logical_widths[0] + logical_widths[1], :
                    ],
                    kernel_weight[logical_widths[0] + logical_widths[1] :, :],
                    kernel_scale[0],
                    kernel_scale[1],
                    kernel_scale[2],
                ]
            )
            processed_res[self.kernel.name] = qkv_rescale
            processed_res[W.attn_qkv_s] = max_scale

            # maybe handle qkv_proj input scale
            if processed_res.get(W.pre_ln_static_quant_reciprocal) is not None:
                assert processed_res[W.pre_ln_static_quant_reciprocal].shape[0] == 3
                processed_res[W.pre_ln_static_quant_reciprocal] = processed_res[
                    W.pre_ln_static_quant_reciprocal
                ].max()
                processed_res[W.pre_ln_static_quant] = (
                    1.0 / processed_res[W.pre_ln_static_quant_reciprocal]
                )
            return processed_res

        return processed_res
