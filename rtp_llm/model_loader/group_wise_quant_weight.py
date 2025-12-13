import functools
from typing import Any, Dict, List, Optional, Union

import torch

from rtp_llm.config.quant_config import AWQConfig, GPTQConfig, QuantizationConfig
from rtp_llm.model_loader.attn_weight import AttnAtomicWeight, MlaAttnAtomicWeight
from rtp_llm.model_loader.ffn_weight import FfnAtomicWeight, MoeAtomicWeight
from rtp_llm.model_loader.load_config import LoadConfig
from rtp_llm.model_loader.weight_module import (
    AtomicWeight,
    CompositeWeight,
    QuantWeight,
    WeightModule,
)
from rtp_llm.utils.model_weight import (
    CkptWeightInfo,
    W,
    identity,
    merge_qkv_hf,
    pad,
    pad_w13,
    stack_,
    stack_moe_w1,
    transpose,
)
from rtp_llm.utils.util import check_with_info

QW_SUFFIX = ".qweight"
QZ_SUFFIX = ".qzeros"
QS_SUFFIX = ".scales"
W_SUFFIX = ".weight"


def get_qkv_quant_weight_info(
    src_weight: Union[AttnAtomicWeight, MlaAttnAtomicWeight]
) -> List[AtomicWeight]:
    weights = src_weight.weights
    assert len(weights) == 1 or len(weights) == 3
    if len(weights) == 3:
        q_name = weights[0].name
        k_name = weights[1].name
        v_name = weights[2].name
        check_with_info(
            q_name.endswith(W_SUFFIX)
            and k_name.endswith(W_SUFFIX)
            and v_name.endswith(W_SUFFIX),
            f"qkv weight name must end with .weight, {q_name}, {k_name}, {v_name}",
        )
        q_name = q_name[: -len(W_SUFFIX)]
        k_name = k_name[: -len(W_SUFFIX)]
        v_name = v_name[: -len(W_SUFFIX)]
        return [
            src_weight.create_from(
                W.attn_qkv_w,
                [
                    CkptWeightInfo(q_name + QW_SUFFIX, transpose),
                    CkptWeightInfo(k_name + QW_SUFFIX, transpose),
                    CkptWeightInfo(v_name + QW_SUFFIX, transpose),
                ],
                functools.partial(merge_qkv_hf),
                data_type=torch.int32,
                config=src_weight.config,
            ),
            src_weight.create_from(
                W.attn_qkv_z,
                [
                    CkptWeightInfo(q_name + QZ_SUFFIX, transpose),
                    CkptWeightInfo(k_name + QZ_SUFFIX, transpose),
                    CkptWeightInfo(v_name + QZ_SUFFIX, transpose),
                ],
                functools.partial(merge_qkv_hf),
                data_type=torch.int32,
                config=src_weight.config,
            ),
            src_weight.create_from(
                W.attn_qkv_s,
                [
                    CkptWeightInfo(q_name + QS_SUFFIX, transpose),
                    CkptWeightInfo(k_name + QS_SUFFIX, transpose),
                    CkptWeightInfo(v_name + QS_SUFFIX, transpose),
                ],
                functools.partial(merge_qkv_hf),
                config=src_weight.config,
            ),
        ]
    else:
        qkv_name = weights[0].name
        assert qkv_name.endswith(W_SUFFIX)
        qkv_name = qkv_name[: -len(W_SUFFIX)]
        return [
            src_weight.create_from(
                W.attn_qkv_w,
                [CkptWeightInfo(qkv_name + QW_SUFFIX, identity)],
                identity,
                data_type=torch.int32,
                config=src_weight.config,
            ),
            src_weight.create_from(
                W.attn_qkv_z,
                [CkptWeightInfo(qkv_name + QZ_SUFFIX, identity)],
                identity,
                data_type=torch.int32,
                config=src_weight.config,
            ),
            src_weight.create_from(
                W.attn_qkv_s,
                [CkptWeightInfo(qkv_name + QS_SUFFIX)],
                identity,
                config=src_weight.config,
            ),
        ]


def get_ffn_quant_weight_info(
    src_weight: Union[FfnAtomicWeight, MoeAtomicWeight], quant_config: Any
) -> List[Union[FfnAtomicWeight, MoeAtomicWeight]]:
    weights = src_weight.weights
    ffn_w_name = src_weight.name
    assert weights[0].name.endswith(W_SUFFIX)
    assert ffn_w_name in [W.ffn_w1, W.ffn_w2, W.ffn_w3, W.ffn_w13, W.moe_w1, W.moe_w2]
    # Use align_size instead of inter_padding_size for dynamic padding
    align_size = src_weight.config.align_size

    if ffn_w_name in [W.ffn_w1, W.ffn_w2, W.ffn_w3]:
        assert len(weights) == 1
    w_name = weights[0].name[: -len(W_SUFFIX)]
    group_size = quant_config.group_size()
    pad_div = 32 // quant_config.bits
    is_awq = isinstance(quant_config, AWQConfig)
    is_gptq = isinstance(quant_config, GPTQConfig)
    w: str = None
    s: str = None
    z: str = None
    stack: Callable = None
    act_w = None
    if ffn_w_name == W.ffn_w2:
        if src_weight.config.need_ffn_act_scale:
            act_w_name = w_name.rsplit(".", 1)[0] + ".act.scales"
            act_w = FfnAtomicWeight(
                W.ffn_act_s,
                [CkptWeightInfo(act_w_name, identity)],
                identity,
                config=src_weight.config,
            )
        return [
            FfnAtomicWeight(
                W.ffn_w2,
                [CkptWeightInfo(w_name + QW_SUFFIX, identity)],
                functools.partial(
                    pad,
                    align_size=(
                        align_size // pad_div if is_gptq else align_size
                    ),
                    dim=0,
                ),
                data_type=torch.int32,
                config=src_weight.config,
            ),
            FfnAtomicWeight(
                W.ffn_z2,
                [CkptWeightInfo(w_name + QZ_SUFFIX, identity)],
                functools.partial(
                    pad, align_size=align_size // group_size, dim=0
                ),
                data_type=torch.int32,
                config=src_weight.config,
            ),
            FfnAtomicWeight(
                W.ffn_s2,
                [CkptWeightInfo(w_name + QS_SUFFIX, identity)],
                functools.partial(
                    pad, align_size=align_size // group_size, dim=0
                ),
                config=src_weight.config,
            ),
            act_w,
        ]
    elif ffn_w_name in [W.moe_w2, W.moe_w1]:
        if ffn_w_name == W.moe_w1:
            w, z, s = (W.moe_w1, W.moe_z1, W.moe_s1)
            stack = stack_moe_w1
        elif ffn_w_name == W.moe_w2:
            w, z, s = (W.moe_w2, W.moe_z2, W.moe_s2)
            stack = stack_

        w_name = [weight.name[: -len(W_SUFFIX)] for weight in weights]
        return [
            MoeAtomicWeight(
                w,
                [CkptWeightInfo(name + QW_SUFFIX, transpose) for name in w_name],
                stack,
                data_type=torch.int32,
                config=src_weight.config,
            ),
            MoeAtomicWeight(
                z,
                [CkptWeightInfo(name + QZ_SUFFIX, transpose) for name in w_name],
                stack,
                data_type=torch.int32,
                config=src_weight.config,
            ),
            MoeAtomicWeight(
                s,
                [CkptWeightInfo(name + QS_SUFFIX, transpose) for name in w_name],
                stack,
                config=src_weight.config,
            ),
            act_w,
        ]
    elif ffn_w_name == W.ffn_w13:
        w, z, s = (W.ffn_w13, W.ffn_z13, W.ffn_s13)
        w1_name = weights[0].name[: -len(W_SUFFIX)]
        w3_name = weights[1].name[: -len(W_SUFFIX)]
        return [
            FfnAtomicWeight(
                w,
                [
                    CkptWeightInfo(w1_name + QW_SUFFIX, identity),
                    CkptWeightInfo(w3_name + QW_SUFFIX, identity),
                ],
                functools.partial(
                    pad_w13,
                    align_size=(
                        align_size // pad_div if is_awq else align_size
                    ),
                    dim=1,
                ),
                data_type=torch.int32,
                config=src_weight.config,
            ),
            FfnAtomicWeight(
                z,
                [
                    CkptWeightInfo(w1_name + QZ_SUFFIX, identity),
                    CkptWeightInfo(w3_name + QZ_SUFFIX, identity),
                ],
                functools.partial(
                    pad_w13,
                    align_size=align_size // pad_div,
                    dim=1,
                ),
                data_type=torch.int32,
                config=src_weight.config,
            ),
            FfnAtomicWeight(
                s,
                [
                    CkptWeightInfo(w1_name + QS_SUFFIX, identity),
                    CkptWeightInfo(w3_name + QS_SUFFIX, identity),
                ],
                functools.partial(
                    pad_w13,
                    align_size=align_size,
                    dim=1,
                ),
                config=src_weight.config,
            ),
            act_w,
        ]
    else:
        w, z, s = (
            (W.ffn_w1, W.ffn_z1, W.ffn_s1)
            if ffn_w_name == W.ffn_w1
            else (W.ffn_w3, W.ffn_z3, W.ffn_s3)
        )
        return [
            FfnAtomicWeight(
                w,
                [CkptWeightInfo(w_name + QW_SUFFIX, identity)],
                functools.partial(
                    pad,
                    align_size=(
                        align_size // pad_div if is_awq else align_size
                    ),
                    dim=1,
                ),
                data_type=torch.int32,
                config=src_weight.config,
            ),
            FfnAtomicWeight(
                z,
                [CkptWeightInfo(w_name + QZ_SUFFIX, identity)],
                functools.partial(
                    pad,
                    align_size=align_size // pad_div,
                    dim=1,
                ),
                data_type=torch.int32,
                config=src_weight.config,
            ),
            FfnAtomicWeight(
                s,
                [CkptWeightInfo(w_name + QS_SUFFIX, identity)],
                functools.partial(
                    pad, align_size=align_size, dim=1
                ),
                config=src_weight.config,
            ),
            act_w,
        ]


class GroupWiseWeight(CompositeWeight, QuantWeight):
    group_wise_w = [
        W.attn_qkv_w,
        W.attn_o_w,
        W.ffn_w1,
        W.ffn_w2,
        W.ffn_w3,
        W.ffn_w13,
        W.moe_w1,
        W.moe_w2,
    ]

    @classmethod
    def support(
        cls, quant_config: QuantizationConfig, src_weight_info: WeightModule
    ) -> bool:
        if not quant_config.is_quanted() or not (
            isinstance(quant_config, AWQConfig) or isinstance(quant_config, GPTQConfig)
        ):
            return False
        name = src_weight_info.name
        return quant_config.group_size() > 0 and name in cls.group_wise_w

    def __init__(
        self, src_weight_info: AtomicWeight, quant_config: QuantizationConfig, **kwargs
    ):
        self.quant_config = quant_config
        kernel: AtomicWeight
        zero: AtomicWeight
        scale: AtomicWeight
        act_scale: Optional[AtomicWeight] = None
        if src_weight_info.name == W.attn_qkv_w:
            (kernel, zero, scale) = get_qkv_quant_weight_info(src_weight_info)
        elif src_weight_info.name == W.attn_o_w:
            w_name = src_weight_info.weights[0].name[: -len(W_SUFFIX)]
            kernel = src_weight_info.create_from(
                W.attn_o_w,
                [CkptWeightInfo(w_name + QW_SUFFIX, identity)],
                identity,
                data_type=torch.int32,
                config=src_weight_info.config,
            )
            zero = src_weight_info.create_from(
                W.attn_o_z,
                [CkptWeightInfo(w_name + QZ_SUFFIX, identity)],
                identity,
                data_type=torch.int32,
                config=src_weight_info.config,
            )
            scale = src_weight_info.create_from(
                W.attn_o_s,
                [CkptWeightInfo(w_name + QS_SUFFIX, identity)],
                identity,
                config=src_weight_info.config,
            )
        elif src_weight_info.name in [
            W.ffn_w1,
            W.ffn_w2,
            W.ffn_w3,
            W.moe_w1,
            W.moe_w2,
            W.ffn_w13,
        ]:
            (kernel, zero, scale, act_scale) = get_ffn_quant_weight_info(
                src_weight_info, quant_config
            )
        else:
            raise ValueError(f"Unsupported weight name {src_weight_info.name}")
        sub_weights = {kernel.name: kernel, zero.name: zero, scale.name: scale}
        if act_scale:
            sub_weights.update({act_scale.name: act_scale})

        super().__init__(sub_weights, quant_config=quant_config, **kwargs)
        self.kernel = self.sub_weights[kernel.name]
        self.zero = self.sub_weights[zero.name]
        self.scale = self.sub_weights[scale.name]
        self.act_scale = self.sub_weights.get(act_scale.name) if act_scale else None
        self.src_weight_info = src_weight_info

    def _postprocess(
        self,
        tensor: Union[torch.Tensor, Dict[str, torch.Tensor]],
        device: str,
        load_config: LoadConfig,
    ):
        kernel = tensor[self.kernel.name]
        zero = tensor[self.zero.name]
        scale = tensor[self.scale.name]
        act_scale = tensor.get(self.act_scale.name) if self.act_scale else None
        if self.kernel.name in [W.attn_qkv_w, W.attn_o_w]:
            post_func = load_config.exported_device.preprocess_groupwise_weight_params
        elif (
            self.kernel.name in [W.ffn_w1, W.ffn_w2, W.ffn_w3, W.ffn_w13]
            or self.kernel.name in [W.moe_w1, W.moe_w2]
        ) and self.kernel.config.is_moe:
            post_func = (
                load_config.exported_device.preprocess_moe_groupwise_weight_params
            )
        elif (
            self.kernel.name in [W.ffn_w1, W.ffn_w2, W.ffn_w3, W.ffn_w13]
        ) and not self.kernel.config.is_moe:
            post_func = load_config.exported_device.preprocess_groupwise_weight_params
        else:
            raise ValueError(f"Unsupported weight name {self.kernel.name}")

        weight, zero, scale = post_func(
            kernel,
            zero,
            scale,
            device,
            isinstance(self.quant_config, GPTQConfig),
            isinstance(self.quant_config, AWQConfig),
            self.quant_config.bits,
        )
        sub_tensors = {
            self.kernel.name: weight,
            self.zero.name: zero,
            self.scale.name: scale,
        }
        if act_scale is not None:
            sub_tensors[self.act_scale.name] = act_scale
        return sub_tensors
