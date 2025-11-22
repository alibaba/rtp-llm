import functools
import logging
from typing import Any, Dict, List, Optional, Union

import torch

from rtp_llm.config.quant_config import QuantizationConfig, SmoothQuantConfig
from rtp_llm.model_loader.ffn_weight import FfnAtomicWeight, MoeAtomicWeight
from rtp_llm.model_loader.load_config import LoadConfig
from rtp_llm.model_loader.w8a8_weight import (
    W8A8Int8AtomicWeight,
    create_w8a8_int8_weight,
)
from rtp_llm.model_loader.weight_module import (
    AtomicWeight,
    CompositeWeight,
    QuantWeight,
    WeightModule,
)
from rtp_llm.utils.model_weight import (
    CkptWeightInfo,
    W,
    WeightStyle,
    concat_w13,
    identity,
    merge_qkv_hf,
    merge_qkv_transpose_concat0,
    merge_te_qkv,
    stack_,
    stack_moe_w1,
    transpose,
    transpose_w13_2,
)

QW_SUFFIX = ".qweight"
QS_SUFFIX = ".scales"
W_SUFFIX = ".weight"
B_SUFFIX = ".bias"
SMOOTHER_SUFFIX = ".smoother"
INT8QW_COL_SUFFIX = ".weight.int8.col"
INT8QS_COL_SUFFIX = ".scale_w_quant_orig.col"


def qkv_transpose(ts, hidden_size):
    return ts[0].reshape(hidden_size, -1)


class SmoothQuantWeightInfo(CompositeWeight, QuantWeight):
    w8a8_weight_list = [
        W.attn_qkv_w,
        W.attn_o_w,
        W.ffn_w1,
        W.ffn_w3,
        W.ffn_w2,
        W.ffn_w13,
    ]

    @classmethod
    def support(
        cls, quant_config: QuantizationConfig, src_weight_info: WeightModule
    ) -> bool:
        if not quant_config.is_quanted() or not isinstance(
            quant_config, SmoothQuantConfig
        ):
            return False
        name = src_weight_info.name
        logging.debug("src_weight_info.weight_style : %s", src_weight_info.weight_style)
        return name in cls.w8a8_weight_list and (
            src_weight_info.weight_style
            not in [WeightStyle.TRT_ENGINE, WeightStyle.RTP_SMOOTH_LLM_STYLE]
        )

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
        smoother: Optional[AtomicWeight] = None
        logging.debug(
            f"SmoothQuantWeightInfo : {self.qs_suffix}, {self.qw_suffix}, {src_weight_info.name}"
        )

        if src_weight_info.name == W.attn_qkv_w:
            (kernel, scale, smoother) = self._get_qkv_quant_weight_info(src_weight_info)
        elif src_weight_info.name == W.attn_o_w:
            (kernel, scale, smoother) = self._get_attn_out_quant_weight_info(
                src_weight_info
            )
        elif src_weight_info.name in [
            W.ffn_w1,
            W.ffn_w2,
            W.ffn_w3,
            W.ffn_w13,
            W.moe_w1,
            W.moe_w2,
        ]:
            (kernel, scale, smoother) = self._get_ffn_quant_weight_info(
                src_weight_info, quant_config
            )
        else:
            raise ValueError(f"Unsupported weight name {src_weight_info.name}")

        sub_weights = {
            kernel.name: kernel,
        }

        if scale is not None:
            sub_weights[scale.name] = scale
        if smoother is not None:
            sub_weights[smoother.name] = smoother

        super().__init__(sub_weights, quant_config=quant_config, *args, **kwargs)
        self.kernel = kernel
        self.scale = scale
        self.smoother = smoother

    @property
    def qw_suffix(self) -> str:
        return QW_SUFFIX

    @property
    def qs_suffix(self) -> str:
        return QS_SUFFIX

    def _get_qkv_quant_weight_info(
        self, src_weight_info: AtomicWeight
    ) -> List[Optional[W8A8Int8AtomicWeight]]:
        weights: List[CkptWeightInfo] = src_weight_info.weights
        assert len(weights) == 1 or len(weights) == 3
        if len(weights) == 3:
            q_name = weights[0].name
            k_name = weights[1].name
            v_name = weights[2].name
            assert (
                q_name.endswith(W_SUFFIX)
                and k_name.endswith(W_SUFFIX)
                and v_name.endswith(W_SUFFIX)
            )
            q_name = q_name[: -len(W_SUFFIX)]
            k_name = k_name[: -len(W_SUFFIX)]
            v_name = v_name[: -len(W_SUFFIX)]
            return [
                create_w8a8_int8_weight(
                    src_weight_info,
                    W.attn_qkv_w,
                    [
                        CkptWeightInfo(q_name + self.qw_suffix, identity),
                        CkptWeightInfo(k_name + self.qw_suffix, identity),
                        CkptWeightInfo(v_name + self.qw_suffix, identity),
                    ],
                    merge_qkv_transpose_concat0,
                    data_type=torch.int8,
                    config=src_weight_info.config,
                ),
                create_w8a8_int8_weight(
                    src_weight_info,
                    W.attn_qkv_s,
                    [
                        CkptWeightInfo(q_name + self.qs_suffix, identity),
                        CkptWeightInfo(k_name + self.qs_suffix, identity),
                        CkptWeightInfo(v_name + self.qs_suffix, identity),
                    ],
                    merge_te_qkv,
                    data_type=torch.float32,
                    config=src_weight_info.config,
                ),
                None,
            ]
        else:
            qkv_name = weights[0].name
            assert qkv_name.endswith(W_SUFFIX)
            qkv_name = qkv_name[: -len(W_SUFFIX)]
            return [
                create_w8a8_int8_weight(
                    src_weight_info,
                    W.attn_qkv_w,
                    [
                        CkptWeightInfo(
                            qkv_name + self.qw_suffix,
                            functools.partial(
                                qkv_transpose,
                                hidden_size=src_weight_info.config.hidden_size,
                            ),
                        )
                    ],
                    transpose,
                    data_type=torch.int8,
                    config=src_weight_info.config,
                ),
                create_w8a8_int8_weight(
                    src_weight_info,
                    W.attn_qkv_s,
                    [CkptWeightInfo(qkv_name + self.qs_suffix)],
                    identity,
                    data_type=torch.float32,
                    config=src_weight_info.config,
                ),
                None,
            ]

    def _get_attn_out_quant_weight_info(
        self, src_weight_info: WeightModule
    ) -> List[W8A8Int8AtomicWeight]:
        w_name = src_weight_info.weights[0].name[: -len(W_SUFFIX)]
        kernel = create_w8a8_int8_weight(
            src_weight_info,
            W.attn_o_w,
            [CkptWeightInfo(w_name + self.qw_suffix, identity)],
            transpose,
            data_type=torch.int8,
            config=src_weight_info.config,
        )
        scale = create_w8a8_int8_weight(
            src_weight_info,
            W.attn_o_s,
            [CkptWeightInfo(w_name + self.qs_suffix)],
            identity,
            data_type=torch.float32,
            config=src_weight_info.config,
        )
        smoother = create_w8a8_int8_weight(
            src_weight_info,
            W.attn_o_smoother,
            [CkptWeightInfo(w_name + SMOOTHER_SUFFIX)],
            identity,
            data_type=torch.float32,
            config=src_weight_info.config,
        )
        return [kernel, scale, smoother]

    def _get_ffn_quant_weight_info(
        self, src_weight: Union[FfnAtomicWeight, MoeAtomicWeight], quant_config: Any
    ) -> List[Optional[W8A8Int8AtomicWeight]]:
        weights = src_weight.weights
        ffn_w_name = src_weight.name
        assert weights[0].name.endswith(
            W_SUFFIX
        ), f"{weights[0].name} not endswith {W_SUFFIX}"
        assert ffn_w_name in [W.ffn_w13, W.ffn_w2, W.moe_w1, W.moe_w2]

        if ffn_w_name in [W.ffn_w2]:
            assert len(weights) == 1
        w_name = weights[0].name[: -len(W_SUFFIX)]

        if ffn_w_name in [W.moe_w2, W.moe_w1]:
            if ffn_w_name == W.moe_w1:
                w, b, s = (W.moe_w1, W.moe_b1, W.moe_s1)
                stack = stack_moe_w1
            elif ffn_w_name == W.moe_w2:
                w, b, s = (W.moe_w2, W.moe_b2, W.moe_s2)
                stack = stack_

            w_name = [weight.name[: -len(W_SUFFIX)] for weight in weights]
            return [
                create_w8a8_int8_weight(
                    src_weight,
                    w,
                    [
                        CkptWeightInfo(name + self.qw_suffix, transpose)
                        for name in w_name
                    ],
                    stack,
                    data_type=torch.int8,
                    config=src_weight.config,
                ),
                create_w8a8_int8_weight(
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
                None,
            ]

        elif ffn_w_name == W.ffn_w2:
            w, b, s = (W.ffn_w2, W.ffn_b2, W.ffn_s2)

            return [
                create_w8a8_int8_weight(
                    src_weight,
                    w,
                    [CkptWeightInfo(w_name + self.qw_suffix, identity)],
                    transpose,
                    data_type=torch.int8,
                    config=src_weight.config,
                ),
                create_w8a8_int8_weight(
                    src_weight,
                    s,
                    [CkptWeightInfo(w_name + self.qs_suffix, identity)],
                    identity,
                    data_type=torch.float32,
                    config=src_weight.config,
                ),
                create_w8a8_int8_weight(
                    src_weight,
                    W.ffn_smoother,
                    [CkptWeightInfo(w_name + SMOOTHER_SUFFIX)],
                    identity,
                    data_type=torch.float32,
                    config=src_weight.config,
                ),
            ]
        elif ffn_w_name == W.ffn_w13:
            w, _, s = (W.ffn_w13, W.ffn_b13, W.ffn_s13)
            w1_name = weights[0].name[: -len(W_SUFFIX)]
            w3_name = weights[1].name[: -len(W_SUFFIX)]
            return [
                create_w8a8_int8_weight(
                    src_weight,
                    w,
                    [
                        CkptWeightInfo(w1_name + self.qw_suffix, identity),
                        CkptWeightInfo(w3_name + self.qw_suffix, identity),
                    ],
                    transpose_w13_2,
                    data_type=torch.int8,
                    config=src_weight.config,
                ),
                None,
                create_w8a8_int8_weight(
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
            ]
        else:
            if ffn_w_name == W.ffn_w1:
                w, b, s = (W.ffn_w1, W.ffn_b1, W.ffn_s1)
            elif ffn_w_name == W.ffn_w3:
                w, b, s = (W.ffn_w3, W.ffn_b3, W.ffn_s3)
            else:
                raise NotImplementedError(
                    f"ffn_w_name {ffn_w_name} not supported in omni_quant"
                )
            return [
                create_w8a8_int8_weight(
                    src_weight,
                    w,
                    [CkptWeightInfo(w_name + self.qw_suffix, identity)],
                    transpose,
                    data_type=torch.int8,
                    config=src_weight.config,
                ),
                create_w8a8_int8_weight(
                    src_weight,
                    s,
                    [CkptWeightInfo(w_name + self.qs_suffix, identity)],
                    identity,
                    data_type=torch.float32,
                    config=src_weight.config,
                ),
                None,
            ]

    def _postprocess(
        self,
        tensor: Union[torch.Tensor, Dict[str, torch.Tensor]],
        device: str,
        load_config: LoadConfig,
    ):
        # need reshape for kernel weight
        processed_res = super()._postprocess(tensor, device, load_config)
        kernel_weight = processed_res[self.kernel.name]
        kernel_weight = kernel_weight.reshape(kernel_weight.shape[-1], -1)
        processed_res[self.kernel.name] = kernel_weight

        return processed_res


class TrtEngineSmoothQuantWeightInfo(SmoothQuantWeightInfo):
    def __init__(
        self,
        src_weight_info: AtomicWeight,
        quant_config: QuantizationConfig,
        *args,
        **kwargs,
    ):
        super().__init__(src_weight_info, quant_config, *args, **kwargs)

    @classmethod
    def support(
        cls, quant_config: QuantizationConfig, src_weight_info: WeightModule
    ) -> bool:
        if not quant_config.is_quanted() or not isinstance(
            quant_config, SmoothQuantConfig
        ):
            return False
        name = src_weight_info.name
        return (
            name in cls.w8a8_weight_list
            and src_weight_info.weight_style == WeightStyle.TRT_ENGINE
        )

    def _get_qkv_quant_weight_info(
        self, src_weight_info: WeightModule
    ) -> List[Optional[W8A8Int8AtomicWeight]]:
        weights: List[CkptWeightInfo] = src_weight_info.weights
        assert len(weights) == 1 or len(weights) == 3
        if len(weights) == 3:
            q_name = weights[0].name
            k_name = weights[1].name
            v_name = weights[2].name
            assert (
                q_name.endswith(W_SUFFIX)
                and k_name.endswith(W_SUFFIX)
                and v_name.endswith(W_SUFFIX)
            )
            q_name = q_name[: -len(W_SUFFIX)]
            k_name = k_name[: -len(W_SUFFIX)]
            v_name = v_name[: -len(W_SUFFIX)]
            return [
                create_w8a8_int8_weight(
                    src_weight_info,
                    W.attn_qkv_w,
                    [
                        CkptWeightInfo(q_name + self.qw_suffix, identity),
                        CkptWeightInfo(k_name + self.qw_suffix, identity),
                        CkptWeightInfo(v_name + self.qw_suffix, identity),
                    ],
                    functools.partial(merge_qkv_hf),
                    data_type=torch.int8,
                    config=src_weight_info.config,
                ),
                create_w8a8_int8_weight(
                    src_weight_info,
                    W.attn_qkv_s,
                    [
                        CkptWeightInfo(q_name + self.qs_suffix, identity),
                        CkptWeightInfo(k_name + self.qs_suffix, identity),
                        CkptWeightInfo(v_name + self.qs_suffix, identity),
                    ],
                    functools.partial(merge_qkv_hf),
                    data_type=torch.float32,
                    config=src_weight_info.config,
                ),
                None,
            ]
        else:
            qkv_name = weights[0].name
            assert qkv_name.endswith(W_SUFFIX)
            qkv_name = qkv_name[: -len(W_SUFFIX)]
            return [
                create_w8a8_int8_weight(
                    src_weight_info,
                    W.attn_qkv_w,
                    [CkptWeightInfo(qkv_name + self.qw_suffix)],
                    identity,
                    data_type=torch.int8,
                    config=src_weight_info.config,
                ),
                create_w8a8_int8_weight(
                    src_weight_info,
                    W.attn_qkv_s,
                    [CkptWeightInfo(qkv_name + self.qs_suffix)],
                    identity,
                    data_type=torch.float32,
                    config=src_weight_info.config,
                ),
                None,
            ]

    def _get_attn_out_quant_weight_info(
        self, src_weight_info: AtomicWeight
    ) -> List[AtomicWeight]:
        w_name = src_weight_info.weights[0].name[: -len(W_SUFFIX)]
        kernel = create_w8a8_int8_weight(
            src_weight_info,
            W.attn_o_w,
            [CkptWeightInfo(w_name + self.qw_suffix, identity)],
            identity,
            data_type=torch.int8,
            config=src_weight_info.config,
        )
        scale = create_w8a8_int8_weight(
            src_weight_info,
            W.attn_o_s,
            [CkptWeightInfo(w_name + self.qs_suffix)],
            identity,
            data_type=torch.float32,
            config=src_weight_info.config,
        )
        smoother = create_w8a8_int8_weight(
            src_weight_info,
            W.attn_o_smoother,
            [CkptWeightInfo(w_name + SMOOTHER_SUFFIX)],
            identity,
            data_type=torch.float32,
            config=src_weight_info.config,
        )
        return [kernel, scale, smoother]

    def _get_ffn_quant_weight_info(
        self, src_weight: FfnAtomicWeight, quant_config: Any
    ) -> List[Optional[W8A8Int8AtomicWeight]]:
        weights = src_weight.weights
        ffn_w_name = src_weight.name
        assert weights[0].name.endswith(W_SUFFIX)
        assert ffn_w_name in [
            W.ffn_w13,
            W.ffn_w2,
            W.moe_w1,
            W.moe_w2,
        ]

        if ffn_w_name in [W.ffn_w2]:
            assert len(weights) == 1
        w_name = weights[0].name[: -len(W_SUFFIX)]

        if ffn_w_name in [W.moe_w2, W.moe_w1]:
            if ffn_w_name == W.moe_w1:
                w, b, s = (W.moe_w1, W.moe_b1, W.moe_s1)
                stack = stack_moe_w1
            elif ffn_w_name == W.moe_w2:
                w, b, s = (W.moe_w2, W.moe_b2, W.moe_s2)
                stack = stack_

            w_name = [weight.name[: -len(W_SUFFIX)] for weight in weights]
            return [
                create_w8a8_int8_weight(
                    src_weight,
                    w,
                    [
                        CkptWeightInfo(name + self.qw_suffix, identity)
                        for name in w_name
                    ],
                    stack,
                    data_type=torch.int8,
                    config=src_weight.config,
                ),
                create_w8a8_int8_weight(
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
                None,
            ]

        elif ffn_w_name == W.ffn_w2:
            w, b, s = (W.ffn_w2, W.ffn_b2, W.ffn_s2)

            return [
                create_w8a8_int8_weight(
                    src_weight,
                    w,
                    [CkptWeightInfo(w_name + self.qw_suffix, identity)],
                    identity,
                    data_type=torch.int8,
                    config=src_weight.config,
                ),
                create_w8a8_int8_weight(
                    src_weight,
                    s,
                    [CkptWeightInfo(w_name + self.qs_suffix, identity)],
                    identity,
                    data_type=torch.float32,
                    config=src_weight.config,
                ),
                create_w8a8_int8_weight(
                    src_weight,
                    W.ffn_smoother,
                    [CkptWeightInfo(w_name + SMOOTHER_SUFFIX)],
                    identity,
                    data_type=torch.float32,
                    config=src_weight.config,
                ),
            ]
        elif ffn_w_name == W.ffn_w13:
            w, _, s = (W.ffn_w13, W.ffn_b13, W.ffn_s13)
            w1_name = weights[0].name[: -len(W_SUFFIX)]
            w3_name = weights[1].name[: -len(W_SUFFIX)]
            return [
                create_w8a8_int8_weight(
                    src_weight,
                    w,
                    [
                        CkptWeightInfo(w1_name + self.qw_suffix, identity),
                        CkptWeightInfo(w3_name + self.qw_suffix, identity),
                    ],
                    transpose_w13_2,
                    data_type=torch.int8,
                    config=src_weight.config,
                ),
                None,
                create_w8a8_int8_weight(
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
            ]
        else:
            if ffn_w_name == W.ffn_w1:
                w, b, s = (W.ffn_w1, W.ffn_b1, W.ffn_s1)
            elif ffn_w_name == W.ffn_w3:
                w, b, s = (W.ffn_w3, W.ffn_b3, W.ffn_s3)
            else:
                raise NotImplementedError(
                    f"ffn_w_name {ffn_w_name} not supported in omni_quant"
                )
            return [
                create_w8a8_int8_weight(
                    src_weight,
                    w,
                    [CkptWeightInfo(w_name + self.qw_suffix, identity)],
                    transpose,
                    data_type=torch.int8,
                    config=src_weight.config,
                ),
                create_w8a8_int8_weight(
                    src_weight,
                    s,
                    [CkptWeightInfo(w_name + self.qs_suffix, identity)],
                    identity,
                    data_type=torch.float32,
                    config=src_weight.config,
                ),
                None,
            ]

    @property
    def qw_suffix(self) -> str:
        return ".weight"

    @property
    def qs_suffix(self) -> str:
        return ".per_channel_scale"


class RtpLLMSmoothQuantWeightInfo(SmoothQuantWeightInfo):
    @classmethod
    def support(
        cls, quant_config: QuantizationConfig, src_weight_info: WeightModule
    ) -> bool:
        if not quant_config.is_quanted() or not isinstance(
            quant_config, SmoothQuantConfig
        ):
            return False
        name = src_weight_info.name
        return name in cls.w8a8_weight_list and (
            src_weight_info.weight_style == WeightStyle.RTP_SMOOTH_LLM_STYLE
        )

    def __init__(
        self,
        src_weight_info: AtomicWeight,
        quant_config: QuantizationConfig,
        *args,
        **kwargs,
    ):
        super().__init__(src_weight_info, quant_config, *args, **kwargs)

    @property
    def qw_suffix(self) -> str:
        return INT8QW_COL_SUFFIX

    @property
    def qs_suffix(self) -> str:
        return INT8QS_COL_SUFFIX
