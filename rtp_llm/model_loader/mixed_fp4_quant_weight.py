import functools
import copy
from typing import Any, List, Dict, Union
import torch

from rtp_llm.config.quant_config import ModelOptFp4Config, QuantizationConfig
from rtp_llm.model_loader.attn_weight import AttnAtomicWeight
from rtp_llm.model_loader.load_config import LoadConfig
from rtp_llm.model_loader.ffn_weight import FfnAtomicWeight, MoeAtomicWeight
from rtp_llm.model_loader.linear_attn_weight import (
    LinearAttnAtomicWeight,
    _linear_attn_split_stratey,
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
    pad,
    pad_w13,
    sp_0,
    sp_0_w13,
    sp_id,
    sp_neg1,
    stack_,
    stack_moe_w1,
    max_scalar,
    stack_moe_w1_s2,
    merge_qkv_hf,
    identity,
    merge_qkvz_transpose_reorder,
    merge_ba_transpose_reorder,
    transpose,
    plus_one,
    split_q_gate,
)

from rtp_llm.utils.util import check_with_info


B_SUFFIX = ".bias"
ACT_S_SUFFIX = ".input_scale"
W_SUFFIX = ".weight"
QW_SUFFIX = ".weight"
QS_SUFFIX = ".weight_scale"
QS_2_SUFFIX = ".weight_scale_2"
IN_PROJ_QKV_SUFFIX = ".in_proj_qkv."
IN_PROJ_Z_SUFFIX = ".in_proj_z."
IN_PROJ_A_SUFFIX = ".in_proj_a."
IN_PROJ_B_SUFFIX = ".in_proj_b."
SELF_ATTN_Q_SUFFIX = ".self_attn.q_proj."
SELF_ATTN_K_SUFFIX = ".self_attn.k_proj."
SELF_ATTN_V_SUFFIX = ".self_attn.v_proj."

def mixed_fp4_gpt_style_tp_strategy():
    mixed_fp4_weight_tp_strategy: Dict[str, Any] = {
        W.ffn_w1: sp_0,
        W.ffn_s1: sp_0,
        W.ffn_w1_s2: sp_id,
        W.ffn_w1_i_s: sp_id,
        W.ffn_w3: sp_0,
        W.ffn_s3: sp_0,
        W.ffn_w3_s2: sp_id,
        W.ffn_w3_i_s: sp_id,
        W.ffn_w2: sp_neg1,
        W.ffn_s2: sp_neg1,
        W.ffn_w2_s2: sp_id,
        W.ffn_w2_i_s: sp_id,
        W.ffn_w13: sp_0_w13,
        W.ffn_s13: sp_0_w13,
        W.ffn_w13_s2: sp_id,
        W.ffn_w13_i_s: sp_id,
    }
    tp_strategy = copy.deepcopy(W.gpt_style_tp_strategy)
    tp_strategy.update(mixed_fp4_weight_tp_strategy)
    tp_strategy.update(_linear_attn_split_stratey)
    return tp_strategy

class MixedFp4PerGroupAtomicWeight(AtomicWeight):
    gpt_style_tp_strategy = mixed_fp4_gpt_style_tp_strategy()

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    def _get_split_func(self):
        return self.gpt_style_tp_strategy[self.name]

class MixedFp4PerGroupLinearAttnAtomicWeight(
    LinearAttnAtomicWeight, MixedFp4PerGroupAtomicWeight
):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

class MixedFp4PerGroupAttnAtomicWeight(
    AttnAtomicWeight, MixedFp4PerGroupAtomicWeight
):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)


class MixedFp4PerGroupFfnAtomicWeight(
    FfnAtomicWeight, MixedFp4PerGroupAtomicWeight
):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)


class MixedFp4PerGroupMoeAtomicWeight(
    MoeAtomicWeight, MixedFp4PerGroupAtomicWeight
):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)


def create_mixed_fp4_per_group_weight(
    src_weight_info: WeightModule, *args: Any, **kwargs: Any
) -> MixedFp4PerGroupAtomicWeight:
    if isinstance(src_weight_info, LinearAttnAtomicWeight):
        return MixedFp4PerGroupLinearAttnAtomicWeight(*args, **kwargs)
    if isinstance(src_weight_info, AttnAtomicWeight):
        return MixedFp4PerGroupAttnAtomicWeight(*args, **kwargs)
    if isinstance(src_weight_info, MoeAtomicWeight):
        return MixedFp4PerGroupMoeAtomicWeight(*args, **kwargs)
    if isinstance(src_weight_info, FfnAtomicWeight):
        return MixedFp4PerGroupFfnAtomicWeight(*args, **kwargs)
    if isinstance(src_weight_info, AtomicWeight):
        return MixedFp4PerGroupAtomicWeight(*args, **kwargs)
    raise NotImplementedError(f"Unsupported weight type: {src_weight_info}")


class MixedFp4Weight(CompositeWeight, QuantWeight):
    w4a4_weight_list = {
        W.ffn_w1: W.ffn_s1,
        W.ffn_w2: W.ffn_s2,
        W.ffn_w3: W.ffn_s3,
        W.ffn_w13: W.ffn_s13,
        W.moe_w1: W.moe_s1,
        W.moe_w2: W.moe_s2,
    }

    unquantized_weight_list = [
        W.attn_qkv_w,
        W.attn_o_w,
        W.attn_gate_w,
        W.q_ln_gamma,
        W.k_ln_gamma,
        W.linear_attn_qkvz_w,
        W.linear_attn_ba_w,
        W.linear_attn_alog,
        W.linear_attn_dt_b,
        W.linear_attn_conv1d_w,
        W.linear_attn_out_w,
        W.linear_attn_norm_w,
    ]

    @classmethod
    def support(
        cls, quant_config: QuantizationConfig, src_weight_info: WeightModule
    ) -> bool:
        if not quant_config.is_quanted() or not isinstance(
            quant_config, ModelOptFp4Config
        ):
            return False
        name = src_weight_info.name
        return (name in cls.unquantized_weight_list or name in cls.w4a4_weight_list) \
               and quant_config.mixed_attention
                

    def __init__(
        self,
        src_weight_info: WeightModule,
        quant_config: QuantizationConfig,
        *args: Any,
        **kwargs: Any,
    ):
        kernel: WeightModule = None
        scale: WeightModule = None
        scale_2: WeightModule = None
        input_scale: WeightModule = None
        
        if src_weight_info.name == W.linear_attn_qkvz_w:
            kernel = self._get_linear_attn_qkvz_weight(src_weight_info)
        elif src_weight_info.name == W.linear_attn_ba_w:
            kernel = self._get_linear_attn_ba_weight(src_weight_info)
        elif src_weight_info.name == W.linear_attn_out_w:
            kernel = self._get_linear_attn_out_weight(src_weight_info)
        elif isinstance(src_weight_info, LinearAttnAtomicWeight):
            kernel = self._get_linear_attn_common_weight(src_weight_info)
        elif src_weight_info.name == W.attn_qkv_w:
            kernel = self._get_qkv_weight(src_weight_info)
        elif src_weight_info.name == W.attn_o_w:
            kernel = self._get_mha_attn_out_weight(src_weight_info)
        elif src_weight_info.name == W.attn_gate_w:
            kernel = self._get_mha_attn_gate_weight(src_weight_info)
        elif src_weight_info.name in [W.q_ln_gamma, W.k_ln_gamma]:
            kernel = self._get_norm_weight(src_weight_info)
        elif src_weight_info.name in [W.ffn_w1, W.ffn_w2, W.ffn_w3, W.ffn_w13]:
            kernel, scale, scale_2, input_scale = self._get_ffn_quant_weight(src_weight_info)
        elif src_weight_info.name == W.moe_w1:
            kernel, scale, scale_2, input_scale = self._get_moe_w1_quant_weight(src_weight_info)
        elif src_weight_info.name == W.moe_w2:
            kernel, scale, scale_2, input_scale = self._get_moe_w2_quant_weight(src_weight_info)

        sub_weights = {kernel.name: kernel}
        if scale is not None:
            sub_weights.update({scale.name: scale})
        if scale_2 is not None:
            sub_weights.update({scale_2.name: scale_2})
        if input_scale is not None:
            sub_weights.update({input_scale.name: input_scale})

        super().__init__(sub_weights, quant_config=quant_config, *args, **kwargs)
        self.kernel = sub_weights.get(kernel.name)
        self.scale = sub_weights.get(scale.name) if scale is not None else None
        self.scale_2 = sub_weights.get(scale_2.name) if scale_2 is not None else None
        self.input_scale = sub_weights.get(input_scale.name) if input_scale is not None else None

    def _get_qkv_weight(self, src_weight_info: AttnAtomicWeight):
        assert src_weight_info.name == W.attn_qkv_w
        weights: List[CkptWeightInfo] = src_weight_info.weights
        assert len(weights) == 3
        head_num = src_weight_info.config.head_num
        head_dim = src_weight_info.config.size_per_head
        q_name = next((w.name for w in weights if SELF_ATTN_Q_SUFFIX in w.name), None)
        k_weight = next((w for w in weights if SELF_ATTN_K_SUFFIX in w.name), None)
        v_weight = next((w for w in weights if SELF_ATTN_V_SUFFIX in w.name), None)
        q_weight = CkptWeightInfo(
                        q_name,
                        functools.partial(
                            split_q_gate,
                            head_num=head_num,
                            head_dim=head_dim,
                            part=0,
                        ),
                   )
        qkv_list = [q_weight, k_weight, v_weight]
        
        kernel = create_mixed_fp4_per_group_weight(
            src_weight_info,
            W.attn_qkv_w,
            qkv_list,
            merge_qkv_hf,
            config=src_weight_info.config,
        )

        return kernel

    def _get_mha_attn_gate_weight(self, src_weight_info: AttnAtomicWeight):
        assert src_weight_info.name == W.attn_gate_w
        weights: List[CkptWeightInfo] = src_weight_info.weights
        assert len(weights) == 1
        head_num = src_weight_info.config.head_num
        head_dim = src_weight_info.config.size_per_head
        q_name = weights[0].name
        q_weight = [CkptWeightInfo(
                        q_name,
                        functools.partial(
                            split_q_gate,
                            head_num=head_num,
                            head_dim=head_dim,
                            part=1,
                        ),
                   )]
        
        kernel = create_mixed_fp4_per_group_weight(
            src_weight_info,
            W.attn_gate_w,
            q_weight,
            transpose,
            config=src_weight_info.config,
        )

        return kernel

    def _get_norm_weight(self, src_weight_info: AtomicWeight):
        """
        process q_ln_gamma 和 k_ln_gamma weights data
        """
        assert src_weight_info.name in (W.q_ln_gamma, W.k_ln_gamma)
        weights: List[CkptWeightInfo] = src_weight_info.weights
        assert len(weights) == 1
        
        kernel = create_mixed_fp4_per_group_weight(
            src_weight_info,
            src_weight_info.name,
            weights,
            plus_one,
            config=src_weight_info.config,
        )
        return kernel

    def _get_mha_attn_out_weight(self, src_weight_info: AttnAtomicWeight):
        check_with_info(
            src_weight_info.name == W.attn_o_w,
            "src_weight_info.name != W.attn_o_w, actual: {}".format(
                src_weight_info.name
            ),
        )
        check_with_info(
            isinstance(src_weight_info, AttnAtomicWeight),
            "src_weight_info is not AttnAtomicWeight, actual: {}".format(
                src_weight_info
            ),
        )

        kernel = create_mixed_fp4_per_group_weight(
            src_weight_info,
            W.attn_o_w,
            src_weight_info.weights,
            transpose,
            config=src_weight_info.config,
        )

        return kernel

    def _get_linear_attn_qkvz_weight(self, src_weight_info: LinearAttnAtomicWeight):
        assert src_weight_info.name == W.linear_attn_qkvz_w
        weights: List[CkptWeightInfo] = src_weight_info.weights
        assert len(weights) == 2
        qkv_weight = next((w for w in weights if IN_PROJ_QKV_SUFFIX in w.name), None)
        z_weight = next((w for w in weights if IN_PROJ_Z_SUFFIX in w.name), None)

        if qkv_weight is None or z_weight is None:
            raise ValueError("Missing required weights: in_proj_qkv or in_proj_z")

        merge_weights = [qkv_weight, z_weight]
            
        kernel = create_mixed_fp4_per_group_weight(
            src_weight_info,
            src_weight_info.name,
            merge_weights,
            merge_qkvz_transpose_reorder,
            config=src_weight_info.config,
        )
        return kernel

    def _get_linear_attn_ba_weight(self, src_weight_info: LinearAttnAtomicWeight):
        assert src_weight_info.name == W.linear_attn_ba_w
        weights: List[CkptWeightInfo] = src_weight_info.weights
        assert len(weights) == 2
        a_weight = next((w for w in weights if IN_PROJ_A_SUFFIX in w.name), None)
        b_weight = next((w for w in weights if IN_PROJ_B_SUFFIX in w.name), None)

        if a_weight is None or b_weight is None:
            raise ValueError("Missing required weights: in_proj_a or in_proj_b")

        merge_weights = [b_weight, a_weight]
            
        kernel = create_mixed_fp4_per_group_weight(
            src_weight_info,
            src_weight_info.name,
            merge_weights,
            merge_ba_transpose_reorder,
            config=src_weight_info.config,
        )
        return kernel

    def _get_linear_attn_out_weight(self, src_weight_info: LinearAttnAtomicWeight):
        assert src_weight_info.name == W.linear_attn_out_w
        weights: List[CkptWeightInfo] = src_weight_info.weights
        assert len(weights) == 1
            
        kernel = create_mixed_fp4_per_group_weight(
            src_weight_info,
            src_weight_info.name,
            weights,
            transpose,
            config=src_weight_info.config,
        )
        return kernel
    
    def _get_linear_attn_common_weight(self, src_weight_info: LinearAttnAtomicWeight):
        assert src_weight_info.name in self.unquantized_weight_list
        weights: List[CkptWeightInfo] = src_weight_info.weights
        assert len(weights) == 1
        kernel = create_mixed_fp4_per_group_weight(
            src_weight_info,
            src_weight_info.name,
            weights,
            identity,
            config=src_weight_info.config,
        )
        return kernel

    def _get_ffn_quant_weight(self, src_weight_info: FfnAtomicWeight):
        assert src_weight_info.name in [W.ffn_w1, W.ffn_w2, W.ffn_w3, W.ffn_w13]
        weights = src_weight_info.weights
        w_name = src_weight_info.weights[0].name[: -len(W_SUFFIX)]
        if src_weight_info.name == W.ffn_w13:
            w, s = (W.ffn_w13, W.ffn_s13)
            s_2, i_s = (W.ffn_w13_s2, W.ffn_w13_i_s)
            w1_name = weights[0].name[: -len(W_SUFFIX)]
            w3_name = weights[1].name[: -len(W_SUFFIX)]
            kernel = create_mixed_fp4_per_group_weight(
                        src_weight_info,
                        w,
                        [
                            CkptWeightInfo(w1_name + QW_SUFFIX, identity),
                            CkptWeightInfo(w3_name + QW_SUFFIX, identity),
                        ],
                        functools.partial(
                            pad_w13,
                            align_size=src_weight_info.config.align_size,
                            dim=0,
                        ),
                        data_type=torch.uint8,
                        config=src_weight_info.config,
            )
            scale = create_mixed_fp4_per_group_weight(
                        src_weight_info,
                        s,
                        [
                            CkptWeightInfo(w1_name + QS_SUFFIX, identity),
                            CkptWeightInfo(w3_name + QS_SUFFIX, identity),
                        ],
                        functools.partial(
                            pad_w13,
                            align_size=src_weight_info.config.align_size,
                            dim=0,
                        ),
                        data_type=torch.float8_e4m3fn,
                        config=src_weight_info.config,
            )
            scale_2 = create_mixed_fp4_per_group_weight(
                        src_weight_info,
                        s_2,
                        [
                            CkptWeightInfo(w1_name + QS_2_SUFFIX, identity),
                            CkptWeightInfo(w3_name + QS_2_SUFFIX, identity),
                        ],
                        max_scalar,
                        data_type=torch.float32,
                        config=src_weight_info.config,
            )
            input_scale = create_mixed_fp4_per_group_weight(
                        src_weight_info,
                        i_s,
                        [
                            CkptWeightInfo(w1_name + ACT_S_SUFFIX, identity),
                            CkptWeightInfo(w3_name + ACT_S_SUFFIX, identity),
                        ],
                        max_scalar,
                        data_type=torch.float32,
                        config=src_weight_info.config,
            )
            return [kernel, scale, scale_2, input_scale]
        elif src_weight_info.name in [W.ffn_w1, W.ffn_w3]:
            if src_weight_info.name == W.ffn_w1:
                w, s = [W.ffn_w1, W.ffn_s1]
                s_2, i_s = [W.ffn_w1_s2, W.ffn_w1_i_s]
            else:
                w, s = [W.ffn_w3, W.ffn_s3]
                s_2, i_s = [W.ffn_w3_s2, W.ffn_w3_i_s]
            kernel = create_mixed_fp4_per_group_weight(
                src_weight_info,
                w,
                [CkptWeightInfo(w_name + QW_SUFFIX, identity)],
                functools.partial(
                    pad,
                    align_size=src_weight_info.config.align_size,
                    dim=0,
                ),
                data_type=torch.uint8,
                config=src_weight_info.config,
            )
            scale = create_mixed_fp4_per_group_weight(
                src_weight_info,
                s,
                [CkptWeightInfo(w_name + QS_SUFFIX, identity)],
                functools.partial(
                    pad,
                    align_size=src_weight_info.config.align_size,
                    dim=0,
                ),
                data_type=torch.float8_e4m3fn,
                config=src_weight_info.config,
            )
            scale_2 = create_mixed_fp4_per_group_weight(
                src_weight_info,
                s_2,
                [CkptWeightInfo(w_name + QS_2_SUFFIX, identity)],
                data_type=torch.float32,
                config=src_weight_info.config,
            )
            input_scale = create_mixed_fp4_per_group_weight(
                src_weight_info,
                i_s,
                [CkptWeightInfo(w_name + ACT_S_SUFFIX, identity)],
                data_type=torch.float32,
                config=src_weight_info.config,
            )
            return [kernel, scale, scale_2, input_scale]
        else:
            kernel = create_mixed_fp4_per_group_weight(
                src_weight_info,
                W.ffn_w2,
                [CkptWeightInfo(w_name + QW_SUFFIX, identity)],
                functools.partial(
                    pad,
                    align_size=src_weight_info.config.align_size,
                    dim=1,
                ),
                data_type=torch.uint8,
                config=src_weight_info.config,
            )
            scale = create_mixed_fp4_per_group_weight(
                src_weight_info,
                W.ffn_s2,
                [CkptWeightInfo(w_name + QS_SUFFIX, identity)],
                data_type=torch.float8_e4m3fn,
                config=src_weight_info.config,
            )
            scale_2 = create_mixed_fp4_per_group_weight(
                src_weight_info,
                W.ffn_w2_s2,
                [CkptWeightInfo(w_name + QS_2_SUFFIX, identity)],
                data_type=torch.float32,
                config=src_weight_info.config,
            )
            input_scale = create_mixed_fp4_per_group_weight(
                src_weight_info,
                W.ffn_w2_i_s,
                [CkptWeightInfo(w_name + ACT_S_SUFFIX, identity)],
                data_type=torch.float32,
                config=src_weight_info.config,
            )
            return [kernel, scale, scale_2, input_scale]

    def _get_moe_w2_quant_weight(self, src_weight_info: MoeAtomicWeight):
        assert src_weight_info.name in [W.moe_w2]
        w_name = src_weight_info.weights[0].name[: -len(W_SUFFIX)]
        kernel = create_mixed_fp4_per_group_weight(
            src_weight_info,
            W.moe_w2,
            [CkptWeightInfo(w_name + QW_SUFFIX, identity)],
            stack_,
            data_type=torch.uint8,
            config=src_weight_info.config,
        )
        scale = create_mixed_fp4_per_group_weight(
            src_weight_info,
            W.moe_s2,
            [CkptWeightInfo(w_name + QS_SUFFIX, identity)],
            stack_,
            data_type=torch.float8_e4m3fn,
            config=src_weight_info.config,
        )
        scale_2 = create_mixed_fp4_per_group_weight(
            src_weight_info,
            W.moe_w2_s2,
            [CkptWeightInfo(w_name + QS_2_SUFFIX, identity)],
            stack_,
            data_type=torch.float32,
            config=src_weight_info.config,
        )
        input_scale = create_mixed_fp4_per_group_weight(
            src_weight_info,
            W.moe_w2_i_s,
            [CkptWeightInfo(w_name + ACT_S_SUFFIX, identity)],
            stack_,
            data_type=torch.float32,
            config=src_weight_info.config,
        )
        return [kernel, scale, scale_2, input_scale]

    def _get_moe_w1_quant_weight(self, src_weight_info: MoeAtomicWeight):
        assert src_weight_info.name in [W.moe_w1]
        kernel = create_mixed_fp4_per_group_weight(
            src_weight_info,
            W.moe_w1,
            [
                CkptWeightInfo(w.name[: -len(W_SUFFIX)] + QW_SUFFIX, identity)
                for w in src_weight_info.weights
            ],
            stack_moe_w1,
            data_type=torch.uint8,
            config=src_weight_info.config,
        )
        scale = create_mixed_fp4_per_group_weight(
            src_weight_info,
            W.moe_s1,
            [
                CkptWeightInfo(w.name[: -len(W_SUFFIX)] + QS_SUFFIX, identity)
                for w in src_weight_info.weights
            ],
            stack_moe_w1,
            data_type=torch.float8_e4m3fn,
            config=src_weight_info.config,
        )
        scale_2 = create_mixed_fp4_per_group_weight(
            src_weight_info,
            W.moe_w1_s2,
            [
                CkptWeightInfo(w.name[: -len(W_SUFFIX)] + QS_2_SUFFIX, identity)
                for w in src_weight_info.weights
            ],
            stack_moe_w1_s2,
            data_type=torch.float32,
            config=src_weight_info.config,
        )
        input_scale = create_mixed_fp4_per_group_weight(
            src_weight_info,
            W.moe_w1_i_s,
            [
                CkptWeightInfo(w.name[: -len(W_SUFFIX)] + ACT_S_SUFFIX, identity)
                for w in src_weight_info.weights
            ],
            stack_moe_w1_s2,
            data_type=torch.float32,
            config=src_weight_info.config,
        )
        return [kernel, scale, scale_2, input_scale]

    def _postprocess(
        self,
        tensor: Union[torch.Tensor, Dict[str, torch.Tensor]],
        device: str,
        load_config: LoadConfig,
    ):
        processed_res = super()._postprocess(tensor, device, load_config)
        kernel_weight = processed_res[self.kernel.name]
        if self.scale is not None:
            scale_weight = processed_res[self.scale.name]
            if kernel_weight.dim() == 2 and scale_weight.dim() == 2:
                kernel_weight, scale_weight = load_config.exported_device.convert_fp4_gemm_weight_params(
                    kernel_weight, scale_weight
                )

            kernel_weight, scale_weight = (
                load_config.exported_device.maybe_prepare_static_weights_for_fp4_moe(
                    self.kernel.name,
                    self.scale.name,
                    kernel_weight,
                    scale_weight,
                )
            )
            processed_res[self.kernel.name] = kernel_weight
            processed_res[self.scale.name] = scale_weight

        return processed_res
