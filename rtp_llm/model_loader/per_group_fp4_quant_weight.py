import copy
import functools
from typing import Any, Dict, List, Union

import torch

from rtp_llm.config.quant_config import ModelOptFp4Config, QuantizationConfig
from rtp_llm.model_loader.attn_weight import AttnAtomicWeight
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
    concat_0,
    identity,
    is_v4_weight,
    max_scalar,
    pad,
    pad_w13,
    sp_0,
    sp_0_w13,
    sp_head_s_gemm_a4,
    sp_head_s_gemm_a4_group,
    sp_id,
    sp_neg1,
    stack_,
    stack_moe_w1,
    stack_moe_w1_s2,
)
from rtp_llm.utils.util import check_with_info

B_SUFFIX = ".bias"
ACT_S_SUFFIX = ".input_scale"
W_SUFFIX = ".weight"
QW_SUFFIX = ".weight"
QS_SUFFIX = ".weight_scale"
QS_2_SUFFIX = ".weight_scale_2"


def gemm_group_fp4_gpt_style_tp_strategy():
    gemm_group_fp4_weight_tp_strategy: Dict[str, Any] = {
        W.attn_o_w: sp_neg1,
        W.attn_o_s: sp_neg1,
        W.attn_o_s2: sp_id,
        W.attn_o_i_s: sp_id,
        W.attn_qkv_w: sp_head_s_gemm_a4,
        W.attn_qkv_s: sp_head_s_gemm_a4_group,
        W.attn_qkv_s2: sp_id,
        W.attn_qkv_i_s: sp_id,
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
    tp_strategy.update(gemm_group_fp4_weight_tp_strategy)
    return tp_strategy


class W4A4Fp4PerGroupAtomicWeight(AtomicWeight):
    gpt_style_tp_strategy = gemm_group_fp4_gpt_style_tp_strategy()

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    def _get_split_func(self):
        return self.gpt_style_tp_strategy[self.name]


class W4A4Fp4PerGroupAttnAtomicWeight(AttnAtomicWeight, W4A4Fp4PerGroupAtomicWeight):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)


class W4A4Fp4PerGroupFfnAtomicWeight(FfnAtomicWeight, W4A4Fp4PerGroupAtomicWeight):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)


class W4A4Fp4PerGroupMoeAtomicWeight(MoeAtomicWeight, W4A4Fp4PerGroupAtomicWeight):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)


def create_w4a4_fp4_per_group_weight(
    src_weight_info: WeightModule, *args: Any, **kwargs: Any
) -> W4A4Fp4PerGroupAtomicWeight:
    if isinstance(src_weight_info, AttnAtomicWeight):
        return W4A4Fp4PerGroupAttnAtomicWeight(*args, **kwargs)
    if isinstance(src_weight_info, MoeAtomicWeight):
        return W4A4Fp4PerGroupMoeAtomicWeight(*args, **kwargs)
    if isinstance(src_weight_info, FfnAtomicWeight):
        return W4A4Fp4PerGroupFfnAtomicWeight(*args, **kwargs)
    if isinstance(src_weight_info, AtomicWeight):
        return W4A4Fp4PerGroupAtomicWeight(*args, **kwargs)
    raise NotImplementedError(f"Unsupported weight type: {src_weight_info}")


class PerGroupFp4Weight(CompositeWeight, QuantWeight):
    w4a4_weight_list = {
        W.attn_qkv_w: W.attn_qkv_s,
        W.attn_o_w: W.attn_o_s,
        W.ffn_w1: W.ffn_s1,
        W.ffn_w2: W.ffn_s2,
        W.ffn_w3: W.ffn_s3,
        W.ffn_w13: W.ffn_s13,
        W.moe_w1: W.moe_s1,
        W.moe_w2: W.moe_s2,
    }

    @classmethod
    def support(
        cls, quant_config: QuantizationConfig, src_weight_info: WeightModule
    ) -> bool:
        if not quant_config.is_quanted() or not isinstance(
            quant_config, ModelOptFp4Config
        ):
            return False
        name = src_weight_info.name
        return (
            name in cls.w4a4_weight_list
            and not quant_config.mixed_attention
            and not is_v4_weight(src_weight_info)
        )

    def __init__(
        self,
        src_weight_info: WeightModule,
        quant_config: QuantizationConfig,
        *args: Any,
        **kwargs: Any,
    ):
        kernel: WeightModule = None
        scale: WeightModule = None

        if src_weight_info.name == W.attn_qkv_w:
            kernel, scale, scale_2, input_scale = self._get_qkv_quant_weight(
                src_weight_info
            )
        elif src_weight_info.name == W.attn_o_w:
            kernel, scale, scale_2, input_scale = self._get_mha_attn_out_quant_weight(
                src_weight_info
            )
        elif src_weight_info.name in [W.ffn_w1, W.ffn_w2, W.ffn_w3, W.ffn_w13]:
            kernel, scale, scale_2, input_scale = self._get_ffn_quant_weight(
                src_weight_info
            )
        elif src_weight_info.name == W.moe_w1:
            kernel, scale, scale_2, input_scale = self._get_moe_w1_quant_weight(
                src_weight_info
            )
        elif src_weight_info.name == W.moe_w2:
            kernel, scale, scale_2, input_scale = self._get_moe_w2_quant_weight(
                src_weight_info
            )
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
        self.input_scale = (
            sub_weights.get(input_scale.name) if input_scale is not None else None
        )

    def _get_qkv_quant_weight(self, src_weight_info: AttnAtomicWeight):
        assert src_weight_info.name == W.attn_qkv_w
        weights: List[CkptWeightInfo] = src_weight_info.weights
        assert len(weights) == 1 or len(weights) == 3
        qkv_w_list = [
            CkptWeightInfo(sub_w.name[: -len(W_SUFFIX)] + QW_SUFFIX, sub_w.merge_fun)
            for sub_w in weights
        ]
        qkv_s_list = [
            CkptWeightInfo(sub_w.name[: -len(W_SUFFIX)] + QS_SUFFIX, sub_w.merge_fun)
            for sub_w in weights
        ]

        qkv_s2_list = [
            CkptWeightInfo(sub_w.name[: -len(W_SUFFIX)] + QS_2_SUFFIX, sub_w.merge_fun)
            for sub_w in weights
        ]

        qkv_i_s_list = [
            CkptWeightInfo(sub_w.name[: -len(W_SUFFIX)] + ACT_S_SUFFIX, sub_w.merge_fun)
            for sub_w in weights
        ]
        kernel = create_w4a4_fp4_per_group_weight(
            src_weight_info,
            W.attn_qkv_w,
            qkv_w_list,
            concat_0,
            data_type=torch.uint8,
            config=src_weight_info.config,
        )

        scale = create_w4a4_fp4_per_group_weight(
            src_weight_info,
            W.attn_qkv_s,
            qkv_s_list,
            concat_0,
            data_type=torch.float8_e4m3fn,
            config=src_weight_info.config,
        )

        scale_2 = create_w4a4_fp4_per_group_weight(
            src_weight_info,
            W.attn_qkv_s2,
            qkv_s2_list,
            max_scalar,
            data_type=torch.float32,
            config=src_weight_info.config,
        )

        input_scale = create_w4a4_fp4_per_group_weight(
            src_weight_info,
            W.attn_qkv_i_s,
            qkv_i_s_list,
            max_scalar,
            data_type=torch.float32,
            config=src_weight_info.config,
        )
        return [kernel, scale, scale_2, input_scale]

    def _get_mha_attn_out_quant_weight(self, src_weight_info: AttnAtomicWeight):
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
        w_name = src_weight_info.weights[0].name[: -len(W_SUFFIX)]

        kernel = create_w4a4_fp4_per_group_weight(
            src_weight_info,
            W.attn_o_w,
            [CkptWeightInfo(w_name + QW_SUFFIX)],
            data_type=torch.uint8,
            config=src_weight_info.config,
        )
        scale = create_w4a4_fp4_per_group_weight(
            src_weight_info,
            W.attn_o_s,
            [CkptWeightInfo(w_name + QS_SUFFIX)],
            data_type=torch.float8_e4m3fn,
            config=src_weight_info.config,
        )
        scale_2 = create_w4a4_fp4_per_group_weight(
            src_weight_info,
            W.attn_o_s2,
            [CkptWeightInfo(w_name + QS_2_SUFFIX)],
            data_type=torch.float32,
            config=src_weight_info.config,
        )

        input_scale = create_w4a4_fp4_per_group_weight(
            src_weight_info,
            W.attn_o_i_s,
            [CkptWeightInfo(w_name + ACT_S_SUFFIX)],
            data_type=torch.float32,
            config=src_weight_info.config,
        )

        return [kernel, scale, scale_2, input_scale]

    def _get_ffn_quant_weight(self, src_weight_info: FfnAtomicWeight):
        assert src_weight_info.name in [W.ffn_w1, W.ffn_w2, W.ffn_w3, W.ffn_w13]
        weights = src_weight_info.weights
        w_name = src_weight_info.weights[0].name[: -len(W_SUFFIX)]
        if src_weight_info.name == W.ffn_w13:
            w, s = (W.ffn_w13, W.ffn_s13)
            s_2, i_s = (W.ffn_w13_s2, W.ffn_w13_i_s)
            w1_name = weights[0].name[: -len(W_SUFFIX)]
            w3_name = weights[1].name[: -len(W_SUFFIX)]
            kernel = create_w4a4_fp4_per_group_weight(
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
            scale = create_w4a4_fp4_per_group_weight(
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
            scale_2 = create_w4a4_fp4_per_group_weight(
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
            input_scale = create_w4a4_fp4_per_group_weight(
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
            kernel = create_w4a4_fp4_per_group_weight(
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
            scale = create_w4a4_fp4_per_group_weight(
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
            scale_2 = create_w4a4_fp4_per_group_weight(
                src_weight_info,
                s_2,
                [CkptWeightInfo(w_name + QS_2_SUFFIX, identity)],
                data_type=torch.float32,
                config=src_weight_info.config,
            )
            input_scale = create_w4a4_fp4_per_group_weight(
                src_weight_info,
                i_s,
                [CkptWeightInfo(w_name + ACT_S_SUFFIX, identity)],
                data_type=torch.float32,
                config=src_weight_info.config,
            )
            return [kernel, scale, scale_2, input_scale]
        else:
            kernel = create_w4a4_fp4_per_group_weight(
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
            scale = create_w4a4_fp4_per_group_weight(
                src_weight_info,
                W.ffn_s2,
                [CkptWeightInfo(w_name + QS_SUFFIX, identity)],
                data_type=torch.float8_e4m3fn,
                config=src_weight_info.config,
            )
            scale_2 = create_w4a4_fp4_per_group_weight(
                src_weight_info,
                W.ffn_w2_s2,
                [CkptWeightInfo(w_name + QS_2_SUFFIX, identity)],
                data_type=torch.float32,
                config=src_weight_info.config,
            )
            input_scale = create_w4a4_fp4_per_group_weight(
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
        kernel = create_w4a4_fp4_per_group_weight(
            src_weight_info,
            W.moe_w2,
            [CkptWeightInfo(w_name + QW_SUFFIX, identity)],
            stack_,
            data_type=torch.uint8,
            config=src_weight_info.config,
        )
        scale = create_w4a4_fp4_per_group_weight(
            src_weight_info,
            W.moe_s2,
            [CkptWeightInfo(w_name + QS_SUFFIX, identity)],
            stack_,
            data_type=torch.float8_e4m3fn,
            config=src_weight_info.config,
        )
        scale_2 = create_w4a4_fp4_per_group_weight(
            src_weight_info,
            W.moe_w2_s2,
            [CkptWeightInfo(w_name + QS_2_SUFFIX, identity)],
            stack_,
            data_type=torch.float32,
            config=src_weight_info.config,
        )
        input_scale = create_w4a4_fp4_per_group_weight(
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
        kernel = create_w4a4_fp4_per_group_weight(
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
        scale = create_w4a4_fp4_per_group_weight(
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
        scale_2 = create_w4a4_fp4_per_group_weight(
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
        input_scale = create_w4a4_fp4_per_group_weight(
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
                kernel_weight, scale_weight = (
                    load_config.exported_device.convert_fp4_gemm_weight_params(
                        kernel_weight, scale_weight
                    )
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


# DSv4 specialization (FP4 routed experts: e2m1 packed int8 + UE8M0 group=32 scale)
# ---------------------------------------------------------------------------
#
# V4 ckpt format differs from ModelOpt-style FP4:
#   * scale suffix is ``.scale`` (V4) not ``.weight_scale`` (ModelOpt).
#   * NO ``weight_scale_2`` (per-tensor outer scale) — single-level FP4.
#   * NO ``input_scale``     — activation_scheme='dynamic' in V4 quant config.
#   * scale dtype is float8_e8m0fnu  shape ``[N, K/32]`` (rather than fp8e4m3 wrapped).
#   * weight dtype is int8 packed (2 nibbles per byte) shape ``[N, K/2]``.
#
# DeepGEMM's ``fp8_fp4_gemm_nt`` consumes these natively, so we deliberately
# bypass ``PerGroupFp4Weight._postprocess`` (which calls
# ``convert_fp4_gemm_weight_params`` + ``maybe_prepare_static_weights_for_fp4_moe``
# — those transform layouts for ModelOpt-style FP4 + static activation scaling).

# V4 routed-expert MoE keys → scale keys.
_V4_FP4_WEIGHT_LIST: Dict[str, str] = {
    W.v4_routed_w1_w: W.v4_routed_w1_s,
    W.v4_routed_w2_w: W.v4_routed_w2_s,
    W.v4_routed_w3_w: W.v4_routed_w3_s,
}

PerGroupFp4Weight.w4a4_weight_list.update(_V4_FP4_WEIGHT_LIST)


class V4PerGroupFp4Weight(PerGroupFp4Weight):
    """V4 FP4 (e2m1 packed) per-group routed-expert weight.

    Differences vs. base ``PerGroupFp4Weight``:
      - ckpt scale suffix is ``.scale`` (V4) instead of ``.weight_scale``.
      - no ``weight_scale_2`` (per-tensor outer scale absent).
      - no ``input_scale``     (activation scheme is dynamic in V4).
      - ``_postprocess`` is a no-op (only device move) — V4 ckpt weight + scale
        layouts are directly consumed by DeepGEMM's ``fp8_fp4_gemm_nt``.
      - ``support()`` is restricted to V4 namespace.
    """

    V4_W_SUFFIX = ".weight"
    V4_S_SUFFIX = ".scale"

    @classmethod
    def support(
        cls, quant_config: QuantizationConfig, src_weight_info: WeightModule
    ) -> bool:
        if not quant_config.is_quanted() or not isinstance(
            quant_config, ModelOptFp4Config
        ):
            return False
        if not is_v4_weight(src_weight_info):
            return False
        return src_weight_info.name in _V4_FP4_WEIGHT_LIST

    def __init__(
        self,
        src_weight_info: WeightModule,
        quant_config: QuantizationConfig,
        *args: Any,
        **kwargs: Any,
    ):
        scale_name = _V4_FP4_WEIGHT_LIST[src_weight_info.name]
        kernel, scale = self._v4_build_pair(
            src_weight_info, src_weight_info.name, scale_name
        )

        sub_weights = {kernel.name: kernel, scale.name: scale}
        CompositeWeight.__init__(
            self, sub_weights, quant_config=quant_config, *args, **kwargs
        )
        self.kernel = sub_weights[kernel.name]
        self.scale = sub_weights[scale.name]
        self.scale_2 = None
        self.input_scale = None

    def _v4_build_pair(
        self,
        src_weight_info: WeightModule,
        weight_key: str,
        scale_key: str,
    ):
        """Build kernel + scale pair for V4 routed experts.  ``src_weight_info``
        is a ``MoeAtomicWeight`` with per-expert ckpt key templates; we keep
        its merge_fun and just retarget ckpt names to ``.weight`` / ``.scale``.
        """
        kernel = create_w4a4_fp4_per_group_weight(
            src_weight_info,
            weight_key,
            [
                CkptWeightInfo(
                    w.name[: -len(self.V4_W_SUFFIX)] + self.V4_W_SUFFIX, w.merge_fun
                )
                for w in src_weight_info.weights
            ],
            src_weight_info.process_fun,
            data_type=torch.int8,
            config=getattr(src_weight_info, "config", None),
        )
        scale = create_w4a4_fp4_per_group_weight(
            src_weight_info,
            scale_key,
            [
                CkptWeightInfo(
                    w.name[: -len(self.V4_W_SUFFIX)] + self.V4_S_SUFFIX, w.merge_fun
                )
                for w in src_weight_info.weights
            ],
            src_weight_info.process_fun,
            data_type=torch.float8_e8m0fnu,
            config=getattr(src_weight_info, "config", None),
        )
        return [kernel, scale]

    def _postprocess(
        self,
        tensor: Union[torch.Tensor, Dict[str, torch.Tensor]],
        device: str,
        load_config: LoadConfig,
    ):
        # V4 ckpt weight + UE8M0 scale layout is directly consumed by
        # DeepGEMM's fp8_fp4_gemm_nt; skip ModelOpt-style transforms.
        return CompositeWeight._postprocess(self, tensor, device, load_config)
