import copy
import functools
from typing import Any, Dict, List, Optional, Union

import torch

from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig, QuantizationConfig
from rtp_llm.model_loader.attn_weight import AttnAtomicWeight, MlaAttnAtomicWeight
from rtp_llm.model_loader.ffn_weight import FfnAtomicWeight, MoeAtomicWeight
from rtp_llm.model_loader.load_config import LoadConfig
from rtp_llm.model_loader.tensor_source import TensorSource
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
    concat_0,
    convert_down_proj_,
    convert_gate_up_proj_,
    identity,
    kv_split,
    merge_block_scale,
    merge_te_qkv,
    mla_pad,
    mla_pad_scale,
    pad,
    pad_w13,
    sp_0,
    sp_0_w13,
    sp_head_gemm_a8,
    sp_head_s_gemm_a8_block,
    sp_neg1,
    stack_,
    stack_moe_w1,
    transpose_slice_k,
    transpose_slice_v,
)
from rtp_llm.utils.util import check_with_info

W_SUFFIX = ".weight"
B_SUFFIX = ".bias"
QW_SUFFIX = ".weight"
QS_SUFFIX = ".weight_scale_inv"
APPEND_SUFFIX = "_scale_inv"


def dequant_weight_split_k(
    ts: List[torch.Tensor],
    block_size: int,
    head_num: int,
    nope_head_dim: int,
    v_head_dim: int,
    lora_rank: int,
) -> torch.Tensor:
    from rtp_llm.models.deepseek_dequant import weight_dequant

    return transpose_slice_k(
        [weight_dequant(ts[0], ts[1], block_size)],
        head_num,
        nope_head_dim,
        v_head_dim,
        lora_rank,
    )


def dequant_weight_split_v(
    ts: List[torch.Tensor],
    block_size: int,
    head_num: int,
    nope_head_dim: int,
    v_head_dim: int,
    lora_rank: int,
) -> torch.Tensor:
    from rtp_llm.models.deepseek_dequant import weight_dequant

    return transpose_slice_v(
        [weight_dequant(ts[0], ts[1], block_size)],
        head_num,
        nope_head_dim,
        v_head_dim,
        lora_rank,
    )


def ceil_div(a, b):
    return (a + b - 1) // b


def cast_to_fp8(x: torch.Tensor):
    return x.to(torch.float8_e4m3fn)


def per_block_cast_to_fp8(
    x: torch.Tensor, group_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    is_2d = x.dim() == 2
    if is_2d:
        x = x.unsqueeze(0)  # (1, m, n)

    b, m, n = x.shape
    m_padded = ceil_div(m, group_size) * group_size
    n_padded = ceil_div(n, group_size) * group_size
    x_padded = torch.zeros(
        (b, m_padded, n_padded), dtype=torch.float32, device=x.device
    )
    x_padded[:, :m, :n] = x
    x_view = x_padded.view(
        b, m_padded // group_size, group_size, n_padded // group_size, group_size
    )
    x_amax = x_view.abs().float().amax(dim=(2, 4), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (FP8_E4M3_MAX / x_amax)).to(torch.float8_e4m3fn)
    x_quantized = x_scaled.view(b, m_padded, n_padded)[:, :m, :n]
    scales = (x_amax / FP8_E4M3_MAX).to(torch.float32)
    squeeze_dims = []

    if scales.size(2) == 1:
        squeeze_dims.append(2)
    if scales.size(4) == 1:
        squeeze_dims.append(4)
    if squeeze_dims:
        scales = scales.squeeze(dim=squeeze_dims)

    if is_2d:
        x_quantized = x_quantized.squeeze(0)
        scales = scales.squeeze(0)

    return x_quantized.contiguous(), scales.contiguous()


def gemm_block_fp8_gpt_style_tp_strategy():
    gemm_block_fp8_weight_tp_strategy: Dict[str, Any] = {
        W.attn_o_w: sp_neg1,
        W.attn_o_s: sp_neg1,
        W.attn_qkv_w: sp_head_gemm_a8,
        W.attn_qkv_s: sp_head_s_gemm_a8_block,
        W.ffn_w1: sp_0,
        W.ffn_s1: sp_0,
        W.ffn_w3: sp_0,
        W.ffn_s3: sp_0,
        W.ffn_w13: sp_0_w13,
        W.ffn_s13: sp_0_w13,
        W.ffn_w2: sp_neg1,
        W.ffn_s2: sp_neg1,
        # mla
        # W.mla_kv_a_w: sp_id,
        W.mla_k_nope_w: sp_0,
        W.mla_k_nope_s: sp_0,
        W.mla_v_w: sp_0,
        W.mla_v_s: sp_0,
        W.mla_q_b_w: sp_0,
        W.mla_q_b_s: sp_0,
    }
    tp_strategy = copy.deepcopy(W.gpt_style_tp_strategy)
    tp_strategy.update(gemm_block_fp8_weight_tp_strategy)
    return tp_strategy


class W8A8Fp8PerBlockAtomicWeight(AtomicWeight):
    gpt_style_tp_strategy = gemm_block_fp8_gpt_style_tp_strategy()

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    def _get_split_func(self):
        return self.gpt_style_tp_strategy[self.name]


class W8A8Fp8PerBlockAttnAtomicWeight(AttnAtomicWeight, W8A8Fp8PerBlockAtomicWeight):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)


class W8A8Fp8PerBlockMlaAttnAtomicWeight(
    MlaAttnAtomicWeight, W8A8Fp8PerBlockAtomicWeight
):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)


class W8A8Fp8PerBlockFfnAtomicWeight(FfnAtomicWeight, W8A8Fp8PerBlockAtomicWeight):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)


class W8A8Fp8PerBlockMoeAtomicWeight(MoeAtomicWeight, W8A8Fp8PerBlockAtomicWeight):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)


def create_w8a8_fp8_per_block_weight(
    src_weight_info: WeightModule, *args: Any, **kwargs: Any
) -> W8A8Fp8PerBlockAtomicWeight:
    if isinstance(src_weight_info, MlaAttnAtomicWeight):
        return W8A8Fp8PerBlockMlaAttnAtomicWeight(*args, **kwargs)
    if isinstance(src_weight_info, AttnAtomicWeight):
        return W8A8Fp8PerBlockAttnAtomicWeight(*args, **kwargs)
    if isinstance(src_weight_info, MoeAtomicWeight):
        return W8A8Fp8PerBlockMoeAtomicWeight(*args, **kwargs)
    if isinstance(src_weight_info, FfnAtomicWeight):
        return W8A8Fp8PerBlockFfnAtomicWeight(*args, **kwargs)
    if isinstance(src_weight_info, AtomicWeight):
        return W8A8Fp8PerBlockAtomicWeight(*args, **kwargs)
    raise NotImplementedError(f"Unsupported weight type: {src_weight_info}")


class PerBlockFp8Weight(CompositeWeight, QuantWeight):
    w8a8_weight_list = {
        W.attn_qkv_w: W.attn_qkv_s,
        W.attn_o_w: W.attn_o_s,
        W.mla_k_nope_w: W.mla_k_nope_s,
        W.mla_v_w: W.mla_v_s,
        W.mla_kc: None,
        W.mla_vc: None,
        W.mla_q_b_w: W.mla_q_b_s,
        W.mla_fusedqkrope_w: W.mla_fusedqkrope_s,
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
            quant_config, Fp8BlockWiseQuantConfig
        ):
            return False
        name = src_weight_info.name
        return name in cls.w8a8_weight_list

    def __init__(
        self,
        src_weight_info: WeightModule,
        quant_config: QuantizationConfig,
        *args: Any,
        **kwargs: Any,
    ):
        kernel: WeightModule = None
        scale: WeightModule = None
        self.group_size = quant_config.group_size()

        if src_weight_info.name == W.attn_qkv_w:
            kernel, scale = self._get_qkv_quant_weight(src_weight_info, self.group_size)
        elif src_weight_info.name == W.attn_o_w:
            if isinstance(src_weight_info, MlaAttnAtomicWeight):
                kernel, scale = self._get_mla_attn_out_quant_weight(
                    src_weight_info, self.group_size
                )
            else:
                kernel, scale = self._get_mha_attn_out_quant_weight(
                    src_weight_info, self.group_size
                )
        elif src_weight_info.name in [W.mla_k_nope_w, W.mla_v_w]:
            kernel, scale = self._get_mla_kv_nope_quant_weight(
                src_weight_info, self.group_size
            )
        elif src_weight_info.name in [W.mla_kc, W.mla_vc]:
            kernel, scale = self._get_mla_kv_c(src_weight_info)
        elif src_weight_info.name == W.mla_q_b_w:
            kernel, scale = self._get_mla_q_quant_weight(src_weight_info)
        elif src_weight_info.name == W.mla_fusedqkrope_w:
            kernel, scale = self._get_fusedqkrope_quant_weight(src_weight_info)
        elif src_weight_info.name in [W.ffn_w1, W.ffn_w2, W.ffn_w3, W.ffn_w13]:
            kernel, scale = self._get_ffn_quant_weight(src_weight_info, self.group_size)
        elif src_weight_info.name == W.moe_w1:
            kernel, scale = self._get_moe_w1_quant_weight(src_weight_info)
        elif src_weight_info.name == W.moe_w2:
            kernel, scale = self._get_moe_w2_quant_weight(src_weight_info)

        sub_weights = {kernel.name: kernel}
        if scale is not None:
            sub_weights.update({scale.name: scale})
        super().__init__(sub_weights, quant_config=quant_config, *args, **kwargs)
        self.kernel = sub_weights.get(kernel.name)
        self.scale = sub_weights.get(scale.name) if scale is not None else None

    def _get_qkv_quant_weight(self, src_weight_info: AttnAtomicWeight, group_size: int):
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
        kernel = create_w8a8_fp8_per_block_weight(
            src_weight_info,
            W.attn_qkv_w,
            qkv_w_list,
            merge_te_qkv,
            data_type=torch.float8_e4m3fn,
            config=src_weight_info.config,
        )

        scale = create_w8a8_fp8_per_block_weight(
            src_weight_info,
            W.attn_qkv_s,
            qkv_s_list,
            merge_block_scale,
            data_type=torch.float32,
            config=src_weight_info.config,
        )
        return [kernel, scale]

    def _get_mha_attn_out_quant_weight(
        self, src_weight_info: AttnAtomicWeight, group_size: int
    ):
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

        kernel = create_w8a8_fp8_per_block_weight(
            src_weight_info,
            W.attn_o_w,
            [CkptWeightInfo(w_name + QW_SUFFIX)],
            data_type=torch.float8_e4m3fn,
            config=src_weight_info.config,
        )
        scale = create_w8a8_fp8_per_block_weight(
            src_weight_info,
            W.attn_o_s,
            [CkptWeightInfo(w_name + QS_SUFFIX)],
            data_type=torch.float32,
            config=src_weight_info.config,
        )
        return [kernel, scale]

    def _get_mla_attn_out_quant_weight(
        self, src_weight_info: MlaAttnAtomicWeight, group_size: int
    ):
        check_with_info(
            src_weight_info.name == W.attn_o_w,
            "src_weight_info.name != W.attn_o_w, actual: {}".format(
                src_weight_info.name
            ),
        )
        check_with_info(
            isinstance(src_weight_info, MlaAttnAtomicWeight),
            "src_weight_info is not MlaAttnAtomicWeight, actual: {}".format(
                src_weight_info
            ),
        )
        w_name = src_weight_info.weights[0].name[: -len(W_SUFFIX)]
        kernel = create_w8a8_fp8_per_block_weight(
            src_weight_info,
            W.attn_o_w,
            [CkptWeightInfo(w_name + QW_SUFFIX, identity)],
            functools.partial(
                mla_pad,
                head_num=src_weight_info.config.head_num,
                nope_head_dim=src_weight_info.nope_head_dim,
                rope_head_dim=0,
            ),
            data_type=torch.float8_e4m3fn,
            config=src_weight_info.config,
        )
        scale = create_w8a8_fp8_per_block_weight(
            src_weight_info,
            W.attn_o_s,
            [CkptWeightInfo(w_name + QS_SUFFIX, identity)],
            functools.partial(
                mla_pad_scale,
                head_num=src_weight_info.config.head_num,
                nope_head_dim=src_weight_info.nope_head_dim,
                rope_head_dim=0,
                group_size=group_size,
            ),
            data_type=torch.float32,
            config=src_weight_info.config,
        )

        return [kernel, scale]

    def _get_mla_kv_nope_quant_weight(
        self, src_weight_info: MlaAttnAtomicWeight, group_size: int
    ):
        is_k = src_weight_info.name == W.mla_k_nope_w
        w_name = src_weight_info.weights[0].name[: -len(W_SUFFIX)]
        if is_k:
            w, s = [W.mla_k_nope_w, W.mla_k_nope_s]
        else:
            w, s = [W.mla_v_w, W.mla_v_s]

        kernel = create_w8a8_fp8_per_block_weight(
            src_weight_info,
            w,
            [CkptWeightInfo(w_name + QW_SUFFIX, identity)],
            functools.partial(
                kv_split,
                kv_lora_rank=src_weight_info.kv_lora_rank,
                nope_head_dim=src_weight_info.nope_head_dim,
                v_head_dim=src_weight_info.v_head_dim,
                idx=0 if is_k else 1,
            ),
            data_type=torch.float8_e4m3fn,
            config=src_weight_info.config,
        )
        scale = create_w8a8_fp8_per_block_weight(
            src_weight_info,
            s,
            [CkptWeightInfo(w_name + QS_SUFFIX, identity)],
            functools.partial(
                kv_split,
                kv_lora_rank=src_weight_info.kv_lora_rank // group_size,
                nope_head_dim=src_weight_info.nope_head_dim // group_size,
                v_head_dim=src_weight_info.v_head_dim // group_size,
                idx=0 if is_k else 1,
            ),
            data_type=torch.float32,
            config=src_weight_info.config,
        )

        return [kernel, scale]

    def _get_mla_kv_c(self, src_weight_info: MlaAttnAtomicWeight):
        is_k = src_weight_info.name == W.mla_kc
        process_func = dequant_weight_split_k if is_k else dequant_weight_split_v
        w_name = src_weight_info.weights[0].name[: -len(W_SUFFIX)]
        w = W.mla_kc if is_k else W.mla_vc

        kernel = create_w8a8_fp8_per_block_weight(
            src_weight_info,
            w,
            [
                CkptWeightInfo(w_name + QW_SUFFIX, identity),
                CkptWeightInfo(w_name + QS_SUFFIX, identity),
            ],
            functools.partial(
                process_func,
                block_size=128,
                head_num=src_weight_info.config.head_num,
                nope_head_dim=src_weight_info.nope_head_dim,
                v_head_dim=src_weight_info.v_head_dim,
                lora_rank=src_weight_info.kv_lora_rank,
            ),
            config=src_weight_info.config,
        )
        return [kernel, None]

    def _get_mla_q_quant_weight(self, src_weight_info: MlaAttnAtomicWeight):
        assert src_weight_info.name == W.mla_q_b_w
        assert src_weight_info.config.q_use_lora
        w_name = src_weight_info.weights[0].name[: -len(W_SUFFIX)]

        kernel = create_w8a8_fp8_per_block_weight(
            src_weight_info,
            W.mla_q_b_w,
            [CkptWeightInfo(w_name + QW_SUFFIX, src_weight_info.weights[0].merge_fun)],
            identity,
            data_type=torch.float8_e4m3fn,
            config=src_weight_info.config,
        )
        scale = create_w8a8_fp8_per_block_weight(
            src_weight_info,
            W.mla_q_b_s,
            [CkptWeightInfo(w_name + QS_SUFFIX, identity)],
            identity,
            data_type=torch.float32,
            config=src_weight_info.config,
        )
        return [kernel, scale]

    def _get_fusedqkrope_quant_weight(self, src_weight_info: MlaAttnAtomicWeight):
        assert src_weight_info.name == W.mla_fusedqkrope_w
        assert src_weight_info.config.q_use_lora
        q_w_name = src_weight_info.weights[0].name[: -len(W_SUFFIX)]
        k_w_name = src_weight_info.weights[1].name[: -len(W_SUFFIX)]
        kernel = create_w8a8_fp8_per_block_weight(
            src_weight_info,
            W.mla_fusedqkrope_w,
            [
                CkptWeightInfo(q_w_name + QW_SUFFIX, identity),
                CkptWeightInfo(
                    k_w_name + QW_SUFFIX, src_weight_info.weights[1].merge_fun
                ),
            ],
            concat_0,
            data_type=torch.float8_e4m3fn,
            config=src_weight_info.config,
        )
        scale = create_w8a8_fp8_per_block_weight(
            src_weight_info,
            W.mla_fusedqkrope_s,
            [
                CkptWeightInfo(q_w_name + QS_SUFFIX, identity),
                CkptWeightInfo(k_w_name + QS_SUFFIX, identity),
            ],
            concat_0,
            data_type=torch.float32,
            config=src_weight_info.config,
        )

        return [kernel, scale]

    def _get_ffn_quant_weight(self, src_weight_info: FfnAtomicWeight, group_size: int):
        assert src_weight_info.name in [W.ffn_w1, W.ffn_w2, W.ffn_w3, W.ffn_w13]
        weights = src_weight_info.weights
        w_name = src_weight_info.weights[0].name[: -len(W_SUFFIX)]
        if src_weight_info.name == W.ffn_w13:
            w, s = (W.ffn_w13, W.ffn_s13)
            w1_name = weights[0].name[: -len(W_SUFFIX)]
            w3_name = weights[1].name[: -len(W_SUFFIX)]

            return [
                create_w8a8_fp8_per_block_weight(
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
                    data_type=torch.float8_e4m3fn,
                    config=src_weight_info.config,
                ),
                create_w8a8_fp8_per_block_weight(
                    src_weight_info,
                    s,
                    [
                        CkptWeightInfo(w1_name + QS_SUFFIX, identity),
                        CkptWeightInfo(w3_name + QS_SUFFIX, identity),
                    ],
                    functools.partial(
                        pad_w13,
                        align_size=src_weight_info.config.align_size
                        // group_size,
                        dim=0,
                    ),
                    data_type=torch.float32,
                    config=src_weight_info.config,
                ),
            ]
        elif src_weight_info.name in [W.ffn_w1, W.ffn_w3]:
            if src_weight_info.name == W.ffn_w1:
                w, s = [W.ffn_w1, W.ffn_s1]
            else:
                w, s = [W.ffn_w3, W.ffn_s3]

            kernel = create_w8a8_fp8_per_block_weight(
                src_weight_info,
                w,
                [CkptWeightInfo(w_name + QW_SUFFIX, identity)],
                functools.partial(
                    pad,
                    align_size=src_weight_info.config.align_size,
                    dim=0,
                ),
                data_type=torch.float8_e4m3fn,
                config=src_weight_info.config,
            )
            scale = create_w8a8_fp8_per_block_weight(
                src_weight_info,
                s,
                [CkptWeightInfo(w_name + QS_SUFFIX, identity)],
                functools.partial(
                    pad,
                    align_size=src_weight_info.config.align_size
                    // group_size,
                    dim=0,
                ),
                data_type=torch.float32,
                config=src_weight_info.config,
            )
            return [kernel, scale]
        else:
            kernel = create_w8a8_fp8_per_block_weight(
                src_weight_info,
                W.ffn_w2,
                [CkptWeightInfo(w_name + QW_SUFFIX, identity)],
                functools.partial(
                    pad,
                    align_size=src_weight_info.config.align_size,
                    dim=1,
                ),
                data_type=torch.float8_e4m3fn,
                config=src_weight_info.config,
            )
            scale = create_w8a8_fp8_per_block_weight(
                src_weight_info,
                W.ffn_s2,
                [CkptWeightInfo(w_name + QS_SUFFIX, identity)],
                functools.partial(
                    pad,
                    align_size=src_weight_info.config.align_size
                    // group_size,
                    dim=1,
                ),
                data_type=torch.float32,
                config=src_weight_info.config,
            )
            return [kernel, scale]

    def _get_moe_w2_quant_weight(self, src_weight_info: MoeAtomicWeight):
        assert src_weight_info.name in [W.moe_w2]
        if not src_weight_info.weights[0].name.endswith(W_SUFFIX):
            w_name = src_weight_info.weights[0].name
            kernel_name = w_name
            scale_name = w_name + APPEND_SUFFIX
            opt1 = convert_down_proj_
            opt2 = identity
        else:
            w_name = src_weight_info.weights[0].name[: -len(W_SUFFIX)]
            kernel_name = w_name + QW_SUFFIX
            scale_name = w_name + QS_SUFFIX
            opt1 = identity
            opt2 = stack_
        kernel = create_w8a8_fp8_per_block_weight(
            src_weight_info,
            W.moe_w2,
            [CkptWeightInfo(kernel_name, opt1)],
            opt2,
            data_type=torch.float8_e4m3fn,
            config=src_weight_info.config,
        )
        scale = create_w8a8_fp8_per_block_weight(
            src_weight_info,
            W.moe_s2,
            [
                CkptWeightInfo(
                    scale_name,
                    opt1,
                )
            ],
            opt2,
            data_type=torch.float32,
            config=src_weight_info.config,
        )
        return [kernel, scale]

    def _get_moe_w1_quant_weight(self, src_weight_info: MoeAtomicWeight):
        assert src_weight_info.name in [W.moe_w1]
        if not src_weight_info.weights[0].name.endswith(W_SUFFIX):
            w_name = src_weight_info.weights[0].name
            kernel_name = w_name
            scale_name = w_name + APPEND_SUFFIX
            opt1 = convert_gate_up_proj_
            opt2 = identity
        else:
            w_name = src_weight_info.weights[0].name[: -len(W_SUFFIX)]
            kernel_name = w_name + QW_SUFFIX
            scale_name = w_name + QS_SUFFIX
            opt1 = identity
            opt2 = stack_moe_w1

        kernel = create_w8a8_fp8_per_block_weight(
            src_weight_info,
            W.moe_w1,
            [CkptWeightInfo(kernel_name, opt1) for w in src_weight_info.weights],
            opt2,
            data_type=torch.float8_e4m3fn,
            config=src_weight_info.config,
        )
        scale = create_w8a8_fp8_per_block_weight(
            src_weight_info,
            W.moe_s1,
            [CkptWeightInfo(scale_name, opt1) for w in src_weight_info.weights],
            opt2,
            data_type=torch.float32,
            config=src_weight_info.config,
        )
        return [kernel, scale]

    def _postprocess(
        self,
        tensor: Union[torch.Tensor, Dict[str, torch.Tensor]],
        device: str,
        load_config: LoadConfig,
    ):
        # need reshape for kernel weight
        processed_res = super()._postprocess(tensor, device, load_config)
        kernel_weight = processed_res[self.kernel.name]
        kernel_weight = (
            kernel_weight.reshape(kernel_weight.shape[-1], -1)
            if kernel_weight.dim() == 2
            else kernel_weight
        )
        processed_res[self.kernel.name] = kernel_weight
        if self.scale is not None:
            scale_weight = processed_res[self.scale.name]
            scale_weight = (
                scale_weight.reshape(scale_weight.shape[-1], -1)
                if scale_weight.dim() == 2
                else scale_weight
            )
            kernel_weight = load_config.exported_device.maybe_rewrite_weight_by_key(
                "weight", kernel_weight
            )
            scale_weight = load_config.exported_device.maybe_rewrite_weight_by_key(
                "scale", scale_weight
            )
            # kernel_weight, scale_weight = load_config.exported_device.convert_fp8_weight_params(kernel_weight, scale_weight)
            processed_res[self.scale.name] = scale_weight
            processed_res[self.kernel.name] = kernel_weight

        return processed_res


class LoadQuantPerBlockFp8Weight(PerBlockFp8Weight):
    @classmethod
    def support(
        cls, quant_config: QuantizationConfig, src_weight_info: WeightModule
    ) -> bool:
        if quant_config.is_quanted() or not isinstance(
            quant_config, Fp8BlockWiseQuantConfig
        ):
            return False
        name = src_weight_info.name
        return name in cls.w8a8_weight_list and name not in [W.mla_kc, W.mla_vc]

    def __init__(
        self,
        src_weight_info: AtomicWeight,
        quant_config: QuantizationConfig,
        *args,
        **kwargs,
    ):
        self.group_size = quant_config.group_size()
        params = src_weight_info.extract_params(
            src_weight_info.__class__, src_weight_info, quant_config
        )
        kernel: AtomicWeight = create_w8a8_fp8_per_block_weight(
            src_weight_info, **params
        )
        sub_weights = {kernel.name: kernel}
        scale_name = self.w8a8_weight_list.get(src_weight_info.name)
        scale = None
        if scale_name:
            scale_params = copy.deepcopy(params)
            scale_params["name"] = scale_name
            scale: AtomicWeight = create_w8a8_fp8_per_block_weight(
                src_weight_info, **scale_params
            )
            sub_weights.update({scale.name: scale})

        CompositeWeight.__init__(
            self, sub_weights, quant_config=quant_config, *args, **kwargs
        )
        self.kernel = kernel
        self.scale = scale

    def _load_raw_tensor(
        self,
        tensor_source: TensorSource,
        layer_id: Optional[int],
        device: str,
        load_config: LoadConfig,
    ):
        kernel = self.kernel._load_raw_tensor(
            tensor_source, layer_id, device, load_config
        )

        res = {}
        scale = None
        if self.scale:
            quant_kernel, scale = per_block_cast_to_fp8(
                kernel.get(self.kernel.name), self.group_size
            )
            if quant_kernel.dim() == 2:
                scale = scale.reshape([scale.shape[0], -1])
        else:
            quant_kernel = cast_to_fp8(kernel.get(self.kernel.name))

        if self.kernel.name == W.moe_w1 or self.kernel.name == W.moe_w2:
            pass
        elif quant_kernel.dim() == 2:
            quant_kernel = quant_kernel.T

        res = {self.kernel.name: quant_kernel.contiguous().to(device)}
        if self.scale:
            scale = scale.T if scale.dim() == 2 else scale
            res.update({self.scale.name: scale.contiguous().to(device)})

        return res

    def get_tensor_names(
        self, layer_id: Optional[int], load_config: LoadConfig
    ) -> set[str]:
        return self.kernel.get_tensor_names(layer_id, load_config)
