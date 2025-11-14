import copy
import functools
from typing import Any, Dict, List, Optional, Union

import torch

from rtp_llm.config.quant_config import (
    Fp8PerChannelCompressedQuantConfig,
    Fp8PerChannelQuarkQuantConfig,
    QuantizationConfig,
)
from rtp_llm.model_loader.attn_weight import AttnAtomicWeight
from rtp_llm.model_loader.ffn_weight import FfnAtomicWeight, MoeAtomicWeight
from rtp_llm.model_loader.load_config import LoadConfig
from rtp_llm.model_loader.weight_module import (
    AtomicWeight,
    CompositeWeight,
    QuantWeight,
    WeightModule,
)
from rtp_llm.utils.database import BaseDatabase
from rtp_llm.utils.model_weight import (
    FP8_E4M3_MAX,
    CkptWeightInfo,
    W,
    concat_0,
    identity,
    pad,
    pad_w13,
    sp_0,
    sp_0_w13,
    sp_head_gemm_a8,
    sp_head_s_gemm_a8_channel,
    sp_id,
    sp_neg1,
    stack_,
    stack_moe_w1,
)
from rtp_llm.utils.util import check_with_info

W_SUFFIX = ".weight"
B_SUFFIX = ".bias"
QW_SUFFIX = ".weight"
QS_SUFFIX = ".weight_scale"

def cast_to_fp8(x: torch.Tensor):
    """Convert tensor to FP8 format."""
    return x.to(torch.float8_e4m3fn)

def per_channel_cast_to_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Per-channel FP8 quantization.
    Args:
        x: Input tensor to be quantized
    Returns:
        tuple[torch.Tensor, torch.Tensor]: Quantized tensor and channel-wise scales
    """
    assert x.dim() in [2, 3], f"weight dim=2 or dim=3 supported, but got shape {x.shape}"
    if x.dim() == 3:
        channel_max = x.abs().amax(dim=-1, keepdim=True).clamp(1e-4)
        scales = (channel_max / FP8_E4M3_MAX).to(torch.float32)
        quantized = (x / scales).to(torch.float8_e4m3fn)
        return quantized.contiguous(), scales.contiguous()
    x = x.T
    # Calculate per-channel maximum absolute values
    channel_max = x.abs().amax(dim=-1, keepdim=True).clamp(1e-4)
    # Compute scaling factors
    scales = (channel_max / FP8_E4M3_MAX).to(torch.float32)
    # Quantize the tensor
    quantized = (x / scales).to(torch.float8_e4m3fn)
    return (quantized.T).contiguous(), (scales.T).contiguous()

def gemm_channel_fp8_gpt_style_tp_strategy():
    gemm_channel_fp8_weight_tp_strategy: Dict[str, Any] = {
        W.attn_o_w: sp_neg1,
        W.attn_o_s: sp_id,
        W.attn_qkv_w: sp_head_gemm_a8,
        W.attn_qkv_s: sp_head_s_gemm_a8_channel,
        W.ffn_w1: sp_0,
        W.ffn_s1: sp_0,
        W.ffn_w3: sp_0,
        W.ffn_s3: sp_0,
        W.ffn_w2: sp_neg1,
        W.ffn_s2: sp_id,
        W.ffn_w13: sp_0_w13,
        W.ffn_s13: sp_0_w13,
    }
    tp_strategy = copy.deepcopy(W.gpt_style_tp_strategy)
    tp_strategy.update(gemm_channel_fp8_weight_tp_strategy)
    return tp_strategy


class W8A8Fp8PerChannelAtomicWeight(AtomicWeight):
    gpt_style_tp_strategy = gemm_channel_fp8_gpt_style_tp_strategy()

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    def _get_split_func(self):
        return self.gpt_style_tp_strategy[self.name]


class W8A8Fp8PerChannelAttnAtomicWeight(
    AttnAtomicWeight, W8A8Fp8PerChannelAtomicWeight
):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)


class W8A8Fp8PerChannelFfnAtomicWeight(FfnAtomicWeight, W8A8Fp8PerChannelAtomicWeight):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)


class W8A8Fp8PerChannelMoeAtomicWeight(MoeAtomicWeight, W8A8Fp8PerChannelAtomicWeight):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)


def create_w8a8_fp8_per_channel_weight(
    src_weight_info: WeightModule, *args: Any, **kwargs: Any
) -> W8A8Fp8PerChannelAtomicWeight:
    if isinstance(src_weight_info, AttnAtomicWeight):
        return W8A8Fp8PerChannelAttnAtomicWeight(*args, **kwargs)
    if isinstance(src_weight_info, MoeAtomicWeight):
        return W8A8Fp8PerChannelMoeAtomicWeight(*args, **kwargs)
    if isinstance(src_weight_info, FfnAtomicWeight):
        return W8A8Fp8PerChannelFfnAtomicWeight(*args, **kwargs)
    if isinstance(src_weight_info, AtomicWeight):
        return W8A8Fp8PerChannelAtomicWeight(*args, **kwargs)
    raise NotImplementedError(f"Unsupported weight type: {src_weight_info}")


class PerChannelFp8Weight(CompositeWeight, QuantWeight):
    w8a8_weight_list = {
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
            quant_config, (Fp8PerChannelCompressedQuantConfig, Fp8PerChannelQuarkQuantConfig)
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

        if src_weight_info.name == W.attn_qkv_w:
            kernel, scale = self._get_qkv_quant_weight(src_weight_info)
        elif src_weight_info.name == W.attn_o_w:
            kernel, scale = self._get_mha_attn_out_quant_weight(src_weight_info)
        elif src_weight_info.name in [W.ffn_w1, W.ffn_w2, W.ffn_w3, W.ffn_w13]:
            kernel, scale = self._get_ffn_quant_weight(src_weight_info)
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
        kernel = create_w8a8_fp8_per_channel_weight(
            src_weight_info,
            W.attn_qkv_w,
            qkv_w_list,
            concat_0,
            data_type=torch.float8_e4m3fn,
            config=src_weight_info.config,
        )

        scale = create_w8a8_fp8_per_channel_weight(
            src_weight_info,
            W.attn_qkv_s,
            qkv_s_list,
            concat_0,
            data_type=torch.float32,
            config=src_weight_info.config,
        )
        return [kernel, scale]

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

        kernel = create_w8a8_fp8_per_channel_weight(
            src_weight_info,
            W.attn_o_w,
            [CkptWeightInfo(w_name + QW_SUFFIX)],
            data_type=torch.float8_e4m3fn,
            config=src_weight_info.config,
        )
        scale = create_w8a8_fp8_per_channel_weight(
            src_weight_info,
            W.attn_o_s,
            [CkptWeightInfo(w_name + QS_SUFFIX)],
            data_type=torch.float32,
            config=src_weight_info.config,
        )
        return [kernel, scale]

    def _get_ffn_quant_weight(self, src_weight_info: FfnAtomicWeight):
        assert src_weight_info.name in [W.ffn_w1, W.ffn_w2, W.ffn_w3, W.ffn_w13]
        weights = src_weight_info.weights
        w_name = src_weight_info.weights[0].name[: -len(W_SUFFIX)]
        if src_weight_info.name == W.ffn_w13:
            w, s = (W.ffn_w13, W.ffn_s13)
            w1_name = weights[0].name[: -len(W_SUFFIX)]
            w3_name = weights[1].name[: -len(W_SUFFIX)]

            return [
                create_w8a8_fp8_per_channel_weight(
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
                create_w8a8_fp8_per_channel_weight(
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
                    data_type=torch.float32,
                    config=src_weight_info.config,
                ),
            ]
        elif src_weight_info.name in [W.ffn_w1, W.ffn_w3]:
            if src_weight_info.name == W.ffn_w1:
                w, s = [W.ffn_w1, W.ffn_s1]
            else:
                w, s = [W.ffn_w3, W.ffn_s3]

            kernel = create_w8a8_fp8_per_channel_weight(
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
            scale = create_w8a8_fp8_per_channel_weight(
                src_weight_info,
                s,
                [CkptWeightInfo(w_name + QS_SUFFIX, identity)],
                functools.partial(
                    pad,
                    align_size=src_weight_info.config.align_size,
                    dim=0,
                ),
                data_type=torch.float32,
                config=src_weight_info.config,
            )
            return [kernel, scale]
        else:
            kernel = create_w8a8_fp8_per_channel_weight(
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
            scale = create_w8a8_fp8_per_channel_weight(
                src_weight_info,
                W.ffn_s2,
                [CkptWeightInfo(w_name + QS_SUFFIX, identity)],
                data_type=torch.float32,
                config=src_weight_info.config,
            )
            return [kernel, scale]

    def _get_moe_w2_quant_weight(self, src_weight_info: MoeAtomicWeight):
        assert src_weight_info.name in [W.moe_w2]
        w_name = src_weight_info.weights[0].name[: -len(W_SUFFIX)]
        kernel = create_w8a8_fp8_per_channel_weight(
            src_weight_info,
            W.moe_w2,
            [CkptWeightInfo(w_name + QW_SUFFIX, identity)],
            stack_,
            data_type=torch.float8_e4m3fn,
            config=src_weight_info.config,
        )
        scale = create_w8a8_fp8_per_channel_weight(
            src_weight_info,
            W.moe_s2,
            [CkptWeightInfo(w_name + QS_SUFFIX, identity)],
            stack_,
            data_type=torch.float32,
            config=src_weight_info.config,
        )
        return [kernel, scale]

    def _get_moe_w1_quant_weight(self, src_weight_info: MoeAtomicWeight):
        assert src_weight_info.name in [W.moe_w1]
        kernel = create_w8a8_fp8_per_channel_weight(
            src_weight_info,
            W.moe_w1,
            [
                CkptWeightInfo(w.name[: -len(W_SUFFIX)] + QW_SUFFIX, identity)
                for w in src_weight_info.weights
            ],
            stack_moe_w1,
            data_type=torch.float8_e4m3fn,
            config=src_weight_info.config,
        )
        scale = create_w8a8_fp8_per_channel_weight(
            src_weight_info,
            W.moe_s1,
            [
                CkptWeightInfo(w.name[: -len(W_SUFFIX)] + QS_SUFFIX, identity)
                for w in src_weight_info.weights
            ],
            stack_moe_w1,
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
            kernel_weight, scale_weight = (
                load_config.exported_device.convert_fp8_weight_params(
                    kernel_weight, scale_weight
                )
            )
            processed_res[self.scale.name] = scale_weight
            processed_res[self.kernel.name] = kernel_weight

        return processed_res

class LoadQuantPerChannelFp8Weight(PerChannelFp8Weight):
    """
    LoadQuantPerChannelFp8Weight class for dynamic per-channel FP8 quantization.

    This class performs per-channel quantization during loading time, similar to
    vLLM's PTPC approach but with per-channel granularity instead of per-token.
    """

    @classmethod
    def support(
        cls, quant_config: QuantizationConfig, src_weight_info: WeightModule
    ) -> bool:
        if quant_config.is_quanted() or not isinstance(
            quant_config, Fp8PerChannelCompressedQuantConfig
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
        # Extract parameters from source weight info
        params = src_weight_info.extract_params(
            src_weight_info.__class__, src_weight_info, quant_config
        )

        # Create kernel weight component
        kernel: AtomicWeight = create_w8a8_fp8_per_channel_weight(
            src_weight_info, **params
        )
        sub_weights = {kernel.name: kernel}

        # Create scale component if needed
        scale_name = self.w8a8_weight_list.get(src_weight_info.name)
        scale = None
        if scale_name:
            scale_params = copy.deepcopy(params)
            scale_params["name"] = scale_name
            scale: AtomicWeight = create_w8a8_fp8_per_channel_weight(
                src_weight_info, **scale_params
            )
            sub_weights.update({scale.name: scale})

        # Initialize composite weight
        CompositeWeight.__init__(
            self, sub_weights, quant_config=quant_config, *args, **kwargs
        )
        self.kernel = kernel
        self.scale = scale

    def _load_raw_tensor(
        self,
        database: BaseDatabase,
        layer_id: Optional[int],
        device: str,
        load_config: LoadConfig,
    ):
        """
        Load raw tensor and perform per-channel quantization.

        This method implements dynamic per-channel quantization similar to vLLM's PTPC
        but with channel-wise granularity instead of token-wise.
        """
        # Load the original weight tensor
        kernel = self.kernel._load_raw_tensor(database, layer_id, device, load_config)

        res = {}
        scale = None

        if self.scale:
            # Perform per-channel quantization
            quant_kernel, scale = per_channel_cast_to_fp8(kernel.get(self.kernel.name))

            # Reshape scale if needed
            if quant_kernel.dim() == 2:
                scale = scale.reshape([scale.shape[0], -1])
        else:
            # Simple cast to FP8 without scaling
            quant_kernel = cast_to_fp8(kernel.get(self.kernel.name))

        # Prepare result dictionary
        if self.kernel.name == W.moe_w1 or self.kernel.name == W.moe_w2:
            pass
        elif quant_kernel.dim() == 2:
            quant_kernel = quant_kernel.T

        res = {self.kernel.name: quant_kernel.contiguous().to(device)}
        if self.scale:
            scale = scale.T if scale.dim() == 2 else scale
            res.update({self.scale.name: scale.contiguous().to(device)})

        return res
