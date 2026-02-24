import copy
import logging
from typing import Dict, Optional, Union

import torch

import rtp_llm.ops.compute_ops as compute_ops
from rtp_llm.config.quant_config import (
    Fp8DynamicPerTensorQuantConfig,
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
from rtp_llm.models_py.utils.arch import is_cuda
from rtp_llm.utils.model_weight import W, WeightStyle


def to_float32_tensor(x, device) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def quantize_weight_to_fp8(ts: torch.Tensor, output: Optional[torch.Tensor] = None):
    if output is None:
        output = torch.empty_like(ts, device=ts.device, dtype=torch.float8_e4m3fn)
    else:
        assert output.dtype == torch.float8_e4m3fn
        assert output.device == ts.device

    if is_cuda() and ts.device.type != "cpu":
        scale = torch.zeros(1, device=ts.device, dtype=torch.float32)
        compute_ops.per_tensor_quant_fp8(ts, output, scale, False)
        return output, scale
    else:
        device = ts.device
        finfo = torch.finfo(torch.float8_e4m3fn)
        max_abs_value = to_float32_tensor(ts.abs().max(), device)
        scaling_factor = max_abs_value / finfo.max
        min_scaling_factor = to_float32_tensor(1.0, device) / (
            to_float32_tensor(512.0, device) * finfo.max
        )
        scaling_factor = torch.max(min_scaling_factor, scaling_factor)
        output = (
            (to_float32_tensor(ts, device) / scaling_factor)
            .clamp(min=finfo.min, max=finfo.max)
            .to(torch.float8_e4m3fn)
        )
        return output.contiguous(), scaling_factor.to(torch.float32)


class LoadQuantDynamicPerTensorFp8Weight(CompositeWeight, QuantWeight):
    fp8_attn_weights_map = {
        W.attn_qkv_w: (
            W.attn_qkv_s,
            None,
            None,
        ),
        W.attn_o_w: (
            W.attn_o_s,
            None,
            None,
        ),
        W.mla_fusedqkrope_w: (W.mla_fusedqkrope_s, None, None),
        W.mla_fusedqkrope_no_lora_w: (W.mla_fusedqkrope_no_lora_s, None, None),
        W.mla_q_b_w: (W.mla_q_b_s, None, None),
        W.mla_kv_b_w: (W.mla_kv_b_s, None, None),
    }

    fp8_ffn_weights_maps = {
        W.ffn_w1: (W.ffn_s1, None, None),
        W.ffn_w3: (W.ffn_s3, None, None),
        W.ffn_w2: (
            W.ffn_s2,
            None,
            None,
        ),
        W.ffn_w13: (
            W.ffn_s13,
            None,
            None,
        ),
    }

    fp8_partial_moe_weights_maps = {
        W.moe_w1: (W.moe_s1, None, None),
        W.moe_w2: (W.moe_s2, None, None),
    }

    weight_scale_map = {
        **fp8_attn_weights_map,
        **fp8_ffn_weights_maps,
        **fp8_partial_moe_weights_maps,
    }
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

    @classmethod
    def support(
        cls, quant_config: QuantizationConfig, src_weight_info: WeightModule
    ) -> bool:

        if quant_config.is_quanted() or not isinstance(
            quant_config, Fp8DynamicPerTensorQuantConfig
        ):
            return False
        name = src_weight_info.name
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
        params = src_weight_info.extract_params(
            src_weight_info.__class__, src_weight_info, quant_config
        )
        kernel: AtomicWeight = create_w8a8_fp8_weight(src_weight_info, **params)
        sub_weights = {kernel.name: kernel}

        scale_name, _, _ = self.weight_scale_map.get(src_weight_info.name)
        scale_params = copy.deepcopy(params)
        scale_params["name"] = scale_name
        scale: AtomicWeight = create_w8a8_fp8_weight(src_weight_info, **scale_params)
        sub_weights.update({scale.name: scale})
        super().__init__(sub_weights, quant_config=quant_config, *args, **kwargs)
        self.kernel = kernel
        self.scale = scale

        self.act_scale = None
        self.act_scale_inv = None

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
        if self.kernel.name in [W.moe_w1, W.moe_w2]:
            # per expert quant moe w13 and w2 to fp8
            kernel_tensor = kernel[self.kernel.name]
            assert len(kernel_tensor.shape) == 3
            num_experts = kernel_tensor.shape[0]

            quant_kernel = torch.empty_like(
                kernel_tensor, device=kernel_tensor.device, dtype=torch.float8_e4m3fn
            )
            scale = torch.ones(
                [num_experts], device=kernel_tensor.device, dtype=torch.float32
            )

            for i in range(num_experts):
                quant_kernel[i, :, :], scale[i] = quantize_weight_to_fp8(
                    kernel_tensor[i, :, :], quant_kernel[i, :, :]
                )

        else:
            quant_kernel, scale = quantize_weight_to_fp8(kernel.get(self.kernel.name))
            quant_kernel = quant_kernel.T

        res = {}
        res = {
            self.kernel.name: quant_kernel.contiguous().to(device),
            self.scale.name: scale.contiguous().to(device),
        }
        return res
