import copy
from typing import Optional

import torch

import rtp_llm.ops.compute_ops as compute_ops

from rtp_llm.config.quant_config import (
    W4a8Int4PerChannelQuantConfig,
    QuantizationConfig,
)
from rtp_llm.model_loader.load_config import LoadConfig
from rtp_llm.model_loader.tensor_source import TensorSource
from rtp_llm.model_loader.w8a8_weight import create_w8a8_fp8_weight
from rtp_llm.model_loader.weight_module import (
    AtomicWeight,
    CompositeWeight,
    QuantWeight,
    WeightModule,
)
from rtp_llm.model_loader.dynamic_fp8_quant_weight import quantize_weight_to_fp8
from rtp_llm.utils.model_weight import W, WeightStyle


def quantize_weight_to_int4b(input: torch.Tensor, group_size: int, eps: float = 1e-12):
    N, K = input.shape
    assert (K % group_size == 0), f"invalid params {K} or {group_size}"

    n_groups = K // group_size
    input_g = input.view(N, n_groups, group_size)

    amax = input_g.abs().amax(dim=2, keepdim=True)
    finfo = torch.finfo(torch.float8_e4m3fn)
    scale = (amax / 7.).clamp(min=eps, max=finfo.max / 8.)
    scale_f = scale.to(torch.float8_e4m3fn).to(input.dtype)

    output_int8 = torch.round(input_g / scale).clamp_(min=-8, max=7).to(torch.int8)
    output_int8_flat = output_int8.flatten()

    first_int8 = output_int8_flat[::2]
    second_int8 = output_int8_flat[1::2]

    first_int4 = first_int8 & 0x0F
    second_int4 = second_int8 & 0x0F

    packed_int4 = (second_int4 << 4) | first_int4
    output_int4_c = packed_int4.reshape(N, K // 2).contiguous()

    scale_c = scale_f.to(torch.float8_e4m3fn).squeeze(-1).t().contiguous()

    output_unified_int4 = compute_ops.unified_encode_int4b(output_int4_c)
    scale_packed = compute_ops.pack_scale_fp8(scale_c)

    return output_unified_int4, scale_packed


class LoadQuantW4a8PerChannelInt4Weight(CompositeWeight, QuantWeight):
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
        W.mla_k_nope_w: (W.mla_k_nope_s, None, None),
        W.mla_v_w: (W.mla_v_s, None, None),
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
    int4_weight_list = [
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
            quant_config, W4a8Int4PerChannelQuantConfig
        ):
            return False
        name = src_weight_info.name
        return name in cls.int4_weight_list and (
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
        self.group_size = quant_config.group_size()

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
            E = kernel_tensor.shape[0]
            N = kernel_tensor.shape[1]
            K = kernel_tensor.shape[2]

            quant_kernel = torch.empty((E, N, K // 2), device=kernel_tensor.device, dtype=torch.int8)
            scale = torch.empty((E, K // self.group_size, N, 8), device=kernel_tensor.device, dtype=torch.float8_e4m3fn)

            for i in range(E):
                quant_kernel[i, :, :], scale[i] = quantize_weight_to_int4b(kernel_tensor[i, :, :], self.group_size)

        else:
            quant_kernel, scale = quantize_weight_to_fp8(kernel.get(self.kernel.name))
            quant_kernel = quant_kernel.T

        res = {
            self.kernel.name: quant_kernel.contiguous().to(device),
            self.scale.name: scale.contiguous().to(device),
        }
        return res
