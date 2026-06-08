import copy
from typing import Optional

import torch

from rtp_llm.config.quant_config import (
    W4a8Int4PerChannelQuantConfig,
    QuantizationConfig,
)
from rtp_llm.model_loader.load_config import LoadConfig
from rtp_llm.model_loader.tensor_source import StackSplitTensorSource, TensorSource
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
    from rtp_kernel.w4a8_group_gemm import unified_encode_int4b, reorder_tensor, pack_scale_fp8

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

    output_unified_int4 = unified_encode_int4b(output_int4_c)
    output_unified_int4 = reorder_tensor(output_unified_int4)
    scale_packed = pack_scale_fp8(scale_c)

    return output_unified_int4, scale_packed


class LoadW4a8Int4PerChannelQuantWeight(CompositeWeight, QuantWeight):
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
        # **fp8_attn_weights_map,
        # **fp8_ffn_weights_maps,
        **fp8_partial_moe_weights_maps,
    }
    int4_weight_list = [
        # W.attn_qkv_w,
        # W.attn_o_w,
        # W.ffn_w1,
        # W.ffn_w3,
        # W.ffn_w2,
        # W.ffn_w13,
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
        # W4A8 online quantization needs the intermediate MoE weight materialized
        # before packing. Avoid the direct-copy preallocation path, which can
        # crash in native extensions during multi-rank startup.
        setattr(kernel, "disable_gpu_preallocate", True)
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
        if self.kernel.name in [W.moe_w1, W.moe_w2]:
            target_device = (
                device if isinstance(device, torch.device) else torch.device(device)
            )
            if torch.cuda.is_available() and target_device.type == "cuda":
                return self._load_moe_int4_streaming(
                    tensor_source, layer_id, device, load_config
                )

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

    def _load_moe_expert_tensor(
        self,
        ckpt_weight,
        layer_id: Optional[int],
        expert_id: int,
        tensor_source: TensorSource,
        convert_type: torch.dtype,
    ) -> torch.Tensor:
        name = ckpt_weight.name.format(
            i=str(layer_id),
            i_1=str(layer_id + 1),
            expert_id=str(expert_id),
        )
        return ckpt_weight.merge_fun(
            tensor_source.load_tensor(name, convert_type)
        )

    def _load_moe_int4_streaming(
        self,
        tensor_source: TensorSource,
        layer_id: Optional[int],
        device: str,
        load_config: LoadConfig,
    ):
        """Load and quantize one expert at a time on CPU.

        GLM-5 W4A8 online quantization previously moved every per-expert tensor to
        CUDA before packing, which can crash in CUDA/native extensions during
        multi-rank startup. Keep the streaming work on CPU and move the final
        packed tensors to CUDA once.
        """
        if self.kernel.stacked_ckpt_keys and tensor_source.has_tensor(
            self.kernel.weights[0].tensor_name(layer_id)
        ):
            tensor_source = StackSplitTensorSource(
                tensor_source,
                self.kernel._build_split_config(layer_id, load_config),
            )

        ckpt_weights = (
            self.kernel._get_expert_weights()
            if self.kernel.stacked_ckpt_keys
            else self.kernel.weights
        )
        selected_experts = load_config.get_selected_experts(
            layer_id, self.kernel.config.expert_num
        )
        convert_type = (
            self.kernel.data_type
            if self.kernel.data_type is not None
            else load_config.compute_dtype
        )

        quant_kernel = None
        scale = None

        for local_idx, expert_id in enumerate(selected_experts):
            expert_parts = [
                self._load_moe_expert_tensor(
                    ckpt_weight,
                    layer_id,
                    expert_id,
                    tensor_source,
                    convert_type,
                )
                for ckpt_weight in ckpt_weights
            ]
            expert_tensor = self.kernel.process_fun(expert_parts).squeeze(0)
            expert_quant_kernel, expert_scale = quantize_weight_to_int4b(
                expert_tensor, self.group_size
            )

            if quant_kernel is None:
                expert_num = len(selected_experts)
                quant_kernel = torch.empty(
                    (expert_num,) + tuple(expert_quant_kernel.shape),
                    device=expert_quant_kernel.device,
                    dtype=expert_quant_kernel.dtype,
                )
                scale = torch.empty(
                    (expert_num,) + tuple(expert_scale.shape),
                    device=expert_scale.device,
                    dtype=expert_scale.dtype,
                )

            quant_kernel[local_idx].copy_(expert_quant_kernel)
            scale[local_idx].copy_(expert_scale)

        assert quant_kernel is not None
        assert scale is not None
        return {
            self.kernel.name: quant_kernel.contiguous().to(device),
            self.scale.name: scale.contiguous().to(device),
        }
