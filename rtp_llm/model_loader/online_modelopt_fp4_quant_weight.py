"""Online MODELOPT_FP4 quantization for MoE weights.

This module implements on-the-fly NVFP4 (modelopt_fp4) quantization of BF16/FP16
MoE weights. Non-MoE weights are kept in their original compute dtype because no
QuantWeight class registers support for them in this mode (``is_quanted()`` is
False).

The math follows ModelOptimizer's NVFP4 reference (block_size=16):
    weight_scale_2 = amax(W) / (FLOAT4_E2M1_MAX * FLOAT8_E4M3_MAX)
    per_block_scale = per_block_amax / (FLOAT4_E2M1_MAX * weight_scale_2)
    q4 = round(W / (per_block_scale.fp8 * weight_scale_2))   -> packed e2m1 uint8

input_scale defaults to 1.0 because no calibration data is available; the FP4
executors fall back to dynamic per-block activation quantization.
"""

from typing import Any, Optional

import torch

from rtp_llm.config.quant_config import ModelOptFp4Config, QuantizationConfig
from rtp_llm.model_loader.ffn_weight import MoeAtomicWeight
from rtp_llm.model_loader.load_config import LoadConfig
from rtp_llm.model_loader.tensor_source import TensorSource
from rtp_llm.model_loader.weight_module import (
    CompositeWeight,
    QuantWeight,
    WeightModule,
)
from rtp_llm.utils.model_weight import W, identity

FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = 448.0
NVFP4_BLOCK_SIZE = 16

_E2M1_BOUNDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
_E2M1_ODD_BOUNDS = _E2M1_BOUNDS[[1, 3, 5]]  # banker-rounding ties


def _cast_to_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Map float values in [-6, 6] to packed E2M1 ordinals (uint8 in [0, 15])."""
    device = x.device
    sign_bit = (x < 0).to(torch.uint8)
    abs_x = x.abs()
    bounds = _E2M1_BOUNDS.to(device=device, dtype=abs_x.dtype)
    ord_ = torch.searchsorted(bounds, abs_x, out_int32=True).to(torch.uint8)
    odd_bounds = _E2M1_ODD_BOUNDS.to(device=device, dtype=abs_x.dtype)
    equals_odd = torch.any(abs_x.unsqueeze(-1) == odd_bounds, dim=-1).to(torch.uint8)
    return (sign_bit << 3) + ord_ + equals_odd


def quantize_expert_to_nvfp4(
    weight: torch.Tensor,
    block_size: int = NVFP4_BLOCK_SIZE,
):
    """Quantize a 2D BF16/FP16 weight tensor to NVFP4 (per-block fp8 + per-tensor fp32).

    Args:
        weight: shape ``[N, K]`` with K divisible by ``block_size``.

    Returns:
        Tuple of (packed uint8 ``[N, K // 2]``, per-block fp8 scale ``[N, K // block_size]``,
        per-tensor fp32 scale_2 scalar).
    """
    assert weight.dim() == 2, f"expected 2D, got {tuple(weight.shape)}"
    N, K = weight.shape
    assert K % block_size == 0, f"K={K} not divisible by block_size={block_size}"

    w_fp32 = weight.to(torch.float32)
    amax = w_fp32.abs().amax().clamp_min(1e-8)
    weight_scale_2 = (amax / (FLOAT4_E2M1_MAX * FLOAT8_E4M3_MAX)).to(torch.float32)

    blocked = w_fp32.view(N, K // block_size, block_size)
    per_block_amax = blocked.abs().amax(dim=-1)  # [N, K//block_size]
    per_block_scale = per_block_amax / (FLOAT4_E2M1_MAX * weight_scale_2)
    per_block_scale = torch.where(
        per_block_scale == 0,
        torch.ones_like(per_block_scale),
        per_block_scale,
    )
    per_block_scale_fp8 = per_block_scale.to(torch.float8_e4m3fn)

    # Re-dequant fp8 scale to use the same value the kernel will see.
    effective_block_scale = per_block_scale_fp8.to(torch.float32)
    combined = (effective_block_scale * weight_scale_2).unsqueeze(-1)  # [N, K//bs, 1]
    scaled = blocked / combined
    scaled = scaled.clamp(-FLOAT4_E2M1_MAX, FLOAT4_E2M1_MAX).reshape(N, K)

    q = _cast_to_e2m1(scaled)  # uint8 [N, K]
    packed = (q[..., 1::2].to(torch.uint8) << 4) | q[..., 0::2].to(torch.uint8)
    packed = packed.contiguous()  # [N, K // 2]
    return packed, per_block_scale_fp8.contiguous(), weight_scale_2


def _quantize_moe_weight(weight: torch.Tensor, block_size: int = NVFP4_BLOCK_SIZE):
    """Quantize a stacked MoE weight ``[E, N, K]`` to NVFP4."""
    assert weight.dim() == 3, f"expected 3D, got {tuple(weight.shape)}"
    E, N, K = weight.shape
    device = weight.device
    packed = torch.empty([E, N, K // 2], dtype=torch.uint8, device=device)
    block_scale = torch.empty(
        [E, N, K // block_size], dtype=torch.float8_e4m3fn, device=device
    )
    scale_2 = torch.empty([E], dtype=torch.float32, device=device)
    for i in range(E):
        p, s, s2 = quantize_expert_to_nvfp4(weight[i], block_size)
        packed[i].copy_(p)
        block_scale[i].copy_(s)
        scale_2[i] = s2
    return packed, block_scale, scale_2


def _moe_kernel_name(name: str) -> str:
    if name == W.moe_w1:
        return W.moe_w1
    if name == W.moe_w2:
        return W.moe_w2
    raise ValueError(f"unsupported moe weight: {name}")


def _moe_scale_name(name: str) -> str:
    if name == W.moe_w1:
        return W.moe_s1
    if name == W.moe_w2:
        return W.moe_s2
    raise ValueError(f"unsupported moe weight: {name}")


def _moe_scale_2_name(name: str) -> str:
    if name == W.moe_w1:
        return W.moe_w1_s2
    if name == W.moe_w2:
        return W.moe_w2_s2
    raise ValueError(f"unsupported moe weight: {name}")


def _moe_input_scale_name(name: str) -> str:
    if name == W.moe_w1:
        return W.moe_w1_i_s
    if name == W.moe_w2:
        return W.moe_w2_i_s
    raise ValueError(f"unsupported moe weight: {name}")


class OnlineModelOptFp4MoeWeight(CompositeWeight, QuantWeight):
    """Online NVFP4 weight loader for MoE ``moe_w1``/``moe_w2`` only."""

    moe_weight_list = [W.moe_w1, W.moe_w2]

    @classmethod
    def support(
        cls, quant_config: QuantizationConfig, src_weight_info: WeightModule
    ) -> bool:
        if not isinstance(quant_config, ModelOptFp4Config):
            return False
        if quant_config.is_quanted():
            return False
        if not isinstance(src_weight_info, MoeAtomicWeight):
            return False
        return src_weight_info.name in cls.moe_weight_list

    def __init__(
        self,
        src_weight_info: MoeAtomicWeight,
        quant_config: QuantizationConfig,
        *args: Any,
        **kwargs: Any,
    ):
        # The kernel sub-weight inherits the original loader so BF16 weights are
        # read from the checkpoint via the existing MoE path (stack_moe_w1 etc.).
        kernel = MoeAtomicWeight(
            name=_moe_kernel_name(src_weight_info.name),
            weights=src_weight_info.weights,
            process_fun=src_weight_info.process_fun,
            data_type=None,  # use load_config.compute_dtype (bf16/fp16)
            config=src_weight_info.config,
            stacked_ckpt_keys=getattr(src_weight_info, "stacked_ckpt_keys", False),
        )
        # Companion sub-weights are synthesized in _load_raw_tensor; their
        # ``weights=[]`` keeps them out of the file-name listing path.
        scale = MoeAtomicWeight(
            name=_moe_scale_name(src_weight_info.name),
            weights=[],
            process_fun=identity,
            data_type=torch.float8_e4m3fn,
            config=src_weight_info.config,
        )
        scale_2 = MoeAtomicWeight(
            name=_moe_scale_2_name(src_weight_info.name),
            weights=[],
            process_fun=identity,
            data_type=torch.float32,
            config=src_weight_info.config,
        )
        input_scale = MoeAtomicWeight(
            name=_moe_input_scale_name(src_weight_info.name),
            weights=[],
            process_fun=identity,
            data_type=torch.float32,
            config=src_weight_info.config,
        )
        sub_weights = {
            kernel.name: kernel,
            scale.name: scale,
            scale_2.name: scale_2,
            input_scale.name: input_scale,
        }
        super().__init__(
            sub_weights,
            quant_config=quant_config,
            name=src_weight_info.name,
            **{k: v for k, v in kwargs.items() if k != "name"},
        )
        self.kernel = kernel
        self.scale = scale
        self.scale_2 = scale_2
        self.input_scale = input_scale
        self._block_size = max(quant_config.group_size(), NVFP4_BLOCK_SIZE)

    def get_tensor_names(
        self, layer_id: Optional[int], load_config: LoadConfig
    ) -> set[str]:
        # Only the kernel sub-weight reads tensors from the checkpoint.
        return self.kernel.get_tensor_names(layer_id, load_config)

    def _load_raw_tensor(
        self,
        tensor_source: TensorSource,
        layer_id: Optional[int],
        device: str,
        load_config: LoadConfig,
    ):
        kernel_dict = self.kernel._load_raw_tensor(
            tensor_source, layer_id, device, load_config
        )
        weight = kernel_dict[self.kernel.name]
        # Quantize on the device the tensor already lives on (typically GPU).
        packed, block_scale, scale_2 = _quantize_moe_weight(
            weight, block_size=self._block_size
        )
        num_experts = weight.shape[0]
        input_scale = torch.ones(
            [num_experts], dtype=torch.float32, device=weight.device
        )
        return {
            self.kernel.name: packed,
            self.scale.name: block_scale,
            self.scale_2.name: scale_2,
            self.input_scale.name: input_scale,
        }

    def _postprocess(
        self,
        tensor,
        device: str,
        load_config: LoadConfig,
    ):
        processed = super()._postprocess(tensor, device, load_config)
        kernel_w = processed[self.kernel.name]
        scale_w = processed[self.scale.name]
        kernel_w, scale_w = (
            load_config.exported_device.maybe_prepare_static_weights_for_fp4_moe(
                self.kernel.name,
                self.scale.name,
                kernel_w,
                scale_w,
            )
        )
        processed[self.kernel.name] = kernel_w
        processed[self.scale.name] = scale_w
        return processed
