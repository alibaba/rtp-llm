"""Explicit SM90 routed W4A8 candidate over installed Humming.

Packed MXFP4/g32 weights are repacked for FP8 WGMMA, not borrowed from W4A16.
RTP's existing activation g128 quantizer is unchanged. Optional K32 promotion
repeats each original g128 scale four times, without requantizing payloads.
No activation decode to BF16 occurs before GEMM. Native FP8 WGMMA has reduced internal accumulation
precision on Hopper: this adapter does not assert the historical numerical
gate has passed. Source-level WGMMA selection still requires a SASS audit.
"""

import dataclasses

import torch
from torch import nn

from ._sm90_humming_linear import unpack_activation_scales


def validate_lossless_a8_scales(scales: torch.Tensor) -> None:
    """Reject Humming's lossy clipping before its fused normalization runs."""
    if scales.dtype != torch.float8_e8m0fnu or scales.ndim != 3 or not scales.numel():
        raise ValueError("Expected nonempty [E,N,K/32] UE8M0 scales")
    raw = scales.contiguous().view(torch.uint8).flatten(1).to(torch.int16)
    # Keep the established finite model exponent range contract. Byte zero has
    # special UE8M0 semantics that fused normalization does not implement.
    if bool(((raw < 1) | (raw > 252)).any()):
        raise ValueError("W4A8 requires UE8M0 exponent bytes in [1,252]")
    if bool(((raw.amax(1) - raw.amin(1)) > 11).any()):
        raise ValueError("W4A8 requires per-expert exponent span <=11; refusing lossy clipping")


class SM90HummingGroupedA8Linear(nn.Module):
    """Grouped-contiguous FP8 activation GEMM; eager route metadata is external.

    Expert offsets must be a valid exclusive prefix sum of E counts ending at
    M. Metadata values are caller-owned, just as for the existing grouped
    adapter; they are not copied to the host in forward or graph capture.
    """

    def __init__(self, weight: torch.Tensor, scales: torch.Tensor, *, promotion_group_size=128):
        super().__init__()
        if promotion_group_size not in (32, 128):
            raise ValueError("W4A8 promotion_group_size must be 32 or 128; activation quantization stays g128")
        self.promotion_group_size = promotion_group_size
        if not weight.is_cuda or torch.cuda.get_device_capability(weight.device) != (9, 0):
            raise ValueError("W4A8 candidate requires CUDA SM90")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Construct W4A8 candidate outside CUDA graph capture")
        if weight.ndim != 3 or weight.dtype != torch.int8:
            raise ValueError("Expected packed MXFP4 int8 [E,N,K/2] weights")
        experts, n, half_k = weight.shape
        k = 2 * half_k
        if experts <= 0 or n <= 0 or k <= 0 or n % 64 or k % 128:
            raise ValueError("W4A8 requires E>0, N%64=0 and K%128=0")
        if scales.device != weight.device or tuple(scales.shape) != (experts, n, k // 32):
            raise ValueError("Expected same-device MXFP4 scales matching weights")
        validate_lossless_a8_scales(scales)
        from humming.config import MmaType
        from humming.schema import HummingInputSchema, HummingWeightSchema
        from humming.transform import prepare_layer_config, transform_humming_tensors

        self.in_features, self.out_features = k, n
        self.config = prepare_layer_config(
            shape_n=n, shape_k=k, num_experts=experts,
            weight_schema=HummingWeightSchema(
                b_dtype="float4e2m1", bs_dtype="float8e8m0",
                weight_scale_group_size=32, weight_scale_type="group",
            ),
            input_schema=HummingInputSchema(
                a_dtype="float8e4m3", input_scale_dtype="float32",
                input_scale_group_size=promotion_group_size,
            ),
            torch_dtype=torch.bfloat16, device=weight.device,
        )
        self.config = dataclasses.replace(self.config, mma_type=MmaType.WGMMA)
        if not self.config.use_fused_e8m0_scale or self.config.use_int_weight_scale:
            raise RuntimeError("Expected installed Humming's fused-E8M0 FP8 WGMMA contract")
        tensors = transform_humming_tensors(self.config, {
            "weight": weight.contiguous().clone().view(torch.int32),
            "weight_scale": scales.contiguous().clone(),
        })
        for name, value in tensors.items():
            self.register_buffer(name, value)
        self.register_buffer("locks", torch.zeros(1024, dtype=torch.int32, device=weight.device))
        self.compute_config = dict(
            gemm_type="grouped_contiguous", use_f16_accum=False,
            use_batch_invariant=True,
        )
        self.arithmetic_contract = (
            "MXFP4_STORAGE_FP8_WGMMA_A_QUANT_G128; "
            f"PROMOTION_METADATA_G{promotion_group_size}; INTERNAL_ACCUM_NOT_FULL_FP32"
        )

    def _promotion_scales(self, packed):
        """Lossless metadata expansion only; packed input is always RTP g128."""
        scale = unpack_activation_scales(packed, self.in_features // 128)
        if self.promotion_group_size == 32:
            scale = scale.repeat_interleave(4, dim=-1).contiguous()
        return scale

    def quantize_input(self, x: torch.Tensor):
        """Original RTP quantizer and lossless row-major FP32 scale conversion."""
        if (
            x.dtype == torch.bfloat16
            and x.ndim == 2
            and x.shape[0] <= 256
            and x.stride(-1) == 1
        ):
            from rtp_llm.models_py.modules.dsv41._sm90_moe_quant_scale_fused import quantize_moe_g128

            # G128 absmax/eps and payload are unchanged. G32 promotion only
            # repeats the stored scale four times, now in the producer.
            return quantize_moe_g128(x, promotion_group_size=self.promotion_group_size)
        from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8

        q, packed = sgl_per_token_group_quant_fp8(
            x.to(torch.bfloat16).contiguous(), group_size=128, eps=1e-4,
            column_major_scales=True, scale_tma_aligned=True, scale_ue8m0=True,
        )
        return q, self._promotion_scales(packed)

    def prepared(self, q, input_scale, expert_offsets, outputs=None):
        from humming.forward import humming_forward

        if (q.ndim != 2 or q.dtype != torch.float8_e4m3fn
                or q.shape[1] != self.in_features or q.device != self.weight.device
                or not q.is_contiguous()):
            raise ValueError("Expected contiguous same-device E4M3 [M,K] activations")
        rows = q.shape[0]
        if (input_scale.dtype != torch.float32 or input_scale.device != q.device
                or tuple(input_scale.shape) != (rows, self.in_features // self.promotion_group_size)
                or not input_scale.is_contiguous()):
            raise ValueError(
                f"Expected contiguous same-device FP32 [M,K/{self.promotion_group_size}] promotion scales"
            )
        if (expert_offsets.dtype != torch.int32 or expert_offsets.device != q.device
                or tuple(expert_offsets.shape) != (self.config.num_experts + 1,)
                or not expert_offsets.is_contiguous()):
            raise ValueError("Expected contiguous same-device int32 expert offsets")
        if outputs is not None:
            if (outputs.dtype != torch.bfloat16 or outputs.device != q.device
                    or tuple(outputs.shape) != (rows, self.out_features)
                    or not outputs.is_contiguous()):
                raise ValueError("Expected contiguous same-device BF16 [M,N] output")
            operands = (q, input_scale, expert_offsets, self.weight, self.weight_scale,
                        self.weight_scale_2, self.locks)
            if any(torch._C._overlaps(outputs, value) for value in operands):
                raise ValueError("Output must not alias GEMM operands")
        if not rows:
            return outputs if outputs is not None else torch.empty(
                (0, self.out_features), device=q.device, dtype=torch.bfloat16)
        return humming_forward(
            self.config, inputs=q, input_scale=input_scale, outputs=outputs,
            weight=self.weight, weight_scale=self.weight_scale,
            weight_scale_2=self.weight_scale_2, expert_layout=expert_offsets,
            valid_shape_m=rows, locks=self.locks, compute_config=self.compute_config,
        )

    def forward_quantized(self, q, packed_scales, expert_offsets, outputs=None):
        if (packed_scales.dtype != torch.int32 or packed_scales.device != q.device
                or tuple(packed_scales.shape) != (q.shape[0], (self.in_features // 128 + 3) // 4)):
            raise ValueError("Expected original packed RTP g128 activation scales")
        if outputs is not None and torch._C._overlaps(outputs, packed_scales):
            raise ValueError("Output must not alias packed input scales")
        return self.prepared(q, self._promotion_scales(packed_scales), expert_offsets, outputs)

    def forward(self, x: torch.Tensor, expert_offsets: torch.Tensor):
        if x.ndim != 2 or x.shape[1] != self.in_features or x.device != self.weight.device:
            raise ValueError("Expected same-device [M,K] input")
        if not x.shape[0]:
            return torch.empty((0, self.out_features), device=x.device, dtype=torch.bfloat16)
        return self.prepared(*self.quantize_input(x), expert_offsets)
