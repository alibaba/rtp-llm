"""Hopper grouped MXFP4 GEMM with the V4 routed activation quantizer.

The checkpoint remains packed E2M1 with group-32 E8M0 scales. Activations
retain RTP's group-128 FP8 quantization, then are decoded exactly to BF16 for
W4A16 tensor-core arithmetic. This explicitly avoids Hopper FP8 accumulation
error without substituting unquantized BF16 inputs or expanding resident weights.
"""

import torch
from torch import nn

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


if triton is not None:

    @triton.jit(do_not_specialize=["M"])
    def _decode_g128_fp8_to_bf16(
        X,
        S,
        Y,
        M,
        K: tl.constexpr,
        X_ROW: tl.constexpr,
        X_COL: tl.constexpr,
        S_ROW: tl.constexpr,
        S_COL: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        rows, cols = offsets // K, offsets % K
        valid = offsets < M * K
        values = tl.load(X + rows * X_ROW + cols * X_COL, valid, other=0.0).to(
            tl.float32
        )
        packed = tl.load(S + rows * S_ROW + (cols // 512) * S_COL, valid, other=0).to(
            tl.uint32
        )
        exponent = (packed >> (((cols // 128) % 4) * 8)) & 255
        # Match torch.float8_e8m0fnu.float(), including its two special bytes.
        # In particular byte zero is 2**-127, NOT zero, and 255 is NaN.
        bits = tl.where(exponent == 0, 0x00400000, exponent << 23)
        bits = tl.where(exponent == 255, 0x7FC00000, bits).to(tl.uint32)
        scales = bits.to(tl.float32, bitcast=True)
        tl.store(Y + offsets, values * scales, valid)


def unpack_activation_scales(scale: torch.Tensor, num_groups: int) -> torch.Tensor:
    """Losslessly decode RTP's packed UE8M0 into FP32 power-of-two scales."""
    if scale.ndim != 2 or scale.dtype != torch.int32:
        raise ValueError("RTP activation scales must be packed int32 [M, groups/4]")
    # Drop the TMA group padding, not valid exponent bytes. The typed conversion
    # preserves byte zero (2**-127) and byte 255 (NaN), unlike bit reinterpretation.
    # [M, 1] TMA scales can be logically contiguous while the singleton last
    # stride is not one. Canonicalize that metadata before changing dtype.
    packed = scale.contiguous()
    packed = packed.as_strided(packed.shape, (packed.shape[1], 1))
    exponents = packed.view(torch.float8_e8m0fnu)
    return exponents[:, :num_groups].float().contiguous()


def decode_quantized_activation(
    payload: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """Decode already quantized g128 values; never use the original input.

    E4M3 times a power of two fits BF16 exactly for the model's finite activation
    range (the RTP quantizer's 1e-4 amax floor also prevents tiny exponents).
    """
    if (
        triton is not None
        and payload.is_cuda
        and payload.ndim == 2
        and payload.dtype == torch.float8_e4m3fn
        and payload.shape[1] > 0
        and payload.shape[1] % 128 == 0
        and scale.device == payload.device
        and scale.ndim == 2
        and scale.dtype == torch.int32
        and scale.shape[0] == payload.shape[0]
        and scale.shape[1] * 4 >= payload.shape[1] // 128
    ):
        m, k = payload.shape
        decoded = torch.empty((m, k), dtype=torch.bfloat16, device=payload.device)
        if m:
            _decode_g128_fp8_to_bf16[(triton.cdiv(m * k, 1024),)](
                payload,
                scale,
                decoded,
                m,
                k,
                payload.stride(0),
                payload.stride(1),
                scale.stride(0),
                scale.stride(1),
                1024,
                enable_fp_fusion=False,
            )
        return decoded
    # Keep the CPU / non-E4M3 path and its validation behavior unchanged.
    scales = unpack_activation_scales(scale, payload.shape[-1] // 128)
    blocks = payload.float().reshape(payload.shape[0], -1, 128)
    return (blocks * scales.unsqueeze(-1)).reshape(payload.shape).bfloat16()


def validate_lossless_humming_scales(scales: torch.Tensor) -> None:
    if scales.dtype != torch.float8_e8m0fnu or scales.ndim != 3:
        raise ValueError("SM90 routed scales must be [E, N, K/32] E8M0")
    exponents = scales.view(torch.uint8).flatten(1).to(torch.int16)
    # W4A16 does not use Humming's fused-E8M0 normalization, so the W4A8
    # exponent-span <= 11 restriction does not apply. Bound absolute exponents
    # instead: all E2M1 values times these scales fit BF16. Byte zero's special
    # E8M0 subnormal conversion is not a validated Humming contract here.
    if bool(((exponents < 1) | (exponents > 252)).any()):
        raise ValueError(
            "SM90 Humming W4A16 requires E8M0 exponent bytes in [1, 252]; "
            "refusing unsupported or overflowing BF16 weight conversion"
        )


class SM90HummingGroupedLinear(nn.Module):
    """Two-dimensional routed rows, grouped by contiguous expert offsets."""

    def __init__(self, weight: torch.Tensor, scales: torch.Tensor):
        super().__init__()
        from humming.schema import HummingInputSchema, HummingWeightSchema
        from humming.transform import prepare_layer_config, transform_humming_tensors

        if weight.ndim != 3 or weight.dtype != torch.int8:
            raise ValueError("SM90 routed weight must be [E, N, K/2] packed int8")
        experts, n, packed_k = weight.shape
        k = packed_k * 2
        if n % 64 or k % 128:
            raise ValueError("SM90 Humming requires N % 64 == 0 and K % 128 == 0")
        if tuple(scales.shape) != (experts, n, k // 32):
            raise ValueError("MXFP4 scale shape does not match routed weight")
        validate_lossless_humming_scales(scales)
        self.in_features, self.out_features = k, n
        self.config = prepare_layer_config(
            shape_n=n,
            shape_k=k,
            num_experts=experts,
            weight_schema=HummingWeightSchema(
                b_dtype="float4e2m1",
                bs_dtype="float8e8m0",
                weight_scale_group_size=32,
                weight_scale_type="group",
            ),
            input_schema=HummingInputSchema(
                a_dtype="bfloat16",
            ),
            torch_dtype=torch.bfloat16,
            device=weight.device,
        )
        if self.config.use_fused_e8m0_scale or self.config.use_int_weight_scale:
            raise RuntimeError("W4A16 must not use lossy fused weight-scale conversion")
        # Humming may modify its input during preprocessing. Own this copy so
        # neither the checkpoint tensors nor another module's views can change.
        tensors = transform_humming_tensors(
            self.config,
            {
                "weight": weight.contiguous().clone().view(torch.int32),
                "weight_scale": scales.contiguous().clone(),
            },
        )
        for name, tensor in tensors.items():
            self.register_buffer(name, tensor)
        self.register_buffer(
            "locks", torch.zeros(1024, dtype=torch.int32, device=weight.device)
        )
        self.compute_config = {
            "gemm_type": "grouped_contiguous",
            "use_f16_accum": False,
            # Disable Humming's BF16 Stream-K partial reductions and select a
            # stable K traversal before the final BF16 conversion.
            "use_batch_invariant": True,
        }

    def forward(self, x: torch.Tensor, expert_offsets: torch.Tensor) -> torch.Tensor:
        from humming.forward import humming_forward

        from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
            sgl_per_token_group_quant_fp8,
        )

        if not x.shape[0]:
            return x.new_empty((0, self.out_features))
        # Match QuantizedLinear._fp4_forward_deepgemm, including its amax floor.
        q, scale = sgl_per_token_group_quant_fp8(
            x.to(torch.bfloat16).contiguous(),
            group_size=128,
            eps=1e-4,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=True,
        )
        # Only the arithmetic dtype changes; amax/rounding/scale grouping above
        # remain identical to RTP. The decoded BF16 activation costs 2*M*K
        # bytes; the fused decoder reads packed scales directly, with no FP32
        # payload/product temporaries and no scale-layout conversion.
        return humming_forward(
            self.config,
            inputs=decode_quantized_activation(q, scale),
            weight=self.weight,
            weight_scale=self.weight_scale,
            weight_scale_2=getattr(self, "weight_scale_2", None),
            expert_layout=expert_offsets,
            valid_shape_m=x.shape[0],
            locks=self.locks,
            compute_config=self.compute_config,
        )
