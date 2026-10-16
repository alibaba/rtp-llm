"""Hopper adapter for the native V4.1 MXFP8 dense contract.

Uses the same Humming MXFP8 backend as vLLM (upstream revision
de8291e24f1312da0239605e33a89fe9cc901e47). Weight conversion is a layout
change, not requantization. RTP remains the owner of activation quantization.
Humming is imported lazily: Blackwell does not require this backend.
"""

import json

import torch


def expand_block32_scales(scales: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Expand checkpoint blocks to per-row g32 without converting E8M0 values."""
    if scales.dtype != torch.float8_e8m0fnu:
        raise TypeError("V4.1 MXFP8 weight scales must be UE8M0")
    expected = ((rows + 31) // 32, (cols + 31) // 32)
    if tuple(scales.shape) != expected:
        raise ValueError(f"Expected block32 scales {expected}, got {scales.shape}")
    # Float8 index/repeat kernels are not implemented by all supported PyTorch
    # versions. Moving the bytes also preserves every checkpoint exponent.
    return (
        scales.view(torch.uint8)
        .repeat_interleave(32, dim=0)[:rows]
        .contiguous()
        .view(torch.float8_e8m0fnu)
    )


def unpack_ue8m0_input_scales(scales: torch.Tensor) -> torch.Tensor:
    """Losslessly decode RTP packed scales for Humming's SM90 FP32-scale ABI.

    The C++ launcher accepts packed int32 only for MXMMA, not Hopper WGMMA.
    Casting the UE8M0 bytes also preserves byte 0 (2**-127) and byte 255 (NaN),
    unlike an unchecked exponent-bit shift. Packed tail groups cover padded K.
    """
    if scales.dtype != torch.int32 or scales.ndim != 2:
        raise ValueError("Expected a 2-D packed int32 UE8M0 scale tensor")
    packed = scales.contiguous()
    # PyTorch ignores singleton strides when deciding contiguity. A TMA
    # [M,1] scale view can therefore retain stride (1,align(M,4)), whereas
    # dtype reinterpretation requires its last stride to be exactly one.
    # The storage is contiguous here; canonicalizing its metadata is safe and
    # avoids another copy, including the [1,1] and empty-row cases.
    packed = packed.as_strided(packed.shape, (packed.shape[1], 1))
    return packed.view(torch.float8_e8m0fnu).float()


class Sm90MXFP8Linear:
    """Prepared Humming W8A8/g32 linear with externally quantized activations."""

    def __init__(self, weight: torch.Tensor, scales: torch.Tensor):
        if not weight.is_cuda or torch.cuda.get_device_capability(weight.device) != (
            9,
            0,
        ):
            raise ValueError("Sm90MXFP8Linear requires a CUDA SM90 device")
        if weight.ndim != 2 or weight.dtype != torch.float8_e4m3fn:
            raise TypeError("V4.1 MXFP8 weights must be a 2-D E4M3 tensor")
        if scales.device != weight.device:
            raise ValueError("V4.1 weight and scales must be on the same device")
        self.N, self.K = weight.shape
        if self.K % 32:
            raise ValueError("V4.1 MXFP8 input dimension must be divisible by 32")
        try:
            from humming.forward import humming_forward
            from humming.schema import HummingInputSchema, HummingWeightSchema
            from humming.transform import (
                prepare_layer_config,
                transform_humming_tensors,
            )
        except ImportError as exc:
            raise RuntimeError(
                "V4.1 on SM90 requires humming-kernels==0.1.16; install the "
                "CUDA13 x86 Bazel requirements. No BF16 fallback is enabled."
            ) from exc

        self._forward = humming_forward
        self.config = prepare_layer_config(
            shape_n=self.N,
            shape_k=self.K,
            weight_schema=HummingWeightSchema(
                b_dtype="float8e4m3",
                bs_dtype="float8e8m0",
                weight_scale_group_size=32,
                weight_scale_type="group",
            ),
            input_schema=HummingInputSchema(
                a_dtype="float8e4m3",
                input_scale_dtype="float32",
                input_scale_group_size=32,
                input_quant_mode="dynamic_group",
            ),
            pad_n_to_multiple=256,
            pad_k_to_multiple=128,
            torch_dtype=torch.bfloat16,
            device=weight.device,
        )
        if (
            self.config.mma_type.value != "wgmma"
            or str(self.config.a_dtype) != "float8e4m3"
        ):
            raise RuntimeError("SM90 native A8 dense requires FP8 WGMMA arithmetic")
        if self.config.use_fused_e8m0_scale or self.config.use_int_weight_scale:
            raise RuntimeError(
                "V4.1 MXFP8 dense requires lossless UE8M0 weight-scale layout "
                "conversion; fused/int scale conversion is not supported"
            )
        tensors = transform_humming_tensors(
            self.config,
            {
                "weight": weight.contiguous().view(torch.int32),
                "weight_scale": expand_block32_scales(scales, self.N, self.K),
            },
        )
        self.weight = tensors["weight"]
        self.weight_scale = tensors["weight_scale"]
        self.weight_scale_2 = tensors.get("weight_scale_2")
        self.locks = torch.zeros(1024, dtype=torch.int32, device=weight.device)
        self.compute_config = json.dumps(
            {
                "gemm_type": "dense",
                "use_f16_accum": False,
                "use_m_major_input_scale": False,
                # H20's default K>1024 Stream-K policy accumulates partial
                # outputs in BF16 (gmem_writer.cuh::reduce_add_f162), violating
                # V41's FP32-accumulate/single-BF16-rounding dense contract.
                # The upstream batch-invariant policy disables Stream-K and
                # retains FP32 accumulation; do not relax numerical tolerances.
                "use_batch_invariant": True,
            }
        )

    def forward_quantized(self, x_q: torch.Tensor, x_s: torch.Tensor, out=None):
        if x_q.dtype != torch.float8_e4m3fn or x_q.shape[-1] != self.K:
            raise ValueError("Expected prequantized E4M3 activations with matching K")
        if x_q.device != self.weight.device or x_s.device != x_q.device:
            raise ValueError("Activations, scales and weights must share a device")
        rows = x_q.numel() // self.K
        expected_scales = (rows, (self.K // 32 + 3) // 4)
        if x_s.dtype != torch.int32 or tuple(x_s.shape) != expected_scales:
            raise ValueError(
                f"Expected packed group32 UE8M0 int32 scales {expected_scales}, "
                f"got {x_s.dtype} {x_s.shape}"
            )
        return self._forward_with_scales(x_q, unpack_ue8m0_input_scales(x_s), out)

    def forward_bf16(self, x: torch.Tensor, out=None):
        """Small-M BF16 producer: exact G32 quantization directly to FP32 scales."""
        from rtp_llm.models_py.modules.dsv41._sm90_quant_scale_fused import quantize_g32

        if (
            x.dtype != torch.bfloat16
            or x.device != self.weight.device
            or x.shape[-1] != self.K
            or x.numel() // self.K > 16
        ):
            raise ValueError("Direct G32 scales require same-device BF16 with M<=16")
        q, scales = quantize_g32(x.reshape(-1, self.K).contiguous())
        return self._forward_with_scales(q.view(x.shape), scales, out)

    def _forward_with_scales(self, x_q, scales, out):
        rows = x_q.numel() // self.K
        shape = (*x_q.shape[:-1], self.N)
        if out is None:
            out = torch.empty(shape, dtype=torch.bfloat16, device=x_q.device)
        elif (
            tuple(out.shape) != shape
            or out.dtype != torch.bfloat16
            or out.device != x_q.device
            or not out.is_contiguous()
        ):
            raise ValueError("out must be contiguous BF16 with the output shape/device")
        if rows:
            self._forward(
                self.config,
                inputs=x_q.reshape(rows, self.K).contiguous(),
                input_scale=scales,
                weight=self.weight,
                weight_scale=self.weight_scale,
                weight_scale_2=self.weight_scale_2,
                outputs=out.view(rows, self.N),
                locks=self.locks,
                compute_config=self.compute_config,
            )
        return out
