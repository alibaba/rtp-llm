"""FP8-storage / BF16-WGMMA adapter used by the opt-in shared-dense selector.

Uses the existing Humming W8A16 implementation: weights remain E4M3 with their
original g32 UE8M0 scales and are converted in registers, not expanded globally.
The original externally quantized activation is decoded with its original
scales to a call-owned BF16 workspace before GEMM. No amax is recomputed and no
new FP8 quantization occurs. Decoding to BF16 changes the operand range at the
arithmetic boundary; exactness requires representability and must be gated.

This is explicitly NOT a native-FP8 arithmetic repair. It is a separate
BF16-WGMMA precision/performance candidate. Include activation decoding in
prepared/full-chain measurements. Warm each row count before graph capture.
Call-owned activation storage follows the CUDA graph allocator lifecycle.
"""

import json

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.modules.dsv41.sm90_dense import (
    expand_block32_scales,
    unpack_ue8m0_input_scales,
)


@triton.jit(do_not_specialize=["ELEMENTS"])
def _decode_original_fp8_to_bf16(
    Q, Scales, Out, ELEMENTS, K: tl.constexpr,
    SCALE_STRIDE_M: tl.constexpr, SCALE_STRIDE_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = index // K
    group = (index % K) // 32
    value = tl.load(Q + index, mask=index < ELEMENTS, other=0.0).to(tl.float32)
    scale = tl.load(
        Scales + row * SCALE_STRIDE_M + group * SCALE_STRIDE_K,
        mask=index < ELEMENTS, other=0.0,
    )
    tl.store(Out + index, (value * scale).to(tl.bfloat16), mask=index < ELEMENTS)


class Sm90Wgmma16Linear:
    """Humming BF16 WGMMA using original FP8 payloads/scales as its source."""

    def __init__(self, weight: torch.Tensor, scales: torch.Tensor):
        if not weight.is_cuda or torch.cuda.get_device_capability(weight.device) != (9, 0):
            raise ValueError("BF16 WGMMA candidate requires CUDA SM90")
        if weight.ndim != 2 or weight.dtype != torch.float8_e4m3fn:
            raise TypeError("Expected 2-D E4M3 weights")
        self.N, self.K = weight.shape
        if self.N <= 0 or self.N % 32 or self.K <= 0 or self.K % 128:
            raise ValueError("Bounded candidate requires N%32=0 and K%128=0")
        if self.N * self.K >= 2**31:
            raise ValueError("Weight exceeds 32-bit indexing bound")
        if (scales.device != weight.device or scales.dtype != torch.float8_e8m0fnu
                or tuple(scales.shape) != (self.N // 32, self.K // 32)):
            raise ValueError("Expected same-device UE8M0 32x32 weight scales")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Construct candidate outside CUDA graph capture")
        from humming.config import MmaType
        from humming.forward import humming_forward
        from humming.schema import HummingInputSchema, HummingWeightSchema
        from humming.transform import prepare_layer_config, transform_humming_tensors

        self._forward = humming_forward
        self.config = prepare_layer_config(
            shape_n=self.N,
            shape_k=self.K,
            weight_schema=HummingWeightSchema(
                b_dtype="float8e4m3", bs_dtype="float8e8m0",
                weight_scale_group_size=32, weight_scale_type="group",
            ),
            # A16 cannot carry separate group scales in Humming. Decode the
            # original A8/group scales explicitly, then disable its quantizer.
            input_schema=HummingInputSchema(a_dtype="bfloat16", input_scale_group_size=0),
            pad_n_to_multiple=256, pad_k_to_multiple=128,
            torch_dtype=torch.bfloat16, device=weight.device,
        )
        if self.config.mma_type != MmaType.WGMMA:
            raise RuntimeError("Expected genuine Humming WGMMA configuration")
        if self.config.has_input_scale or self.config.input_quant_mode.should_quantize:
            raise RuntimeError("Humming must not requantize already decoded activations")
        if self.config.use_fused_e8m0_scale or self.config.use_int_weight_scale:
            raise RuntimeError("Candidate requires lossless original weight-scale layout conversion")
        # New BF16-activation layout, never reuse an A8 MMA/WGMMA weight repack.
        tensors = transform_humming_tensors(
            self.config,
            {
                "weight": weight.contiguous().clone().view(torch.int32),
                "weight_scale": expand_block32_scales(scales, self.N, self.K),
            },
        )
        self.weight = tensors["weight"]
        self.weight_scale = tensors["weight_scale"]
        self.weight_scale_2 = tensors.get("weight_scale_2")
        self.locks = torch.zeros(1024, device=weight.device, dtype=torch.int32)
        self.compute_config = json.dumps(
            dict(gemm_type="dense", use_f16_accum=False, use_batch_invariant=True)
        )
        # Leave tile selection to installed Humming's existing legal WGMMA
        # heuristic. Batch-invariant disables BF16 Stream-K output reductions.
        self.tuning_config = None
        self.arithmetic_contract = "FP8_STORAGE_BF16_WGMMA_FP32_ACCUM; SASS_AUDIT_REQUIRED"
        self.last_dequant_kernel = None

    def _output(self, q, out):
        if (q.ndim < 1 or q.dtype != torch.float8_e4m3fn or q.shape[-1] != self.K
                or q.device != self.weight.device or not q.is_contiguous()):
            raise ValueError("Expected contiguous same-device E4M3 activations with matching K")
        rows = q.numel() // self.K
        if max(rows * self.K, rows * self.N) >= 2**31:
            raise ValueError("Activation/output exceeds 32-bit indexing bound")
        shape = (*q.shape[:-1], self.N)
        if out is None:
            out = torch.empty(shape, device=q.device, dtype=torch.bfloat16)
        if (out.shape != shape or out.dtype != torch.bfloat16 or out.device != q.device
                or not out.is_contiguous()):
            raise ValueError("out must be contiguous same-device BF16 with matching shape")
        if any(torch._C._overlaps(out, value) for value in (q, self.weight, self.weight_scale)):
            raise ValueError("out must not overlap activation or weights")
        return rows, out

    def _workspace(self, rows):
        # Allocation is graph-pool managed during capture; no mutable per-row
        # cache that can overflow on arbitrary prefill lengths or alias calls.
        return torch.empty((rows, self.K), device=self.weight.device, dtype=torch.bfloat16)

    def forward_quantized(self, q, packed_scales, out=None):
        rows, out = self._output(q, out)
        if (packed_scales.device != q.device or packed_scales.dtype != torch.int32
                or tuple(packed_scales.shape) != (rows, self.K // 128)):
            raise ValueError("Expected original packed group32 UE8M0 activation scales")
        if torch._C._overlaps(out, packed_scales):
            raise ValueError("out must not overlap input scales")
        if not rows:
            return out
        return self.prepared(q, unpack_ue8m0_input_scales(packed_scales), out)

    def prepared(self, q, float_scales, out=None):
        rows, out = self._output(q, out)
        if (float_scales.device != q.device or float_scales.dtype != torch.float32
                or tuple(float_scales.shape) != (rows, self.K // 32)
                or any(stride <= 0 for stride in float_scales.stride())):
            raise ValueError("Expected positive-stride original [M,K/32] FP32 scales")
        if torch._C._overlaps(out, float_scales):
            raise ValueError("out must not overlap input scales")
        if not rows:
            return out
        decoded = self._workspace(rows)
        if any(torch._C._overlaps(decoded, value) for value in (q, float_scales, out)):
            raise ValueError("Caller tensors must not alias cached activation workspace")
        self.last_dequant_kernel = _decode_original_fp8_to_bf16[(triton.cdiv(q.numel(), 512),)](
            q, float_scales, decoded, q.numel(), K=self.K,
            SCALE_STRIDE_M=float_scales.stride(0), SCALE_STRIDE_K=float_scales.stride(1),
            BLOCK=512,
        )
        self._forward(
            self.config, inputs=decoded, weight=self.weight,
            weight_scale=self.weight_scale, weight_scale_2=self.weight_scale_2,
            outputs=out.view(rows, self.N), locks=self.locks,
            compute_config=self.compute_config, tuning_config=self.tuning_config,
        )
        return out
