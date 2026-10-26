"""SM90 WO_A using native DeepGEMM G32 for every TP-local group.

Weights retain one checkpoint FP8 layout. All groups are submitted as a single
native batched GEMM (`fp8_einsum`); the SM90 grouped-GEMM API does not support
the G32 recipe. Wheels without `fp8_einsum` fall back to one dense GEMM per
group with identical operands and recipe. Inverse RoPE and activation
quantization keep their existing numerical contract.
"""

import logging

import torch
from rtp_llm.models_py.modules.dsv41._fused_inv_rope_fp8_quant_triton import fused_inv_rope_fp8_quant
from rtp_llm.models_py.modules.dsv41.fp8._v41_output_projection import quantization_scale_min

logger = logging.getLogger(__name__)


class Sm90GroupedOutputProjection(torch.nn.Module):
    """Native FP8 [G,N,K] weights and BF16 [M,G*N] outputs.

    Raw UE8M0 block32 scales and the historical expanded packed-int32 ABI
    are accepted. Expanded scales must be constant within every N32 block.
    Eagerly warm intended token counts before CUDA graph capture.
    """

    backend = "deepgemm"

    def __init__(self, weight: torch.Tensor, block_scales: torch.Tensor):
        super().__init__()
        if not weight.is_cuda or torch.cuda.get_device_capability(weight.device) != (9, 0):
            raise ValueError("SM90 grouped WO_A requires a CUDA SM90 device")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Construct grouped WO_A outside CUDA graph capture")
        if weight.dtype != torch.float8_e4m3fn or weight.ndim != 3:
            raise ValueError("Expected E4M3 weight [G,N,K]")
        self.G, self.N, self.K = weight.shape
        if self.G < 1 or self.N < 1 or self.N % 256 or self.K < 1 or self.K % 512:
            raise ValueError("Expected positive G, N divisible by 256, K divisible by 512")
        if block_scales.device != weight.device:
            raise ValueError("Weights and scales must share a CUDA device")
        if block_scales.dtype == torch.float8_e8m0fnu and tuple(block_scales.shape) == (self.G, self.N // 32, self.K // 32):
            native_scales = block_scales
        elif block_scales.dtype == torch.int32 and tuple(block_scales.shape) == (self.G, self.N, self.K // 128):
            packed = block_scales.contiguous().reshape(self.G * self.N, self.K // 128)
            packed = packed.as_strided(packed.shape, (self.K // 128, 1))
            expanded = packed.view(torch.uint8).reshape(self.G, self.N // 32, 32, self.K // 32)
            if not torch.equal(expanded, expanded[:, :, :1, :].expand_as(expanded)):
                raise ValueError("DeepGEMM WO_A requires equal scales within N32 blocks")
            native_scales = expanded[:, :, 0, :].contiguous().view(torch.float8_e8m0fnu)
        else:
            raise ValueError("Expected UE8M0 [G,N/32,K/32] or packed int32 [G,N,K/128]")
        import deep_gemm

        self._forward = deep_gemm.fp8_gemm_nt
        # Batched einsum keeps all G groups inside one kernel launch. Wheels
        # predating that API only expose the per-group GEMM, which stays as a
        # numerically identical fallback instead of failing at construction.
        self._einsum = getattr(deep_gemm, "fp8_einsum", None)
        if self._einsum is None:
            logger.warning(
                "deep_gemm.fp8_einsum is unavailable; grouped WO_A falls back to "
                "%d per-group GEMM launches",
                self.G,
            )
        self.register_buffer("weight", weight.contiguous(), persistent=False)
        self.register_buffer("weight_scale", native_scales.float().contiguous(), persistent=False)
        self._warmed_shapes = set()

    def _valid_input(self, o, freqs):
        if (
            o.device != self.weight.device
            or o.dtype != torch.bfloat16
            or o.ndim not in (3, 4)
            or o.shape[-1] != 512
            or o.shape[-2] != self.G * (self.K // 512)
            or o.stride(-1) != 1
            or o.requires_grad
            or freqs.requires_grad
            or freqs.device != o.device
            or freqs.dtype != torch.complex64
            or freqs.ndim != 2
            or freqs.shape[1] != 32
            or freqs.is_conj()
        ):
            return False
        tokens = o.numel() // (self.G * self.K)
        if not tokens:
            return True
        if freqs.shape[0] == 0 or tokens % freqs.shape[0]:
            return False
        if o.ndim == 4 and (
            freqs.shape[0] not in (o.shape[0], tokens)
            or (
                o.shape[0] > 1
                and o.shape[1] > 1
                and o.stride(0) != o.shape[1] * o.stride(1)
            )
        ):
            return False
        return True

    def supports(self, o, freqs, *, max_tokens=16):
        """Bounded static shape query; forward itself has no max_tokens cap."""
        return (
            self._valid_input(o, freqs) and o.numel() // (self.G * self.K) <= max_tokens
        )

    def quantize(self, o, freqs, *, imaginary_fma_first=True):
        """E4M3 [M,G,K], packed UE8M0 [M,G,K/128], physically group-major.

        Preserves the eager inverse-RoPE BF16 intermediate before G32 absmax;
        quantizer legacy/v2 policy is evaluated per group, not over G*M rows.
        Hopper eager complex multiplication contracts the imaginary coordinate
        as fma(a, -sin, b*cos). Preserve that order at BF16 midpoints. The
        explicit alternative is retained for the independent diagnostic test;
        other callers of the shared quantizer retain its unchanged default.
        """
        if not self._valid_input(o, freqs):
            raise ValueError(
                "Expected inference BF16 attention output and matching complex64 RoPE"
            )
        tokens = o.numel() // (self.G * self.K)
        if not tokens:
            q = torch.empty(
                (self.G, 0, self.K), dtype=torch.float8_e4m3fn, device=o.device
            )
            packed = torch.empty(
                (self.G, 0, self.K // 128), dtype=torch.int32, device=o.device
            )
            return q.transpose(0, 1), packed.transpose(0, 1)
        return fused_inv_rope_fp8_quant(
            o,
            freqs,
            n_groups=self.G,
            heads_per_group=self.K // 512,
            nope_dim=448,
            rope_head_dim=64,
            quant_group_size=32,
            eps=torch.finfo(torch.float32).tiny,
            round_rope_to_input_dtype=True,
            scale_min=quantization_scale_min(tokens, self.K),
            imaginary_fma_first=imaginary_fma_first,
        )

    def _output(self, tokens, out, operands):
        if out is None:
            return torch.empty(
                (tokens, self.G * self.N),
                dtype=torch.bfloat16,
                device=self.weight.device,
            )
        if (
            out.device != self.weight.device
            or out.dtype != torch.bfloat16
            or tuple(out.shape) != (tokens, self.G * self.N)
            or not out.is_contiguous()
            or out.requires_grad
        ):
            raise ValueError("Expected contiguous inference BF16 out [M,G*N]")
        protected = (*operands, *self.buffers())
        if any(t is not None and torch._C._overlaps(out, t) for t in protected):
            raise ValueError(
                "Output must not alias inputs, weights or grouped metadata"
            )
        return out

    def forward_quantized(self, q, packed, *, out=None):
        if (
            q.device != self.weight.device
            or q.dtype != torch.float8_e4m3fn
            or q.ndim != 3
            or tuple(q.shape[1:]) != (self.G, self.K)
            or q.requires_grad
            or packed.device != q.device
            or packed.dtype != torch.int32
            or tuple(packed.shape) != (q.shape[0], self.G, self.K // 128)
        ):
            raise ValueError(
                "Expected E4M3 [M,G,K] and packed G32 UE8M0 int32 [M,G,K/128]"
            )
        tokens = q.shape[0]
        out = self._output(tokens, out, (q, packed))
        if not tokens:
            return out
        group_packed = (
            packed.transpose(0, 1).contiguous().view(self.G * tokens, self.K // 128)
        )
        from rtp_llm.models_py.modules.dsv41._sm90_quant_scale_fused import unpack_ue8m0_to_fp32

        return self._forward_with_scales(
            q, unpack_ue8m0_to_fp32(group_packed, mn_major=True, n_groups=self.G), out
        )

    def _forward_with_scales(self, q, scales, out):
        tokens = q.shape[0]
        if (
            tokens not in self._warmed_shapes
            and torch.cuda.is_current_stream_capturing()
        ):
            raise RuntimeError(
                "Eagerly warm each grouped WO_A token count before CUDA graph capture"
            )
        if scales.dtype != torch.float32 or scales.device != q.device:
            raise ValueError("Expected same-device FP32 activation scales")
        if tuple(scales.shape) == (self.G, tokens, self.K // 32):
            group_scales = scales
        elif tuple(scales.shape) == (self.G * tokens, self.K // 32):
            group_scales = scales.reshape(self.G, tokens, self.K // 32)
        else:
            raise ValueError("Expected group-major scales [G,M,K/32]")
        if self._einsum is not None:
            # One native batched GEMM covers every TP-local group. `q` is
            # already logical [M,G,K] and `out` is [M,G,N], so neither operand
            # needs a copy; the transposed group-major scales are the
            # TMA-ready [G,M,K/32] view the kernel consumes without a
            # transpose. Same weights, scales and recipe as the per-group
            # loop, so only the launch count changes.
            self._einsum(
                "bhr,hdr->bhd",
                (q, group_scales.transpose(0, 1)),
                (self.weight, self.weight_scale),
                out.view(tokens, self.G, self.N),
                recipe=(1, 32, 32),
            )
            self._warmed_shapes.add(tokens)
            return out
        # Fallback for wheels without `fp8_einsum`: one dense GEMM per group.
        group_q = q.transpose(0, 1).contiguous()
        if tokens == 1 or self.G == 1:
            group_out = out.view(self.G, tokens, self.N)
        else:
            group_out = torch.empty((self.G, tokens, self.N), dtype=torch.bfloat16, device=q.device)
        # G is fixed model metadata. Capture records each group's native GEMM
        # and its scale producer; replay observes updated inputs and scales.
        for group in range(self.G):
            self._forward(
                (group_q[group], group_scales[group]),
                (self.weight[group], self.weight_scale[group]),
                group_out[group], recipe=(1, 32, 32), compiled_dims="",
            )
        if tokens > 1 and self.G > 1:
            out.view(tokens, self.G, self.N).copy_(group_out.transpose(0, 1))
        self._warmed_shapes.add(tokens)
        return out

    def forward(self, o, freqs, *, out=None):
        if not self._valid_input(o, freqs):
            raise ValueError(
                "Expected inference BF16 attention output and matching complex64 RoPE"
            )
        tokens = o.numel() // (self.G * self.K)
        out = self._output(tokens, out, (o, freqs))
        if not tokens:
            return out
        if (
            tokens not in self._warmed_shapes
            and torch.cuda.is_current_stream_capturing()
        ):
            raise RuntimeError(
                "Eagerly warm each grouped WO_A token count before CUDA graph capture"
            )
        if tokens <= 16:
            from rtp_llm.models_py.modules.dsv41._sm90_quant_scale_fused import (
                quantize_wo_a_g32,
            )

            q, scales = quantize_wo_a_g32(
                o, freqs, n_groups=self.G, heads_per_group=self.K // 512, mn_major=True
            )
            group_scales = scales.transpose(0, 1)
            return self._forward_with_scales(q, group_scales, out)
        return self.forward_quantized(*self.quantize(o, freqs), out=out)
