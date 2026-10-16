"""Hopper WO_A: inverse RoPE/G32 quantization and one Humming grouped GEMM.

Reuses Humming 0.1.16's native FP8 WGMMA ``grouped_contiguous`` backend, or
its dense backend for a single TP-local group. No new GEMM kernel or weight
requantization is involved. The FP32 accumulator
policy deliberately disables Stream-K's BF16 partial-output reduction.
Construct and eagerly run each intended shape before CUDA graph capture.
Prefill and decode shapes are qualified independently: the public forward
supports both, without claiming one scheduler is optimal for every M. The
optional ``supports`` helper defaults to a bounded decode-shaped query; it
does not restrict production forward or stand in for performance validation.
"""

import json

import torch

from rtp_llm.models_py.modules.dsv41._fused_inv_rope_fp8_quant_triton import (
    fused_inv_rope_fp8_quant,
)
from rtp_llm.models_py.modules.dsv41.fp8._v41_output_projection import (
    quantization_scale_min,
)
from rtp_llm.models_py.modules.dsv41.sm90_dense import unpack_ue8m0_input_scales


class Sm90GroupedOutputProjection(torch.nn.Module):
    """Prepared G32 W8A8 projection, returning BF16 ``[M, G*N]``.

    ``weight`` is checkpoint E4M3 ``[G,N,K]``. ``block_scales`` accepts raw
    UE8M0 block32 ``[G,N/32,K/32]`` or the existing stacked packed-int32
    per-row ``[G,N,K/128]`` ABI. Only one transformed weight set is kept.
    Like the existing Humming dense adapter, one instance is not intended
    for simultaneous execution on multiple CUDA streams (shared locks).
    """

    def __init__(self, weight: torch.Tensor, block_scales: torch.Tensor):
        super().__init__()
        if not weight.is_cuda or torch.cuda.get_device_capability(weight.device) != (
            9,
            0,
        ):
            raise ValueError("SM90 grouped WO_A requires a CUDA SM90 device")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Construct grouped WO_A outside CUDA graph capture")
        if weight.dtype != torch.float8_e4m3fn or weight.ndim != 3:
            raise ValueError("Expected E4M3 weight [G,N,K]")
        self.G, self.N, self.K = weight.shape
        if self.G < 1 or self.N < 1 or self.N % 256 or self.K < 1 or self.K % 512:
            raise ValueError(
                "Expected positive G, N divisible by 256, K divisible by 512"
            )
        if block_scales.device != weight.device:
            raise ValueError("Weights and scales must share a CUDA device")
        if block_scales.dtype == torch.float8_e8m0fnu and tuple(block_scales.shape) == (
            self.G,
            self.N // 32,
            self.K // 32,
        ):
            scales = (
                block_scales.contiguous().view(torch.uint8).repeat_interleave(32, dim=1)
            )
        elif block_scales.dtype == torch.int32 and tuple(block_scales.shape) == (
            self.G,
            self.N,
            self.K // 128,
        ):
            # Canonical strides also handle singleton dimensions in PyTorch's
            # otherwise-contiguous TMA views. Reinterpretation preserves bytes.
            scales = block_scales.contiguous().view(self.G * self.N, self.K // 128)
            scales = scales.as_strided(scales.shape, (self.K // 128, 1))
            scales = scales.view(torch.uint8).view(self.G, self.N, self.K // 32)
        else:
            raise ValueError("Expected UE8M0 [G,N/32,K/32] or packed int32 [G,N,K/128]")

        from humming.forward import humming_forward
        from humming.schema import HummingInputSchema, HummingWeightSchema
        from humming.transform import prepare_layer_config, transform_humming_tensors

        self._forward = humming_forward
        # A single TP-local group is an ordinary dense GEMM. Grouped H20
        # scheduling regresses large M for G=1; prepare the same dense layout
        # and scheduler as the existing linear, retaining fused RoPE/quant.
        # This is one transform/one weight set, not an additional fallback.
        self.config = prepare_layer_config(
            shape_n=self.N,
            shape_k=self.K,
            num_experts=0 if self.G == 1 else self.G,
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
            raise RuntimeError("Grouped WO_A requires native FP8 WGMMA")
        if self.config.use_fused_e8m0_scale or self.config.use_int_weight_scale:
            raise RuntimeError(
                "Grouped WO_A requires lossless UE8M0 weight-scale conversion"
            )
        transform_weight = weight[0] if self.G == 1 else weight
        transform_scales = scales[0] if self.G == 1 else scales
        tensors = transform_humming_tensors(
            self.config,
            {
                "weight": transform_weight.contiguous().clone().view(torch.int32),
                "weight_scale": transform_scales.contiguous()
                .clone()
                .view(torch.float8_e8m0fnu),
            },
        )
        for name, value in tensors.items():
            self.register_buffer(name, value, persistent=False)
        if not hasattr(self, "weight_scale_2"):
            self.register_buffer("weight_scale_2", None, persistent=False)
        self.register_buffer(
            "locks",
            torch.zeros(1024, dtype=torch.int32, device=weight.device),
            persistent=False,
        )
        self.compute_config = json.dumps(
            {
                "gemm_type": "dense" if self.G == 1 else "grouped_contiguous",
                "use_f16_accum": False,
                "use_m_major_input_scale": False,
                "use_batch_invariant": True,
            }
        )
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

    def _offsets(self, tokens):
        name = f"_expert_offsets_{tokens}"
        if not hasattr(self, name):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "Eagerly warm each grouped WO_A token count before CUDA graph capture"
                )
            self.register_buffer(
                name,
                torch.arange(self.G + 1, dtype=torch.int32, device=self.weight.device)
                * tokens,
                persistent=False,
            )
        return getattr(self, name)

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
        return self._forward_with_scales(
            q, unpack_ue8m0_input_scales(group_packed), out
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
        offsets = None if self.G == 1 else self._offsets(tokens)
        # Fused quant already writes this physical layout: the Q transpose is
        # a zero-copy view, not an M*G*K permutation on the production path.
        group_q = q.transpose(0, 1).contiguous().view(self.G * tokens, self.K)
        if tokens == 1 or self.G == 1:
            group_out = out.view(self.G * tokens, self.N)
        else:
            group_out = torch.empty(
                (self.G * tokens, self.N), dtype=torch.bfloat16, device=q.device
            )
        self._forward(
            self.config,
            inputs=group_q,
            input_scale=scales,
            weight=self.weight,
            weight_scale=self.weight_scale,
            weight_scale_2=self.weight_scale_2,
            outputs=group_out,
            expert_layout=offsets,
            locks=self.locks,
            compute_config=self.compute_config,
        )
        if tokens > 1 and self.G > 1:
            out.view(tokens, self.G, self.N).copy_(
                group_out.view(self.G, tokens, self.N).transpose(0, 1)
            )
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
                o, freqs, n_groups=self.G, heads_per_group=self.K // 512
            )
            group_scales = scales.transpose(0, 1).reshape(self.G * tokens, self.K // 32)
            return self._forward_with_scales(q, group_scales, out)
        return self.forward_quantized(*self.quantize(o, freqs), out=out)
