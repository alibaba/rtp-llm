"""Bounded M=1, G=8 WO_A adapter using installed Humming grouped GEMM.

This module owns no RoPE or new arithmetic kernel. Its input is the original
post-inverse-RoPE BF16 intermediate. RTP owns E4M3/group32 quantization and
Humming only changes eight independent GEMMs into one grouped launch. Build
and eagerly warm the adapter before CUDA graph capture. No production routing
is changed by importing this module.
"""

import json

import torch


class Sm90GroupedWOA(torch.nn.Module):
    """[1,8,4096] BF16 -> [1,8192] BF16; native W8A8, original g32 scales."""

    GROUPS, N, K = 8, 1024, 4096

    def __init__(self, weight: torch.Tensor, block_scales: torch.Tensor):
        super().__init__()
        if not weight.is_cuda or torch.cuda.get_device_capability(weight.device) != (9, 0):
            raise ValueError("Sm90GroupedWOA requires SM90")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Prepare grouped WO_A outside CUDA graph capture")
        if weight.dtype != torch.float8_e4m3fn or tuple(weight.shape) != (8, 1024, 4096):
            raise ValueError("Expected E4M3 weight [8,1024,4096]")
        if (
            block_scales.device != weight.device
            or block_scales.dtype != torch.float8_e8m0fnu
            or tuple(block_scales.shape) != (8, 32, 128)
        ):
            raise ValueError("Expected same-device UE8M0 block32 scales [8,32,128]")
        from humming.forward import humming_forward
        from humming.schema import HummingInputSchema, HummingWeightSchema
        from humming.transform import prepare_layer_config, transform_humming_tensors

        self._forward = humming_forward
        self.config = prepare_layer_config(
            shape_n=self.N, shape_k=self.K, num_experts=self.GROUPS,
            weight_schema=HummingWeightSchema(
                b_dtype="float8e4m3", bs_dtype="float8e8m0",
                weight_scale_group_size=32, weight_scale_type="group",
            ),
            input_schema=HummingInputSchema(
                a_dtype="float8e4m3", input_scale_dtype="float32",
                input_scale_group_size=32, input_quant_mode="dynamic_group",
            ),
            pad_n_to_multiple=256, pad_k_to_multiple=128,
            torch_dtype=torch.bfloat16, device=weight.device,
        )
        if self.config.mma_type.value != "wgmma" or str(self.config.a_dtype) != "float8e4m3":
            raise RuntimeError("Grouped WO_A requires native FP8 WGMMA")
        if self.config.use_fused_e8m0_scale or self.config.use_int_weight_scale:
            raise RuntimeError("Grouped WO_A disallows lossy weight-scale conversion")
        # Expand N's block32 scale without modifying checkpoint exponent bytes.
        expanded = block_scales.contiguous().view(torch.uint8).repeat_interleave(32, dim=1)
        tensors = transform_humming_tensors(self.config, {
            "weight": weight.contiguous().clone().view(torch.int32),
            "weight_scale": expanded.contiguous().view(torch.float8_e8m0fnu),
        })
        for name, value in tensors.items():
            self.register_buffer(name, value, persistent=False)
        if not hasattr(self, "weight_scale_2"):
            self.register_buffer("weight_scale_2", None, persistent=False)
        self.register_buffer("locks", torch.zeros(1024, dtype=torch.int32, device=weight.device), persistent=False)
        # Static row ownership: one row per group. No sort, padding or CPU read.
        self.register_buffer("expert_offsets", torch.arange(9, dtype=torch.int32, device=weight.device), persistent=False)
        self.compute_config = json.dumps({
            "gemm_type": "grouped_contiguous", "use_f16_accum": False,
            "use_m_major_input_scale": False, "use_batch_invariant": True,
        })
        self._warmed = False

    def supports(self, x: torch.Tensor) -> bool:
        return (
            x.device == self.weight.device and x.dtype == torch.bfloat16
            and tuple(x.shape) == (1, self.GROUPS, self.K) and x.is_contiguous()
            and not (torch.is_grad_enabled() and x.requires_grad)
        )

    def quantize(self, x: torch.Tensor):
        if not self.supports(x):
            raise ValueError("Grouped WO_A supports contiguous inference BF16 [1,8,4096] only")
        from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8

        # Both M=1 original groups and the combined eight rows stay below the
        # original auto-v2 threshold. Explicit legacy/v2 selections are kept.
        return sgl_per_token_group_quant_fp8(
            x.view(self.GROUPS, self.K), group_size=32,
            eps=torch.finfo(torch.float32).tiny, column_major_scales=True,
            scale_tma_aligned=True, scale_ue8m0=True,
        )

    def forward_quantized(self, q: torch.Tensor, packed: torch.Tensor, *, out=None):
        if not self._warmed and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Eagerly warm grouped WO_A before CUDA graph capture")
        if (
            q.device != self.weight.device or q.dtype != torch.float8_e4m3fn
            or tuple(q.shape) != (self.GROUPS, self.K) or not q.is_contiguous()
            or packed.device != q.device or packed.dtype != torch.int32
            or tuple(packed.shape) != (self.GROUPS, self.K // 128)
        ):
            raise ValueError("Expected E4M3 [8,4096] and packed UE8M0 int32 [8,32]")
        if out is None:
            out = torch.empty((1, self.GROUPS * self.N), dtype=torch.bfloat16, device=q.device)
        if (
            out.device != q.device or out.dtype != torch.bfloat16
            or tuple(out.shape) != (1, self.GROUPS * self.N) or not out.is_contiguous()
        ):
            raise ValueError("Expected contiguous BF16 output [1,8192]")
        operands = (q, packed, self.weight, self.weight_scale,
                    self.weight_scale_2, self.expert_offsets, self.locks)
        if any(t is not None and torch._C._overlaps(out, t) for t in operands):
            raise ValueError("Output must not alias input, weights or grouped metadata")
        from rtp_llm.models_py.modules.dsv41.sm90_dense import unpack_ue8m0_input_scales

        self._forward(
            self.config, inputs=q, input_scale=unpack_ue8m0_input_scales(packed),
            weight=self.weight, weight_scale=self.weight_scale,
            weight_scale_2=self.weight_scale_2, outputs=out.view(self.GROUPS, self.N),
            expert_layout=self.expert_offsets, locks=self.locks,
            compute_config=self.compute_config,
        )
        self._warmed = True
        return out

    def forward(self, x: torch.Tensor, *, out=None):
        return self.forward_quantized(*self.quantize(x), out=out)
