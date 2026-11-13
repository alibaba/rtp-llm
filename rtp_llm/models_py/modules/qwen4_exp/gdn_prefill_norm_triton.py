"""Qwen4 sigmoid-gated GDN norm with multiple head reductions per CTA."""

import os

import torch
import triton
import triton.language as tl

try:
    from .gdn_prefill_norm_gluon import flat_head_norm
    from .gdn_prefill_norm_quant_gluon import norm_quant
except ImportError:
    # Older platform Triton versions retain the existing implementation.
    flat_head_norm = None
    norm_quant = None


@triton.jit
def _gdn_prefill_gated_norm(
    X,
    Z,
    W,
    Y,
    N: tl.constexpr,
    ZX: tl.constexpr,
    XX: tl.constexpr,
    EPS: tl.constexpr,
    GROUPS: tl.constexpr,
    GH: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1) * GH + tl.arange(0, GH)
    col = tl.arange(0, 128)
    offset = head[:, None] * 128 + col[None, :]
    value = tl.load(X + row * XX + offset, head[:, None] < GROUPS, other=0.0).to(
        tl.float32
    )
    variance = tl.sum(value * value, 1) / 128.0
    inverse = 1.0 / tl.sqrt(variance + EPS)
    weight = tl.load(W + col).to(tl.float32)
    gate = tl.load(Z + row * ZX + offset, head[:, None] < GROUPS, other=0.0).to(
        tl.float32
    )
    result = (value * inverse[:, None]) * weight[None, :]
    result *= tl.sigmoid(gate)
    tl.store(Y + row * N + offset, result, head[:, None] < GROUPS)


def _prefill_gated_norm(x, z, weight, eps=1e-6, groups_per_cta=4):
    rows, width = x.shape
    if flat_head_norm is not None and width in (1536, 3072):
        return flat_head_norm(x, z, weight, eps)
    groups = width // 128
    out = torch.empty_like(x)
    _gdn_prefill_gated_norm[(rows, triton.cdiv(groups, groups_per_cta))](
        x,
        z,
        weight,
        out,
        N=width,
        ZX=z.stride(0),
        XX=x.stride(0),
        EPS=eps,
        GROUPS=groups,
        GH=groups_per_cta,
        num_warps=groups_per_cta,
    )
    return out


class Qwen4PrefillGatedNorm(torch.nn.Module):
    """Explicit Qwen4-only prefill wrapper; preserve the native norm fallback."""

    def __init__(self, native, *, role):
        super().__init__()
        self.native = native
        self.role = str(role).rsplit(".", 1)[-1]

    @property
    def weight(self):
        return self.native.weight

    @property
    def bias(self):
        return self.native.bias

    @property
    def group_size(self):
        return self.native.group_size

    @property
    def eps(self):
        return self.native.eps

    @property
    def activation(self):
        return self.native.activation

    def _supports_prefill(self, x, gate):
        width = x.shape[-1] if x.ndim == 2 else 0
        return (
            self.role in ("PREFILL", "PDFUSION")
            and x.is_cuda
            and torch.version.hip is None
            and x.ndim == 2
            and width in (768, 1536, 3072)
            and 2048 <= x.shape[0] <= 12288
            and (width != 3072 or x.shape[0] <= 8192)
            and self.group_size == 128
            and self.activation == "sigmoid"
            and self.bias is None
            and self.weight.shape == (128,)
            and self.weight.dtype in (torch.bfloat16, torch.float32)
            and x.dtype == gate.dtype == torch.bfloat16
            and x.shape == gate.shape
            and x.device == gate.device == self.weight.device
            and x.is_contiguous()
            and gate.stride(-1) == 1
            and gate.stride(0) > 0
            and self.weight.is_contiguous()
            and not any(t.requires_grad for t in (x, gate, self.weight))
            and torch.cuda.get_device_capability(x.device) == (12, 0)
        )

    def forward(self, x, gate):
        if self._supports_prefill(x, gate):
            return _prefill_gated_norm(
                x, gate, self.weight, self.eps, 8 if x.shape[1] == 768 else 4
            )
        return self.native(x, gate)

    def forward_project(self, x, gate, projection):
        """Explicit GDN consumer; preserve the BF16 norm quantization boundary."""
        if not (
            norm_quant is not None
            and self._supports_prefill(x, gate)
            and x.shape[1] in (1536, 3072)
            and x.shape[0] <= 8192
        ):
            return projection(self(x, gate))

        from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_deepgemm_linear import (
            CudaFp8DeepGEMMLinear,
        )
        from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_gemm_linear import (
            CudaFp8GEMMLinear,
        )

        def unmodified(module, methods):
            return (
                not module._forward_hooks
                and not module._forward_pre_hooks
                and all(name not in module.__dict__ for name in methods)
            )

        consumer = projection
        if type(projection) is CudaFp8GEMMLinear:
            # The model's LinearFactory returns this dispatcher, not its inner
            # backend. Follow the same BF16-input dispatch as its forward; do
            # not force DeepGEMM where the wrapper selects FlashInfer.
            if not unmodified(projection, ("forward", "_should_use_flashinfer")):
                return projection(self(x, gate))
            if projection._should_use_flashinfer(x):
                return projection(self(x, gate))
            consumer = projection._deepgemm_linear

        if (
            type(consumer) is CudaFp8DeepGEMMLinear
            and consumer.scale_ue8m0
            and consumer.K == x.shape[1]
            and os.environ.get("DSV4_FP8_QUANT_KERNEL", "auto").strip().lower()
            in ("auto", "v2")
            and (
                x.numel() >= 4 * 1024 * 1024
                or os.environ.get("DSV4_FP8_QUANT_KERNEL", "auto").strip().lower()
                == "v2"
            )
            and consumer.N == 2560
            and consumer.weight.device == x.device
            and unmodified(self, ("forward",))
            and unmodified(consumer, ("forward", "quantize_input", "forward_quantized"))
            and not torch.nn.modules.module._global_forward_pre_hooks
            and not torch.nn.modules.module._global_forward_hooks
        ):
            quantized, scales = norm_quant(x, gate, self.weight, self.eps)
            return consumer.forward_quantized(quantized, scales)
        return projection(self(x, gate))
