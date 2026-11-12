"""Qwen4 GDN norm admission, fallback and projection dispatch."""

import os

import torch

from rtp_llm.models_py.kernels.cuda.gdn_sm120.norm import (
    _prefill_gated_norm,
    norm_quant,
)


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
