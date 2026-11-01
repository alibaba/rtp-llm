"""PPU DeepGEMM FP8 per-block dense Linear implementation."""

import logging
from typing import Optional

import torch

from rtp_llm.models_py.kernels.ppu.deepgemm_warmup import (
    get_deep_gemm_warmup_mode,
    resolve_deep_gemm_warmup_max_tokens,
    warmup_dense_fp8_gemm,
)
from rtp_llm.models_py.kernels.ppu.deepgemm_wrapper import (
    configure_deep_gemm_num_sms,
    deep_gemm_default_num_sms,
    fp8_gemm_nt,
    has_deep_gemm_fp8_dense,
)
from rtp_llm.models_py.modules.factory.linear import LinearBase
from rtp_llm.ops import HWKernelConfig
from rtp_llm.ops.compute_ops import trt_fp8_quantize_128

logger = logging.getLogger(__name__)


class PpuFp8DeepGemmLinear(LinearBase):
    """FP8 e4m3 weights with 128x128 block scales, dynamically quantized input."""

    QUANT_METHOD = "FP8_PER_BLOCK"

    @classmethod
    def can_handle(
        cls,
        quant_config: object,
        weight: torch.Tensor,
        weight_scales: Optional[torch.Tensor],
        hw_kernel_config: Optional["HWKernelConfig"] = None,
        weight_scale_2: Optional[torch.Tensor] = None,
        input_scale: Optional[torch.Tensor] = None,
    ) -> bool:
        return (
            weight.dtype == torch.float8_e4m3fn
            and weight_scales is not None
            and weight_scales.dtype == torch.float32
            and quant_config is not None
            and quant_config.get_method() == cls.QUANT_METHOD
            and has_deep_gemm_fp8_dense()
        )

    def __init__(
        self,
        weight: torch.Tensor,
        weight_scales: Optional[torch.Tensor] = None,
        input_scales: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        quant_config: object = None,
        weight_scale_2: Optional[torch.Tensor] = None,
    ):
        super().__init__(
            weight, weight_scales, input_scales, bias, quant_config, weight_scale_2
        )
        if weight.dim() != 2 or weight_scales is None or weight_scales.dim() != 2:
            raise ValueError(
                "FP8 dense weight and scale must be 2D, got "
                f"{tuple(weight.shape)} and "
                f"{None if weight_scales is None else tuple(weight_scales.shape)}"
            )

        # RTP-LLM stores dense kernels as a [K, N] view over checkpoint [N, K]
        # storage; DeepGEMM consumes row-major [N, K] with (N//128, K//128)
        # block scales.
        self.k, self.n = weight.shape
        scale_k, scale_n = weight_scales.shape
        self.weight = weight.reshape(self.n, self.k).contiguous()
        self.weight_scales = weight_scales.reshape(scale_n, scale_k).contiguous()
        if scale_n != (self.n + 127) // 128 or scale_k != (self.k + 127) // 128:
            raise ValueError(
                f"FP8 weight scale shape {(scale_k, scale_n)} does not match "
                f"N/K={(self.n, self.k)} with 128x128 blocks"
            )
        self.bias = bias

        # The wheel's own grid width for every shape -- see
        # deep_gemm_default_num_sms(). One source for warmup and execution.
        self.num_gemm_sms = deep_gemm_default_num_sms()

        self._maybe_warmup()

    def _maybe_warmup(self) -> None:
        """Compile-only JIT the dense kernel so the first request skips JIT."""
        try:
            mode = get_deep_gemm_warmup_mode()
            if mode == "skip":
                return
            warmup_dense_fp8_gemm(
                (self.weight, self.weight_scales),
                max_m=resolve_deep_gemm_warmup_max_tokens(),
                mode=mode,
                num_sms=self.num_gemm_sms,
            )
        except Exception:
            logger.exception(
                "DeepGEMM dense FP8 warmup failed; execution will use lazy JIT"
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.shape[-1] != self.k:
            raise ValueError(
                f"FP8 Linear expected input K={self.k}, got {input.shape[-1]}"
            )
        if input.dtype != torch.bfloat16:
            raise ValueError(f"FP8 Linear expects BF16 input, got {input.dtype}")
        output_shape = (*input.shape[:-1], self.n)
        input_2d = input.reshape(-1, self.k)
        if input_2d.shape[0] == 0:
            return input.new_empty(output_shape)

        input_q, input_scale = trt_fp8_quantize_128(input_2d, False)
        output = torch.empty(
            (input_2d.shape[0], self.n),
            device=input.device,
            dtype=torch.bfloat16,
        )
        with configure_deep_gemm_num_sms(self.num_gemm_sms):
            fp8_gemm_nt(
                (input_q, input_scale),
                (self.weight, self.weight_scales),
                output,
            )
        if self.bias is not None:
            output.add_(self.bias.to(output.dtype))
        return output.to(input.dtype).view(output_shape)
