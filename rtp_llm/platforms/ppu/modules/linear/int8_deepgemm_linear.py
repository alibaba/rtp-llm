"""PPU DeepGEMM W8A8 INT8 dense Linear implementation."""

import logging
from typing import Optional

import torch
from rtp_llm.models_py.modules.factory.linear import LinearBase
from rtp_llm.ops import HWKernelConfig
from rtp_llm.platforms.ppu.kernels.int8.deepgemm_warmup import (
    get_deep_gemm_warmup_mode,
    resolve_deep_gemm_warmup_max_tokens,
    warmup_dense_int8_gemm,
)
from rtp_llm.platforms.ppu.kernels.int8.deepgemm_wrapper import (
    configure_deep_gemm_num_sms,
    deep_gemm_default_num_sms,
    has_deep_gemm_int8_dense,
    int8_gemm_nt,
)
from rtp_llm.platforms.ppu.kernels.int8.int8_kernel.int8_quant import (
    per_token_quant_int8,
)

logger = logging.getLogger(__name__)


class PpuInt8DeepGemmLinear(LinearBase):
    """Static per-channel INT8 weights with dynamic per-token activations."""

    QUANT_METHOD = "W8A8_INT8_PER_CHANNEL_COMPRESSED"

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
            weight.dtype == torch.int8
            and weight_scales is not None
            and weight_scales.dtype == torch.float32
            and quant_config is not None
            and quant_config.get_method() == cls.QUANT_METHOD
            and has_deep_gemm_int8_dense()
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
                "W8A8 dense weight and scale must be 2D, got "
                f"{tuple(weight.shape)} and "
                f"{None if weight_scales is None else tuple(weight_scales.shape)}"
            )

        # RTP-LLM stores dense kernels as a [K, N] view over checkpoint [N, K]
        # storage. DeepGEMM consumes row-major [N, K].
        self.k, self.n = weight.shape
        self.weight = weight.reshape(self.n, self.k).contiguous()
        self.weight_scale = weight_scales.reshape(self.n, 1).contiguous()
        self.bias = bias

        # The wheel's own grid width, for every shape, with no knob -- see
        # deep_gemm_default_num_sms(). One source for warmup and execution
        # both: num_sms is not part of DeepGEMM's JIT key but the tile config
        # it picks is, so resolving it differently in the two places makes
        # warmup compile kernels that forward() never looks up.
        self.num_gemm_sms = deep_gemm_default_num_sms()

        self._maybe_warmup()

    def _maybe_warmup(self) -> None:
        """Compile-only JIT the dense kernel so the first request skips JIT.

        Independent of the engine ``--warm_up`` forward: that only exercises a
        single (max) M, whereas this sweeps the decode/prefill M range. Errors
        are non-fatal; execution falls back to lazy JIT.
        """
        try:
            mode = get_deep_gemm_warmup_mode()
            if mode == "skip":
                return
            warmup_dense_int8_gemm(
                (self.weight, self.weight_scale),
                max_m=resolve_deep_gemm_warmup_max_tokens(),
                mode=mode,
                num_sms=self.num_gemm_sms,
            )
        except Exception:
            logger.exception(
                "DeepGEMM dense INT8 warmup failed; execution will use lazy JIT"
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.shape[-1] != self.k:
            raise ValueError(
                f"W8A8 Linear expected input K={self.k}, got {input.shape[-1]}"
            )
        output_shape = (*input.shape[:-1], self.n)
        input_2d = input.reshape(-1, self.k)
        if input_2d.shape[0] == 0:
            return input.new_empty(output_shape)

        input_q, input_scale = per_token_quant_int8(input_2d)
        output = torch.empty(
            (input_2d.shape[0], self.n),
            device=input.device,
            dtype=torch.bfloat16,
        )
        with configure_deep_gemm_num_sms(self.num_gemm_sms):
            int8_gemm_nt(
                (input_q, input_scale),
                (self.weight, self.weight_scale),
                output,
            )
        if self.bias is not None:
            output.add_(self.bias.to(output.dtype))
        return output.to(input.dtype).view(output_shape)
