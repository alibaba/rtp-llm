"""ROCm FP8 PTPC (Per-Token Per-Channel) quantized Linear implementation"""

import logging
from functools import lru_cache
from typing import Optional

import aiter
import torch

from rtp_llm.models_py.modules.factory.linear import LinearBase

logger = logging.getLogger(__name__)

from rtp_llm.models_py.kernels.rocm.fp8_kernel import rocm_per_token_quant_fp8
from rtp_llm.ops import HWKernelConfig


def _unshuffle_weight(weight: torch.Tensor, layout=(16, 16)) -> torch.Tensor:
    """Inverse of aiter.ops.shuffle.shuffle_weight, used by hipb_mm path that
    expects the original [K,N] layout instead of CK preshuffled layout."""
    weight_type = weight.dtype
    in_block, ik_block = layout
    block_k = ik_block * 2
    vector_k = 16 // weight.element_size()
    block_n = in_block
    assert weight.shape[-2] % block_n == 0
    assert weight.shape[-1] % block_k == 0
    weight_view = weight.view(
        -1,
        weight.shape[-2] // block_n,
        weight.shape[-1] // block_k,
        block_k // vector_k,
        block_n,
        vector_k,
    )
    weight_view = weight_view.permute(0, 1, 4, 2, 3, 5)
    weight_view = weight_view.contiguous().view(*weight.shape)
    return weight_view.view(weight_type)


class RocmFp8PTPCLinear(LinearBase):
    """ROCm FP8 PTPC (Per-Token Per-Channel) quantized Linear"""

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
        """Handle FP8_PER_CHANNEL_COMPRESSED and FP8_PER_CHANNEL_QUARK"""
        if weight_scales is None or quant_config is None:
            return False

        if weight.dtype not in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
            return False

        quant_method = quant_config.get_method()
        return quant_method in ("FP8_PER_CHANNEL_COMPRESSED", "FP8_PER_CHANNEL_QUARK")

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
        self.hidden_size = weight.shape[0]  # k
        self.output_size = weight.shape[1]  # n
        # Reshape weight from [k, n] to [n, k] as done in C++ code
        self.weight = weight.reshape([weight.shape[1], weight.shape[0]])
        self.weight_scales = weight_scales.reshape(
            [weight_scales.shape[1], weight_scales.shape[0]]
        )
        self.bias = bias

    @staticmethod
    @lru_cache(maxsize=1)
    def init_hipblas() -> None:
        aiter.hipb_create_extension()

    def _get_hipb_weight_and_scales(self):
        """Lazy: convert CK-preshuffled weight back to [K,N] for hipb_mm,
        and transpose scale from [N,1] to [1,N]. Cached on first call."""
        if not hasattr(self, "_hipb_weight"):
            raw_weight = _unshuffle_weight(self.weight, layout=(16, 16))
            self._hipb_weight = raw_weight.T.contiguous()
            self._hipb_weight_scales = self.weight_scales.T.contiguous()
        return self._hipb_weight, self._hipb_weight_scales

    def _quantize_input(self, input: torch.Tensor):
        original_dtype = input.dtype
        if input.dtype != torch.bfloat16:
            input_bf16 = input.to(torch.bfloat16)
        else:
            input_bf16 = input
        input_fp8, input_scales = rocm_per_token_quant_fp8(input_bf16, eps=1e-10)
        input_scales = input_scales.to(torch.float32)
        return input_fp8, input_scales, input_bf16, original_dtype

    def _forward_impl(self, input: torch.Tensor, add_bias: bool) -> torch.Tensor:
        input_fp8, input_scales, input_bf16, original_dtype = self._quantize_input(
            input
        )

        # Always use aiter.gemm_a8w8_bpreshuffle (ck backend) so the tuned
        # lookup table in module_gemm_a8w8_bpreshuffle.so kicks in. The previous
        # cktile dispatch was a workaround for missing tuning; with the tuned .so
        # in place ck is consistently faster on MI308X for visionbert shapes.
        output = aiter.gemm_a8w8_bpreshuffle(
            input_fp8,
            self.weight,
            input_scales,
            self.weight_scales,
            None,
            input_bf16.dtype,
        )

        if add_bias and self.bias is not None:
            output = output + self.bias.to(output.dtype)

        if output.dtype != original_dtype:
            output = output.to(original_dtype)
        return output

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(input, add_bias=True)

    def forward_without_bias(self, input: torch.Tensor) -> torch.Tensor:
        """Same as forward() but skips bias add. Caller is responsible for
        adding self.bias somewhere downstream (residual add / GELU / etc)."""
        return self._forward_impl(input, add_bias=False)

    def forward_hipb_bias(self, input: torch.Tensor) -> torch.Tensor:
        """Run GEMM via aiter.hipb_mm with fused per-token/per-channel scale and
        fused bias epilogue. Used by QKV linear on MI308X for large-M shapes
        where hipBLASLt's bias epilogue saves an independent bias-add kernel.

        Caller must guarantee self.bias is not None.
        """
        assert self.bias is not None
        self.init_hipblas()

        input_fp8, input_scales, input_bf16, original_dtype = self._quantize_input(
            input
        )
        hipb_weight, hipb_weight_scales = self._get_hipb_weight_and_scales()

        output = aiter.hipb_mm(
            input_fp8,
            hipb_weight,
            -1,
            bias=self.bias,
            out_dtype=input_bf16.dtype,
            scaleA=input_scales,
            scaleB=hipb_weight_scales,
        )

        if output.dtype != original_dtype:
            output = output.to(original_dtype)
        return output

    def forward_hipb_bias_gelu(self, input: torch.Tensor) -> torch.Tensor:
        """Run GEMM + bias + GELU in a single hipBLASLt kernel via
        HIPBLASLT_EPILOGUE_GELU_BIAS. Eliminates the independent
        _bias_gelu_kernel that otherwise follows FC1 GEMM.

        Caller must guarantee self.bias is not None.
        """
        assert self.bias is not None
        self.init_hipblas()

        input_fp8, input_scales, input_bf16, original_dtype = self._quantize_input(
            input
        )
        hipb_weight, hipb_weight_scales = self._get_hipb_weight_and_scales()

        output = aiter.hipb_mm(
            input_fp8,
            hipb_weight,
            -1,
            bias=self.bias,
            out_dtype=input_bf16.dtype,
            scaleA=input_scales,
            scaleB=hipb_weight_scales,
            use_gelu=True,
        )

        if output.dtype != original_dtype:
            output = output.to(original_dtype)
        return output
