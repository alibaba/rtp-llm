"""CUDA FP4 quantized Linear implementation using flashinfer mm_fp4 operator"""

import logging
import os
from typing import Optional

import torch

from rtp_llm.models_py.modules.factory.linear import LinearBase

logger = logging.getLogger(__name__)

from flashinfer import fp4_quantize, mm_fp4

from rtp_llm.models_py.kernels.cuda.fp4_kernel import cutlass_scaled_fp4_mm_wrapper
from rtp_llm.models_py.kernels.cuda.modelopt_fp4_quant import (
    residual_norm_fp4_quant,
    silu_fp4_quant,
)
from rtp_llm.models_py.utils.arch import get_sm
from rtp_llm.ops import HWKernelConfig


class CudaFp4GEMMLinear(LinearBase):
    """CUDA FP4 quantized Linear"""

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
        """Check if this strategy can handle the given configuration"""
        if (
            weight_scales is None
            or quant_config is None
            or weight_scale_2 is None
            or input_scale is None
        ):
            return False

        # Check quantization method
        quant_method = quant_config.get_method()
        return (
            quant_method in ("modelopt_fp4", "modelopt_mixed")
            and weight.dtype == torch.uint8
        )

    @torch.inference_mode()
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
        # [n, k // 2]
        self.hidden_size = weight.shape[1] * 2  # k
        self.output_size = weight.shape[0]  # n
        self.weight = weight
        self.weight_scales = weight_scales
        self.weight_scale_2 = weight_scale_2
        self.input_scale = input_scales
        self.bias = bias
        self.backend = os.getenv("RTP_LLM_FP4_GEMM_BACKEND", "cutlass")
        self.alpha = self.weight_scale_2 * self.input_scale
        self.input_scale_inv = 1 / self.input_scale
        self.silu_quant_mask = None
        # The private CuTe runner is validated with the pinned CUDA 13 stack.
        sm120_decode_enabled = (
            os.getenv("RTP_LLM_SM120_FUSION", "0") == "1"
            and self.backend == "cutlass"
            and torch.version.cuda is not None
            and torch.version.cuda.startswith("13.")
            and weight.is_cuda
            and get_sm(weight.device) == (12, 0)
        )
        self._decode_auto_backend = sm120_decode_enabled and (
            self.hidden_size,
            self.output_size,
        ) == (8704, 5120)
        self._decode_tiled_runner = None
        if sm120_decode_enabled and (self.hidden_size, self.output_size) in (
            (5120, 17408),
            (5120, 124160),
        ):
            try:
                from flashinfer.gemm.gemm_base import _b12x_gemm_fp4_runner

                self._decode_tiled_runner = _b12x_gemm_fp4_runner(
                    12, 0, True, torch.bfloat16, True
                )
            except (ImportError, AttributeError, TypeError) as error:
                logger.warning("SM120 FP4 tiled runner unavailable: %s", error)
        if os.getenv("RTP_LLM_SM120_FUSION", "0") == "1" and weight.is_cuda:
            # The expert quantizer also bounds every row by the input shape.
            # All rows of a dense matrix are valid; prepare this outside capture.
            self.silu_quant_mask = torch.full(
                (1,),
                torch.iinfo(torch.int32).max,
                device=weight.device,
                dtype=torch.int32,
            )

        self._swiglu_weights = None
        if (
            self._decode_tiled_runner is not None
            and (self.hidden_size, self.output_size) == (5120, 17408)
            and self.bias is None
        ):
            from rtp_llm.models_py.kernels.cuda.sm120_fp4_swiglu import prepare_weights

            self._swiglu_weights = prepare_weights(self.weight, self.weight_scales)
            logger.info(
                "SM120 fused FFN retains %d additional weight/scale bytes on %s",
                sum(t.numel() * t.element_size() for t in self._swiglu_weights),
                self.weight.device,
            )

    def forward_fused_mlp(
        self, input, down, *, residual=None, norm=None, communicator=None
    ):
        if (
            self._swiglu_weights is None
            or not isinstance(down, CudaFp4GEMMLinear)
            or down.backend != "cutlass"
            or (down.hidden_size, down.output_size) != (8704, 5120)
            or input.dtype != torch.bfloat16
            or input.ndim != 2
            or not 0 < input.shape[0] <= 64
            or input.shape[1] != 5120
            or not input.is_contiguous()
            or input.device != self.weight.device
        ):
            return None
        from rtp_llm.models_py.kernels.cuda.sm120_fp4_swiglu import fused_swiglu_fp4

        if norm is None:
            quantized = fp4_quantize(input, self.input_scale_inv)
        else:
            quantized = residual_norm_fp4_quant(
                input,
                residual,
                norm.weight,
                self.input_scale_inv,
                norm.variance_epsilon,
                communicator,
            )
            if quantized is None:
                return None
        output, scales = fused_swiglu_fp4(self, *quantized, down.input_scale_inv)
        return down._forward_quantized(output, scales, input.shape[0], input.dtype)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Get input dimensions
        input_m = input.shape[0]
        output_dtype = input.dtype

        # Check input dtype - accept BF16 and FP16
        if input.dtype not in (torch.bfloat16, torch.float16):
            error_msg = (
                f"CudaFp4GEMMLinear accepts bfloat16 and float16 input, but got {input.dtype}. "
                "Please convert input to bfloat16 or float16 before calling CudaFp4GEMMLinear."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        if self.backend == "trtllm" and input.dtype == torch.float16:
            error_msg = (
                "CudaFp4GEMMLinear with trtllm backend only supoorts bfloat16 input, "
                f"but got {input.dtype}. Please convert input to bfloat16 with trtllm backend."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Quantize BF16 or FP16 input to (FP4 and interleaved block scale)
        input_fp4, input_scale_interleaved = fp4_quantize(input, self.input_scale_inv)
        assert input_fp4.dtype == torch.uint8

        return self._forward_quantized(
            input_fp4, input_scale_interleaved, input_m, output_dtype
        )

    def forward_with_silu(self, gate_up: torch.Tensor) -> Optional[torch.Tensor]:
        if gate_up.ndim != 2 or gate_up.shape[1] != self.hidden_size * 2:
            return None
        quantized = silu_fp4_quant(gate_up, self.input_scale_inv, self.silu_quant_mask)
        if quantized is None:
            return None
        return self._forward_quantized(*quantized, gate_up.shape[0], gate_up.dtype)

    def forward_with_residual_norm(self, input, residual, norm, communicator=None):
        if (self.hidden_size, self.output_size) != (5120, 17408):
            return None
        quantized = residual_norm_fp4_quant(
            input,
            residual,
            norm.weight,
            self.input_scale_inv,
            norm.variance_epsilon,
            communicator,
        )
        if quantized is None:
            return None
        return self._forward_quantized(*quantized, input.shape[0], input.dtype)

    def _forward_quantized(
        self, input_fp4, input_scale_interleaved, input_m, output_dtype
    ):
        output_shape = [input_m, self.output_size]
        backend = self.backend
        if output_dtype == torch.bfloat16:
            if self._decode_auto_backend and 0 < input_m <= 64:
                backend = "auto"
        if (
            self._decode_tiled_runner is not None
            and output_dtype == torch.bfloat16
            and 0 < input_m <= 64
        ):
            # The 64x128 SM120 tile also wins with rotating, uncached weights.
            # Keep the existing FP4/SF layout and device-resident alpha.
            output = torch.empty(
                output_shape, device=input_fp4.device, dtype=output_dtype
            )
            self._decode_tiled_runner.forward(
                [
                    input_fp4,
                    self.weight.T,
                    input_scale_interleaved,
                    self.weight_scales.T,
                    self.alpha,
                    output_dtype,
                    output,
                    16,
                    True,
                    None,
                ],
                tactic=((64, 128), (1, 1), False, False, "sm120", None),
            )
        elif self.backend == "sgl_cutlass":
            output = cutlass_scaled_fp4_mm_wrapper(
                input_fp4,
                self.weight,
                input_scale_interleaved.view(torch.float8_e4m3fn),
                self.weight_scales.view(torch.float8_e4m3fn),
                self.alpha,
                output_dtype,
            ).view(*output_shape)
        else:
            output = mm_fp4(
                input_fp4,
                self.weight.T,
                input_scale_interleaved,
                self.weight_scales.T,
                self.alpha,
                output_dtype,
                backend=backend,
            ).view(*output_shape)

        if self.bias is not None:
            output = output + self.bias.to(output.dtype)
        return output
