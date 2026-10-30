"""CUDA FP4 quantized Linear implementation using flashinfer mm_fp4 operator"""

import logging
import os
from functools import partial
from typing import NamedTuple, Optional

import torch

from rtp_llm.models_py.modules.factory.linear import LinearBase

logger = logging.getLogger(__name__)

from flashinfer import fp4_quantize, mm_fp4

from rtp_llm.models_py.kernels.cuda.fp4_kernel import cutlass_scaled_fp4_mm_wrapper
from rtp_llm.ops import HWKernelConfig


class Fp4Activation(NamedTuple):
    values: torch.Tensor
    block_scales: torch.Tensor
    scale: torch.Tensor


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
        if weight_scales is None or quant_config is None or weight_scale_2 is None:
            return False
        if (
            input_scale is None
            and getattr(quant_config, "activation_scheme", "static") != "dynamic"
        ):
            return False

        # Check quantization method
        quant_method = quant_config.get_method()
        return quant_method == "modelopt_fp4"

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
        self.quant_config = quant_config
        self.dynamic_input_scale = (
            getattr(quant_config, "activation_scheme", "static") == "dynamic"
        )
        self.backend = os.getenv("RTP_LLM_FP4_GEMM_BACKEND", "cutlass")
        if self.dynamic_input_scale:
            if self.backend != "cutlass":
                raise ValueError("Dynamic NVFP4 Linear requires the cutlass backend")
        else:
            self.alpha = self.weight_scale_2 * self.input_scale
            self.input_scale_inv = 1 / self.input_scale

    @staticmethod
    def quantize_dynamic(input: torch.Tensor) -> Fp4Activation:
        # Short batches benefit from keeping the per-tensor reduction and pack
        # in one launch. Keep FlashInfer's large-batch path for prefill.
        if (
            input.is_cuda
            and input.dtype == torch.bfloat16
            and input.stride(1) == 1
            and 0 < input.shape[0] <= 512
            and input.shape[1] % 64 == 0
            and input.stride(0) >= input.shape[1]
            and torch.cuda.get_device_capability(input.device) in ((10, 0), (10, 3))
        ):
            from rtp_llm.models_py.triton_kernels.common.nvfp4 import (
                quantize_strided_nvfp4,
            )

            return quantize_strided_nvfp4(input)
        input = input.contiguous()
        # Singleton row slices can be contiguous with a misaligned storage offset.
        if input.data_ptr() % 16:
            input = input.clone(memory_format=torch.contiguous_format)
        scale = input.abs().amax().float().clamp_min(1e-8).reshape(1) / (6.0 * 448.0)
        values, block_scales = fp4_quantize(input, scale.reciprocal())
        return Fp4Activation(values, block_scales, scale)

    def prepare_quantized_gemms(self, activation: Fp4Activation, out, *, num_streams):
        """Reuse the CUTLASS dispatch for same-shape shards, with one workspace per stream."""
        from flashinfer.gemm import gemm_base

        from rtp_llm.models_py.kernels.cuda.fp4_kernel.flashinfer_dual_scale import (
            select_runner,
        )

        if not self.dynamic_input_scale or self.bias is not None:
            raise ValueError(
                "Prepared NVFP4 GEMMs require a dynamic projection without bias"
            )
        device = activation.values.device
        workspaces = [
            torch.empty(
                gemm_base.DEFAULT_WORKSPACE_SIZE, dtype=torch.uint8, device=device
            )
            for _ in range(num_streams)
        ]

        def operands(a, target, workspace):
            return [
                a.values,
                self.weight,
                a.block_scales,
                self.weight_scales,
                a.scale,
                torch.bfloat16,
                target,
                16,
                True,
                workspace,
                self.weight_scale_2,
            ]

        runner, tactic = select_runner(
            operands(activation, out, workspaces[0]),
            row_scale=activation.scale.numel() != 1,
        )

        def launch(a, target, *, workspace):
            runner(inputs=operands(a, target, workspace), tactic=tactic)
            return target

        return tuple(partial(launch, workspace=workspace) for workspace in workspaces)

    def forward_quantized(self, activation: Fp4Activation, *, out=None, gemm=None):
        return self(activation, out=out, gemm=gemm)

    def _dynamic_gemm(self, activation: Fp4Activation, out, gemm=None):
        values, block_scales, scale = activation
        if (
            not self.dynamic_input_scale
            or values.ndim != 2
            or values.shape[1] * 2 != self.hidden_size
            or values.dtype != torch.uint8
            or scale.shape not in ((1,), (values.shape[0],))
            or scale.dtype != torch.float32
            or not scale.is_contiguous()
            or scale.device != values.device
        ):
            raise ValueError(
                "Prequantized NVFP4 GEMM requires packed values and scalar or row FP32 scales"
            )
        shape = (values.shape[0], self.output_size)
        if out is not None and (
            out.shape != shape
            or out.dtype != torch.bfloat16
            or not out.is_contiguous()
            or out.device != values.device
            or self.bias is not None
        ):
            raise ValueError(
                "Preallocated NVFP4 output requires contiguous BF16 and no bias"
            )
        if not values.shape[0]:
            return (
                out
                if out is not None
                else values.new_empty(shape, dtype=torch.bfloat16)
            )
        if gemm is not None:
            if out is None:
                raise ValueError("Prepared NVFP4 GEMM requires a caller-owned output")
            return gemm(activation, out)
        from flashinfer.gemm import gemm_base

        from rtp_llm.models_py.kernels.cuda.fp4_kernel.flashinfer_dual_scale import (
            select_runner,
        )

        if out is None:
            out = values.new_empty(shape, dtype=torch.bfloat16)
        workspace = gemm_base._get_cache_buf(
            f"rtp_nvfp4_workspace_{torch.cuda.current_stream(values.device).cuda_stream}",
            gemm_base.DEFAULT_WORKSPACE_SIZE,
            values.device,
        )
        inputs = [
            values,
            self.weight,
            block_scales,
            self.weight_scales,
            scale,
            torch.bfloat16,
            out,
            16,
            True,
            workspace,
            self.weight_scale_2,
        ]
        runner, tactic = select_runner(inputs, row_scale=scale.numel() != 1)
        output = runner(inputs=inputs, tactic=tactic)
        return output if self.bias is None else output + self.bias.to(output.dtype)

    def forward(
        self, input: torch.Tensor | Fp4Activation, *, out=None, gemm=None
    ) -> torch.Tensor:
        if isinstance(input, Fp4Activation):
            return self._dynamic_gemm(input, out, gemm)
        if out is not None or gemm is not None:
            raise ValueError(
                "A preallocated output or prepared GEMM requires prequantized NVFP4 input"
            )
        if self.dynamic_input_scale and (
            input.ndim != 2 or input.shape[1] != self.hidden_size
        ):
            raise ValueError(
                f"Dynamic FP4 GEMM expects [M,{self.hidden_size}], got {tuple(input.shape)}"
            )
        # Get input dimensions
        input_m = input.shape[0]
        output_n = self.output_size
        output_dtype = torch.bfloat16 if self.dynamic_input_scale else input.dtype
        output_shape = [input_m, output_n]

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

        if self.dynamic_input_scale:
            if input_m == 0:
                return input.new_empty(output_shape, dtype=output_dtype)
            return self._dynamic_gemm(self.quantize_dynamic(input), None)

        input_scale_inv, alpha = self.input_scale_inv, self.alpha

        # Quantize BF16 or FP16 input to (FP4 and interleaved block scale)
        input_fp4, input_scale_interleaved = fp4_quantize(input, input_scale_inv)
        assert input_fp4.dtype == torch.uint8

        if self.backend == "sgl_cutlass":
            output = cutlass_scaled_fp4_mm_wrapper(
                input_fp4,
                self.weight,
                input_scale_interleaved.view(torch.float8_e4m3fn),
                self.weight_scales.view(torch.float8_e4m3fn),
                alpha,
                output_dtype,
            ).view(*output_shape)
        else:
            output = mm_fp4(
                input_fp4,
                self.weight.T,
                input_scale_interleaved,
                self.weight_scales.T,
                alpha,
                output_dtype,
                backend=self.backend,
            ).view(*output_shape)

        if self.bias is not None:
            output = output + self.bias.to(output.dtype)
        return output
