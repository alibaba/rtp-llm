import logging
import os
from typing import Optional

import torch
from torch import nn

from rtp_llm.config.quant_config import QuantizationConfig
from rtp_llm.models_py.modules import utils

logger = logging.getLogger(__name__)

try:
    from rtp_llm.models_py.modules.fp8_kernel import (
        scaled_fp8_per_tensor_quant,
        sgl_per_token_group_quant_fp8,
    )

    FP8_AVAILABLE = True
    # FP8 quantization available
except ImportError:
    # FP8 quantization not available
    FP8_AVAILABLE = False

if utils.is_cuda():
    try:
        from rtp_llm.models_py.kernels.deepgemm_wrapper import fp8_gemm_nt

        DEEPGEMM_AVAILABLE = True
        # Setup CUTLASS include paths for JIT compilation
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Search for CUTLASS headers
        cutlass_paths = []
        parts = current_dir.split("/")
        for i in range(len(parts)):
            base_path = "/".join(parts[: i + 1])
            for subpath in [
                "deep_gemm/third-party/cutlass/include",
                "external/deep_gemm/third-party/cutlass/include",
            ]:
                path = os.path.join(base_path, subpath)
                if os.path.exists(path):
                    cutlass_paths.append(path)
        # Check runfiles directory
        if "runfiles" in current_dir:
            runfiles_root = current_dir.split("runfiles")[0] + "runfiles"
            for subdir in ["deep_gemm", "external/deep_gemm"]:
                path = os.path.join(
                    runfiles_root, subdir, "third-party/cutlass/include"
                )
                if os.path.exists(path):
                    cutlass_paths.append(path)
        # Set environment variables if CUTLASS found
        if cutlass_paths:
            cutlass_path = cutlass_paths[0]
            for env_var in ["CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH", "CPATH"]:
                current_val = os.environ.get(env_var, "")
                os.environ[env_var] = (
                    f"{cutlass_path}:{current_val}" if current_val else cutlass_path
                )
            nvcc_flags = os.environ.get("NVCC_PREPEND_FLAGS", "")
            os.environ["NVCC_PREPEND_FLAGS"] = f"-I{cutlass_path} {nvcc_flags}".strip()
        logger.info(f"DeepGEMM successfully imported with fp8_gemm_nt: {fp8_gemm_nt}")
    except ImportError as e:
        logger.warning(f"DeepGEMM not available: {e}")
        DEEPGEMM_AVAILABLE = False
        fp8_gemm_nt = None
    except Exception as e:
        logger.warning(f"Error importing DeepGEMM: {e}")
        DEEPGEMM_AVAILABLE = False
        fp8_gemm_nt = None
else:
    # Not CUDA device, deep_gemm not available
    DEEPGEMM_AVAILABLE = False
    fp8_gemm_nt = None


class Fp8DeepGEMMLinear(nn.Module):
    """FP8 Linear layer with DeepGEMM quantized matrix multiplication."""

    def __init__(
        self,
        weight: torch.Tensor,
        weight_scales: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        config=None,
    ) -> None:
        super().__init__()
        self.hidden_size = weight.shape[0]  # k
        self.output_size = weight.shape[1]  # n
        self.weight = weight.reshape([weight.shape[1], weight.shape[0]])
        self.weight_scales = weight_scales.reshape(
            [weight_scales.shape[1], weight_scales.shape[0]]
        )
        self.bias = bias

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Get input dimensions
        input_m = input.shape[0]
        input_k = input.shape[1]
        output_n = self.output_size

        # Check input dtype - only accept BF16
        if input.dtype != torch.bfloat16:
            error_msg = (
                f"Fp8DeepGEMMLinear only accepts bfloat16 input, but got {input.dtype}. "
                "Please convert input to bfloat16 before calling Fp8DeepGEMMLinear."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        input_bf16 = input

        # Quantize input to FP8
        if not FP8_AVAILABLE:
            # FP8 not available - fail fast for easier debugging
            error_msg = (
                "FP8 quantization is not available but required for Fp8DeepGEMMLinear. "
                "Please ensure FP8 kernel is properly installed and imported."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        alignment = self._get_padding_size(input_m)
        target_m = (input_m + alignment - 1) // alignment * alignment
        need_padding = target_m > input_m

        if need_padding:
            input_for_quant = torch.zeros(
                target_m, input_k, dtype=torch.bfloat16, device=input.device
            )
            input_for_quant[:input_m, :] = input_bf16
        else:
            input_for_quant = input_bf16

        # Quantize using sgl_per_token_group_quant_fp8
        quantization_eps = 1e-4
        input_fp8, input_scales = sgl_per_token_group_quant_fp8(
            input_for_quant,
            group_size=128,
            eps=quantization_eps,
            column_major_scales=False,
        )

        FP8_E4M3_MAX = 448.0
        min_scale_threshold = 1e-4 / FP8_E4M3_MAX
        input_scales = torch.clamp(input_scales, min=min_scale_threshold)
        input_scales = input_scales.to(torch.float32)
        output_m = input_for_quant.shape[0]
        output = torch.zeros(
            output_m, output_n, dtype=torch.bfloat16, device=input.device
        )

        # Call DeepGEMM
        if DEEPGEMM_AVAILABLE:
            deepgemm_input_scales = input_scales
            input_fp8 = input_fp8.contiguous()
            deepgemm_input_scales = deepgemm_input_scales.contiguous()
            weight = self.weight.contiguous()
            weight_scales = self.weight_scales.contiguous()
            output = output.contiguous()
            try:
                fp8_gemm_nt(
                    (input_fp8, deepgemm_input_scales),
                    (weight, weight_scales),
                    output,
                    c=None,
                    disable_ue8m0_cast=True,
                )
            except Exception as e:
                # DeepGEMM call failed - log error and re-raise
                error_msg = f"DeepGEMM fp8_gemm_nt call failed: {type(e).__name__}: {e}"
                logger.error(error_msg)
                import traceback

                logger.error(f"Traceback: {traceback.format_exc()}")
                raise RuntimeError(error_msg) from e
        else:
            # DeepGEMM not available - fail fast for easier debugging
            error_msg = (
                "DeepGEMM is not available but required for FP8 computation. "
                "Please ensure DeepGEMM is properly installed and imported."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        if need_padding:
            output = output[:input_m, :]
        if self.bias is not None:
            output = output + self.bias.to(output.dtype)
        return output

    def _get_padding_size(self, m):
        """Calculate padding size based on DeepGEMM requirements."""
        if self._gemm_swap_ab_heuristic(m):
            if m < 16:
                return 16
            else:
                return 8
        else:
            return 64

    def _gemm_swap_ab_heuristic(self, m):
        return False

    def _expand_input_scales(self, input_scales, target_shape):
        """Expand input scales to target shape."""
        # input_scales: [m, k/128] - always row-major
        # target_shape: [m, k]
        m, k = target_shape
        expected_scales_shape = (m, (k + 127) // 128)
        if input_scales.shape != expected_scales_shape:
            raise ValueError(
                f"Input scales shape mismatch! Expected {expected_scales_shape}, got {input_scales.shape}"
            )
        expanded = torch.zeros(
            target_shape, dtype=input_scales.dtype, device=input_scales.device
        )
        for i in range(input_scales.shape[0]):  # m tokens
            for j in range(input_scales.shape[1]):  # k/128 groups
                k_start = j * 128
                k_end = min((j + 1) * 128, k)
                expanded[i, k_start:k_end] = input_scales[i, j]
        return expanded

    def _expand_weight_scales(self):
        """Expand weight scales to weight tensor shape."""
        expanded = torch.zeros_like(self.weight, dtype=torch.float32)
        for i in range(self.weight_scales.shape[0]):  # output_size blocks (60)
            for j in range(self.weight_scales.shape[1]):  # hidden_size blocks (20)
                h_start = i * 128  # output_size dimension
                h_end = min((i + 1) * 128, self.weight.shape[0])
                w_start = j * 128  # hidden_size dimension
                w_end = min((j + 1) * 128, self.weight.shape[1])
                expanded[h_start:h_end, w_start:w_end] = self.weight_scales[i, j]
        return expanded


class Fp8PerTensorLinear(nn.Module):
    def __init__(
        self,
        quant_config: QuantizationConfig,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        input_scale: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.weight = weight.T
        self.weight_scale = weight_scale
        self.input_scale = input_scale
        self.bias = bias
        self.block_quant = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:

        input_2d = input.view(-1, input.shape[-1])
        output_shape = [*input.shape[:-1], self.weight.shape[1]]

        qinput, x_scale = scaled_fp8_per_tensor_quant(input_2d, self.input_scale)

        # TODO(serina.wzq): Use high performance kernel
        output = torch._scaled_mm(
            qinput,
            self.weight,
            out_dtype=input.dtype,
            scale_a=x_scale,
            scale_b=self.weight_scale,
            bias=self.bias,
        )

        if type(output) is tuple and len(output) == 2:
            output = output[0]

        return torch.narrow(output, 0, 0, input_2d.shape[0]).view(*output_shape)
