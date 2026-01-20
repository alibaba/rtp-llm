"""CUDA FP8 DeepGEMM quantized Linear implementation"""

import logging
from typing import Optional

import torch

from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
    fp8_gemm_nt,
    has_deep_gemm,
    is_deep_gemm_e8m0_used,
)
from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
    create_per_token_group_quant_fp8_output_scale,
    requant_weight_ue8m0,
    sgl_per_token_group_quant_fp8,
)
from rtp_llm.models_py.modules.factory.linear import LinearBase
from rtp_llm.ops import HWKernelConfig

# from rtp_llm.test.utils.numeric_util import per_token_cast_back

logger = logging.getLogger(__name__)


class CudaFp8DeepGEMMLinear(LinearBase):
    """CUDA FP8 DeepGEMM quantized Linear"""

    # 全局共享的 scale cache，key = (device, K, max_len)
    _global_scale_cache: dict = {}

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
        """Handle other FP8 methods (FP8, FP8_PER_BLOCK, etc.)"""
        if weight_scales is None or quant_config is None:
            return False

        # Check if weight is FP8 format
        if weight.dtype not in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
            return False

        # Check quantization method - handle all other FP8 methods
        quant_method = quant_config.get_method()
        return quant_method == "FP8_PER_BLOCK"

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
        # Initialize parameters
        self.weight = weight
        self.weight_scales = weight_scales
        self.input_scales = input_scales
        self.bias = bias

        # Check if DeepGEMM is available
        if not has_deep_gemm():
            error_msg = "DeepGEMM is not available. Please install the `deep_gemm` package to enable DeepGEMM kernels."
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        # Check weight and weight scale dimensions
        if self.weight.dim() != 2 or self.weight_scales.dim() != 2:
            error_msg = f"Weight and weight scale must be 2D tensors, but got weight dim: {self.weight.dim()} and weight scale dim: {self.weight_scales.dim()}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        # Reshape weight and weight scale
        self.K, self.N = self.weight.shape
        self.scale_K, self.scale_N = self.weight_scales.shape
        self.weight = self.weight.reshape(self.N, self.K)
        self.weight_scales = self.weight_scales.reshape(self.scale_N, self.scale_K)
        # Check weight scale sizes
        if (self.N + 127) // 128 != self.scale_N or (
            self.K + 127
        ) // 128 != self.scale_K:
            error_msg = f"Weight scale dimension mismatch! N: {self.N}, scale_N: {self.scale_N}, K: {self.K}, scale_K: {self.scale_K}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        # Check weight and weight scale dtypes
        if self.weight.dtype != torch.float8_e4m3fn:
            error_msg = f"Weight dtype must be float8_e4m3fn, got {self.weight.dtype}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        if self.weight_scales.dtype != torch.float32:
            error_msg = (
                f"Weight scale dtype must be float32, got {self.weight_scales.dtype}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        # Check bias
        if self.bias is not None:
            if self.bias.dim() != 1 and self.bias.dim() != 2:
                error_msg = f"Bias dimension must be 1 or 2, got {self.bias.dim()}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            if self.bias.shape[-1] != self.N:
                error_msg = (
                    f"Bias last dimension must be {self.N}, got {self.bias.shape[-1]}"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
            if self.bias.dim() == 2 and self.bias.shape[0] != 1:
                error_msg = f"Bias first dimension must be 1, got {self.bias.shape[0]}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            if self.bias.dtype != torch.bfloat16:
                error_msg = f"Bias dtype must be bfloat16, got {self.bias.dtype}"
                logger.error(error_msg)
                raise ValueError(error_msg)
        self.scale_ue8m0 = is_deep_gemm_e8m0_used()
        # Initialize cached scales attributes
        self.cached_scales = None
        self.cached_scales_max_len = 0
        # Disable UE8M0 for small tensors due to performance/accuracy trade-offs.
        # TODO: Re-evaluate this threshold after further optimization of UE8M0 kernels.
        if self.weight.shape[0] < 128 or self.weight.shape[1] < 256:
            self.scale_ue8m0 = False
        if self.scale_ue8m0:
            w_tmp, self.weight_scales = requant_weight_ue8m0(
                self.weight, self.weight_scales
            )
            self.weight.copy_(w_tmp)
            del w_tmp

    def maybe_cache_quant_scale(self, max_len: int) -> None:
        if not self.scale_ue8m0:
            return

        # 使用 (device, K, max_len) 作为 cache key
        cache_key = (str(self.weight.device), self.K, max_len)

        if cache_key not in CudaFp8DeepGEMMLinear._global_scale_cache:
            cache_shape = [max_len, self.K]
            cached_scales = create_per_token_group_quant_fp8_output_scale(
                x_shape=cache_shape,
                device=self.weight.device,
                group_size=128,
                column_major_scales=True,
                scale_tma_aligned=True,
                scale_ue8m0=self.scale_ue8m0,
            )
            cached_scales.fill_(0x7F7F7F7F)
            CudaFp8DeepGEMMLinear._global_scale_cache[cache_key] = cached_scales
            logging.info(
                f"Created global cached scales for key {cache_key}, shape: {cached_scales.shape}"
            )

        # 引用全局 cache
        self.cached_scales = CudaFp8DeepGEMMLinear._global_scale_cache[cache_key]
        self.cached_scales_max_len = self.cached_scales.shape[0]
        logging.info(
            f"successfully cached {self.cached_scales_max_len} scales, shape: {self.cached_scales.shape}"
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Check input dtype - only accept bfloat16
        if input.dtype != torch.bfloat16 and input.dtype != torch.float8_e4m3fn:
            error_msg = f"Input tensor dtype must be bfloat16 or float8_e4m3fn, got {input.dtype}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        # Check input tensor dimension
        if input.dim() != 2:
            error_msg = f"Input tensor dimension must be 2, got {input.dim()}D tensor"
            logger.error(error_msg)
            raise ValueError(error_msg)
        M, K = input.shape
        # Check input tensor inner dimension expected to be K
        if K != self.K:
            error_msg = f"Input tensor inner dimension expected to be {self.K}, got {K}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        if input.dtype == torch.float8_e4m3fn:
            if not self.scale_ue8m0:
                error_msg = "Scale UE8M0 is required for float8_e4m3fn input"
                logger.error(error_msg)
                raise ValueError(error_msg)
            input_fp8 = input
            if self.cached_scales is not None and self.cached_scales_max_len >= M:
                aligned_M = (M + 3) // 4 * 4  # ceil_align(M, 4)
                K_scaled = self.cached_scales.shape[1]  # K // 128 维度
                input_scales = self.cached_scales[:M, :].as_strided(
                    (M, K_scaled), (1, aligned_M)
                )
            else:
                input_scales = create_per_token_group_quant_fp8_output_scale(
                    x_shape=input.shape,
                    device=input.device,
                    group_size=128,
                    column_major_scales=True,
                    scale_tma_aligned=True,
                    scale_ue8m0=self.scale_ue8m0,
                )
                # filling with ue8m0 encoded ones, skip actual quantizing
                input_scales.fill_(0x7F7F7F7F)

        else:
            # Quantize x to FP8
            input_fp8, input_scales = sgl_per_token_group_quant_fp8(
                input,
                group_size=128,
                eps=1e-4,
                column_major_scales=True,
                scale_tma_aligned=True,
                scale_ue8m0=self.scale_ue8m0,
            )

        # Prepare output tensor
        output = torch.empty(M, self.N, dtype=torch.bfloat16, device=input.device)
        # Invoke DeepGEMM
        fp8_gemm_nt(
            (input_fp8, input_scales),
            (self.weight, self.weight_scales),
            output,
            c=None,
            disable_ue8m0_cast=not self.scale_ue8m0,
        )
        if self.bias is not None:
            output = output + self.bias.to(output.dtype)
        return output
