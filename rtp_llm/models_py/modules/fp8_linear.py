from typing import Optional

import torch
from torch import nn

try:
    from rtp_llm.models_py.modules.fp8_kernel import sgl_per_token_group_quant_fp8

    FP8_AVAILABLE = True
    # FP8 quantization available
except ImportError as e:
    # FP8 quantization not available
    FP8_AVAILABLE = False

try:
    import deep_gemm
    from deep_gemm import fp8_gemm_nt

    DEEPGEMM_AVAILABLE = True
except ImportError:
    DEEPGEMM_AVAILABLE = False


class Fp8Linear(nn.Module):
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
        original_dtype = input.dtype

        # Convert to BF16 if needed
        if input.dtype != torch.bfloat16:
            input_bf16 = input.to(torch.bfloat16)
        else:
            input_bf16 = input

        # Quantize input to FP8
        if FP8_AVAILABLE:
            alignment = self._get_small_batch_padding(input_m)
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
            use_column_major = need_padding
            input_fp8, input_scales = sgl_per_token_group_quant_fp8(
                input_for_quant,
                group_size=128,
                eps=quantization_eps,
                column_major_scales=use_column_major,
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
                    # DeepGEMM call failed, fallback to torch
                    print(f"Fp8Linear forward error type: {type(e)}")
                    import traceback

                    traceback.print_exc()
                    raise
            else:
                # DeepGEMM not available
                output = self._torch_fallback(input_fp8, input_scales)
        else:
            # FP8 not available, use FP16 fallback
            output = self._fp16_fallback(input_bf16)
            need_padding = False

        if need_padding:
            output = output[:input_m, :]
        if self.bias is not None:
            output = output + self.bias.to(output.dtype)
        if output.dtype != original_dtype:
            output = output.to(original_dtype)
        final_output = output
        return final_output

    def _get_small_batch_padding(self, m):
        """Calculate padding size for memory alignment optimization."""
        # Small batch optimization: use smaller padding to reduce memory waste
        if m < 64:
            return 16 if m < 16 else 8
        else:
            # Large batch: use standard padding for optimal performance
            return 64

    def _torch_fallback(self, input_fp8, input_scales):
        """Fallback implementation using torch."""
        expanded_input_scales = self._expand_input_scales(input_scales, input_fp8.shape)
        input_fp32 = input_fp8.to(torch.float32) * expanded_input_scales
        weight_fp32 = self.weight.to(torch.float32) * self._expand_weight_scales()
        # Perform matrix multiplication: [m, k] @ [n, k].T = [m, n]
        output = torch.matmul(input_fp32, weight_fp32.T)
        return output.to(torch.bfloat16)

    def _fp16_fallback(self, input_tensor):
        """Pure FP16 fallback logic."""
        weight_fp32 = self.weight.to(torch.float32) * self._expand_weight_scales()
        weight_fp16 = weight_fp32.to(torch.float16)
        input_fp16 = input_tensor.to(torch.float16)
        output = torch.matmul(input_fp16, weight_fp16.T)
        return output.to(torch.bfloat16)

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
