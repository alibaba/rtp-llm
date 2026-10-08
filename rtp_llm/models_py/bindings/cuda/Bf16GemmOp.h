#pragma once

#include <torch/extension.h>

namespace torch_ext {

at::Tensor cublas_gemm_bf16_bf16_fp32(const at::Tensor& input, const at::Tensor& weight);

at::Tensor cublas_gemm_bf16_fp32_accum(const at::Tensor& input, const at::Tensor& weight);

at::Tensor cublas_gemm_bf16_fp32_accum_add(const at::Tensor& input,
                                        const at::Tensor& weight,
                                        const at::Tensor& residual);

}  // namespace torch_ext
