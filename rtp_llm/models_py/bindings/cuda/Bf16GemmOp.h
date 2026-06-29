#pragma once

#include <torch/extension.h>

namespace torch_ext {

at::Tensor cublas_gemm_bf16_bf16_fp32(const at::Tensor& input, const at::Tensor& weight);
at::Tensor cublas_gemm_bf16_bf16_fp32_out(const at::Tensor& input,
                                          const at::Tensor& weight,
                                          at::Tensor&       out);

}  // namespace torch_ext
