#pragma once

#include <torch/extension.h>

namespace torch_ext {

at::Tensor cublas_gemm_bf16_bf16_fp32(const at::Tensor& input, const at::Tensor& weight);
at::Tensor batch_invariant_cublaslt_gemm(const at::Tensor& input, const at::Tensor& weight, bool fp32_output);

}  // namespace torch_ext
