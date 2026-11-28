#pragma once

#include <torch/extension.h>
#include <vector>

namespace torch_ext {

at::Tensor cublas_gemm_bf16_bf16_fp32(const at::Tensor& input, const at::Tensor& weight);
// BF16 inputs with FP32 accumulation; output is contiguous FP32 or BF16 [M,N].
at::Tensor cublas_gemm_bf16_bf16_out(const at::Tensor& input,
                                     const at::Tensor& weight,
                                     const at::Tensor& output,
                                     at::ScalarType    output_dtype);
void       cublas_gemm_bf16_bf16_fp32_packed(const at::Tensor&              input,
                                             const std::vector<at::Tensor>& weights,
                                             const at::Tensor&              output);

}  // namespace torch_ext
