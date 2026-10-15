#pragma once

#include <ATen/ATen.h>

namespace rtp_llm {
at::Tensor w4a16Sm120Hadamard(at::Tensor input, double scale);
void       w4a16Sm120Transform(const at::Tensor& weight, at::Tensor packed, at::Tensor scales, at::Tensor scratch);
void       w4a16Sm120Gemm(const at::Tensor& input,
                          const at::Tensor& packed,
                          const at::Tensor& scales,
                          at::Tensor        output,
                          int64_t           output_size,
                          int64_t           input_size,
                          int64_t           split_k);
}  // namespace rtp_llm
