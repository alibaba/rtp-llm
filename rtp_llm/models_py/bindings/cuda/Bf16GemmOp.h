#pragma once

#include <torch/extension.h>

#include <optional>

namespace torch_ext {

at::Tensor cublas_gemm_bf16_bf16_fp32(const at::Tensor&                input,
                                      const at::Tensor&                weight,
                                      const std::optional<at::Tensor>& output = std::nullopt);

}  // namespace torch_ext
