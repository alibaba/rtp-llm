#pragma once
#include <torch/extension.h>

namespace torch_ext {
std::vector<at::Tensor> v41_qb_gemm(const at::Tensor& a, const at::Tensor& sa,
    const at::Tensor& b, const at::Tensor& sb, const at::Tensor& weights,
    const at::Tensor& frequencies, const at::Tensor& positions);
}
