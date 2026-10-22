#pragma once
#include <ATen/ATen.h>
#include <vector>

namespace torch_ext {
std::vector<at::Tensor> v41_front_gemm(const at::Tensor& x, const at::Tensor& a,
    const at::Tensor& sa, const at::Tensor& b, const at::Tensor& sb, const at::Tensor& head);
}
