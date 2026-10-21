#pragma once
#include <ATen/ATen.h>
#include <cstdint>
#include <optional>

void hy4_qkv_norm_launch(const at::Tensor& x, const at::Tensor& qw, const at::Tensor& kw,
                        const at::Tensor& q8, const at::Tensor& scale,
                        const std::optional<at::Tensor>& qb, const at::Tensor& kb,
                        int qr, int kr, float qe, float ke, bool pdl, std::uint64_t event,
                        const std::optional<at::Tensor>& cache, const std::optional<at::Tensor>& slots,
                        const std::optional<at::Tensor>& positions, const std::optional<at::Tensor>& cos_sin, int release);

void qkv_test_copy(const at::Tensor&, const at::Tensor&, bool, bool);
