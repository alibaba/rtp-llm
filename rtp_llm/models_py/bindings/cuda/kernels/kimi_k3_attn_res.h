#pragma once
#include <ATen/ATen.h>
#include <optional>

void kimi_k3_attn_res(at::Tensor& prefix, std::optional<at::Tensor> delta,
                     at::Tensor& blocks, const at::Tensor& norm_weight,
                     const at::Tensor& qk_weight,
                     std::optional<at::Tensor> output_norm_weight,
                     at::Tensor& output, int64_t num_blocks,
                     int64_t block_write_idx, double eps, double output_norm_eps);
