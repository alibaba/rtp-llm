#pragma once
#include <ATen/ATen.h>
#include <tuple>
std::tuple<at::Tensor, at::Tensor> kimi_k3_grouped_topk(
    const at::Tensor& scores, const at::Tensor& bias, int64_t n_group,
    int64_t topk_group, int64_t topk, bool renormalize, double scale);
