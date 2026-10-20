#pragma once
#include <torch/all.h>

namespace torch_ext {
bool dsv41_flashmla_pdl_available();
void dsv41_flashmla_decode_out(torch::Tensor q, torch::Tensor pool, torch::Tensor indices,
    torch::Tensor extra, torch::Tensor extra_indices, torch::Tensor positions,
    torch::Tensor cos_sin, torch::Tensor sink, torch::Tensor output, torch::Tensor scales,
    torch::Tensor lse, double sm_scale, bool wait_pdl);
} // namespace torch_ext
