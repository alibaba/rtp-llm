#pragma once

#include <torch/all.h>

namespace torch_ext {

// Materialize the DSpARK logits + native-dtype Markov bias, divided by
// per-row temperature, as contiguous FP32 input for candidate selection.
torch::Tensor
markov_scaled_logits_sm90(const torch::Tensor& logits, const torch::Tensor& markov_bias, const torch::Tensor& temperature);

}  // namespace torch_ext
