#pragma once
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include "rocm/pa_decode_dot_kernel.h"
#include "rocm/pa_decode_reduce_kernel.h"
void paged_attention_atrex(torch::Tensor&                      out,
                           torch::Tensor&                      exp_sums,
                           torch::Tensor&                      max_logits,
                           torch::Tensor&                      tmp_out,
                           torch::Tensor&                      query,
                           torch::Tensor&                      key_cache,
                           torch::Tensor&                      value_cache,
                           torch::Tensor&                      context_lens,
                           torch::Tensor&                      block_tables,
                           float                               scale,
                           int64_t                             max_context_len,
                           const std::optional<torch::Tensor>& alibi_slopes);
