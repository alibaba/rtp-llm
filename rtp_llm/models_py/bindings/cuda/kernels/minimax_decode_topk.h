// Adapted from SGLang MiniMax-M3 decode top-k kernel.
// Licensed under the Apache License, Version 2.0.
#pragma once

#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <torch/all.h>

namespace rtp_llm {

void minimax_decode_topk(
    const at::Tensor& score, const at::Tensor& seq_lens, at::Tensor& topk_idx, int64_t block_size, int64_t topk);

}  // namespace rtp_llm
