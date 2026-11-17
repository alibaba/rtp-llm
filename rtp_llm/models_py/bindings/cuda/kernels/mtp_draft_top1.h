#pragma once

#include <torch/extension.h>

namespace rtp_llm {

// Each int64 is an opaque (ordered score, inverse global token ID) pair.
// Only the local rank's slot is initialized; execAllGather fills the others.
torch::Tensor mtpDraftLocalTop1(const torch::Tensor& logits, int tp_size, int tp_rank, int64_t vocab_size);
torch::Tensor mtpDraftGlobalTop1(const torch::Tensor& candidates);

}  // namespace rtp_llm
