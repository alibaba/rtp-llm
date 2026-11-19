#pragma once
#include <ATen/core/Tensor.h>

namespace rtp_llm {
// Argmax of each draft vocabulary row, returning int64 token IDs with keepdim=true.
at::Tensor atomDecodeDraftArgmax(const at::Tensor& logits);
}  // namespace rtp_llm
