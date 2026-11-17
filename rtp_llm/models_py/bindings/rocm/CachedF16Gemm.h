#pragma once

#include "rtp_llm/models_py/bindings/common/Torch_ext.h"
#include <optional>

namespace rtp_llm {
// None means unsupported/capturing/cache full: the caller must use AITER.
// Execution failures throw; they must never be retried on a possibly bad stream.
std::optional<at::Tensor> rocmCachedF16Gemm(const at::Tensor&                input,
                                            const at::Tensor&                weight,
                                            const std::optional<at::Tensor>& bias,
                                            bool                             preshuffle);
// Current thread/device/stream only: hits, misses, plans, workspace bytes.
std::vector<int64_t> rocmF16GemmCacheStats();
}  // namespace rtp_llm
