#pragma once

#include <torch/types.h>

namespace rtp_llm {

// Shared CPU/CUDA entry used by the Python binding and the LLM processor.
#if defined(__GNUC__)
__attribute__((visibility("default")))
#endif
torch::Tensor
getMultimodalFeatureHash(const torch::Tensor& embedding);

}  // namespace rtp_llm
