#pragma once

#include <torch/types.h>

namespace rtp_llm {

torch::Tensor getMultimodalFeatureHash(const torch::Tensor& embedding);

}  // namespace rtp_llm
