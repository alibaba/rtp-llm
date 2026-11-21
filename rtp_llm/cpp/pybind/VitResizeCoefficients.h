#pragma once

#include <ATen/core/Tensor.h>
#include <cstdint>
#include <utility>

namespace rtp_llm {

std::pair<at::Tensor, at::Tensor> vitResizeCoefficients(int64_t input, int64_t output);

}  // namespace rtp_llm
